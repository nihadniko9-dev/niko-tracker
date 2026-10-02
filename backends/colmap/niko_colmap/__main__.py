"""`python -m niko_colmap job.json` - COLMAP 4.2 (incremental or global mapper) camera candidate.

Job options:
  mapper        "global" (ex-GLOMAP, default) | "incremental"
  camera_model  COLMAP model for the single shared camera (default SIMPLE_RADIAL)
  images        "proxy" (default) | "frames"
  matcher       "auto" (exhaustive up to 60 frames, else sequential) | "exhaustive" | "sequential"
  overlap       sequential matching overlap (default 20)
  max_features  SIFT features per image (default 8192)
  stride        use every n-th frame (default 1). 60 fps and long clips: neighbouring frames add
                cost, not geometry; refinement registers the frames in between (PnP + BA).

Outputs: camera_raw.npz (full-resolution K), points.ply, colmap/ (database + sparse model).
COLMAP's pixel convention is corner origin, the same as ours, so intrinsics only scale.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from niko_backend_common import ResultWriter, read_job, save_camera_npz, write_ply


def camera_to_K_dist(cam) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(cam.params, np.float64)
    name = cam.model_name if hasattr(cam, "model_name") else str(cam.model).split(".")[-1]
    dist = np.zeros(5)
    if name == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif name == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif name == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; dist[0] = p[3]
    elif name == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; dist[:2] = p[3:5]
    elif name == "OPENCV":
        fx, fy, cx, cy = p[:4]; dist[:4] = p[4:8]
    else:
        raise ValueError(f"camera model {name} has no 5-coefficient OpenCV equivalent")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), dist


def cam_from_world(image):
    cfw = image.cam_from_world
    cfw = cfw() if callable(cfw) else cfw
    M = np.asarray(cfw.matrix(), np.float64)  # 3x4
    return M[:, :3], M[:, 3]


def write_colmap_masks(shot_dir: Path, names: list[str], size_wh, dst: Path) -> bool:
    """Our masks (255 = excluded) -> COLMAP masks (0 = ignore), named <image stem>.png."""
    src = shot_dir / "masks"
    files = sorted(src.glob("*.png")) if src.is_dir() else []
    if not files:
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for name, f in zip(names, files):
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, size_wh, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dst / (Path(name).stem + ".png")), np.where(m > 127, 0, 255).astype(np.uint8))
    return True


def main() -> None:
    job = read_job(sys.argv[1])
    opt = job.get("options", {})
    shot_dir, out_dir = Path(job["shot_dir"]), Path(job["out_dir"])
    if job.get("task") in ("refine", "sift_tracks", "mesh"):
        # refine: Ceres bundle adjustment of a problem prepared by niko.refine
        # sift_tracks: multi-view tracks from the verified SIFT matches of a COLMAP database
        # mesh: multi-view stereo + Poisson mesh on the solve's cameras (niko.mesh)
        if job["task"] == "refine":
            from .refine import run
        elif job["task"] == "mesh":
            from .mesh import run
        else:
            from .sift_tracks import run

        with ResultWriter(job, versions={"pycolmap": pycolmap.__version__}) as res:
            run(job, res)
        return
    with ResultWriter(job, versions={"pycolmap": pycolmap.__version__}) as res:
        shot = json.loads((shot_dir / "shot.json").read_text())
        W, H, T = shot["width"], shot["height"], shot["n_frames"]
        image_dir = shot_dir / opt.get("images", "proxy")
        names = sorted(p.name for p in image_dir.iterdir() if p.suffix in (".jpg", ".png"))
        if len(names) != T:
            raise ValueError(f"{len(names)} images in {image_dir}, shot has {T}")
        all_names = names
        stride = max(1, int(opt.get("stride", 1)))
        names = names[::stride]
        iw, ih = cv2.imread(str(image_dir / names[0])).shape[1::-1]
        sx, sy = W / iw, H / ih

        work = out_dir / "colmap"
        if work.exists():
            shutil.rmtree(work)
        (work / "sparse").mkdir(parents=True)
        db = work / "database.db"
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = opt.get("camera_model", "SIMPLE_RADIAL")
        has_masks = write_colmap_masks(shot_dir, all_names, (iw, ih), work / "masks")
        if has_masks:
            reader.mask_path = str(work / "masks")
        matcher = opt.get("matcher", "auto")
        if matcher == "auto":  # video: sequential pairs (overlap + quadratic long-range); short clips: all pairs
            matcher = "exhaustive" if T <= 60 else "sequential"
        # Features + matches depend only on these settings, not on the mapper: another candidate
        # (e.g. colmap_global) that already built an identical database is reused as is.
        settings = {"images": str(image_dir), "names": names, "camera_model": reader.camera_model,
                    "max_features": opt.get("max_features", 8192), "masks": has_masks, "matcher": matcher,
                    "overlap": opt.get("overlap", 20), "pycolmap": pycolmap.__version__}
        reuse = opt.get("reuse_db_from")
        src = out_dir.parent / reuse / "colmap" if reuse else None
        if src is not None and (src / "database.db").exists() and (src / "features.json").exists() \
                and json.loads((src / "features.json").read_text()) == settings:
            shutil.copyfile(src / "database.db", db)
            res.stats["database_reused_from"] = reuse
        else:
            fx_opts = pycolmap.FeatureExtractionOptions()
            fx_opts.sift.max_num_features = settings["max_features"]
            pycolmap.extract_features(db, image_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                      reader_options=reader, extraction_options=fx_opts, device=pycolmap.Device.cuda)
            if matcher == "exhaustive":
                pycolmap.match_exhaustive(db, device=pycolmap.Device.cuda)
            else:
                pairing = pycolmap.SequentialPairingOptions()
                pairing.overlap = settings["overlap"]
                pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cuda)
        (work / "features.json").write_text(json.dumps(settings))

        mapper = opt.get("mapper", "global")
        if mapper == "global":
            recs = pycolmap.global_mapping(db, image_dir, work / "sparse")
        elif mapper == "incremental":
            recs = pycolmap.incremental_mapping(db, image_dir, work / "sparse")
        else:
            raise ValueError(f"unknown mapper {mapper!r}")
        if not recs:
            raise RuntimeError("COLMAP produced no reconstruction")
        rec = max(recs.values(), key=lambda r: r.num_reg_images())

        K = np.tile(np.eye(3), (T, 1, 1)); dist = np.zeros((T, 5))
        R = np.tile(np.eye(3), (T, 1, 1)); t = np.zeros((T, 3)); valid = np.zeros(T, bool)
        index = {n: i for i, n in enumerate(all_names)}
        for image in rec.images.values():
            if not image.has_pose:
                continue
            i = index[image.name]
            Kp, d = camera_to_K_dist(rec.cameras[image.camera_id])
            K[i] = np.diag([sx, sy, 1.0]) @ Kp
            dist[i] = d
            R[i], t[i] = cam_from_world(image)
            valid[i] = True

        pts = rec.points3D
        if len(pts) < 50:  # real clip 02 (indoor, little parallax): 158 cameras "registered" on 6 points
            raise RuntimeError(f"COLMAP model is degenerate: {rec.num_reg_images()} images on {len(pts)} points")
        xyz = np.array([p.xyz for p in pts.values()]) if len(pts) else np.zeros((0, 3))
        rgb = np.array([p.color for p in pts.values()]) if len(pts) else np.zeros((0, 3))
        write_ply(out_dir / "points.ply", xyz, rgb)
        meta = {"method": f"colmap_{mapper}", "camera_model": reader.camera_model, "matcher": matcher,
                "images": opt.get("images", "proxy"), "image_size": [iw, ih], "masks": has_masks,
                "n_models": len(recs), "stride": stride}
        save_camera_npz(out_dir / "camera_raw.npz", K, dist, R, t, valid, meta)
        res.outputs.update({"camera": "camera_raw.npz", "points": "points.ply"})
        res.stats.update({"registered": int(valid.sum()), "n_frames": T, "stride": stride,
                          "keyframes": len(names), "n_points": int(len(xyz)),
                          "n_models": len(recs), "mean_reproj_px_proxy": float(rec.compute_mean_reprojection_error())})


if __name__ == "__main__":
    main()
