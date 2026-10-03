"""`python -m niko_da3 job.json` - Depth Anything 3 camera candidate (poses + per-frame intrinsics).

Job options:
  process_res   longest side DA3 works at (default 504, DA3's own default)
  max_frames    keyframes per call (default 40); longer clips use evenly spaced keyframes and the
                other frames stay invalid here - refinement registers them by PnP on the tracks
  ref_view      DA3 reference view strategy (default "saddle_balanced")
  max_points    points kept for points.ply

DA3 resizes the longest side to process_res and rounds each side to a multiple of 14, so x
and y scale independently; its intrinsics refer to that processed image. Both are scaled back
to full resolution here (corner-origin pixels scale linearly).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from niko_backend_common import ResultWriter, read_job, save_camera_npz, write_ply

from .common import CKPT_DIR, load_model


def repo_commit() -> str:
    import depth_anything_3  # a namespace package: no __file__, use __path__

    try:
        src = Path(list(depth_anything_3.__path__)[0]).resolve().parents[1]
        return subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    job = read_job(sys.argv[1])
    opt = job.get("options", {})
    shot_dir, out_dir = Path(job["shot_dir"]), Path(job["out_dir"])
    if job.get("task") in ("mesh", "texture"):  # surface / texture (Open3D lives in this env)
        import open3d

        from .mesh import run, texture

        with ResultWriter(job, versions={"open3d": open3d.__version__}) as res:
            (texture if job["task"] == "texture" else run)(job, res)
        return
    with ResultWriter(job, versions={"da3_commit": repo_commit(), "checkpoint": CKPT_DIR.name}) as res:
        import torch

        shot = json.loads((shot_dir / "shot.json").read_text())
        W, H, T = shot["width"], shot["height"], shot["n_frames"]
        files = sorted((shot_dir / "proxy").glob("*.jpg"))
        if len(files) != T:
            raise ValueError(f"{len(files)} proxy frames, shot has {T}")
        max_frames = opt.get("max_frames", 40)  # 40 x 1080p keyframes: 13.3 GB; refinement registers the rest
        keys = np.arange(T) if T <= max_frames else np.unique(np.linspace(0, T - 1, max_frames).round().astype(int))
        images = [cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB) for i in keys]

        model = load_model()
        with torch.no_grad():
            pred = model.inference(images, process_res=opt.get("process_res", 504),
                                   ref_view_strategy=opt.get("ref_view", "saddle_balanced"))
        ext = np.asarray(pred.extrinsics, np.float64)  # [N,3,4] world->camera, OpenCV
        ixt = np.asarray(pred.intrinsics, np.float64)  # [N,3,3] at processed size
        proc = np.asarray(pred.processed_images)
        hp, wp = proc.shape[1:3]
        S = np.diag([W / wp, H / hp, 1.0])

        K = np.tile(np.eye(3), (T, 1, 1)); dist = np.zeros((T, 5))
        R = np.tile(np.eye(3), (T, 1, 1)); t = np.zeros((T, 3)); valid = np.zeros(T, bool)
        K[keys] = S @ ixt
        R[keys], t[keys] = ext[:, :, :3], ext[:, :, 3]
        valid[keys] = True

        # point cloud: unproject confident depth, subsampled
        depth = np.asarray(pred.depth, np.float32)  # [N,hp,wp]
        conf = np.asarray(pred.conf, np.float32) if getattr(pred, "conf", None) is not None else np.ones_like(depth)
        thr = np.percentile(conf, 60)
        ys, xs = np.mgrid[0:hp, 0:wp]
        pts, cols = [], []
        for k in range(len(keys)):
            m = conf[k] >= thr
            z = depth[k][m]
            u, v = xs[m] + 0.5, ys[m] + 0.5  # processed pixel centres, corner origin
            Kinv = np.linalg.inv(ixt[k])
            rays = (Kinv @ np.stack([u, v, np.ones_like(u)]).reshape(3, -1)).T * z[:, None]
            Rk, tk = ext[k, :, :3], ext[k, :, 3]
            pts.append((rays - tk) @ Rk)
            cols.append(proc[k][m])
        pts, cols = np.concatenate(pts), np.concatenate(cols)
        rng = np.random.default_rng(0)
        keep = rng.choice(len(pts), min(len(pts), opt.get("max_points", 300_000)), replace=False)
        write_ply(out_dir / "points.ply", pts[keep], cols[keep])

        meta = {"method": "da3", "checkpoint": CKPT_DIR.name, "process_size": [int(wp), int(hp)],
                "keyframes": keys.tolist(), "intrinsics_mode": "per_frame"}
        save_camera_npz(out_dir / "camera_raw.npz", K, dist, R, t, valid, meta)
        res.outputs.update({"camera": "camera_raw.npz", "points": "points.ply"})
        res.stats.update({"n_keyframes": int(len(keys)), "process_size": [int(wp), int(hp)],
                          "focal_px_full_median": float(np.median(K[keys, 0, 0]))})


if __name__ == "__main__":
    main()
