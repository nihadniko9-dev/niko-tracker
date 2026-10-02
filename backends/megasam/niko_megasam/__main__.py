"""`python -m niko_megasam job.json` - MegaSaM camera candidate (poses + shared focal).

Runs MegaSaM's own three steps unchanged (apart from port.py), each as a subprocess in a
per-job work folder so runs never share state:
  1. Depth-Anything v1 ViT-L mono disparity   (Depth-Anything/run_videos.py)
  2. UniDepth v2 ViT-L metric depth + FOV      (UniDepth/scripts/demo_mega-sam.py)
  3. DROID-based tracking with megasam_final   (camera_tracking_scripts/test_demo.py)
Step 3 writes outputs/shot_droid.npz with cam_c2w (OpenCV axes) and one K at the resized
tracking resolution (area ~384x512, as in test_demo.image_stream); K is scaled back here.
Optional CVD depth refinement is not run: it does not change the cameras.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from niko_backend_common import ResultWriter, read_job, save_camera_npz, write_ply

from .common import DA_CKPT, SRC, UNIDEPTH_DIR, WEIGHTS

SCENE = "shot"


def run(cmd: list[str], cwd: Path, log: Path, env: dict) -> None:
    with log.open("a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"step failed (rc={rc}): {' '.join(cmd[:3])} ... see {log}")


def main() -> None:
    job = read_job(sys.argv[1])
    opt = job.get("options", {})
    shot_dir, out_dir = Path(job["shot_dir"]), Path(job["out_dir"])
    commit = subprocess.check_output(["git", "-C", str(SRC), "rev-parse", "HEAD"], text=True).strip()
    with ResultWriter(job, versions={"megasam_commit": commit}) as res:
        shot = json.loads((shot_dir / "shot.json").read_text())
        W, H, T = shot["width"], shot["height"], shot["n_frames"]
        img_dir = shot_dir / "proxy"
        if len(list(img_dir.glob("*.jpg"))) != T:
            raise ValueError("proxy frame count does not match shot.json")
        h0, w0 = cv2.imread(str(img_dir / "000000.jpg")).shape[:2]

        work = out_dir / "work"
        work.mkdir(parents=True, exist_ok=True)
        stride = max(1, int(opt.get("stride", 1)))
        keys = np.arange(0, T, stride)
        if stride > 1:  # every n-th frame through a folder of links; refinement fills the others
            kdir = work / "keyframes"
            if kdir.exists():
                shutil.rmtree(kdir)
            kdir.mkdir()
            for j, i in enumerate(keys):
                os.symlink(img_dir / f"{i:06d}.jpg", kdir / f"{j:06d}.jpg")
            img_dir = kdir
        log = out_dir / "steps.log"
        log.write_text("")
        env = {**os.environ, "NIKO_UNIDEPTH_DIR": str(UNIDEPTH_DIR),
               "PYTHONPATH": os.pathsep.join([str(SRC / "base/droid_slam"), str(SRC / "UniDepth"),
                                              os.environ.get("PYTHONPATH", "")])}
        py = sys.executable
        run([py, str(SRC / "Depth-Anything/run_videos.py"), "--encoder", "vitl", "--load-from", str(DA_CKPT),
             "--img-path", str(img_dir), "--outdir", str(work / "mono" / SCENE)], work, log, env)
        run([py, str(SRC / "UniDepth/scripts/demo_mega-sam.py"), "--scene-name", SCENE,
             "--img-path", str(img_dir), "--outdir", str(work / "metric")], work, log, env)
        run([py, str(SRC / "camera_tracking_scripts/test_demo.py"), f"--datapath={img_dir}",
             f"--weights={WEIGHTS}", "--scene_name", SCENE, "--mono_depth_path", str(work / "mono"),
             "--metric_depth_path", str(work / "metric"), "--disable_vis"], work, log, env)

        d = np.load(work / "outputs" / f"{SCENE}_droid.npz")
        c2w_k = d["cam_c2w"]
        if len(c2w_k) != len(keys):
            raise RuntimeError(f"MegaSaM returned {len(c2w_k)} poses for {len(keys)} frames")
        c2w = np.tile(np.eye(4), (T, 1, 1))
        c2w[keys] = c2w_k
        valid = np.zeros(T, bool)
        valid[keys] = True
        # test_demo.image_stream: w1 = int(w0 * sqrt(384*512 / (h0*w0))), K scaled by w1/w0, h1/h0
        s = np.sqrt((384 * 512) / (h0 * w0))
        w1, h1 = int(w0 * s), int(h0 * s)
        K_track = d["intrinsic"].astype(np.float64)
        K_full = np.diag([W / w1, H / h1, 1.0]) @ K_track
        if not K_full[0, 0] > 0:  # seen on a real 60 fps drone clip: -1158 px
            raise RuntimeError(f"MegaSaM's focal went non-positive ({K_full[0, 0]:.0f} px): its solve diverged")

        R = np.transpose(c2w[:, :3, :3], (0, 2, 1))
        t = -np.einsum("nij,nj->ni", R, c2w[:, :3, 3])
        K = np.repeat(K_full[None], T, 0)
        save_camera_npz(out_dir / "camera_raw.npz", K, np.zeros((T, 5)), R, t, valid,
                        {"method": "megasam", "track_size": [w1, h1], "intrinsics_mode": "shared",
                         "cvd": False, "stride": stride})

        depths, images = d["depths"], d["images"]  # [T,h,w], [T,h,w,3] at tracking size (cropped to /8)
        rng = np.random.default_rng(0)
        pts, cols = [], []
        Kinv = np.linalg.inv(K_track)
        for k in range(0, len(keys), max(1, len(keys) // 30)):
            i = k  # depths / images are per keyframe; poses are placed at frame keys[k]
            hh, ww = depths[i].shape
            ys, xs = np.mgrid[0:hh, 0:ww]
            z = depths[i].ravel()
            m = (z > 0) & np.isfinite(z) & (rng.random(z.size) < 0.1)
            rays = (Kinv @ np.stack([xs.ravel()[m] + 0.5, ys.ravel()[m] + 0.5, np.ones(m.sum())])).T * z[m, None]
            pts.append(rays @ c2w_k[i, :3, :3].T + c2w_k[i, :3, 3])
            cols.append(images[i].reshape(-1, 3)[m])
        write_ply(out_dir / "points.ply", np.concatenate(pts), np.concatenate(cols))
        res.outputs.update({"camera": "camera_raw.npz", "points": "points.ply"})
        res.stats.update({"track_size": [w1, h1], "focal_px_full": float(K_full[0, 0]), "stride": stride,
                          "keyframes": int(len(keys))})


if __name__ == "__main__":
    main()
