"""COLMAP / Ceres bundle adjustment of a prepared problem (task "refine").

Input  <out_dir>/ba_input.npz  (written by niko.refine.prepare):
  frames [F] int, K [F,3,3], dist [F,5], R [F,3,3], t [F,3]   initial cameras (full-res, corner origin)
  obs_f [M] int (index into frames), obs_p [M] int (point index), uv [M,2]   observations
  X [P,3]                                                             initial points
Options: intrinsics shared_focal | per_frame_focal | fixed ; distortion none | k1 | k1k2 ;
         loss_scale_px (Cauchy), outlier_px, rounds, max_iterations
Output <out_dir>/ba_output.npz: K, dist, R, t [F...], X [P,3], used [M] bool, err [M] (px)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pycolmap

MODEL = {"none": "SIMPLE_PINHOLE", "k1": "SIMPLE_RADIAL", "k1k2": "RADIAL"}


def _params(model, f, cx, cy, d):
    return {"SIMPLE_PINHOLE": [f, cx, cy], "SIMPLE_RADIAL": [f, cx, cy, d[0]],
            "RADIAL": [f, cx, cy, d[0], d[1]]}[model]


def _read_camera(cam):
    p = np.asarray(cam.params, float)
    d = np.zeros(5)
    d[:len(p) - 3] = p[3:]
    return p[0], p[1], p[2], d


def _project(K, dist, R, t, X):
    Xc = X @ R.T + t
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    s = 1 + r2 * (dist[0] + r2 * dist[1])
    return np.stack([K[0, 0] * x * s + K[0, 2], K[1, 1] * y * s + K[1, 2]], 1), Xc[:, 2]


def run(job: dict, res) -> None:
    out_dir = Path(job["out_dir"])
    opt = job.get("options", {})
    d = np.load(out_dir / "ba_input.npz")
    frames, K0, dist0, R0, t0 = d["frames"], d["K"], d["dist"], d["R"], d["t"]
    obs_f, obs_p, uv, X0 = d["obs_f"], d["obs_p"], d["uv"].astype(float), d["X"]
    F, P, M = len(frames), len(X0), len(obs_f)
    W, H = int(opt["width"]), int(opt["height"])
    model = MODEL[opt.get("distortion", "none")]
    per_frame = opt.get("intrinsics", "shared_focal") == "per_frame_focal"
    used = np.ones(M, bool)
    K, dist, R, t, X = K0.copy(), dist0.copy(), R0.copy(), t0.copy(), X0.copy()
    rounds = []
    for rnd in range(int(opt.get("rounds", 2)) + 1):
        rec = pycolmap.Reconstruction()
        cam_ids = []
        for i in range(F if per_frame else 1):
            cam = pycolmap.Camera(model=model, width=W, height=H, camera_id=i + 1,
                                  params=_params(model, float(K[i if per_frame else 0, 0, 0]),
                                                 float(K[0, 0, 2]), float(K[0, 1, 2]), dist[0]))
            rec.add_camera_with_trivial_rig(cam)
            cam_ids.append(i + 1)
        m = used
        order = np.argsort(obs_f[m], kind="stable")
        of, op, ouv = obs_f[m][order], obs_p[m][order], uv[m][order]
        starts = np.searchsorted(of, np.arange(F + 1))
        idx_in_image = np.empty(len(of), int)
        for i in range(F):
            a, b = starts[i], starts[i + 1]
            idx_in_image[a:b] = np.arange(b - a)
            img = pycolmap.Image(name=f"{int(frames[i]):06d}", keypoints=ouv[a:b],
                                 camera_id=cam_ids[i if per_frame else 0], image_id=i + 1)
            pose = pycolmap.Rigid3d(pycolmap.Rotation3d(R[i]), t[i])
            rec.add_image_with_trivial_frame(img, pose)
        # tracks: group observations by point
        porder = np.argsort(op, kind="stable")
        pstarts = np.searchsorted(op[porder], np.arange(P + 1))
        pid_of = {}
        for p in range(P):
            a, b = pstarts[p], pstarts[p + 1]
            if b - a < 2:
                continue
            els = [pycolmap.TrackElement(int(of[porder[k]]) + 1, int(idx_in_image[porder[k]])) for k in range(a, b)]
            pid_of[p] = rec.add_point3D(X[p], pycolmap.Track(els))
        ba = pycolmap.BundleAdjustmentOptions()
        ba.refine_focal_length = opt.get("intrinsics", "shared_focal") != "fixed"
        ba.refine_principal_point = False
        ba.refine_extra_params = model != "SIMPLE_PINHOLE"
        ba.print_summary = False
        ba.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
        ba.ceres.loss_function_scale = float(opt.get("loss_scale_px", 1.0))
        ba.ceres.solver_options.max_num_iterations = int(opt.get("max_iterations", 100))
        cfg = pycolmap.BundleAdjustmentConfig()
        for i in range(F):
            cfg.add_image(i + 1)
        cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        adj = pycolmap.create_default_bundle_adjuster(ba, cfg, rec)
        summary = adj.solve()
        # read back
        for i in range(F):
            img = rec.image(i + 1)
            cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            Mx = np.asarray(cfw.matrix())
            R[i], t[i] = Mx[:, :3], Mx[:, 3]
            f, cx, cy, dd = _read_camera(rec.camera(cam_ids[i if per_frame else 0]))
            K[i] = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
            dist[i] = dd
        for p, pid in pid_of.items():
            X[p] = rec.point3D(pid).xyz
        # residuals of every observation (also the ones dropped earlier)
        err = np.full(M, np.inf)
        for i in range(F):
            sel = obs_f == i
            if sel.any():
                proj, z = _project(K[i], dist[i], R[i], t[i], X[obs_p[sel]])
                e = np.linalg.norm(proj - uv[sel], axis=1)
                e[z <= 0] = np.inf
                err[sel] = e
        eu = err[used & np.isfinite(err)]
        med = float(np.median(eu))
        mad = 1.4826 * float(np.median(np.abs(eu - med)))
        thr = max(float(opt.get("outlier_px", 3.0)), med + 4 * mad)
        rounds.append({"observations": int(used.sum()), "points": len(pid_of), "median_px": med,
                       "rms_px": float(np.sqrt(np.mean(eu ** 2))), "threshold_px": thr,
                       "termination": str(getattr(summary, "termination_type", "")),
                       "iterations": int(getattr(summary, "num_iterations", -1))})
        if rnd < int(opt.get("rounds", 2)):
            used &= err <= thr
    np.savez(out_dir / "ba_output.npz", K=K, dist=dist, R=R, t=t, X=X, used=used, err=err)
    (out_dir / "ba_rounds.json").write_text(json.dumps(rounds, indent=1))
    res.outputs["ba"] = "ba_output.npz"
    res.stats.update({"rounds": rounds, "frames": F, "points": P})
