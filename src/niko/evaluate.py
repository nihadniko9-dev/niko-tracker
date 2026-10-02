"""Score an estimated cameras.json against ground truth (benchmark metrics, docs: metrics.py)."""

from __future__ import annotations

import numpy as np

from . import metrics
from .camio import CameraTrack


def evaluate_against_gt(est: CameraTrack, gt: CameraTrack, est_scene_size: float | None = None) -> dict:
    if est.n_frames != gt.n_frames:
        raise ValueError(f"frame count differs: est {est.n_frames}, gt {gt.n_frames}")
    if (est.width, est.height) != (gt.width, gt.height):
        raise ValueError("image size differs between estimate and ground truth")
    scene_size = float(gt.extra["scene_size"])
    ok = est.valid & gt.valid
    out: dict = {"success_rate": metrics.success_rate(est.valid), "n_valid": int(ok.sum()),
                 "n_frames": int(gt.n_frames), "scene_size": scene_size}
    if ok.sum() < 3:
        out["failed"] = True
        return out

    C_e, C_g = est.centers[ok], gt.centers[ok]
    R_e, R_g = est.R_c2w[ok], gt.R_c2w[ok]
    a = metrics.ate(C_e, C_g, scene_size, R_e, R_g, est_scene_size=est_scene_size)
    rot = metrics.rotation_errors_deg(R_e, R_g)
    focal = metrics.focal_error_pct(est.K[ok, 0, 0], gt.K[ok, 0, 0])
    out.update({
        "failed": False,
        "ate_pct": a["rmse_pct"], "ate_max_pct": a["max_pct"], "ate_mode": a["mode"], "sim3_scale": a["scale"],
        "rot_err_deg_mean": float(rot.mean()), "rot_err_deg_max": float(rot.max()),
        "focal_err_pct_mean": float(focal.mean()), "focal_err_pct_max": float(focal.max()),
    })
    # relative drift only over runs of consecutive valid frames
    if ok.all() and a["mode"] != "static":
        for d in (1, 10):
            if gt.n_frames > d:
                r = metrics.rpe(C_e, R_e, C_g, R_g, scene_size, a["scale"], delta=d)
                out[f"rpe{d}_rot_deg"] = r["rot_deg_rmse"]
                out[f"rpe{d}_trans_pct"] = r["trans_pct_rmse"]
    depth = float(gt.extra.get("median_depth") or scene_size / 2)
    j_e = metrics.jitter(a["sim3"].apply(C_e), R_e, depth)
    j_g = metrics.jitter(C_g, R_g, depth)
    out.update({"jitter_trans_pct": j_e["trans_pct_rms"], "jitter_rot_deg": j_e["rot_deg_rms"],
                "gt_jitter_trans_pct": j_g["trans_pct_rms"], "gt_jitter_rot_deg": j_g["rot_deg_rms"]})
    out["meets_targets"] = bool(out["rot_err_deg_max"] < 0.2 and out["focal_err_pct_max"] < 2.0
                                and out["ate_pct"] < 1.0 and out["success_rate"] == 1.0)
    return out


def summary_line(name: str, r: dict) -> str:
    if r.get("failed"):
        return f"{name:<22} FAILED ({r['n_valid']}/{r['n_frames']} frames)"
    return (f"{name:<22} ok {r['success_rate'] * 100:5.1f}%  ATE {r['ate_pct']:.3f}%  "
            f"rot {r['rot_err_deg_mean']:.3f}/{r['rot_err_deg_max']:.3f} deg  "
            f"focal {r['focal_err_pct_mean']:.2f}/{r['focal_err_pct_max']:.2f}%  "
            f"{'TARGETS MET' if r['meets_targets'] else 'targets missed'}")
