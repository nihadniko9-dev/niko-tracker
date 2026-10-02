"""Benchmark metrics.

All functions take camera-to-world rotations (`R_c2w`, [T,3,3]) and camera
centres (`C`, [T,3]) unless noted, and only the frames both tracks marked valid.

Definitions
-----------
ATE        RMSE of camera centres after Sim(3) Umeyama alignment of the estimated
           centres onto ground truth, in % of `scene_size`. If the ground-truth path
           is (nearly) collinear the Sim(3) rotation about the path is not defined
           by positions, so the rotation comes from the orientations
           (`align_rotations`) and only scale + translation are fitted to positions.
           If the ground-truth path is static (tripod), no scale is observable: the
           estimate is compared by its own spread relative to `est_scene_size`.
Rotation   Per-frame geodesic angle after the best global rotation between the two
           orientation sets (gauge-free, well conditioned for any path), degrees.
RPE        Relative pose error over a frame step `delta`: rotation in degrees and
           translation in % of `scene_size` (estimate scaled by the ATE scale).
Focal      |f_est - f_gt| / f_gt in %, per frame (fx).
Jitter     Translation: RMS of the second difference of centres / `scale`, in %.
           Rotation: RMS angle of (dR_i dR_{i-1}^T), dR_i = R_{i+1} R_i^T, degrees.
           Needs no ground truth; used by auto-select.
"""

from __future__ import annotations

import numpy as np

from .geometry import project, rotation_angle_deg
from .sim3 import Sim3, align_rotations, is_collinear, umeyama


def align_sim3_trajectory(
    C_est: np.ndarray,
    C_gt: np.ndarray,
    R_est_c2w: np.ndarray | None = None,
    R_gt_c2w: np.ndarray | None = None,
) -> tuple[Sim3, str]:
    """Alignment that maps the estimate into the ground-truth frame. Returns (sim3, mode)."""
    C_est = np.asarray(C_est, dtype=np.float64)
    C_gt = np.asarray(C_gt, dtype=np.float64)
    gt_spread = np.linalg.norm(C_gt - C_gt.mean(axis=0), axis=1).max()
    est_spread = np.linalg.norm(C_est - C_est.mean(axis=0), axis=1).max()
    if gt_spread < 1e-9:
        mode = "static"
    elif est_spread < 1e-12:
        mode = "est_static"  # rotation-only estimate of a moving camera: its error is the GT motion
    elif is_collinear(C_gt):
        mode = "collinear"
    else:
        mode = "umeyama"

    if mode == "umeyama":
        return umeyama(C_est, C_gt, with_scale=True), mode

    if R_est_c2w is None or R_gt_c2w is None:
        raise ValueError(f"{mode} trajectory: orientations are needed to align")
    A = align_rotations(R_est_c2w, R_gt_c2w)
    x = (C_est - C_est.mean(axis=0)) @ A.T
    y = C_gt - C_gt.mean(axis=0)
    if mode in ("static", "est_static"):
        s = 1.0
    else:
        denom = float(np.sum(x * x))
        s = float(np.sum(x * y) / denom) if denom > 1e-18 else 1.0
    t = C_gt.mean(axis=0) - s * A @ C_est.mean(axis=0)
    return Sim3(s, A, t), mode


def ate(
    C_est: np.ndarray,
    C_gt: np.ndarray,
    scene_size: float,
    R_est_c2w: np.ndarray | None = None,
    R_gt_c2w: np.ndarray | None = None,
    est_scene_size: float | None = None,
) -> dict:
    sim3, mode = align_sim3_trajectory(C_est, C_gt, R_est_c2w, R_gt_c2w)
    if mode == "static":
        if est_scene_size is None:
            raise ValueError("static ground truth: est_scene_size is required")
        err = np.linalg.norm(C_est - C_est.mean(axis=0), axis=1)
        pct = 100.0 * err / est_scene_size
    else:
        err = np.linalg.norm(sim3.apply(C_est) - C_gt, axis=1)
        pct = 100.0 * err / scene_size
    return {
        "rmse_pct": float(np.sqrt(np.mean(pct**2))),
        "mean_pct": float(pct.mean()),
        "max_pct": float(pct.max()),
        "mode": mode,
        "scale": sim3.s,
        "sim3": sim3,
    }


def rotation_errors_deg(R_est_c2w: np.ndarray, R_gt_c2w: np.ndarray) -> np.ndarray:
    A = align_rotations(R_est_c2w, R_gt_c2w)
    return rotation_angle_deg(np.einsum("ij,njk->nik", A, R_est_c2w), R_gt_c2w)


def rpe(
    C_est: np.ndarray,
    R_est_c2w: np.ndarray,
    C_gt: np.ndarray,
    R_gt_c2w: np.ndarray,
    scene_size: float,
    scale: float,
    delta: int = 1,
) -> dict:
    """Relative pose error between frames i and i+delta."""
    if len(C_gt) <= delta:
        raise ValueError("trajectory shorter than delta")
    # relative motion expressed in the first camera's frame: invariant to the global gauge
    def rel(C, Rc):
        dR = np.einsum("nji,njk->nik", Rc[:-delta], Rc[delta:])  # R_i^T R_{i+d}
        dC = np.einsum("nji,nj->ni", Rc[:-delta], C[delta:] - C[:-delta])
        return dR, dC

    dR_e, dC_e = rel(np.asarray(C_est) * scale, R_est_c2w)
    dR_g, dC_g = rel(np.asarray(C_gt), R_gt_c2w)
    rot = rotation_angle_deg(dR_e, dR_g)
    trans = 100.0 * np.linalg.norm(dC_e - dC_g, axis=1) / scene_size
    return {
        "delta": delta,
        "rot_deg_rmse": float(np.sqrt(np.mean(rot**2))),
        "trans_pct_rmse": float(np.sqrt(np.mean(trans**2))),
    }


def focal_error_pct(f_est: np.ndarray, f_gt: np.ndarray) -> np.ndarray:
    f_est = np.asarray(f_est, dtype=np.float64)
    f_gt = np.asarray(f_gt, dtype=np.float64)
    return 100.0 * np.abs(f_est - f_gt) / f_gt


def reprojection_errors(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray, uv: np.ndarray, dist=None
) -> np.ndarray:
    """Per-point pixel error of X [N,3] against observations uv [N,2] in one frame."""
    proj, depth = project(K, R, t, X, dist)
    err = np.linalg.norm(proj - np.asarray(uv), axis=1)
    err[depth <= 0] = np.inf  # behind the camera counts as a failure, never as a small error
    return err


def jitter(C: np.ndarray, R_c2w: np.ndarray, scale: float) -> dict:
    """High-frequency camera-path energy. `scale` sets the translation unit (e.g. median scene depth)."""
    C = np.asarray(C, dtype=np.float64)
    if len(C) < 3:
        return {"trans_pct_rms": 0.0, "rot_deg_rms": 0.0}
    acc = C[2:] - 2.0 * C[1:-1] + C[:-2]
    trans = 100.0 * np.linalg.norm(acc, axis=1) / scale
    dR = np.einsum("nij,nkj->nik", R_c2w[1:], R_c2w[:-1])  # R_{i+1} R_i^T
    rot = rotation_angle_deg(dR[1:], dR[:-1])
    return {
        "trans_pct_rms": float(np.sqrt(np.mean(trans**2))),
        "rot_deg_rms": float(np.sqrt(np.mean(rot**2))),
    }


def success_rate(valid: np.ndarray) -> float:
    valid = np.asarray(valid, dtype=bool)
    return float(valid.mean()) if valid.size else 0.0
