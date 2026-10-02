"""Stage 5 - refinement: bundle adjustment of a camera candidate against CoTracker3 tracks.

Pipeline (orchestrator side, this module):
  1. Tracks are split into segments of `max_track_len` frames (niko.tracks: CoTracker3 drift grows
     with distance from the query frame, so each segment is its own 3D point).
  2. Non-held-out segments with >= 3 observations in registered frames are triangulated with the
     candidate cameras; frames the candidate missed are registered by RANSAC PnP; observations
     further than 20 px from their triangulated point are dropped before the adjustment.
  3. The problem goes to the COLMAP env (backends/colmap/niko_colmap/refine.py): Ceres bundle
     adjustment with COLMAP's analytic cost functions, Cauchy loss, gauge fixed on two cameras,
     outlier rounds (> max(3 px, median + 4 MAD) dropped, solved again).
Intrinsics modes: "fixed", "shared_focal", "per_frame_focal" (zoom); distortion "none", "k1",
"k1k2". The principal point stays where the candidate put it.

A first scipy (finite-difference, LSMR) version did not converge on DA3 starts (status "max
evaluations" after 300 evaluations, worst error at the fixed gauge frame), hence Ceres.
Held-out segments are never used here; auto-select scores on them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .backend import run_backend
from .camio import CameraTrack
from .tracks import split_tracks
from .triangulate import reprojection_errors, triangulate_tracks


@dataclass
class RefineOptions:
    intrinsics: str = "shared_focal"   # fixed | shared_focal | per_frame_focal
    distortion: str = "none"           # none | k1 | k1k2
    loss_scale_px: float = 1.0
    outlier_px: float = 3.0
    rounds: int = 1                    # outlier rounds after the first solve
    max_tracks: int = 4000             # CoTracker segments
    max_sift: int = 8000               # SIFT tracks (longest first), when given
    max_track_len: int = 16            # 0 = keep whole tracks
    max_iterations: int = 100
    seed: int = 0


def copy_track(c: CameraTrack) -> CameraTrack:
    t = CameraTrack(**{**c.__dict__})
    t.K, t.dist, t.R, t.t, t.valid = c.K.copy(), c.dist.copy(), c.R.copy(), c.t.copy(), c.valid.copy()
    t.extra = dict(c.extra)
    return t


def _select(tracks: dict, valid: np.ndarray, max_tracks: int, rng) -> np.ndarray:
    vis = tracks["vis"] & valid[:, None] & ~tracks["holdout"][None, :]
    counts = vis.sum(0)
    cand = np.nonzero(counts >= 3)[0]
    if len(cand) > max_tracks:  # prefer long segments, keep randomness for spatial spread
        w = counts[cand].astype(float)
        cand = rng.choice(cand, max_tracks, replace=False, p=w / w.sum())
    return np.sort(cand)


def interpolate_gaps(trk: CameraTrack) -> list[int]:
    """Frames between registered ones get interpolated cameras (rotation slerp, centre and
    intrinsics linear; ends copied): the start for keyframe-strided candidates, which bundle
    adjustment then refines. Returns the frames filled."""
    from scipy.spatial.transform import Rotation, Slerp

    v = np.nonzero(trk.valid)[0]
    miss = np.nonzero(~trk.valid)[0]
    if len(v) < 2 or not len(miss):
        return []
    C = trk.centers[v]
    rots = Rotation.from_matrix(trk.R_c2w[v])
    tq = np.clip(miss, v[0], v[-1]).astype(float)
    Rc2w = Slerp(v.astype(float), rots)(tq).as_matrix()
    Cm = np.stack([np.interp(tq, v, C[:, k]) for k in range(3)], 1)
    K = np.stack([np.stack([np.interp(tq, v, trk.K[v, i, j]) for j in range(3)], 1) for i in range(3)], 1)
    D = np.stack([np.interp(tq, v, trk.dist[v, k]) for k in range(5)], 1)
    trk.R[miss] = np.transpose(Rc2w, (0, 2, 1))
    trk.t[miss] = -np.einsum("nij,nj->ni", trk.R[miss], Cm)
    trk.K[miss], trk.dist[miss] = K, D
    trk.valid[miss] = True
    return [int(i) for i in miss]


def pnp_fill(trk: CameraTrack, X: np.ndarray, ok: np.ndarray, xy: np.ndarray, vis: np.ndarray) -> list[int]:
    """Register frames the candidate missed, from triangulated points (RANSAC PnP + LM)."""
    added = []
    if not trk.valid.any():
        return added
    K = np.median(trk.K[trk.valid], axis=0)
    dist = np.median(trk.dist[trk.valid], axis=0)
    for t in np.nonzero(~trk.valid)[0]:
        m = vis[t] & ok
        if m.sum() < 12:
            continue
        obj = X[m].astype(np.float64)
        img = (xy[t, m] - 0.5).astype(np.float64)  # OpenCV pixel-centre indices
        good, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, dist, reprojectionError=3.0,
                                                   iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        if not good or inl is None or len(inl) < 10:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl[:, 0]], img[inl[:, 0]], K, dist, rvec, tvec)
        trk.R[t] = cv2.Rodrigues(rvec)[0]
        trk.t[t] = tvec.ravel()
        trk.K[t] = K
        trk.dist[t] = dist
        trk.valid[t] = True
        added.append(int(t))
    return added


def _combined(tracks: dict, sift: dict | None, valid, opts: RefineOptions, rng):
    """CoTracker segments (non-held-out) + SIFT tracks as one [T, N] set for the adjustment."""
    sel = _select(tracks, valid, opts.max_tracks, rng)
    xy, vis = [tracks["xy"][:, sel].astype(np.float64)], [tracks["vis"][:, sel]]
    n_sift = 0
    if sift is not None:
        sv = sift["vis"] & valid[:, None]
        cnt = sv.sum(0)
        ssel = np.nonzero(cnt >= 3)[0]
        ssel = ssel[np.argsort(cnt[ssel])[::-1][: opts.max_sift]]
        xy.append(sift["xy"][:, ssel].astype(np.float64))
        vis.append(sift["vis"][:, ssel])
        n_sift = len(ssel)
    return np.concatenate(xy, 1), np.concatenate(vis, 1), n_sift


def refine(cand: CameraTrack, tracks: dict, work_dir: str | Path, opts: RefineOptions | None = None,
           sift: dict | None = None) -> tuple[CameraTrack, dict]:
    """sift: optional multi-view SIFT tracks of the shot (colmap task "sift_tracks"): no drift and
    long baselines (orbit_yard vs ray-cast GT: 0.38 px median, 0.48 px 48+ frames apart, against
    1.67 / 3.9 px for CoTracker3), so they carry the geometry; CoTracker segments add coverage."""
    opts = opts or RefineOptions()
    t_start = time.time()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.seed)
    trk = copy_track(cand)
    interpolated = []
    if int(cand.extra.get("backend_meta", {}).get("stride", 1) or 1) > 1:
        interpolated = interpolate_gaps(trk)
    if opts.max_track_len:
        tracks = split_tracks(tracks, opts.max_track_len)

    xy, vis, n_sift = _combined(tracks, sift, trk.valid, opts, rng)
    X, ok = triangulate_tracks(trk, xy, vis)
    added = pnp_fill(trk, X, ok, xy, vis)
    if added:
        xy, vis, n_sift = _combined(tracks, sift, trk.valid, opts, rng)
        X, ok = triangulate_tracks(trk, xy, vis)
    err0 = reprojection_errors(trk, X, xy, vis)
    use = vis & trk.valid[:, None] & ok[None, :] & (np.nan_to_num(err0, nan=1e9) < 20.0)
    keep_pts = np.nonzero(use.sum(0) >= 2)[0]
    use = use[:, keep_pts]
    err0 = err0[:, keep_pts]
    frames = np.nonzero(trk.valid)[0]
    fidx = -np.ones(trk.n_frames, int)
    fidx[frames] = np.arange(len(frames))
    ot, on = np.nonzero(use)
    np.savez(work_dir / "ba_input.npz", frames=frames, K=trk.K[frames], dist=trk.dist[frames], R=trk.R[frames],
             t=trk.t[frames], obs_f=fidx[ot], obs_p=on, uv=xy[:, keep_pts][ot, on], X=X[keep_pts])
    options = {**asdict(opts), "width": trk.width, "height": trk.height}
    result = run_backend("colmap", "refine", work_dir, work_dir, options)
    out = np.load(work_dir / "ba_output.npz")
    trk.K[frames], trk.dist[frames], trk.R[frames], trk.t[frames] = out["K"], out["dist"], out["R"], out["t"]
    trk.method = f"refined:{cand.method}"
    trk.intrinsics_mode = "per_frame_focal" if opts.intrinsics == "per_frame_focal" else "shared"
    rounds = result["stats"]["rounds"]
    report = {"method": cand.method, "options": asdict(opts), "frames_added_by_pnp": added,
              "frames_interpolated": len(interpolated),
              "segments_used": int(len(keep_pts)), "sift_tracks_offered": int(n_sift), "observations": int(len(ot)),
              "initial_median_px": float(np.median(err0[use & np.isfinite(err0)])) if use.any() else None,
              "rounds": rounds, "k1": float(np.median(out["dist"][:, 0])), "k2": float(np.median(out["dist"][:, 1])),
              "focal_median": float(np.median(out["K"][:, 0, 0])), "seconds": round(time.time() - t_start, 2)}
    trk.extra["refine"] = report
    (work_dir / "refine.json").write_text(json.dumps(report, indent=1))
    return trk, report
