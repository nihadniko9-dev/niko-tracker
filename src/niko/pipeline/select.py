"""Stage 6 - auto-select: one neutral score for every candidate, no ground truth needed.

score_px = median held-out reprojection error (px)
         + 0.5 * path jitter expressed in pixels (rotation jitter * f, translation jitter * f / depth)
Held-out tracks are triangulated with the candidate's own cameras - or, for a rotation-only
(tripod) candidate, turned into directions - and reprojected into every frame that sees them.
The held-out tracks are the SIFT tracks refinement never saw (niko.tracks.split_sift) when the
shot has at least MIN_SIFT_OBS of their observations, else CoTracker3's held-out tracks cut into
16-frame segments. CoTracker3 drift misled the choice: on zoom_in the drift is radial, a 6-dof
camera absorbs it into point depth, and a wrong dolly solution (focal 72 % off) scored 0.260 px
against 0.307 px for the true rotation + zoom; on unseen SIFT tracks both score 0.188 px and
the jitter term decides for the true one. distortion_barrel: k1k2 0.195 vs 0.321 px on SIFT,
0.243 vs 0.257 px on CoTracker3. Candidates that register fewer than 95 % of frames rank after
all others.

The jitter term also catches broken solves the reprojection cannot see: on whip_pan_blur no track
survives the whip, so each half of a broken solve is self-consistent (0.39 px) while the path
jumps; a jump shows up as a large rotation second difference.
"""

from __future__ import annotations

import numpy as np

from .. import metrics
from ..camio import CameraTrack
from ..tracks import split_tracks
from ..triangulate import reprojection_errors, triangulate_tracks, undistort_points

MIN_SUCCESS = 0.95
SEGMENT = 16
MIN_SIFT_OBS = 2000
INLIER_PX = 3.0  # held-out observations further off are mismatches or movers, not solve error


def is_rotation_only(trk: CameraTrack) -> bool:
    if trk.extra.get("rotation_only"):
        return True
    C = trk.centers[trk.valid]
    return len(C) > 1 and float(np.ptp(C, axis=0).max()) < 1e-9


def direction_errors(trk: CameraTrack, xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Rotation-only reprojection: mean back-projected ray per track, reprojected. [T, N] px."""
    T, N = vis.shape
    d = np.zeros((N, 3))
    frames = np.nonzero(trk.valid)[0]
    for t in frames:
        m = vis[t]
        if not m.any():
            continue
        u = undistort_points(xy[t, m].astype(float), trk.K[t], trk.dist[t])
        rays = (np.linalg.inv(trk.K[t]) @ np.column_stack([u, np.ones(m.sum())]).T).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        d[m] += rays @ trk.R[t]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    err = np.full((T, N), np.nan)
    for t in frames:
        m = vis[t] & (np.linalg.norm(d, axis=1) > 0)
        if not m.any():
            continue
        v = d[m] @ trk.R[t].T
        x, y = v[:, 0] / v[:, 2], v[:, 1] / v[:, 2]
        k1, k2 = trk.dist[t, 0], trk.dist[t, 1]
        r2 = x * x + y * y
        s = 1 + r2 * (k1 + r2 * k2)
        p = np.stack([trk.K[t, 0, 0] * x * s + trk.K[t, 0, 2], trk.K[t, 1, 1] * y * s + trk.K[t, 1, 2]], 1)
        e = np.linalg.norm(p - xy[t, m], axis=1)
        e[v[:, 2] <= 0] = np.inf
        err[t, m] = e
    return err


def _heldout_errors(trk: CameraTrack, xy: np.ndarray, vis: np.ndarray, rot_only: bool):
    """[T, N] reprojection errors of held-out tracks, and the triangulated points (None if rotation-only)."""
    vis = vis & trk.valid[:, None]
    if rot_only:
        return direction_errors(trk, xy, vis), None
    X, good = triangulate_tracks(trk, xy, vis)
    return reprojection_errors(trk, X, xy, vis), X[good]


def score_candidate(trk: CameraTrack, tracks: dict, sift_holdout: dict | None = None) -> dict:
    """sift_holdout: SIFT tracks refinement never saw; when given, they are the reprojection measure
    (CoTracker3's figure is kept as reproj_cotracker_px)."""
    ok = trk.valid
    out = {"success_rate": metrics.success_rate(ok)}
    if ok.sum() < 3:
        return {**out, "score_px": float("inf"), "reason": "fewer than 3 valid frames"}
    seg = split_tracks(tracks, SEGMENT) if "query_frame" in tracks else tracks
    hold = seg["holdout"]
    rot_only = is_rotation_only(trk)
    err, Xg = _heldout_errors(trk, seg["xy"][:, hold].astype(float), seg["vis"][:, hold], rot_only)
    depth = 1.0
    if Xg is not None and len(Xg):
        depth = float(np.median([np.median(np.einsum("ij,nj->ni", trk.R[t], Xg)[:, 2] + trk.t[t, 2])
                                 for t in np.nonzero(ok)[0][:: max(1, ok.sum() // 10)]]))
    e_cot = err[np.isfinite(err)]
    if sift_holdout is not None:
        err, _ = _heldout_errors(trk, sift_holdout["xy"].astype(float), sift_holdout["vis"], rot_only)
    e = err[np.isfinite(err)]
    if e.size < 50:
        return {**out, "score_px": float("inf"), "reason": f"only {e.size} held-out observations"}
    f = float(np.median(trk.K[ok, 0, 0]))
    j = metrics.jitter(trk.centers[ok], trk.R_c2w[ok], scale=max(depth, 1e-9))
    jitter_px = f * np.radians(j["rot_deg_rms"]) + (0.0 if rot_only else f * j["trans_pct_rms"] / 100.0)
    reproj = float(np.median(e))
    inl = e[e < INLIER_PX]
    return {**out, "reproj_median_px": reproj, "reproj_p90_px": float(np.percentile(e, 90)),
            "reproj_mean_px": float(inl.mean()) if inl.size else None,  # the "average error" users see
            "inlier_fraction": float(inl.size / e.size),
            "reproj_source": "sift" if sift_holdout is not None else "cotracker",
            "reproj_cotracker_px": float(np.median(e_cot)) if e_cot.size else None,
            "n_heldout_obs": int(e.size), "jitter_px": float(jitter_px), "median_depth": depth,
            "rotation_only": rot_only, "score_px": reproj + 0.5 * float(jitter_px)}


def frame_errors(trk: CameraTrack, tracks: dict, sift_holdout: dict | None = None) -> dict:
    """Per-frame held-out reprojection of one candidate (the UI's error graph): median px, mean px of
    observations within INLIER_PX, and counts. Same held-out set and rule as score_candidate."""
    sift_holdout = usable_sift_holdout(sift_holdout)
    rot_only = is_rotation_only(trk)
    if sift_holdout is not None:
        xy, vis, source = sift_holdout["xy"].astype(float), sift_holdout["vis"], "sift"
    else:
        seg = split_tracks(tracks, SEGMENT) if "query_frame" in tracks else tracks
        xy, vis, source = seg["xy"][:, seg["holdout"]].astype(float), seg["vis"][:, seg["holdout"]], "cotracker"
    err, _ = _heldout_errors(trk, xy, vis, rot_only)
    median, mean, n = [], [], []
    for t in range(trk.n_frames):
        e = err[t][np.isfinite(err[t])]
        inl = e[e < INLIER_PX]
        median.append(round(float(np.median(e)), 4) if e.size else None)
        mean.append(round(float(inl.mean()), 4) if inl.size else None)
        n.append(int(e.size))
    return {"schema": "niko.frame_errors/1", "source": source, "inlier_px": INLIER_PX,
            "frame_start": trk.frame_start, "median_px": median, "mean_px": mean, "n_obs": n}


def usable_sift_holdout(sift_holdout: dict | None) -> dict | None:
    """The held-out SIFT set if it is big enough to score on, else None (CoTracker3 fallback)."""
    if sift_holdout is None or int(sift_holdout["vis"].sum()) < MIN_SIFT_OBS:
        return None
    return sift_holdout


# a simpler camera model wins when it is within 3 % of the best on the score AND on held-out
# reprojection alone. The score alone is not enough: real camera shake shows up as jitter in every
# candidate and compresses relative gaps (distortion_barrel: k1k2 0.243 vs 0.257 px reprojection,
# 5.8 %, but 0.870 vs 0.886 score, 1.8 %, so the model without distortion won at 18.7 % focal error).
PARSIMONY = 0.03


def select(cands: dict[str, CameraTrack], tracks: dict, sift_holdout: dict | None = None) -> tuple[str | None, dict]:
    """Best score among candidates solving >= 95 % of frames, then the simplest camera model
    (extra["complexity"]: tripod 0, tripod zoom 1, BA 1, BA k1k2 2, raw 2, BA zoom 3) within 3 %
    of the best score and within 3 % of the best held-out reprojection in that group."""
    sift_holdout = usable_sift_holdout(sift_holdout)
    scores = {name: score_candidate(trk, tracks, sift_holdout) for name, trk in cands.items()}
    for name, trk in cands.items():
        scores[name]["complexity"] = trk.extra.get("complexity", 2)
    ranked = sorted(scores, key=lambda n: (scores[n]["success_rate"] < MIN_SUCCESS, scores[n]["score_px"]))
    if not ranked or not np.isfinite(scores[ranked[0]]["score_px"]):
        return None, {"ranking": ranked, "scores": scores}
    top = scores[ranked[0]]
    pool = [n for n in ranked if (scores[n]["success_rate"] >= MIN_SUCCESS) == (top["success_rate"] >= MIN_SUCCESS)
            and scores[n]["score_px"] <= top["score_px"] * (1 + PARSIMONY)]
    best_reproj = min(scores[n]["reproj_median_px"] for n in pool)
    pool = [n for n in pool if scores[n]["reproj_median_px"] <= best_reproj * (1 + PARSIMONY)]
    best = min(pool, key=lambda n: (scores[n]["complexity"], scores[n]["score_px"]))
    return best, {"ranking": ranked, "scores": scores, "best_score": ranked[0], "parsimony_pool": pool,
                  "reproj_source": "sift" if sift_holdout is not None else "cotracker",
                  "rule": __doc__.strip().splitlines()[2:8]}
