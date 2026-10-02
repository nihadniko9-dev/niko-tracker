"""Track utilities.

CoTracker3 tracks drift with distance from their query frame (orbit_yard, vs ray-cast GT:
median 0.34 px at 1-2 frames, 0.62 px at 2-8, 1.4 px at 8-24, 2.4 px at 24-48, 3.9 px beyond).
Solvers therefore use tracks split into short segments: each segment is its own 3D point, so
drift inside a long track cannot bend the geometry.
"""

from __future__ import annotations

import numpy as np


def split_tracks(tracks: dict, max_len: int = 16) -> dict:
    """Cut every track into pieces of at most max_len frames, counted from its query frame.

    Returns a dict with the same keys (xy, vis, conf, query_frame, holdout) where each column is
    one segment; a segment inherits holdout from its parent track and keeps >= 2 observations.
    """
    xy, vis = tracks["xy"], tracks["vis"]
    T, N = vis.shape
    qf = tracks["query_frame"].astype(int)
    seg = np.floor((np.arange(T)[:, None] - qf[None, :]) / max_len).astype(int)  # [T, N]
    seg_min = seg.min()
    key = (seg - seg_min) * N + np.arange(N)[None, :]  # unique per (track, segment)
    keys = np.unique(key[vis])
    counts = np.bincount(np.searchsorted(keys, key[vis]), minlength=len(keys))
    keys = keys[counts >= 2]
    col = -np.ones(key.max() + 1, int)
    col[keys] = np.arange(len(keys))
    M = len(keys)
    out_xy = np.full((T, M, 2), np.nan, np.float32)
    out_vis = np.zeros((T, M), bool)
    out_conf = np.zeros((T, M), np.float32)
    ts, ns = np.nonzero(vis)
    cs = col[key[ts, ns]]
    keep = cs >= 0
    ts, ns, cs = ts[keep], ns[keep], cs[keep]
    out_xy[ts, cs] = xy[ts, ns]
    out_vis[ts, cs] = True
    if "conf" in tracks:
        out_conf[ts, cs] = tracks["conf"][ts, ns]
    parent = keys % N
    first = np.full(M, T, int)
    np.minimum.at(first, cs, ts)
    return {"xy": out_xy, "vis": out_vis, "conf": out_conf, "query_frame": first,
            "holdout": tracks["holdout"][parent], "parent": parent}


def split_sift(sift: dict, n_train: int = 8000, min_views: int = 3) -> tuple[dict, dict]:
    """SIFT tracks (colmap task "sift_tracks") -> (train, held out).

    Refinement gets the n_train longest tracks; every other track with >= min_views observations
    is never seen by refinement and scores the candidates in auto-select. Unlike CoTracker3
    segments, SIFT tracks do not drift (orbit_yard vs ray-cast GT: 0.48 px at 48+ frames apart).
    Raw COLMAP candidates did match these tracks, so they are not unseen for those.
    """
    cnt = sift["vis"].sum(0)
    order = np.argsort(-cnt, kind="stable")
    order = order[cnt[order] >= min_views]
    train, held = np.sort(order[:n_train]), np.sort(order[n_train:])
    return ({"xy": sift["xy"][:, train], "vis": sift["vis"][:, train]},
            {"xy": sift["xy"][:, held], "vis": sift["vis"][:, held]})


# tracks that must tie the two sides of a frame boundary together (scripts/dev/track_gaps.py,
# 2026-10-01): whip_pan_blur 0-2 across its whip; every other synthetic shot >= 604, Nihad's
# clips >= 232 (clip 02), so 50 leaves a wide margin either way
GAP_MIN_TRACKS = 50


def track_gaps(vis: np.ndarray, window: int = 3, min_tracks: int = GAP_MIN_TRACKS) -> list[dict]:
    """Breaks in tracking, from the multi-view SIFT tracks' visibility [T, N]: boundaries between
    consecutive frames that have observations (keyframes of a strided solve, frames COLMAP
    registered) crossed by fewer than min_tracks tracks seen in the `window` such frames before
    AND the `window` after. Consecutive weak boundaries merge into one break. Across a break
    nothing ties one side's camera to the other's: each side can be right and the turn between
    them off (whip_pan_blur: 2.7 deg). Returns [{"last_before", "first_after": frame index (0-based),
    "tracks": fewest crossing tracks}]."""
    rows = np.nonzero(vis.any(1))[0]
    if len(rows) < 2:
        return []
    v = vis[rows]
    counts = np.array([int((v[max(0, i - window + 1): i + 1].any(0) & v[i + 1: i + 1 + window].any(0)).sum())
                       for i in range(len(rows) - 1)])
    gaps, i = [], 0
    while i < len(counts):
        if counts[i] >= min_tracks:
            i += 1
            continue
        j = i
        while j + 1 < len(counts) and counts[j + 1] < min_tracks:
            j += 1
        gaps.append({"last_before": int(rows[i]), "first_after": int(rows[j + 1]), "tracks": int(counts[i:j + 1].min())})
        i = j + 1
    return gaps
