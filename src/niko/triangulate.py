"""Multi-view triangulation of point tracks with known cameras, and reprojection errors.

Used to (a) check tracks against ground-truth cameras and (b) score candidates on
held-out tracks (auto-select).
"""

from __future__ import annotations

import numpy as np

from .camio import CameraTrack


def projection_matrices(trk: CameraTrack) -> np.ndarray:
    """P = K [R | t] per frame, [T, 3, 4] (NaN for invalid frames)."""
    Rt = np.concatenate([trk.R, trk.t[..., None]], axis=2)
    return np.einsum("tij,tjk->tik", trk.K, Rt)


def undistort_points(uv: np.ndarray, K: np.ndarray, dist: np.ndarray, iters: int = 10) -> np.ndarray:
    """Inverse of geometry.project's OpenCV model by fixed-point iteration (pixels in, pixels out)."""
    if not np.any(dist):
        return uv
    k1, k2, p1, p2, k3 = dist
    x = (uv[..., 0] - K[0, 2]) / K[0, 0]
    y = (uv[..., 1] - K[1, 2]) / K[1, 1]
    xu, yu = x.copy(), y.copy()
    for _ in range(iters):
        r2 = xu * xu + yu * yu
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
        dy = p1 * (r2 + 2 * yu * yu) + 2 * p2 * xu * yu
        xu = (x - dx) / radial
        yu = (y - dy) / radial
    return np.stack([xu * K[0, 0] + K[0, 2], yu * K[1, 1] + K[1, 2]], -1)


def triangulate_tracks(trk: CameraTrack, xy: np.ndarray, vis: np.ndarray, min_views: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """DLT triangulation of every track. xy [T,N,2], vis [T,N]. Returns X [N,3] and ok [N]."""
    T, N = vis.shape
    P = projection_matrices(trk)
    use = vis & trk.valid[:, None]
    uv = np.empty_like(xy, dtype=np.float64)
    for t in range(T):
        uv[t] = undistort_points(xy[t].astype(np.float64), trk.K[t], trk.dist[t]) if trk.valid[t] else np.nan
    X = np.full((N, 3), np.nan)
    ok = np.zeros(N, bool)
    counts = use.sum(0)
    for n in np.nonzero(counts >= min_views)[0]:
        ts = np.nonzero(use[:, n])[0]
        A = np.concatenate([uv[ts, n, 0:1] * P[ts, 2] - P[ts, 0], uv[ts, n, 1:2] * P[ts, 2] - P[ts, 1]])
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]
        if abs(h[3]) < 1e-12:
            continue
        Xn = h[:3] / h[3]
        depth = np.einsum("tj,j->t", trk.R[ts, 2], Xn) + trk.t[ts, 2]
        if np.all(depth > 0):
            X[n] = Xn
            ok[n] = True
    return X, ok


def reprojection_errors(trk: CameraTrack, X: np.ndarray, xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Pixel error for every visible observation of every triangulated track: [T, N] (NaN elsewhere)."""
    from .geometry import project

    T, N = vis.shape
    err = np.full((T, N), np.nan)
    good = ~np.isnan(X[:, 0])
    for t in range(T):
        if not trk.valid[t]:
            continue
        m = vis[t] & good
        if not m.any():
            continue
        uv, z = project(trk.K[t], trk.R[t], trk.t[t], X[m], trk.dist[t])
        e = np.linalg.norm(uv - xy[t, m], axis=1)
        e[z <= 0] = np.inf
        err[t, m] = e
    return err
