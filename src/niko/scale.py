"""Real-world (metric) scale of a solve.

A camera solve from one video has no unit: the whole scene can be any size. UniDepth v2, which
MegaSaM runs on every keyframe (candidates/megasam/work/metric/<scene>/<j>.npz: depth in metres
along the view axis, at a reduced size), measures depth in metres from the picture itself. For
every SIFT track the solve triangulated, depth in metres / depth in solve units at the pixel where
the track was seen is one estimate of the scale; the median over each frame, then over frames, is
the solve's metres per unit. A tripod solve has no depth and needs none.

Depth from one picture is an estimate (synthetic benchmark: see PROGRESS.md 2026-10-02); the
add-on also lets the user set the size from a known distance, which is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camio import CameraTrack
from .triangulate import triangulate_tracks

MIN_POINTS_PER_FRAME = 20
# the scene is put in metres only when the two depth models agree this well. Measured 2026-10-02:
# synthetic shots 6-152 % apart (median 32 %), Nihad's real clips 46-1984 % apart (drone footage
# worst): single-image metric depth is too uncertain to trust alone, the add-on's "Set real size"
# (a known distance or the camera height) is the exact way
AGREE_MAX_PCT = 50.0


def _metric_frames(megasam_dir: Path) -> dict[int, Path]:
    """frame index -> UniDepth file of that frame (MegaSaM runs on every `stride`-th frame)."""
    files = sorted((megasam_dir / "work" / "metric").glob("*/*.npz"))
    if not files:
        return {}
    stride = 1
    res = megasam_dir / "result.json"
    if res.exists():
        stride = max(1, int(json.loads(res.read_text()).get("stats", {}).get("stride", 1) or 1))
    return {int(f.stem) * stride: f for f in files}


def _da3_depth_maps(da3_dir: Path, width: int, height: int, cell: int = 4) -> dict[int, np.ndarray]:
    """frame -> sparse depth (metres) at 1/cell size: DA3-Nested's metric points (points.ply, its own
    world) seen from its own cameras, nearest point per cell. DA3 keeps no depth maps itself."""
    from .plyio import read_ply_xyz

    cam, ply = da3_dir / "cameras.json", da3_dir / "points.ply"
    if not cam.exists() or not ply.exists():
        return {}
    c = CameraTrack.load(cam)
    K, R, t, valid = c.K, c.R, c.t, c.valid
    P = read_ply_xyz(ply)
    h, w = height // cell, width // cell
    out = {}
    for f in np.nonzero(valid)[0]:
        c = P @ R[f].T + t[f]
        z = c[:, 2]
        m = z > 1e-6
        u = (K[f, 0, 0] * c[m, 0] / z[m] + K[f, 0, 2]) / cell
        v = (K[f, 1, 1] * c[m, 1] / z[m] + K[f, 1, 2]) / cell
        zi = z[m]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui, vi, zi = u[inside].astype(int), v[inside].astype(int), zi[inside]
        img = np.full(h * w, np.inf)
        np.minimum.at(img, vi * w + ui, zi)
        out[int(f)] = img.reshape(h, w)
    return out


def _depth_at(img: np.ndarray, u: np.ndarray, v: np.ndarray, radius: int = 1) -> np.ndarray:
    """Smallest finite depth in a (2 radius + 1)^2 window around each cell (sparse maps)."""
    h, w = img.shape
    best = np.full(len(u), np.inf)
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            best = np.minimum(best, img[np.clip(v + dv, 0, h - 1), np.clip(u + du, 0, w - 1)])
    return best


def _ratios(trk: CameraTrack, xy, vis, X, ok, frames, depth_of) -> list[float]:
    """Per frame: median of (depth in metres / depth in solve units) at the frame's tracks."""
    per_frame = []
    for t in frames:
        m = vis[t] & ok
        if m.sum() < MIN_POINTS_PER_FRAME:
            continue
        z = (X[m] @ trk.R[t].T + trk.t[t])[:, 2]
        dm = depth_of(t, xy[t, m])
        good = (z > 0) & np.isfinite(dm) & (dm > 0)
        if good.sum() >= MIN_POINTS_PER_FRAME:
            per_frame.append(float(np.median(dm[good] / z[good])))
    return per_frame


def _summary(per_frame: list[float], source: str) -> dict | None:
    if len(per_frame) < 3:
        return None
    s = float(np.median(per_frame))
    q1, q3 = np.percentile(per_frame, [25, 75])
    return {"metres_per_unit": s, "spread_pct": round(float(100 * (q3 - q1) / s), 2),
            "frames": len(per_frame), "source": source}


def estimate_metric_scale(trk: CameraTrack, xy: np.ndarray, vis: np.ndarray, solve_dir: Path,
                          max_frames: int = 60, max_tracks: int = 6000, seed: int = 0) -> dict | None:
    """Metres per solve unit from two independent single-image metric depth models at the solve's
    own tracks: UniDepth v2 (MegaSaM's keyframes) and DA3-Nested (the DA3 candidate). None for a
    tripod (no depth) or when neither model ran. xy [T,N,2] full-frame pixels, vis [T,N].
    'agree_pct': how far the two are apart; the combined value is their geometric mean."""
    if trk.extra.get("rotation_only"):
        return None
    cands = Path(solve_dir) / "candidates"
    rng = np.random.default_rng(seed)
    N = vis.shape[1]
    cols = rng.permutation(N)[:max_tracks] if N > max_tracks else np.arange(N)
    xy, vis = xy[:, cols], vis[:, cols]
    X, ok = None, None
    found = {}

    def pick(frames):
        frames = [t for t in sorted(frames) if t < trk.n_frames and trk.valid[t]]
        if len(frames) > max_frames:
            frames = [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))]
        return frames

    uni = _metric_frames(cands / "megasam")
    da3 = _da3_depth_maps(cands / "da3", trk.width, trk.height)
    if not uni and not da3:
        return None
    X, ok = triangulate_tracks(trk, xy, vis)

    if uni:
        cache = {}

        def uni_depth(t, uv):
            if t not in cache:
                cache[t] = np.load(uni[t])["depth"].astype(np.float64)
            d = cache.pop(t)
            h, w = d.shape
            u = np.clip((uv[:, 0] * w / trk.width).astype(int), 0, w - 1)
            v = np.clip((uv[:, 1] * h / trk.height).astype(int), 0, h - 1)
            return d[v, u]
        found["unidepth_v2"] = _summary(_ratios(trk, xy, vis, X, ok, pick(uni), uni_depth), "unidepth_v2")
    if da3:
        def da3_depth(t, uv):
            return _depth_at(da3[t], (uv[:, 0] / 4).astype(int), (uv[:, 1] / 4).astype(int))
        found["da3_nested"] = _summary(_ratios(trk, xy, vis, X, ok, pick(da3), da3_depth), "da3_nested")
    found = {k: v for k, v in found.items() if v}
    if not found:
        return None
    vals = [v["metres_per_unit"] for v in found.values()]
    s = float(np.exp(np.mean(np.log(vals))))
    agree = round(float(100 * (max(vals) / min(vals) - 1)), 1) if len(vals) > 1 else None
    return {"metres_per_unit": s, "agree_pct": agree, "reliable": agree is not None and agree <= AGREE_MAX_PCT,
            "models": found}
