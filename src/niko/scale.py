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


# real size from the drone's own telemetry (niko.telemetry). Measured 2026-10-02 on Nihad's Air 3S
# clips: GPS against the solved camera path, 0079: 133 m path, 0.36 m median residual, the two
# halves of the flight 1.1 % apart; 0148: 143 m, 0.26 m, 0.8 % apart, and its barometric altitude
# (17.6 m descent) 1.8 % from the GPS scale. The depth models were 2.4x and 1.7x off on the same clips.
GPS_MIN_EXTENT_M = 20.0        # consumer GPS is good to well under a metre over a short flight
GPS_MAX_RESIDUAL_FRAC = 0.03   # median residual / path extent
HALVES_MAX_PCT = 5.0           # the two halves of the flight must give the same scale
ALT_MIN_CHANGE_M = 5.0         # barometric altitude is reported in 0.1 m steps
MAX_UNCERTAINTY_PCT = 15.0     # used for the scene only up to this (still far better than the depth models)
MIN_UNCERTAINTY_PCT = 1.0


def _halves_pct(a: float, b: float) -> float:
    return 100.0 * abs(a - b) / (0.5 * (a + b))


def telemetry_scale(trk: CameraTrack, tel: dict) -> dict | None:
    """Metres per solve unit from the drone's GPS track (all three axes) or, without GPS, from its
    altitude changes, with an uncertainty and whether it is reliable; None without usable telemetry."""
    from .sim3 import umeyama
    from .telemetry import align_rotation, enu, gimbal_rotations, updates

    if trk.extra.get("rotation_only"):  # a tripod solve has no camera path to measure
        return None
    n = min(trk.n_frames, tel.get("n", 0))
    out = {}
    if tel.get("lat") is not None and tel.get("alt") is not None:
        u = updates(tel["lat"][:n])
        u = u[trk.valid[u] & np.isfinite(tel["alt"][u])]
        if len(u) >= 10:
            E = enu(tel["lat"][u], tel["lon"][u], tel["alt"][u])
            C = trk.centers[u]
            extent = float(np.linalg.norm(np.ptp(E, 0)))
            if extent > 1.0:
                sim = umeyama(C, E, with_scale=True)
                r = np.linalg.norm(sim.apply(C) - E, axis=1)
                h = len(u) // 2
                s1 = umeyama(C[:h], E[:h], with_scale=True).s if h >= 5 else np.nan
                s2 = umeyama(C[h:], E[h:], with_scale=True).s if len(u) - h >= 5 else np.nan
                halves = _halves_pct(s1, s2) if np.isfinite(s1) and np.isfinite(s2) else None
                res = float(np.median(r))
                unc = max(MIN_UNCERTAINTY_PCT, (halves or 0.0) / 2, 200.0 * res / extent)
                out["gps"] = {"metres_per_unit": float(sim.s), "fixes": int(len(u)), "extent_m": round(extent, 2),
                              "residual_median_m": round(res, 3), "halves_apart_pct": None if halves is None else round(halves, 2),
                              "uncertainty_pct": round(unc, 2),
                              "reliable": bool(extent >= GPS_MIN_EXTENT_M and res <= GPS_MAX_RESIDUAL_FRAC * extent
                                               and halves is not None and halves <= HALVES_MAX_PCT
                                               and unc <= MAX_UNCERTAINTY_PCT)}
    if tel.get("rel_alt") is not None:
        ua = updates(tel["rel_alt"][:n])
        ua = ua[trk.valid[ua]]
        hgt = tel["rel_alt"][ua]
        change = float(np.ptp(hgt)) if len(ua) else 0.0
        up = None
        Rg = gimbal_rotations(tel)
        if Rg is not None:
            al = align_rotation(trk, Rg)
            if al is not None and al[1] < 2.0:
                up = al[0].T @ np.array([0.0, 0.0, 1.0])  # true up in the solve's world
        if up is not None and len(ua) >= 10 and change > 0.5:
            z = trk.centers[ua] @ up
            k, c = np.polyfit(z, hgt, 1)
            res = float(np.median(np.abs(k * z + c - hgt)))
            h = len(ua) // 2
            halves = None
            if np.ptp(hgt[:h]) > 0.3 * change and np.ptp(hgt[h:]) > 0.3 * change:
                halves = _halves_pct(np.polyfit(z[:h], hgt[:h], 1)[0], np.polyfit(z[h:], hgt[h:], 1)[0])
            unc = max(MIN_UNCERTAINTY_PCT, 100.0 * 0.1 / change, 200.0 * res / change, (halves or 0.0) / 2)
            out["altitude"] = {"metres_per_unit": float(k), "readings": int(len(ua)), "change_m": round(change, 2),
                               "residual_median_m": round(res, 3),
                               "halves_apart_pct": None if halves is None else round(halves, 2),
                               "uncertainty_pct": round(unc, 2),
                               "reliable": bool(k > 0 and change >= ALT_MIN_CHANGE_M and res <= 0.05 * change
                                                and unc <= MAX_UNCERTAINTY_PCT)}
    if not out:
        return None
    pick = "gps" if out.get("gps", {}).get("reliable") else "altitude" if out.get("altitude", {}).get("reliable") else None
    best = out[pick] if pick else (out.get("gps") or out.get("altitude"))
    result = {"metres_per_unit": best["metres_per_unit"], "source": pick or ("gps" if "gps" in out else "altitude"),
              "uncertainty_pct": best["uncertainty_pct"], "reliable": pick is not None, **out}
    if "gps" in out and "altitude" in out and out["altitude"]["metres_per_unit"] > 0:
        result["gps_vs_altitude_pct"] = round(_halves_pct(out["gps"]["metres_per_unit"],
                                                          out["altitude"]["metres_per_unit"]), 2)
    return result
