"""Lock test: the solve's 3D points drawn over the footage, as an MP4 - the check for real clips.

Markers are held-out tracks (SIFT tracks refinement never used; CoTracker3 held-out segments if
there are too few), triangulated with the solved cameras - or turned into directions for a
rotation-only solve. In every frame where a track is actually seen, its 3D point is projected
(cross, coloured by the distance to the tracked position: green < 0.5 px, amber < 1 px, red
beyond) next to the tracked position (small ring). A locked camera keeps every cross inside its
ring through the whole shot; sliding or swimming shows as crosses drifting off.
Occluded points vanish with their track, so nothing is drawn through foreground objects.

Planted ground patches: a RANSAC plane through the points whose normal is closest to the
cameras' up direction; small 2 x 2 grids are planted at up to 10 ground points and drawn while
that point is seen - they must look nailed to the floor, the way a CG object would.
HUD: frame, the frame's held-out error, the solve's average error. ffmpeg libx264, CRF 18.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .camio import CameraTrack
from .geometry import project

GREEN, AMBER, RED = (80, 200, 60), (40, 170, 255), (60, 60, 230)  # BGR
PATCH = (0, 230, 255)  # yellow


def fit_ground(X: np.ndarray, up: np.ndarray, rng, iters: int = 800):
    """RANSAC plane with the most inliers among planes within ~45 deg of `up`.
    Returns (normal pointing up, d with n.x + d = 0, inlier mask) or None."""
    if len(X) < 50:
        return None
    scale = float(np.median(np.linalg.norm(X - np.median(X, 0), axis=1)))
    thr = 0.01 * scale
    best = None
    for _ in range(iters):
        p = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up) < 0.7:
            continue
        inl = np.abs(X @ n - n @ p[0]) < thr
        if best is None or inl.sum() > best[1].sum():
            best = (n, inl)
    if best is None or best[1].sum() < 30:
        return None
    c = X[best[1]].mean(0)  # least-squares refit on the inliers: smallest-eigenvalue direction
    D = X[best[1]] - c      # (a full SVD of D would build an N x N matrix: 26 GB for 57k points)
    n = np.linalg.eigh(D.T @ D)[1][:, 0]
    if n @ up < 0:
        n = -n
    return n, -n @ c, np.abs(X @ n - n @ c) < thr


def _patch(center: np.ndarray, n: np.ndarray, size: float, samples: int = 24) -> list[np.ndarray]:
    """A 2 x 2 grid of `size` on the plane with normal n, centred at `center`, as 3D polylines."""
    a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    s = np.linspace(-size / 2, size / 2, samples)[:, None]
    lines = []
    for v in (-size / 2, 0.0, size / 2):
        lines.append(center + v * a + s * b)
        lines.append(center + v * b + s * a)
    return lines


def _directions(trk: CameraTrack, xy: np.ndarray, vis: np.ndarray, far: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotation-only solve: mean back-projected ray per track, placed `far` from the centre."""
    from .triangulate import undistort_points

    d = np.zeros((vis.shape[1], 3))
    for t in np.nonzero(trk.valid)[0]:
        m = vis[t]
        if not m.any():
            continue
        u = undistort_points(xy[t, m].astype(float), trk.K[t], trk.dist[t])
        r = (np.linalg.inv(trk.K[t]) @ np.column_stack([u, np.ones(m.sum())]).T).T
        d[m] += (r / np.linalg.norm(r, axis=1, keepdims=True)) @ trk.R[t]
    norm = np.linalg.norm(d, axis=1)
    ok = norm > 0
    d[ok] /= norm[ok, None]
    return trk.centers[trk.valid].mean(0) + far * d, ok


def _held_out_tracks(solve_dir: Path, trk: CameraTrack) -> tuple[np.ndarray, np.ndarray, str]:
    from .pipeline.select import MIN_SIFT_OBS, SEGMENT
    from .refine import RefineOptions
    from .tracks import split_sift, split_tracks

    sift = solve_dir / "sift" / "sift_tracks.npz"
    if sift.exists():
        _, held = split_sift(dict(np.load(sift)), RefineOptions().max_sift)
        if int(held["vis"].sum()) >= MIN_SIFT_OBS:
            return held["xy"].astype(float), held["vis"], "sift"
    tr = dict(np.load(solve_dir / "tracks" / "tracks.npz"))
    seg = split_tracks(tr, SEGMENT)
    return seg["xy"][:, seg["holdout"]].astype(float), seg["vis"][:, seg["holdout"]], "cotracker"


def _label(img, text, org, scale, color):
    """Text on a translucent dark box (readable on any footage)."""
    th = max(1, int(round(1.5 * scale)))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)
    x, y = org
    pad = int(round(6 * scale)) + 2
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - h - pad), min(img.shape[1], x + w + pad), min(img.shape[0], y + base + pad)
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.4).astype(np.uint8)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, th, cv2.LINE_AA)


def _polyline(img, uv, z, color, thickness=1):
    seg = []
    for p, ok in zip(uv, z > 1e-6):
        if ok:
            seg.append(p)
            continue
        if len(seg) > 1:
            cv2.polylines(img, [np.int32(np.round(seg))], False, color, thickness, cv2.LINE_AA)
        seg = []
    if len(seg) > 1:
        cv2.polylines(img, [np.int32(np.round(seg))], False, color, thickness, cv2.LINE_AA)


def lock_test(solve_dir: str | Path, out: str | Path | None = None, scale: float = 1.0, n_points: int = 1500,
              seed: int = 0, log=print) -> dict:
    from .pipeline.select import frame_errors, is_rotation_only
    from .triangulate import triangulate_tracks

    solve_dir = Path(solve_dir)
    sel = solve_dir / "selected"
    trk = CameraTrack.load(sel / "cameras.json")
    shot = json.loads((solve_dir / "shot.json").read_text())
    rep = json.loads((solve_dir / "solve.json").read_text()) if (solve_dir / "solve.json").exists() else {}
    rng = np.random.default_rng(seed)

    xy, vis, source = _held_out_tracks(solve_dir, trk)
    vis = vis & trk.valid[:, None]
    rot_only = is_rotation_only(trk)
    X, ok = _directions(trk, xy, vis, 1000.0) if rot_only else triangulate_tracks(trk, xy, vis)
    ok &= vis.sum(0) >= 3
    idx = np.nonzero(ok)[0]
    if len(idx) > n_points:  # longer tracks first, random among them for spread
        w = vis[:, idx].sum(0).astype(float)
        idx = np.sort(rng.choice(idx, n_points, replace=False, p=w / w.sum()))
    X, xy, vis = X[idx], xy[:, idx], vis[:, idx]
    # keyframe-strided SIFT tracks are seen every k-th frame only: draw a track's cross on the
    # frames between two of its observations too (ring = tracked position, on observed frames)
    obs_frames = np.nonzero(vis.any(1))[0]
    step = int(np.median(np.diff(obs_frames))) if len(obs_frames) > 1 else 1
    draw = vis.copy()
    last_col = np.zeros(vis.shape[1], int)
    if step > 1:
        T = vis.shape[0]
        last = np.full(vis.shape, -10**9)
        nxt = np.full(vis.shape, 10**9)
        cur_l = np.full(vis.shape[1], -10**9)
        for t in range(T):
            cur_l = np.where(vis[t], t, cur_l)
            last[t] = cur_l
        cur_n = np.full(vis.shape[1], 10**9)
        for t in range(T - 1, -1, -1):
            cur_n = np.where(vis[t], t, cur_n)
            nxt[t] = cur_n
        draw = (nxt - last) <= 2 * step

    patches = []
    if not rot_only and len(X) >= 50:
        up = -np.mean(trk.R_c2w[trk.valid][:, :, 1], 0)  # OpenCV camera -Y, averaged over the shot
        g = fit_ground(X, up / np.linalg.norm(up), rng)
        # a real floor holds a good share of the points; without one (real clip 02: people on a
        # sofa, telephoto) the best "up" plane was a few points in the air and the patches floated
        if g is not None and g[2].sum() < 0.15 * len(X):
            g = None
        if g is not None:
            n, d, inl = g
            size = 0.12 * float(np.median(np.linalg.norm(X - np.median(X, 0), axis=1)))
            cand = np.nonzero(inl)[0]
            cand = cand[np.argsort(-vis[:, cand].sum(0))][:10]
            patches = [(j, _patch(X[j] - (X[j] @ n + d) * n, n, size)) for j in cand]
    log(f"[locktest] {len(X)} held-out {source} tracks, {len(patches)} ground patches")

    errs = json.loads((sel / "errors.json").read_text()) if (sel / "errors.json").exists() else None
    if errs is None and (solve_dir / "tracks" / "tracks.npz").exists():  # solves from before errors.json
        from .refine import RefineOptions
        from .tracks import split_sift

        held = None
        if (solve_dir / "sift" / "sift_tracks.npz").exists():
            _, held = split_sift(dict(np.load(solve_dir / "sift" / "sift_tracks.npz")), RefineOptions().max_sift)
        errs = frame_errors(trk, dict(np.load(solve_dir / "tracks" / "tracks.npz")), held)
        (sel / "errors.json").write_text(json.dumps(errs), encoding="utf-8")

    W, H = int(round(trk.width * scale)) // 2 * 2, int(round(trk.height * scale)) // 2 * 2
    s = W / trk.width
    out = Path(out) if out else sel / "locktest.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", f"{trk.fps:g}", "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    avg = (rep.get("solve_error") or {}).get("average_px")
    ext = shot["frame_format"]
    r = max(3, int(round(5 * s)))
    fs = 0.8 * max(s, 0.6)
    counts = {"green": 0, "amber": 0, "red": 0}
    # colour thresholds in HD pixels: a 4K frame has twice the pixels for the same angle, so the same
    # solve would look twice as bad in 4K pixels (real clip 04: 54 % green at 4K, 84 % at HD scale)
    k = max(1.0, trk.width / 1920.0)
    for i in range(trk.n_frames):
        img = cv2.imread(str(solve_dir / "frames" / f"{i:06d}.{ext}"))
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if trk.valid[i]:
            K, R, t, dist = trk.K[i], trk.R[i], trk.t[i], trk.dist[i]
            for j, lines in patches:
                if draw[i, j]:
                    for line in lines:
                        uv, z = project(K, R, t, line, dist)
                        _polyline(img, uv * s, z, PATCH, max(2, int(round(3 * s))))
            m = np.nonzero(draw[i])[0]
            uv, z = project(K, R, t, X[m], dist)
            for (u, v), zz, j in zip(uv * s, z, m):
                if zz <= 0:
                    continue
                if vis[i, j]:
                    ee = float(np.hypot(u / s - xy[i, j, 0], v / s - xy[i, j, 1]))
                    if ee > 3.0 * k:
                        continue  # a mismatched track, not the camera: auto-select counts these apart too
                    last_col[j] = 0 if ee < 0.5 * k else (1 if ee < 1.0 * k else 2)
                    counts[("green", "amber", "red")[last_col[j]]] += 1
                    ou, ov = xy[i, j] * s
                    cv2.circle(img, (int(round(ou)), int(round(ov))), r + 2, (255, 255, 255), 1, cv2.LINE_AA)
                col = (GREEN, AMBER, RED)[last_col[j]]
                u, v = int(round(u)), int(round(v))
                cv2.line(img, (u - r, v), (u + r, v), col, 1, cv2.LINE_AA)
                cv2.line(img, (u, v - r), (u, v + r), col, 1, cv2.LINE_AA)
        txt = f"frame {trk.frame_start + i}"
        fe = errs["mean_px"][i] if errs else None
        if not trk.valid[i]:
            txt += "   not solved"
        elif fe is not None:
            txt += f"   {fe:.2f} px"
        _label(img, txt, (int(20 * s) + 8, int(40 * s) + 16), fs, (255, 255, 255))
        foot = f"Niko Tracker   {rep.get('selected') or trk.method}"
        if avg is not None:
            foot += f"   average error {avg:.2f} px"
        _label(img, foot, (int(20 * s) + 8, H - int(24 * s) - 10), fs * 0.75, (230, 230, 230))
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    total = max(1, sum(counts.values()))
    shares = {k: round(100 * v / total, 1) for k, v in counts.items()}
    log(f"[locktest] markers green {shares['green']} %, amber {shares['amber']} %, red {shares['red']} % -> {out}")
    return {"mp4": str(out), "tracks": int(len(X)), "patches": len(patches), "marker_share_pct": shares,
            "source": source}
