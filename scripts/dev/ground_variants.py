"""Ground-plane fit variants against synthetic truth (tilt of the found up, degrees) and the camera
height they give on real solves (share of the median depth).

usage (niko env): python scripts/dev/ground_variants.py <solve_dir>[=<gt shot dir>] ...
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.pipeline.export import median_depth
from niko.plyio import read_ply_xyz


def up_from_right_axes(trk):
    """Up from the cameras' right (x) axes, which stay level unless the camera rolls: the normal of
    the plane they lie in. (up, reliable)"""
    v = np.nonzero(trk.valid)[0]
    Rc = trk.R_c2w[v]
    x = Rc[:, :, 0]
    w, V = np.linalg.eigh(x.T @ x / len(x))  # ascending
    up0 = -Rc[:, :, 1].mean(0)
    up0 /= np.linalg.norm(up0)
    n = V[:, 0] * np.sign(V[:, 0] @ up0)
    return n, up0, w


def fit(X, trk, depth, prior, cos_win, rng, min_h=None, iters=800, max_roll=None):
    scale = float(np.median(np.linalg.norm(X - np.median(X, 0), axis=1)))
    thr = 0.01 * scale
    C = trk.centers[trk.valid]
    xm = trk.R_c2w[trk.valid][:, :, 0].mean(0)
    xm /= np.linalg.norm(xm)
    sin_roll = None if max_roll is None else np.sin(np.radians(max_roll))
    best = None
    for _ in range(iters):
        p = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if n @ prior < 0:
            n = -n
        if n @ prior < cos_win:
            continue
        if min_h is not None and np.median((C - p[0]) @ n) < min_h:
            continue
        if sin_roll is not None and abs(n @ xm) > sin_roll:
            continue
        inl = np.abs(X @ n - n @ p[0]) < thr
        if best is None or inl.sum() > best[1].sum():
            best = (n, inl)
    if best is None or best[1].sum() < 30:
        return None
    c = X[best[1]].mean(0)
    D = X[best[1]] - c
    n = np.linalg.eigh(D.T @ D)[1][:, 0]
    if n @ prior < 0:
        n = -n
    inl = np.abs(X @ n - n @ c) < thr
    return n, np.median((C - c) @ n), int(inl.sum()), float(np.degrees(np.arcsin(min(1.0, abs(n @ xm)))))


def main():
    for arg in sys.argv[1:]:
        d, _, g = arg.partition("=")
        d = Path(d)
        trk = CameraTrack.load(d / "selected" / "cameras.json")
        if not (d / "selected" / "points.ply").exists():
            continue
        X = read_ply_xyz(d / "selected" / "points.ply")
        depth = median_depth(trk, X)
        n_x, up0, w = up_from_right_axes(trk)
        ratio = w[1] / max(w[0], 1e-12)
        x_ok = w[1] > 3e-4 and ratio > 20
        variants = {
            "V0 now": (up0, 0.7, None),
            "V1 above": (up0, 0.7, 0.02 * depth),
            "V2 xprior": ((n_x if x_ok else up0), (np.cos(np.radians(25)) if x_ok else 0.7), 0.02 * depth),
            "V3 noroll": ((n_x if x_ok else up0), (np.cos(np.radians(10)) if x_ok else 0.7), 0.02 * depth, 8.0),
        }
        Ra = None
        if g:
            gt = CameraTrack.load(Path(g) / "gt" / "cameras.json")
            ok = trk.valid & gt.valid
            U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", gt.R_c2w[ok], trk.R_c2w[ok]))
            Ra = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
            true_h = np.median(gt.centers[gt.valid][:, 2]) / json.loads((Path(g) / "scene.json").read_text())["median_depth"]
        tilt = (lambda u: np.degrees(np.arccos(np.clip((Ra @ u)[2], -1, 1)))) if Ra is not None else None
        line = f"{d.name[:26]:26s} x-axes {'ok ' if x_ok else 'no '}(l2 {w[1]:.1e}, ratio {ratio:7.1f})"
        if tilt:
            line += f" xprior tilt {tilt(n_x):5.1f} | true h {100 * true_h:5.1f}%"
        print(line)
        for name, (prior, cw, mh, *roll) in variants.items():
            r = fit(X, trk, depth, prior, cw, np.random.default_rng(0), mh, max_roll=roll[0] if roll else None)
            if r is None:
                u, h = (n_x if x_ok else up0), None
                s = f"    {name:10s} no plane      "
            else:
                u, h, k, rl = r
                if roll and x_ok:
                    u = n_x  # V3: up from the level right axes, the plane only places the ground
                s = (f"    {name:10s} h {100 * h / depth:6.1f}% {k:6d}/{len(X)} inl, roll {rl:4.1f},"
                     f" {np.degrees(np.arccos(np.clip(u @ n_x, -1, 1))):5.1f} deg from x-axes up")
            if tilt:
                s += f" tilt {tilt(u):5.1f}"
            print(s, flush=True)


if __name__ == "__main__":
    main()
