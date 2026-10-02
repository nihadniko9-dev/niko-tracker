"""Rotation-only (tripod / nodal pan) solver.

Model: one camera centre, per-frame rotation R_t, points are directions d_n on the unit sphere:
    x_tn = K_t * distort( R_t d_n )        (corner-origin pixels)
Focal: shared or per frame (zoom on a tripod); distortion none | k1 | k1k2; principal point fixed.
Gauge: frame 0's rotation is fixed (a global rotation is unobservable).

Solver: Levenberg-Marquardt with a Schur complement on the 2x2 point blocks, Jacobians by
central differences per parameter block (vectorised over observations), Cauchy weights (IRLS),
two outlier rounds. The reduced camera system is at most (3F + F + 2)^2, i.e. tiny and dense.
Used when parallax is too small for SfM (tripod_rotation_only: COLMAP focal error 31 %).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from .camio import CameraTrack
from .refine import copy_track
from .triangulate import undistort_points


@dataclass
class TripodOptions:
    intrinsics: str = "shared_focal"   # shared_focal | per_frame_focal
    distortion: str = "none"           # none | k1 | k1k2
    loss_scale_px: float = 1.0
    outlier_px: float = 3.0
    rounds: int = 2
    max_iterations: int = 60
    max_points: int = 2500  # longest tracks; 6000 took 100-200 s per solve on 150 x 1080p frames


def _acc(index: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(index, weights=weights, minlength=size)


def _skew(v):
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def _exp(w):
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = _skew(w / np.maximum(th[..., 0], 1e-12))
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return np.where(th < 1e-12, np.eye(3) + _skew(w), R)


def _tangent(d):
    """Orthonormal tangent basis [N,3,2] at unit vectors d [N,3]."""
    a = np.where(np.abs(d[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    b1 = np.cross(d, a)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 = np.cross(d, b1)
    return np.stack([b1, b2], 2)


class _Problem:
    def __init__(self, obs_f, obs_p, uv, F, P, cx, cy, per_frame, nk):
        self.of, self.op, self.uv = obs_f, obs_p, uv
        self.F, self.P, self.cx, self.cy = F, P, cx, cy
        self.per_frame, self.nk = per_frame, nk

    def residual(self, R, d, f, k):
        v = np.einsum("oij,oj->oi", R[self.of], d[self.op])
        x, y = v[:, 0] / v[:, 2], v[:, 1] / v[:, 2]
        r2 = x * x + y * y
        s = 1 + r2 * (k[0] + r2 * k[1])
        ff = f[self.of] if self.per_frame else f[0]
        res = np.stack([ff * x * s + self.cx - self.uv[:, 0], ff * y * s + self.cy - self.uv[:, 1]], 1)
        res[v[:, 2] <= 0] = 1e3
        return res


def solve_tripod(cand: CameraTrack, tracks: dict, opts: TripodOptions | None = None) -> tuple[CameraTrack, dict]:
    opts = opts or TripodOptions()
    t0 = time.time()
    trk = copy_track(cand)
    valid = trk.valid
    frames = np.nonzero(valid)[0]
    F = len(frames)
    fidx = -np.ones(trk.n_frames, int)
    fidx[frames] = np.arange(F)
    xy, vis = tracks["xy"].astype(np.float64), tracks["vis"] & valid[:, None]
    if "holdout" in tracks:
        vis = vis & ~tracks["holdout"][None, :]
    cnt = vis.sum(0)
    pts = np.nonzero(cnt >= 2)[0]
    if len(pts) > opts.max_points:
        pts = pts[np.argsort(cnt[pts])[::-1][: opts.max_points]]
    xy, vis = xy[:, pts], vis[:, pts]
    P = len(pts)
    cx, cy = float(np.median(trk.K[frames, 0, 2])), float(np.median(trk.K[frames, 1, 2]))
    per_frame = opts.intrinsics == "per_frame_focal"
    nk = {"none": 0, "k1": 1, "k1k2": 2}[opts.distortion]
    R = trk.R[frames].copy()  # world -> camera; the shared centre is irrelevant for rotations
    f = trk.K[frames, 0, 0].copy() if per_frame else np.array([float(np.median(trk.K[frames, 0, 0]))])
    k = np.array([float(np.median(trk.dist[frames, 0])), float(np.median(trk.dist[frames, 1]))])
    if nk < 2:
        k[1] = 0.0
    if nk < 1:
        k[0] = 0.0
    # initial directions: mean of the back-projected rays
    d = np.zeros((P, 3))
    for i, t in enumerate(frames):
        m = vis[t]
        if not m.any():
            continue
        K = np.array([[f[i] if per_frame else f[0], 0, cx], [0, f[i] if per_frame else f[0], cy], [0, 0, 1]])
        u = undistort_points(xy[t, m], K, np.array([k[0], k[1], 0, 0, 0]))
        rays = (np.linalg.inv(K) @ np.column_stack([u, np.ones(m.sum())]).T).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        d[m] += rays @ R[i]  # camera -> world direction
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    ot, on = np.nonzero(vis)  # [obs] frame (clip index), point
    use = np.ones(len(ot), bool)
    report = {"method": cand.method, "options": asdict(opts), "points": int(P), "rounds": []}

    nc = 3 * (F - 1) + (F if per_frame else 1) + nk  # camera-side parameters
    for rnd in range(opts.rounds + 1):
        of, op, uv = fidx[ot[use]], on[use], xy[ot[use], on[use]]
        prob = _Problem(of, op, uv, F, P, cx, cy, per_frame, nk)
        lam = 1e-3
        r = prob.residual(R, d, f, k)
        e = np.linalg.norm(r, axis=1)
        cost = np.sum(np.log1p((e / opts.loss_scale_px) ** 2))
        it = 0
        for it in range(opts.max_iterations):
            w = 1.0 / (1.0 + (e / opts.loss_scale_px) ** 2)  # Cauchy IRLS weights
            h = 1e-6
            # Jacobian blocks by central differences, vectorised over observations
            Jr = np.zeros((len(of), 2, 3))
            for a in range(3):
                dw = np.zeros(3)
                dw[a] = h
                Rp, Rm = _exp(dw)[None] @ R, _exp(-dw)[None] @ R
                Jr[:, :, a] = (prob.residual(Rp, d, f, k) - prob.residual(Rm, d, f, k)) / (2 * h)
            B = _tangent(d)
            Jp = np.zeros((len(of), 2, 2))
            for a in range(2):
                dp = d + h * B[:, :, a]
                dm = d - h * B[:, :, a]
                Jp[:, :, a] = (prob.residual(R, dp / np.linalg.norm(dp, axis=1, keepdims=True), f, k)
                               - prob.residual(R, dm / np.linalg.norm(dm, axis=1, keepdims=True), f, k)) / (2 * h)
            nf = F if per_frame else 1
            Jf = np.zeros((len(of), 2))
            hf = 1e-3
            if per_frame:
                ff = np.ones(F)
                Jf = (prob.residual(R, d, f + hf * ff, k) - prob.residual(R, d, f - hf * ff, k)) / (2 * hf)
            else:
                Jf = (prob.residual(R, d, f + hf, k) - prob.residual(R, d, f - hf, k)) / (2 * hf)
            Jk = np.zeros((len(of), 2, nk))
            for a in range(nk):
                dk = np.zeros(2)
                dk[a] = 1e-6
                Jk[:, :, a] = (prob.residual(R, d, f, k + dk) - prob.residual(R, d, f, k - dk)) / 2e-6
            # local camera-side block per observation: [rot x3 | focal | k x nk], with its columns
            m_loc = 4 + nk
            Jl = np.zeros((len(of), 2, m_loc))
            Jl[:, :, :3] = Jr
            Jl[:, :, 3] = Jf
            Jl[:, :, 4:] = Jk
            cols = np.zeros((len(of), m_loc), int)
            cols[:, :3] = 3 * (of[:, None] - 1) + np.arange(3)
            cols[:, 3] = 3 * (F - 1) + (of if per_frame else 0)
            cols[:, 4:] = 3 * (F - 1) + nf + np.arange(nk)
            live = np.ones((len(of), m_loc), bool)
            live[:, :3] = (of > 0)[:, None]  # frame 0's rotation is the gauge
            Jl = Jl * live[:, None, :]
            cols = np.where(live, cols, 0)
            ww = w[:, None, None]
            # scatter-adds via bincount (np.add.at is ~100x slower at this size)
            JtJ = np.einsum("oai,oaj->oij", Jl * ww, Jl)  # [obs, m, m]
            U = _acc((cols[:, :, None] * nc + cols[:, None, :]).ravel(), JtJ.ravel(), nc * nc).reshape(nc, nc)
            gc = _acc(cols.ravel(), np.einsum("oai,oa->oi", Jl * ww, r).ravel(), nc)
            V = _acc((op[:, None] * 4 + np.arange(4)).ravel(),
                     np.einsum("oai,oaj->oij", Jp * ww, Jp).reshape(-1), P * 4).reshape(P, 2, 2)
            gp = _acc((op[:, None] * 2 + np.arange(2)).ravel(),
                      np.einsum("oai,oa->oi", Jp * ww, r).ravel(), P * 2).reshape(P, 2)
            JW = np.einsum("oai,oaj->oij", Jl * ww, Jp)  # [obs, m, 2]
            idx = (op[:, None, None] * nc + cols[:, :, None]) * 2 + np.arange(2)[None, None, :]
            Wsum = _acc(idx.ravel(), JW.ravel(), P * nc * 2).reshape(P, nc, 2)
            while True:
                Vd = V + lam * (V * np.eye(2)[None] + 1e-9 * np.eye(2)[None])
                Vinv = np.linalg.inv(Vd)
                Ud = U + lam * np.diag(np.diag(U) + 1e-9)
                # Schur: S = U - sum_n W_n V_n^-1 W_n^T ; rhs = gc - sum_n W_n V_n^-1 gp_n
                WV = np.einsum("pij,pjk->pik", Wsum, Vinv)
                S = Ud - WV.transpose(1, 0, 2).reshape(nc, -1) @ Wsum.transpose(1, 0, 2).reshape(nc, -1).T
                rhs = gc - np.einsum("pij,pj->i", WV, gp)
                dc = np.linalg.solve(S, -rhs)
                dpnt = -np.einsum("pij,pj->pi", Vinv, gp + np.einsum("pji,j->pi", Wsum, dc))
                # apply
                Rn = R.copy()
                Rn[1:] = _exp(dc[: 3 * (F - 1)].reshape(F - 1, 3)) @ R[1:]
                fn = f + dc[3 * (F - 1): 3 * (F - 1) + nf]
                kn = k.copy()
                kn[:nk] += dc[3 * (F - 1) + nf:]
                dn = d + np.einsum("pij,pj->pi", B, dpnt)
                dn /= np.linalg.norm(dn, axis=1, keepdims=True)
                rn = prob.residual(Rn, dn, fn, kn)
                en = np.linalg.norm(rn, axis=1)
                cn = np.sum(np.log1p((en / opts.loss_scale_px) ** 2))
                if cn < cost:
                    R, f, k, d, r, e = Rn, fn, kn, dn, rn, en
                    rel = (cost - cn) / max(cost, 1e-12)
                    cost = cn
                    lam = max(lam / 3, 1e-9)
                    break
                lam *= 5
                if lam > 1e8:
                    break
            if lam > 1e8 or rel < 1e-9:
                break
        # outlier rounds on all observations
        prob_all = _Problem(fidx[ot], on, xy[ot, on], F, P, cx, cy, per_frame, nk)
        e_all = np.linalg.norm(prob_all.residual(R, d, f, k), axis=1)
        eu = e_all[use]
        med = float(np.median(eu))
        mad = 1.4826 * float(np.median(np.abs(eu - med)))
        thr = max(opts.outlier_px, med + 4 * mad)
        report["rounds"].append({"observations": int(use.sum()), "iterations": it + 1, "median_px": med,
                                 "rms_px": float(np.sqrt(np.mean(eu ** 2))), "threshold_px": thr})
        if rnd < opts.rounds:
            use = e_all <= thr
    # write back: shared centre at the candidate's mean centre, t = -R C
    C = trk.centers[frames].mean(axis=0)
    trk.R[frames] = R
    trk.t[frames] = -np.einsum("tij,j->ti", R, C)
    trk.K[frames] = 0
    trk.K[frames, 0, 0] = trk.K[frames, 1, 1] = f if per_frame else f[0]
    trk.K[frames, 0, 2], trk.K[frames, 1, 2], trk.K[frames, 2, 2] = cx, cy, 1
    trk.dist[frames] = 0
    trk.dist[frames, 0], trk.dist[frames, 1] = k
    trk.method = f"tripod:{cand.method}"
    trk.intrinsics_mode = "per_frame_focal" if per_frame else "shared"
    report.update({"focal_median": float(np.median(f)), "k1": float(k[0]), "k2": float(k[1]),
                   "seconds": round(time.time() - t0, 2)})
    trk.extra["tripod"] = report
    trk.extra["rotation_only"] = True
    return trk, report
