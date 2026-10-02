"""Similarity alignment (Umeyama 1991) and rotation-only alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import nearest_rotation


@dataclass(frozen=True)
class Sim3:
    """x' = s * R @ x + t"""

    s: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        return self.s * np.asarray(X) @ self.R.T + self.t

    def inverse(self) -> "Sim3":
        Rinv = self.R.T
        return Sim3(1.0 / self.s, Rinv, -(Rinv @ self.t) / self.s)


class DegenerateAlignment(ValueError):
    pass


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares Sim(3) (or SE(3)) with dst ~= s R src + t.

    src, dst: [N, 3], N >= 3, not all coincident. For collinear points the
    rotation about the line is not determined by positions; callers that need
    orientations should use `align_rotations` for the rotational part.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"expected matching [N,3] arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise DegenerateAlignment("need at least 3 points")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    var_s = float(np.mean(np.sum(xs * xs, axis=1)))
    if var_s < 1e-18:
        raise DegenerateAlignment("source points are coincident")

    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)


def align_rotations(R_src_c2w: np.ndarray, R_dst_c2w: np.ndarray) -> np.ndarray:
    """Global rotation A minimising sum ||R_dst_i - A R_src_i||_F (camera-to-world stacks)."""
    M = np.einsum("nij,nkj->ik", R_dst_c2w, R_src_c2w)  # sum R_dst R_src^T
    return nearest_rotation(M)


def is_collinear(X: np.ndarray, rel_tol: float = 1e-3) -> bool:
    """True if the spread of X is (nearly) one-dimensional or zero."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3:
        return True
    sv = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    if sv[0] < 1e-12:
        return True
    return bool(sv[1] < rel_tol * sv[0])
