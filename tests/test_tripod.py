"""Rotation-only solver on a synthetic pan: recovers rotations and focal from a poor start."""

import numpy as np
import pytest

from niko.camio import CameraTrack
from niko.metrics import rotation_errors_deg
from niko.tripod import TripodOptions, solve_tripod

from conftest import axis_angle


def make_pan(n=40, f=1400.0, zoom=False, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    W, H = 1920, 1080
    gt = CameraTrack.empty("gt", W, H, 25.0, 1, n)
    base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # camera looking along world +y
    fs = np.linspace(f, 2.2 * f, n) if zoom else np.full(n, f)
    for i in range(n):
        R_c2w = axis_angle([0, 0, 1], 1.2 * i) @ axis_angle([1, 0, 0], 3 * np.sin(i / 7)) @ base.T
        gt.R[i] = R_c2w.T
        gt.t[i] = -gt.R[i] @ np.array([1.0, 2.0, 1.5])
        gt.K[i] = [[fs[i], 0, W / 2], [0, fs[i], H / 2], [0, 0, 1]]
        gt.dist[i] = 0
    gt.valid[:] = True
    # directions spread over the swept field of view
    az = rng.uniform(-40, 100, 3000)
    el = rng.uniform(-25, 25, 3000)
    dw = np.stack([np.sin(np.radians(az)) * np.cos(np.radians(el)), np.cos(np.radians(az)) * np.cos(np.radians(el)),
                   np.sin(np.radians(el))], 1)
    xy = np.full((n, len(dw), 2), np.nan)
    vis = np.zeros((n, len(dw)), bool)
    for i in range(n):
        v = dw @ gt.R[i].T
        p = v[:, :2] / v[:, 2:3] * fs[i] + [W / 2, H / 2]
        m = (v[:, 2] > 0) & (p[:, 0] > 0) & (p[:, 0] < W) & (p[:, 1] > 0) & (p[:, 1] < H)
        xy[i, m] = p[m] + rng.normal(scale=noise, size=(m.sum(), 2))
        vis[i, m] = True
    return gt, {"xy": xy, "vis": vis, "holdout": np.zeros(len(dw), bool)}


@pytest.mark.parametrize("zoom", [False, True])
def test_tripod_recovers_rotation_and_focal(zoom):
    gt, tracks = make_pan(zoom=zoom)
    start = CameraTrack(**{**gt.__dict__})
    start.R, start.t, start.K = gt.R.copy(), gt.t.copy(), gt.K.copy()
    rng = np.random.default_rng(1)
    for i in range(gt.n_frames):  # 1 deg rotation noise, 15 % focal error
        start.R[i] = axis_angle(rng.normal(size=3), 1.0) @ gt.R[i]
    start.K[:, 0, 0] *= 1.15
    start.K[:, 1, 1] *= 1.15
    est, rep = solve_tripod(start, tracks, TripodOptions(intrinsics="per_frame_focal" if zoom else "shared_focal"))
    rot = rotation_errors_deg(est.R_c2w, gt.R_c2w)
    focal = np.abs(est.K[:, 0, 0] / gt.K[:, 0, 0] - 1) * 100
    print(f"\nzoom={zoom}: rot max {rot.max():.4f} deg, focal max {focal.max():.3f} %, rounds {rep['rounds']}")
    assert rot.max() < 0.02
    assert focal.max() < 0.3
    assert np.ptp(est.centers, axis=0).max() < 1e-9  # one shared centre
