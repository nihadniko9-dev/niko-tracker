"""Real-world size from single-image metric depth (niko.scale) and the no-masks prompt choice."""

import json

import numpy as np

from niko.camio import CameraTrack
from niko.cli import _prompts
from niko.scale import estimate_metric_scale


def _scene(n=12, W=640, H=360, f=500.0, seed=0):
    """A camera sliding sideways past points 4-8 units away; tracks in every frame."""
    rng = np.random.default_rng(seed)
    trk = CameraTrack.empty("test", W, H, 25.0, 1, n)
    for i in range(n):
        trk.K[i] = [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]
        trk.R[i] = np.eye(3)
        trk.t[i] = [-0.2 * i, 0, 0]
        trk.dist[i] = 0
    trk.valid[:] = True
    X = np.column_stack([rng.uniform(-3, 5, 400), rng.uniform(-1.5, 1.5, 400), rng.uniform(4, 8, 400)])
    xy = np.zeros((n, len(X), 2))
    for i in range(n):
        c = X @ trk.R[i].T + trk.t[i]
        xy[i] = np.column_stack([f * c[:, 0] / c[:, 2] + W / 2, f * c[:, 1] / c[:, 2] + H / 2])
    vis = (xy[..., 0] >= 0) & (xy[..., 0] < W) & (xy[..., 1] >= 0) & (xy[..., 1] < H)
    return trk, X, xy, vis


def _unidepth(tmp_path, trk, X, metres_per_unit, h=90, w=160):
    """UniDepth files as MegaSaM leaves them: per-frame depth in metres at a reduced size."""
    d = tmp_path / "candidates" / "megasam"
    (d / "work" / "metric" / "shot").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"stats": {"stride": 1}}))
    for i in range(trk.n_frames):
        depth = np.full((h, w), np.nan, np.float32)
        c = X @ trk.R[i].T + trk.t[i]
        u = (trk.K[i, 0, 0] * c[:, 0] / c[:, 2] + trk.K[i, 0, 2]) * w / trk.width
        v = (trk.K[i, 1, 1] * c[:, 1] / c[:, 2] + trk.K[i, 1, 2]) * h / trk.height
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        depth[v[ok].astype(int), u[ok].astype(int)] = metres_per_unit * c[ok, 2]
        np.savez(d / "work" / "metric" / "shot" / f"{i:06d}.npz", depth=depth, fov=60.0)


def test_metric_scale_recovers_known_size(tmp_path):
    trk, X, xy, vis = _scene()
    _unidepth(tmp_path, trk, X, 3.7)
    est = estimate_metric_scale(trk, xy, vis, tmp_path)
    assert est is not None
    assert abs(est["metres_per_unit"] / 3.7 - 1) < 0.02
    # one model alone is never trusted to set the size
    assert est["agree_pct"] is None and not est["reliable"]


def test_metric_scale_none_without_depth_models_or_for_a_tripod(tmp_path):
    trk, X, xy, vis = _scene()
    assert estimate_metric_scale(trk, xy, vis, tmp_path) is None
    _unidepth(tmp_path, trk, X, 2.0)
    trk.extra["rotation_only"] = True
    assert estimate_metric_scale(trk, xy, vis, tmp_path) is None


def test_prompts_none_means_no_masks():
    assert _prompts(None) is None                  # the default list
    assert _prompts("none") == [] and _prompts("") == [] and _prompts(" , ") == []
    assert _prompts("person, car") == ["person", "car"]
