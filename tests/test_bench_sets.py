import numpy as np

from niko.bench.generate import add_noise, dist_vector, nominal_K, overscan
from niko.bench.sets import load_set, shot_specs
from niko.geometry import project
from niko.triangulate import undistort_points


def test_synthetic_v1_specs():
    bench = load_set("synthetic_v1")
    specs = shot_specs(bench)
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names)) >= 20
    for s in specs:
        assert (s["width"], s["height"], s["n_frames"]) == (1920, 1080, 150)
        assert s["camera"]["path"] in {"orbit", "walk", "line", "pan", "whip", "follow"}
    town = next(s for s in specs if s["name"] == "drone_flyover")
    assert town["scene"]["kind"] == "town" and town["scene"]["extent"] == 90.0  # one-level merge
    assert next(s for s in specs if s["name"] == "orbit_yard")["scene"]["clear_width"] == 3.5


def test_preview_keeps_duration_and_whip_timing():
    full = {s["name"]: s for s in shot_specs(load_set("synthetic_v1"))}
    prev = {s["name"]: s for s in shot_specs(load_set("synthetic_v1"), preview=True)}
    for name in full:
        a, b = full[name], prev[name]
        assert b["n_frames"] == 16 and b["width"] == a["width"] // 4
        assert abs(a["n_frames"] / a["fps"] - b["n_frames"] / b["fps"]) < 1e-9
    w = prev["whip_pan_blur"]["camera"]
    assert 0 < w["whip_start"] < 16 and w["whip_frames"] >= 1


def test_overscan_covers_every_distorted_pixel():
    spec = {"width": 1920, "height": 1080, "camera": {"lens": 16, "sensor_width": 36},
            "distortion": {"k1": -0.10, "k2": 0.02}}
    mx, my = overscan(spec)
    K, d = nominal_K(spec), dist_vector(spec)
    ys, xs = np.mgrid[0:1080:7, 0:1920:7]
    und = undistort_points(np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1), K, d, iters=50)
    assert und[:, 0].min() > -mx and und[:, 0].max() < 1920 + mx
    assert und[:, 1].min() > -my and und[:, 1].max() < 1080 + my
    # and the inverse really inverts the forward model used by the metrics
    X = np.column_stack([(und - K[:2, 2]) / K[0, 0], np.ones(len(und))])
    uv, _ = project(K, np.eye(3), np.zeros(3), X, d)
    assert np.abs(uv - np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)).max() < 1e-6
    assert overscan({"width": 100, "height": 100, "camera": {}}) == (0, 0)


def test_noise_statistics():
    img = np.full((200, 200, 3), 100, np.uint8)
    out = add_noise(img, {"read": 6.0, "shot": 0.08}, np.random.default_rng(0))
    assert abs(out.mean() - 100) < 0.2
    assert abs(out.astype(float).std() - np.sqrt(36 + 0.08 * 100)) < 0.2
