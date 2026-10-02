"""End-to-end check of the ground-truth path against pixels Cycles actually renders.

Blender camera (native settings) -> nikobpy.export_cameras -> niko.blendercam ->
OpenCV K, R, t -> projection, compared with the centroids of tiny emissive spheres
in the rendered image. A half-pixel convention error would show up as 0.5 px.
"""

import json
import subprocess

import cv2
import numpy as np
import pytest

from niko.blendercam import load_blender_cameras
from niko.geometry import blender_to_K, project
from niko.paths import REPO, find_blender

from test_blender_convention import _euler_xyz, _host_path

pytestmark = pytest.mark.blender

SCRIPT = REPO / "blender" / "tests" / "render_markers.py"
BLOB_RADIUS_PX = 1.6
WINDOW = 7  # centroid window half-size in pixels
TOL_PX = 0.1

CASES = [
    {"name": "hd_centered", "resolution": [1920, 1080],
     "camera": {"lens": 35.0, "sensor_width": 36.0, "sensor_height": 24.0, "sensor_fit": "AUTO",
                "shift_x": 0.0, "shift_y": 0.0}},
    {"name": "hd720_shifted", "resolution": [1280, 720],
     "camera": {"lens": 50.0, "sensor_width": 36.0, "sensor_height": 24.0, "sensor_fit": "HORIZONTAL",
                "shift_x": 0.1, "shift_y": -0.07}},
    {"name": "portrait_auto", "resolution": [1080, 1920],
     "camera": {"lens": 28.0, "sensor_width": 36.0, "sensor_height": 24.0, "sensor_fit": "AUTO",
                "shift_x": 0.0, "shift_y": 0.05}},
    {"name": "vertical_fit", "resolution": [1600, 900],
     "camera": {"lens": 70.0, "sensor_width": 36.0, "sensor_height": 20.25, "sensor_fit": "VERTICAL",
                "shift_x": -0.05, "shift_y": 0.02}},
]


def _place_markers(rng, case):
    W, H = case["resolution"]
    c = case["camera"]
    K = blender_to_K(c["lens"], c["sensor_width"], c["sensor_height"], c["sensor_fit"],
                     c["shift_x"], c["shift_y"], W, H)
    euler = rng.uniform(-np.pi, np.pi, 3)
    loc = rng.normal(scale=3.0, size=3)
    R = (_euler_xyz(*euler) @ np.diag([1.0, -1.0, -1.0])).T
    t = -R @ loc
    us = np.linspace(40, W - 40, 9)
    vs = np.linspace(40, H - 40, 6)
    uv = np.array([(u, v) for u in us for v in vs]) + rng.uniform(-5, 5, (len(us) * len(vs), 2))
    depth = rng.uniform(4.0, 30.0, len(uv))
    Xc = (np.linalg.inv(K) @ np.column_stack([uv, np.ones(len(uv))]).T).T * depth[:, None]
    Xw = (Xc - t) @ R
    radii = BLOB_RADIUS_PX * depth / K[0, 0]
    return {**case, "location": loc.tolist(), "rotation_euler": euler.tolist(),
            "markers": Xw.tolist(), "radii": radii.tolist()}


def _centroids(img, uv_pred):
    out = []
    H, W = img.shape
    for u, v in uv_pred:
        c0, r0 = int(np.floor(u)), int(np.floor(v))
        rs, cs = slice(max(r0 - WINDOW, 0), min(r0 + WINDOW + 1, H)), slice(max(c0 - WINDOW, 0), min(c0 + WINDOW + 1, W))
        patch = img[rs, cs]
        rows, cols = np.mgrid[rs, cs]
        w = patch.sum()
        assert w > 0.5, f"no marker near ({u:.1f}, {v:.1f})"
        out.append([(patch * (cols + 0.5)).sum() / w, (patch * (rows + 0.5)).sum() / w])
    return np.array(out)


def _render(spec, tmp_path):
    blender = find_blender()
    if blender is None:
        pytest.skip("Blender not found (set NIKO_BLENDER)")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", _host_path(SCRIPT, blender), "--",
           _host_path(spec_path, blender), _host_path(tmp_path, blender)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def _pose(euler, loc):
    R = (_euler_xyz(*euler) @ np.diag([1.0, -1.0, -1.0])).T
    return R, -R @ np.asarray(loc)


def test_distorted_markers(tmp_path):
    """Overscan render + exact inverse remap (niko.bench.generate) == OpenCV model with distortion."""
    from niko.bench.generate import overscan, undistort_maps
    from niko.triangulate import undistort_points

    W, H, lens = 1280, 720, 24.0
    spec = {"width": W, "height": H, "camera": {"lens": lens, "sensor_width": 36.0},
            "distortion": {"k1": -0.12, "k2": 0.03}}
    mx, my = overscan(spec)
    K = np.array([[lens / 36 * W, 0, W / 2], [0, lens / 36 * W, H / 2], [0, 0, 1.0]])
    dist = np.array([-0.12, 0.03, 0, 0, 0])
    rng = np.random.default_rng(3)
    euler, loc = rng.uniform(-np.pi, np.pi, 3), rng.normal(size=3)
    R, t = _pose(euler, loc)
    grid = np.array([(u, v) for u in np.linspace(30, W - 30, 10) for v in np.linspace(30, H - 30, 6)])
    und = undistort_points(grid, K, dist, iters=50)
    depth = rng.uniform(4.0, 30.0, len(grid))
    Xc = (np.linalg.inv(K) @ np.column_stack([und, np.ones(len(und))]).T).T * depth[:, None]
    Xw = (Xc - t) @ R
    case = {"name": "distorted", "resolution": [W + 2 * mx, H + 2 * my],
            "camera": {"lens": lens * W / (W + 2 * mx), "sensor_width": 36.0, "sensor_height": 24.0,
                       "sensor_fit": "HORIZONTAL", "shift_x": 0.0, "shift_y": 0.0},
            "location": list(loc), "rotation_euler": list(euler), "markers": Xw.tolist(),
            "radii": (BLOB_RADIUS_PX * depth / K[0, 0]).tolist()}
    _render({"samples": 256, "cases": [case]}, tmp_path)

    trk = load_blender_cameras(tmp_path / "distorted_cameras.json")
    assert np.allclose(trk.K[0, 0, 0], K[0, 0]) and np.allclose(trk.K[0, :2, 2], [W / 2 + mx, H / 2 + my])
    img = cv2.imread(str(tmp_path / "distorted.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    mapx, mapy = undistort_maps(K, dist, W, H, mx, my)
    out = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    uv, _ = project(K, R, t, Xw, dist)
    err = np.linalg.norm(_centroids(np.clip(out, 0, None), uv) - uv, axis=1)
    print(f"\ndistortion k1=-0.12 k2=0.03, overscan {mx}x{my}px: mean {err.mean():.3f} px, max {err.max():.3f} px")
    assert err.max() < TOL_PX


def test_rolling_shutter_markers(tmp_path):
    """Cycles rolling shutter (type TOP): row y is exposed at frame + (y/H - 0.5) * readout."""
    W, H, lens = 1280, 720, 35.0
    readout, w = 0.6, np.radians(1.0)  # 1 deg/frame pan -> ~13 px top-to-bottom skew
    K = np.array([[lens / 36 * W, 0, W / 2], [0, lens / 36 * W, H / 2], [0, 0, 1.0]])
    rng = np.random.default_rng(5)
    euler, loc = np.array([np.radians(80), 0.0, rng.uniform(-np.pi, np.pi)]), rng.normal(size=3)
    R0, t0 = _pose(euler, loc)
    grid = np.array([(u, v) for u in np.linspace(60, W - 60, 8) for v in np.linspace(40, H - 40, 7)])
    depth = rng.uniform(4.0, 30.0, len(grid))
    Xw = ((np.linalg.inv(K) @ np.column_stack([grid, np.ones(len(grid))]).T).T * depth[:, None] - t0) @ R0
    case = {"name": "rolling", "resolution": [W, H],
            "camera": {"lens": lens, "sensor_width": 36.0, "sensor_height": 24.0, "sensor_fit": "HORIZONTAL",
                       "shift_x": 0.0, "shift_y": 0.0},
            "location": list(loc), "rotation_euler": list(euler), "markers": Xw.tolist(),
            "radii": (BLOB_RADIUS_PX * depth / K[0, 0]).tolist(),
            "animation": {"yaw_rad_per_frame": float(w),
                          "rolling_shutter": {"readout": readout, "row_exposure": 0.02}}}
    _render({"samples": 256, "cases": [case]}, tmp_path)

    def pose_at(dt):
        e = euler.copy()
        e[2] += w * dt
        return _pose(e, loc)

    pred = np.empty((len(Xw), 2))
    for k, X in enumerate(Xw):  # fixed point: the row decides the time, the time decides the row
        y = grid[k, 1]
        for _ in range(20):
            R, t = pose_at((y / H - 0.5) * readout)
            uv, _ = project(K, R, t, X[None])
            y = uv[0, 1]
        pred[k] = uv[0]
    img = cv2.imread(str(tmp_path / "rolling.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 65535.0
    cen = _centroids(img, pred)
    err = np.linalg.norm(cen - pred, axis=1)
    static, _ = project(K, R0, t0, Xw)
    skew = np.linalg.norm(cen - static, axis=1)
    print(f"\nrolling shutter readout {readout}: model error mean {err.mean():.3f} px, max {err.max():.3f} px; "
          f"global-shutter model error max {skew.max():.2f} px")
    assert err.max() < 0.15
    assert skew.max() > 3.0  # the effect is real and the direction/timing model explains it


def test_rendered_markers_match_projection(tmp_path):
    blender = find_blender()
    if blender is None:
        pytest.skip("Blender not found (set NIKO_BLENDER)")
    rng = np.random.default_rng(7)
    spec = {"samples": 256, "cases": [_place_markers(rng, c) for c in CASES]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", _host_path(SCRIPT, blender), "--",
           _host_path(spec_path, blender), _host_path(tmp_path, blender)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    report = json.loads((tmp_path / "report.json").read_text())
    print(f"\nrendered with {report['device']} in Blender {report['blender_version']}")

    worst = 0.0
    for case in spec["cases"]:
        trk = load_blender_cameras(tmp_path / f"{case['name']}_cameras.json")
        uv, _ = project(trk.K[0], trk.R[0], trk.t[0], np.array(case["markers"]))
        img = cv2.imread(str(tmp_path / f"{case['name']}.png"), cv2.IMREAD_UNCHANGED)
        assert img is not None and img.dtype == np.uint16 and img.ndim == 2
        assert (img.shape[1], img.shape[0]) == tuple(case["resolution"])
        cen = _centroids(img.astype(np.float64) / 65535.0, uv)
        err = np.linalg.norm(cen - uv, axis=1)
        bias = (cen - uv).mean(axis=0)
        print(f"{case['name']:>14}: mean {err.mean():.3f} px, max {err.max():.3f} px, "
              f"bias ({bias[0]:+.3f}, {bias[1]:+.3f}) px, {len(err)} markers")
        worst = max(worst, float(err.max()))
    assert worst < TOL_PX
