"""OpenCV <-> Blender conversion checked against Blender 5.2 itself.

Blender projects the points (bpy_extras world_to_camera_view, which uses the
C-side camera view frame); we project the same points through our K, R, t.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from niko.geometry import (
    K_to_blender,
    blender_matrix_world_to_opencv,
    blender_to_K,
    opencv_to_blender_matrix_world,
    project,
)
from niko.paths import REPO, find_blender

from conftest import random_rotation

pytestmark = pytest.mark.blender

PROBE = REPO / "blender" / "tests" / "probe_projection.py"
# Blender works in float32: ~1e-7 relative error on world coordinates, times focal
# lengths up to ~4e4 px here. A formula mistake (half-pixel origin, aspect, shift
# units) shows up as >= 0.5 px, so this still separates right from wrong.
TOL_PX = 2e-2
NDC_MARGIN = 0.25  # compare points in (or just around) the frame, where precision matters


def _host_path(p: Path, blender: str) -> str:
    """Windows blender.exe called from WSL needs Windows paths."""
    if sys.platform != "win32" and blender.endswith(".exe"):
        return subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()
    return str(p)


def run_probe(cases: list, tmp_path: Path) -> dict:
    blender = find_blender()
    if blender is None:
        pytest.skip("Blender not found (set NIKO_BLENDER)")
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    src.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", _host_path(PROBE, blender), "--",
           _host_path(src, blender), _host_path(dst, blender)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = json.loads(dst.read_text(encoding="utf-8"))
    assert out["blender_version"].startswith("5.2"), out["blender_version"]
    return out


def _points_in_front(rng, R, t, half_tan=0.7, n=400):
    """World points in front of an OpenCV camera (R, t), inside a cone of half-angle atan(half_tan)."""
    z = rng.uniform(2.0, 60.0, n)
    xy = rng.uniform(-half_tan, half_tan, (n, 2)) * z[:, None]
    Xc = np.column_stack([xy, z])
    return (Xc - t) @ R  # R^T (Xc - t)


def _in_frame(ndc):
    ndc = np.asarray(ndc)
    return np.all((ndc[:, :2] > -NDC_MARGIN) & (ndc[:, :2] < 1 + NDC_MARGIN), axis=1)


def _euler_xyz(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _ndc_to_pixels(ndc, w, h):
    ndc = np.asarray(ndc)
    return np.column_stack([ndc[:, 0] * w, (1.0 - ndc[:, 1]) * h]), ndc[:, 2]


def test_blender_camera_to_opencv(rng, tmp_path):
    """Random Blender cameras (every sensor fit, shifts, pixel aspect, object scale)."""
    cases, expected_R = [], []
    fits = ["AUTO", "HORIZONTAL", "VERTICAL"]
    # Only percentages that divide the size exactly: with e.g. 37 %, Blender renders an
    # integer size (710x399) but world_to_camera_view keeps the unscaled 16:9 frame, so
    # the two disagree inside Blender itself. We follow the render (integer size).
    sizes = [(1920, 1080), (1080, 1920), (3840, 1600), (1000, 1000), (1280, 720)]
    for k in range(60):
        w, h = sizes[k % len(sizes)]
        pa = [1.0, 1.0]
        if k % 4 == 3:
            pa[rng.integers(2)] = float(rng.uniform(1.0, 2.0))
        euler = rng.uniform(-np.pi, np.pi, 3)
        loc = rng.normal(scale=5.0, size=3)
        R_c2w_bl = _euler_xyz(*euler)
        R = (R_c2w_bl @ np.diag([1.0, -1.0, -1.0])).T
        t = -R @ loc
        expected_R.append(R)
        lens = float(rng.uniform(10, 200))
        sw, sh = float(rng.uniform(15, 40)), float(rng.uniform(12, 30))
        # coarse cone that covers the frame for any fit / aspect / shift (no exact formula needed)
        half_tan = 1.2 * (0.5 * max(w, h) / min(w, h) + 0.3) * max(sw, sh) / lens
        cases.append({
            "camera": {
                "lens": lens,
                "sensor_width": sw,
                "sensor_height": sh,
                "sensor_fit": fits[k % 3],
                "shift_x": float(rng.uniform(-0.3, 0.3)),
                "shift_y": float(rng.uniform(-0.3, 0.3)),
                "resolution_x": w, "resolution_y": h,
                "resolution_percentage": int(rng.choice([100, 50])),
                "pixel_aspect_x": pa[0], "pixel_aspect_y": pa[1],
            },
            "location": loc.tolist(),
            "rotation_euler": euler.tolist(),
            "scale": rng.uniform(0.2, 3.0, 3).tolist(),
            "points": _points_in_front(rng, R, t, half_tan=half_tan).tolist(),
        })

    out = run_probe(cases, tmp_path)
    worst, compared = 0.0, 0
    for case, res, R_exp in zip(cases, out["cases"], expected_R):
        c = case["camera"]
        W, H = res["render"]["width"], res["render"]["height"]
        K = blender_to_K(c["lens"], c["sensor_width"], c["sensor_height"], c["sensor_fit"],
                         c["shift_x"], c["shift_y"], W, H, c["pixel_aspect_x"], c["pixel_aspect_y"])
        R, t = blender_matrix_world_to_opencv(np.array(res["matrix_world"]))
        assert np.abs(R - R_exp).max() < 1e-5, "euler composition / scale removal mismatch"
        uv, z = project(K, R, t, np.array(case["points"]))
        uv_bl, z_bl = _ndc_to_pixels(res["ndc"], W, H)
        assert np.allclose(z, z_bl, rtol=1e-5, atol=1e-4)
        keep = _in_frame(res["ndc"])
        assert keep.sum() >= 5, "too few points inside the frame"
        compared += int(keep.sum())
        worst = max(worst, float(np.abs(uv - uv_bl)[keep].max()))
    print(f"\nBlender->OpenCV: max pixel difference {worst:.2e} px "
          f"({compared} points, {len(cases)} cameras)")
    assert worst < TOL_PX


def test_opencv_camera_to_blender(rng, tmp_path):
    """Random OpenCV cameras pushed into Blender through K_to_blender + matrix_world."""
    cases, Ks, Rs, ts = [], [], [], []
    sizes = [(1920, 1080), (1080, 1920), (2048, 858), (720, 720)]
    for k in range(40):
        w, h = sizes[k % len(sizes)]
        fx = rng.uniform(400, 5000)
        fy = fx * (1.0 if k % 3 else rng.uniform(0.85, 1.2))
        K = np.array([[fx, 0, w / 2 + rng.uniform(-0.08, 0.08) * w],
                      [0, fy, h / 2 + rng.uniform(-0.08, 0.08) * h], [0, 0, 1.0]])
        R = random_rotation(rng)
        t = rng.normal(scale=5.0, size=3)
        b = K_to_blender(K, w, h, sensor_width=float(rng.uniform(10, 40)))
        cam = {key: b[key] for key in ("lens", "sensor_width", "sensor_height", "sensor_fit",
                                       "shift_x", "shift_y", "resolution_x", "resolution_y",
                                       "pixel_aspect_x", "pixel_aspect_y")}
        half_tan = 1.3 * 0.5 * max(w, h) / min(fx, fy) + 0.1
        cases.append({
            "camera": cam,
            "matrix_world": opencv_to_blender_matrix_world(R, t).tolist(),
            "points": _points_in_front(rng, R, t, half_tan=half_tan).tolist(),
        })
        Ks.append(K), Rs.append(R), ts.append(t)

    out = run_probe(cases, tmp_path)
    worst, compared = 0.0, 0
    for case, res, K, R, t in zip(cases, out["cases"], Ks, Rs, ts):
        W, H = res["render"]["width"], res["render"]["height"]
        assert (W, H) == (case["camera"]["resolution_x"], case["camera"]["resolution_y"])
        uv, _ = project(K, R, t, np.array(case["points"]))
        uv_bl, _ = _ndc_to_pixels(res["ndc"], W, H)
        keep = _in_frame(res["ndc"])
        assert keep.sum() >= 5, "too few points inside the frame"
        compared += int(keep.sum())
        worst = max(worst, float(np.abs(uv - uv_bl)[keep].max()))
    print(f"\nOpenCV->Blender: max pixel difference {worst:.2e} px "
          f"({compared} points, {len(cases)} cameras)")
    assert worst < TOL_PX
