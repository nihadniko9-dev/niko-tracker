"""cameras.json -> import_blender.py -> Blender 5.2 -> evaluated camera -> cameras.json must match.

Exercises the export script, the per-frame Blender keys and the import-camera path
(nikobpy.export_cameras + niko.blendercam) together.
"""

import json
import subprocess

import cv2
import numpy as np
import pytest

from niko.blendercam import load_blender_cameras
from niko.camio import CameraTrack
from niko.paths import REPO, find_blender
from niko.pipeline.export import export_solve

from conftest import axis_angle
from test_blender_convention import _host_path

pytestmark = pytest.mark.blender


def make_track(n=12):
    trk = CameraTrack.empty("unit", 1280, 720, 24.0, 1001, n, name="exp")
    for i in range(n):
        a = np.radians(3 * i)
        C = np.array([6 * np.cos(a), 6 * np.sin(a), 1.5 + 0.05 * i])
        z = -C / np.linalg.norm(C)
        x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        Rc2w = np.column_stack([x, y, z]) @ axis_angle([0, 0, 1], 0.7 * i)  # add some roll
        trk.R[i] = Rc2w.T
        trk.t[i] = -Rc2w.T @ C
        f = 1100.0 + 15 * i  # zoom
        trk.K[i] = [[f, 0, 640 + 7.5], [0, f, 360 - 4.25], [0, 0, 1]]
        trk.dist[i] = 0
    trk.valid[:] = True
    trk.valid[5] = False
    trk.K[5] = trk.R[5] = trk.t[5] = trk.dist[5] = np.nan
    return trk


def test_export_roundtrip_through_blender(tmp_path):
    blender = find_blender()
    if blender is None:
        pytest.skip("Blender not found")
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(12):
        cv2.imwrite(str(frames / f"{i:06d}.jpg"), np.full((720, 1280, 3), 40 + 10 * i, np.uint8))
    trk = make_track()
    export_solve(trk, None, tmp_path / "selected", frames, "000000.jpg")
    out = tmp_path / "back.json"
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", _host_path(tmp_path / "selected" / "import_blender.py", blender),
           "--python", _host_path(REPO / "blender" / "tests" / "export_back.py", blender),
           "--", _host_path(out, blender)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "footage not found" not in proc.stdout

    back = load_blender_cameras(out)
    assert (back.width, back.height, back.frame_start, back.n_frames) == (1280, 720, 1001, 12)
    ok = trk.valid
    assert np.abs(back.R[ok] - trk.R[ok]).max() < 2e-6
    assert np.abs(back.centers[ok] - trk.centers[ok]).max() < 2e-5
    assert np.abs(back.K[ok] - trk.K[ok]).max() < 2e-3  # px, Blender stores lens/shift as float32
    objects = json.loads(out.read_text())["extra"]["objects"]
    assert any(o.startswith("niko_exp") for o in objects)
