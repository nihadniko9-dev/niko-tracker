"""Smoke test: a 30-frame 540p synthetic shot through the whole pipeline (needs WSL envs + GPU).

    uv run pytest -m smoke -s
"""

import json
import os
import time
from pathlib import Path

import pytest

from niko.camio import CameraTrack
from niko.paths import REPO, env_python, niko_home

pytestmark = pytest.mark.smoke

BACKENDS = ("sam3", "cotracker", "colmap", "megasam", "da3")


@pytest.mark.skipif(not all(env_python(b).exists() for b in BACKENDS), reason="backend envs not installed")
def test_smoke_pipeline(tmp_path):
    from niko.bench.generate import generate_shot
    from niko.pipeline.solve import solve

    t0 = time.time()
    shot = tmp_path / "smoke_orbit"
    info = generate_shot(REPO / "configs/shots/smoke_orbit.json", shot)
    assert info["spec"]["n_frames"] == 30 and info["spec"]["height"] == 540

    lines = []
    report = solve(shot, tmp_path / "solve", log=lambda s: (lines.append(s), print(s)))
    elapsed = time.time() - t0
    print(f"smoke total {elapsed:.0f}s")

    failed = {k: v.get("error") for k, v in report["stages"].items() if not v["ok"]}
    assert not failed, failed
    assert report["ok"] and report["selected"]
    sel = report["gt_eval"][report["selected"]]
    assert sel["meets_targets"], sel
    out = tmp_path / "solve" / "selected"
    trk = CameraTrack.load(out / "cameras.json")  # validates against the schema
    assert trk.valid.all()
    assert (out / "points.ply").stat().st_size > 1000
    assert (out / "import_blender.py").exists()
    masks = json.loads((tmp_path / "solve/masks/masks.json").read_text())
    assert set(masks["prompts"]) >= {"person", "car", "sky"}
    assert elapsed < 600, "smoke test should take a few minutes"
