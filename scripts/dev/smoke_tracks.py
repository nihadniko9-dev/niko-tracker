"""Dev check: ingest the smoke shot, run CoTracker3, measure tracks against ground-truth cameras.

    uv run python scripts/dev/smoke_tracks.py
"""

import json
import os
from pathlib import Path

import numpy as np

from niko.backend import run_backend
from niko.camio import CameraTrack
from niko.pipeline.ingest import ingest
from niko.triangulate import reprojection_errors, triangulate_tracks

home = Path(os.environ["NIKO_HOME"])
bench = home / "bench/smoke/smoke_orbit"
shot_dir = home / "runs/smoke/smoke_orbit"
spec = json.loads((bench / "spec.json").read_text())
shot = ingest(bench / "frames", shot_dir, fps=spec["fps"], frame_start=spec["frame_start"])
print(f"ingest: {shot['n_frames']} frames {shot['width']}x{shot['height']}, proxy {shot['proxy']}")

res = run_backend("cotracker", "tracks", shot_dir, shot_dir / "tracks", {"query_every": 10})
print(f"cotracker: {res['runtime_s']:.1f}s, peak VRAM {res['peak_vram_mb']:.0f} MB, stats {res['stats']}")

d = np.load(shot_dir / "tracks/tracks.npz")
gt = CameraTrack.load(bench / "gt/cameras.json")
X, ok = triangulate_tracks(gt, d["xy"], d["vis"])
err = reprojection_errors(gt, X, d["xy"], d["vis"])
e = err[np.isfinite(err)]
print(f"tracks triangulated with GT cameras: {ok.sum()}/{len(ok)}")
print(f"reprojection vs GT cameras: median {np.median(e):.3f} px, mean {e.mean():.3f} px, "
      f"p90 {np.percentile(e, 90):.3f} px, p99 {np.percentile(e, 99):.3f} px over {e.size} observations")
