"""Dev: CoTracker3 runtime / VRAM / accuracy vs model size on an ingested benchmark shot.

    uv run python scripts/dev/track_budget.py <run_shot_dir> <gt_cameras.json> H W CHUNK [H W CHUNK ...]
"""

import sys
from pathlib import Path

import numpy as np

from niko.backend import run_backend
from niko.camio import CameraTrack
from niko.triangulate import reprojection_errors, triangulate_tracks

shot_dir, gt = Path(sys.argv[1]), CameraTrack.load(sys.argv[2])
cfgs = [tuple(int(x) for x in sys.argv[i:i + 3]) for i in range(3, len(sys.argv), 3)]
for h, w, chunk in cfgs:
    out = shot_dir / f"tracks_budget_{h}x{w}_{chunk}"
    try:
        r = run_backend("cotracker", "tracks", shot_dir, out,
                        {"query_every": 10, "model_size": [h, w], "chunk": chunk}, timeout=900)
    except Exception as e:
        print(f"{h}x{w} chunk {chunk}: FAILED {type(e).__name__}: {str(e)[-200:]}")
        continue
    d = np.load(out / "tracks.npz")
    X, ok = triangulate_tracks(gt, d["xy"], d["vis"])
    err = reprojection_errors(gt, X, d["xy"], d["vis"])
    e = err[np.isfinite(err)]
    print(f"{h}x{w} chunk {chunk}: {r['runtime_s']:.0f}s, peak VRAM {r['peak_vram_mb']:.0f} MB, "
          f"{r['stats']['n_tracks']} tracks; vs GT median {np.median(e):.2f} px (full-res), "
          f"p90 {np.percentile(e, 90):.2f}, >3px {100 * (e > 3).mean():.1f}%")
