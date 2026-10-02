"""Dev check: accuracy of a tracks.npz against ground-truth cameras (signed bias, per-query-distance).

    uv run python scripts/dev/track_accuracy.py <tracks_dir> [<model_size_h> <model_size_w>]
    (with a model size, CoTracker is first re-run into <tracks_dir> at that size)
"""

import os
import sys
from pathlib import Path

import numpy as np

from niko.backend import run_backend
from niko.camio import CameraTrack
from niko.geometry import project
from niko.triangulate import reprojection_errors, triangulate_tracks

home = Path(os.environ["NIKO_HOME"])
shot_dir = home / "runs/smoke/smoke_orbit"
out = Path(sys.argv[1])
if len(sys.argv) > 3:
    size = [int(sys.argv[2]), int(sys.argv[3])]
    r = run_backend("cotracker", "tracks", shot_dir, out, {"query_every": 10, "model_size": size})
    print(f"cotracker at {size}: {r['runtime_s']:.1f}s, peak VRAM {r['peak_vram_mb']:.0f} MB")

d = np.load(out / "tracks.npz")
gt = CameraTrack.load(home / "bench/smoke/smoke_orbit/gt/cameras.json")
xy, vis, qf = d["xy"], d["vis"], d["query_frame"]
X, ok = triangulate_tracks(gt, xy, vis)
err = reprojection_errors(gt, X, xy, vis)

# signed residuals (projection - observation) over inlier observations
T, N = vis.shape
res = []
dist = []
for t in range(T):
    m = vis[t] & ok & (err[t] < 3)
    uv, _ = project(gt.K[t], gt.R[t], gt.t[t], X[m])
    res.append(uv - xy[t, m])
    dist.append(np.abs(t - qf[m]))
res, dist = np.concatenate(res), np.concatenate(dist)
e = err[np.isfinite(err)]
print(f"all obs: median {np.median(e):.3f} px, mean {e.mean():.3f}, p90 {np.percentile(e, 90):.3f}, "
      f"outliers >3px {100 * (e > 3).mean():.1f}%")
print(f"inlier signed bias (proj - obs): x {res[:, 0].mean():+.3f} px, y {res[:, 1].mean():+.3f} px")
for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 30)):
    m = (dist >= lo) & (dist < hi)
    if m.any():
        print(f"  {lo:>2}-{hi:<2} frames from query: median |r| {np.median(np.linalg.norm(res[m], axis=1)):.3f} px ({m.sum()} obs)")
