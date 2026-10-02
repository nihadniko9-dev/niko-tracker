"""Dev: exact track errors vs ground truth (ray-cast in scene.blend), overall and by distance to query.

    uv run python scripts/dev/track_truth_test.py <solve_dir> <bench_shot_dir>
"""

import sys
from pathlib import Path

import numpy as np

from niko.bench.track_truth import gt_tracks, track_errors

solve_dir, shot = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(solve_dir / "tracks/tracks.npz")
xy, vis, qf = d["xy"].astype(np.float64), d["vis"], d["query_frame"]
truth = gt_tracks(shot, xy, qf, solve_dir / "track_truth")
e = track_errors(xy, vis, truth)
v = e[np.isfinite(e)]
print(f"tracks {xy.shape[1]}, static {truth['static'].sum()}, on movers {truth['dynamic'].sum()}")
print(f"ALL   median {np.median(v):.3f} px  mean(<3) {v[v < 3].mean():.3f}  p90 {np.percentile(v, 90):.3f}  "
      f">1px {100 * (v > 1).mean():.1f}%  >3px {100 * (v > 3).mean():.1f}%")
dt = np.abs(np.arange(xy.shape[0])[:, None] - qf[None, :])
for lo, hi in ((1, 2), (2, 8), (8, 24), (24, 48), (48, 200)):
    m = np.isfinite(e) & (dt >= lo) & (dt < hi)
    if m.any():
        w = e[m]
        print(f"  {lo:>3}-{hi:<3} frames from query: median {np.median(w):.3f} px  >3px {100 * (w > 3).mean():.1f}%  ({m.sum()} obs)")
signed = (xy - truth["xy"])[np.isfinite(e) & (e < 3)]
print(f"signed bias (tracker - truth): x {signed[:, 0].mean():+.3f} px, y {signed[:, 1].mean():+.3f} px")
np.save(solve_dir / "track_truth" / "errors.npy", e)
