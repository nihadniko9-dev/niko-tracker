"""Dev: does sub-pixel Lucas-Kanade refinement at full resolution make CoTracker3 tracks more accurate?

    uv run python scripts/dev/lk_refine_test.py <solve_dir> <gt_cameras.json>

Accuracy proxy: triangulate each track with the ground-truth cameras and measure the reprojection
error of every observation (lower = the 2D positions agree better with the true geometry).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from niko.camio import CameraTrack
from niko.triangulate import reprojection_errors, triangulate_tracks

solve_dir, gt = Path(sys.argv[1]), CameraTrack.load(sys.argv[2])
d = np.load(solve_dir / "tracks/tracks.npz")
xy, vis, qf = d["xy"].astype(np.float64), d["vis"], d["query_frame"]
files = sorted((solve_dir / "frames").glob("*.*"))
T, N = vis.shape
gray = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY) for f in files]


def accuracy(label, pts):
    X, ok = triangulate_tracks(gt, pts, vis)
    e = reprojection_errors(gt, X, pts, vis)
    e = e[np.isfinite(e)]
    print(f"{label:<28} median {np.median(e):.3f} px  mean(<3px) {e[e < 3].mean():.3f}  "
          f"p90 {np.percentile(e, 90):.3f}  >3px {100 * (e > 3).mean():.1f}%")


def lk_refine(max_shift=1.5, win=15, level=1, max_gap=None, template="query"):
    out = xy.copy()
    changed = 0
    for t in range(T):
        for q in np.unique(qf):
            m = vis[t] & (qf == q)
            if template == "prev":
                continue
            if max_gap is not None and abs(t - q) > max_gap:
                continue
            if t == q or not m.any():
                continue
            p0 = (xy[q, m] - 0.5).astype(np.float32).reshape(-1, 1, 2)
            p1 = (xy[t, m] - 0.5).astype(np.float32).reshape(-1, 1, 2)
            p1r, st, err = cv2.calcOpticalFlowPyrLK(gray[q], gray[t], p0, p1.copy(), winSize=(win, win), maxLevel=level,
                                                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                                                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            shift = np.linalg.norm((p1r - p1).reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (shift < max_shift)
            idx = np.nonzero(m)[0][good]
            out[t, idx] = p1r.reshape(-1, 2)[good] + 0.5
            changed += int(good.sum())
    return out, changed


accuracy("CoTracker3 (as is)", xy)
for gap in (8, 24, None):
    t0 = time.time()
    ref, n = lk_refine(max_gap=gap)
    accuracy(f"LK from query frame, gap<={gap}", ref)
    print(f"   changed {n} observations in {time.time() - t0:.0f}s")
