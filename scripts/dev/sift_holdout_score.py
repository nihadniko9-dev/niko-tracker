"""Would auto-select choose better if it scored on SIFT tracks that refinement never saw?

CoTracker3 held-out segments drift (niko.tracks); in a zoom the drift is radial, and a 6-dof
camera can absorb radial motion into point depth, so a wrong dolly solution can out-score the
true rotation+zoom one (zoom_in: 0.260 vs 0.307 px). SIFT tracks do not drift. Refinement takes
the 8000 longest SIFT tracks (rotation-only: 2500), so every SIFT track ranked after 8000 is
unseen by all refined candidates (raw COLMAP did match them).

Prints, per shot: the candidate chosen by the current rule, and by the median reprojection of
those unseen SIFT tracks (plus the same jitter term), next to the GT rotation / focal error.

usage: python scripts/dev/sift_holdout_score.py <run_dir> [shot ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.pipeline.select import direction_errors, is_rotation_only
from niko.triangulate import reprojection_errors, triangulate_tracks

SKIP = 8000


def sift_reproj(trk: CameraTrack, xy, vis) -> float:
    vis = vis & trk.valid[:, None]
    if is_rotation_only(trk):
        err = direction_errors(trk, xy, vis)
    else:
        X, ok = triangulate_tracks(trk, xy, vis)
        err = reprojection_errors(trk, X, xy, vis)
    e = err[np.isfinite(err)]
    return float(np.median(e)) if e.size >= 50 else float("inf")


def main():
    run = Path(sys.argv[1])
    shots = sys.argv[2:] or sorted(p.name for p in run.iterdir() if (p / "solve.json").exists())
    for shot in shots:
        d = run / shot
        rep = json.loads((d / "solve.json").read_text())
        sel, gt = rep.get("select") or {}, rep.get("gt_eval", {})
        if not (d / "sift/sift_tracks.npz").exists() or not sel:
            print(f"== {shot}: no SIFT tracks / no selection")
            continue
        sift = dict(np.load(d / "sift/sift_tracks.npz"))
        cnt = sift["vis"].sum(0)
        order = np.argsort(cnt)[::-1]
        hold = order[SKIP:]
        hold = hold[cnt[hold] >= 3]
        xy, vis = sift["xy"][:, hold].astype(float), sift["vis"][:, hold]
        rows = []
        for name in sel["ranking"]:
            s = sel["scores"][name]
            if not np.isfinite(s["score_px"]):
                continue
            trk = CameraTrack.load(d / "candidates" / name / "cameras.json")
            r = sift_reproj(trk, xy, vis)
            rows.append((name, s, r, r + 0.5 * s["jitter_px"], gt.get(name, {})))
        print(f"== {shot}: {len(hold)} unseen SIFT tracks, current pick {rep['selected']}")
        for name, s, r, sc, g in sorted(rows, key=lambda x: (x[1]["success_rate"] < 0.95, x[3])):
            print(f"  {name:<30} cotracker {s['reproj_median_px']:.3f}  sift {r:.3f}  score {sc:.3f}  "
                  f"cx {s['complexity']} | rot {g.get('rot_err_deg_max', float('nan')):.3f} "
                  f"focal {g.get('focal_err_pct_max', float('nan')):.2f}")


if __name__ == "__main__":
    main()
