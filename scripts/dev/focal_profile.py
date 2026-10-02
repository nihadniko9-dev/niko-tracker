"""Is the focal length observable on a shot? Bundle-adjust with the focal FIXED at a range of
values and look at the held-out reprojection (auto-select's measure) against the GT focal error.

A flat profile means the tracks cannot tell the focal (pure forward motion: image flow does not
depend on it), so any choice between candidates' focals is noise.

usage: python scripts/dev/focal_profile.py <run_dir> <shot> <candidate> [focal_px ...]
Scores on the held-out SIFT tracks (auto-select's measure), refinement uses the training SIFT tracks.
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.pipeline.select import score_candidate
from niko.refine import RefineOptions, copy_track, refine
from niko.tracks import split_sift


def main():
    run, shot, cand = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    focals = [float(s) for s in sys.argv[4:]]
    d = run / shot
    spec = json.loads((d / "shot.json").read_text())
    gt = CameraTrack.load(Path(spec["source"]).parent / "gt/cameras.json")
    tracks = dict(np.load(d / "tracks/tracks.npz"))
    sift, held = split_sift(dict(np.load(d / "sift/sift_tracks.npz")), RefineOptions().max_sift)
    c0 = CameraTrack.load(d / "candidates" / cand / "cameras.json")
    f_gt = float(np.median(gt.K[gt.valid, 0, 0]))
    f0 = float(np.median(c0.K[c0.valid, 0, 0]))
    print(f"{shot} / {cand}: start focal {f0:.1f} px, GT {f_gt:.1f} px ({100 * (f0 / f_gt - 1):+.1f} %)")
    print(f"{'focal px':>9}{'vs GT %':>9}{'BA median px':>13}{'held-out px':>12}{'score px':>10}")
    for fx in focals or [f0 * s for s in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)]:
        c = copy_track(c0)
        c.K[c.valid, 0, 0] = c.K[c.valid, 1, 1] = fx
        trk, rep = refine(c, tracks, d / "dev_focal" / f"{fx:.0f}", RefineOptions(intrinsics="fixed"), sift=sift)
        sc = score_candidate(trk, tracks, held)
        f = float(np.median(trk.K[trk.valid, 0, 0]))
        print(f"{f:>9.1f}{100 * (f / f_gt - 1):>+9.1f}{rep['rounds'][-1]['median_px']:>13.3f}"
              f"{sc['reproj_median_px']:>12.3f}{sc['score_px']:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
