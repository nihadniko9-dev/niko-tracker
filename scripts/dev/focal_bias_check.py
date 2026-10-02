"""Does the bundle adjustment pull the focal away from the truth, and do CoTracker3 segments cause it?

usage: python scripts/dev/focal_bias_check.py <run_dir> <shot> <candidate> [cotracker_budget ...]
env SIFT_TRAIN=n: SIFT tracks given to the adjustment (default: the pipeline's 8000 longest)
Refines the candidate with the SIFT training tracks plus N CoTracker3 segments (N = 0 means SIFT
only) and prints the focal error against ground truth for each N.
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.refine import RefineOptions, refine
from niko.tracks import split_sift


def main():
    run, shot, cand = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    budgets = [int(x) for x in sys.argv[4:]] or [0, 1000, 4000]
    d = run / shot
    spec = json.loads((d / "shot.json").read_text())
    gt = CameraTrack.load(Path(spec["source"]).parent / "gt/cameras.json")
    f_gt = float(np.median(gt.K[gt.valid, 0, 0]))
    tracks = dict(np.load(d / "tracks/tracks.npz"))
    import os
    n_train = int(os.environ.get("SIFT_TRAIN", RefineOptions().max_sift))
    sift, _ = split_sift(dict(np.load(d / "sift/sift_tracks.npz")), n_train)
    c0 = CameraTrack.load(d / "candidates" / cand / "cameras.json")
    f0 = float(np.median(c0.K[c0.valid, 0, 0]))
    print(f"SIFT tracks for the adjustment: {sift['vis'].shape[1]}")
    print(f"{shot}/{cand}: GT focal {f_gt:.1f} px, start {f0:.1f} px ({100 * (f0 / f_gt - 1):+.2f} %)")
    for n in budgets:
        trk, rep = refine(c0, tracks, d / "dev_bias" / f"cot{n}", RefineOptions(max_tracks=n, max_sift=n_train),
                          sift=sift)
        f = float(np.median(trk.K[trk.valid, 0, 0]))
        print(f"  CoTracker segments {n:5d}: focal {f:.1f} px ({100 * (f / f_gt - 1):+.2f} %), "
              f"BA median {rep['rounds'][-1]['median_px']:.3f} px", flush=True)


if __name__ == "__main__":
    main()
