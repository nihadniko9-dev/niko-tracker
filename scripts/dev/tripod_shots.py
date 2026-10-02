"""Dev: rotation-only solver on the static-camera shots, from the COLMAP candidate, with SIFT tracks.

    uv run python scripts/dev/tripod_shots.py zoom_in,tripod_rotation_only,pan_low_parallax,whip_pan_blur
"""

import os
import sys
from pathlib import Path

import numpy as np

from niko.backend import run_backend
from niko.camio import CameraTrack
from niko.evaluate import evaluate_against_gt
from niko.pipeline.select import score_candidate
from niko.tripod import TripodOptions, solve_tripod

home = Path(os.environ["NIKO_HOME"])
for shot in sys.argv[1].split(","):
    S = home / "runs/cp2_colmap_v1" / shot
    gt = CameraTrack.load(home / "bench/synthetic_v1" / shot / "gt/cameras.json")
    db = S / "candidates/colmap_global/colmap/database.db"
    if not (S / "sift/sift_tracks.npz").exists():
        run_backend("colmap", "sift_tracks", S, S / "sift", {"db": str(db), "width": 1920, "height": 1080,
                                                           "proxy_width": 1920, "proxy_height": 1080})
    sift = dict(np.load(S / "sift/sift_tracks.npz"))
    tracks = dict(np.load(S / "tracks/tracks.npz"))
    print(f"\n######## {shot}")
    for base in ("colmap_global", "colmap_incremental", "megasam", "da3"):
        p = S / "candidates" / base / "cameras.json"
        if not p.exists():
            continue
        cand = CameraTrack.load(p)
        if cand.valid.mean() < 0.2:
            print(f"{base:<19} skipped ({int(cand.valid.sum())} frames)")
            continue
        for intr in ("shared_focal", "per_frame_focal"):
            try:
                est, rep = solve_tripod(cand, sift, TripodOptions(intrinsics=intr))
            except Exception as e:
                print(f"{base:<19} {intr}: FAILED {type(e).__name__}: {str(e)[:200]}")
                continue
            r = evaluate_against_gt(est, gt, est_scene_size=1.0)
            s = score_candidate(est, tracks)
            print(f"{base:<19} {intr:<16} frames {int(est.valid.sum()):>3}  rot {r['rot_err_deg_mean']:.4f}/"
                  f"{r['rot_err_deg_max']:.4f}  focal {r['focal_err_pct_mean']:.2f}/{r['focal_err_pct_max']:.2f}%  "
                  f"held-out {s.get('reproj_median_px', float('nan')):.3f}  "
                  f"{'MET' if r['meets_targets'] else 'missed'}  ({rep['seconds']}s, "
                  f"{rep['rounds'][-1]['median_px']:.3f} px)")
