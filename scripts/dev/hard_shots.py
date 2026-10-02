"""Dev: on the shots COLMAP failed, run MegaSaM + DA3, build SIFT tracks, refine every candidate in
several modes, and compare with GT.

    uv run python scripts/dev/hard_shots.py shot1,shot2 [modes: shared_focal:none,per_frame_focal:none,...]
"""

import os
import sys
from pathlib import Path

import numpy as np

from niko.backend import run_backend
from niko.camio import CameraTrack
from niko.evaluate import evaluate_against_gt
from niko.pipeline.candidates import run_candidate
from niko.pipeline.select import score_candidate
from niko.refine import RefineOptions, refine

home = Path(os.environ["NIKO_HOME"])
shots = sys.argv[1].split(",")
modes = [m.split(":") for m in (sys.argv[2] if len(sys.argv) > 2 else "shared_focal:none").split(",")]


def fmt(trk, gt, tracks):
    s = score_candidate(trk, tracks)
    est_size = (gt.extra["scene_size"] * s["median_depth"] / gt.extra["median_depth"]
                if s.get("median_depth") and gt.extra.get("median_depth") else None)
    r = evaluate_against_gt(trk, gt, est_scene_size=est_size)
    if r.get("failed"):
        return f"FAILED ({r['n_valid']}/{r['n_frames']})"
    return (f"frames {int(trk.valid.sum()):>3}  ATE {r['ate_pct']:.4f}%  rot {r['rot_err_deg_max']:.4f}  "
            f"focal {r['focal_err_pct_max']:.2f}%  held-out {s.get('reproj_median_px', float('nan')):.3f}  "
            f"{'MET' if r['meets_targets'] else 'missed'}")


for shot in shots:
    S = home / "runs/cp2_colmap_v1" / shot
    gt = CameraTrack.load(home / "bench/synthetic_v1" / shot / "gt/cameras.json")
    for m in ("megasam", "da3"):
        if not (S / "candidates" / m / "cameras.json").exists():
            try:
                run_candidate(m, S, S / "candidates", {"max_frames": 40} if m == "da3" else {})
            except Exception as e:
                print(f"[{shot}] {m} candidate FAILED: {str(e)[-300:]}")
    db = S / "candidates/colmap_global/colmap/database.db"
    sift = None
    if db.exists():
        if not (S / "sift/sift_tracks.npz").exists():
            run_backend("colmap", "sift_tracks", S, S / "sift", {"db": str(db), "width": 1920, "height": 1080,
                                                               "proxy_width": 1920, "proxy_height": 1080})
        sift = dict(np.load(S / "sift/sift_tracks.npz"))
    tracks = dict(np.load(S / "tracks/tracks.npz"))
    print(f"\n######## {shot}  (SIFT tracks: {None if sift is None else sift['vis'].shape[1]})")
    for cdir in sorted((S / "candidates").iterdir()):
        if not (cdir / "cameras.json").exists():
            continue
        cand = CameraTrack.load(cdir / "cameras.json")
        print(f"{cdir.name:<20} before   {fmt(cand, gt, tracks)}")
        for intr, dist in modes:
            try:
                ref, rep = refine(cand, tracks, S / "refine_dev" / f"{cdir.name}_{intr}_{dist}",
                                  RefineOptions(intrinsics=intr, distortion=dist), sift=sift)
                print(f"{'':<20} {intr[:9]}/{dist:<4} {fmt(ref, gt, tracks)}  k1 {rep['k1']:+.4f} k2 {rep['k2']:+.4f}")
            except Exception as e:
                print(f"{'':<20} {intr}/{dist} refine FAILED: {str(e)[-200:]}")
