"""Dev: refine every candidate of a solved shot and compare with ground truth, before / after.

    uv run python scripts/dev/refine_check.py <solve_dir> <gt_cameras.json> [intrinsics] [distortion]
"""

import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.evaluate import evaluate_against_gt
from niko.pipeline.select import score_candidate
from niko.refine import RefineOptions, refine

import os

solve_dir, gt = Path(sys.argv[1]), CameraTrack.load(sys.argv[2])
opts = RefineOptions(intrinsics=sys.argv[3] if len(sys.argv) > 3 else "shared_focal",
                     distortion=sys.argv[4] if len(sys.argv) > 4 else "none",
                     max_track_len=int(os.environ.get("SEG", "16")),
                     rounds=int(os.environ.get("ROUNDS", "2")))
only = set(filter(None, os.environ.get("ONLY", "").split(",")))
tracks = dict(np.load(solve_dir / os.environ.get("TRACKS", "tracks/tracks.npz")))
sift = dict(np.load(solve_dir / os.environ["SIFT"])) if os.environ.get("SIFT") else None
print("SIFT tracks:", None if sift is None else sift["vis"].shape)
print(f"options: {opts}")


def line(tag, trk):
    r = evaluate_against_gt(trk, gt)
    s = score_candidate(trk, tracks)
    if r.get("failed"):
        return f"{tag:<7} FAILED"
    return (f"{tag:<7} frames {int(trk.valid.sum()):>3}/{trk.n_frames}  ATE {r['ate_pct']:.4f}%  "
            f"rot {r['rot_err_deg_mean']:.4f}/{r['rot_err_deg_max']:.4f} deg  "
            f"focal {r['focal_err_pct_max']:.3f}%  held-out {s.get('reproj_median_px', float('nan')):.3f} px  "
            f"{'MET' if r['meets_targets'] else 'missed'}")


for cdir in sorted((solve_dir / "candidates").iterdir()):
    p = cdir / "cameras.json"
    if not p.exists() or (only and cdir.name not in only):
        continue
    cand = CameraTrack.load(p)
    print(f"== {cdir.name}")
    print(line("before", cand))
    try:
        ref, rep = refine(cand, tracks, solve_dir / "refine_dev" / cdir.name, opts, sift=sift)
    except Exception as e:
        print(f"refine FAILED: {type(e).__name__}: {str(e)[-800:]}")
        continue
    print(line("after", ref))
    print(f"        BA: {rep['seconds']}s, {rep['observations']} obs, {rep['segments_used']} segments, "
          f"initial median {rep['initial_median_px']:.3f} px, k1 {rep['k1']:.4f}, "
          f"pnp-added {len(rep['frames_added_by_pnp'])}")
    for k, r in enumerate(rep["rounds"]):
        print(f"        round {k}: {r['iterations']} it, {r['termination']}, median {r['median_px']:.3f} "
              f"rms {r['rms_px']:.3f} thr {r['threshold_px']:.2f}")
    ev = evaluate_against_gt(ref, gt)
    if not ev.get("failed"):
        from niko.metrics import rotation_errors_deg
        ok = ref.valid & gt.valid
        err = rotation_errors_deg(ref.R_c2w[ok], gt.R_c2w[ok])
        worst = np.argsort(err)[-5:][::-1]
        print(f"        worst frames: {[(int(np.nonzero(ok)[0][i]), round(float(err[i]), 3)) for i in worst]}")
