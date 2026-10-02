"""Dev check: run camera candidates on the ingested smoke shot and score them against ground truth.

    uv run python scripts/dev/smoke_candidates.py colmap_global colmap_incremental da3
"""

import os
import sys
from pathlib import Path

from niko.camio import CameraTrack
from niko.evaluate import evaluate_against_gt, summary_line
from niko.pipeline.candidates import run_candidate

home = Path(os.environ["NIKO_HOME"])
shot_dir = home / "runs/smoke/smoke_orbit"
gt = CameraTrack.load(home / "bench/smoke/smoke_orbit/gt/cameras.json")
for method in sys.argv[1:]:
    try:
        trk, res = run_candidate(method, shot_dir, shot_dir / "candidates")
    except Exception as e:
        print(f"{method:<22} ERROR {type(e).__name__}: {str(e)[-600:]}")
        continue
    r = evaluate_against_gt(trk, gt)
    print(summary_line(method, r))
    vram = res["peak_vram_mb"]
    print(f"{'':<22} runtime {res['runtime_s']:.1f}s, peak VRAM {vram if vram is None else round(vram)} MB, "
          f"stats {res.get('stats')}")
    if not r.get("failed"):
        print(f"{'':<22} ATE mode {r['ate_mode']}, rpe1 rot {r.get('rpe1_rot_deg', float('nan')):.4f} deg, "
              f"jitter {r['jitter_trans_pct']:.4f}% / {r['jitter_rot_deg']:.4f} deg "
              f"(GT {r['gt_jitter_trans_pct']:.4f}% / {r['gt_jitter_rot_deg']:.4f} deg)")
