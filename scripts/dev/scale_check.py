"""How close is the real-world scale estimate (niko.scale) to the truth? Benchmark shots know it:
gt_eval[selected].sim3_scale maps the solve to the ground truth in metres.

usage: python scripts/dev/scale_check.py <run_dir> [shot ...]
The synthetic scenes are not built at a realistic size (both depth models agree they look ~2x
smaller, drone shots ~20x): "error" against them says little; "apart" (UniDepth vs DA3) does.
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.scale import estimate_metric_scale


def tracks_of(d: Path):
    for f in (d / "sift" / "sift_tracks.npz", d / "tracks" / "tracks.npz"):
        if f.exists():
            z = np.load(f)
            return z["xy"], z["vis"], f.parent.name
    return None, None, None


def main():
    run = Path(sys.argv[1])
    shots = sys.argv[2:] or sorted(p.parent.name for p in run.glob("*/solve.json"))
    errs, apart = [], []
    print(f"{'shot':26s}{'pick':24s}{'true m/unit':>12}{'estimate':>10}{'error':>9}{'UniDepth':>10}{'DA3':>10}{'apart':>8}")
    for shot in shots:
        d = run / shot
        rep = json.loads((d / "solve.json").read_text())
        pick = rep.get("selected")
        truth = ((rep.get("gt_eval") or {}).get(pick) or {}).get("sim3_scale")
        trk = CameraTrack.load(d / "selected" / "cameras.json")
        xy, vis, src = tracks_of(d)
        est = estimate_metric_scale(trk, xy, vis, d) if xy is not None else None
        if est is None or not truth:
            print(f"{shot[:26]:26s}{str(pick)[:24]:24s}{'-' if not truth else f'{truth:.4g}':>12}"
                  f"{'none':>10}   (tripod, no MegaSaM or no tracks)")
            continue
        e = 100 * (est["metres_per_unit"] / truth - 1)
        errs.append(e)
        m = est["models"]
        u = (m.get("unidepth_v2") or {}).get("metres_per_unit")
        a = (m.get("da3_nested") or {}).get("metres_per_unit")
        print(f"{shot[:26]:26s}{pick[:24]:24s}{truth:>12.4g}{est['metres_per_unit']:>10.4g}{e:>+8.1f}%"
              f"{u if u is None else round(u, 4):>10}{a if a is None else round(a, 4):>10}"
              f"{'' if est['agree_pct'] is None else str(est['agree_pct']) + '%':>8}", flush=True)
        apart.append(est["agree_pct"])
    if errs:
        a = np.abs(errs)
        print(f"\n{len(errs)} shots: |error| median {np.median(a):.1f} %, mean {a.mean():.1f} %, max {a.max():.1f} %; "
              f"within 10 %: {(a <= 10).sum()}, within 20 %: {(a <= 20).sum()}")
        ap = np.array([x for x in apart if x is not None])
        if ap.size:
            print(f"UniDepth vs DA3 apart: median {np.median(ap):.1f} %, max {ap.max():.1f} %")


if __name__ == "__main__":
    main()
