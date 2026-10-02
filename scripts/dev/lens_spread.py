"""How far apart are the lenses of candidates the footage cannot tell apart, and does that spread
predict the picked lens's real (GT) error? Uses the pipeline's own rule (niko.pipeline.solve.lens_spread)
on the stored auto-select scores; nothing is re-solved.

usage: python scripts/dev/lens_spread.py <solve.json | run_dir> ...
"""

import json
import sys
from pathlib import Path

from niko.camio import CameraTrack
from niko.pipeline.solve import LENS_SPREAD_WARN, lens_spread


def main():
    paths = []
    for a in map(Path, sys.argv[1:]):
        paths += sorted(a.glob("*/solve.json")) if a.is_dir() else [a]
    print(f"{'shot':34s}{'n':>3}{'lo px':>8}{'hi px':>8}{'spread':>9}{'GT err of pick':>16}  pick")
    for p in paths:
        rep = json.loads(p.read_text())
        scores = (rep.get("select") or {}).get("scores") or {}
        pick = rep.get("selected")
        if not pick or pick not in scores:
            continue
        cands = {n: CameraTrack.load(p.parent / "candidates" / n / "cameras.json") for n in scores
                 if (p.parent / "candidates" / n / "cameras.json").exists()}
        sp = lens_spread(cands, scores, pick)
        g = ((rep.get("gt_eval") or {}).get(pick) or {}).get("focal_err_pct_max")
        ge = "" if g is None else f"{g:.2f} %"
        if sp is None:
            print(f"{p.parent.name[:34]:34s}{'tripod pick, not checked':>36}{ge:>16}  {pick}")
            continue
        lo, hi = sp["focal_px"]
        warn = " !" if sp["spread_pct"] > 100 * LENS_SPREAD_WARN else "  "
        print(f"{p.parent.name[:34]:34s}{len(sp['equally_good']):>3}{lo:>8.0f}{hi:>8.0f}{sp['spread_pct']:>7.1f}%"
              f"{warn}{ge:>16}  {pick}")


if __name__ == "__main__":
    main()
