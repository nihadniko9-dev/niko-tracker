"""Recompute the checks a solve reports, with the current rules, from what the solve stored
(scores, candidates, SIFT tracks), and write them back: lens_check (niko.pipeline.solve.lens_check:
depth models + equally good fits) and track_gaps (niko.tracks.track_gaps).

usage: python scripts/dev/backfill_checks.py <solve_dir> [--also <copy/solve.json> ...] ...
  --also: flat copies of that solve (e.g. reports/test_shots/03/solve.json) to patch the same way.
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.pipeline.select import MIN_SIFT_OBS
from niko.pipeline.solve import lens_check
from niko.tracks import track_gaps


def main():
    args = sys.argv[1:]
    jobs, i = [], 0
    while i < len(args):
        if args[i] == "--also":
            jobs[-1][1].append(Path(args[i + 1]))
            i += 2
        else:
            jobs.append((Path(args[i]), []))
            i += 1
    for solve_dir, copies in jobs:
        rep = json.loads((solve_dir / "solve.json").read_text())
        best, scores = rep.get("selected"), (rep.get("select") or {}).get("scores") or {}
        patch = {}
        if best and not rep.get("known_lens"):
            cands = {n: CameraTrack.load(solve_dir / "candidates" / n / "cameras.json") for n in scores
                     if (solve_dir / "candidates" / n / "cameras.json").exists()}
            lc = lens_check(cands, best, scores)
            if lc:
                patch["lens_check"] = lc
        npz = solve_dir / "sift" / "sift_tracks.npz"
        if npz.exists():
            vis = np.load(npz)["vis"]
            if int(vis.sum()) >= MIN_SIFT_OBS:
                fs = int(json.loads((solve_dir / "shot.json").read_text()).get("frame_start", 1))
                patch["track_gaps"] = [{"from_frame": fs + g["last_before"], "to_frame": fs + g["first_after"],
                                        "tracks": g["tracks"]} for g in track_gaps(vis)]
        for path in [solve_dir / "solve.json", *copies]:
            r = json.loads(path.read_text())
            r.update(patch)
            path.write_text(json.dumps(r, indent=1))
        lc = patch.get("lens_check") or {}
        print(f"{solve_dir.name}: lens uncertain {bool(lc.get('uncertain'))} {lc.get('reasons')}, "
              f"spread {(lc.get('spread') or {}).get('spread_pct')} %; track gaps {patch.get('track_gaps')}; "
              f"patched {1 + len(copies)} file(s)")

if __name__ == "__main__":
    main()
