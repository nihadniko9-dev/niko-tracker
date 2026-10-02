"""Bring finished solves up to the current export: selected/ (cameras.json, points.ply,
import_blender.py, blender.json), errors.json, the After Effects .jsx, and solve.json's
solve_error.worst_frame. For runs that finished before these outputs existed; nothing is re-solved.

usage: python scripts/dev/backfill_exports.py <run_dir> [shot ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.export_ae import export_after_effects
from niko.pipeline.export import export_solve
from niko.pipeline.select import frame_errors
from niko.pipeline.solve import lens_check
from niko.refine import RefineOptions
from niko.tracks import split_sift


def main():
    run = Path(sys.argv[1])
    shots = sys.argv[2:] or sorted(p.name for p in run.iterdir() if (p / "solve.json").exists())
    for name in shots:
        d = run / name
        rep = json.loads((d / "solve.json").read_text())
        best = rep.get("selected")
        if not best:
            print(f"{name}: no selection, skipped")
            continue
        shot = json.loads((d / "shot.json").read_text())
        trk = CameraTrack.load(d / "candidates" / best / "cameras.json")
        export_solve(trk, d / "candidates" / best / "points.ply", d / "selected", d / "frames",
                     f"000000.{shot['frame_format']}")
        held = None
        if (d / "sift" / "sift_tracks.npz").exists():
            _, held = split_sift(dict(np.load(d / "sift" / "sift_tracks.npz")), RefineOptions().max_sift)
        fe = frame_errors(trk, dict(np.load(d / "tracks" / "tracks.npz")), held)
        (d / "selected" / "errors.json").write_text(json.dumps(fe), encoding="utf-8")
        cands = {best: trk}
        for mth in ("megasam", "da3"):
            if (d / "candidates" / mth / "cameras.json").exists():
                cands[mth] = CameraTrack.load(d / "candidates" / mth / "cameras.json")
        if not rep.get("known_lens"):
            rep["lens_check"] = lens_check(cands, best)
        m = [(v, i) for i, v in enumerate(fe["mean_px"]) if v is not None]
        if m and rep.get("solve_error") is not None:
            v, i = max(m)
            rep["solve_error"]["worst_frame"] = {"frame": trk.frame_start + i, "mean_px": v}
        (d / "solve.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        export_after_effects(d, log=lambda *_: None)
        print(f"{name}: {best}, average {(rep.get('solve_error') or {}).get('average_px')}, "
              f"worst frame {(rep.get('solve_error') or {}).get('worst_frame')}, lens {rep.get('lens_check')}")


if __name__ == "__main__":
    main()
