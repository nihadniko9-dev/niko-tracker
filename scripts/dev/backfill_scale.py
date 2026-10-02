"""Give finished solves the real-size estimate and the levelled world of 0.4 (niko.scale,
export_ae.world_alignment): solve.json -> metric_scale, selected/blender.json -> world (ground and
up, in metres when known), units, median_depth, and import_blender.py to match. The camera keys are
not touched; only the levelled world they hang under changes, as a new solve would export it.

usage: python scripts/dev/backfill_scale.py <solve_dir> [--also <flat copy folder> ...] ...
  --also: a flat copy of selected/ (e.g. reports/test_shots/03) to patch the same way.
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.pipeline.export import BPY_TEMPLATE, blender_world, median_depth
from niko.plyio import read_ply_xyz
from niko.scale import estimate_metric_scale


def main():
    args, jobs, i = sys.argv[1:], [], 0
    while i < len(args):
        if args[i] == "--also":
            jobs[-1][1].append(Path(args[i + 1]))
            i += 2
        else:
            jobs.append((Path(args[i]), []))
            i += 1
    for d, copies in jobs:
        sel = d / "selected"
        trk = CameraTrack.load(sel / "cameras.json")
        f = d / "sift" / "sift_tracks.npz"
        src = np.load(f) if f.exists() else np.load(d / "tracks" / "tracks.npz")
        est = estimate_metric_scale(trk, src["xy"], src["vis"], d)
        ply = sel / "points.ply"
        X = read_ply_xyz(ply) if ply.exists() else None
        mpu = (est or {}).get("metres_per_unit") if (est or {}).get("reliable") else None
        world, how = blender_world(trk, ply if ply.exists() else None, metres_per_unit=mpu)
        units = ({"kind": "metres_estimated", "metres_per_unit": mpu, "agree_pct": est.get("agree_pct")}
                 if mpu else {"kind": "arbitrary"})
        for folder in [sel, *copies]:
            b = json.loads((folder / "blender.json").read_text())
            b.update({"world": world, "world_how": how, "units": units, "median_depth": median_depth(trk, X)})
            (folder / "blender.json").write_text(json.dumps(b))
            if (folder / "import_blender.py").exists():
                (folder / "import_blender.py").write_text(BPY_TEMPLATE.replace("__DATA__", json.dumps(b)), encoding="utf-8")
            sj = folder / "solve.json" if (folder / "solve.json").exists() else d / "solve.json"
            r = json.loads(sj.read_text())
            if est:
                r["metric_scale"] = est
            sj.write_text(json.dumps(r, indent=1, default=str))
        if (d / "solve.json").exists() and est:
            r = json.loads((d / "solve.json").read_text())
            r["metric_scale"] = est
            (d / "solve.json").write_text(json.dumps(r, indent=1, default=str))
        models = {k: round(v["metres_per_unit"], 4) for k, v in ((est or {}).get("models") or {}).items()}
        verdict = "no estimate (tripod or no depth model)" if not est else (
            f"{mpu:.4g} m/unit, used" if mpu else f"{est['metres_per_unit']:.4g} m/unit, NOT used (models disagree)")
        print(f"{d.name}: {verdict} "
              f"{models} apart {(est or {}).get('agree_pct')} %; patched {1 + len(copies)} folder(s)", flush=True)


if __name__ == "__main__":
    main()
