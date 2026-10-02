"""Give finished solves the real size and levelled world a new solve would export: solve.json ->
metric_scale (the drone's GPS / altitude when the video has telemetry, else the depth models) and
true_up (the drone's gimbal), selected/blender.json -> world, units, median_depth, and
import_blender.py to match; the video's telemetry is kept as telemetry.npz. The camera keys are not
touched; only the levelled world they hang under changes.

usage: python scripts/dev/backfill_scale.py <solve_dir> [--also <flat copy folder> ...] ...
  --also: a flat copy of selected/ (e.g. reports/test_shots/03) to patch the same way.
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camera import identify
from niko.camio import CameraTrack
from niko.pipeline.export import BPY_TEMPLATE, blender_world, median_depth, units_of
from niko.pipeline.solve import _metric_scale
from niko.plyio import read_ply_xyz
from niko.pipeline.ingest import probe_video
from niko.telemetry import TELEMETRY_FILE, for_solve, save, summary, true_up


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
        tel = for_solve(d)
        if tel is not None and not (d / TELEMETRY_FILE).exists():
            save(d / TELEMETRY_FILE, tel)
        shot = json.loads((d / "shot.json").read_text())
        if "camera" not in shot:  # solves from before 0.5: which camera made the clip
            tags = {}
            if Path(shot["source"]).is_file():
                tags = probe_video(Path(shot["source"])).get("camera_tags", {})
            shot["telemetry"] = summary(tel) if tel else None
            shot["camera"] = identify(tags, shot["telemetry"], shot["width"], shot["height"])
            (d / "shot.json").write_text(json.dumps(shot, indent=1), encoding="utf-8")
        metric = _metric_scale(trk, d, dict(np.load(d / "tracks" / "tracks.npz")))
        up = true_up(trk, tel)
        ply = sel / "points.ply"
        X = read_ply_xyz(ply) if ply.exists() else None
        mpu, units = units_of(metric)
        world, how = blender_world(trk, ply if ply.exists() else None, metres_per_unit=mpu, up=up)
        for folder in [sel, *copies]:
            b = json.loads((folder / "blender.json").read_text())
            b.update({"world": world, "world_how": how, "units": units, "median_depth": median_depth(trk, X)})
            (folder / "blender.json").write_text(json.dumps(b))
            if (folder / "import_blender.py").exists():
                (folder / "import_blender.py").write_text(BPY_TEMPLATE.replace("__DATA__", json.dumps(b)), encoding="utf-8")
        for sj in {d / "solve.json", *[c / "solve.json" for c in copies if (c / "solve.json").exists()]}:
            r = json.loads(sj.read_text())
            r["camera"] = shot.get("camera")
            if metric:
                r["metric_scale"] = metric
            if up is not None:
                r["true_up"] = {"source": "gimbal", "up": [float(x) for x in up]}
            else:
                r.pop("true_up", None)
            sj.write_text(json.dumps(r, indent=1, default=str))
        src = (metric or {}).get("source")
        if mpu:
            verdict = f"{mpu:.4g} m/unit from {src}" + (f" +-{metric.get('uncertainty_pct')} %" if src != "depth_models" else
                                                         f" (models {metric.get('agree_pct')} % apart)")
        else:
            verdict = "size unknown"
        print(f"{d.name}: camera {(shot.get('camera') or {}).get('name')}; {verdict}; level from {'the gimbal' if up is not None else how}; "
              f"patched {1 + len(copies)} folder(s)", flush=True)


if __name__ == "__main__":
    main()
