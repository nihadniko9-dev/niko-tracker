"""`niko` command line."""

from __future__ import annotations

import argparse
import sys


def _csv(s):
    return s.split(",") if s else None


def _bench_generate(args) -> int:
    from .bench.sets import contact_sheet, generate_set, load_set, set_dir

    generate_set(args.set, only=_csv(args.shots), preview=args.preview, force=args.force)
    root = set_dir(load_set(args.set)["name"], args.preview)
    print("contact sheet:", contact_sheet(root, root / "contact_sheet.jpg"))
    return 0


def _bench_run(args) -> int:
    from .bench.run import run_set
    from .pipeline.solve import DEFAULT_METHODS

    out = run_set(args.set, args.name, _csv(args.methods) or DEFAULT_METHODS, only=_csv(args.shots),
                  force=args.force, prompts=_csv(args.prompts), reuse=args.reuse)
    print("run folder:", out)
    return 0


def _bench_report(args) -> int:
    from pathlib import Path

    from .bench.report import make_report
    from .paths import niko_home

    import shutil

    from .paths import REPO

    run_dir = Path(args.run) if Path(args.run).is_dir() else niko_home() / "runs" / args.run
    r = make_report(run_dir)
    for s in r["summary"]:
        print(f"{s['method']:<20} solved {s['solved']}/{s['shots']}, targets met {s['targets_met']}/{s['shots']}, "
              f"median ATE {s['ate_pct']}, median rot max {s['rot_err_deg_max']}")
    # a copy next to the source (D:\Niko Tracker\reports\<run>) so it opens from Windows directly
    dst = REPO / "reports" / run_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in (r["html"], r["csv"]):
        shutil.copyfile(f, dst / Path(f).name)
    print("report:", r["html"], "\ncsv:   ", r["csv"], "\ncopy:  ", dst)
    return 0


def _bench_import(args) -> int:
    from .bench.import_camera import import_camera
    from .camio import CameraTrack
    from .evaluate import evaluate_against_gt, summary_line

    frames = tuple(int(x) for x in args.frames.split(":")) if args.frames else None
    trk = import_camera(args.blend, args.out, camera=args.camera, frames=frames)
    print(f"{trk.n_frames} frames, {trk.width}x{trk.height} @ {trk.fps:g} fps -> {args.out}")
    if args.gt:
        print(summary_line(trk.method, evaluate_against_gt(trk, CameraTrack.load(args.gt))))
    return 0


def _doctor(args) -> int:
    from .doctor import run_doctor

    return run_doctor(as_json=args.json)


def _prompts(text):
    """--prompts: None = the default list; "none" (or an empty value) = mask nothing, no masks step."""
    if text is None:
        return None
    words = [w.strip() for w in text.split(",") if w.strip()]
    return [] if not words or [w.lower() for w in words] == ["none"] else words


def _solve(args) -> int:
    from pathlib import Path

    from .pipeline.solve import DEFAULT_METHODS, solve

    clip = Path(args.clip)
    out = Path(args.out) if args.out else clip.with_suffix("").parent / f"{clip.stem}_niko"
    methods = args.methods.split(",") if args.methods else DEFAULT_METHODS
    prompts = _prompts(args.prompts)
    try:
        report = solve(clip, out, methods=methods, prompts=prompts, fps=args.fps, frame_start=args.frame_start,
                       reuse=args.reuse, focal_mm=args.focal_mm, sensor_mm=args.sensor_mm, stride=args.stride)
    except (ValueError, OSError) as exc:
        print(f"Cannot start solve: {exc}", file=sys.stderr)
        return 1
    print(f"{'OK' if report['ok'] else 'INCOMPLETE'}: selected {report.get('selected')} "
          f"in {report['total_seconds']}s -> {out / 'solve.json'}")
    return 0 if report["ok"] else 1


def _locktest(args) -> int:
    from .locktest import lock_test

    r = lock_test(args.solve_dir, args.out, scale=args.scale, n_points=args.points)
    print("lock test:", r["mp4"])
    return 0


def _export_ae(args) -> int:
    from .export_ae import export_after_effects

    r = export_after_effects(args.solve_dir, args.out, n_nulls=args.nulls)
    print("After Effects script:", r["jsx"])
    return 0


def _mesh(args) -> int:
    from .mesh import build_mesh

    r = build_mesh(args.solve_dir, quality=args.quality, max_images=args.images, max_image_size=args.size,
                   reuse_stereo=args.reuse)
    print("mesh:", r["mesh"])
    return 0


def _report(args) -> int:
    from .clip_report import clip_report

    out = clip_report(args.solve_dirs, args.out, title=args.title, subtitle=args.subtitle)
    print("report:", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="niko", description="Niko Tracker Engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check GPU, CUDA, backend envs, checkpoints, Blender")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_doctor)

    p = sub.add_parser("solve", help="solve the camera of a clip (video, image folder or bench shot)")
    p.add_argument("clip")
    p.add_argument("-o", "--out", help="output folder (default: <clip>_niko next to the clip)")
    p.add_argument("--methods", help="comma list: colmap_global,colmap_incremental,megasam,da3")
    p.add_argument("--prompts", help='comma list of SAM 3 prompts (default: person,car,animal,sky,water); '
                                     '"none": ignore nothing (no masks)')
    p.add_argument("--fps", type=float, help="fps for image folders")
    p.add_argument("--frame-start", type=int, default=1)
    p.add_argument("--focal-mm", type=float, help="known lens focal length in mm (kept fixed in the solve)")
    p.add_argument("--sensor-mm", type=float, default=36.0,
                   help="sensor width in mm for --focal-mm (default 36: full frame / 35 mm equivalent)")
    p.add_argument("--reuse", action="store_true", help="keep finished stages found in the output folder")
    p.add_argument("--stride", type=int, help="keyframe step for COLMAP / MegaSaM (default: automatic)")
    p.set_defaults(func=_solve)

    p = sub.add_parser("locktest", help="MP4 of the footage with the solve's 3D points and a ground grid")
    p.add_argument("solve_dir", help="output folder of niko solve (or a bench run's shot folder)")
    p.add_argument("-o", "--out", help="MP4 path (default: <solve_dir>/selected/locktest.mp4)")
    p.add_argument("--scale", type=float, default=1.0, help="output size relative to the clip")
    p.add_argument("--points", type=int, default=1500, help="held-out tracks to draw")
    p.set_defaults(func=_locktest)

    p = sub.add_parser("export-ae", help="After Effects .jsx: comp, footage, 3D camera, track nulls")
    p.add_argument("solve_dir", help="output folder of niko solve (or a bench run's shot folder)")
    p.add_argument("-o", "--out", help="jsx path (default: <solve_dir>/selected/niko_after_effects.jsx)")
    p.add_argument("--nulls", type=int, default=24, help="number of 3D track nulls")
    p.set_defaults(func=_export_ae)

    p = sub.add_parser("mesh", help="editable 3D mesh of the static scene (multi-view stereo on the solve)")
    p.add_argument("solve_dir", help="output folder of niko solve")
    p.add_argument("--quality", choices=("fast", "good", "high"), default="good",
                   help="fast: 40 frames at 1280 px; good: 60 at 1600; high: 90 at 2400 (slower)")
    p.add_argument("--images", type=int, help="frames used for stereo (overrides the quality)")
    p.add_argument("--size", type=int, help="max image size for stereo (overrides the quality)")
    p.add_argument("--reuse", action="store_true", help="keep the earlier stereo points, redo the surface")
    p.set_defaults(func=_mesh)

    p = sub.add_parser("report", help="one HTML page about solved clips (embedded images, reads on a phone)")
    p.add_argument("solve_dirs", nargs="+", help="output folders of niko solve")
    p.add_argument("-o", "--out", required=True, help="HTML file to write")
    p.add_argument("--title", default="your clips")
    p.add_argument("--subtitle", default="")
    p.set_defaults(func=_report)

    bench = sub.add_parser("bench", help="benchmark").add_subparsers(dest="bench_cmd", required=True)
    p = bench.add_parser("generate", help="render a synthetic shot set with ground truth (resumable)")
    p.add_argument("--set", default="synthetic_v1", help="configs/benchmark/<set>.json")
    p.add_argument("--shots", help="comma list of shot names (default: all)")
    p.add_argument("--preview", action="store_true", help="1/4 size, 16 frames: quick composition check")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_bench_generate)

    p = bench.add_parser("run", help="solve every shot of a set (resumable)")
    p.add_argument("--set", default="synthetic_v1", help="set name or a folder of shots")
    p.add_argument("--name", required=True, help="run name: results go to $NIKO_HOME/runs/<name>")
    p.add_argument("--methods", help="comma list (default: all methods)")
    p.add_argument("--shots", help="comma list of shot names (default: all)")
    p.add_argument("--prompts", help="SAM 3 prompts (default: the set's own list, else the solve default)")
    p.add_argument("--force", action="store_true", help="re-solve shots that already have a solve.json")
    p.add_argument("--reuse", action="store_true",
                   help="keep finished stages found in the shot folder (ingest, masks, tracks, candidates)")
    p.set_defaults(func=_bench_run)

    p = bench.add_parser("report", help="HTML + CSV report of a run")
    p.add_argument("run", help="run name or folder")
    p.set_defaults(func=_bench_report)

    p = bench.add_parser("import-camera", help="camera of any .blend -> cameras.json")
    p.add_argument("blend")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--camera", help="camera object name (default: the scene camera)")
    p.add_argument("--frames", help="first:last (default: the scene range)")
    p.add_argument("--gt", help="ground-truth cameras.json to score against")
    p.set_defaults(func=_bench_import)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
