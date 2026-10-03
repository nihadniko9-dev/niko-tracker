"""Background check that a solve's Blender project matches the video exactly: resolution (upright phone
videos portrait), frame rate (29.97, 59.94, variable-rate clips at their nominal rate...), frame range,
and footage behind the camera with exactly one picture per camera key; and the After Effects script
asks for the same comp.

blender.exe -b --factory-startup --python tests/blender/addon_project_check.py -- <solve folder> [<solve folder> ...]
"""

import json
import os
import re
import sys
from fractions import Fraction

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io  # noqa: E402

niko_tracker.register()
ctx = bpy.context
bad = 0
for folder in (os.path.normpath(a) for a in sys.argv[sys.argv.index("--") + 1:]):
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for clip in list(bpy.data.movieclips):
        bpy.data.movieclips.remove(clip)
    with open(os.path.join(folder, "shot.json"), encoding="utf-8") as fh:
        shot = json.load(fh)
    s = solve_io.get(folder, refresh=True)
    ctx.scene.niko.solve_dir = folder
    ctx.scene.niko.clip = s.source_clip() or ""
    ops.build_scene(ctx, s)
    r = ctx.scene.render
    fps_video = Fraction(shot["fps_fraction"]) if shot.get("fps_fraction") else Fraction(shot["fps"]).limit_denominator(1001)
    fps_scene = r.fps / r.fps_base
    n = shot["n_frames"]
    problems = []
    if (r.resolution_x, r.resolution_y, r.resolution_percentage) != (shot["width"], shot["height"], 100):
        problems.append(f"resolution {r.resolution_x}x{r.resolution_y} @{r.resolution_percentage}% vs video {shot['width']}x{shot['height']}")
    if abs(fps_scene / float(fps_video) - 1) > 1e-6:
        problems.append(f"fps {fps_scene:.6f} vs video {float(fps_video):.6f}")
    if ctx.scene.frame_end - ctx.scene.frame_start + 1 != n:
        problems.append(f"frame range {ctx.scene.frame_end - ctx.scene.frame_start + 1} vs {n} frames")
    cam = bpy.data.objects["Niko camera"]
    bg = cam.data.background_images[0]
    if bg.source == "MOVIE_CLIP":
        clip = bg.clip
        what = f"movie clip {os.path.basename(clip.filepath)} ({clip.frame_duration} frames, {clip.size[0]}x{clip.size[1]})"
        if clip.frame_duration != n:
            problems.append(f"Blender reads the video as {clip.frame_duration} frames, the solve has {n}")
        if tuple(clip.size) != (shot["width"], shot["height"]):
            problems.append(f"Blender reads the video as {clip.size[0]}x{clip.size[1]}")
    else:
        iu = bg.image_user
        what = f"image sequence ({iu.frame_duration} frames)"
        if iu.frame_duration != n:
            problems.append(f"footage sequence {iu.frame_duration} frames vs {n}")
    jsx = os.path.join(solve_io.selected_dir(folder), "niko_after_effects.jsx")
    if os.path.exists(jsx):
        txt = open(jsx, encoding="utf-8").read()
        m = re.search(r"var D = (\{.*?\});\n", txt, re.S)
        if m:
            D = json.loads(m.group(1))
            if (D["width"], D["height"]) != (shot["width"], shot["height"]) or abs(D["fps"] / float(fps_video) - 1) > 1e-6 \
                    or len(D["frames"]) != n:
                problems.append(f"After Effects comp {D['width']}x{D['height']} {D['fps']:.4f} fps {len(D['frames'])} frames")
    flags = [k for k in ("vfr", "interlaced") if shot.get(k)] + (["upright"] if (shot.get("rotation") or 0) % 180 else [])
    print(f"NIKO_PROJECT {os.path.basename(folder)}: {shot['width']}x{shot['height']} {float(fps_video):.4f} fps "
          f"{n} frames {flags}; footage {what}: {'OK' if not problems else 'PROBLEM ' + '; '.join(problems)}")
    bad += bool(problems)
print("NIKO_PROJECT", "OK" if not bad else f"{bad} with problems")
