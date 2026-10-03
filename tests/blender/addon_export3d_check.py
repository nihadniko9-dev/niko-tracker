"""Background check of "Export camera" (FBX, Alembic, USD): each file, imported into an empty scene,
must give the same camera position, orientation and lens on several frames as the Niko scene.

blender.exe -b --factory-startup --python tests/blender/addon_export3d_check.py -- <flat solve folder> <scratch>
"""

import os
import shutil
import sys

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io  # noqa: E402

src, work = (os.path.normpath(a) for a in sys.argv[sys.argv.index("--") + 1:][:2])
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(src, work)
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
niko_tracker.register()
ctx = bpy.context
sc = ctx.scene
sc.niko.solve_dir = work
ops.build_scene(ctx, solve_io.get(work))
cam = bpy.data.objects["Niko camera"]
f0, f1 = sc.frame_start, sc.frame_end
frames = [f0, (f0 + f1) // 2, f1]
ref = {}
for f in frames:
    sc.frame_set(f)
    ref[f] = (cam.matrix_world.copy(), cam.data.lens)
scale = max(bpy.data.objects["Niko world"].matrix_world.to_scale())
depth = (solve_io.get(work).blender.get("median_depth") or 1.0) * scale

for fmt, ext, imp in (("FBX", "fbx", lambda p: bpy.ops.import_scene.fbx(filepath=p)),
                      ("ABC", "abc", lambda p: bpy.ops.wm.alembic_import(filepath=p)),
                      ("USD", "usdc", lambda p: bpy.ops.wm.usd_import(filepath=p))):
    assert bpy.ops.niko.export_3d(fmt=fmt) == {"FINISHED"}, fmt
    path = os.path.join(work, f"niko_camera.{ext}")
    assert os.path.getsize(path) > 1000, path
    before = set(bpy.data.objects)
    imp(path)
    added = [o for o in bpy.data.objects if o not in before]
    icams = [o for o in added if o.type == "CAMERA"]
    assert icams, f"{fmt}: no camera imported ({[o.name for o in added]})"
    ic = icams[0]
    ad = ic.animation_data
    act = ad.action if ad else None
    par = ic.parent.name if ic.parent else None
    pad = ic.parent.animation_data.action if ic.parent and ic.parent.animation_data else None
    print(f"   {fmt} imported: {[o.name for o in added]}, camera parent {par}, action {act and act.name}, "
          f"frame range {act and tuple(act.frame_range)}, parent action {pad and pad.name}")
    worst_pos, worst_rot, worst_lens = 0.0, 0.0, 0.0
    for f in frames:
        if fmt == "FBX":
            # the file keys frame f at f / fps seconds (checked in the binary: 59.94 fps, frame 1700 at
            # 28.3617 s); Blender's own FBX importer reads that back as t * round(fps) + 1, so the
            # imported camera is compared at that (sub)frame
            g = f / sc.render.fps * sc.render.fps_base * sc.render.fps + 1
            sc.frame_set(int(g), subframe=g - int(g))
        else:
            sc.frame_set(f)
        bpy.context.view_layer.update()
        m = ic.matrix_world
        r = ref[f][0]
        worst_pos = max(worst_pos, (m.translation - r.translation).length / max(depth, 1e-9))
        worst_rot = max(worst_rot, m.to_quaternion().rotation_difference(r.to_quaternion()).angle)
        worst_lens = max(worst_lens, abs(ic.data.lens - ref[f][1]))
        print(f"   {fmt} frame {f}: pos {(m.translation - r.translation).length:.5f} angle "
              f"{m.to_quaternion().rotation_difference(r.to_quaternion()).angle * 57.2958:.4f} deg")
    print(f"NIKO_EXPORT {fmt}: {os.path.getsize(path)} bytes, camera position off {100 * worst_pos:.4f} % of depth, "
          f"angle off {worst_rot * 57.2958:.4f} deg, lens off {worst_lens:.4f} mm")
    assert worst_pos < 1e-3 and worst_rot < 1e-3 and worst_lens < 0.01, fmt
    for o in added:
        bpy.data.objects.remove(o, do_unlink=True)
print("NIKO_EXPORT OK")
