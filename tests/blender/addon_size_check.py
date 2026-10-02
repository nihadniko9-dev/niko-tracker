"""Background check of "Set real size" (camera height and two points), on a scratch copy of a solve.

blender.exe -b --factory-startup --python tests/blender/addon_size_check.py -- <flat solve folder> <scratch folder>
"""

import os
import shutil
import sys

import bpy
from mathutils import Vector

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
ctx.scene.niko.solve_dir = work
ops.build_scene(ctx, solve_io.get(work))
cam = bpy.data.objects["Niko camera"]
frame = ctx.scene.frame_current
z0 = cam.matrix_world.translation.z
assert z0 > 0, z0

# camera height 50 m
assert bpy.ops.niko.set_size(mode="HEIGHT", metres=50.0) == {"FINISHED"}
z1 = cam.matrix_world.translation.z
print("NIKO_SIZE height", round(z0, 4), "->", round(z1, 4))
assert abs(z1 - 50.0) < 1e-3, z1
assert abs(solve_io.read_real_scale(work)["factor"] - 50.0 / z0) < 1e-6 * 50.0 / z0

# the size survives a rebuild
ops.build_scene(ctx, solve_io.get(work, refresh=True))
cam = bpy.data.objects["Niko camera"]
ctx.scene.frame_set(frame)
z2 = cam.matrix_world.translation.z
print("NIKO_SIZE after rebuild", round(z2, 4))
assert abs(z2 - 50.0) < 1e-3, z2

# two points: two empties 10 m apart afterwards
world = bpy.data.objects["Niko world"]
a, b = (bpy.data.objects.new(n, None) for n in ("A", "B"))
for e, p in ((a, Vector((0.1, 0.2, 0.0))), (b, Vector((0.4, -0.3, 0.05)))):
    ctx.scene.collection.objects.link(e)
    e.parent = world
    e.location = p  # in the world empty's space: they scale with it
bpy.ops.object.select_all(action="DESELECT")
a.select_set(True)
b.select_set(True)
ctx.view_layer.objects.active = a
ctx.view_layer.update()
assert bpy.ops.niko.set_size(mode="POINTS", metres=10.0) == {"FINISHED"}
ctx.view_layer.update()
d = (a.matrix_world.translation - b.matrix_world.translation).length
print("NIKO_SIZE two points", round(d, 5), "m; saved:", solve_io.read_real_scale(work))
assert abs(d - 10.0) < 1e-3, d
print("NIKO_SIZE OK")
