"""Background check of "Set ground from points" with "Set real size" and "Reset", on a scratch copy of a solve.

blender.exe -b --factory-startup --python tests/blender/addon_ground_check.py -- <flat solve folder> <scratch folder>
Four empties on a tilted, raised plane (in the solve's levelled world) become the ground: afterwards
they lie on Z = 0, level, around the origin. Then a camera height of 12 m, a rebuild (both kept),
and a reset (back to the engine's world).
"""

import math
import os
import shutil
import sys

import bpy
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io  # noqa: E402

src, work = (os.path.normpath(a) for a in sys.argv[sys.argv.index("--") + 1:][:2])
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(src, work)
for name in ("real_scale.json", "real_scale.json.old"):
    if os.path.exists(os.path.join(work, name)):
        os.remove(os.path.join(work, name))
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
niko_tracker.register()
ctx = bpy.context
ctx.scene.niko.solve_dir = work
ops.build_scene(ctx, solve_io.get(work))
world = bpy.data.objects["Niko world"]
engine_world = world.matrix_world.copy()
cam = bpy.data.objects["Niko camera"]
ctx.scene.frame_set(ctx.scene.frame_start)

# a plane tilted 12 deg about X and 7 deg about Y, 0.3 units below the camera, in scene space
cz = cam.matrix_world.translation
tilt = Matrix.Rotation(math.radians(12), 4, "X") @ Matrix.Rotation(math.radians(7), 4, "Y")
base = Vector((cz.x, cz.y + 2.0, cz.z - 0.3))
scene_pts = [base + (tilt @ Vector((x, y, 0.0))) for x, y in ((-1, -1), (1, -1), (1, 1), (-1.2, 0.8))]
empties = []
inv = world.matrix_world.inverted()
for i, p in enumerate(scene_pts):
    e = bpy.data.objects.new(f"G{i}", None)
    ctx.scene.collection.objects.link(e)
    e.parent = world
    e.location = inv @ p  # in the world empty's space: it moves with the world
    empties.append(e)
bpy.ops.object.select_all(action="DESELECT")
for e in empties:
    e.select_set(True)
ctx.view_layer.objects.active = empties[0]
ctx.view_layer.update()
assert bpy.ops.niko.set_ground() == {"FINISHED"}
ctx.view_layer.update()
z = [e.matrix_world.translation.z for e in empties]
centre = sum((e.matrix_world.translation for e in empties), Vector()) / 4
print("NIKO_GROUND points z after:", [round(v, 6) for v in z], "centre", tuple(round(v, 6) for v in centre))
assert max(abs(v) for v in z) < 1e-4, z
assert centre.length < 1e-4, centre
assert cam.matrix_world.translation.z > 0
saved = solve_io.read_real_scale(work)
assert saved.get("ground", "").startswith("4 points"), saved

# camera height after the ground: 12 m, the floor stays level
h0 = cam.matrix_world.translation.z
assert bpy.ops.niko.set_size(mode="HEIGHT", metres=12.0) == {"FINISHED"}
ctx.view_layer.update()
z = [e.matrix_world.translation.z for e in empties]
print("NIKO_GROUND height", round(h0, 5), "-> camera z", round(cam.matrix_world.translation.z, 5), "floor z", [round(v, 6) for v in z])
assert abs(cam.matrix_world.translation.z - 12.0) < 1e-3
assert max(abs(v) for v in z) < 1e-3
local = [e.location.copy() for e in empties]

# rebuild: the same world (size and ground) comes back from real_scale.json
ops.build_scene(ctx, solve_io.get(work, refresh=True))
world2 = bpy.data.objects["Niko world"]
cam = bpy.data.objects["Niko camera"]
ctx.scene.frame_set(ctx.scene.frame_start)
z2 = [(world2.matrix_world @ p).z for p in local]
print("NIKO_GROUND after rebuild: camera z", round(cam.matrix_world.translation.z, 5), "floor z", [round(v, 6) for v in z2])
assert abs(cam.matrix_world.translation.z - 12.0) < 1e-3
assert max(abs(v) for v in z2) < 1e-3

# reset: the engine's world again, the setting kept as .old
assert bpy.ops.niko.reset_adjust() == {"FINISHED"}
diff = max(abs(a - b) for ra, rb in zip(world2.matrix_world, engine_world) for a, b in zip(ra, rb))
print("NIKO_GROUND reset: max difference from the engine world", diff)
assert diff < 1e-6
assert solve_io.read_real_scale(work) == {}
assert os.path.exists(os.path.join(work, "real_scale.json.old"))

# an old file (0.4.0: factor only) still loads
solve_io.write_real_scale(work, factor=2.0, how="old file")
os.replace(os.path.join(work, "real_scale.json"), os.path.join(work, "tmp.json"))
import json  # noqa: E402
with open(os.path.join(work, "tmp.json"), encoding="utf-8") as fh:
    old = {k: v for k, v in json.load(fh).items() if k in ("factor", "how", "set")}
with open(os.path.join(work, "real_scale.json"), "w", encoding="utf-8") as fh:
    json.dump(old, fh)
ops.build_scene(ctx, solve_io.get(work, refresh=True))
w3 = bpy.data.objects["Niko world"].matrix_world
expect = Matrix.Scale(2.0, 4) @ engine_world
diff = max(abs(a - b) for ra, rb in zip(w3, expect) for a, b in zip(ra, rb))
print("NIKO_GROUND 0.4.0 file: max difference", diff)
assert diff < 1e-6
print("NIKO_GROUND OK")
