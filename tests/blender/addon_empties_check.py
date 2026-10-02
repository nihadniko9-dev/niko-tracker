"""Background check of 'Empty at selected points'.

blender.exe -b --factory-startup --python tests/blender/addon_empties_check.py -- <solve folder>

Builds the scene, picks three points in Edit Mode, adds one empty per point and one in the middle,
and checks each empty sits exactly on its point in world space (also after moving the levelling empty).
"""

import os
import sys

import bmesh
import bpy
from mathutils import Vector

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io  # noqa: E402

folder = os.path.normpath(sys.argv[sys.argv.index("--") + 1])
niko_tracker.register()
scene = bpy.context.scene
scene.niko.solve_dir = folder
ops.build_scene(bpy.context, solve_io.get(folder))
pts = bpy.data.objects["Niko points"]
assert not pts.modifiers[0].show_in_editmode

bpy.ops.niko.edit_points()
assert pts.mode == "EDIT"
bm = bmesh.from_edit_mesh(pts.data)
bm.verts.ensure_lookup_table()
pick = [10, 500, 2500]
for i in pick:
    bm.verts[i].select = True
bmesh.update_edit_mesh(pts.data)
want = [pts.matrix_world @ bm.verts[i].co for i in pick]

r = bpy.ops.niko.empties(mode="EACH")
made = sorted((o for o in bpy.data.objects if o.name.startswith("Niko point ")), key=lambda o: o.name)
err = max((o.matrix_world.translation - w).length for o, w in zip(made, want))
print("NIKO_EMPTIES each:", r, len(made), f"max offset {err:.2e}")
assert len(made) == 3 and err < 1e-5

bpy.ops.niko.edit_points()
bm = bmesh.from_edit_mesh(pts.data)
bm.verts.ensure_lookup_table()
for i in pick:
    bm.verts[i].select = True
bmesh.update_edit_mesh(pts.data)
bpy.ops.niko.empties(mode="CENTER")
mid = [o for o in bpy.data.objects if o.name == "Niko point 04"][0]
c = sum(want, Vector()) / 3
print("NIKO_EMPTIES middle offset", f"{(mid.matrix_world.translation - c).length:.2e}")
assert (mid.matrix_world.translation - c).length < 1e-5

world = bpy.data.objects["Niko world"]  # the empties follow the levelled scene
world.location.z += 1.0
bpy.context.view_layer.update()
err = max((o.matrix_world.translation - pts.matrix_world @ pts.data.vertices[i].co).length for o, i in zip(made, pick))
print("NIKO_EMPTIES after moving Niko world:", f"{err:.2e}")
assert err < 1e-5
print("NIKO_EMPTIES OK")
