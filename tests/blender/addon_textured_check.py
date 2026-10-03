"""Background check of "Add textured mesh": the OBJ lands where the coloured mesh is (same world, no axis
swap), with UVs and the albedo texture in its material.

blender.exe -b --factory-startup --python tests/blender/addon_textured_check.py -- <flat solve folder with mesh.ply and mesh_textured.*>
"""

import os
import sys

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io  # noqa: E402

folder = os.path.normpath(sys.argv[sys.argv.index("--") + 1])
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
niko_tracker.register()
ctx = bpy.context
ctx.scene.niko.solve_dir = folder
ops.build_scene(ctx, solve_io.get(folder))
assert bpy.ops.niko.textured_mesh() == {"FINISHED"}
tex = bpy.data.objects[ops.TEX_NAME]
ref = ops.import_scene_mesh(ctx, os.path.join(folder, "mesh.ply"))
ctx.view_layer.update()


def world_verts(ob, n=20000):
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)[:: max(1, len(ob.data.vertices) // n)]
    M = np.array(ob.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


a, b = world_verts(tex), world_verts(ref)
size = np.linalg.norm(b.max(0) - b.min(0))
d = np.array([np.min(np.linalg.norm(b - p, axis=1)) for p in a[:: max(1, len(a) // 1500)]])
img = [n.image for m in tex.data.materials if m and m.use_nodes for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
print(f"NIKO_TEX {len(tex.data.polygons):,} faces, UV layers {len(tex.data.uv_layers)}, texture "
      f"{img[0].size[:] if img else None}; distance to the coloured mesh: median {100 * np.median(d) / size:.3f} %, "
      f"90% {100 * np.percentile(d, 90) / size:.3f} % of the scene size")
assert len(tex.data.uv_layers) >= 1 and img and img[0].size[0] >= 1024
assert np.median(d) / size < 0.005
print("NIKO_TEX OK")
