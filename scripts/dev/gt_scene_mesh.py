"""Blender: the ground-truth geometry of a benchmark scene as one PLY (world coordinates, static
objects only), for checking scene meshes against it (scripts/dev/mesh_check.py).

blender.exe -b <bench shot>/scene.blend --python scripts/dev/gt_scene_mesh.py -- <out.ply>
Objects with animation (the moving people and cars) are left out: the scene mesh leaves them out too.
"""

import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
scene.frame_set(scene.frame_start)
dg = bpy.context.evaluated_depsgraph_get()
verts, tris, base = [], [], 0


def animated(ob):
    while ob is not None:
        if ob.animation_data and ob.animation_data.action:
            return True
        ob = ob.parent
    return False


for ob in scene.objects:
    if ob.type != "MESH" or animated(ob) or ob.hide_render:
        continue
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world)
    co = co @ M[:3, :3].T + M[:3, 3]
    t = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", t)
    verts.append(co)
    tris.append(t.reshape(-1, 3) + base)
    base += len(co)
    ev.to_mesh_clear()

V, F = np.concatenate(verts), np.concatenate(tris)
with open(out, "wb") as fh:
    fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
    fh.write(V.astype("<f4").tobytes())
    rec = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
    rec["n"], rec["i"] = 3, F
    fh.write(rec.tobytes())
print("NIKO_GT", out, len(V), "vertices", len(F), "triangles")
