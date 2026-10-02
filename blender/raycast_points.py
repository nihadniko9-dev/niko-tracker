"""Run inside Blender with a benchmark scene.blend open: exact 3D points behind pixels.

    blender -b scene.blend --python raycast_points.py -- rays.npz out.npz

rays.npz: frame [N] int (Blender frame number), origin [N,3], direction [N,3] (world, unit).
out.npz:  hit [N] bool, point [N,3], dynamic [N] bool (hit a moving person/car proxy).
Rays are cast with the scene evaluated at each ray's frame, so moving objects are where they
were on that frame.
"""

import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nikobpy  # noqa: E402

src, dst = nikobpy.script_args()[:2]
d = np.load(src)
frames, origins, dirs = d["frame"], d["origin"], d["direction"]
scene = bpy.context.scene
hit = np.zeros(len(frames), bool)
point = np.full((len(frames), 3), np.nan)
dynamic = np.zeros(len(frames), bool)
for f in np.unique(frames):
    scene.frame_set(int(f))
    dg = bpy.context.evaluated_depsgraph_get()
    for i in np.nonzero(frames == f)[0]:
        ok, loc, _n, _idx, obj, _m = scene.ray_cast(dg, Vector(origins[i]), Vector(dirs[i]), distance=1e5)
        if ok:
            hit[i] = True
            point[i] = tuple(loc)
            name = obj.name if obj is not None else ""
            root = obj.parent.name if obj is not None and obj.parent is not None else name
            dynamic[i] = root.startswith(("person_", "car_"))
np.savez(dst, hit=hit, point=point, dynamic=dynamic)
print(f"NIKO_RAYCAST_DONE {int(hit.sum())}/{len(hit)} hits, {int(dynamic.sum())} on movers")
