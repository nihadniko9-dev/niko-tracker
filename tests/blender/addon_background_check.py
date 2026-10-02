"""Background check of the Blender add-on in the user's Windows Blender (no UI).

blender.exe -b --factory-startup --python tests/blender/addon_background_check.py -- <solve folder>

Registers the add-on from the repo, builds a solve's scene with the add-on's own importer, and
checks that Blender's camera sees the solve's 3D points exactly where the engine's cameras.json
puts them (levelling empty, parenting, lens/shift keys included). Also checks every icon name the
panels use exists in this Blender.
"""

import json
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io, ui  # noqa: E402

folder = os.path.normpath(sys.argv[sys.argv.index("--") + 1])
niko_tracker.register()
scene = bpy.context.scene
scene.niko.solve_dir = folder
s = solve_io.get(folder)
assert s is not None, "solve not readable"
ops.build_scene(bpy.context, s)

cam = bpy.data.objects["Niko camera"]
world = bpy.data.objects["Niko world"]
with open(os.path.join(solve_io.selected_dir(folder), "cameras.json"), encoding="utf-8") as fh:
    cams = json.load(fh)
W, H = cams["clip"]["width"], cams["clip"]["height"]
P = s.points[np.random.default_rng(0).choice(len(s.points), min(60, len(s.points)), replace=False)] \
    if len(s.points) else np.zeros((0, 3))
if not len(P):  # a tripod solve has no 3D points: test points along the view rays of some frames
    rays = []
    for fr in cams["frames"][:: max(1, len(cams["frames"]) // 12)]:
        if fr["valid"]:
            K, R, t = np.array(fr["K"]), np.array(fr["R"]), np.array(fr["t"])
            uv1 = np.array([[u, v, 1.0] for u in (0.2 * W, 0.5 * W, 0.8 * W) for v in (0.2 * H, 0.8 * H)])
            d = (np.linalg.inv(K) @ uv1.T).T @ R  # camera rays in the world
            rays.append(-R.T @ t + 10.0 * d / np.linalg.norm(d, axis=1, keepdims=True))
    P = np.concatenate(rays) if rays else P
worst = 0.0
checked = 0
for fr in cams["frames"][:: max(1, len(cams["frames"]) // 12)]:
    if not fr["valid"] or not len(P):
        continue
    scene.frame_set(cams["clip"]["frame_start"] + fr["i"])
    K, R, t = np.array(fr["K"]), np.array(fr["R"]), np.array(fr["t"])
    Xc = P @ R.T + t
    front = Xc[:, 2] > 0.1 * np.median(np.abs(Xc[:, 2]))
    uv_cv = np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], 1)
    # within a frame's size of the picture: far off to the side (20 000+ px on a 90 deg whip) float32
    # in Blender's matrices reaches 0.1 px, which says nothing about the picture
    front &= (np.abs(uv_cv[:, 0] - W / 2) < 1.5 * W) & (np.abs(uv_cv[:, 1] - H / 2) < 1.5 * H)
    for p, ok, ref in zip(P, front, uv_cv):
        if not ok:
            continue
        q = world.matrix_world @ Vector(p)
        c = world_to_camera_view(scene, cam, q)
        uv = np.array([c.x * W, (1 - c.y) * H])
        worst = max(worst, float(np.linalg.norm(uv - ref)))
        checked += 1
print(f"NIKO_CHECK projections {checked}, worst |blender - engine| = {worst:.6f} px")
assert checked > 0 and worst < 0.01 * max(1.0, W / 1920), worst  # Blender float32; HD pixels

icons = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
import re  # noqa: E402
src = "".join(open(os.path.join(REPO, "addon", "niko_tracker", f), encoding="utf-8").read()
              for f in os.listdir(os.path.join(REPO, "addon", "niko_tracker")) if f.endswith(".py"))
# every icon the add-on names: icon="..." and the (key, "ICON") pairs of the mask buttons
used = (set(re.findall(r'icon="([A-Z0-9_]+)"', src)) | set(re.findall(r'\("[a-z]+", "([A-Z0-9_]+)"\)', src))
        | set(ui.STATE_ICON.values()))
missing = sorted(used - icons)
print("NIKO_CHECK missing icons:", missing)
assert not missing, missing
pts = bpy.data.objects.get("Niko points")
print("NIKO_CHECK points object:", pts is not None and len(pts.data.vertices), "modifier:",
      pts is not None and pts.modifiers[0].node_group.name)
print("NIKO_CHECK average", s.average_px, "worst frame", s.worst_frame(), "camera", s.camera_kind)
print("NIKO_CHECK OK")
