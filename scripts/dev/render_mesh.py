"""Blender (background): render scene meshes of a solve through its own camera, for comparing them.

blender.exe -b --factory-startup --python scripts/dev/render_mesh.py -- <solve folder> <frame> <out.png> <mesh.ply> [<mesh.ply> ...]
Builds the solve's scene with the add-on, then renders each mesh alone (Workbench, vertex colours,
no footage) from the solve camera at <frame>; the images are placed side by side in <out.png>.
"""

import os
import sys

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
folder, frame, out = os.path.normpath(args[0]), int(args[1]), os.path.normpath(args[2])
meshes = [os.path.normpath(m) for m in args[3:]]
for ob in list(bpy.data.objects):  # the factory scene's cube and light
    bpy.data.objects.remove(ob, do_unlink=True)
niko_tracker.register()
scene = bpy.context.scene
scene.niko.solve_dir = folder
ops.build_scene(bpy.context, solve_io.get(folder))
for name in ("Niko points",):
    ob = bpy.data.objects.get(name)
    if ob is not None:
        ob.hide_render = True
cam = bpy.data.objects["Niko camera"]
cam.data.show_background_images = False
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "VERTEX"
scene.render.film_transparent = False
scene.world = scene.world or bpy.data.worlds.new("w")
scale = 960 / scene.render.resolution_x
scene.render.resolution_percentage = int(round(100 * scale))
scene.frame_set(frame)
images = []
for i, path in enumerate(meshes):
    ob = ops.import_scene_mesh(bpy.context, path)
    ob.hide_render = False
    png = f"{out}.{i}.png"
    scene.render.filepath = png
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(png)
    w, h = img.size
    px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
    images.append(px)
    bpy.data.objects.remove(ob, do_unlink=True)
canvas = np.concatenate(images, axis=1)
im = bpy.data.images.new("cmp", canvas.shape[1], canvas.shape[0], alpha=True)
im.pixels = canvas.ravel()
im.filepath_raw = out
im.file_format = "PNG"
im.save()
print("NIKO_RENDER", out, canvas.shape)
