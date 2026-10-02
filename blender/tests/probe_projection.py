"""Run inside Blender: report how Blender itself projects points through cameras.

    blender -b --factory-startup --python-exit-code 1 --python probe_projection.py -- in.json out.json

in.json:  {"cases": [{"camera": {...settings...}, "matrix_world": 4x4 | null,
                      "location": [...], "rotation_euler": [...], "scale": [...],
                      "points": [[x,y,z], ...]}, ...]}
out.json: {"blender_version": "...", "cases": [{"matrix_world": 4x4, "ndc": [[x,y,z], ...],
                                                "render": {...}}, ...]}

`ndc` is bpy_extras.object_utils.world_to_camera_view: x, y in [0, 1] of the camera
frame with the origin at the bottom-left, z = depth along the view axis.
"""

import json
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
with open(src, encoding="utf-8") as fh:
    job = json.load(fh)

scene = bpy.context.scene
out = {"blender_version": bpy.app.version_string, "cases": []}

for case in job["cases"]:
    c = case["camera"]
    r = scene.render
    r.resolution_x = c["resolution_x"]
    r.resolution_y = c["resolution_y"]
    r.resolution_percentage = c.get("resolution_percentage", 100)
    r.pixel_aspect_x = c.get("pixel_aspect_x", 1.0)
    r.pixel_aspect_y = c.get("pixel_aspect_y", 1.0)

    cam_data = bpy.data.cameras.new("probe_cam")
    cam_data.type = "PERSP"
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.lens = c["lens"]
    cam_data.sensor_width = c["sensor_width"]
    cam_data.sensor_height = c["sensor_height"]
    cam_data.sensor_fit = c["sensor_fit"]
    cam_data.shift_x = c["shift_x"]
    cam_data.shift_y = c["shift_y"]
    cam = bpy.data.objects.new("probe_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    if case.get("matrix_world") is not None:
        cam.matrix_world = Matrix(case["matrix_world"])
    else:
        cam.location = case["location"]
        cam.rotation_mode = "XYZ"
        cam.rotation_euler = case["rotation_euler"]
        cam.scale = case.get("scale", [1.0, 1.0, 1.0])
    bpy.context.view_layer.update()

    ndc = [list(world_to_camera_view(scene, cam, Vector(p))) for p in case["points"]]
    out["cases"].append({
        "matrix_world": [list(row) for row in cam.matrix_world],
        "ndc": ndc,
        "render": {
            "width": r.resolution_x * r.resolution_percentage // 100,
            "height": r.resolution_y * r.resolution_percentage // 100,
        },
    })
    bpy.data.objects.remove(cam)
    bpy.data.cameras.remove(cam_data)

with open(dst, "w", encoding="utf-8") as fh:
    json.dump(out, fh)
