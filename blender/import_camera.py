"""Run inside Blender with a .blend open: write its camera as niko.blender_cameras/1.

    blender -b scene.blend --python import_camera.py -- out.json [camera_name] [frame_start frame_end]

The camera is evaluated per frame (matrix_world, lens, shift, sensor), so constraints,
parents and drivers are all included; no F-curves are parsed. If the scene has a movie clip,
its tracking-camera lens model is recorded in "extra" for reference.
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nikobpy  # noqa: E402

args = nikobpy.script_args()
out = args[0]
scene = bpy.context.scene
cam = bpy.data.objects[args[1]] if len(args) > 1 and args[1] not in ("", "-") else scene.camera
if cam is None or cam.type != "CAMERA":
    raise SystemExit(f"no camera found (scene camera: {scene.camera}); pass its object name")
fs = int(args[2]) if len(args) > 3 else scene.frame_start
fe = int(args[3]) if len(args) > 3 else scene.frame_end

extra = {"blend": bpy.data.filepath, "camera_object": cam.name, "scene": scene.name}
for clip in bpy.data.movieclips:
    tc = clip.tracking.camera
    extra.setdefault("movie_clips", []).append({
        "name": clip.name, "filepath": clip.filepath, "size": list(clip.size),
        "distortion_model": tc.distortion_model, "focal_px": tc.focal,
        "principal_point_px": list(getattr(tc, "principal_point_pixels", (0, 0))),
        "k1": tc.k1, "k2": tc.k2, "k3": tc.k3, "sensor_width": tc.sensor_width,
    })
nikobpy.export_cameras(scene, cam, out, fs, fe, extra=extra)
print(f"NIKO_IMPORT_DONE {cam.name} frames {fs}-{fe}")
