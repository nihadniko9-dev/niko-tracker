"""Helpers shared by the bpy scripts in this folder (run inside Blender 5.2).

Camera data leaves Blender only as raw Blender settings (`niko.blender_cameras/1`);
the conversion to OpenCV cameras.json happens once, in niko.blendercam, which is
tested against Blender itself.
"""

import json

import bpy


def script_args():
    import sys

    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def setup_cycles(scene, samples=64, device="GPU", seed=0):
    """Cycles on the GPU (OptiX, then CUDA), falling back to CPU. Returns the device used."""
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.seed = seed
    used = "CPU"
    if device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                used = f"{backend}:{gpus[0].name}"
                break
    if used == "CPU":
        scene.cycles.device = "CPU"
    return used


def camera_frame_record(scene, cam_obj):
    """Raw Blender camera state at the current (already evaluated) frame."""
    cam = cam_obj.data
    return {
        "frame": scene.frame_current,
        "matrix_world": [list(row) for row in cam_obj.matrix_world],
        "lens": cam.lens,
        "sensor_width": cam.sensor_width,
        "sensor_height": cam.sensor_height,
        "sensor_fit": cam.sensor_fit,
        "shift_x": cam.shift_x,
        "shift_y": cam.shift_y,
        "type": cam.type,
    }


def export_cameras(scene, cam_obj, path, frame_start, frame_end, extra=None):
    """Evaluate the camera on every frame (no F-curve parsing) and write niko.blender_cameras/1."""
    r = scene.render
    frames = []
    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        frames.append(camera_frame_record(scene, cam_obj))
    data = {
        "schema": "niko.blender_cameras/1",
        "blender_version": bpy.app.version_string,
        "resolution_x": r.resolution_x,
        "resolution_y": r.resolution_y,
        "resolution_percentage": r.resolution_percentage,
        "pixel_aspect_x": r.pixel_aspect_x,
        "pixel_aspect_y": r.pixel_aspect_y,
        "fps": r.fps / r.fps_base,
        "frame_start": frame_start,
        "camera_object": cam_obj.name,
        "frames": frames,
        "extra": extra or {},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return data
