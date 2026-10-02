"""Run inside Blender 5.2: render one synthetic shot with exact ground truth.

    blender -b --factory-startup --python-exit-code 1 --python gen_shot.py -- spec.json out_dir

Spec keys (see configs/shots/*.json and niko.bench.generate):
  name, seed, width, height, fps, frame_start, n_frames, samples, max_bounces, device
  scene      {"kind": "yard"|"town", "extent", "n_objects", "contrast"}
  movers     {"people", "cars", "center", "radius", "seed"}           (optional)
  camera     {"path": ..., path params, "lens", "lens_end", "sensor_width", "shake_deg", "seed"}
  motion_blur      shutter in frames (0 = off)
  rolling_shutter  {"readout": fraction of a frame, "row_exposure": frames}   (optional)
  render     {"width", "height", "lens_factor"}   overscan for lens distortion, set by niko

Writes into out_dir: render/000000.png ..., blender_cameras.json, scene.json, scene.blend.
The orchestrator turns render/ into frames/ (distortion, noise) and writes gt/cameras.json.
"""

import json
import math
import os
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nikobpy  # noqa: E402
import scene_lib  # noqa: E402
import shots  # noqa: E402

spec_path, out_dir = nikobpy.script_args()[:2]
with open(spec_path, encoding="utf-8") as fh:
    spec = json.load(fh)
os.makedirs(os.path.join(out_dir, "render"), exist_ok=True)

scene = bpy.context.scene
for obj in list(scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

n = spec["n_frames"]
f0 = spec.get("frame_start", 1)
fps = spec.get("fps", 25)
scene.frame_start, scene.frame_end = f0, f0 + n - 1
scene.render.fps, scene.render.fps_base = int(round(fps)), int(round(fps)) / fps
render = spec.get("render", {})
scene.render.resolution_x = render.get("width", spec["width"])
scene.render.resolution_y = render.get("height", spec["height"])
scene.render.resolution_percentage = 100  # see PROGRESS.md: non-exact percentages break the camera frame
scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0

sc = spec.get("scene", {})
scene_lib.CONTRAST = sc.get("contrast", 1.0)
cam_spec = spec["camera"]
def _corridors(cam, n, fps):
    """Keep props off the camera's own path (x0, y0, x1, y1, half_width)."""
    p = cam["path"]
    if p == "walk":
        a = math.radians(cam.get("yaw_deg", 0.0))
        L = cam.get("speed", 1.2) * n / fps + 2.0
        x0, y0 = cam["start"][:2]
        return [(x0, y0, x0 + L * math.cos(a), y0 + L * math.sin(a), 1.5)]
    if p == "line":
        return [(*cam["start"][:2], *cam["end"][:2], 1.5)]
    if p in ("pan", "whip"):
        return [(*cam["eye"][:2], *cam["eye"][:2], 2.5)]
    return []


if sc.get("kind", "yard") == "town":
    shots.build_town(spec.get("seed", 0), extent=sc.get("extent", 90.0))
else:
    scene_lib.build_yard(spec.get("seed", 0), extent=sc.get("extent", 20.0), n_objects=sc.get("n_objects", 30),
                         clear_radius=cam_spec.get("radius", 0.0) if cam_spec["path"] == "orbit" else 0.0,
                         clear_segments=_corridors(cam_spec, n, fps), clear_width=sc.get("clear_width", 2.0))
movers = shots.add_movers(spec.get("movers"), n, fps, f0)

cam_data = bpy.data.cameras.new("camera")
lens_factor = render.get("lens_factor", 1.0)  # keeps fx in pixels when rendering with overscan
cam_data.lens = cam_spec.get("lens", 28.0) * lens_factor
cam_data.sensor_width = cam_spec.get("sensor_width", 36.0)
cam_data.sensor_fit = "HORIZONTAL"
cam_data.clip_start, cam_data.clip_end = 0.05, 2000.0
cam = bpy.data.objects.new("camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

mats, lenses = shots.camera_path(cam_spec, n, fps, movers)
scene_lib.keyframe_camera(cam, mats, f0, None if lenses is None else [l * lens_factor for l in lenses])

device = nikobpy.setup_cycles(scene, samples=spec.get("samples", 64), device=spec.get("device", "GPU"),
                              seed=spec.get("seed", 0))
scene.cycles.max_bounces = spec.get("max_bounces", 4)

# Motion blur / rolling shutter. Cycles exposes row y (0 = top with type TOP) at shutter-normalised
# time tau = (y/H)(1-d) + d*u, u in [0,1]; with position CENTER the middle row is centred on the
# frame time, so the GT pose of a frame is the pose of its middle row, and the rows span
# readout = shutter * (1 - d) frames. Checked by tests/test_render_markers.py (rolling shutter case).
blur = spec.get("motion_blur", 0.0)
rs = spec.get("rolling_shutter")
if rs:
    exposure = rs.get("row_exposure", max(blur, 0.02))
    shutter = rs["readout"] + exposure
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = shutter
    scene.cycles.rolling_shutter_type = "TOP"
    scene.cycles.rolling_shutter_duration = exposure / shutter
elif blur > 0:
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = blur
else:
    scene.render.use_motion_blur = False
scene.render.motion_blur_position = "CENTER"
scene.render.use_persistent_data = True
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"

cams = nikobpy.export_cameras(scene, cam, os.path.join(out_dir, "blender_cameras.json"), f0, f0 + n - 1)

hits, dists = scene_lib.sample_scene_points(scene, cam, range(f0, f0 + n, max(1, n // 30)))
centers = [tuple(fr["matrix_world"][r][3] for r in range(3)) for fr in cams["frames"]]


def _pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


pts = hits + centers
lo = [_pct([p[k] for p in pts], 0.02) for k in range(3)]
hi = [_pct([p[k] for p in pts], 0.98) for k in range(3)]
scene_size = math.dist(lo, hi)

t0 = time.time()
for i in range(n):
    scene.frame_set(f0 + i)
    scene.render.filepath = os.path.join(out_dir, "render", f"{i:06d}.png")
    bpy.ops.render.render(write_still=True)
render_s = time.time() - t0

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "scene.blend"), compress=True)
info = {
    "schema": "niko.synthetic_shot/1",
    "spec": spec,
    "blender_version": bpy.app.version_string,
    "device": device,
    "render_seconds": render_s,
    "scene_size": scene_size,
    "scene_size_definition": "diagonal of the 2-98 percentile box of ray-cast scene hits and camera centres",
    "median_depth": _pct(dists, 0.5) if dists else None,
    "movers": [{"kind": k, "start": p[0], "end": p[-1]} for k, p, _ in movers],
    "motion_blur_shutter": scene.render.motion_blur_shutter if scene.render.use_motion_blur else 0.0,
    "rolling_shutter": ({"readout": rs["readout"], "direction": "top_to_bottom",
                         "row_exposure": scene.render.motion_blur_shutter * scene.cycles.rolling_shutter_duration}
                        if rs else None),
    "points_sample": [list(p) for p in hits[:: max(1, len(hits) // 2000)]],
}
with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as fh:
    json.dump(info, fh, indent=1)
print(f"NIKO_GEN_DONE {n} frames in {render_s:.1f}s on {device}, scene_size={scene_size:.2f}")
