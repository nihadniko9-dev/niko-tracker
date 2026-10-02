"""Run inside Blender: render tiny emissive spheres at known 3D points.

    blender -b --factory-startup --python-exit-code 1 --python render_markers.py -- spec.json out_dir

For each case writes out_dir/<name>.png (16-bit grayscale, linear 'Raw' view) and
out_dir/<name>_cameras.json (niko.blender_cameras/1, exported by evaluating the
camera, the same path the benchmark generator uses). The test compares marker
centroids in the rendered image with the projection through our K, R, t.
"""

import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nikobpy  # noqa: E402
import scene_lib  # noqa: E402

spec_path, out_dir = nikobpy.script_args()[:2]
with open(spec_path, encoding="utf-8") as fh:
    spec = json.load(fh)
os.makedirs(out_dir, exist_ok=True)

scene = bpy.context.scene
for obj in list(scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

world = bpy.data.worlds.new("black")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
scene.world = world

device = nikobpy.setup_cycles(scene, samples=spec.get("samples", 64), device=spec.get("device", "GPU"))
scene.cycles.max_bounces = 0
scene.view_settings.view_transform = "Raw"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.film_transparent = False
scene.render.use_motion_blur = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "BW"
scene.render.image_settings.color_depth = "16"
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = 1.0
scene.render.pixel_aspect_y = 1.0
scene.frame_start = scene.frame_end = scene.frame_current = 1

mat = bpy.data.materials.new("marker")
mat.use_nodes = True
nodes = mat.node_tree.nodes
nodes.clear()
emit = nodes.new("ShaderNodeEmission")
emit.inputs["Strength"].default_value = spec.get("strength", 0.8)
out = nodes.new("ShaderNodeOutputMaterial")
mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])

report = {"device": device, "blender_version": bpy.app.version_string, "cases": []}
for case in spec["cases"]:
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    scene.render.resolution_x, scene.render.resolution_y = case["resolution"]
    cam_data = bpy.data.cameras.new(case["name"])
    for key, value in case["camera"].items():
        setattr(cam_data, key, value)
    cam = bpy.data.objects.new(case["name"], cam_data)
    scene.collection.objects.link(cam)
    cam.location = case["location"]
    cam.rotation_mode = "XYZ"
    cam.rotation_euler = case["rotation_euler"]
    scene.camera = cam

    # optional motion: constant yaw rate about world Z (Euler XYZ: Z is applied last, in world axes),
    # linear keys at frames 0, 1, 2 around the rendered frame 1, plus Cycles rolling shutter
    anim = case.get("animation")
    scene.render.use_motion_blur = False
    scene.cycles.rolling_shutter_type = "NONE"
    if anim:
        w = anim["yaw_rad_per_frame"]
        for f in (0, 1, 2):
            e = list(case["rotation_euler"])
            e[2] += w * (f - 1)
            cam.rotation_euler = e
            cam.keyframe_insert("rotation_euler", frame=f)
        for fc in scene_lib._fcurves(cam.animation_data):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        rs = anim["rolling_shutter"]
        shutter = rs["readout"] + rs["row_exposure"]
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = shutter
        scene.render.motion_blur_position = "CENTER"
        scene.cycles.rolling_shutter_type = "TOP"
        scene.cycles.rolling_shutter_duration = rs["row_exposure"] / shutter
        scene.frame_set(1)

    for k, (p, radius) in enumerate(zip(case["markers"], case["radii"])):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius, location=p)
        sphere = bpy.context.active_object
        sphere.name = f"marker_{k:03d}"
        sphere.data.materials.append(mat)

    bpy.context.view_layer.update()
    nikobpy.export_cameras(scene, cam, os.path.join(out_dir, f"{case['name']}_cameras.json"), 1, 1)
    scene.render.filepath = os.path.join(out_dir, f"{case['name']}.png")
    bpy.ops.render.render(write_still=True)
    report["cases"].append(case["name"])

with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh)
