"""Stage 7 - export: cameras.json + points.ply + a self-contained bpy import script.

The Blender script carries per-frame Blender camera values computed here with the tested
niko.geometry conversion, so Blender-side code does no camera maths of its own.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..camio import CameraTrack
from ..geometry import K_to_blender, opencv_to_blender_matrix_world

BPY_TEMPLATE = '''"""Import a Niko Tracker solve into Blender 5.2.

Niko Tracker Engine - author: Nihad Jihad. Generated file, do not edit.
Run:  blender --python import_blender.py      (or open it in Blender's Text Editor and Run)
Creates the solved camera (one key per frame), the footage as camera background, and points.ply.
"""
import json
import os

import bpy
from mathutils import Matrix

DATA = json.loads(r"""__DATA__""")
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ""


def resolve(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    cand = os.path.join(HERE, path)
    return cand if os.path.exists(cand) else path


scene = bpy.context.scene
r = scene.render
r.resolution_x, r.resolution_y = DATA["width"], DATA["height"]
r.resolution_percentage = 100
r.pixel_aspect_x, r.pixel_aspect_y = DATA["pixel_aspect_x"], DATA["pixel_aspect_y"]
r.fps, r.fps_base = DATA["fps_int"], DATA["fps_int"] / DATA["fps"]
scene.frame_start = DATA["frame_start"]
scene.frame_end = DATA["frame_start"] + len(DATA["frames"]) - 1

cam_data = bpy.data.cameras.new(DATA["name"])
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = DATA["sensor_width"]
cam = bpy.data.objects.new(DATA["name"], cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam.rotation_mode = "QUATERNION"
for f in DATA["frames"]:
    if not f["valid"]:
        continue
    cam.matrix_world = Matrix(f["matrix_world"])
    cam.keyframe_insert("location", frame=f["frame"])
    cam.keyframe_insert("rotation_quaternion", frame=f["frame"])
    cam_data.lens = f["lens"]
    cam_data.shift_x, cam_data.shift_y = f["shift_x"], f["shift_y"]
    for prop in ("lens", "shift_x", "shift_y"):
        cam_data.keyframe_insert(prop, frame=f["frame"])

frames_dir = resolve(DATA["frames_dir"])
first = os.path.join(frames_dir, DATA["first_frame_file"])
if os.path.exists(first):
    img = bpy.data.images.load(first)
    img.source = "SEQUENCE"
    cam_data.show_background_images = True
    bg = cam_data.background_images.new()
    bg.image = img
    bg.image_user.frame_start = DATA["frame_start"]
    bg.image_user.frame_duration = len(DATA["frames"])
    bg.image_user.frame_offset = -1
    bg.alpha = 1.0
else:
    print("footage not found:", first)

ply = resolve(DATA["points"]) if DATA["points"] else None
if ply and os.path.exists(ply):
    bpy.ops.wm.ply_import(filepath=ply)
    bpy.context.active_object.name = DATA["name"] + "_points"
scene.frame_set(DATA["frame_start"])
print("Niko Tracker import done:", DATA["name"], DATA["method"])
'''


def blender_frames(trk: CameraTrack, sensor_width: float = 36.0) -> tuple[list[dict], dict]:
    frames = []
    aspect = None
    for i in range(trk.n_frames):
        rec = {"frame": trk.frame_start + i, "valid": bool(trk.valid[i])}
        if trk.valid[i]:
            b = K_to_blender(trk.K[i], trk.width, trk.height, sensor_width)
            M = opencv_to_blender_matrix_world(trk.R[i], trk.t[i])
            rec.update({"matrix_world": M.tolist(), "lens": b["lens"], "shift_x": b["shift_x"],
                        "shift_y": b["shift_y"]})
            aspect = aspect or (b["pixel_aspect_x"], b["pixel_aspect_y"])
        frames.append(rec)
    return frames, {"pixel_aspect_x": (aspect or (1, 1))[0], "pixel_aspect_y": (aspect or (1, 1))[1]}


def median_depth(trk: CameraTrack, X: np.ndarray | None) -> float | None:
    """Median depth of the solve's points in front of the cameras, in solve units."""
    if X is None or not len(X):
        return None
    v = np.nonzero(trk.valid)[0]
    d = float(np.median([np.median((X @ trk.R[t].T + trk.t[t])[:, 2]) for t in v[:: max(1, len(v) // 10)]]))
    return d if d > 0 else None


def blender_world(trk: CameraTrack, ply: Path | None, target_depth: float = 10.0,
                  metres_per_unit: float | None = None) -> tuple[list, str]:
    """4x4 (row-major) taking the solve's world to a levelled Blender world (export_ae.world_alignment):
    up on +Z, the first camera looking along +Y, origin on the ground under the scene; in metres when `metres_per_unit` is known (niko.scale), else scaled so the median point
    depth is `target_depth` units. The add-on puts it on an empty that parents the camera and
    points, so the per-frame camera values stay exactly the engine's."""
    from ..export_ae import world_alignment
    from ..plyio import read_ply_xyz

    rng = np.random.default_rng(0)
    X = read_ply_xyz(ply) if ply is not None and Path(ply).exists() else None
    A, origin, how = world_alignment(trk, X, rng)  # rows: right, down, forward (After Effects axes)
    B = np.stack([A[0], A[2], -A[1]])  # Blender: X right, Y forward, Z up
    s = 1.0
    depth = median_depth(trk, X)
    if metres_per_unit:
        s = float(metres_per_unit)
    elif depth:
        s = target_depth / depth
    M = np.eye(4)
    M[:3, :3] = s * B
    M[:3, 3] = -s * B @ origin
    return M.tolist(), how


def export_solve(trk: CameraTrack, points_src: Path | None, out_dir: Path, frames_dir: Path,
                 first_frame_file: str, metric: dict | None = None) -> dict:
    """metric: niko.scale's estimate; the Blender scene is then in metres (approximately)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if points_src is not None and Path(points_src).exists():
        shutil.copyfile(points_src, out_dir / "points.ply")
        trk.points = "points.ply"
    else:
        trk.points = None
    trk.save(out_dir / "cameras.json")
    frames, aspect = blender_frames(trk)
    if any(np.asarray(trk.dist)[trk.valid].ravel() != 0):
        note = "lens distortion is in cameras.json; the Blender camera is the undistorted pinhole"
    else:
        note = ""
    ply = out_dir / "points.ply" if trk.points else None
    mpu = (metric or {}).get("metres_per_unit")
    world, world_how = blender_world(trk, ply, metres_per_unit=mpu)
    from ..plyio import read_ply_xyz
    depth = median_depth(trk, read_ply_xyz(ply)) if ply is not None and ply.exists() else None
    units = ({"kind": "metres_estimated", "metres_per_unit": mpu, "agree_pct": metric.get("agree_pct")}
             if mpu else {"kind": "arbitrary"})
    data = {"name": f"niko_{trk.name or 'shot'}", "method": trk.method, "width": trk.width,
            "height": trk.height, "fps": trk.fps, "fps_int": int(round(trk.fps)),
            "frame_start": trk.frame_start, "sensor_width": 36.0, **aspect, "frames": frames,
            "frames_dir": str(frames_dir), "first_frame_file": first_frame_file,
            "points": "points.ply" if trk.points else None, "note": note,
            "world": world, "world_how": world_how, "units": units, "median_depth": depth}
    script = BPY_TEMPLATE.replace("__DATA__", json.dumps(data))
    (out_dir / "import_blender.py").write_text(script, encoding="utf-8")
    (out_dir / "blender.json").write_text(json.dumps(data), encoding="utf-8")  # read by the Blender add-on
    return {"cameras": str(out_dir / "cameras.json"), "script": str(out_dir / "import_blender.py"),
            "points": trk.points}
