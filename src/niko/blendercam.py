"""Raw Blender camera export (niko.blender_cameras/1) -> CameraTrack (cameras.json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camio import CameraTrack
from .geometry import blender_matrix_world_to_opencv, blender_to_K


def render_size(d: dict) -> tuple[int, int]:
    """Effective render size, as Blender computes it (integer percentage scaling)."""
    pct = d.get("resolution_percentage", 100)
    return d["resolution_x"] * pct // 100, d["resolution_y"] * pct // 100


def track_from_blender(d: dict, method: str = "gt", name: str = "") -> CameraTrack:
    if d.get("schema") != "niko.blender_cameras/1":
        raise ValueError(f"not a niko.blender_cameras/1 file: {d.get('schema')!r}")
    W, H = render_size(d)
    frames = d["frames"]
    trk = CameraTrack.empty(method, W, H, float(d["fps"]), int(d["frame_start"]), len(frames), name=name)
    for i, f in enumerate(frames):
        if f["frame"] != trk.frame_start + i:
            raise ValueError(f"frames must be consecutive: got {f['frame']} at index {i}")
        if f.get("type", "PERSP") != "PERSP":
            raise ValueError(f"frame {f['frame']}: only perspective cameras are supported, got {f['type']}")
        trk.K[i] = blender_to_K(f["lens"], f["sensor_width"], f["sensor_height"], f["sensor_fit"],
                                f["shift_x"], f["shift_y"], W, H,
                                d.get("pixel_aspect_x", 1.0), d.get("pixel_aspect_y", 1.0))
        trk.R[i], trk.t[i] = blender_matrix_world_to_opencv(np.array(f["matrix_world"]))
        trk.dist[i] = 0.0
        trk.valid[i] = True
    lenses = {round(f["lens"], 9) for f in frames}
    trk.intrinsics_mode = "shared" if len(lenses) == 1 else "per_frame_focal"
    trk.extra = {"blender_version": d.get("blender_version"), **d.get("extra", {})}
    return trk


def load_blender_cameras(path: str | Path, **kw) -> CameraTrack:
    return track_from_blender(json.loads(Path(path).read_text(encoding="utf-8")), **kw)
