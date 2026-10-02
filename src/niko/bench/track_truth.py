"""Exact ground-truth positions for point tracks of a synthetic shot.

For every track, the ray through its query pixel (GT camera of the query frame, lens distortion
removed) is cast into the shot's scene.blend; the 3D hit is projected into every frame with the
GT cameras (with distortion). Tracks that start on a moving person/car proxy are flagged and
excluded from static-scene accuracy. Occlusion is not modelled: compare only where the tracker
itself says "visible".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..camio import CameraTrack
from ..geometry import project
from ..paths import REPO, find_blender
from ..triangulate import undistort_points
from .generate import _host_path

SCRIPT = REPO / "blender" / "raycast_points.py"


def gt_tracks(shot: Path, xy: np.ndarray, query_frame: np.ndarray, work: Path) -> dict:
    gt = CameraTrack.load(shot / "gt" / "cameras.json")
    T, N = xy.shape[:2]
    q = query_frame
    uv = xy[q, np.arange(N)].astype(np.float64)
    origins = np.empty((N, 3))
    dirs = np.empty((N, 3))
    for t in np.unique(q):
        m = q == t
        K, R, tt, dist = gt.K[t], gt.R[t], gt.t[t], gt.dist[t]
        und = undistort_points(uv[m], K, dist, iters=50)
        rays = (np.linalg.inv(K) @ np.column_stack([und, np.ones(m.sum())]).T).T
        d = rays @ R  # R^T ray: camera -> world
        dirs[m] = d / np.linalg.norm(d, axis=1, keepdims=True)
        origins[m] = -R.T @ tt
    work.mkdir(parents=True, exist_ok=True)
    np.savez(work / "rays.npz", frame=(q + gt.frame_start).astype(np.int32), origin=origins, direction=dirs)
    blender = find_blender()
    cmd = [blender, "-b", _host_path(shot / "scene.blend", blender), "--python-exit-code", "1",
           "--python", _host_path(SCRIPT, blender), "--",
           _host_path(work / "rays.npz", blender), _host_path(work / "hits.npz", blender)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or "NIKO_RAYCAST_DONE" not in proc.stdout:
        raise RuntimeError(proc.stdout[-1500:] + proc.stderr[-1500:])
    h = np.load(work / "hits.npz")
    true_xy = np.full((T, N, 2), np.nan)
    good = h["hit"] & ~h["dynamic"]
    for t in range(T):
        p, z = project(gt.K[t], gt.R[t], gt.t[t], h["point"][good], gt.dist[t])
        p[z <= 0] = np.nan
        true_xy[t, good] = p
    return {"xy": true_xy, "static": good, "dynamic": h["dynamic"], "point": h["point"]}


def track_errors(xy: np.ndarray, vis: np.ndarray, truth: dict) -> np.ndarray:
    """Per-observation pixel error vs ground truth (NaN where not comparable)."""
    e = np.linalg.norm(xy - truth["xy"], axis=-1)
    e[~vis] = np.nan
    e[:, ~truth["static"]] = np.nan
    return e
