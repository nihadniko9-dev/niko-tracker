"""Stage 4 - camera candidates: run a backend, turn its camera_raw.npz into cameras.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..backend import run_backend
from ..camio import CameraTrack

# Keyframe stride for the SfM / SLAM candidates: at 60 fps neighbouring frames add cost, not
# geometry, and long clips make COLMAP slow (418 frames at 60 fps: 1656 s for colmap_global).
# Refinement fills the frames in between (niko.refine.interpolate_gaps + bundle adjustment).
STRIDE_FPS = 30.0        # at most ~30 keyframes per second
MAX_KEYFRAMES = 200      # and at most this many keyframes
STRIDED = ("colmap_global", "colmap_incremental", "megasam")


def keyframe_stride(n_frames: int, fps: float) -> int:
    """synthetic_v1 (150 frames, 25 fps) -> 1; 418 frames at 59.94 fps -> 3; 1323 -> 7."""
    s = max(1, round(fps / STRIDE_FPS))
    return max(s, -(-n_frames // MAX_KEYFRAMES))


def candidate_stride(trk: CameraTrack) -> int:
    return int(trk.extra.get("backend_meta", {}).get("stride", 1) or 1)


def coverage(trk: CameraTrack) -> float:
    """Registered share of the frames the method was given (all frames, or its keyframes)."""
    s = candidate_stride(trk)
    return float(trk.valid.sum()) / max(1, -(-trk.n_frames // s))


# method name -> (backend env, fixed options)
METHODS = {
    "colmap_global": ("colmap", {"mapper": "global"}),
    # same features + matches as colmap_global when that ran first (checked by the adapter)
    "colmap_incremental": ("colmap", {"mapper": "incremental", "reuse_db_from": "colmap_global"}),
    "da3": ("da3", {}),
    "megasam": ("megasam", {}),
}


def track_from_raw(raw_path: Path, shot: dict, method: str) -> CameraTrack:
    d = np.load(raw_path)
    meta = json.loads(str(d["meta"]))
    T = shot["n_frames"]
    if d["valid"].shape != (T,):
        raise ValueError(f"{raw_path}: {d['valid'].shape[0]} frames, shot has {T}")
    trk = CameraTrack.empty(method, shot["width"], shot["height"], shot["fps"], shot["frame_start"], T,
                            name=Path(shot["source"]).name)
    v = d["valid"].copy()
    # a candidate frame is only valid if its numbers are finite and R is a rotation
    for i in np.nonzero(v)[0]:
        R = d["R"][i]
        if not (np.all(np.isfinite(d["K"][i])) and np.all(np.isfinite(R)) and np.all(np.isfinite(d["t"][i]))
                and np.abs(R.T @ R - np.eye(3)).max() < 1e-4 and abs(np.linalg.det(R) - 1) < 1e-4):
            v[i] = False
    trk.K[v], trk.dist[v], trk.R[v], trk.t[v] = d["K"][v], d["dist"][v], d["R"][v], d["t"][v]
    # tiny numerical cleanup so the file passes the strict schema
    trk.K[v, 0, 1] = trk.K[v, 1, 0] = trk.K[v, 2, 0] = trk.K[v, 2, 1] = 0.0
    trk.K[v, 2, 2] = 1.0
    for i in np.nonzero(v)[0]:
        U, _, Vt = np.linalg.svd(trk.R[i])
        trk.R[i] = U @ Vt
    trk.valid = v
    trk.intrinsics_mode = meta.get("intrinsics_mode", "shared")
    trk.extra = {"backend_meta": meta}
    return trk


def run_candidate(method: str, shot_dir: Path, out_root: Path, options: dict | None = None) -> tuple[CameraTrack, dict]:
    backend, fixed = METHODS[method]
    out_dir = out_root / method
    result = run_backend(backend, "camera", shot_dir, out_dir, {**fixed, **(options or {})})
    shot = json.loads((shot_dir / "shot.json").read_text())
    trk = track_from_raw(out_dir / result["outputs"]["camera"], shot, method)
    trk.points = "points.ply" if "points" in result["outputs"] else None
    trk.extra.update({"runtime_s": result["runtime_s"], "peak_vram_mb": result["peak_vram_mb"]})
    trk.save(out_dir / "cameras.json")
    return trk, result
