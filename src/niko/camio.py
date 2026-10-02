"""cameras.json read / write / validate (schema: docs/SCHEMA.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__

SCHEMA_ID = "niko.cameras/1"
SCHEMA_FILE = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "cameras.schema.json"


class CameraFileError(ValueError):
    pass


@dataclass
class CameraTrack:
    """In-memory form of cameras.json. Invalid frames hold NaN in K/dist/R/t."""

    method: str
    width: int
    height: int
    fps: float
    frame_start: int
    K: np.ndarray  # [T,3,3]
    dist: np.ndarray  # [T,5]
    R: np.ndarray  # [T,3,3] world->camera
    t: np.ndarray  # [T,3]
    valid: np.ndarray  # [T] bool
    name: str = ""
    world_up: str = "unknown"
    units: str = "arbitrary"
    intrinsics_mode: str = "shared"
    rolling_shutter: dict | None = None
    points: str | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.valid.shape[0])

    @property
    def centers(self) -> np.ndarray:
        return -np.einsum("nji,nj->ni", self.R, self.t)

    @property
    def R_c2w(self) -> np.ndarray:
        return np.transpose(self.R, (0, 2, 1))

    @classmethod
    def empty(cls, method: str, width: int, height: int, fps: float, frame_start: int, n: int, **kw):
        nan33 = np.full((n, 3, 3), np.nan)
        return cls(
            method=method, width=width, height=height, fps=fps, frame_start=frame_start,
            K=nan33.copy(), dist=np.full((n, 5), np.nan), R=nan33.copy(),
            t=np.full((n, 3), np.nan), valid=np.zeros(n, dtype=bool), **kw,
        )

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        frames = []
        for i in range(self.n_frames):
            ok = bool(self.valid[i])
            frames.append({
                "i": i,
                "frame": self.frame_start + i,
                "valid": ok,
                "K": self.K[i].tolist() if ok else None,
                "dist": self.dist[i].tolist() if ok else None,
                "R": self.R[i].tolist() if ok else None,
                "t": self.t[i].tolist() if ok else None,
            })
        return {
            "schema": SCHEMA_ID,
            "producer": {
                "method": self.method,
                "version": f"niko {__version__}",
                "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "notes": self.notes,
            },
            "clip": {
                "name": self.name,
                "width": int(self.width),
                "height": int(self.height),
                "fps": float(self.fps),
                "frame_start": int(self.frame_start),
                "n_frames": self.n_frames,
            },
            "conventions": {
                "camera": "opencv",
                "extrinsics": "world_to_camera",
                "pixel_origin": "corner",
                "world_up": self.world_up,
                "units": self.units,
            },
            "intrinsics_mode": self.intrinsics_mode,
            "distortion_model": "opencv5",
            "rolling_shutter": self.rolling_shutter,
            "points": self.points,
            "frames": frames,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraTrack":
        validate(d)
        clip = d["clip"]
        n = clip["n_frames"]
        trk = cls.empty(d["producer"]["method"], clip["width"], clip["height"], clip["fps"],
                        clip["frame_start"], n)
        for f in d["frames"]:
            i = f["i"]
            if f["valid"]:
                trk.K[i] = f["K"]
                trk.dist[i] = f["dist"]
                trk.R[i] = f["R"]
                trk.t[i] = f["t"]
                trk.valid[i] = True
        conv = d["conventions"]
        trk.name = clip.get("name", "")
        trk.world_up = conv.get("world_up", "unknown")
        trk.units = conv.get("units", "arbitrary")
        trk.intrinsics_mode = d.get("intrinsics_mode", "shared")
        trk.rolling_shutter = d.get("rolling_shutter")
        trk.points = d.get("points")
        trk.notes = d["producer"].get("notes", "")
        trk.extra = d.get("extra", {})
        return trk

    def save(self, path: str | Path) -> None:
        d = self.to_dict()
        validate(d)
        Path(path).write_text(json.dumps(d, indent=1, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CameraTrack":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------- #

_validator = None


def _json_validator():
    global _validator
    if _validator is None:
        import jsonschema

        schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        _validator = jsonschema.Draft202012Validator(schema)
    return _validator


def validate(d: dict, rot_tol: float = 1e-5) -> None:
    """Raise CameraFileError if `d` breaks the JSON Schema or the semantic rules."""
    errors = sorted(_json_validator().iter_errors(d), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "<root>"
        raise CameraFileError(f"schema: {where}: {e.message}")

    clip = d["clip"]
    frames = d["frames"]
    if len(frames) != clip["n_frames"]:
        raise CameraFileError(f"n_frames={clip['n_frames']} but {len(frames)} frames listed")
    for k, f in enumerate(frames):
        if f["i"] != k:
            raise CameraFileError(f"frames[{k}].i = {f['i']}, expected {k}")
        if f["frame"] != clip["frame_start"] + k:
            raise CameraFileError(f"frames[{k}].frame = {f['frame']}, expected {clip['frame_start'] + k}")
        if not f["valid"]:
            continue
        K = np.asarray(f["K"], dtype=np.float64)
        R = np.asarray(f["R"], dtype=np.float64)
        vals = np.concatenate([K.ravel(), R.ravel(), np.asarray(f["t"], float), np.asarray(f["dist"], float)])
        if not np.all(np.isfinite(vals)):
            raise CameraFileError(f"frames[{k}]: non-finite values")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise CameraFileError(f"frames[{k}]: focal must be positive")
        if K[0, 1] != 0 or K[1, 0] != 0 or not np.array_equal(K[2], [0.0, 0.0, 1.0]):
            raise CameraFileError(f"frames[{k}]: K must be [[fx,0,cx],[0,fy,cy],[0,0,1]]")
        if np.abs(R.T @ R - np.eye(3)).max() > rot_tol or abs(np.linalg.det(R) - 1.0) > rot_tol:
            raise CameraFileError(f"frames[{k}]: R is not a rotation")
