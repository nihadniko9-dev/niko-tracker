"""`niko bench import-camera`: any .blend camera (MotionMaster solve, Blender tracker) -> cameras.json."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..blendercam import load_blender_cameras
from ..camio import CameraTrack
from ..paths import REPO, find_blender
from .generate import _host_path

SCRIPT = REPO / "blender" / "import_camera.py"


def import_camera(blend: str | Path, out: str | Path, camera: str | None = None,
                  frames: tuple[int, int] | None = None, name: str | None = None) -> CameraTrack:
    blend, out = Path(blend), Path(out)
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 5.2 not found")
    raw = out.with_suffix(".blender_cameras.json")
    args = [_host_path(raw, blender), camera or "-"]
    if frames:
        args += [str(frames[0]), str(frames[1])]
    cmd = [blender, "-b", _host_path(blend, blender), "--python-exit-code", "1",
           "--python", _host_path(SCRIPT, blender), "--", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or "NIKO_IMPORT_DONE" not in proc.stdout:
        raise RuntimeError(f"Blender import failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    trk = load_blender_cameras(raw, method=f"import:{blend.name}", name=name or blend.stem)
    trk.save(out)
    return trk
