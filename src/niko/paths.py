"""Locations: repo, $NIKO_HOME (heavy data on the Linux filesystem), tools."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

WINDOWS_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
WSL_WINDOWS_BLENDER = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"


def niko_home() -> Path:
    return Path(os.environ.get("NIKO_HOME", Path.home() / "niko")).expanduser()


def env_dir(backend: str) -> Path:
    return niko_home() / "envs" / backend


def env_python(backend: str) -> Path:
    d = env_dir(backend)
    return d / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def checkpoints_dir() -> Path:
    return niko_home() / "checkpoints"


def find_blender(allow_windows_from_wsl: bool = True) -> str | None:
    """NIKO_BLENDER, then Linux Blender under $NIKO_HOME, then PATH, then the Windows install."""
    cands = [os.environ.get("NIKO_BLENDER")]
    if sys.platform != "win32":
        cands += sorted((niko_home() / "blender").glob("blender-5.2*/blender"), reverse=True)
    cands.append(shutil.which("blender"))
    if sys.platform == "win32":
        cands.append(WINDOWS_BLENDER)
    elif allow_windows_from_wsl:
        cands.append(WSL_WINDOWS_BLENDER)
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    return None
