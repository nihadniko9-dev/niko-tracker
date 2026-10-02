"""Validate and stage an add-on update before replacing the installed directory."""

import os
from pathlib import Path
import shutil
import tempfile
import uuid

REQUIRED = {"__init__.py", "engine.py", "ops.py", "props.py", "ui.py", "draw.py", "solve_io.py"}


def install_files(files: dict[str, bytes], here: str) -> str:
    if not REQUIRED.issubset(files):
        raise ValueError("Incomplete add-on update; installed files were kept")
    for name, data in files.items():
        if Path(name).name != name or not name.endswith(".py") or "\\" in name:
            raise ValueError(f"Invalid update filename: {name}")
        compile(data, name, "exec")
    target = Path(here).resolve()
    # Keep backups outside addons/ so Blender does not discover duplicate add-ons.
    backups = target.parent.parent / "niko_tracker_backups"
    backups.mkdir(exist_ok=True)
    backup = backups / (target.name + "-" + uuid.uuid4().hex[:8])
    with tempfile.TemporaryDirectory(prefix="niko-update-", dir=target.parent) as scratch:
        staged = Path(scratch) / target.name
        shutil.copytree(target, staged, ignore=shutil.ignore_patterns("__pycache__"))
        for name, data in files.items():
            (staged / name).write_bytes(data)
        os.replace(target, backup)
        try:
            os.replace(staged, target)
        except Exception:
            os.replace(backup, target)
            raise
    return str(backup)


def zip_files(zf) -> dict[str, bytes]:
    names = [n for n in zf.namelist() if n.startswith("niko_tracker/") and n.endswith(".py")]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate files in update")
    files = {}
    for name in names:
        short = name.removeprefix("niko_tracker/")
        if "/" in short or "\\" in short:
            raise ValueError("Unexpected nested path in add-on update")
        files[short] = zf.read(name)
    return files
