"""Conservative reuse: match source contents, options and engine code before reading old stages."""

import hashlib
import json
from pathlib import Path


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(source: Path, settings: dict) -> dict:
    from .ingest import IMAGE_EXTS
    from ..paths import REPO

    files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS) if source.is_dir() else [source]
    code = hashlib.sha256()
    for folder in (REPO / "src", REPO / "backends"):
        for path in sorted(folder.rglob("*.py")):
            code.update(str(path.relative_to(REPO)).encode())
            code.update(path.read_bytes())
    return {"schema": "niko.reuse/1", "source": str(source.resolve()),
            "files": [[p.name, _digest(p)] for p in files], "settings": settings,
            "code": code.hexdigest()}


def check_reuse(out_dir: Path, expected: dict) -> None:
    try:
        previous = json.loads((out_dir / "reuse.json").read_text(encoding="utf-8"))
        report = json.loads((out_dir / "solve.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot safely reuse this older or incomplete solve. Choose a new output folder.") from exc
    if previous != expected or not report.get("ok"):
        raise ValueError("Cannot reuse: the source, settings or engine changed, or the previous solve failed. "
                         "Choose a new output folder.")
