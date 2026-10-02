"""`niko bench run`: solve every shot of a set with the given methods (resumable)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..paths import niko_home
from ..pipeline.solve import solve
from .sets import SETS_DIR, load_set, set_dir


def shots_in(root: Path) -> list[Path]:
    """Synthetic shots: folders with spec.json + frames/ + gt/."""
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "spec.json").exists() and (p / "gt").is_dir())


def run_set(set_name: str, run_name: str, methods, only=None, force=False, prompts=None, log=print,
            reuse: bool = False) -> Path:
    if Path(set_name).is_dir():
        root, cfg = Path(set_name), {}
    else:
        cfg = load_set(set_name) if (SETS_DIR / f"{set_name}.json").exists() else {}
        root = set_dir(cfg.get("name", set_name))
    prompts = prompts or cfg.get("prompts")
    out_root = niko_home() / "runs" / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "run.json"
    meta = {"set": str(root), "methods": list(methods), "prompts": prompts,
            "started": datetime.now(timezone.utc).isoformat(), "shots": {}}
    for shot in shots_in(root):
        if only and shot.name not in only:
            continue
        out = out_root / shot.name
        if (out / "solve.json").exists() and not force:
            log(f"[{shot.name}] already solved, skipping")
            meta["shots"][shot.name] = "skipped"
            continue
        t0 = time.time()
        if force and not reuse and out.exists():
            from uuid import uuid4
            backup = out_root / ".previous" / f"{out.name}-{uuid4().hex[:8]}"
            backup.parent.mkdir(exist_ok=True)
            out.rename(backup)
            log(f"[{shot.name}] previous output kept at {backup}")
        rep = solve(shot, out, methods=methods, prompts=prompts, reuse=reuse,
                    log=lambda s, n=shot.name: log(f"[{n}] {s}"))
        meta["shots"][shot.name] = {"ok": rep["ok"], "seconds": round(time.time() - t0, 1),
                                    "selected": rep.get("selected")}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out_root
