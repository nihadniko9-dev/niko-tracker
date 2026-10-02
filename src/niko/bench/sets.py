"""Benchmark shot sets: configs/benchmark/<set>.json -> one spec per shot; generation of a set."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np

from ..paths import REPO, niko_home
from .generate import generate_shot

SETS_DIR = REPO / "configs" / "benchmark"


def load_set(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = SETS_DIR / f"{name_or_path}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def shot_specs(bench: dict, only: list[str] | None = None, preview: bool = False) -> list[dict]:
    """Merge defaults into every shot (one level deep for dicts). Preview: 1/4 size, 16 frames, 16 spp."""
    out = []
    for shot in bench["shots"]:
        if only and shot["name"] not in only:
            continue
        spec = copy.deepcopy(bench["defaults"])
        for k, v in shot.items():
            if isinstance(v, dict) and isinstance(spec.get(k), dict):
                spec[k] = {**spec[k], **v}
            else:
                spec[k] = copy.deepcopy(v)
        if preview:  # same shot duration and path, 16 frames at 1/4 size: a quick composition check
            k = 16 / spec["n_frames"]
            spec["width"] //= 4
            spec["height"] //= 4
            spec["n_frames"] = 16
            spec["samples"] = 16
            spec["fps"] = spec.get("fps", 25) * k
            cam = spec["camera"]
            for key in ("whip_start", "whip_frames"):
                if key in cam:
                    cam[key] = max(1, round(cam[key] * k))
            if "slow_deg_per_frame" in cam:
                cam["slow_deg_per_frame"] /= k
        out.append(spec)
    return out


def set_dir(name: str, preview: bool = False) -> Path:
    return niko_home() / "bench" / (f"{name}_preview" if preview else name)


def generate_set(name: str, only=None, preview=False, force=False, log=print) -> list[dict]:
    bench = load_set(name)
    root = set_dir(bench["name"], preview)
    infos = []
    for spec in shot_specs(bench, only, preview):
        info = generate_shot(spec, root / spec["name"], force=force)
        state = "skipped (already done)" if info.get("skipped") else f"{info['render_seconds']:.0f}s render"
        log(f"[generate] {spec['name']:<26} {spec['width']}x{spec['height']} x{spec['n_frames']}  {state}, "
            f"scene {info['scene_size']:.1f} m, {info['device']}")
        infos.append({"name": spec["name"], **info})
    return infos


def contact_sheet(root: Path, out: Path, thumb_w: int = 320) -> Path:
    """First / middle / last frame of every shot in a grid (quick visual check)."""
    rows = []
    for shot in sorted(p for p in root.iterdir() if (p / "frames").is_dir()):
        files = sorted((shot / "frames").glob("*.png"))
        if not files:
            continue
        picks = [files[0], files[len(files) // 2], files[-1]]
        tiles = []
        for f in picks:
            img = cv2.imread(str(f))
            h = int(img.shape[0] * thumb_w / img.shape[1])
            tiles.append(cv2.resize(img, (thumb_w, h), interpolation=cv2.INTER_AREA))
        row = np.hstack(tiles)
        label = np.full((22, row.shape[1], 3), 255, np.uint8)
        cv2.putText(label, shot.name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.vstack([label, row]))
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out
