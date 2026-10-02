"""Stage 1 - ingest: video or image folder -> frames/, proxy/, shot.json (docs/SCHEMA.md)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".bmp"}
PROXY_HEIGHT = 1080

# Container / stream tags that sometimes carry lens information (phones, DJI, cinema cameras).
_FOCAL_KEYS = re.compile(r"focal|lens", re.IGNORECASE)
_CAMERA_KEYS = re.compile(r"make|model|manufacturer|lens|encoder", re.IGNORECASE)


def probe_video(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    vs = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not vs:
        raise ValueError(f"{path}: no video stream")
    v = vs[0]
    rate = v.get("avg_frame_rate") if v.get("avg_frame_rate", "0/0") != "0/0" else v.get("r_frame_rate")
    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    tags = {**info.get("format", {}).get("tags", {}), **v.get("tags", {})}
    lens_tags = {k: val for k, val in tags.items() if _FOCAL_KEYS.search(k)}
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": float(Fraction(rate)),
        "fps_fraction": rate,
        "n_frames": int(v["nb_frames"]) if v.get("nb_frames", "").isdigit() else None,
        "codec": v.get("codec_name"),
        "pix_fmt": v.get("pix_fmt"),
        "rotation": rotation,
        "lens_tags": lens_tags,
        "camera_tags": {k: val for k, val in tags.items() if _CAMERA_KEYS.search(k)},
    }


def _decode_frames(path: Path, width: int, height: int):
    """Yield RGB uint8 frames via ffmpeg (every decoded frame, no resampling of time)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed decoding {path}")


def _write(img_rgb: np.ndarray, path: Path, fmt: str, quality: int) -> None:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(str(path), bgr, params):
        raise OSError(f"could not write {path}")


def ingest(
    clip: str | Path,
    shot_dir: str | Path,
    frame_format: str = "jpg",
    jpg_quality: int = 95,
    fps: float | None = None,
    frame_start: int = 1,
    proxy_height: int = PROXY_HEIGHT,
) -> dict:
    """Extract frames + proxy and write shot.json. For image folders pass fps (and frame_start)."""
    clip, shot_dir = Path(clip), Path(shot_dir)
    frames_dir, proxy_dir = shot_dir / "frames", shot_dir / "proxy"
    for d in (frames_dir, proxy_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    if clip.is_dir():
        files = sorted(p for p in clip.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"{clip}: no images")
        first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        meta = {"width": first.shape[1], "height": first.shape[0], "fps": float(fps or 25.0),
                "fps_fraction": None, "codec": "images", "pix_fmt": None, "rotation": 0, "lens_tags": {},
                "camera_tags": {}}
        source_frames = (cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in files)
    else:
        meta = probe_video(clip)
        if fps:
            meta["fps"] = float(fps)
        if meta["rotation"]:
            raise NotImplementedError(f"{clip}: rotated video ({meta['rotation']} deg) not handled yet")
        source_frames = _decode_frames(clip, meta["width"], meta["height"])
    tel = None
    if not clip.is_dir():
        from ..telemetry import TELEMETRY_FILE, read_dji, save
        try:
            tel = read_dji(clip)
        except Exception:  # telemetry is a bonus: a stream that cannot be read must not stop the solve
            tel = None
        if tel is not None:
            save(shot_dir / TELEMETRY_FILE, tel)

    W, H = meta["width"], meta["height"]
    scale = min(1.0, proxy_height / H)
    pw, ph = int(round(W * scale / 2) * 2), int(round(H * scale / 2) * 2)
    n = 0
    for i, rgb in enumerate(source_frames):
        if rgb.shape[:2] != (H, W):
            raise ValueError(f"frame {i} has size {rgb.shape[1]}x{rgb.shape[0]}, expected {W}x{H}")
        _write(rgb, frames_dir / f"{i:06d}.{frame_format}", frame_format, jpg_quality)
        proxy = rgb if scale == 1.0 else cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
        _write(proxy, proxy_dir / f"{i:06d}.jpg", "jpg", jpg_quality)
        n = i + 1

    shot = {
        "schema": "niko.shot/1",
        "source": str(clip.resolve()),
        "width": W, "height": H, "fps": meta["fps"], "fps_fraction": meta["fps_fraction"],
        "n_frames": n, "frame_start": frame_start,
        "frame_format": frame_format, "jpg_quality": jpg_quality if frame_format == "jpg" else None,
        # proxy pixel = full pixel * scale, both in corner-origin coordinates
        "proxy": {"width": pw, "height": ph, "scale_x": pw / W, "scale_y": ph / H},
        "lens": {"focal_mm": None, "focal_35mm": None, "sensor_width_mm": None,
                 "tags": meta["lens_tags"], "source": "tags" if meta["lens_tags"] else "none"},
        "codec": meta["codec"], "pix_fmt": meta["pix_fmt"], "rotation": meta["rotation"],
        "telemetry": None,
    }
    if tel is not None:
        from ..telemetry import summary
        shot["telemetry"] = summary(tel)
    from ..camera import identify
    shot["camera"] = identify(meta.get("camera_tags", {}), shot["telemetry"], W, H)
    (shot_dir / "shot.json").write_text(json.dumps(shot, indent=1), encoding="utf-8")
    return shot
