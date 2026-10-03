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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".dpx", ".bmp"}
# read through ffmpeg (this OpenCV build has no EXR codec; DPX it never had); linear EXR gets the sRGB
# curve so the tracker sees a normal picture, not a dark one
_FFMPEG_IMAGE_EXTS = {".exr", ".dpx"}
PROXY_HEIGHT = 1080

# Container / stream tags that sometimes carry lens information (phones, DJI, cinema cameras).
_FOCAL_KEYS = re.compile(r"focal|lens", re.IGNORECASE)
_CAMERA_KEYS = re.compile(r"make|model|manufacturer|lens|encoder", re.IGNORECASE)


# camera RAW formats that only their maker's software decodes
_RAW_FORMATS = {".braw": "Blackmagic RAW", ".r3d": "RED R3D", ".ari": "ARRIRAW", ".arx": "ARRIRAW",
                ".crm": "Canon Cinema RAW Light", ".mxf": "a RAW MXF (ARRIRAW, Canon or Sony RAW)"}


def _raw_hint(path: Path, tag: str) -> str:
    kind = _RAW_FORMATS.get(Path(path).suffix.lower()) or ("Blackmagic RAW" if tag == "brst" else "this video format")
    return (f"{Path(path).name}: {kind} needs its maker's software to decode. Export it from DaVinci Resolve "
            "(free) as ProRes 422 HQ at full resolution, or as an EXR / TIFF sequence, and solve that.")


# frame rates cameras and editors use; a variable-rate clip is set to the nearest one of these
STANDARD_FPS = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60), Fraction(100),
                Fraction(120000, 1001), Fraction(120), Fraction(240)]
VFR_SHARE = 0.01   # a clip is variable-rate when more than 1 % of its frame intervals are 10 % off the usual one


def frames_as_footage(shot: dict) -> bool:
    """Whether Blender / After Effects should show the engine's own frames rather than the video: a
    variable frame rate, an upright phone video or interlaced fields are read differently by every
    program, while the extracted frames are exactly the solve's frames, one per frame."""
    return bool(shot.get("vfr") or (shot.get("rotation") or 0) % 180 or shot.get("interlaced"))


def frame_timing(path: Path) -> dict:
    """{"vfr", "n_packets", "fps_nominal" (Fraction), "fps_average"} from the video's packet times.
    Phones record a variable frame rate in low light (real iPhone 11 clip: 13.8 % of the intervals
    33 -> 43 ms, 28.97 fps on average); every recorded frame is kept, one frame each, at the nominal
    rate, as editors conform such clips."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    t = np.sort(np.array([float(x) for x in out.split() if x.strip() and x.strip() != "N/A"]))
    if len(t) < 3:
        return {"vfr": False, "n_packets": len(t), "fps_nominal": None, "fps_average": None}
    d = np.diff(t)
    m = float(np.median(d))
    if m <= 0:
        return {"vfr": False, "n_packets": len(t), "fps_nominal": None, "fps_average": None}
    vfr = bool(np.mean(np.abs(d / m - 1) > 0.1) > VFR_SHARE)
    raw = 1.0 / m
    nominal = min(STANDARD_FPS, key=lambda f: abs(float(f) - raw))
    if abs(float(nominal) / raw - 1) > 0.01:  # not near a standard rate: keep what the clip says
        nominal = Fraction(raw).limit_denominator(1001)
    return {"vfr": vfr, "n_packets": len(t), "fps_nominal": nominal,
            "fps_average": (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else None}


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
    if v.get("codec_name") in (None, "unknown", "none"):
        raise ValueError(_raw_hint(path, v.get("codec_tag_string") or ""))
    # the container's numbers can mislead (real Sony Z90 1080i MXF: avg_frame_rate 25/2 for a 25 fps clip;
    # one late frame made an iPhone 17 Pro Max clip 59.971): the measured frame spacing decides below
    rate = v.get("r_frame_rate") if v.get("r_frame_rate", "0/0") != "0/0" else v.get("avg_frame_rate")
    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    tags = {**info.get("format", {}).get("tags", {}), **v.get("tags", {})}
    lens_tags = {k: val for k, val in tags.items() if _FOCAL_KEYS.search(k)}
    timing = frame_timing(path)
    if timing["fps_nominal"]:
        rate = f"{timing['fps_nominal'].numerator}/{timing['fps_nominal'].denominator}"
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": float(Fraction(rate)),
        "fps_fraction": rate,
        "vfr": timing["vfr"],
        "fps_average": timing["fps_average"],
        "interlaced": v.get("field_order") in ("tt", "bb", "tb", "bt"),
        "n_frames": int(v["nb_frames"]) if v.get("nb_frames", "").isdigit() else None,
        "codec": v.get("codec_name"),
        "pix_fmt": v.get("pix_fmt"),
        "rotation": rotation,
        "lens_tags": lens_tags,
        "camera_tags": {k: val for k, val in tags.items() if _CAMERA_KEYS.search(k)},
    }


def _read_image(path: Path) -> np.ndarray:
    """One image of a sequence as RGB uint8 (16-bit and float images are brought to 8 bit)."""
    if path.suffix.lower() in _FFMPEG_IMAGE_EXTS:
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_streams", str(path)],
                                         capture_output=True, text=True, check=True).stdout)["streams"][0]
        w, h = int(info["width"]), int(info["height"])
        pre = ["-apply_trc", "iec61966_2_1"] if path.suffix.lower() == ".exr" else []
        raw = subprocess.run(["ffmpeg", "-v", "error", *pre, "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{path}: cannot read this image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _decode_frames(path: Path, width: int, height: int, deinterlace: bool = False, autorotate: bool = True):
    """Yield RGB uint8 frames via ffmpeg (every decoded frame, no resampling of time). Interlaced video
    (camcorders' 1080i) is deinterlaced to one progressive frame per frame (yadif): tracking combed
    fields would see every moving edge twice."""
    vf = ["-vf", "yadif=0:-1:0"] if deinterlace else []
    pre = [] if autorotate else ["-noautorotate"]
    cmd = ["ffmpeg", "-v", "error", *pre, "-i", str(path), "-map", "0:v:0", *vf, "-fps_mode", "passthrough",
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
    ignore_rotation: bool = False,
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
        first = _read_image(files[0])
        meta = {"width": first.shape[1], "height": first.shape[0], "fps": float(fps or 25.0),
                "fps_fraction": None, "codec": "images", "pix_fmt": None, "rotation": 0, "lens_tags": {},
                "camera_tags": {}}
        source_frames = (_read_image(p) for p in files)
    else:
        meta = probe_video(clip)
        if fps:
            meta["fps"] = float(fps)
        if ignore_rotation:  # a wrong rotation flag (an edit pack's iPhone clip: landscape picture flagged -90)
            meta["rotation_ignored"], meta["rotation"] = meta["rotation"], 0
        if meta["rotation"] % 180:
            # a phone held upright stores the picture sideways with a rotation flag; ffmpeg turns the
            # frames upright when decoding, so the frames (and the solve) are portrait
            meta["width"], meta["height"] = meta["height"], meta["width"]
        source_frames = _decode_frames(clip, meta["width"], meta["height"], deinterlace=meta.get("interlaced", False),
                                       autorotate=not ignore_rotation)
    tel = None
    if not clip.is_dir():
        from ..telemetry import TELEMETRY_FILE, read_telemetry, save
        try:
            tel = read_telemetry(clip)
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
        # variable frame rate (phones): every recorded frame is one frame at the nominal "fps"
        "vfr": bool(meta.get("vfr")), "fps_average": meta.get("fps_average"),
        "interlaced": bool(meta.get("interlaced")),
        "rotation_ignored": meta.get("rotation_ignored"),
        "telemetry": None,
    }
    if tel is not None:
        from ..telemetry import summary
        shot["telemetry"] = summary(tel)
    from ..camera import identify
    shot["camera"] = identify(meta.get("camera_tags", {}), shot["telemetry"], W, H)
    (shot_dir / "shot.json").write_text(json.dumps(shot, indent=1), encoding="utf-8")
    return shot
