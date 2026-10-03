"""Which camera made a clip, and what is known about its lens.

The camera is recognised from the clip itself: a DJI drone's telemetry names the camera module, its
video mode's sensor area and the digital zoom (niko.telemetry); phones and other cameras write a make
and model into the container. A recognised camera can have a profile: its lens measured once from a
calibration clip (`niko calibrate`), then used as the known lens of every later clip from the same
camera and mode. Clips whose camera motion does not measure the lens (a drone flying sideways
without turning, real clip DJI 0079: lenses 12 % apart fit the footage, the gimbal and the GPS
equally well) are then solved with the right lens instead of a guessed one.

Profiles live in $NIKO_HOME/camera_profiles.json, keyed by camera and video mode.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

# container tags that name the camera, most specific first
_MAKE_TAGS = ("com.apple.quicktime.make", "com.android.manufacturer", "make", "Make")
_MODEL_TAGS = ("com.apple.quicktime.model", "com.android.model", "model", "Model")
_LENS_TAGS = ("com.apple.quicktime.camera.lens_model", "lens_model", "LensModel")
PROFILE_FILE = "camera_profiles.json"
PROFILE_MAX_UNCERTAINTY_PCT = 1.5   # a profile is used as the known lens only when measured this well


def identify(tags: dict, telemetry: dict | None, width: int, height: int) -> dict | None:
    """{"id", "name", "key", ...} of the camera that made a clip, or None when nothing says."""
    if telemetry and telemetry.get("camera"):
        src = telemetry.get("source", "dji_djmd")
        sensor = telemetry.get("sensor_mm")
        key = f"{telemetry['camera']}|{width}x{height}"
        z = None
        if src == "dji_djmd":
            zoom = telemetry.get("zoom")
            if isinstance(zoom, (list, tuple)) and len(zoom) == 2:
                z = zoom[0] if abs(zoom[1] - zoom[0]) < 1e-3 else None  # a zoom that changes has no single lens
            if sensor:
                key += f"|sensor {sensor[0]:.3f}x{sensor[1]:.3f}"
            key += f"|zoom {z:.2f}" if z is not None else "|zoom varies"
            model = telemetry.get("model")
            name = model if model == telemetry["camera"] else f"{model or 'DJI'} ({telemetry['camera']})"
        else:
            # GoPro: the lens mode (W, L, S...) and stabilisation change the picture; Sony: the lens
            extra = [x for x in (telemetry.get("lens_mode") and f"lens {telemetry['lens_mode']}",
                                 telemetry.get("stabilised") and "stabilised", telemetry.get("lens_model")) if x]
            key += "".join(f"|{x}" for x in extra)
            name = telemetry["camera"] + (f", {telemetry['lens_model']}" if telemetry.get("lens_model") else "")
        return {"id": telemetry["camera"], "name": name, "key": key, "source": "telemetry",
                "sensor_mm": sensor, "zoom": z}
    low = {k.lower(): v for k, v in (tags or {}).items()}

    def first(names):
        for n in names:
            if n.lower() in low and str(low[n.lower()]).strip():
                return str(low[n.lower()]).strip()
        return None

    make, model, lens = first(_MAKE_TAGS), first(_MODEL_TAGS), first(_LENS_TAGS)
    if not (make or model):
        enc = first(("encoder",))
        if enc and enc.upper().startswith("DJI"):
            make, model = "DJI", enc
    if not (make or model):
        return None
    # "DJI" + "DJI Mini4 Pro" is one name, not "DJI DJI Mini4 Pro"
    ident = model if make and model and model.lower().startswith(make.lower()) else " ".join(x for x in (make, model) if x)
    key = f"{ident}|{lens or 'lens ?'}|{width}x{height}"
    return {"id": ident, "name": ident + (f", {lens}" if lens else ""), "key": key, "source": "tags", "lens": lens}


FULL_FRAME_DIAGONAL_MM = 43.2666


def focus_factor(focal_mm: float | None, focus_m: float | None) -> float:
    """How much longer than its focal length a lens focused closer than infinity behaves for a pinhole
    camera: the lens sits v = (D - sqrt(D^2 - 4 f D)) / 2 from the sensor for a subject D from the sensor
    (thin lens, 1/f = 1/u + 1/v, u + v = D), and the pinhole focal length is v. 1 when unknown."""
    if not focal_mm or not focus_m or not math.isfinite(focus_m):
        return 1.0
    f, d = focal_mm / 1000.0, float(focus_m)
    if d <= 4.2 * f:  # closer than the lens can focus (or macro): no correction
        return 1.0
    return (d - (d * d - 4 * f * d) ** 0.5) / 2 / f


def metadata_focal_px(shot: dict) -> dict | None:
    """The lens a cinema camera records in the clip (Sony RDD 18 metadata: the 35 mm equivalent focal
    length, per frame), in pixels, when it stays the same through the clip: the equivalent is measured on
    the diagonal, so focal_px = f35 x diagonal_px / 43.27, times the focus factor for the clip's median
    focus distance. Real FX6 clips in Super 35: a 24-70 at 24 mm (38.1 mm equivalent, focus 1.35 m):
    3880 x 1.018 = 3951 px, the solve measured 3915 px and every value 3800-4060 px fits equally well;
    a 70-200 at 200 mm (317.4 mm equivalent, focus 2.13 m): 32,321 x 1.117 = 36,100 px; held at each
    value, the footage fits best at 34,000-36,100 px (0.834 px median at 36,100, 0.875 px at 32,321)."""
    tel = shot.get("telemetry") or {}
    f35 = tel.get("focal_35mm")
    if tel.get("source") != "sony" or not f35 or f35[0] <= 0:
        return None
    if f35[1] - f35[0] > 0.005 * f35[0]:  # a zoom during the clip: no single lens
        return None
    w, h = shot["width"], shot["height"]
    f = 0.5 * (f35[0] + f35[1])
    mm = tel.get("focal_mm")
    mm = None if not mm else 0.5 * (mm[0] + mm[1])
    focus = (tel.get("focus_m") or [None, None, None])[1]
    k = focus_factor(mm, focus)
    return {"focal_px": f * (w * w + h * h) ** 0.5 / FULL_FRAME_DIAGONAL_MM * k, "focal_35mm": f,
            "focal_mm": mm, "lens": tel.get("lens_model"), "focus_m": focus, "focus_factor": round(k, 4)}


# cameras whose maker gives the lens (35 mm equivalent focal length, on the full sensor it refers to),
# used only when the footage cannot measure the lens (the camera hardly turns): drones flying straight on
# a steady gimbal. Air 3S wide: Nihad's clips measured 2654-2940 px where they could, the spec gives 2662;
# real DJI 0079 (0.18 deg turn) was solved with 2979 px and a Mini 5 Pro clip (0.18 deg) with 1795 px.
MAKER_LENS = {
    "DJI FC9113": {"name": "DJI Air 3S wide camera", "f35": 24.0, "sensor_mm": (13.107, 9.830), "uncertainty_pct": 10.0},
    "DJI FC9313": {"name": "DJI Mini 5 Pro camera", "f35": 24.0, "sensor_mm": (13.107, 9.830), "uncertainty_pct": 10.0},
}


def maker_focal_px(shot: dict) -> dict | None:
    """The lens from the maker's spec for this camera and video mode, in pixels, or None."""
    cam = shot.get("camera") or {}
    spec = MAKER_LENS.get(cam.get("id") or "")
    tel = shot.get("telemetry") or {}
    area = tel.get("sensor_mm")  # the part of the sensor this video mode uses
    zoom = tel.get("zoom")
    if not spec or not area:
        return None
    z = 1.0
    if zoom:
        if abs(zoom[1] - zoom[0]) > 1e-3:
            return None  # zooming during the clip
        z = float(zoom[0])
    f_mm = spec["f35"] * (spec["sensor_mm"][0] ** 2 + spec["sensor_mm"][1] ** 2) ** 0.5 / FULL_FRAME_DIAGONAL_MM
    return {"focal_px": f_mm * shot["width"] / area[0] * z, "focal_mm": f_mm, "focal_35mm": spec["f35"],
            "name": spec["name"], "uncertainty_pct": spec["uncertainty_pct"], "zoom": z}


def _profiles_path() -> Path:
    return Path(os.environ.get("NIKO_HOME", Path.home() / "niko")) / PROFILE_FILE


def load_profiles() -> dict:
    try:
        return json.loads(_profiles_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def profile_for(camera: dict | None) -> dict | None:
    """The calibrated lens of this camera and mode, when measured well enough to be used."""
    if not camera:
        return None
    p = load_profiles().get(camera["key"])
    if p and p.get("uncertainty_pct", 100.0) <= PROFILE_MAX_UNCERTAINTY_PCT:
        return p
    return None


def save_profile(camera: dict, focal_px: float, width: int, uncertainty_pct: float, how: str) -> dict:
    profiles = load_profiles()
    entry = {"name": camera["name"], "focal_px": float(focal_px), "width": int(width),
             "uncertainty_pct": round(float(uncertainty_pct), 2), "how": how,
             "date": time.strftime("%Y-%m-%d %H:%M")}
    profiles[camera["key"]] = entry
    path = _profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(profiles, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return entry
