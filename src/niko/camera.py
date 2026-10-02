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
        zoom = telemetry.get("zoom")
        z = None
        if isinstance(zoom, (list, tuple)) and len(zoom) == 2:
            z = zoom[0] if abs(zoom[1] - zoom[0]) < 1e-3 else None  # a zoom that changes has no single lens
        sensor = telemetry.get("sensor_mm")
        key = f"{telemetry['camera']}|{width}x{height}"
        if sensor:
            key += f"|sensor {sensor[0]:.3f}x{sensor[1]:.3f}"
        key += f"|zoom {z:.2f}" if z is not None else "|zoom varies"
        return {"id": telemetry["camera"], "name": f"{telemetry.get('model') or 'DJI'} ({telemetry['camera']})",
                "key": key, "source": "telemetry", "sensor_mm": sensor, "zoom": z}
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
    ident = " ".join(x for x in (make, model) if x)
    key = f"{ident}|{lens or 'lens ?'}|{width}x{height}"
    return {"id": ident, "name": ident + (f", {lens}" if lens else ""), "key": key, "source": "tags", "lens": lens}


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
