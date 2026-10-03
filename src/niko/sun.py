"""Where the sun was: azimuth and elevation from the place (GPS) and the time (UTC) of a recording,
NOAA's general solar position equations (accurate to a fraction of a degree, plenty for lighting).

With the levelled world's up (true gravity from the drone's gimbal) and its north (the GPS track, or
the gimbal's compass), the sun becomes a direction in the Blender scene: a sun lamp there lights 3D
objects like the footage and casts their shadows the same way.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np


def solar_position(lat_deg: float, lon_deg: float, when: datetime) -> tuple[float, float]:
    """(azimuth degrees clockwise from north, elevation degrees above the horizon) at a UTC time."""
    t = when.astimezone(timezone.utc)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + (t.second + t.microsecond / 1e6) / 3600
    g = 2 * math.pi / (366 if t.year % 4 == 0 and (t.year % 100 or t.year % 400 == 0) else 365) * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon_deg          # true solar time, minutes
    ha = math.radians(tst / 4 - 180)                # hour angle
    lat = math.radians(lat_deg)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    elev = 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180
    return az % 360, elev


def sun_vector_enu(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector towards the sun in east, north, up."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def parse_utc(text: str | None) -> datetime | None:
    """A container's creation_time (ISO, "Z") as an aware UTC datetime, or None."""
    if not text:
        return None
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def sun_for_solve(trk, tel: dict | None, utc: datetime | None, up: np.ndarray | None) -> dict | None:
    """The sun in the solve's world: {"direction" (unit, solve world, towards the sun), "azimuth_deg",
    "elevation_deg", "utc", "north" (solve world), "heading_from"}, or None without place, time or north."""
    from .telemetry import enu_rotation

    if tel is None or utc is None or tel.get("lat") is None:
        return None
    lat = tel["lat"][np.isfinite(tel["lat"])]
    lon = tel["lon"][np.isfinite(tel["lon"])]
    if not len(lat):
        return None
    rot = enu_rotation(trk, tel, up)
    if rot is None:
        return None
    R, how = rot
    az, el = solar_position(float(np.median(lat)), float(np.median(lon)), utc)
    d = R.T @ sun_vector_enu(az, el)
    return {"direction": [float(v) for v in d], "azimuth_deg": round(az, 2), "elevation_deg": round(el, 2),
            "utc": utc.isoformat(), "north": [float(v) for v in R.T @ np.array([0.0, 1.0, 0.0])],
            "heading_from": how, "lat": round(float(np.median(lat)), 6), "lon": round(float(np.median(lon)), 6)}
