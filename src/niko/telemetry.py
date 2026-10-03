"""Per-frame telemetry recorded inside the video file.

DJI drones write a "DJI meta" data stream (codec tag djmd): one protobuf message per video frame with
the camera (model, sensor area of the video mode, digital zoom), the aircraft (GPS, absolute and
take-off-relative altitude, attitude) and the gimbal (angles and an attitude quaternion). The field
numbers were read from the user's DJI Air 3S files (dvtm_Air3s.proto) and checked for sense: GPS in
radians at Zakho (37.14 N, 42.69 E), speed from the GPS track equal to the speed field, gimbal yaw
from the quaternion equal to the yaw angle field, sensor area 13.107 x 7.372 mm = the 16:9 part of
the camera's 1-inch IMX989.

No schema is needed to read the stream: the protobuf wire format is decoded generically and the known
field paths are picked out; unknown DJI models fall back to a search for the GPS pair.
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

# field paths (message numbers joined by "-") per DJI proto file
# The DJI cameras seen so far share one layout (Air 3S dvtm_Air3s, Mini 5 Pro dvtm_Mini5Pro, Osmo Pocket 3):
# the paths below. They differ in the GPS unit: radians on the Air 3S, degrees on the Mini 5 Pro
# (37.2467, 43.2939), told apart by size (a latitude in radians is at most pi / 2).
DJI_DEFAULT = "dvtm_Air3s.proto"
DJI_FIELDS = {
    "dvtm_Air3s.proto": {
        "proto": "1-1-1", "model": "1-1-10", "camera": "2-2-1-4", "width": "2-3-1", "height": "2-3-2",
        "fps": "2-3-3", "sensor_w_um": "3-2-38-1", "sensor_h_um": "3-2-38-2", "zoom": "3-2-23-1",
        "lat_rad": "3-3-4-1-2", "lon_rad": "3-3-4-1-3", "alt_mm": "3-3-4-2", "rel_alt_mm": "3-3-5-1",
        "drone_pitch_d10": "3-3-3-1", "drone_roll_d10": "3-3-3-2", "drone_yaw_d10": "3-3-3-3",
        "gimbal_pitch_d10": "3-4-3-1", "gimbal_roll_d10": "3-4-3-2", "gimbal_yaw_d10": "3-4-3-3",
        "gimbal_qw": "3-4-4-1", "gimbal_qx": "3-4-4-2", "gimbal_qy": "3-4-4-3", "gimbal_qz": "3-4-4-4",
    },
}


def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def _parse(b: bytes) -> list | None:
    """[(field, wire type, raw value)] of one protobuf message, or None if b is not one."""
    out, i = [], 0
    try:
        while i < len(b):
            key, i = _varint(b, i)
            f, wt = key >> 3, key & 7
            if f == 0:
                return None
            if wt == 0:
                v, i = _varint(b, i)
            elif wt in (1, 5):
                n = 8 if wt == 1 else 4
                v, i = b[i:i + n], i + n
            elif wt == 2:
                n, i = _varint(b, i)
                v, i = b[i:i + n], i + n
            else:
                return None
            if i > len(b):
                return None
            out.append((f, wt, v))
    except IndexError:
        return None
    return out


def flatten(b: bytes, path: str = "", out: dict | None = None) -> dict:
    """{field path: value} of a protobuf message: varints as int, 64-bit as double, 32-bit as float,
    nested messages recursed into, other bytes as text when ASCII. The first value of a path wins."""
    out = {} if out is None else out
    for f, wt, v in _parse(b) or []:
        p = f"{path}-{f}" if path else str(f)
        if wt == 0:
            val = v - (1 << 64) if v >= 1 << 63 else v
        elif wt == 1:
            val = struct.unpack("<d", v)[0]
        elif wt == 5:
            val = struct.unpack("<f", v)[0]
        else:
            sub = _parse(v) if v else None
            if sub:
                flatten(v, p, out)
                continue
            try:
                val = v.decode("ascii")
            except UnicodeDecodeError:
                continue
        out.setdefault(p, val)
    return out


def _ffprobe(args: list[str]) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-of", "json", *args], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def dji_stream(video: str | Path) -> int | None:
    """Index of the DJI meta (djmd) stream, or None."""
    for s in _ffprobe(["-show_streams", str(video)]).get("streams", []):
        if s.get("codec_type") == "data" and s.get("codec_tag_string") == "djmd":
            return int(s["index"])
    return None


def _guess_gps(rows: list[dict]) -> tuple[str, str] | None:
    """For a DJI model without a field map: the sibling pair of doubles that look like latitude and
    longitude in radians and move a little (a drone, not noise)."""
    cands = {}
    for p, v in rows[0].items():
        if isinstance(v, float) and -np.pi / 2 < v < np.pi / 2 and v != 0:
            parent, _, last = p.rpartition("-")
            cands.setdefault(parent, []).append(p)
    for parent, ps in cands.items():
        ps = sorted(ps)
        for a, b in zip(ps, ps[1:]):
            la = np.array([r.get(a, np.nan) for r in rows], float)
            lo = np.array([r.get(b, np.nan) for r in rows], float)
            if np.all(np.abs(la[~np.isnan(la)]) < np.pi / 2) and np.nanstd(la) < 1e-3 and np.nanstd(lo) < 1e-3:
                return a, b
    return None


def read_dji(video: str | Path) -> dict | None:
    """Telemetry of a DJI video, one entry per video frame (packet), or None if there is none.
    Angles in degrees, altitudes and sensor in metres / millimetres, GPS in degrees."""
    video = Path(video)
    idx = dji_stream(video)
    if idx is None:
        return None
    sizes = [int(p["size"]) for p in _ffprobe(["-select_streams", str(idx), "-show_entries", "packet=size",
                                                str(video)])["packets"]]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:{idx}", "-c", "copy", "-f", "data", "-"],
                         capture_output=True, check=True).stdout
    rows, off = [], 0
    for n in sizes:
        rows.append(flatten(raw[off:off + n]))
        off += n
    head = rows[0]
    proto = head.get("1-1-1", "")
    fmap = DJI_FIELDS.get(proto)
    how = "field map"
    if fmap is None and any(p in rows[0] for p in ("3-3-4-1-2", "3-4-4-1", "3-2-38-1")):
        fmap, how = DJI_FIELDS[DJI_DEFAULT], "DJI common layout"
    if fmap is None:
        g = _guess_gps(rows)
        fmap = {"proto": "1-1-1", "model": "1-1-10"}
        if g:
            fmap.update({"lat_rad": g[0], "lon_rad": g[1]})
        how = "guessed GPS" if g else "unknown fields"

    def series(key, scale=1.0):
        p = fmap.get(key)
        if p is None:
            return None
        a = np.array([r.get(p, np.nan) if not isinstance(r.get(p), str) else np.nan for r in rows], float)
        return None if np.all(np.isnan(a)) else a * scale

    def first(key):
        p = fmap.get(key)
        for r in rows:
            if p in r:
                return r[p]
        return None

    out = {"source": "dji_djmd", "proto": proto, "fields": how, "n": len(rows),
           "model": first("model"), "camera": first("camera") or first("model"),  # Pocket 3 names no module
           "width": first("width"), "height": first("height"), "fps": first("fps")}
    sw, sh = first("sensor_w_um"), first("sensor_h_um")
    out["sensor_mm"] = [sw / 1000.0, sh / 1000.0] if sw and sh else None
    for key, name, scale in (("zoom", "zoom", 1.0), ("lat_rad", "lat", 180 / np.pi), ("lon_rad", "lon", 180 / np.pi),
                             ("alt_mm", "alt", 1e-3), ("rel_alt_mm", "rel_alt", 1e-3),
                             ("drone_pitch_d10", "drone_pitch", 0.1), ("drone_roll_d10", "drone_roll", 0.1),
                             ("drone_yaw_d10", "drone_yaw", 0.1), ("gimbal_pitch_d10", "gimbal_pitch", 0.1),
                             ("gimbal_roll_d10", "gimbal_roll", 0.1), ("gimbal_yaw_d10", "gimbal_yaw", 0.1)):
        out[name] = series(key, scale)
    if out["gimbal_pitch"] is not None and out["gimbal_roll"] is None:
        out["gimbal_roll"] = np.zeros(len(rows))  # a zero roll is left out of the message
    q = [series(k) for k in ("gimbal_qw", "gimbal_qx", "gimbal_qy", "gimbal_qz")]
    out["gimbal_q"] = np.column_stack(q) if all(x is not None for x in q) else None
    if out["lat"] is not None:
        raw_lat = series("lat_rad")
        raw_lon = series("lon_rad")
        if np.nanmax(np.abs(raw_lat)) > np.pi / 2 or np.nanmax(np.abs(raw_lon)) > np.pi:  # degrees already
            out["lat"], out["lon"] = raw_lat, raw_lon
    if out["lat"] is not None:  # no fix: zeros
        bad = (np.abs(out["lat"]) < 1e-9) & (np.abs(out["lon"]) < 1e-9)
        out["lat"][bad] = np.nan
        out["lon"][bad] = np.nan
        if np.all(np.isnan(out["lat"])):
            out["lat"] = out["lon"] = None
    return out


# ---------------------------------------------------------------- GoPro (GPMF)
# GoPro cameras write telemetry as GPMF (KLV: four-character key, type, struct size, repeat; big
# endian; nested when the type is 0) in a "GoPro MET" data stream, one payload per second, and the
# camera's settings in a GPMF box in the file's udta (model MINF, firmware FMWR, lens mode VFOV
# W/L/S/N/H, field of view ZFOV in degrees on HERO8 and later, stabilisation EISE). Read from GoPro's
# own sample files (github.com/gopro/gpmf-parser): HERO5 to HERO8 and the Karma drone.
_GPMF_TYPES = {"b": ("b", 1), "B": ("B", 1), "d": ("d", 8), "f": ("f", 4), "j": ("q", 8), "J": ("Q", 8),
               "l": ("l", 4), "L": ("L", 4), "s": ("h", 2), "S": ("H", 2)}
GOPRO_MAX_DOP = 5.0   # GPS fixes with a dilution of precision above this (GPSP / 100) are not used


def gpmf(b: bytes) -> list:
    """[(key, type, values or nested list)] of a GPMF payload."""
    out, i = [], 0
    while i + 8 <= len(b):
        key = b[i:i + 4].decode("latin1")
        typ, size = b[i + 4], b[i + 5]
        rep = struct.unpack(">H", b[i + 6:i + 8])[0]
        n = size * rep
        data = b[i + 8:i + 8 + n]
        i += 8 + ((n + 3) // 4) * 4
        if typ == 0:
            out.append((key, None, gpmf(data)))
            continue
        t = chr(typ)
        if t in ("c", "U"):
            val = data.decode("latin1").rstrip("\x00")
        elif t in _GPMF_TYPES:
            f, w = _GPMF_TYPES[t]
            val = struct.unpack(">" + f * (n // w), data[:(n // w) * w])
        else:
            val = data
        out.append((key, t, val))
    return out


def _udta_settings(video: Path) -> dict:
    """The camera's settings from the GPMF box in the MP4's udta (empty when there is none)."""
    try:
        with open(video, "rb") as fh:
            head = fh.read(8 << 20)
            fh.seek(max(0, Path(video).stat().st_size - (8 << 20)))
            tail = fh.read()
    except OSError:
        return {}
    for blob in (head, tail):
        i = blob.find(b"udta")
        j = blob.find(b"GPMF", i) if i >= 0 else -1
        if j > 4:
            size = struct.unpack(">I", blob[j - 4:j])[0]
            out = {}

            def walk(items):
                for k, t, v in items:
                    if t is None:
                        walk(v)
                    else:
                        out.setdefault(k, v[0] if isinstance(v, tuple) and len(v) == 1 else v)
            walk(gpmf(blob[j + 4:j - 4 + size]))
            return out
    return {}


def read_gopro(video: str | Path) -> dict | None:
    """Telemetry of a GoPro video (same layout as read_dji: one entry per video frame), or None."""
    video = Path(video)
    streams = _ffprobe(["-show_streams", str(video)]).get("streams", [])
    gs = [x for x in streams if x.get("codec_type") == "data" and x.get("codec_tag_string") == "gpmd"]
    vs = [x for x in streams if x.get("codec_type") == "video"]
    if not gs or not vs:
        return None
    idx = int(gs[0]["index"])
    v = vs[0]
    fps = float(Fraction(v.get("avg_frame_rate") or v.get("r_frame_rate")))
    n_frames = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None
    pk = _ffprobe(["-select_streams", str(idx), "-show_entries", "packet=pts_time,duration_time,size",
                   str(video)])["packets"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:{idx}", "-c", "copy", "-f", "data", "-"],
                         capture_output=True, check=True).stdout
    gps_t, gps, grav_t, grav, names = [], [], [], [], []
    off = 0
    for p in pk:
        n = int(p["size"])
        t0 = float(p.get("pts_time") or 0.0)
        dur = float(p.get("duration_time") or 1.0)
        for dev_key, _, dev in gpmf(raw[off:off + n]):
            if dev_key != "DEVC":
                continue
            for k, t, val in dev:
                if k == "DVNM":
                    names.append(val)
                if k != "STRM":
                    continue
                d = {kk: vv for kk, _, vv in val}
                scal = d.get("SCAL", (1,))
                for key in ("GPS5", "GPS9"):
                    if key in d:
                        w = 5 if key == "GPS5" else 9
                        sc = np.array(scal if len(scal) == w else [scal[0]] * w, float)
                        vals = np.array(d[key], float).reshape(-1, w) / sc
                        fix = d.get("GPSF", (vals[:, 8].astype(int)[0] if key == "GPS9" else 0,))[0]
                        dop = (d.get("GPSP", (9999,))[0] / 100.0) if key == "GPS5" else float(np.median(vals[:, 7]))
                        good = fix >= 3 and dop <= GOPRO_MAX_DOP
                        times = t0 + dur * np.arange(len(vals)) / max(len(vals), 1)
                        for tt, row in zip(times, vals):
                            gps_t.append(tt)
                            gps.append(row[:3] if good else (np.nan, np.nan, np.nan))
                if "GRAV" in d:
                    vals = np.array(d["GRAV"], float).reshape(-1, 3) / float(scal[0])
                    vals[np.linalg.norm(vals, axis=1) < 0.5] = np.nan  # zeros while the sensor starts
                    times = t0 + dur * np.arange(len(vals)) / max(len(vals), 1)
                    grav_t.extend(times)
                    grav.extend(vals)
        off += n
    settings = _udta_settings(video)
    model = settings.get("MINF") or next((x for x in names if x and x not in ("Camera", "Global Settings")), None)
    fw = settings.get("FMWR") or (_ffprobe(["-show_format", str(video)]).get("format", {}).get("tags", {}) or {}).get("firmware")
    if not model and fw:
        # the firmware names the model: HD5.02.02.00.00 is a HERO5, H23.01... (from 2021: H21 = HERO10) a HERO12
        m = re.match(r"HD(\d+)\.", fw)
        m2 = re.match(r"H(2\d)\.", fw)
        model = f"HERO{m.group(1)}" if m else f"HERO{int(m2.group(1)) - 11}" if m2 else f"GoPro ({fw})"
    if n_frames is None:
        n_frames = int(round(float(v.get("duration", 0)) * fps))
    ft = np.arange(n_frames) / fps

    def hold(times, rows, width):
        out = np.full((n_frames, width), np.nan)
        if not len(times):
            return out
        times = np.asarray(times)
        rows = np.asarray(rows, float)
        j = np.searchsorted(times, ft + 1e-9, side="right") - 1
        ok = j >= 0
        out[ok] = rows[j[ok]]
        return out

    g = hold(gps_t, gps, 3)
    zoom = settings.get("DZST")
    out = {"source": "gopro_gpmf", "proto": "GPMF", "fields": "GPMF", "n": n_frames,
           "model": model, "camera": (f"GoPro {model}" if model and not model.startswith("GoPro") else model),
           "width": int(v["width"]), "height": int(v["height"]), "fps": fps, "sensor_mm": None,
           "lens_mode": settings.get("VFOV"), "fov_deg": settings.get("ZFOV"),
           "stabilised": settings.get("EISE") == "Y", "firmware": fw,
           "zoom": None, "lat": g[:, 0], "lon": g[:, 1], "alt": g[:, 2], "rel_alt": None,
           "gimbal_q": None, "gravity_cam": hold(grav_t, grav, 3) if grav else None}
    for k in ("drone_pitch", "drone_roll", "drone_yaw", "gimbal_pitch", "gimbal_roll", "gimbal_yaw"):
        out[k] = None
    if np.all(np.isnan(out["lat"])):
        out["lat"] = out["lon"] = out["alt"] = None
    if zoom not in (None, 0):
        out["zoom_setting"] = zoom
    return out


# ---------------------------------------------------------------- Sony (XML sidecar + MXF metadata)
# Sony cinema cameras (FX6 and its family) write <clip>M01.XML next to the clip (camera model, lens
# model, capture gamma) and, in every MXF content package, a data element with SMPTE RDD 18
# acquisition metadata: the lens set (key ...0c02010101010000; 8005 actual focal length, 8004 35 mm
# equivalent, 8001 focus distance, 8000 iris) and the camera set (...0c02010102010000; 8104 / 8105
# the imager's effective width / height in micrometres, 8106 capture frame rate). Distances are
# 16-bit: a signed 4-bit power of ten and a 12-bit mantissa, in metres (FX6 clip with a 70-200 mm at
# 200 mm: 0xC7D0 = 2000e-4 m).
_SONY_DATA_KEY = bytes.fromhex("060e2b34010201010d01030117010201")
_RDD18_LENS = bytes.fromhex("060e2b34025301010c02010101010000")
_RDD18_CAMERA = bytes.fromhex("060e2b34025301010c02010102010000")


def rdd18_distance(v: int) -> float:
    """RDD 18 16-bit distance (metres): signed 4-bit exponent of ten, 12-bit mantissa."""
    e = v >> 12
    e = e - 16 if e >= 8 else e
    return (v & 0x0FFF) * 10.0 ** e


def _ber(b: bytes, i: int) -> tuple[int, int]:
    n = b[i]
    if n < 0x80:
        return n, i + 1
    k = n & 0x7F
    return int.from_bytes(b[i + 1:i + 1 + k], "big"), i + 1 + k


def _rdd18_sets(b: bytes) -> dict:
    """{set key: {tag: raw bytes}} of the RDD 18 sets inside one frame's metadata element."""
    out, pos = {}, 0
    while True:
        i = b.find(b"\x06\x0e\x2b\x34\x02\x53", pos)
        if i < 0 or i + 17 > len(b):
            return out
        key = b[i:i + 16]
        n, j = _ber(b, i + 16)
        val, k, items = b[j:j + n], 0, {}
        while k + 4 <= len(val):
            tag, ln = struct.unpack(">HH", val[k:k + 4])
            items[tag] = val[k + 4:k + 4 + ln]
            k += 4 + ln
        out.setdefault(key, items)
        pos = j + max(n, 1)


def _mxf_frames(path: Path, max_frames: int = 100000) -> list[bytes]:
    """The per-frame metadata elements of an MXF (top-level KLV walk; essence is skipped, not read)."""
    out = []
    with open(path, "rb") as f:
        while len(out) < max_frames:
            key = f.read(16)
            if len(key) < 16:
                break
            lb = f.read(1)
            if not lb:
                break
            n = lb[0]
            if n >= 0x80:
                n = int.from_bytes(f.read(n & 0x7F), "big")
            if key == _SONY_DATA_KEY:
                out.append(f.read(n))
            else:
                f.seek(n, 1)
    return out


def _mp4_sony_xml(video: Path) -> bytes | None:
    """The NonRealTimeMeta XML a Sony camera writes into its MP4 (a top-level meta box), or None."""
    try:
        size = video.stat().st_size
        with open(video, "rb") as f:
            pos = 0
            while pos + 8 <= size:
                f.seek(pos)
                h = f.read(16)
                n, kind = struct.unpack(">I4s", h[:8])
                if n == 1 and len(h) == 16:
                    n = struct.unpack(">Q", h[8:16])[0]
                elif n == 0:
                    n = size - pos
                if n < 8:
                    return None
                if kind in (b"meta", b"uuid") and n < 1 << 20:
                    f.seek(pos)
                    b = f.read(n)
                    i, j = b.find(b"<NonRealTimeMeta"), b.find(b"</NonRealTimeMeta>")
                    if 0 <= i < j:
                        return b[i:j + len(b"</NonRealTimeMeta>")]
                pos += n
    except (OSError, struct.error):
        return None
    return None


def _sony_xml(video: Path) -> dict:
    import xml.etree.ElementTree as ET

    xml = video.with_name(video.stem + "M01.XML")
    if not xml.exists():
        xml = video.with_name(video.stem + "M01.xml")
    try:
        if xml.exists():
            root = ET.parse(xml).getroot()
        elif video.suffix.lower() in (".mp4", ".mov"):
            text = _mp4_sony_xml(video)
            if not text:
                return {}
            root = ET.fromstring(text)
        else:
            return {}
    except (ET.ParseError, OSError):
        return {}
    out = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "Device":
            out["manufacturer"], out["model"] = el.get("manufacturer"), el.get("modelName")
        elif tag == "Lens":
            out["lens"] = el.get("modelName")
        elif tag == "Item" and el.get("name") == "CaptureGammaEquation":
            out["gamma"] = el.get("value")
        elif tag == "VideoFrame":
            out["capture_fps"] = el.get("captureFps")
    return out


def read_sony(video: str | Path) -> dict | None:
    """Camera and per-frame lens data of a Sony clip (XML sidecar or the XML inside an MP4, MXF RDD 18
    metadata), or None."""
    video = Path(video)
    meta = _sony_xml(video)
    frames = _mxf_frames(video) if video.suffix.lower() == ".mxf" else []
    if not frames and (meta.get("manufacturer") or "").lower() != "sony":
        return None
    focal, eq35, focus = [], [], []
    sensor = None
    for b in frames:
        sets = _rdd18_sets(b)
        lens = sets.get(_RDD18_LENS, {})
        cam = sets.get(_RDD18_CAMERA, {})

        def dist(tag):
            v = lens.get(tag)
            return rdd18_distance(struct.unpack(">H", v)[0]) if v and len(v) == 2 and v != b"\xff\xff" else np.nan
        focal.append(dist(0x8005) * 1000.0)
        eq35.append(dist(0x8004) * 1000.0)
        focus.append(dist(0x8001))
        if sensor is None and 0x8104 in cam and 0x8105 in cam:
            w, h = struct.unpack(">H", cam[0x8104])[0], struct.unpack(">H", cam[0x8105])[0]
            sensor = [w / 1000.0, h / 1000.0]
    v = (_ffprobe(["-show_streams", "-select_streams", "v:0", str(video)]).get("streams") or [{}])[0]
    n = len(frames) or (int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else 0)
    model = meta.get("model")
    out = {"source": "sony", "proto": "RDD 18" if frames else "XML", "fields": "RDD 18", "n": n,
           "model": model, "camera": f"Sony {model}" if model else "Sony", "width": v.get("width"),
           "height": v.get("height"), "fps": None, "sensor_mm": sensor, "lens_model": meta.get("lens"),
           "gamma": meta.get("gamma"), "zoom": None, "lat": None, "lon": None, "alt": None, "rel_alt": None,
           "gimbal_q": None, "gravity_cam": None,
           "focal_mm": np.array(focal) if frames and np.isfinite(focal).any() else None,
           "focal_35mm": np.array(eq35) if frames and np.isfinite(eq35).any() else None,
           "focus_m": np.array(focus) if frames and np.isfinite(focus).any() else None}
    for k in ("drone_pitch", "drone_roll", "drone_yaw", "gimbal_pitch", "gimbal_roll", "gimbal_yaw"):
        out[k] = None
    return out


def read_telemetry(video: str | Path) -> dict | None:
    """The per-frame telemetry a video carries (DJI or GoPro), or None."""
    return read_dji(video) or read_gopro(video) or read_sony(video)


# WGS84
_A, _F = 6378137.0, 1 / 298.257223563
_E2 = _F * (2 - _F)


def ecef(lat_deg, lon_deg, h):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.column_stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                            (n * (1 - _E2) + h) * np.sin(lat)])


def enu(lat_deg, lon_deg, h, origin=None) -> np.ndarray:
    """East, north, up in metres from `origin` (lat, lon, h; default the first point)."""
    lat_deg, lon_deg, h = (np.atleast_1d(np.asarray(x, float)) for x in (lat_deg, lon_deg, h))
    o = origin if origin is not None else (lat_deg[0], lon_deg[0], h[0])
    lat0, lon0 = np.radians(o[0]), np.radians(o[1])
    d = ecef(lat_deg, lon_deg, h) - ecef([o[0]], [o[1]], [o[2]])
    R = np.array([[-np.sin(lon0), np.cos(lon0), 0],
                  [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
                  [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)]])
    return d @ R.T


def updates(a: np.ndarray) -> np.ndarray:
    """Indices where a sampled-and-held series takes a new value (the moments a reading arrived)."""
    a = np.asarray(a, float)
    ok = ~np.isnan(a)
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return idx
    keep = [idx[0]] + [j for i, j in zip(idx, idx[1:]) if a[j] != a[i]]
    return np.array(keep)


# camera axes (OpenCV: x right, y down, z forward) in DJI body axes (x forward, y right, z down)
CAM_TO_BODY = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
NED_TO_ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)


def quat_matrix(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def gimbal_rotations(tel: dict) -> np.ndarray | None:
    """Camera-to-ENU rotation per frame from the gimbal quaternion (body to NED, checked against the
    gimbal's yaw / pitch fields on the Air 3S), or None."""
    q = tel.get("gimbal_q")
    if q is None:
        return None
    return np.array([NED_TO_ENU @ quat_matrix(x) @ CAM_TO_BODY if np.all(np.isfinite(x)) else np.full((3, 3), np.nan)
                     for x in q])


def align_rotation(trk, Rg: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """(Ra, median deg, 90th percentile deg): the one rotation taking the solve's world to ENU that
    best matches the solve's cameras to the gimbal's, and how far the cameras still differ."""
    n = min(trk.n_frames, len(Rg))
    v = np.array([i for i in range(n) if trk.valid[i] and np.all(np.isfinite(Rg[i]))])
    if len(v) < 3:
        return None
    U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", Rg[v], trk.R_c2w[v]))
    Ra = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    rel = np.einsum("nji,njk->nik", np.einsum("ij,njk->nik", Ra, trk.R_c2w[v]), Rg[v])
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return Ra, float(np.median(ang)), float(np.percentile(ang, 90))


TELEMETRY_FILE = "telemetry.npz"
_ARRAYS = ("zoom", "lat", "lon", "alt", "rel_alt", "drone_pitch", "drone_roll", "drone_yaw", "gimbal_pitch",
           "gimbal_roll", "gimbal_yaw", "gimbal_q", "gravity_cam", "focal_mm", "focal_35mm", "focus_m")
_INFO = ("source", "proto", "fields", "n", "model", "camera", "width", "height", "fps", "sensor_mm",
         "lens_mode", "fov_deg", "stabilised", "firmware", "lens_model", "gamma")


def summary(tel: dict) -> dict:
    """What shot.json records about the telemetry."""
    info = {k: tel.get(k) for k in _INFO}
    info.update({"gps": tel.get("lat") is not None, "altitude": tel.get("rel_alt") is not None,
                 "gimbal": tel.get("gimbal_q") is not None, "gravity": tel.get("gravity_cam") is not None})
    for k in ("focal_mm", "focal_35mm"):
        a = tel.get(k)
        if a is not None and np.isfinite(a).any():
            info[k] = [round(float(np.nanmin(a)), 2), round(float(np.nanmax(a)), 2)]
    a = tel.get("focus_m")
    if a is not None and np.isfinite(a).any():
        info["focus_m"] = [round(float(np.nanmin(a)), 3), round(float(np.nanmedian(a)), 3),
                           round(float(np.nanmax(a)), 3)]
    if tel.get("zoom") is not None and np.isfinite(tel["zoom"]).any():
        info["zoom"] = [float(np.nanmin(tel["zoom"])), float(np.nanmax(tel["zoom"]))]
    return info


def save(path: str | Path, tel: dict) -> None:
    arrays = {k: np.asarray(tel[k], float) for k in _ARRAYS if tel.get(k) is not None}
    np.savez_compressed(path, info=json.dumps({k: tel.get(k) for k in _INFO}), **arrays)


def load(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    tel = json.loads(str(z["info"]))
    for k in _ARRAYS:
        tel[k] = z[k] if k in z.files else None
    return tel


def for_solve(solve_dir: str | Path) -> dict | None:
    """The telemetry of a solve: its telemetry.npz, else read again from the source video (solves made
    before 0.5 did not keep it), else None."""
    solve_dir = Path(solve_dir)
    tel = load(solve_dir / TELEMETRY_FILE)
    if tel is not None:
        return tel
    try:
        src = json.loads((solve_dir / "shot.json").read_text())["source"]
        return read_telemetry(src) if Path(src).is_file() else None
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        return None


def true_up(trk, tel: dict | None, max_median_deg: float = 1.0) -> np.ndarray | None:
    """Up (against gravity) in the solve's world from the gimbal, when the solve's cameras match the
    gimbal's to within max_median_deg after one fixed rotation; else None."""
    if tel is None:
        return None
    Rg = gimbal_rotations(tel)
    if Rg is None:
        return None
    al = align_rotation(trk, Rg)
    if al is None or al[1] > max_median_deg:
        return None
    return al[0].T @ np.array([0.0, 0.0, 1.0])


def enu_rotation(trk, tel: dict | None, up: np.ndarray | None) -> tuple[np.ndarray, str] | None:
    """(R, how): the rotation taking the solve's world to east-north-up, or None. Up is `up` (the
    gimbal's gravity); north comes from the GPS track when it spans 20 m or more (the camera path's
    horizontal shape fitted to it), else from the gimbal's compass. On Nihad's Air 3S clips the
    compass heading was about 3 deg off the GPS track's."""
    if tel is None:
        return None
    Rg = gimbal_rotations(tel)
    al = align_rotation(trk, Rg) if Rg is not None else None
    if up is None and al is not None and al[1] < 1.0:
        up = al[0].T @ np.array([0.0, 0.0, 1.0])
    if up is None:
        return None
    z = np.asarray(up, float) / np.linalg.norm(up)
    x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    B = np.stack([x, y, z])  # solve world -> a frame with z up
    n = min(trk.n_frames, tel.get("n", 0))
    if tel.get("lat") is not None and tel.get("alt") is not None:
        u = updates(tel["lat"][:n])
        u = u[trk.valid[u] & np.isfinite(tel["alt"][u])]
        if len(u) >= 10:
            E = enu(tel["lat"][u], tel["lon"][u], tel["alt"][u])[:, :2]
            if np.linalg.norm(np.ptp(E, 0)) >= 20.0:
                P = (trk.centers[u] @ B.T)[:, :2]
                Pc, Ec = P - P.mean(0), E - E.mean(0)
                h = (Pc.T @ Ec)
                th = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])  # best 2D rotation P -> E
                c, s = np.cos(th), np.sin(th)
                return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ B, "gps"
    if al is not None:
        return al[0], "compass"
    return None


def recorded_utc(video: str | Path, tel: dict | None = None):
    """When the clip was recorded (UTC), from the container's creation_time (DJI writes UTC there:
    real clip 0079 says 20:49 UTC, 23:49 local time, and is a night shot), or None."""
    from .sun import parse_utc

    try:
        tags = _ffprobe(["-show_format", str(video)]).get("format", {}).get("tags", {}) or {}
    except (subprocess.CalledProcessError, OSError, ValueError):
        return None
    return parse_utc(tags.get("creation_time"))
