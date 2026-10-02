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
import struct
import subprocess
from pathlib import Path

import numpy as np

# field paths (message numbers joined by "-") per DJI proto file
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
           "model": first("model"), "camera": first("camera"),
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
    if out["lat"] is not None:  # no fix: zeros
        bad = (np.abs(out["lat"]) < 1e-9) & (np.abs(out["lon"]) < 1e-9)
        out["lat"][bad] = np.nan
        out["lon"][bad] = np.nan
        if np.all(np.isnan(out["lat"])):
            out["lat"] = out["lon"] = None
    return out


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
           "gimbal_roll", "gimbal_yaw", "gimbal_q")
_INFO = ("source", "proto", "fields", "n", "model", "camera", "width", "height", "fps", "sensor_mm")


def summary(tel: dict) -> dict:
    """What shot.json records about the telemetry."""
    info = {k: tel.get(k) for k in _INFO}
    info.update({"gps": tel.get("lat") is not None, "altitude": tel.get("rel_alt") is not None,
                 "gimbal": tel.get("gimbal_q") is not None})
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
        return read_dji(src) if Path(src).is_file() else None
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
