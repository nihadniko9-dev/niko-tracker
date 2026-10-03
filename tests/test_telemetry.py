"""Drone telemetry (niko.telemetry) and the real size it gives (niko.scale.telemetry_scale)."""

import struct

import numpy as np

from niko.camio import CameraTrack
from niko.scale import telemetry_scale
from niko.telemetry import CAM_TO_BODY, NED_TO_ENU, enu, flatten, rdd18_distance, read_sony, true_up, updates


def _varint(v):
    out = b""
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out += bytes([b | 0x80])
        else:
            return out + bytes([b])


def _field(n, wt, payload):
    key = _varint((n << 3) | wt)
    if wt == 2:
        return key + _varint(len(payload)) + payload
    return key + payload


def test_flatten_reads_nested_fields():
    inner = _field(2, 1, struct.pack("<d", 0.648)) + _field(3, 0, _varint(404971))
    msg = _field(1, 2, _field(10, 2, b"DJI Air3s")) + _field(3, 2, _field(4, 2, inner)) + _field(5, 5, struct.pack("<f", 50.0))
    f = flatten(msg)
    assert f["1-10"] == "DJI Air3s"
    assert abs(f["3-4-2"] - 0.648) < 1e-12 and f["3-4-3"] == 404971
    assert f["5"] == 50.0


def test_enu_distances():
    # 0.001 deg of latitude at 37 N is 110.98 m north (meridian arc); 0.001 deg of longitude 88.9 m east
    e = enu([37.0, 37.001, 37.0], [42.0, 42.0, 42.001], [400.0, 400.0, 400.0])
    assert abs(e[1, 1] - 110.98) < 0.05 and abs(e[1, 0]) < 1e-6
    assert abs(e[2, 0] - 88.95) < 0.1
    assert abs(e[1, 2]) < 0.01  # the curvature of the Earth over 111 m is a millimetre


def _flight(n=600, fps=50.0, mpu=12.5, path_m=120.0, climb_m=0.0, gps_noise=0.3, seed=0):
    """A drone flying sideways (and climbing) over `path_m` metres; the solve has it 1/mpu the size.
    Returns the solve's CameraTrack (solve world = ENU / mpu, camera looking north, tilted down 30 deg)
    and telemetry as the DJI stream gives it (GPS and altitude held between 10 Hz / 5 Hz readings)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    E = np.column_stack([path_m * t / t[-1], 0.2 * path_m * np.sin(t / t[-1] * np.pi), climb_m * t / t[-1]])
    trk = CameraTrack.empty("flight", 3840, 2160, fps, 1, n)
    pitch = np.radians(30)
    fwd = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(fwd, right)
    Rc2w = np.column_stack([right, down, fwd])
    for i in range(n):
        trk.R[i] = Rc2w.T
        trk.t[i] = -Rc2w.T @ (E[i] / mpu)
        trk.K[i] = [[2660, 0, 1920], [0, 2660, 1080], [0, 0, 1]]
        trk.dist[i] = 0
    trk.valid[:] = True
    lat0, lon0, h0 = 37.13, 42.69, 405.0
    hold = (np.arange(n) // int(fps / 10)) * int(fps / 10)  # 10 Hz, held between readings
    noisy = E[hold] + rng.normal(0, gps_noise, (n, 3)) * (np.arange(n) == hold)[:, None]
    noisy = noisy[hold]
    lat = lat0 + noisy[:, 1] / 110977.6
    lon = lon0 + noisy[:, 0] / (111319.49 * np.cos(np.radians(lat0)))
    hold5 = (np.arange(n) // int(fps / 5)) * int(fps / 5)
    rel = np.round(E[hold5, 2] + 30.0, 1)  # barometric, 0.1 m steps
    # gimbal: body (FRD) to NED for the same camera attitude
    Rb2ned = NED_TO_ENU.T @ Rc2w @ CAM_TO_BODY.T
    w = np.sqrt(max(1e-12, 1 + np.trace(Rb2ned))) / 2
    q = np.array([w, (Rb2ned[2, 1] - Rb2ned[1, 2]) / (4 * w), (Rb2ned[0, 2] - Rb2ned[2, 0]) / (4 * w),
                  (Rb2ned[1, 0] - Rb2ned[0, 1]) / (4 * w)])
    tel = {"n": n, "lat": lat, "lon": lon, "alt": h0 + noisy[:, 2], "rel_alt": rel,
           "gimbal_q": np.tile(q, (n, 1))}
    return trk, tel


def test_gps_gives_the_real_size():
    trk, tel = _flight(mpu=12.5)
    r = telemetry_scale(trk, tel)
    assert r["reliable"] and r["source"] == "gps"
    assert abs(r["metres_per_unit"] / 12.5 - 1) < 0.01
    assert r["gps"]["fixes"] >= 50 and r["uncertainty_pct"] <= 2.0


def test_short_flight_is_not_trusted():
    trk, tel = _flight(mpu=0.5, path_m=4.0, gps_noise=0.5)
    r = telemetry_scale(trk, tel)
    assert not r["reliable"]


def test_altitude_without_gps():
    trk, tel = _flight(mpu=7.0, path_m=10.0, climb_m=30.0, gps_noise=0.0)
    tel["lat"] = tel["lon"] = None
    r = telemetry_scale(trk, tel)
    assert r["source"] == "altitude" and r["reliable"]
    assert abs(r["metres_per_unit"] / 7.0 - 1) < 0.02


def test_true_up_from_the_gimbal():
    trk, tel = _flight()
    up = true_up(trk, tel)
    assert up is not None and np.degrees(np.arccos(np.clip(up @ [0, 0, 1], -1, 1))) < 0.01
    assert len(updates(tel["lat"])) == 120  # 12 s of GPS at 10 Hz


def test_rdd18_distance():
    assert abs(rdd18_distance(0xC7D0) - 0.2) < 1e-12        # 2000e-4 m: a 200 mm focal length
    assert abs(rdd18_distance(0x00A6) - 166.0) < 1e-9       # exponent 0
    assert abs(rdd18_distance(0xE0A6) - 1.66) < 1e-12       # exponent -2: a 1.66 m focus distance


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def test_sony_camera_from_the_xml_inside_an_mp4(tmp_path):
    xml = (b'<?xml version="1.0" encoding="UTF-8"?>\n<NonRealTimeMeta xmlns="urn:schemas-professionalDisc:'
           b'nonRealTimeMeta:ver.2.00"><Device manufacturer="Sony" modelName="ILCE-7M3" serialNo="1"/>'
           b'<AcquisitionRecord><Group name="CameraUnitMetadataSet"><Item name="CaptureGammaEquation" '
           b'value="s-log3-cine"/></Group></AcquisitionRecord></NonRealTimeMeta>')
    clip = tmp_path / "C0022.MP4"
    clip.write_bytes(_box(b"ftyp", b"XAVC\0\0\0\0") + _box(b"mdat", bytes(64)) + _box(b"moov", bytes(16))
                     + _box(b"meta", bytes(12) + xml))
    tel = read_sony(clip)
    assert tel["camera"] == "Sony ILCE-7M3" and tel["gamma"] == "s-log3-cine" and tel["focal_mm"] is None
    other = tmp_path / "other.mp4"
    other.write_bytes(_box(b"ftyp", b"isom\0\0\0\0") + _box(b"mdat", bytes(64)))
    assert read_sony(other) is None
