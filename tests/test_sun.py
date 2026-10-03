"""Sun position (niko.sun) and the scene's north from GPS (niko.telemetry.enu_rotation)."""

from datetime import datetime, timezone

import numpy as np

from niko.sun import parse_utc, solar_position, sun_for_solve, sun_vector_enu
from niko.telemetry import enu_rotation

from test_telemetry import _flight


def test_solar_position_known_cases():
    # Greenwich, midsummer noon: elevation 90 - 51.48 + 23.44 = 61.96, just before solar noon
    az, el = solar_position(51.4769, 0.0, datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc))
    assert abs(el - 61.96) < 0.2 and abs(az - 179.5) < 1.0
    # equator at the March equinox, solar noon at Greenwich: nearly overhead
    az, el = solar_position(0.0, 0.0, datetime(2025, 3, 20, 12, 7, tzinfo=timezone.utc))
    assert el > 89.0
    # an afternoon sun is in the west, a night sun below the horizon (real clip 0079: 23:49 local time)
    az, el = solar_position(37.14, 42.68, datetime(2025, 3, 3, 13, 27, tzinfo=timezone.utc))
    assert 200 < az < 280 and 10 < el < 25
    assert solar_position(37.14, 42.69, datetime(2025, 3, 3, 20, 49, tzinfo=timezone.utc))[1] < -50


def test_sun_vector_and_time():
    v = sun_vector_enu(90.0, 0.0)
    assert np.allclose(v, [1, 0, 0], atol=1e-12)
    assert np.allclose(sun_vector_enu(0.0, 90.0), [0, 0, 1], atol=1e-12)
    t = parse_utc("2025-03-03T20:49:14.000000Z")
    assert t.hour == 20 and t.utcoffset().total_seconds() == 0
    assert parse_utc(None) is None and parse_utc("not a date") is None


def test_north_from_the_gps_track():
    # the synthetic flight's solve world is east-north-up / scale: the rotation must be identity
    trk, tel = _flight(mpu=12.5)
    R, how = enu_rotation(trk, tel, np.array([0.0, 0.0, 1.0]))
    assert how == "gps" and np.allclose(R, np.eye(3), atol=0.01)
    # turn the solve's world 40 deg about up: north must turn back with it
    a = np.radians(40)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    for i in range(trk.n_frames):
        trk.R[i] = trk.R[i] @ Rz.T
    R2, _ = enu_rotation(trk, tel, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(R2 @ Rz, np.eye(3), atol=0.01)
    s = sun_for_solve(trk, tel, datetime(2025, 3, 3, 13, 27, tzinfo=timezone.utc), np.array([0.0, 0.0, 1.0]))
    d_enu = R2 @ np.array(s["direction"])
    assert np.allclose(d_enu, sun_vector_enu(s["azimuth_deg"], s["elevation_deg"]), atol=0.02)
