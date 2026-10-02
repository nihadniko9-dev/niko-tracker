"""Recognising the camera of a clip, camera profiles, and the lens check of a camera that hardly turns."""

import numpy as np

from niko import camera
from niko.camio import CameraTrack
from niko.pipeline.solve import lens_check


def test_identify_dji_from_telemetry():
    tel = {"camera": "DJI FC9113", "model": "DJI Air3s", "sensor_mm": [13.107, 7.372], "zoom": [1.0, 1.0]}
    c = camera.identify({"encoder": "DJI Air3s"}, tel, 3840, 2160)
    assert c["id"] == "DJI FC9113" and c["source"] == "telemetry"
    assert c["key"] == "DJI FC9113|3840x2160|sensor 13.107x7.372|zoom 1.00"
    tel["zoom"] = [1.0, 2.0]  # a zoom that changes has no single lens
    assert camera.identify({}, tel, 3840, 2160)["key"].endswith("zoom varies")


def test_identify_from_tags_and_unknown():
    tags = {"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 15 Pro",
            "com.apple.quicktime.camera.lens_model": "iPhone 15 Pro back triple camera 6.765mm f/1.78"}
    c = camera.identify(tags, None, 3840, 2160)
    assert c["id"] == "Apple iPhone 15 Pro" and "6.765mm" in c["key"] and c["key"].endswith("3840x2160")
    assert camera.identify({"encoder": "DJI Mini4 Pro"}, None, 1920, 1080)["id"] == "DJI DJI Mini4 Pro"
    assert camera.identify({"encoder": "Lavf60.16.100"}, None, 1920, 1080) is None
    assert camera.identify({}, None, 1920, 1080) is None


def test_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("NIKO_HOME", str(tmp_path))
    cam = {"key": "X|1920x1080", "name": "X"}
    assert camera.profile_for(cam) is None
    camera.save_profile(cam, 1500.0, 1920, 4.0, "a short turn")
    assert camera.profile_for(cam) is None  # measured, but not well enough to be used
    camera.save_profile(cam, 1490.0, 1920, 0.6, "a full circle")
    p = camera.profile_for(cam)
    assert p["focal_px"] == 1490.0 and p["how"] == "a full circle"
    assert camera.profile_for({"key": "other", "name": "Y"}) is None


def _sliding(n=40, turn_deg=0.0):
    trk = CameraTrack.empty("slide", 1920, 1080, 25.0, 1, n)
    for i in range(n):
        a = np.radians(turn_deg * i / (n - 1))
        trk.R[i] = [[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]]
        trk.t[i] = [-0.1 * i, 0, 0]
        trk.K[i] = [[1500, 0, 960], [0, 1500, 540], [0, 0, 1]]
        trk.dist[i] = 0
    trk.valid[:] = True
    return trk


def test_lens_check_flags_a_camera_that_hardly_turns():
    lc = lens_check({"a": _sliding(turn_deg=0.2)}, "a")
    assert lc["uncertain"] and "hardly_turns" in lc["reasons"] and lc["max_turn_deg"] < 0.25
    lc = lens_check({"a": _sliding(turn_deg=5.0)}, "a")
    assert lc is None or "hardly_turns" not in lc["reasons"]
