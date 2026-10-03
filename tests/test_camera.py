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
    assert camera.identify({"encoder": "DJI Mini4 Pro"}, None, 1920, 1080)["id"] == "DJI Mini4 Pro"
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


def test_batch_finds_videos_in_folders_and_files(tmp_path):
    from niko.cli import batch_clips
    (tmp_path / "a.MP4").write_bytes(b"x")
    (tmp_path / "b.mov").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.mxf").write_bytes(b"x")
    found = batch_clips([str(tmp_path), str(tmp_path / "sub" / "c.mxf")])
    assert [p.name for p in found] == ["a.MP4", "b.mov", "c.mxf"]


def test_focus_factor():
    assert camera.focus_factor(200.0, None) == 1.0 and camera.focus_factor(None, 2.0) == 1.0
    assert abs(camera.focus_factor(50.0, 1e6) - 1.0) < 1e-4                # infinity
    assert abs(camera.focus_factor(200.0, 2.134) - 1.1168) < 1e-3
    assert camera.focus_factor(200.0, 0.5) == 1.0                           # closer than it can focus


def test_maker_lens_for_dji_cameras():
    shot = {"width": 3840, "height": 2160, "camera": {"id": "DJI FC9113"},
            "telemetry": {"sensor_mm": [13.107, 7.372], "zoom": [1.0, 1.0]}}
    mk = camera.maker_focal_px(shot)
    assert abs(mk["focal_px"] - 2662.6) < 1.0 and mk["focal_35mm"] == 24.0
    shot["telemetry"]["zoom"] = [2.0, 2.0]
    assert abs(camera.maker_focal_px(shot)["focal_px"] - 5325.1) < 2.0     # digital zoom 2x
    shot["telemetry"]["zoom"] = [1.0, 2.0]
    assert camera.maker_focal_px(shot) is None                             # zooming
    assert camera.maker_focal_px({"width": 3840, "height": 2160, "camera": {"id": "Apple iPhone 15 Pro"}}) is None


def test_metadata_lens_from_sony_35mm_equivalent():
    shot = {"width": 3840, "height": 2160,
            "telemetry": {"source": "sony", "focal_35mm": [38.1, 38.1], "focal_mm": [24.0, 24.0],
                          "lens_model": "FE 24-70mm F2.8 GM II"}}
    md = camera.metadata_focal_px(shot)
    assert abs(md["focal_px"] - 3879.7) < 0.5 and md["focal_mm"] == 24.0   # focus unknown: as recorded
    shot["telemetry"]["focus_m"] = [1.339, 1.354, 2.181]
    md = camera.metadata_focal_px(shot)
    assert abs(md["focal_px"] - 3951) < 1 and md["focus_m"] == 1.354       # the solve measured 3915 px
    shot["telemetry"].update(focal_35mm=[317.4, 317.4], focal_mm=[200.0, 200.0], focus_m=[1.663, 2.134, 2.204])
    assert abs(camera.metadata_focal_px(shot)["focal_px"] - 36100) < 5     # footage best at 34,000-36,100
    shot["telemetry"]["focal_35mm"] = [38.1, 76.2]                          # zooming: no single lens
    assert camera.metadata_focal_px(shot) is None
    assert camera.metadata_focal_px({"width": 1920, "height": 1080, "telemetry": {"source": "dji_djmd"}}) is None
