"""Frame rate, variable frame rate, interlacing and upright phone videos at ingest (made with ffmpeg)."""

import subprocess
from fractions import Fraction

import pytest

from niko.pipeline.ingest import frames_as_footage, ingest, probe_video


def _ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("timing")
    src = ["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25", "-t", "2"]
    _ff(*src, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "cfr25.mp4"))
    # 30 fps with every 6th frame 10 ms late: what a phone does in low light (real iPhone 11 clip). The
    # delay needs a 1 ms clock in the filter and the file, or it rounds away to a regular 30 fps
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-t", "3",
        "-vf", r"settb=1/1000,setpts=N*1000/30+if(eq(mod(N\,6)\,5)\,10\,0)", "-fps_mode", "passthrough",
        "-enc_time_base", "1/1000", "-video_track_timescale", "1000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "vfr30.mp4"))
    _ff(*src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-flags", "+ildct+ilme", "-x264opts", "tff=1",
        str(d / "interlaced.mp4"))
    _ff("-display_rotation", "90", "-i", str(d / "cfr25.mp4"), "-c", "copy", str(d / "upright.mp4"))
    return d


def test_constant_rate(clips):
    m = probe_video(clips / "cfr25.mp4")
    assert Fraction(m["fps_fraction"]) == 25 and not m["vfr"] and not m["interlaced"]
    assert not frames_as_footage({"vfr": m["vfr"], "rotation": m["rotation"], "interlaced": m["interlaced"]})


def test_variable_rate_gets_its_nominal_rate(clips):
    m = probe_video(clips / "vfr30.mp4")
    assert m["vfr"] and Fraction(m["fps_fraction"]) == 30
    assert frames_as_footage({"vfr": True})


def test_interlaced_is_detected_and_deinterlaced(clips, tmp_path):
    m = probe_video(clips / "interlaced.mp4")
    assert m["interlaced"] and Fraction(m["fps_fraction"]) == 25
    shot = ingest(clips / "interlaced.mp4", tmp_path / "shot")
    assert shot["interlaced"] and shot["n_frames"] == 50 and shot["fps"] == 25


def test_upright_phone_video_is_portrait(clips, tmp_path):
    shot = ingest(clips / "upright.mp4", tmp_path / "shot")
    assert (shot["width"], shot["height"]) == (180, 320) and shot["rotation"] % 180 == 90
    assert frames_as_footage(shot)


def test_tracker_size_divides_by_four():
    from niko.pipeline.solve import tracker_size
    assert tracker_size(854, 480) == [480, 856]          # GoPro samples
    assert tracker_size(960, 540) == [540, 960]          # unchanged
    for w, h in [(1920, 1080), (608, 1080), (1080, 1350), (750, 422), (5312, 2988)]:
        th, tw = tracker_size(w, h)
        assert th % 4 == 0 and tw % 4 == 0 and th * tw <= 960 * 540 * 1.03


def test_a_wrong_rotation_flag_can_be_ignored(clips, tmp_path):
    shot = ingest(clips / "upright.mp4", tmp_path / "shot", ignore_rotation=True)
    assert (shot["width"], shot["height"]) == (320, 180) and shot["rotation"] == 0
    assert shot["rotation_ignored"] % 180 == 90 and not frames_as_footage(shot)
