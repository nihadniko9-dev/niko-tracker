import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from niko.pipeline.ingest import ingest

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _frames(n, w, h):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return [np.roll(base, 3 * i, axis=1) for i in range(n)]


def test_image_folder(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i, f in enumerate(_frames(5, 640, 360)):
        cv2.imwrite(str(src / f"f{i:04d}.png"), f)
    shot = ingest(src, tmp_path / "shot", fps=24.0, frame_start=1001)
    assert shot["n_frames"] == 5 and shot["width"] == 640 and shot["height"] == 360
    assert shot["fps"] == 24.0 and shot["frame_start"] == 1001
    assert shot["proxy"]["scale_x"] == 1.0  # smaller than 1080p: proxy is not upscaled
    assert len(list((tmp_path / "shot" / "frames").glob("*.jpg"))) == 5
    back = cv2.imread(str(tmp_path / "shot" / "frames" / "000002.jpg"))
    ref = cv2.imread(str(src / "f0002.png"))
    assert np.abs(back.astype(int) - ref.astype(int)).mean() < 3.0  # JPG q95
    assert json.loads((tmp_path / "shot" / "shot.json").read_text())["schema"] == "niko.shot/1"


@needs_ffmpeg
def test_video_and_proxy(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i, f in enumerate(_frames(12, 2560, 1440)):
        cv2.imwrite(str(src / f"{i:04d}.png"), f)
    mp4 = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "30000/1001", "-i", str(src / "%04d.png"),
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    shot = ingest(mp4, tmp_path / "shot")
    assert shot["n_frames"] == 12
    assert (shot["width"], shot["height"]) == (2560, 1440)
    assert shot["fps"] == pytest.approx(29.97, abs=1e-3)
    assert (shot["proxy"]["width"], shot["proxy"]["height"]) == (1920, 1080)
    proxy = cv2.imread(str(tmp_path / "shot" / "proxy" / "000011.jpg"))
    assert proxy.shape == (1080, 1920, 3)
    full = cv2.imread(str(tmp_path / "shot" / "frames" / "000011.jpg"))
    ref = cv2.imread(str(src / "0011.png"))
    assert np.abs(full.astype(int) - ref.astype(int)).mean() < 4.0
