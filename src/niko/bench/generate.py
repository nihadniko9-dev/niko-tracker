"""Render synthetic shots with Blender and write their exact ground truth as gt/cameras.json.

Blender renders an undistorted pinhole image into render/. When the spec asks for lens
distortion, the render has an overscan border (same focal length in pixels, larger canvas)
and every output pixel is looked up through the exact inverse of the OpenCV model, so the
ground truth is (K, dist) of the final frames. Sensor noise is added last. Both steps are
checked against projected markers in tests/test_render_markers.py.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..blendercam import load_blender_cameras
from ..paths import REPO, find_blender
from ..triangulate import undistort_points

GEN_SCRIPT = REPO / "blender" / "gen_shot.py"


def _host_path(p: Path, blender: str) -> str:
    """A Windows blender.exe started from WSL needs Windows paths."""
    if sys.platform != "win32" and blender.endswith(".exe"):
        return subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()
    return str(p)


def nominal_K(spec: dict) -> np.ndarray:
    cam = spec["camera"]
    f = cam.get("lens", 28.0) * spec["width"] / cam.get("sensor_width", 36.0)
    return np.array([[f, 0, spec["width"] / 2], [0, f, spec["height"] / 2], [0, 0, 1.0]])


def dist_vector(spec: dict) -> np.ndarray:
    d = spec.get("distortion") or {}
    return np.array([d.get("k1", 0.0), d.get("k2", 0.0), d.get("p1", 0.0), d.get("p2", 0.0), d.get("k3", 0.0)])


def overscan(spec: dict) -> tuple[int, int]:
    """Pixels to add on each side so every distorted output pixel samples inside the render."""
    dist = dist_vector(spec)
    if not np.any(dist):
        return 0, 0
    if "lens_end" in spec["camera"]:
        raise ValueError("zoom together with lens distortion is not supported by the generator")
    W, H = spec["width"], spec["height"]
    s = np.linspace(0, 1, 200)
    border = np.concatenate([np.stack([s * W, np.zeros_like(s)], 1), np.stack([s * W, np.full_like(s, H)], 1),
                             np.stack([np.zeros_like(s), s * H], 1), np.stack([np.full_like(s, W), s * H], 1)])
    und = undistort_points(border, nominal_K(spec), dist, iters=50)
    mx = max(0.0, -und[:, 0].min(), und[:, 0].max() - W)
    my = max(0.0, -und[:, 1].min(), und[:, 1].max() - H)
    return int(math.ceil(mx)) + 8, int(math.ceil(my)) + 8


def undistort_maps(K: np.ndarray, dist: np.ndarray, W: int, H: int, mx: int, my: int):
    """cv2.remap maps: output (distorted) pixel -> position in the overscan render."""
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    und = undistort_points(pts, K, dist, iters=50)
    # corner-origin render coordinates -> cv2 pixel-centre indices
    map_x = (und[:, 0] + mx - 0.5).reshape(H, W).astype(np.float32)
    map_y = (und[:, 1] + my - 0.5).reshape(H, W).astype(np.float32)
    return map_x, map_y


def add_noise(img: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Read noise (sigma in 8-bit levels) + signal-dependent shot noise, per channel."""
    x = img.astype(np.float32)
    var = cfg.get("read", 2.0) ** 2 + cfg.get("shot", 0.0) * x
    x += rng.normal(size=x.shape).astype(np.float32) * np.sqrt(var)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def generate_shot(spec: dict | str | Path, out_dir: str | Path, blender: str | None = None,
                  force: bool = False) -> dict:
    """Render one shot into out_dir (frames/, gt/cameras.json, scene.json). Skips finished shots."""
    if not isinstance(spec, dict):
        spec = json.loads(Path(spec).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    done = out_dir / "gt" / "cameras.json"
    if done.exists() and not force:
        old = json.loads((out_dir / "spec.json").read_text())
        if old == spec:
            info = json.loads((out_dir / "scene.json").read_text())
            info["skipped"] = True
            return info
    for d in ("render", "frames", "gt"):
        if (out_dir / d).exists():
            shutil.rmtree(out_dir / d)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    blender = blender or find_blender()
    if blender is None:
        raise RuntimeError("Blender 5.2 not found (set NIKO_BLENDER)")

    W, H = spec["width"], spec["height"]
    mx, my = overscan(spec)
    run_spec = dict(spec)
    if mx or my:
        run_spec["render"] = {"width": W + 2 * mx, "height": H + 2 * my, "lens_factor": W / (W + 2 * mx)}
    spec_path = out_dir / "blender_spec.json"
    spec_path.write_text(json.dumps(run_spec, indent=1), encoding="utf-8")
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", _host_path(GEN_SCRIPT, blender), "--",
           _host_path(spec_path, blender), _host_path(out_dir, blender)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "blender.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Blender failed (rc={proc.returncode}), see {out_dir / 'blender.log'}")
    info = json.loads((out_dir / "scene.json").read_text(encoding="utf-8"))

    # ground truth for the final frames
    trk = load_blender_cameras(out_dir / "blender_cameras.json", method="gt", name=spec["name"])
    trk.K[:, 0, 2] -= mx
    trk.K[:, 1, 2] -= my
    trk.width, trk.height = W, H
    dist = dist_vector(spec)
    trk.dist[:] = dist
    trk.world_up, trk.units = "+z", "meters"
    if info.get("rolling_shutter"):
        trk.rolling_shutter = {"readout": info["rolling_shutter"]["readout"], "direction": "top_to_bottom"}
    trk.extra.update({"scene_size": info["scene_size"], "median_depth": info["median_depth"],
                      "render_device": info["device"], "motion_blur_shutter": info["motion_blur_shutter"],
                      "noise": spec.get("noise"), "movers": len(info.get("movers", []))})

    # frames: distortion (exact inverse model) then sensor noise
    renders = sorted((out_dir / "render").glob("*.png"))
    if len(renders) != spec["n_frames"]:
        raise RuntimeError(f"expected {spec['n_frames']} renders, found {len(renders)}")
    frames = out_dir / "frames"
    frames.mkdir()
    maps = undistort_maps(trk.K[0], dist, W, H, mx, my) if np.any(dist) else None
    noise = spec.get("noise")
    for i, f in enumerate(renders):
        if maps is None and not noise:
            f.rename(frames / f.name)
            continue
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if maps is not None:
            img = cv2.remap(img, maps[0], maps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if noise:
            img = add_noise(img, noise, np.random.default_rng(spec.get("seed", 0) * 100003 + i))
        cv2.imwrite(str(frames / f.name), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    shutil.rmtree(out_dir / "render")

    trk.save(done)
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    info["overscan_px"] = [mx, my]
    info["wall_seconds"] = time.time() - t0
    (out_dir / "scene.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info
