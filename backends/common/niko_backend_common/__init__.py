"""Job / result protocol shared by every backend env (docs/SCHEMA.md, "Backend job protocol").

A backend entry point looks like:

    def main():
        job = read_job(sys.argv[1])
        with ResultWriter(job) as res:
            ...                                   # do the work, write files into job["out_dir"]
            res.outputs["cameras"] = "cameras.json"
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


def read_job(path: str | os.PathLike) -> dict:
    job = json.loads(Path(path).read_text(encoding="utf-8"))
    if job.get("schema") != "niko.job/1":
        raise ValueError(f"{path}: not a niko.job/1 file")
    Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
    return job


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def torch_versions() -> dict:
    torch = _torch()
    v = {"python": platform.python_version()}
    if torch is not None:
        v["torch"] = torch.__version__
        v["torch_cuda"] = torch.version.cuda
    return v


class ResultWriter:
    """Times the work, records peak VRAM, and always writes out_dir/result.json."""

    def __init__(self, job: dict, versions: dict | None = None):
        self.job = job
        self.out_dir = Path(job["out_dir"])
        self.outputs: dict = {}
        self.stats: dict = {}
        self.versions = {**torch_versions(), **(versions or {})}

    def __enter__(self):
        torch = _torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        runtime = time.perf_counter() - self.t0
        torch = _torch()
        peak = None
        if torch is not None and torch.cuda.is_available():
            peak = torch.cuda.max_memory_reserved() / 2**20
        result = {
            "schema": "niko.result/1",
            "ok": exc is None,
            "error": None if exc is None else "".join(traceback.format_exception(exc_type, exc, tb)),
            "runtime_s": runtime,
            "peak_vram_mb": peak,
            "versions": self.versions,
            "outputs": self.outputs,
            "stats": self.stats,
        }
        (self.out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        if exc is not None:
            print(result["error"], file=sys.stderr)
            sys.exit(1)
        return False


def probe_torch_cuda() -> dict:
    """Checks every GPU backend runs before its own probe: CUDA build, sm_120 kernels, a real op."""
    torch = _torch()
    if torch is None:
        return {"ok": False, "error": "torch not installed"}
    info = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        return {**info, "ok": False, "error": "torch.cuda.is_available() is False"}
    cap = torch.cuda.get_device_capability(0)
    info["device"] = torch.cuda.get_device_name(0)
    info["capability"] = f"sm_{cap[0]}{cap[1]}"
    arches = torch.cuda.get_arch_list()
    info["sm_120_kernels"] = "sm_120" in arches
    a = torch.randn(512, 512, device="cuda")
    info["matmul_ok"] = bool(torch.allclose((a @ a.T).cpu(), a.cpu() @ a.cpu().T, atol=1e-2))
    info["ok"] = info["sm_120_kernels"] and info["matmul_ok"]
    return info


def save_camera_npz(path, K, dist, R, t, valid, meta: dict) -> None:
    """Raw camera candidate (docs/SCHEMA.md "camera_raw.npz"); the orchestrator writes cameras.json.

    K [T,3,3] full-resolution pixels, corner origin; dist [T,5] OpenCV order; R, t world->camera
    (OpenCV axes); valid [T] bool. Invalid frames may hold anything.
    """
    import numpy as np

    np.savez(path, K=np.asarray(K, np.float64), dist=np.asarray(dist, np.float64),
             R=np.asarray(R, np.float64), t=np.asarray(t, np.float64),
             valid=np.asarray(valid, bool), meta=np.array(json.dumps(meta)))


def write_ply(path, xyz, rgb=None) -> None:
    """Binary little-endian PLY with x, y, z (float32) and red, green, blue (uint8)."""
    import numpy as np

    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    rgb = np.full((len(xyz), 3), 200, np.uint8) if rgb is None else np.asarray(rgb, np.uint8).reshape(-1, 3)
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz.T
    data["red"], data["green"], data["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def print_probe(info: dict) -> None:
    """Probes end with one JSON line; `niko doctor` reads the last line."""
    print(json.dumps(info))
    sys.exit(0 if info.get("ok") else 1)
