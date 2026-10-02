"""Run a research backend in its own env: write job.json, run `python -m niko_<name>`, read result.json."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .paths import env_python


class BackendError(RuntimeError):
    pass


class GpuMemorySampler:
    """Peak whole-GPU memory above the level at start, sampled with nvidia-smi every 0.25 s.

    Approximate: other programs on the same GPU (e.g. Adobe apps) move the baseline. Used for
    backends whose GPU work runs outside torch in the adapter process (COLMAP, MegaSaM steps).
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        self.samples: list[float] = []
        self._stop = threading.Event()

    def _read(self) -> float | None:
        try:
            out = subprocess.run([self.smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return float(out.split()[0])
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline = self._read()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        return False

    @property
    def peak_delta_mb(self) -> float | None:
        if self.baseline is None or not self.samples:
            return None
        return max(0.0, max(self.samples) - self.baseline)


def run_backend(name: str, task: str, shot_dir: str | Path, out_dir: str | Path,
                options: dict | None = None, timeout: float | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {"schema": "niko.job/1", "task": task, "shot_dir": str(Path(shot_dir).resolve()),
           "out_dir": str(out_dir.resolve()), "options": options or {}}
    job_path = out_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    result_path = out_dir / "result.json"
    # A crashed process must never inherit a previous run's success.
    result_path.unlink(missing_ok=True)
    py = env_python(name)
    if not py.exists():
        raise BackendError(f"backend env missing: {py}")
    t0 = time.time()
    # less VRAM fragmentation on long clips; the GPU is shared with other apps
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF",
                                                                   "expandable_segments:True")}
    with (out_dir / "log.txt").open("w", encoding="utf-8") as log, GpuMemorySampler() as gpu:
        proc = subprocess.run([str(py), "-m", f"niko_{name}", str(job_path)], stdout=log,
                              stderr=subprocess.STDOUT, timeout=timeout, cwd=out_dir, env=env)
    if proc.returncode != 0:
        raise BackendError(f"{name}/{task} exited with code {proc.returncode}; see {out_dir / 'log.txt'}")
    if not result_path.exists():
        raise BackendError(f"{name}/{task} wrote no result.json (rc={proc.returncode}); see {out_dir / 'log.txt'}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["wall_s"] = time.time() - t0
    result["gpu_mem_peak_delta_mb"] = gpu.peak_delta_mb
    if not result.get("peak_vram_mb"):  # no torch in the adapter process: use the sampler
        result["peak_vram_mb"] = gpu.peak_delta_mb
        result["peak_vram_source"] = "nvidia-smi delta (approx)"
    if not result.get("ok"):
        raise BackendError(f"{name}/{task} failed: {(result.get('error') or '')[-2000:]}")
    return result
