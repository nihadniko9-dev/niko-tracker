"""`niko doctor`: is this machine ready? GPU, CUDA, every backend env, every checkpoint."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .registry import BACKENDS, CHECKPOINTS

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def engine_image() -> bool:
    """An installed engine image (installer/engine): no CUDA toolkit or Linux Blender inside, by design."""
    return (Path(os.environ.get("NIKO_REPO", "")) / "ENGINE_VERSION").is_file()


def _dev_only(ok: bool) -> str:
    """FAIL on a development machine; on an engine image only a warning."""
    return OK if ok else (WARN if engine_image() else FAIL)


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str


def _run(cmd: list[str], timeout: float = 60, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"


def check_system() -> list[Check]:
    out = []
    version = Path("/proc/version").read_text() if Path("/proc/version").exists() else ""
    wsl = "microsoft" in version.lower()
    out.append(Check("system", "WSL2", OK if wsl else FAIL, version.split(" (")[0] or "not Linux"))

    osr = Path("/etc/os-release")
    pretty = ""
    if osr.exists():
        m = re.search(r'PRETTY_NAME="([^"]+)"', osr.read_text())
        pretty = m.group(1) if m else ""
    out.append(Check("system", "Ubuntu 24.04", OK if "24.04" in pretty else FAIL, pretty or "unknown"))

    home = paths.niko_home()
    resolved = str(home.resolve()) if home.exists() else str(home)
    on_linux_fs = not resolved.startswith("/mnt/")
    if not home.exists():
        out.append(Check("system", "NIKO_HOME", FAIL, f"{home} does not exist"))
    else:
        free = shutil.disk_usage(home).free / 1e9
        st = OK if on_linux_fs and free > 100 else (WARN if on_linux_fs else FAIL)
        out.append(Check("system", "NIKO_HOME", st,
                         f"{home} ({'ext4' if on_linux_fs else 'Windows mount!'}), {free:.0f} GB free"))

    arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    out.append(Check("system", "TORCH_CUDA_ARCH_LIST", _dev_only(arch == "12.0"),
                     arch or ("unset (only to build backends)" if engine_image() else "unset")))
    # only needed to download gated checkpoints; once they are on disk it may be revoked
    out.append(Check("system", "HF_TOKEN", OK if os.environ.get("HF_TOKEN") else WARN,
                     "set (value not shown)" if os.environ.get("HF_TOKEN") else "unset (only needed for downloads)"))
    for tool in ("uv", "ffmpeg", "ffprobe", "git"):
        p = shutil.which(tool)
        out.append(Check("system", tool, OK if p else FAIL, p or "not on PATH"))
    return out


def check_gpu() -> list[Check]:
    out = []
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    rc, txt = _run([smi, "--query-gpu=name,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader"])
    if rc == 0 and txt:
        name, driver, mem, cap = [s.strip() for s in txt.splitlines()[0].split(",")]
        out.append(Check("gpu", "nvidia-smi", OK if cap == "12.0" else WARN,
                         f"{name}, driver {driver}, {mem}, sm_{cap.replace('.', '')}"))
    else:
        out.append(Check("gpu", "nvidia-smi", FAIL, txt[:200]))

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = shutil.which("nvcc") or str(Path(cuda_home) / "bin" / "nvcc")
    rc, txt = _run([nvcc, "--version"])
    m = re.search(r"release (\d+)\.(\d+)", txt)
    if rc == 0 and m:
        ver = (int(m.group(1)), int(m.group(2)))
        out.append(Check("gpu", "nvcc", OK if ver >= (12, 8) else FAIL, f"CUDA {ver[0]}.{ver[1]} at {nvcc}"))
    else:
        out.append(Check("gpu", "nvcc", _dev_only(False),
                         "not in the engine image (only to build backends)" if engine_image()
                         else (txt[:200] or "not found")))
    return out


def check_backends() -> list[Check]:
    out = []
    for name, b in BACKENDS.items():
        py = paths.env_python(name)
        if not py.exists():
            out.append(Check("backend", name, FAIL, f"env missing: {paths.env_dir(name)}"))
            continue
        # probes print one JSON line on stdout; libraries log freely on stderr
        try:
            p = subprocess.run([str(py), "-m", f"niko_{name}.probe"], capture_output=True, text=True, timeout=600)
            rc, stdout, stderr = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            rc, stdout, stderr = 124, "", "timeout after 600s"
        info = None
        for line in reversed(stdout.splitlines()):
            if line.startswith("{"):
                try:
                    info = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
        if info is None:
            out.append(Check("backend", name, FAIL, f"probe failed (rc={rc}): {(stdout + stderr)[-300:]}"))
            continue
        status = OK if rc == 0 and info.get("ok") else FAIL
        detail = ", ".join(f"{k}={v}" for k, v in info.items() if k not in ("ok",))
        out.append(Check("backend", name, status, detail))
    return out


def check_checkpoints() -> list[Check]:
    out = []
    root = paths.checkpoints_dir()
    for ck in CHECKPOINTS:
        d = root / ck.key
        missing = [f for f in ck.files if not (d / f).is_file()]
        if missing:
            out.append(Check("checkpoint", ck.key, FAIL, f"missing {', '.join(missing)} in {d}"))
            continue
        size = sum((d / f).stat().st_size for f in ck.files)
        if ck.size_bytes and abs(size - ck.size_bytes) > 0.02 * ck.size_bytes:
            out.append(Check("checkpoint", ck.key, FAIL,
                             f"size {size / 1e9:.2f} GB, expected ~{ck.size_bytes / 1e9:.2f} GB"))
        else:
            out.append(Check("checkpoint", ck.key, OK, f"{size / 1e9:.2f} GB, {ck.license}"))
    return out


def check_blender() -> list[Check]:
    exe = paths.find_blender(allow_windows_from_wsl=False)
    win = Path(paths.WSL_WINDOWS_BLENDER)
    out = []
    if exe is None:
        out.append(Check("blender", "Linux Blender 5.2", _dev_only(False),
                         "not in the engine image (only to render benchmark shots)" if engine_image()
                         else "not found under $NIKO_HOME/blender"))
    else:
        rc, txt = _run([exe, "--version"])
        first = txt.splitlines()[0] if txt else ""
        out.append(Check("blender", "Linux Blender 5.2", OK if "5.2" in first else FAIL, f"{first} ({exe})"))
    out.append(Check("blender", "Windows fallback", OK if win.is_file() else WARN, str(win)))
    return out


def run_doctor(as_json: bool = False) -> int:
    checks = check_system() + check_gpu() + check_backends() + check_checkpoints() + check_blender()
    if as_json:
        print(json.dumps([c.__dict__ for c in checks], indent=1))
    else:
        width = max(len(c.name) for c in checks)
        group = None
        for c in checks:
            if c.group != group:
                group = c.group
                print(f"\n{group}")
            print(f"  [{c.status:>4}] {c.name:<{width}}  {c.detail}")
        n_fail = sum(c.status == FAIL for c in checks)
        print(f"\n{'ALL GREEN' if n_fail == 0 else f'{n_fail} check(s) failed'}")
    return 0 if all(c.status != FAIL for c in checks) else 1
