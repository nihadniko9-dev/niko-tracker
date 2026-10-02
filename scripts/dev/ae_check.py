"""Round trip through real After Effects: export a solve as .jsx, run it in AE with a probe that
reads back where AE puts every null in comp pixels, and compare with our own projection.

usage (WSL, niko env): python scripts/dev/ae_check.py <solve_dir> [AfterFX.exe path]
The jsx is written to a Windows temp folder; AE is started with -r and quits by itself.
Refuses to start while After Effects has anything but an empty, unchanged "Untitled Project" open:
-r hands the script to that session, and the probe closes its project unsaved and quits (found
2026-10-02 with Nihad's unsaved project open).
The script also checks inside AE and leaves an open project untouched (export_ae.PROBE_GUARD).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.export_ae import export_after_effects
from niko.geometry import project

AE = "/mnt/c/Program Files/Adobe/Adobe After Effects 2026/Support Files/AfterFX.exe"


def winpath(p) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()


def ae_windows() -> list[str] | None:
    """Titles of running After Effects windows ([] = running without a window yet), None = not running."""
    ps = ("$p = @(Get-Process AfterFX -ErrorAction SilentlyContinue); if ($p.Count -eq 0) { 'NONE' } "
          "else { $p | ForEach-Object { 'T:' + $_.MainWindowTitle } }")
    r = subprocess.run(["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True)
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    if "NONE" in lines:
        return None
    return [ln[2:] for ln in lines if ln.startswith("T:") and ln[2:]]


def main():
    titles = ae_windows()
    if titles is not None:  # running: only an empty, unchanged "Untitled Project" may be used (and closed)
        if not titles or any("*" in t or "Untitled Project" not in t for t in titles):
            sys.exit(f"After Effects is open with {titles or 'no window yet'}: save and close it first "
                     "(this check quits AE without saving)")
        print("After Effects is open with an empty, unchanged project: using it; it quits at the end")
    solve_dir = Path(sys.argv[1])
    ae = sys.argv[2] if len(sys.argv) > 2 else AE
    win_temp = subprocess.run(["/mnt/c/Windows/System32/cmd.exe", "/c", "echo %TEMP%"], capture_output=True,
                              text=True, cwd="/mnt/c").stdout.strip()
    work = Path(subprocess.run(["wslpath", "-u", win_temp], capture_output=True, text=True,
                               check=True).stdout.strip()) / "niko_ae_check"
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{solve_dir.name}_probe.json"
    if out.exists():
        out.unlink()
    trk = CameraTrack.load(solve_dir / "selected" / "cameras.json")
    v = np.nonzero(trk.valid)[0]
    frames = sorted({int(v[0]), int(v[len(v) // 3]), int(v[2 * len(v) // 3]), int(v[-1])})
    jsx = work / f"{solve_dir.name}.jsx"
    r = export_after_effects(solve_dir, jsx, windows_path=winpath,
                             probe={"frames": frames, "out": winpath(out).replace("\\", "/")})
    t0 = time.time()
    subprocess.Popen([ae, "-r", winpath(jsx)])
    while not out.exists() and time.time() - t0 < 300:
        time.sleep(2)
    if not out.exists():
        sys.exit("After Effects wrote no probe result within 300 s")
    time.sleep(1)
    res = json.loads(out.read_text())
    if "error" in res:
        sys.exit(f"After Effects: {res['error']}")
    uv_ae = np.array(res["uv"], float)  # [frames, nulls, 2]
    X = r["X_solve"]
    errs = []
    for fi, t in enumerate(res["frames"]):
        K = trk.K[t].copy()
        uv, z = project(K, trk.R[t], trk.t[t], X)  # pinhole: AE has no lens distortion
        e = np.linalg.norm(uv - uv_ae[fi], axis=1)
        errs.append(e)
        print(f"frame {t:4d}: {len(e)} nulls, AE vs niko max {e.max():.6f} px, median {np.median(e):.6f} px")
    e = np.concatenate(errs)
    print(f"footage imported: {res['footage_ok']}; overall max {e.max():.6f} px over {e.size} projections "
          f"in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
