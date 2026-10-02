"""Publish a release where other computers can reach it: the add-on zip, the Windows installer, the
engine code and latest.json, which the add-on's update button reads; with --engine also the engine
image parts (installer/engine/build_engine.ps1).

Niko Tracker - author: Nihad Jihad ("Niko").
usage: python scripts/publish_release.py (<release folder> | --github) [--notes "what changed"]
                                         [--engine | --engine-image <version>] [--engine-dir D:\\NikoEngine]
  <release folder>  a shared drive, USB stick or cloud folder (Preferences > Niko Tracker folder)
  --github          a GitHub release v<version> of this repo (gh CLI logged in); the add-on's default
                    update address reads releases/latest/download/latest.json
  --engine          this release carries a new engine image (its parts are uploaded too)
  --engine-image    otherwise: the newest engine image the code runs on (installed engines older than
                    that are told to run the new setup)
Build first: python scripts/package_addon.py and installer/build.ps1 (same version).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_addon import version  # noqa: E402


def gh_exe() -> str:
    found = shutil.which("gh")
    if found:
        return found
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "gh" / "bin" / "gh.exe"
    if local.exists():
        return str(local)
    sys.exit("GitHub CLI (gh) not found")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--github", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--engine-image")
    ap.add_argument("--engine-dir", default=r"D:\NikoEngine")
    a = ap.parse_args()
    if bool(a.folder) == a.github:
        ap.error("give a release folder or --github")
    v = version()
    zip_ = ROOT / "dist" / f"niko_tracker-{v}.zip"
    exe = ROOT / "installer" / "Output" / f"NikoTracker-Setup-{v}.exe"
    missing = [str(p) for p in (zip_, exe) if not p.exists()]
    parts = []
    if a.engine:
        pj = Path(a.engine_dir) / f"niko-engine-{v}.parts.json"
        if not pj.exists():
            sys.exit(f"no engine image {v}: run installer/engine/build_engine.ps1 first ({pj})")
        parts = [Path(a.engine_dir) / p["name"] for p in json.loads(pj.read_text())["parts"]]
        missing += [str(p) for p in parts if not p.exists()]
    if missing:
        sys.exit(f"build {v} first, missing: {', '.join(missing)}")
    image = v if a.engine else a.engine_image
    if a.github and not image:
        sys.exit("say which engine image this code runs on: --engine (new image) or --engine-image <version>")
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / f"niko-engine-code-{v}.tar.gz"
        subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(code), "HEAD"], cwd=ROOT, check=True)
        info = {"schema": "niko.release/1", "version": v, "addon_zip": zip_.name, "installer": exe.name,
                "engine_code": code.name, "engine_image": image,
                "published": time.strftime("%Y-%m-%dT%H:%M:%S"), "notes": a.notes}
        latest = Path(tmp) / "latest.json"
        latest.write_text(json.dumps(info, indent=1), encoding="utf-8")
        files = [zip_, exe, code, latest, *parts]
        if a.github:
            subprocess.run([gh_exe(), "release", "create", f"v{v}", *map(str, files),
                            "--title", f"Niko Tracker {v}", "--notes", a.notes or f"Niko Tracker {v}"],
                           cwd=ROOT, check=True)
            print(f"published {v} to GitHub ({len(files)} files)")
            return
        dest = Path(a.folder)
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
        print(f"published {v} to {dest} ({len(files)} files)")


if __name__ == "__main__":
    main()
