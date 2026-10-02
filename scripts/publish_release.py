"""Publish a release where other computers can reach it: the add-on zip, the Windows installer and
latest.json, which the add-on's update button reads.

Niko Tracker - author: Nihad Jihad ("Niko").
usage: python scripts/publish_release.py <release folder> [--notes "what changed"]
       python scripts/publish_release.py --github [--notes "what changed"]
  <release folder>: a shared drive, USB stick or cloud folder (Preferences > Niko Tracker folder).
  --github: a GitHub release v<version> of this repo (gh CLI logged in); the add-on's default update
            address reads releases/latest/download/latest.json.
Build first: python scripts/package_addon.py and installer/build.ps1 (same version).
"""

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
    args = sys.argv[1:]
    notes = ""
    if "--notes" in args:
        i = args.index("--notes")
        notes = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        sys.exit(__doc__)
    v = version()
    zip_ = ROOT / "dist" / f"niko_tracker-{v}.zip"
    exe = ROOT / "installer" / "Output" / f"NikoTracker-Setup-{v}.exe"
    missing = [str(p) for p in (zip_, exe) if not p.exists()]
    if missing:
        sys.exit(f"build {v} first, missing: {', '.join(missing)}")
    info = {"schema": "niko.release/1", "version": v, "addon_zip": zip_.name, "installer": exe.name,
            "published": time.strftime("%Y-%m-%dT%H:%M:%S"), "notes": notes}
    if args[0] == "--github":
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest.json"
            latest.write_text(json.dumps(info, indent=1), encoding="utf-8")
            subprocess.run([gh_exe(), "release", "create", f"v{v}", str(zip_), str(exe), str(latest),
                            "--title", f"Niko Tracker {v}", "--notes", notes or f"Niko Tracker {v}"],
                           cwd=ROOT, check=True)
        print(f"published {v} to GitHub")
        return
    dest = Path(args[0])
    dest.mkdir(parents=True, exist_ok=True)
    for f in (zip_, exe):
        shutil.copy2(f, dest / f.name)
    (dest / "latest.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"published {v} to {dest}")


if __name__ == "__main__":
    main()
