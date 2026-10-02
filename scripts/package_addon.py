"""Zip the Blender add-on for installing on another computer (Blender: Edit > Preferences >
Add-ons > Install from Disk).  usage: python scripts/package_addon.py  ->  dist/niko_tracker-<version>.zip"""

import ast
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "addon" / "niko_tracker"


def version() -> str:
    tree = ast.parse((SRC / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "bl_info" for t in node.targets):
            return ".".join(map(str, ast.literal_eval(node.value)["version"]))
    raise RuntimeError("no bl_info")


def main():
    out = ROOT / "dist" / f"niko_tracker-{version()}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SRC.glob("*.py")):
            z.write(f, f"niko_tracker/{f.name}")
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
