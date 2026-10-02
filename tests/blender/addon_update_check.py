"""Background check of the add-on's update button from a release folder (no UI).

blender.exe -b --factory-startup --python tests/blender/addon_update_check.py -- <scratch folder>
Copies the add-on into <scratch>/addons/niko_tracker (never updates the repo's own files), makes a
release folder with a newer version (latest.json + zip, as scripts/publish_release.py writes),
points the preferences at it, runs niko.update and checks the files and the version were replaced.
"""

import json
import os
import shutil
import sys
import zipfile

import addon_utils
import bpy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
work = os.path.normpath(sys.argv[sys.argv.index("--") + 1])
addons = os.path.join(work, "addons")
here = os.path.join(addons, "niko_tracker")
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(os.path.join(REPO, "addon", "niko_tracker"), here,
                ignore=shutil.ignore_patterns("__pycache__"))
sys.path.insert(0, addons)
addon_utils.enable("niko_tracker", default_set=True)
from niko_tracker import ops  # noqa: E402

cur = ops._version_of(os.path.join(here, "__init__.py"))
newer = (cur[0], cur[1], cur[2] + 1)
v = ".".join(map(str, newer))
rel = os.path.join(work, "release")
os.makedirs(rel)
with zipfile.ZipFile(os.path.join(rel, f"niko_tracker-{v}.zip"), "w") as z:
    for f in sorted(os.listdir(here)):
        if f.endswith(".py"):
            text = open(os.path.join(here, f), encoding="utf-8").read()
            if f == "__init__.py":
                text = text.replace(f'"version": {tuple(cur)}', f'"version": {tuple(newer)}')
            if f == "ui.py":
                text += "\n# NIKO_UPDATE_MARKER\n"
            z.writestr(f"niko_tracker/{f}", text)
json.dump({"version": v, "addon_zip": f"niko_tracker-{v}.zip"}, open(os.path.join(rel, "latest.json"), "w"))

bpy.context.preferences.addons["niko_tracker"].preferences.repo_dir = rel
res = bpy.ops.niko.update()
status = bpy.context.scene.niko.status
after = ops._version_of(os.path.join(here, "__init__.py"))
marker = "NIKO_UPDATE_MARKER" in open(os.path.join(here, "ui.py"), encoding="utf-8").read()
print("NIKO_UPDATE", res, status, "version on disk", after, "marker", marker)
assert res == {"FINISHED"} and after == newer and marker, (res, status, after, marker)
assert os.path.isfile(os.path.join(REPO, "addon", "niko_tracker", "ui.py"))
assert "NIKO_UPDATE_MARKER" not in open(os.path.join(REPO, "addon", "niko_tracker", "ui.py"), encoding="utf-8").read()
print("NIKO_UPDATE OK")
