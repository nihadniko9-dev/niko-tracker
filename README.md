# Niko Tracker Engine

Camera tracker research prototype and benchmark: solves the camera of a video shot, checks itself
on tracks it never saw, and hands the result to Blender and After Effects.
Author: Nihad Jihad ("Niko"), Art Director and Motion Designer. Personal, non-commercial.

**Copyright (c) 2026 Nihad Jihad ("Niko"). All rights reserved.** The code is visible so the
releases can be downloaded; it may not be copied, modified or redistributed. You may install and
update the official releases for personal, non-commercial use. See [LICENSE](LICENSE).
Third-party components keep their own licenses ([docs/LICENSES.md](docs/LICENSES.md)).

## Install (Windows)

Download `NikoTracker-Setup-<version>.exe` from the
[latest release](https://github.com/nihadniko9-dev/niko-tracker/releases/latest) and run it. It
checks the computer (NVIDIA GPU, driver, memory, disk, WSL2), downloads the tracking engine
(about 4 GB, checked piece by piece) and installs it into WSL2 as `NikoEngine`, downloads the
models (about 13 GB, from their official sources; SAM 3 needs your own Hugging Face token and
Meta's license accepted on huggingface.co/facebook/sam3.1) and installs the Blender add-on.
Updates: the round-arrow button in the add-on's Niko tab installs the newest add-on and engine
code from here; a new engine is only needed when its environments change (the button says so).

- Brief: [docs/BRIEF.md](docs/BRIEF.md)
- Design, environments, disk budget, later phases: [docs/DESIGN.md](docs/DESIGN.md)
- File formats (cameras.json, tracks.npz, errors.json, ...): [docs/SCHEMA.md](docs/SCHEMA.md)
- Model / repo licenses: [docs/LICENSES.md](docs/LICENSES.md)
- Status, results and exact commands: [PROGRESS.md](PROGRESS.md)

## Command line

Run inside WSL, from the repo, with `UV_PROJECT_ENVIRONMENT=$NIKO_HOME/envs/niko`:

```
uv run niko doctor                                   # GPU, CUDA, every backend env, every checkpoint
uv run niko solve <clip|image folder|bench shot> -o <out> [--prompts person,car] [--focal-mm 28 --sensor-mm 36] [--reuse]
                                                     # ingest -> masks -> tracks -> candidates -> refine -> select -> export
uv run niko locktest <out> [--scale 0.5]             # lock-test MP4: held-out tracks drawn over the footage
uv run niko export-ae <out>                          # After Effects .jsx (also written by every solve)
uv run niko mesh <out> [--images 40] [--size 1280]   # editable scene mesh (multi-view stereo on the solve)
uv run niko bench generate [--set synthetic_v1] [--preview] [--shots a,b]
uv run niko bench run --name <run> [--methods ...] [--shots a,b] [--reuse] [--force]
uv run niko bench report <run>                       # runs/<run>/report.html + metrics.csv, copied to reports/<run>
uv run niko bench import-camera file.blend -o cameras.json [--camera NAME] [--gt gt/cameras.json]
```

`solve` output (`<out>/selected/`): `cameras.json`, `points.ply`, `errors.json` (error per frame),
`blender.json` + `import_blender.py` (Blender 5.2), `niko_after_effects.jsx` (After Effects:
File > Scripts > Run Script File), and `<out>/solve.json` with every stage, every candidate's
score, the average tracking error and the lens check.

`--focal-mm`: give the lens when you know it (camera metadata, lens barrel; phones and drones: the
35 mm equivalent with the default 36 mm sensor). A camera that only moves straight ahead cannot
measure its lens; the solve then says "lens uncertain" and this option fixes it.

## Blender add-on

`addon/niko_tracker` (Blender 5.2; installed in `%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons`).
Open the **Niko track** tab: 1 Clip, 2 Ignore, 3 Solve, 4 Result, 5 Use it. The engine runs in
WSL2 in the background; the result is built in the scene (camera, footage, points on a levelled
ground) with the average error and an error-per-frame strip on the camera view.
Load a finished solve: the engine's folder or a flat copy of it (e.g. `reports/test_shots/01`).
5 Use it: pick points and put empties on them, build the scene mesh, lock-test video, After
Effects. The refresh button in the panel header installs the newest add-on: from this folder when
the computer has it, from a release folder (`scripts/publish_release.py <folder>`), or else from
the GitHub releases (`scripts/publish_release.py --github`).

The engine runs inside WSL2 (Ubuntu 24.04). Source lives here; heavy data lives in
`$NIKO_HOME` (`~/niko`) on the Linux filesystem.
