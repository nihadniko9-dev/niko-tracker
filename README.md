# Niko Tracker Engine

Camera tracker research prototype and benchmark: solves the camera of a video shot, checks itself
on tracks it never saw, and hands the result to Blender and After Effects.
Author: Nihad Jiad Yousef, Art Director and Motion Designer. Personal, non-commercial.

**Copyright (c) 2026 Nihad Jiad Yousef. All rights reserved.** The code is visible so the
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
uv run niko mesh <out> [--quality good]             # editable scene mesh (multi-view stereo on the solve)
uv run niko calibrate <out>                          # measure the camera's lens from a calibration clip, once
uv run niko batch <folder|clips...> [-o <root>]      # solve many clips one after another (finished ones skipped)
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
35 mm equivalent with the default 36 mm sensor). A camera that only moves and hardly turns (under
1 degree: a forward dolly, a drone flying sideways on a steady gimbal) cannot measure its lens; the
solve then says "lens not measured". Better than typing a lens: calibrate the camera once.

Maker's lens: for DJI drones whose camera spec is known (Air 3S, Mini 5 Pro: 24 mm equivalent on a
1-inch sensor) the engine uses the spec when the camera hardly turns (under 1 degree: flying straight on
a steady gimbal), because such footage cannot measure the lens (real clips: 2979 px and 1795 px chosen
freely, the spec gives 2662 px; where the footage could measure it, 2654-2940 px).

Lens metadata: a Sony cinema camera records its lens and focus distance in every frame; the engine
uses them as the known lens when the lens does not change in the clip. A lens focused close behaves
longer than its focal length (FX6 at 200 mm focused at 2.1 m: 11.7 % longer; held at each value, the
footage fits best right there, while 24 mm at 1.35 m fits any value within 3 %). `--ignore-rotation`
uses the picture as stored when a file's rotation flag is wrong. A solve where fewer than 60 % of the
tracked points fit is marked "Not reliable".

Cameras: the engine reads which camera made the clip (a DJI drone's telemetry names the camera
module, its video mode and zoom; phones write a make and model). `niko calibrate <out>` on a solved
calibration clip (hover or stand still and turn slowly, a full circle is best) measures the lens and
keeps it in `$NIKO_HOME/camera_profiles.json`; every later clip from that camera and video mode is
then solved with it (when measured to 1.5 % or better; DJI clips are cross-checked with the gimbal).

Cameras tested on real footage: DJI Air 3S and Mini 5 Pro drones, DJI Osmo Pocket 3, GoPro HERO5 to HERO8,
HERO12 and Karma, iPhone 11 / 14 Pro / 15 Pro / 17 Pro Max (H.264, HEVC HDR, ProRes, upright and
variable-rate videos), Sony FX6 (S-Log3 MXF with lens metadata), Sony PXW-Z90 (1080i MXF) and a7 III
(XAVC S MP4). Camera RAW (BRAW, R3D, ARRIRAW, Canon RAW) needs its maker's software: export it from
DaVinci Resolve as ProRes or an EXR / TIFF / DPX sequence first.

Project settings: the Blender scene and the After Effects comp get the video's own resolution (upright
phone videos portrait) and frame rate, measured from the frames (29.97, 59.94, 23.976 exactly; a
variable-rate phone clip at its nominal rate with every frame kept; interlaced video deinterlaced), and
the footage shows exactly one picture per camera key. Export camera writes FBX, Alembic or USD for other
programs (frame 1 at time 1/fps).

The sun: with GPS and the recording time (DJI and GoPro) the engine knows where the sun was;
Blender > Add the real sun puts a sun lamp there, turned with the scene (north from the GPS track).
Add shadow catcher puts a shadow-only ground plane on the floor for compositing.

Drone telemetry (DJI): GPS, altitude and gimbal angles recorded in the video are read automatically.
The camera path is fitted to the GPS track for the real size (Nihad's Air 3S clips: within 1 %, the
scene's camera height changes match the barometer to 1-4 %), and the gimbal gives true gravity for
the level. Without GPS the altitude changes give the size.

## Blender add-on

First-time users: [Sorani Kurdish walkthrough](docs/QUICKSTART.ku.md).
The interface guides you through choosing a video, solving the camera and checking the result.
Before solving, choose what to ignore: nothing (the masks step is skipped), some or all moving things. Advanced settings reveal the lens controls. Low pixel error is not a guarantee:
review lens/tracking warnings and check for sliding before placing your final 3D objects.

Real size: a solve from one video has no unit. With drone telemetry the engine takes metres from the
GPS (or the altitude); otherwise it estimates them from two single-image depth models (UniDepth,
DA3) and uses that only when they agree within 50 %. In Blender (Use your camera > Real size) set it
exactly from two points at a known distance or the camera height, and set the ground from 3 or more
points on the floor (Set ground from points: they become Z = 0, level). Camera, points and mesh
follow together, the setting is saved with the solve (`real_scale.json`), and the reset button goes
back to the engine's own size and ground.

Scene mesh (`niko mesh <solve> --quality fast|good|high`): full-resolution frames for good/high,
points on moving things and far junk removed, floating pieces dropped, small holes filled, light
smoothing; `selected/mesh_sim.ply` is a simplified hole-filled copy for physics (Blender: Add
simulation collider); `selected/mesh_textured.obj` carries the footage itself as a 4K texture on a
much lighter surface (real clip 03: 300k triangles show what the 1.8M-triangle coloured mesh shows;
Blender: Add textured mesh).

CLI reuse is conservative: only a completed solve with matching source contents, settings and
engine code can be reused. Older solves remain loadable; use a new output folder to solve them
again. Existing output folders are preserved. `bench run --force` keeps previous outputs under
the run's `.previous/` directory.

`addon/niko_tracker` (Blender 5.2; installed in `%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons`).
Open the **Niko track** tab: 1 Choose your video, 2 Solve camera, 3 Check your result. The engine runs in
WSL2 in the background; the result is built in the scene (camera, footage, points on a levelled
ground) with the average error and an error-per-frame strip on the camera view.
Load a finished solve: the engine's folder or a flat copy of it (e.g. `reports/test_shots/01`).
Use your camera: pick points and put empties on them, build the optional scene mesh, lock-test video, After
Effects. The refresh button in the panel header installs the newest add-on: from this folder when
the computer has it, from a release folder (`scripts/publish_release.py <folder>`), or else from
the GitHub releases (`scripts/publish_release.py --github`).

The engine runs inside WSL2 (Ubuntu 24.04). Source lives here; heavy data lives in
`$NIKO_HOME` (`~/niko`) on the Linux filesystem.
