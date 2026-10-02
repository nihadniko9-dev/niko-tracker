# Niko Tracker Engine — Design (Phase 1)

Author: Nihad Jihad ("Niko"), Art Director and Motion Designer.
Personal, non-commercial research tool. See `BRIEF.md` for the full brief.

## 1. Where things live

Two roots, on purpose:

| Root | Filesystem | Holds |
|---|---|---|
| `D:\Pack\Track Nhad` = `/mnt/d/Pack/Track Nhad` in WSL | NTFS (Windows) | **Source only**: code, docs, tests, small configs. Small, versioned, editable from Windows. |
| `$NIKO_HOME` = `~/niko` inside WSL | ext4 (Linux) | **Everything heavy**: uv environments, cloned research repos, checkpoints, Blender, benchmark shots, run outputs. |

Heavy I/O never touches `/mnt/c` or `/mnt/d` (9P is ~10x slower than ext4).
uv environments are placed on ext4 with `UV_PROJECT_ENVIRONMENT`, never as `.venv` inside the repo.

### Repo layout (`D:\Pack\Track Nhad`)

```
Track Nhad/
├─ README.md                 quick start
├─ PROGRESS.md               what works, what failed, exact commands
├─ pyproject.toml            orchestrator package `niko` (Python 3.12, no torch)
├─ docs/
│  ├─ BRIEF.md               the brief, verbatim
│  ├─ DESIGN.md              this file
│  ├─ SCHEMA.md              every file exchanged between stages, field by field
│  ├─ LICENSES.md            every repo / model and its license
│  └─ schemas/cameras.schema.json
├─ src/niko/                 orchestrator: CLI, geometry, metrics, pipeline glue
│  ├─ cli.py                 `niko solve | bench … | doctor`
│  ├─ paths.py               $NIKO_HOME layout
│  ├─ backend.py             run a backend env as a subprocess (job.json → result.json)
│  ├─ camio.py               cameras.json read / write / validate
│  ├─ geometry.py            OpenCV ↔ Blender camera conversion, projection
│  ├─ sim3.py                Umeyama Sim(3) alignment
│  ├─ metrics.py             ATE, RPE/drift, rotation, focal, reprojection, jitter
│  ├─ doctor.py
│  ├─ pipeline/              ingest, masks, tracks, candidates, refine, select, export
│  └─ bench/                 generate, import_camera, run, report
├─ backends/                 one uv project per research backend (own Python/PyTorch)
│  ├─ sam3/                  SAM 3.1 multiplex video masks
│  ├─ cotracker/             CoTracker3 offline point tracks
│  ├─ colmap/                COLMAP 4.x incremental + global mapper (ex-GLOMAP)
│  ├─ megasam/               MegaSaM (ported to torch 2.10 / sm_120)
│  └─ da3/                   Depth Anything 3 poses + intrinsics
├─ blender/                  bpy scripts run inside Blender 5.2
├─ scripts/                  WSL bootstrap, env setup, resumable checkpoint download
└─ tests/                    pytest
```

### `$NIKO_HOME` layout (`~/niko` inside WSL)

```
~/niko/
├─ envs/<backend>/           uv virtualenvs (orchestrator = envs/niko)
├─ third_party/<repo>/       research repos, pinned to a commit (recorded in PROGRESS.md)
├─ checkpoints/<model>/      weights (+ .sha256 written after download)
├─ blender/                  Linux Blender 5.2
├─ bench/<set>/<shot>/       synthetic shots with ground truth
└─ runs/<run>/<shot>/        per-shot work dirs (see SCHEMA.md)
```

## 2. Process model

`niko` (orchestrator env, no torch) drives every stage. A research backend is called as

```
$NIKO_HOME/envs/<backend>/bin/python -m niko_<backend> <job.json>
```

The job file names inputs, outputs and options; the backend writes its outputs plus
`result.json` (`ok`, `error`, `runtime_s`, `peak_vram_mb`, versions). Nothing is passed
through pipes or pickles, so every stage can be re-run and inspected on its own.
Refinement, auto-select, metrics and reports run in the orchestrator env (CPU; pycolmap/Ceres).

## 3. Environments

All GPU environments use the **same** PyTorch build so uv hard-links one copy
(one ~5 GB install instead of five). Every version below is re-checked at install time
and the exact resolved versions are written to PROGRESS.md.

| Env | Python | PyTorch | Main packages | Notes |
|---|---|---|---|---|
| `niko` (orchestrator) | 3.12 | — | numpy, scipy, opencv-headless, pycolmap, jsonschema, pytest | also runs refinement / select / report |
| `sam3` | 3.12 | 2.10.0 + cu128 | `sam3` (git, pinned) | SAM 3.1 multiplex checkpoint (gated on HF) |
| `cotracker` | 3.12 | 2.10.0 + cu128 | `cotracker` (git, pinned) | CoTracker3 `scaled_offline.pth` |
| `colmap` | 3.12 | — | pycolmap-cuda12 4.2.0 wheel (GPU SIFT works on sm_120; no source build needed) | global mapper = former GLOMAP (merged into COLMAP 4.0) |
| `megasam` | 3.12 | 2.10.0 + cu128 | DROID-SLAM ext. (`lietorch`, `droid_backends`) compiled with `TORCH_CUDA_ARCH_LIST="12.0"`, torch-scatter shim, **no xformers** | upstream pins py3.10 / torch 2.0.1 / CUDA 11.8; ported by `backends/megasam/port.py` |
| `da3` | 3.12 | 2.10.0 + cu128 | `depth_anything_3` (git, pinned), xformers | DA3NESTED-GIANT-LARGE-1.1 |
| Blender | (bundled) | — | Blender 5.2 Linux in WSL; fallback `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` | Windows 5.2 already installed |

System: Ubuntu 24.04 (WSL2), CUDA toolkit 12.8 (nvcc for sm_120), gcc 13, cmake, ninja, ffmpeg.

## 4. Disk budget (WSL virtual disk)

| Item | Size |
|---|---|
| Ubuntu 24.04 + build tools + ffmpeg | ~6 GB |
| CUDA toolkit 12.8 | ~7 GB |
| uv cache + 6 envs (one shared torch 2.10 cu128) | ~9 GB |
| COLMAP (wheel, or source build if needed) | 0.5–5 GB |
| Checkpoints (SAM 3.1 3.5 GB, DA3 6.8 GB, MegaSaM priors ~3 GB, CoTracker3 0.1 GB) | ~13.5 GB |
| Blender 5.2 Linux | ~1.5 GB |
| **Software subtotal** | **~40 GB** |
| Synthetic benchmark (20 shots × 150 frames, 1080p PNG + GT) | ~15 GB |
| Run outputs (~15 GB per full synthetic run, keep ~3) | ~45 GB |
| Real clips + their runs (checkpoint 4, depends on footage) | ~30 GB |
| **Total** | **~130 GB (budget 200 GB)** |

C: has ~189 GB free, D: ~687 GB (same NVMe). Install the WSL distro on **D:**.

### Downloads over 1 GB (announced before downloading, all resumable)

| Download | Size |
|---|---|
| CUDA toolkit 12.8 (apt) | ~3.5 GB |
| PyTorch 2.10 cu128 + NVIDIA libs (once, uv cache) | ~3 GB |
| `facebook/sam3.1` → `sam3.1_multiplex.pt` (gated) | 3.5 GB |
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` → `model.safetensors` | 6.76 GB |
| MegaSaM priors: Depth Anything v1 ViT-L + UniDepth ViT-L | ~2.7 GB (verify) |
| **Total** | **~19.5 GB** |

## 5. Coordinate conventions (summary — full detail in SCHEMA.md)

- Camera: OpenCV (+X right, +Y down, +Z forward). Extrinsics are world→camera: `x_cam = R · X + t`.
- Pixels: `corner` origin — (0, 0) is the top-left corner of the top-left pixel; that pixel's centre is (0.5, 0.5). Same as COLMAP and Blender. Backends that use pixel-centre-at-integer (OpenCV-style indices) convert with ±0.5 in their adapter.
- Blender camera = OpenCV camera with Y and Z flipped: `R_c2w_blender = R_c2w_opencv · diag(1, −1, −1)`. The world frame is not changed.
- Timing is in frames. `frame` = the clip's frame number, `i` = 0-based index.

## 6. Later phases — agreed with Nihad (2026-09-29)

Not part of Phase 1 (CLI + benchmark); recorded so the engine produces what they need.

- **Host: Blender add-on** is the main UI. Panel layout sketched in chat: clip, "ignore moving
  things" prompt chips (person / car / sky), camera (auto / moving / tripod), lens (auto / known
  mm / zoom), one "Solve camera" button with stage progress; result card with the **average
  tracking error** large and colour-rated (< 0.5 px excellent, 0.5–1 good, > 1 check), an error
  graph per frame, frames solved, lens, camera type; buttons: create camera, lock test, export.
- **After Effects export for compositing**: a `.jsx` that builds a comp matching the clip (size,
  fps, duration), the footage layer, an animated 3D camera (position, orientation, zoom per
  frame) and 3D nulls at selected track points; run in AE with File > Scripts > Run Script File.
  Needs an OpenCV -> AE conversion with a round-trip test (project known points in AE, compare
  with our projection), tested in Nihad's installed AE version.
- Engine side, done 2026-09-29: `solve.json` → `solve_error`, `lens_check`, `known_lens`;
  `selected/errors.json` (per frame); `selected/blender.json`; `niko locktest`; `niko export-ae`
  (verified in After Effects 26.5 to 0.0004 px).
- **Add-on v0.1 built 2026-09-29** (`addon/niko_tracker`): `engine.py` (wsl.exe bridge, Windows
  <-> WSL paths, stage parsing), `solve_io.py` (reads a solve folder), `ops.py` (solve job,
  scene build, workspace, lock test, After Effects, load solve), `ui.py` (sidebar sections),
  `draw.py` (camera-view HUD + error strip, timeline graph). Blender does no camera maths: it
  applies the engine's per-frame matrices under a levelling empty. Checked by the scripts in
  `tests/blender/` (background projection check, GUI screenshot, end-to-end solve).
- **Scene from the shot (asked 2026-09-29)**: turn a solved shot into an editable 3D scene in
  Blender. Ingredients already in the pipeline: the solved camera, per-frame depth (MegaSaM for
  every frame, DA3 on keyframes) and SAM 3 masks to leave moving people and cars out. Plan:
  depth + camera -> TSDF fusion -> mesh, with colour/texture projected from the frames; a
  shadow-catcher ground plane as the quick option; Gaussian splatting as the photoreal option.
  Needs parallax: a tripod or zoom shot gives only single-view depth, not real 3D.
  **Built 2026-09-29 (first version)**: `niko mesh` = COLMAP multi-view stereo on the solve's
  cameras (masks painted out) + Open3D Poisson with data-based trimming; add-on "Build scene
  mesh". Not yet: texture baking from the frames (vertex colours only), Gaussian splatting.
- **Windows installer (asked 2026-09-29)**: one `NikoTracker-Setup.exe` (Inno Setup, custom
  artwork) that is a small bootstrapper, downloading the parts resumably with checksums.
  Measured on this machine (2026-09-29): Python envs 9.8 GB (torch shared by hard links),
  checkpoints 13 GB (SAM 3.1 3.3, DA3 giant 6.3, UniDepth 1.4, DA v1 1.3, CoTracker3 0.1),
  CUDA toolkit 6.7 GB (build only, not shipped), Linux Blender 1.6 GB (benchmark only, not
  shipped), bench + runs 71 GB (dev only). Shipped install ≈ 25 GB (Ubuntu base + envs + code +
  models); download ≈ 18 GB (envs compress, weights do not).
  Steps: check Windows / virtualisation / WSL2 (install, reboot, resume) -> NVIDIA GPU, driver,
  VRAM, disk, RAM -> import the prebuilt engine distro (`wsl --import`) -> models from their
  official sources (SAM 3 is gated: the user's own HF login, never bundled) -> Blender add-on
  into the user's Blender -> `niko doctor` + 30-frame smoke test.
  Hardware profile written at install and re-checked by `niko doctor`: VRAM decides SAM 3 chunk,
  CoTracker window and DA3 model size; MegaSaM's CUDA ops built for sm_86/89/120 + PTX (RTX 30,
  40, 50) instead of 12.0 only. NVIDIA only (CUDA).
  Updates in three layers: code + add-on (MB, one click in Blender), a Python env only when its
  lock file changes, a model only when the registry checksum changes; previous version kept for
  rollback. Hosting (private GitHub / Hugging Face repo) is Nihad's call.
- **"Niko track" workspace tab (asked 2026-09-29)**: the add-on adds its own workspace next to
  Layout / Modeling / Shading (appended from a template .blend shipped in the add-on). Areas:
  footage (Movie Clip editor) with track points drawn in green / amber / red by their error and
  the error under the mouse; the 3D scene with point cloud and camera path; the Niko sidebar
  panel in numbered sections (1 clip, 2 ignore, 3 solve, 4 result, 5 use it); a timeline with
  the error per frame, click a bar to jump to that frame, worst frame named in the result.
  Built from Blender's own widgets plus custom icons (bpy.utils.previews) and GPU-drawn
  overlays and graph (gpu module), so it looks native in Blender's theme.
