PROJECT: Niko Tracker Engine — Phase 1: solver prototype + benchmark

GOAL
I'm Nihad Jihad, Art Director and Motion Designer. I want a camera tracker that beats MotionMaster 3D (a Blender add-on; its Heavy mode is COLMAP + GLOMAP + OpenMVS, with SAM2 masks) on hard shots — low parallax, motion blur, moving people/cars, handheld and drone footage — while staying close to it on easy shots.
Personal, non-commercial research tool: never distributed or sold. Still record each model's license.
Talk to me in Sorani Kurdish; keep code, comments and files in English.

MACHINE
Windows + NVIDIA RTX 5090 (32 GB, Blackwell sm_120). The engine runs inside WSL2 Ubuntu 24.04 with CUDA.
- PyTorch must be a CUDA 12.8+ build; compile custom CUDA ops with TORCH_CUDA_ARCH_LIST="12.0".
- Keep heavy data inside the Linux filesystem, not /mnt/c.
- Tell me the size before any download over 1 GB; downloads must be resumable. Read HF_TOKEN from the environment; never write tokens into the repo.

GROUND RULES
- Phase 1 = no Blender add-on, no UI. A Python CLI (`niko`) + benchmark only. The Blender add-on comes later as a thin client.
- One uv environment per research backend (they need different Python/PyTorch versions; SAM 3 needs Python 3.12+). An orchestrator calls each backend as a subprocess and exchanges data through files (.npz/.json/.png) with a documented schema.
- Verify, don't assume: run every step on real data before saying it works, and show me the numbers. If a repo won't build on Blackwell/WSL2, report it and propose the next best option. Never fake results.
- First save this brief as docs/BRIEF.md. Keep PROGRESS.md updated (what works, what failed, exact commands to reproduce).

PIPELINE — `niko solve <clip>`
1. Ingest: video → frames (JPG q95 default, PNG optional) + 1080p proxy; read fps, resolution, lens metadata.
2. Masks: SAM 3 (prefer the SAM 3.1 multiplex video tracker) with text prompts, default ["person", "car", "animal", "sky", "water"], editable per shot → per-frame binary masks.
3. Point tracks: CoTracker3 (offline) on unmasked regions; grid + texture-aware sampling; long tracks with visibility/confidence. Pluggable backend (TAPNext later).
4. Camera candidates:
   a) Classic baseline: COLMAP/GLOMAP with masks (the same family MotionMaster uses).
   b) MegaSaM (deep visual SLAM for casual dynamic videos: poses, focal, depth).
   c) Depth Anything 3 (DA3NESTED-GIANT-LARGE-1.1) on keyframes for poses + intrinsics, chunked to fit VRAM.
5. Refinement: bundle adjustment of every candidate against the CoTracker3 tracks (robust loss, outlier rejection). Intrinsics modes: shared focal / per-frame focal (zoom) / radial k1,k2. Plus a rotation-only (tripod) solver when parallax is too small.
6. Auto-select: score all refined candidates with one neutral metric (held-out track reprojection error + camera-path jitter), pick the best, keep the rest for the report.
7. Export: cameras.json (per frame: K, distortion, R, t in OpenCV convention + tested conversion to Blender's camera convention), points.ply, and a small bpy import script (camera + background footage + points) for a visual check.

BENCHMARK — `niko bench ...`
- generate: synthetic shots rendered with Blender 5.2 (Cycles, background mode; Linux Blender inside WSL, fall back to the Windows blender.exe if GPU rendering fails). Well-textured scenes (procedural materials or CC0 assets). ~20 shots × 150 frames at 1080p: handheld walk, drone fly-over, orbit, slow pan with little parallax, fast whip with motion blur, zoom, tripod rotation-only, moving people/car proxies, sensor noise, mild lens distortion from a known model, rolling shutter. Save exact ground-truth cameras in the cameras.json schema.
- import-camera: read the camera from any .blend (a MotionMaster solve or Blender's own tracker) by evaluating matrix_world and lens per frame (don't parse F-curves) → cameras.json.
- run: run every method on a shot set (synthetic, or a folder of my real clips).
- report: HTML + CSV. Metrics: success rate, ATE after Sim(3) alignment (% of scene size), relative drift, rotation error (deg), focal error (%), held-out reprojection error (px), jitter, runtime, peak VRAM. For real clips, also render a "lock test" MP4: a checkerboard plane + cube placed on the solved ground, composited over the footage.
- Targets on synthetic: rotation error < 0.2°, focal error < 2%, ATE < 1% of scene size. On real clips: reprojection < 1 px (0.5 px ideal) and no visible sliding in the lock test.

TESTS
- pytest: OpenCV ↔ Blender camera conversion round-trip, Sim(3)/Umeyama alignment, metrics, cameras.json schema.
- `niko doctor`: GPU, CUDA, every backend environment, every checkpoint.
- Smoke test: a 30-frame 540p synthetic shot through the whole pipeline in a few minutes.

CHECKPOINTS — stop and wait for my OK after each
1. WSL2 + environments + `niko doctor` all green + smoke test passing.
2. Synthetic generator + first report with the classic COLMAP/GLOMAP baseline only.
3. MegaSaM + DA3 + CoTracker3 refinement + auto-select → full synthetic report.
4. My real clips + MotionMaster comparison through import-camera.

Before writing code, reply with: the repo layout, the cameras.json schema, the environments with their Python/PyTorch versions, and a disk-space estimate. Then start checkpoint 1.
