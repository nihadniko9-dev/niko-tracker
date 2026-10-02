# PROGRESS

Newest first. Every claim here was run on this machine; the command and the numbers are listed.

## 2026-10-02 (late) — 0.4.0: optional masks, real size, better scene mesh, correct ground

Nihad (on 0.3.9, clip DJI 0079): keep his 0.3.9 changes, credit "Nihad Jihad" without "Niko",
masks optional before solving, mesh and camera at real size, a clean complete mesh that works for
simulation, test everything. His 0.3.9 work is commit 0b7012d; this builds on it unchanged.

- **Masks optional.** Add-on: "Ignore moving things" switch (on by default) with All / None and
  per-kind toggles; off (or nothing chosen) runs `niko solve --prompts none`: no masks stage at all.
  Before, an empty choice fell back to the default prompts. Test: 30-frame synthetic solve without
  masks (`$NIKO_HOME/solves/test_nomasks`): 0.276 px, 190 s, targets met.
- **Real size.** `niko.scale`: the two single-image metric depth models already in the pipeline
  (UniDepth v2 inside MegaSaM, DA3-Nested) are compared with the solve's own tracks, frame by frame;
  the Blender world is put in metres only when they agree within 50 % (`metric_scale` in solve.json,
  `units` in blender.json). On Nihad's clips they are 46-1984 % apart: only clip 02 qualifies
  (6.1 m/unit, 46 %); the others say "size unknown" instead of guessing. Exact sizing in the add-on:
  "Real size" -> From two points (a known distance) or Camera height, saved as real_scale.json and
  kept on rebuild (`tests/blender/addon_size_check.py`: height 4.36 -> 50.00 m, two points 10.000 m).
  Camera display size, clip range and point size follow the scene's scale.
- **Scene mesh.** Full-resolution stereo; stereo points on the solve's masks (by projection, no
  painted frames) and far away are removed; Poisson with a density / distance trim; floating pieces
  removed; Taubin smoothing; small holes filled (Open3D 0.20's fill_holes returns garbage colours:
  nearest-vertex colours instead). Quality fast / good / high. New `mesh_sim.ply` (about 150k
  triangles, no non-manifold edges, bigger holes filled) and an add-on button that adds it as a
  hidden, wireframe Collision object ("Niko scene collider").
  Against synthetic truth (drone_orbit, `scripts/dev/mesh_check.py`, % of median depth):
  pieces 562 -> 18, accuracy 0.35 -> 0.35 %, precision at 1 % / 2 % 98.6 / 99.7 -> 98.3 / 99.7,
  recall 32.2 / 40.3 -> 33.1 / 40.1. A radius-outlier filter and a finer grid were measured and
  dropped (recall 18.9 %). Clip DJI 0079 (4K, good quality, ~20 min): 398 396 fused points, 30 437 on
  water / sky and 4 962 far away removed, 1 074 212 triangles (492 floating pieces removed),
  simulation copy 152 704 triangles; old vs new render: `reports/mesh_0079_old_vs_new.png`.
- **Ground and up** (Blender world and After Effects world, `export_ae.world_alignment`). Measured
  against synthetic truth (`scripts/dev/ground_check.py --gt`): up was 18, 12.5, 48.5, 3.5, 3.2 deg
  off on dolly_forward, drone_high_low_parallax, zoom_in (ground above the camera),
  pan_low_parallax, rolling_shutter_handheld; on Nihad's clip 02 the "ground" ran through the camera.
  Now: up from the cameras' level right axes when the camera turns (`level_up`); the ground must be
  2 % of the median depth below the cameras, roll the camera at most 8 deg, be re-checked after the
  refit, and, when it decides up, hold 5 % of the points. Result: every synthetic shot within
  0.0-0.5 deg (tripods 1.7 / 2.7 -> 0.0 / 0.1) except dolly_forward (18 deg: its points are up to 7 m
  under the true floor) and tele_orbit_short (7.9 deg: the solve's own rotation); clip 02 now says
  no ground (Camera height asks for two points). Real clip 03: vertical guides over the porch pillars
  agree with the new up (`scripts/dev/ground_view.py`). Nihad's solves and report copies re-levelled
  (`scripts/dev/backfill_scale.py`), After Effects scripts re-exported.
- **Name**: "Nihad Jihad" everywhere (licence, add-on, installer and its art, docs, exports, GitHub
  description); no "Niko" nickname.
- Checks: `pytest -m "not smoke"` 82 passed (new: tests/test_scale.py, tests/test_level.py); smoke
  1 passed in 209.7 s (0.272 px, 0.35 m/unit, models 37 % apart); Blender background check on
  clips 02 / 03 / 04 / DJI 0079 / no-masks: OK; size check OK; update check 0.4.0 -> 0.4.1 OK; GUI
  screenshot of the new panel checked.

## 2026-10-02 — local 0.3.9 preview: guided workflow and safer results

- User approved implementation after the read-only review. Local changes only; no GitHub release
  or push. Installed add-on 0.3.9 in Blender 5.2, with the original 0.3.8 preserved under
  `%APPDATA%/Blender Foundation/Blender/5.2/scripts/niko_tracker_backups/`.
  The user's open, unsaved Blender session was left running; restart after saving to load the update.
- Guided UI: Choose your video -> Solve camera -> Check your result; advanced lens and mask controls,
  actionable result warnings, inlier fraction and an explicit inlier-average label. The camera HUD
  also says Needs review for an uncertain lens instead of calling the solve Excellent. Mesh is
  labelled optional. Sorani walkthrough: `docs/QUICKSTART.ku.md`.
- Backend execution removes stale result.json and rejects nonzero exit codes. Successful camera
  export can complete with warnings when an alternative candidate fails; essential export/selection
  failures remain failures. A known-lens solve cannot silently fall back to an unconstrained raw result.
- Conservative reuse: source content hashes, options, hardware profile and engine Python code must
  match a completed run's reuse.json. Legacy solves remain loadable but cannot be reused without a
  new solve. Existing solve folders are protected; benchmark --force archives under `.previous/`.
- Add-on updates validate all Python sources before a directory swap, preserve a backup outside the
  add-on discovery directory, and restore it on a failed swap. Engine code updates check staged imports
  before swapping and retain the previous engine. The update shell was exercised against disposable
  valid and deliberately broken releases, not the user's installed engine.
- Validation: 67 non-Blender tests passed initially; 6 Blender tests passed (73 total excluding smoke).
  Final non-smoke run: 76 passed in 21.08 s, including the new relative-source regression and two
  staged engine-update checks. Full GPU smoke:
  1 passed in 229.45 s. Reliability suite after final backup change: 11 passed. Commands used:
  `python -B -m pytest -q -p no:cacheprovider -m 'not smoke'` and separate `-m smoke`, in the WSL
  niko environment, with isolated basetemp directories. Blender add-on update check passed in a
  scratch installation; installed 0.3.9 enable check passed. Existing clip 03 projection check:
  780 projections, max 0.000517 px difference between Blender and engine.
- Real test: a 2-second, 30 fps, 1280x720 copy from clip 03 (`reports/phase1/clip03-preview.mp4`).
  Fresh solve in `$NIKO_HOME/solves/phase1_clip03_preview`: 60/60 frames, selected megasam+ba_k1k2,
  0.19834 px inlier mean, 0.13106 px median, 99.8827% inliers; 488.6 s. Both COLMAP alternatives
  failed; result correctly says completed_with_warnings. Lens remains uncertain (21% spread).
  This short sample verifies the workflow, not ground-truth accuracy or improved performance.
  Lock test: 1500 held-out SIFT tracks, 93.1% green / 5.1% amber / 1.8% red; no ground patches.
  Loading this new result in Blender exposed relative CLI source paths; new solves now record
  absolute paths, and legacy solves use shot.json's canonical source. After the fix: 720 projections,
  maximum Blender/engine difference 0.000413 px. Source and installed add-on files match.
  `reports/phase1/lock-test.mp4` and `preview.png` are local review artifacts. ZIP: `dist/niko_tracker-0.3.9.zip`.
- Still outside this first milestone: faster inference, hard-shot accuracy improvements, lower-VRAM
  hardware validation, and the separate installer hardware/disk checks identified in the review.

## 2026-10-02 (night) — own folder, cleanup

Nihad: everything for work and updates in its own folder; the extra files removed.

- **Project folder: `D:\Niko Tracker`** (was D:\Pack\Track Nhad, among editing packs). Copied with
  robocopy (664 files, 537 MB, git history intact), then every path repointed: WSL env.sh
  NIKO_REPO, the six envs' editable installs (24 .pth / direct_url files), add-on default folder,
  WSL setup scripts, lock files, docs, the clip paths in Nihad's solves and their AE scripts.
  Checked from the new folder: `pytest` 64 passed, `niko doctor` all backends and models OK,
  Blender check clip 03 0.0005 px. The old folder could not be moved in place: this session's
  own process (claude.exe) watches it; it goes to the Recycle Bin once the session has moved.
  `START HERE.txt` explains the folder.
- **Recycle Bin** (recoverable): D:\NikoEngine 7.95 GB (engine build output, now on GitHub;
  its parts list is kept in `installer\engine\releases`), add-on zips 0.2.0-0.3.7, the AE check's
  temp folder.
- **Engine (WSL) extras** gathered in `~/niko/_extra_old` (40 GB listed, ~15 GB of it not shared
  with kept data by hard links): earlier attempts and tests (DJI 0148 first try, four "New folder (4)"
  attempts, clip 03 4 s cut, test_01 stride variant, smoke and reference solves), old runs
  (cp2_colmap, cp2_colmap_v1, cp3_lens, smoke, smoke_solve, stride_test), pytest_tmp, 24 logs.
  Nihad deletes them with `Delete extra Niko files.bat` (asks first). Kept: his solves test_01-04,
  dji_0148 and DJI 0036 (solved by him today from Blender: 974 / 974 frames, 0.573 px, lens
  measured (equally good fits within 1.7 %), no tracking break, mesh built, 42.5 min), the
  benchmark runs cp3_full and tele_v1, the benchmark shots, models, envs.
- `installer\build.ps1` takes the newest engine parts list <= the version from the repo, so a
  code-only release still installs the last engine image.

## 2026-10-02 (evening) — all rights reserved; the engine image; installer installs everything

Nihad: build the engine; on GitHub all rights must stay his, others may only install and update.

- **Rights**: `LICENSE` = copyright Nihad Jihad (Niko), all rights reserved: installing and updating
  the official releases is allowed for personal, non-commercial use; copying, changing,
  redistributing or commercial use need his written permission; third-party parts keep their own
  licenses (SAM License: redistribution with the license, which ships in sam3/; CoTracker3
  CC-BY-NC 4.0: non-commercial with attribution; MegaSaM and Depth Anything 3: Apache-2.0 - read
  from the license files in third_party). The installer shows LICENSE; README and the add-on say it.
  GitHub: wiki, issues, projects, discussions off; main protected (no force push, no deletion);
  interaction limit "collaborators only" until 2027-04-02 (GitHub's longest; renew then). On a
  public repository GitHub still lets anyone view and fork; the license says a fork gives no right.
- **Engine image** (`installer/engine/build_engine.ps1`: stage.sh -> import as a throw-away copy ->
  prepare.sh -> export -> xz -T0 -> parts): the working distro is only read. Left out: Nihad's
  solves (49 GB), runs (69 GB), benchmark data, checkpoints (13 GB, downloaded at install), the uv
  cache (the envs keep their hard-linked files), CUDA toolkit 6.7 GB and Nsight 2.1 GB (build tools,
  not redistributable; the compiled extensions link only against PyTorch: checked with ldd and
  by running), secrets.sh, shell history, ssh, caches, logs. Checked in the staged tar (181 355
  entries): no solves, runs, bench, checkpoints, secrets, history, ssh, CUDA, footage; 11 .mp4 =
  the third-party repos' demo clips. prepare.sh repoints the envs' editable installs from the
  Windows checkout to `$NIKO_HOME/engine` (committed tree of the version), writes env.sh
  (UV_NO_SYNC, no CUDA_HOME), ENGINE_VERSION (code) and IMAGE_VERSION (environments).
  0.3.7 image: 13.2 GB staged -> **4.25 GB tar.xz** (xz -6, all cores, 6 min) -> 3 parts of < 2 GB.
- **Installer** (Inno Setup, per user): hardware check (also in silent installs), WSL2 check (turns
  it on with an administrator prompt when missing), engine download (Inno's download page, SHA-256
  per part), join, `wsl --import NikoEngine %LOCALAPPDATA%\NikoTracker\engine`, hardware.json into
  the engine, models in a console window (`fetch_checkpoints.py --all --ask-token`: shows 13.2 GB,
  resumable, asks for the Hugging Face token SAM 3 needs and keeps it in secrets.sh), add-on;
  uninstall asks before removing the engine. Silent-safe questions (default: keep).
  **Tested end to end** with the parts served from this PC (`python -m http.server`, installer
  /ENGINEURL): 3 parts downloaded and checked, joined, imported: 196 s. In NikoEngine: user rudaw,
  NIKO_REPO=$NIKO_HOME/engine, all five backends pass `niko doctor` on the RTX 5090 (CUDA kernels,
  MegaSaM extensions, pycolmap CUDA); developer-only checks (nvcc, TORCH_CUDA_ARCH_LIST, Linux
  Blender) are warnings inside an engine image. Smoke shot (30 frames) solved there:
  colmap_global+ba 0.272 px, rot 0.147°, focal 1.21 %, targets met - the development engine on
  the same frames: colmap_global+ba 0.272 px, rot 0.143°, focal 1.19 %.
- **Updates of an installed engine**: the update button now also installs the release's engine code
  (`niko-engine-code-<ver>.tar.gz`, ~1 MB) into NikoEngine when the image can run it
  (`engine_image` in latest.json), else says the new setup is needed; refuses while a solve runs.
  Tested with a local 0.3.8 release: add-on 0.3.7 -> 0.3.8 and engine code 0.3.7 -> 0.3.8,
  IMAGE_VERSION kept 0.3.7, engine imports and runs afterwards.
- **Final image 0.3.8** (committed code c4c5ea4): 3.96 GiB tar.xz (4 250 215 260 bytes), 3 parts,
  sha256 29324f88...4710. Final installer NikoTracker-Setup-0.3.8.exe (GitHub release URLs)
  installed it from the local server (/ENGINEURL) in 187 s: code 0.3.8, image 0.3.8, `niko doctor`
  0 failures (with the models copied in), smoke shot colmap_incremental+ba 0.274 px, rot 0.139°,
  focal 1.15 %, targets met. Test engine, test install and server removed afterwards; this PC
  keeps its development engine (Ubuntu-24.04). **Released** (Nihad: "upload now"):
  https://github.com/nihadniko9-dev/niko-tracker/releases/tag/v0.3.8 - 7 files; GitHub's SHA-256
  of the three parts equals the local ones (044eed1b..., 574196..., 157f53...); every download
  address answers 200 with the right size; latest.json -> 0.3.8, engine_image 0.3.8.
- Add-on 0.3.8 finds the engine distro by itself (NikoEngine, else Ubuntu-24.04).
- `publish_release.py`: engine code tarball, `--engine` (upload the image parts) or
  `--engine-image <ver>`. `pytest` 64 passed; Blender checks OK.

## 2026-10-02 (afternoon) — on GitHub, updates from GitHub releases, After Effects check passed

- **After Effects round trip with undistorted footage (clip 03)**: AE was open with an empty,
  unchanged "Untitled Project" (Nihad: "After Effects is empty now, you can test"). ae_check now
  accepts exactly that case (the in-AE guard double-checks). 24 nulls at 4 frames, AE vs niko max
  **0.000703 px** over 96 projections; footage = the undistorted sequence (k1 -0.2470, k2 1.1905,
  12.3 px at the frame corners) imported as a sequence; AE quit by itself afterwards.
- **GitHub**: GitHub CLI 2.102.0 (official release, SHA-256 matches the published checksums,
  signed GitHub, Inc.) in %LOCALAPPDATA%\Programs\gh; Nihad approved the device login himself
  (account nihadniko9-dev). Nihad chose a public repository:
  https://github.com/nihadniko9-dev/niko-tracker (181 files; his footage, reports, dist and
  installer output stay out via .gitignore; scanned for tokens and keys first: none; his AE
  project name taken out of these notes; commits use the GitHub no-reply address).
- **Release v0.3.7**: NikoTracker-Setup-0.3.7.exe, niko_tracker-0.3.7.zip, latest.json
  (`python scripts/publish_release.py --github --notes ...`).
- **Add-on 0.3.7 updates from GitHub**: the update button uses the project folder when the
  computer has it, else a release folder, else `releases/latest/download/latest.json` (preference
  "Update address"); it only installs a newer version. `tests/blender/addon_update_check.py ...
  github`: a copy marked 0.3.6 updated to 0.3.7 from the live release: OK. Release-folder variant
  0.3.7 -> 0.3.8: OK. `pytest` 64 passed.

## 2026-10-02 — installer built, updates from a release folder, After Effects check made safe

Nihad: "do those things now, you have the authority for what is needed".

- **After Effects check made safe.** AE was open with one of Nihad's work projects
  and unsaved changes. `scripts/dev/ae_check.py` hands its script to a running AE (`-r`) and the
  probe ends with `app.project.close(DO_NOT_SAVE_CHANGES); app.quit()`: it would have thrown that
  work away. Not run. Now (1) ae_check refuses while AfterFX.exe runs, (2) the probe script itself
  leaves AE untouched when a project is saved, edited or has items (`export_ae.PROBE_GUARD`, before
  anything is added; test in tests/test_export_ae.py). The undistorted-footage check (clip 03)
  waits until AE is closed.
- **Inno Setup 6.7.3** downloaded from the official page's link (github.com/jrsoftware/issrc
  release; 10.6 MB, not the 7 MB guessed earlier), Authenticode signature valid (Pyrsys B.V.),
  SHA-256 9C73C3BA...6E97B732, installed per user (%LOCALAPPDATA%\Programs\Inno Setup 6).
- **Installer built**: `installer\build.ps1` -> `installer\Output\NikoTracker-Setup-0.3.6.exe`
  (2.2 MB, version read from the add-on). Per user (no administrator prompt), hardware check page
  that stops on blockers (accepts PowerShell 5 and 7 JSON spacing), Blender add-on into every
  Blender, docs, Start menu entries, uninstaller, an after-install page saying what is and is not
  installed. Artwork from `installer/art/make_art.py` (side image, corner image, icon; 100 / 200 %).
  Tested: silent install -> files, Start menu, add-on step exit 0; silent uninstall -> folder gone.
  Not in it yet: WSL, the engine image and the models (need the slim engine image and hosting).
- **Updates from a release folder** (add-on 0.3.6): the update button reads `latest.json` + the
  add-on zip from the "Niko Tracker folder" when that is a release folder (else the project folder
  as before). `scripts/publish_release.py <folder>` writes zip, installer and latest.json.
  `tests/blender/addon_update_check.py`: a copy of the add-on updated 0.3.5 -> 0.3.6 from a release
  folder, files replaced, repo untouched: OK. Where the release folder lives is Nihad's choice.
- Checks: `pytest` 64 passed; Blender background check on clip 03 0.0005 px, OK; add-on 0.3.6
  installed in Blender 5.2.

## 2026-10-01/02 — whip pans: bridging tried and not adopted; tracking breaks now reported

### Bridging the whip with direct image alignment (`scripts/dev/whip_bridge_check.py`): not adopted

whip_pan_blur turns 91.6° in 10 motion-blurred frames; no SIFT track crosses it, each half is
right to ~0.01° and the turn between them 2.7° off. Measured the turn of every frame pair through
the whip from the pixels alone (GT lens, half resolution), against GT, frames 66-84 (GT 90.8°):

| method | per pair (slow / whip) | chained error |
|---|---|---|
| homography ECC from a phase-correlation start | 0.002-0.005° / up to 2.78° | 5.0° |
| 3-parameter rotation warp, least squares coarse to fine | 0.005-0.01° / up to 0.94° | 5.0° |
| + 1-D search of the turn size at every level | / up to 0.36° (all short) | 1.5° |
| + each image given the other's blur (box, 0.5 shutter) | / ±0.05 speeding up, +0.21 slowing | 0.76° |
| + lopsided blur from a monotone spline of the turns, passes 1-6 | | 0.90, 0.42, **0.19**, 0.47, 0.53, 0.74° |

The cost at the true turn is lower than where the optimiser stopped (72-73: mse 0.00049 vs
0.00072), so the first shortfall was the start and local dips, not the blur. With blur
equalised the pairs scatter ±0.1-0.2° and the iteration does not settle. The target is rot max
< 0.2°; a bridge that lands anywhere in 0.2-0.9° is not a dependable fix, so the engine does not
use it. whip_pan_blur stays the one open synthetic miss.

### New: tracking breaks are reported (`niko.tracks.track_gaps`, `solve.json` -> `track_gaps`)

For each boundary between consecutive frames with SIFT observations (keyframes of a strided
solve), the SIFT tracks seen in the 3 frames before AND the 3 after (`scripts/dev/track_gaps.py`):
whip_pan_blur 0-2 across the whip; every other synthetic shot >= 604 (low_texture_walk), Nihad's
clips >= 232 (clip 02; 01 2046, 03 4682, 04 3240, DJI 891, the 4 s cut of 03 5326). Below 50 is a
break; neighbouring weak boundaries merge. whip_pan_blur: "frames 72 - 80"; none of Nihad's clips.
- Solve log line, `track_gaps` in `solve.json` (docs/SCHEMA.md), a "Tracking breaks" row in the
  clip report, and add-on 0.3.5: a "Tracking broke" box with the frames and a jump button.
- `scripts/dev/backfill_checks.py` (was backfill_lens_check.py) writes lens_check and track_gaps
  into finished solves; run on Nihad's solves, their copies and cp3_full/whip_pan_blur.
- `tests/blender/addon_background_check.py` now also checks tripod solves (no 3D points: test
  points along the view rays) and only points within a frame of the picture: on the whip, points
  20 000+ px to the side differed by up to 0.24 px from float32 alone. Whip 0.0011 px, clip 03
  0.0005 px, DJI 0.0061 px: OK. Screenshot checked; add-on 0.3.5 installed.
- Lens metadata: Nihad's 01-04 carry none (re-exported, brand mp42); the DJI file says only
  "DJI Air3s", which has a 24 mm and a 70 mm camera, so the tag alone cannot give the lens.
- Checks after these changes: `pytest` 63 passed. Smoke test: **failed once** (MegaSaM crashed
  in its first step, the depth encoder; After Effects was open holding ~11 GB of VRAM), then
  **passed** on the re-run (265 s, colmap_incremental+ba 0.274 px, 30/30). The failed run's log was
  gone: WSL clears /tmp when the distro goes idle. Test folders now go to `$NIKO_HOME/pytest_tmp`
  (tests/conftest.py) and stay until the next run. Cause of the crash not confirmed.

## 2026-09-29 evening — long-lens shots: what the footage can measure, and a new lens warning

### Synthetic telephoto set `tele_v1` (`configs/benchmark/synthetic_tele.json`, like clip 03)

`niko bench generate --set synthetic_tele && niko bench run --set synthetic_tele --name tele_v1`
(report `reports/tele_v1/`). colmap_incremental fails on both orbits ("no reconstruction");
colmap_global solves all three.

| shot | lens | pick | rot max | focal error | targets |
|---|---|---|---|---|---|
| tele_orbit_slow | 80 mm, 12° arc at 140 m | megasam+ba | 0.007° | 0.10 % | met |
| tele_line_side | 70 mm, 16 m slide at ~150 m | colmap_incremental+ba | 0.021° | 1.54 % | met |
| tele_orbit_short | 100 mm, 5° arc at 160 m | colmap_global (raw) | 0.625° | 28.3 % | missed |

- tele_orbit_short: colmap_global+ba had rot 0.083° / focal 3.5 % but lost on path jitter (0.295
  vs 0.106 px) with the same held-out reprojection (0.200 vs 0.198 px). Every candidate from
  megasam (-39 %) to colmap_global+ba (-3.5 %) fits the footage to within 5 %: the footage does
  not measure this lens.
- tele_line_side focal +1.54 % after refinement (raw COLMAP +0.17 %): **not a bug**.
  `scripts/dev/focal_bias_check.py`: SIFT only +1.37 % (8000 tracks) / +1.29 % (all 20000), so
  not CoTracker3 (+0.17 % more with 4000 segments) and not the SIFT subset. The adjustment has
  converged (focal still 3773.8 px after 800 iterations; COLMAP's "no convergence" at 100 is its
  zero function/parameter tolerances, the cost moves in the 6th digit after that).
  `scripts/dev/focal_profile.py` with the focal fixed at -1 / 0 / +0.5 / +1 / +1.4 / +2 %:
  adjustment median 0.187 px and held-out 0.323-0.324 px for all six. Anything within +-2 %
  reprojects identically, so a composite cannot show it.

### Selector jitter: two on-screen variants tried, neither adopted (`scripts/dev/reselect.py`)

The path jitter adds rotation * f and sideways translation * f / depth, so a turn and a slide that
cancel on screen count twice. Re-scoring stored candidates (no re-solve) with the shake measured
on screen instead:
- image acceleration of the held-out points: near points blow it up (car_follow 1467 px),
  distortion_barrel goes to a 31 % focal error. Dropped.
- a 3 x 3 grid of virtual points at the median depth (`screen_jitter_px`): a sound measure of
  visible shake (tele_orbit_short: 0.011-0.012 px for every good candidate), but zoom_in goes to
  a dolly solution 72 % off and telephoto 1 / 3 instead of 2 / 3. The double counting is what
  marks those paths as implausible. `niko/pipeline/select.py` is unchanged.

### New: lens warning from equally good fits (`solve.lens_spread`, in `lens_check`)

Candidates whose held-out reprojection is within 10 % of the best (>= 95 % of their frames or
keyframes, tripod candidates excluded) fit the footage equally well; when their lenses differ by
more than 10 %, `lens_check.uncertain` with reason `equally_good_fits`. Not checked when the pick
is a tripod solve (its lens comes from the rotation). `scripts/dev/lens_spread.py` on every run:

- synthetic, pick not tripod: 18 of 21 spread <= 5.6 % (their picks' focal error <= 1.54 %);
  flagged: dolly_forward 1026 % (pick 86 % off), tele_orbit_short 58 % (28 % off),
  pan_low_parallax 36 % (pick right - a false alarm; the only cost is advice to give the lens).
- Nihad's clips: 01 2.6 %, 04 0.5 %, DJI 0148 5.5 % measured; **02 21.8 % (79-96 mm) and 03
  14.4 % (80-92 mm; its 4 s cut 4451-5500 px) flagged** - the 80 vs 103 mm of clip 03 explained.
- Existing solves rewritten with `scripts/dev/backfill_lens_check.py` (solves + the copies in
  `reports/test_shots/`, `reports/dji_0148/`); report page rebuilt.
- Add-on 0.3.4 (installed): the "Lens uncertain" box says "Equally good fits: 80 - 92 mm"; panel
  texts shortened to fit the sidebar. Background check OK (worst |blender - engine| 0.0005 px),
  screenshot checked. `pytest`: 61 passed (6 new in `tests/test_lens_check.py`).

## 2026-09-29 afternoon — add-on 0.2: pick points, update button, scene mesh, undistorted AE footage

Nihad: real-clip test "very good"; asked for the point-picking button, updates, and to keep
building the agreed features.

- **Empty at selected points** (sidebar 5 Use it): "Pick points" puts 'Niko points' in Edit Mode
  (the point spheres hide there so the vertices themselves are clicked, in the camera view over
  the footage); "Empty at each point" / "One in the middle" add empties parented to Niko world.
  `tests/blender/addon_empties_check.py`: each empty on its point to 1.6e-6 units, the middle one
  at the centroid, and they follow when Niko world moves.
- **Update button** (sidebar header and add-on preferences): copies the newest add-on from the
  project folder into Blender (the engine there is always current), then restart Blender. Tested
  0.2.0 -> 0.2.1 from a fake newer copy; add-on is now 0.2.0.
- **Scene mesh** (`niko mesh <solve_dir>`, sidebar "Build scene mesh"): COLMAP multi-view stereo
  (PatchMatch on the GPU, pycolmap 4.2) on 40 frames spread over the shot, with the solve's own
  cameras and the SAM 3 masks painted black (no surface on people, cars, water), fused points,
  then the surface in Open3D (da3 env): outlier removal, screened Poisson, only surface with
  data behind it kept (lowest-density 8 % and anything > 3 voxels from a point removed).
  COLMAP's own Poisson mesher gave vertices at +-1e38 and NaN on clip 03, hence Open3D.
  The first version closed a bubble around the scene (the camera sat inside it): fixed by the
  trimming. Results: clip 03 1.2 M fused points -> 222 k vertices (60 frames at 1600 px: 20 min;
  now 40 at 1280 px), clip 01 343 k fused points -> 127 k vertices in 10 min. In Blender the
  mesh lands on the footage in the camera view and shows in its own colours in the free view.
  A rotation-only (tripod) solve has no parallax: the command says so instead of meshing.
- **Project footage on mesh** (sidebar, once a scene mesh is loaded): camera projection - a UV
  Project modifier from 'Niko camera' and an emission material with the footage (movie or
  sequence, following the frame). Clip 03: the building mesh wears the real footage in the free
  view (walls, windows, roof); black where stereo found no surface. The colour button goes back
  to vertex colours. `tests/blender/addon_gui_project.py`.
- **Undistorted footage for After Effects**: when the lens distortion moves the frame corners by
  more than 0.5 HD px, the export writes selected/undistorted/ (same size, same K) and the .jsx
  uses it, so the AE pinhole camera matches everywhere. Clip 03 (k1 -0.247, k2 1.19: 12.3 px at
  the corners). `tests/test_export_ae.py::test_undistort_maps_match_our_distortion_model`: the
  maps fetch each pixel where our model distorts it, < 0.001 px (corner-origin handled).
  **Not yet checked in After Effects itself**: AE did not run any script this afternoon (even a
  one-line test that worked at 02:20) - it stops at something on screen the locked session hides.
  Nihad: open After Effects once, close whatever it shows, then `scripts/dev/ae_check.py` again.
- **Keyframe stride checked on real footage**: clip 01 re-solved with stride 3 (140 of 418 frames
  to COLMAP / MegaSaM) against the full-rate solve (`scripts/dev/compare_solves.py`): rotation
  difference median 0.0028°, max 0.0105°; path difference 0.016 % of median depth; focal 0.98 %
  apart; average error 0.431 vs 0.429 px; 19 min instead of ~75 min (COLMAP global 256 s instead
  of 1656 s). MegaSaM, which diverged at full rate on this clip, worked on the keyframes.
- **Scene mesh on the 4 clips**: 01 127 k vertices, 03 222 k, 04 125 k (9.8 min); 02 none: the
  telephoto camera barely moves, stereo finds no frame pairs with enough angle (even relaxed to
  0.2°: 213 consistent points). The command now says so ("the camera moves too little for 3D;
  the camera track itself is not affected").
- **Nihad's DJI clip** (`Desktop\DJI_20260809185601_0148_D.MP4`, DJI Air 3S, 4K, 1700 frames at
  59.94 fps; his own add-on try this morning had stopped): solved in 46 min (masks 967 s with
  20 people / 41 cars / water = 43 % of each frame, tracks 365 s, keyframes every 9th = 189,
  colmap_global+ba picked), 1700 / 1700 frames, average 0.70 px (0.35 HD px), 94.9 % of held-out
  SIFT within 3 px, worst frame 892 (1.20 px); lens 2654 px = 24.9 mm full-frame equivalent,
  MegaSaM 26.4 mm, DA3 25.4 mm, and the Air 3S main camera is 24 mm. Lock test 76.7 / 16.3 / 7.0 %;
  mesh 395 k stereo points -> 203 k vertices (10.7 min). `reports/dji_0148/`.
- Add-on **0.3.0** (installed, `dist/niko_tracker-0.3.0.zip`): the HUD and panel show the HD
  equivalent next to 4K errors ("0.61 px (0.30 HD) Excellent"); Load a finished solve opens in
  the engine's solves folder.
- **End to end from Blender on real 60 fps footage** (a 4 s cut of clip 03, 240 frames, stride
  2): Solve camera -> 7 / 7 stages -> megasam+ba, 0.29 px, 240 / 240 frames (17 min). Two add-on
  bugs found and fixed: (1) the panel said "No result": a draw callback had cached the solve while
  the engine was between writing blender.json and solve.json; the cache now also watches
  solve.json and errors.json (checked by writing them in that order); (2) points drawn as balls of
  a fixed size covered the footage with a long lens; now ~2.5 px on screen whatever the focal.
  Add-on 0.3.3.
- **Lens on telephoto, little-parallax shots is weakly measured**: the same footage gave 80 mm
  (clip 03, 8.6 s) and 103 mm (its first 4 s); the depth models stayed put (DA3 ~71 mm, MegaSaM
  ~82 mm on both). The panel now also shows "depth models say 72 - 83 mm" under the solved lens.
  A synthetic telephoto set with GT (`configs/benchmark/synthetic_tele.json`: 80 / 70 / 100 mm,
  drone far away, slow move): results in the evening section above.
- **Regression check after all of today's engine changes**: `pytest` 48 passed, smoke test passed
  (216 s), `niko bench run --name cp3_full --reuse --force` -> niko_selected 19 / 21 again, same
  medians (ATE 0.0008 %, rot max 0.0078°), same two misses.
- **`niko report <solve dirs> -o page.html`** (`src/niko/clip_report.py`): one self-contained page
  (images embedded, reads on a phone) per clip: lock-test frame, average error with rating (HD
  pixels for bigger footage), error-per-frame strip, lens with the depth-model check, pick,
  keyframes, what was ignored, mesh, time per stage. `reports/test_shots/index.html`.
- **SAM 3 on every 2nd frame, tried and not adopted** (`frame_stride` option in the sam3 backend,
  default 1; `scripts/dev/mask_stride_check.py`): clip 01, 173 s instead of 252 s (1.46x: chunks
  and re-detection cost the same), median recall 0.994 against the every-frame masks but the 1st
  percentile 0.41 and one frame at 0.035 (a whole region, probably the water, not re-detected). A
  mover left unmasked costs more than the minutes saved.
- **Add-on zip** for another computer: `python scripts/package_addon.py` -> `dist/niko_tracker-0.2.0.zip`.
- **Installer, first parts** (`installer/`):
  - `hardware_check.ps1`: Windows build, RAM, free disk, CPU, WSL version and distros, NVIDIA
    GPUs (name, VRAM, driver, compute capability), Blender and After Effects installs -> JSON with
    a profile (full >= 24 GB, medium 12-24, light 8-12, none) and blockers (no NVIDIA, driver < 570,
    compute < 7.5, < 8 GB, < 60 GB disk, Windows too old). This PC: RTX 5090 31.8 GB, driver
    616.92, 12.0 -> "full", no blockers.
  - The engine reads `$NIKO_HOME/hardware.json` (`solve.hardware_settings`): SAM 3 chunk size and
    CoTracker3 size follow the profile (defaults = this machine's values).
  - `install_addon.ps1`: copies the add-on into every installed Blender's user add-ons and
    enables it (background Blender + save preferences). Run here: 0.2.0 installed and enabled.
  - `NikoTracker.iss`: the Inno Setup wizard (hardware page that stops on blockers, add-on
    component, steps for WSL / engine image / models / smoke test written down). Not compiled:
    Inno Setup is a 7 MB download (asking Nihad first), and the engine image needs a slim distro
    (without the 70 GB of benchmark data) and a place to host it.

### Nihad's 4 real clips (`D:\Pack\Track Nhad\Test SHot`), all solved, results in `reports/test_shots/<n>/`

`uv run niko solve "/mnt/d/Pack/Track Nhad/Test SHot/<n>.mp4" -o $NIKO_HOME/solves/test_<n> --prompts person,car,animal,sky,water,flag`
then `uv run niko locktest $NIKO_HOME/solves/test_<n> --scale 0.5`. No ground truth and no
MotionMaster files: judged by held-out error, lens agreement and the lock test watched frame by frame.

| clip | content | frames | stride | picked | avg error | lens (full-frame eq.) | lock test green / amber / red |
|---|---|---|---|---|---|---|---|
| 01 | drone, river + stone bridge, walkers | 418 1080p | 1 (ran before stride) | colmap_global+ba | 0.43 px | 1411 px ~ 26 mm | 73.5 / 17.9 / 8.7 % |
| 02 | indoor, people on a sofa, telephoto | 316 1080p | 2 | megasam+ba | 0.38 px* | 5144 px ~ 96 mm | 78.8 / 15.8 / 5.4 % |
| 03 | drone over a building, flag | 514 1080p | 3 | megasam+ba_k1k2 | 0.33 px | 4269 px ~ 80 mm | 81.5 / 12.8 / 5.7 % |
| 04 | 4K drone toward Dalal bridge | 1323 4K | 7 | colmap_global+ba | 0.61 px (0.30 HD) | 2712 px ~ 25 mm | 82.2 / 13.0 / 4.8 % (HD scale) |

\* clip 02 had too few held-out SIFT tracks (484): scored on CoTracker3 held-out segments.
Lens: MegaSaM and DA3 agree with every pick (|log ratio| <= 0.51); no "lens uncertain".
Every frame solved on all four. Timings: 01 ~80 min at full rate (COLMAP 1656 + 1620 s),
02 ~10 min, 03 ~30 min, 04 ~38 min (masks 758 s, tracks 467 s at 4K).

What the real clips broke, all fixed and re-checked:
- MegaSaM diverged on 01 (focal -1158 px): now a clear candidate failure.
- COLMAP on 02 "registered" 158 frames on 6 points; incremental gave up: models under 50 points
  are failures now. MegaSaM carried the shot.
- SAM 3.1 on 04: "No points are provided" from its tracker in a chunk with nothing to track:
  taken as "no object" for those frames (counted in masks.json).
- Refinement on 04: the per-frame focal model sent one frame's focal <= 0 and the exception
  aborted the whole refine stage (losing the good models): each model now fails on its own,
  with a focal check.
- Ground patches floated in the air on 02 (no floor in view): planted only when the plane holds
  >= 15 % of the points.
- 4K: lock-test colours and the add-on's rating now use HD-pixel thresholds (a 4K frame has
  twice the pixels per degree); clip 04 read 54 % green in 4K pixels, 82 % at HD scale.
  Blender vs engine on clip 04: 778 projections, max 0.013 px (4K pixels).

- **Nihad's own add-on tries this morning** (found in `$NIKO_HOME/solves`): a 4K DJI clip (1700
  frames) stopped during masks (long clip, no sense of progress); "New folder (4)" failed 4 times
  at once: the clip field held a folder of 4 MP4s, the engine expected one video or an image
  sequence, and the panel only said "engine stopped". Fixed: the add-on now checks the clip
  first (one video in a folder is taken, several -> "This folder has 4 videos (01.mp4, ...):
  pick one video file"; wrong type; missing file), shows the engine's actual error in a red box,
  shows elapsed time per stage and in total, and Cancel also stops the engine inside WSL
  (pkill on the solve's output folder). Checked in Blender 5.2 (background) on those cases.
- **Keyframe stride** for COLMAP and MegaSaM (`candidates.keyframe_stride`: <= ~30 keyframes per
  second and <= 200 keyframes; synthetic_v1 stays at 1). Real clip 01 (418 frames, 59.94 fps)
  took 1656 s for colmap_global alone at full rate. Refinement fills the frames in between
  (`refine.interpolate_gaps`: slerp / linear start, then BA with CoTracker3 segments); SIFT tracks
  keep the shot length; the tripod gate and the lock test handle keyframe-only SIFT tracks.
  Check on orbit_yard with `--stride 3` (keyframes 0.12 s apart, like stride 7 at 60 fps):
  whole solve 341 s instead of 829 s; selected BA rot max 0.034° (0.013° at full rate), focal
  0.00 %, ATE 0.002 %: targets met; frame-to-frame jitter 0.61 px vs 0.2 px (frames between
  keyframes are held by CoTracker3 segments only). Strides for the test clips: 3, 2, 3, 7.

## 2026-09-29 night — CHECKPOINT 3 DONE, then Phase 2 started (Nihad: "do the next phases yourself")

### Checkpoint 3 result: 19 / 21 synthetic shots meet every target (COLMAP alone: 14 / 21)

`uv run niko bench run --name cp3_full --reuse --force && uv run niko bench report cp3_full`
(masks, tracks and raw candidates reused, refinement and selection recomputed with the final code;
report: `reports/cp3_full/report.html`, `metrics.csv`; lock-test videos: `reports/cp3_full/locktests/`).

| method | targets met | median ATE % | median rot max ° |
|---|---|---|---|
| **niko_selected** (the engine's answer) | **19 / 21** | **0.0008** | **0.0078** |
| oracle_best (best candidate by GT: measures the selector) | 19 / 21 | 0.0008 | 0.0062 |
| colmap_global (checkpoint 2 baseline) | 14 / 21 | 0.0015 | 0.0166 |
| megasam raw | 13 / 21 | 0.0135 | 0.0731 |
| colmap_incremental raw | 12 / 21 | 0.0019 | 0.0151 |
| da3 raw (40 keyframes only, so never "solved") | 0 / 21 | — | — |

| shot | niko picked | rot max ° | focal max % | ATE % | avg error px | targets | COLMAP (cp2) rot ° / focal % |
|---|---|---|---|---|---|---|---|
| car_follow | colmap_incremental+ba | 0.008 | 0.57 | 0.007 | 0.44 | yes | 0.006 / 0.2 |
| crowd_orbit | colmap_global | 0.006 | 0.11 | 0.001 | 0.32 | yes | 0.006 / 0.1 |
| distortion_barrel | colmap_global+ba_k1k2 | 0.005 | 0.13 | 0.000 | 0.29 | yes | 0.139 / 14.7 |
| distortion_pincushion | colmap_global | 0.008 | 0.03 | 0.001 | 0.38 | yes | 0.008 / 0.0 |
| dolly_forward | colmap_global+ba | 0.013 | 86.21 | 1.787 | 0.29 | **no** | failed |
| drone_flyover | colmap_global | 0.017 | 0.12 | 0.013 | 0.27 | yes | 0.017 / 0.1 |
| drone_high_low_parallax | colmap_incremental+ba | 0.008 | 0.46 | 0.001 | 0.33 | yes | 0.019 / 0.8 |
| drone_orbit | colmap_incremental+ba | 0.004 | 0.01 | 0.001 | 0.31 | yes | 0.007 / 0.0 |
| low_texture_walk | megasam+ba | 0.019 | 0.67 | 0.002 | 0.40 | yes | 0.020 / 0.0 |
| noise_heavy | colmap_global | 0.008 | 0.00 | 0.001 | 0.48 | yes | 0.008 / 0.0 |
| orbit_people | colmap_global | 0.006 | 0.01 | 0.001 | 0.33 | yes | 0.006 / 0.0 |
| orbit_yard | colmap_global | 0.019 | 0.05 | 0.001 | 0.32 | yes | 0.019 / 0.0 |
| pan_low_parallax | megasam+ba | 0.029 | 0.06 | 0.000 | 0.30 | yes | 0.110 / 34.5 |
| rolling_shutter_handheld | megasam | 0.114 | 0.15 | 0.019 | 0.48 | yes | 0.129 / 0.3 |
| tripod_rotation_only | colmap_incremental+tripod | 0.003 | 0.01 | 0.000 | 0.24 | yes | 0.038 / 31.5 |
| walk_forward | colmap_global+ba | 0.003 | 0.19 | 0.001 | 0.29 | yes | 0.003 / 0.0 |
| walk_people | colmap_global+ba | 0.005 | 0.03 | 0.000 | 0.31 | yes | 0.005 / 0.0 |
| walk_turn | megasam+ba | 0.007 | 0.01 | 0.001 | 0.28 | yes | 0.011 / 0.0 |
| whip_pan_blur | megasam+tripod | 1.581 | 0.00 | 0.000 | 0.43 | **no** | 27.2 / 36.1 |
| zoom_in | colmap_incremental+tripod_zoom | 0.003 | 0.01 | 0.000 | 0.28 | yes | 1.766 / 204.4 |
| zoom_walk | colmap_global+ba_zoom | 0.009 | 0.17 | 0.004 | 0.34 | yes | 1.289 / 49.9 |

"avg error" = the solve's average tracking error: mean held-out reprojection (SIFT tracks the
solver never saw, observations under 3 px; 97-99.8 % of them are).

What changed after the first full pass (17 / 21), all measured:
- **Auto-select scores on held-out SIFT tracks** (the 12 000 shorter ones refinement never
  sees; `niko.tracks.split_sift`) instead of CoTracker3 segments. CoTracker3 drift is radial in a
  zoom and a 6-dof camera absorbs it into depth: on zoom_in the wrong dolly solution (focal 72 %
  off) scored 0.260 px against 0.307 px for the true rotation + zoom; on SIFT both are 0.188 px
  and the jitter term picks the true one (0.617 vs 0.652) -> zoom_in fixed. Fragile margin (5 %),
  noted. Raw COLMAP candidates did match those SIFT tracks: a small advantage to them
  (orbit_yard 0.207 vs 0.208 px), inside the parsimony margin.
- **Parsimony also needs held-out reprojection within 3 %**, not only the score: real camera
  shake enters every candidate's jitter and compressed the gap (distortion_barrel: k1k2 0.195 vs
  0.321 px on SIFT; before, the no-distortion model won at 18.7 % focal error) -> fixed.
- Runtime: the three camera models on the best candidate only, one outlier round, rotation-only
  solver gated by a homography test (`homography_residual_px` < 1 px: 7 shots). ~215 s refinement
  per shot instead of ~700 s. With 4000 SIFT tracks instead of 8000 the BA was faster but noisier
  (rotation jitter 0.078 vs 0.057 px) and lost to raw COLMAP: kept 8000.

The two misses, honestly:
- **dolly_forward**: pure forward motion does not measure the lens. Measured with a focal sweep
  (`scripts/dev/focal_profile.py`: BA with the focal fixed, scored on held-out SIFT): 232 px ->
  0.189 px, 1493 px (true) -> 0.187 px, 2298 px -> 0.188 px: flat over 10x. No automatic choice
  can be trusted, so the engine now **says so**: `lens_check` compares the solve's focal with the
  learned depth models' (MegaSaM, DA3); |log ratio| > 1 -> "lens uncertain". On the 21 shots it
  fires on dolly_forward only (2.41; all others <= 0.66). And a **known lens** fixes it:
  `niko solve ... --focal-mm 28` -> rot 0.006°, focal 0.00 %, ATE 0.004 % (targets met).
- **whip_pan_blur**: 91.6° whip in 10 blurred frames with a 65.5° field of view: the two sides do
  not overlap and SIFT finds 0-15 matches across the whip. Each side is solved to ~0.01°, the jump
  between them is 2.7° off (rot max 1.58° after global alignment). The per-frame error graph shows
  it (frame 80: 2.33 px). Would need tracking through motion blur; left open.

### Engine additions (all verified)
- `selected/errors.json`: per-frame held-out error (the UI graph); `solve.json` -> `solve_error`
  (average / median px, inlier fraction, worst frame), `lens_check`, `known_lens`.
- **Lock test** `niko locktest <solve_dir>`: MP4 with held-out tracks drawn only where they are
  seen (cross = projected 3D point coloured green < 0.5 / amber < 1 / red px, ring = tracked
  position) and planted ground patches. Five examples in `reports/cp3_full/locktests/`.
- **After Effects export** `niko export-ae <solve_dir>` (also written by every solve):
  `selected/niko_after_effects.jsx` builds comp, footage, keyed 3D camera, 24 track nulls, a
  guide ground grid. AE conventions **measured in After Effects 26.5** (probe script run by
  `AfterFX.exe -r`): orientation = Rx·Ry·Rz, camera looks down +Z with Y down,
  u = W/2 + zoom·x/z (0.000000 px vs our model). Round trip `scripts/dev/ae_check.py` (export ->
  AE -> read back every null's comp position): orbit_yard max 0.00026 px, zoom_in (tripod zoom)
  0.00038 px, distortion_barrel 0.023 px; footage imported from the WSL path. Solves with lens
  distortion get a note in the script (AE cameras are pinholes).
- `--focal-mm / --sensor-mm` (known lens: BA keeps it fixed) and `--reuse` on `niko solve`.
- Levelled world for Blender (`blender.json` -> `world`): ground plane on Z = 0, median depth
  10 units, via an empty that parents camera and points (camera keys stay the engine's values).

### Phase 2 started: Blender add-on `addon/niko_tracker` (installed and enabled in Blender 5.2)
- Workspace tab **"Niko track"** (added next to Layout in every file, user stays on their tab;
  preference to turn off): camera view with footage + points + big colour-rated average error +
  per-frame error strip; free 3D view with the Niko sidebar; sections 1 Clip, 2 Ignore
  (person / car / animal / sky / water + more), 3 Solve (lens Auto / Known mm + sensor),
  4 Result (average error, frames, lens, camera type, worst frame button, this frame, lens
  warning), 5 Use it (create camera, lock test video, send to After Effects, open folder).
- The engine runs in WSL2 as a background job; progress per stage in the panel; Cancel.
- Checks: `tests/blender/addon_background_check.py` (Blender's camera vs the engine's projection,
  orbit_yard 780 projections max 0.00054 px, dolly_forward 722 max 0.003 px; every icon exists);
  `tests/blender/addon_gui_screenshot.py` (real GUI, screenshot); `tests/blender/addon_gui_solve.py`
  (end to end: MP4 on the WSL disk -> Solve camera in Blender -> 7/7 stages -> 0.27 px -> scene
  built, 258 s for 30 frames 540p).
- Crashed Blender 5.2 once (EXCEPTION_ACCESS_VIOLATION in ED_workspace_change) by switching
  workspaces twice in one step: workspace changes are deferred; now arranged by a timer.

### Problems met tonight
- **Out of memory in WSL**: the ground-plane fit used a full SVD (N x N matrix: 26 GB for 57k
  points); WSL's OOM killer took the process and then WSL itself ("Catastrophic failure
  E_UNEXPECTED", fixed by `wsl --shutdown`). Now a 3 x 3 eigenproblem: 7.56 GB -> 0.07 GB.
  A second WSL crash during a 9-value focal sweep in one process was not diagnosed (no log
  survived); the sweep then ran one value per process without trouble. Watch this.

### Waiting for Nihad
- Checkpoint 4 needs **your real clips and the MotionMaster .blend files** (not found in the
  project folder). Everything else for it is ready: video ingest, lock test, After Effects export.
- 2026-09-29 09:06: Nihad added 4 real clips in `D:\Pack\Track Nhad\Test SHot` (his own tracks of
  them were not good; no MotionMaster files: judge by lock test + average error). All 59.94 fps,
  no lens metadata: 01 1080p 418 fr (drone, river + stone bridge, many walkers), 02 1080p 316 fr
  (indoor, people on a sofa fill most of the frame, red wall: hardest), 03 1080p 514 fr (drone
  over a building, waving flag), 04 4K 1323 fr (drone flying toward Dalal bridge, Zakho: mostly
  forward motion, lens may come out uncertain). Run started 09:08 and was **stopped on request**
  (Nihad working); nothing finished. To resume, for c in 01..04:
  `uv run niko solve "/mnt/d/Pack/Track Nhad/Test SHot/$c.mp4" -o $NIKO_HOME/solves/test_$c --prompts person,car,animal,sky,water,flag`
  then `uv run niko locktest $NIKO_HOME/solves/test_$c --scale 0.5`; check lock tests by eye,
  deliver Blender + After Effects files per clip.
- Try the add-on: open Blender -> tab "Niko track" -> 1 Clip -> Solve camera.

## 2026-09-28 night — Checkpoint 3 in progress (Nihad OK'd checkpoint 2)

### Exact track accuracy (new tool: ray-cast GT tracks, `niko.bench.track_truth`)

Every track's query pixel is ray-cast into the shot's `scene.blend`; the hit is projected with the
GT cameras. orbit_yard, 150 frames 1080p (`scripts/dev/track_truth_test.py`):

| frames from query | CoTracker3 (exact queries) | CoTracker3 (model's own query) | SIFT tracks (COLMAP DB) |
|---|---|---|---|
| 1–2 | 0.50 px | **0.34 px** | 0.24 px (1–8) |
| 2–8 | 0.68 | 0.62 | |
| 8–24 | 1.38 | 1.40 | 0.43 (8–48) |
| 24–48 | 2.39 | 2.40 | |
| 48+ | 3.92 | 3.92 | **0.48** |
| all, > 3 px | 31.6 % | 31.7 % | **4.6 %** |

- The earlier "0.88 px" (triangulation proxy) hid CoTracker's **drift**: error grows with distance
  from the query frame. Replacing the query frame with the exact query created a constant +0.3 px
  disagreement; the adapter now keeps the model's own prediction there (bias → +0.12 px).
- Sub-pixel LK refinement at full resolution made tracks slightly worse (0.875 → 0.90 px) — dropped.
- SIFT tracks (verified COLMAP matches → connected components, `colmap` task `sift_tracks`):
  20 000 tracks, median 30 views, no drift.

### Refinement = Ceres bundle adjustment on SIFT + CoTracker segments

- scipy trust-region BA did not converge on poor starts (DA3: "max evaluations", worst error at the
  fixed gauge frame) → moved to Ceres through pycolmap (analytic costs, Schur, Cauchy loss).
- CoTracker segments only (16 frames): every start converges to the *same* wrong optimum
  (rot max 0.257°): short segments cannot stop the path from bending.
- **SIFT + CoTracker segments**, orbit_yard: all candidates → rot max **0.013°**, focal **0.008 %**,
  ATE 0.0008 % (raw COLMAP 0.019° / 0.049 %; DA3 from 40 keyframes 0.485° → 0.014°, 150/150 frames).
- Hard shots (`scripts/dev/hard_shots.py`):
  - **distortion_barrel solved**: k1k2 model from any start → rot 0.0047°, focal 0.12 %,
    k1 −0.0999 / k2 +0.0200 (GT −0.10 / 0.02). COLMAP: 0.14°, focal 15 %.
  - dolly_forward: DA3 + BA rot 0.003°, ATE 0.01 %, focal **3.5 %** (COLMAP 84 %). Pure forward
    motion leaves focal almost unobservable; DA3's focal prior is what gets it to 3.5 %.
  - zoom_in: 6-dof BA with per-frame focal is not enough (best rot 0.52°): the camera does not move,
    so this needs the rotation-only solver (below).
- Rotation-only solver `niko.tripod` (LM + Schur, directions on the sphere, shared or per-frame
  focal): synthetic pan from 1° / 15 % focal error → rot 0.003°, focal 0.002 %; with zoom
  0.009° / 0.014 % (`tests/test_tripod.py`).
- Auto-select: held-out segments, direction-based scoring for rotation-only candidates, and a
  parsimony rule (simplest camera model within 3 % of the best score).

### Full pipeline on one shot, then runtime cuts

`uv run niko bench run --name cp3_full --reuse --shots orbit_yard` (masks/candidates reused from
cp2): selected `colmap_global+ba`, ATE 0.001 %, rot 0.013° max, focal 0.01 %, 829 s for the shot.
Refinement was the slow part (6 BA solves × 68–90 s, 2 tripod solves × 136–193 s), so:

- The three BA camera models run on the best candidate only; the runner-up gets plain BA (every
  good start reaches the same optimum). One outlier round instead of two. 6 → 4 BA solves.
- Kept 8000 SIFT tracks: with 4000, BA ran 32–46 s but its frame-to-frame rotation noise rose
  (0.057 → 0.078 px) and auto-select preferred raw COLMAP (0.019°) over BA (0.014°). With 8000:
  49–62 s per solve, rot 0.013°, BA selected.
- Rotation-only solver only when a homography explains the SIFT tracks
  (`homography_residual_px`: RANSAC homography between frames 15 apart, median transfer residual).
  `scripts/dev/parallax_gate.py` on all 21 shots:

  | shot | H residual px | GT path / depth |
  |---|---|---|
  | tripod_rotation_only | 0.196 | 0 |
  | zoom_in | 0.228 | 0 |
  | drone_orbit | 0.292 | 0.54 (distant, near-planar) |
  | pan_low_parallax | 0.299 | 0.004 |
  | whip_pan_blur | 0.387 | 0 |
  | dolly_forward | 0.720 | 0.89 |
  | drone_high_low_parallax | 0.850 | 0.07 |
  | all other 14 shots | 1.10 – 87 | 0.61 – 3.7 |

  Gate at 1.0 px → the tripod runs on the 7 shots above (drone_orbit and dolly_forward are false
  positives: extra time only, auto-select scores them).
- orbit_yard after the cuts: refinement ≈ 215 s (was ≈ 700 s), same result.

## 2026-09-28 — CHECKPOINT 2 DONE (waiting for Nihad's OK)

Run `cp2_colmap_v1`: 21 synthetic shots × 150 frames, 1920×1080, COLMAP 4.2 global (ex-GLOMAP)
and incremental, SIMPLE_RADIAL single camera, sequential matching, SAM 3.1 masks.
Report: `D:\Pack\Track Nhad\reports\cp2_colmap_v1\report.html` (+ `metrics.csv`).

| | colmap_global | colmap_incremental | best of the two |
|---|---|---|---|
| shots solved (≥ 95 % frames) | 20/21 | 19/21 | 21/21 |
| **targets met** (rot < 0.2°, focal < 2 %, ATE < 1 %) | **14/21** | **12/21** | **14/21** |
| median ATE / rot max (solved shots) | 0.0014 % / 0.018° | 0.0019 % / 0.015° | |
| runtime per shot | 62–345 s | 6–1164 s | |

Easy shots (orbit, walk, drone, people/cars, noise, pincushion, rolling shutter 0.6 frame):
rot max 0.003–0.14°, focal ≤ 0.8 %, ATE ≤ 0.02 %.

**The 7 shots COLMAP cannot do** — exactly the hard cases the brief targets:

| shot | best COLMAP result | why |
|---|---|---|
| zoom_in | ATE 19 %, rot 1.5°, focal 72 % | one shared camera: zoom cannot be modelled |
| zoom_walk | ATE 0.76 %, rot 1.3°, focal 48 % | same |
| tripod_rotation_only | rot 0.03°, focal 31 %, ATE 1.1 % | no parallax: focal and depth unconstrained for SfM |
| pan_low_parallax | rot 0.04°, focal 35 % | 5 cm nodal offset only |
| distortion_barrel | rot 0.14°, focal 15 % | SIMPLE_RADIAL (k1 only) cannot fit k1 −0.10, k2 0.02 at 16 mm |
| dolly_forward | global: no model after 16 min; incremental focal 84 %, ATE 1.6 % | pure forward motion |
| whip_pan_blur | global: rot 27°, ATE 13 %; incremental: 2/150 frames | 90° whip with 0.5-frame blur |

Other failures: incremental broke on walk_turn (rot 63°) and failed to initialise on
low_texture_walk (2/150); global solved both.

Auto-select (no GT) picked a target-meeting method in all 14 shots where one existed.
**Weakness found:** on whip_pan_blur the broken global solve scored a *low* held-out
reprojection (0.39 px): tracks do not survive the whip, so each half is self-consistent and the
metric cannot see that the halves are misaligned. Checkpoint 3 must add a continuity check across
the whole shot (path jumps / track coverage) to the select score.

## 2026-09-28 — Checkpoint 2 in progress (Nihad OK'd checkpoint 1)

### Generator (verified before rendering the set)

- New content: town scene (drone shots, 5 m+ streets), mannequin people (legs/arms swing) and car
  proxies, 7 camera paths (orbit, walk, line, pan, whip, follow, zoom via `lens_end`), hand shake
  (1/f sum of sines), motion blur, Cycles rolling shutter, lens distortion, sensor noise,
  low-contrast textures, props kept off the camera path.
- **Distortion GT**: overscan render + exact inverse remap vs. OpenCV model with (k1,k2) =
  (-0.12, 0.03): max **0.034 px** (60 markers, `test_distorted_markers`).
- **Rolling shutter GT**: Cycles TOP rolling shutter, readout 0.6 frame, 1°/frame pan: row y is
  exposed at `frame + (y/H - 0.5) * readout` — max **0.025 px**; a global-shutter model would be
  off by 7.2 px (`test_rolling_shutter_markers`). GT pose of a frame = pose of its middle row.
- `import-camera` on a generated `scene.blend` reproduces its GT exactly (150 frames, max |ΔR|,
  |ΔC|, |ΔK| = 0.0).
- Preview of all 21 shots (480×270, 16 frames): 33 s; fixes after looking at the contact sheet:
  wider camera corridor in orbits (±3.5 m), drone fly-over raised to 60 m.
- SAM 3.1 on mannequin proxies (orbit_people, 5 frames): 1 of 4 people found, car not found;
  most proxies were occluded by props in those frames. Checkpoint 3: render GT dynamic masks
  (object index pass) to measure mask recall properly.
- Full set rendered: 21 shots × 150 frames, 1920×1080, 48 spp, **1 807 s total** (66–108 s per
  shot), 12 GB. `uv run niko bench generate` (contact sheet in the set folder).

### Long clips: VRAM and the Windows GPU watchdog (found on the first 1080p/150-frame shot)

- SAM 3.1 on the whole 150-frame clip reserved **26.8 GB** (151 s); with CoTracker the 32 GB GPU
  was full. SAM 3 now runs in 50-frame chunks with frames kept in RAM (masks are a per-frame
  union, so object identity across chunks does not matter).
- CoTracker3 on the whole clip: 384×512 → 24.7 GB, 35 s, **2.65 px** median vs GT at 1080p
  (45 % > 3 px); 544×960 → *"CUDA driver error: device not ready"* (Windows TDR: one GPU call
  too long on the display GPU).
- **Windowed tracking** (48-frame windows, 8 overlap, tracks chained across windows, 400
  queries per call), 544×960, same shot: **0.88 px** median vs GT (12 % > 3 px), **10.4 GB**,
  17 s, 5 495 tracks, tracks up to the full 150 frames (median 48).
- COLMAP global and incremental now share one feature/match database (identical settings
  checked); smoke: incremental 16 s → 12 s. COLMAP results vary slightly run to run
  (smoke rot max 0.023° vs 0.029°): it is not bit-deterministic.
- Synthetic runs use prompts person, car, sky (no animals / water in these scenes).
- First 1080p/150-frame shot end to end (orbit_yard, before the matcher change): masks 70 s,
  tracks 21 s (10.4 GB), colmap_global 310 s, colmap_incremental 273 s; both 150/150 frames,
  ATE 0.001 %, rot max 0.020° / 0.025°, focal 0.05 % / 0.06 % — targets met. COLMAP now uses
  sequential matching (overlap 20 + quadratic) above 60 frames, as usual for video.

### Status when stopping for the day (2026-09-28 evening)

- Running: `niko bench run --name cp2_colmap_v1 --methods colmap_global,colmap_incremental`
  then `niko bench report cp2_colmap_v1` (log: `$NIKO_HOME/cp2_run.log`). Resumable: shots with a
  `solve.json` are skipped, an interrupted shot is redone. To resume:
  ```
  uv run niko bench run --name cp2_colmap_v1 --methods colmap_global,colmap_incremental
  uv run niko bench report cp2_colmap_v1        # -> D:\Pack\Track Nhad\reports\cp2_colmap_v1\report.html
  ```

### Checkpoint 3 preview (code only, not run on the benchmark): bundle adjustment

`niko.refine` (scipy trust-region, sparse Jacobian, Cauchy loss, 2 outlier rounds, PnP for
missing frames, shared focal + k1). On the smoke shot (`scripts/dev/refine_check.py`):

| candidate | rot max before → after | held-out reproj before → after |
|---|---|---|
| colmap_global | 0.024° → 0.079° | 0.555 → 0.525 px |
| colmap_incremental | 0.032° → 0.071° | 0.555 → 0.526 px |
| da3 | 0.289° → 0.289° | 0.782 → 0.521 px |
| megasam | 0.293° → 0.291° | 0.563 → 0.529 px |

Honest reading: BA against CoTracker3 tracks alone lowers the reprojection error but makes
COLMAP's rotations *worse* and does not fix the DA3/MegaSaM worst frame. The tracks (~0.5 px
median, 12 % > 3 px) are less precise than COLMAP's SIFT, so the BA fits tracker error. Next:
find the frame(s) behind the 0.29° max, sub-pixel track refinement, and BA with COLMAP features
+ tracks together. Refinement is not in `solve` yet.

## 2026-09-28 — CHECKPOINT 1 DONE (waiting for Nihad's OK)

**`niko doctor`: ALL GREEN** (30 checks). **pytest: 38 passed. Smoke test: passed, 140 s**
(generate a 30-frame 960×540 shot + the whole pipeline; 152 s including pytest start-up).

### Smoke run (`pytest -m smoke -s`)

| Stage | Result |
|---|---|
| ingest | 30 frames 960×540 @ 25 fps |
| masks (SAM 3.1, 5 prompts) | 50.5 s; 1 sky object, 17.5 % of pixels excluded; no person/car (correct) |
| tracks (CoTracker3 at 540×960) | 3.1 s, 4 283 tracks, 2 714 visible/frame, 13.2 GB VRAM |
| colmap_global / colmap_incremental | 22.5 s / 16.6 s, 30/30 frames |
| megasam / da3 | 34.1 s / 12.1 s, 30/30 frames |
| auto-select (no GT) | colmap_global 0.943 px < colmap_incremental 0.953 < megasam 0.977 < da3 2.090 |
| export | selected/cameras.json, points.ply, import_blender.py |

Against ground truth (not used by select):

| Method | ATE | rot max | focal max | targets |
|---|---|---|---|---|
| **colmap_global (selected)** | 0.002 % | 0.023° | 0.08 % | met |
| colmap_incremental | 0.002 % | 0.024° | 0.06 % | met |
| megasam (unrefined) | 0.006 % | 0.263° | 1.06 % | rotation missed |
| da3 (unrefined) | 0.017 % | 0.263° | 1.71 % | rotation missed |

The smoke shot is an easy orbit; this proves the plumbing, not the accuracy on hard shots.

### Track accuracy (CoTracker3 vs GT cameras, smoke shot, `scripts/dev/track_accuracy.py`)

| Model resolution | median | inliers (<3 px) | >3 px outliers | VRAM |
|---|---|---|---|---|
| 384×512 (CoTracker default) | 1.00 px | ~0.70 px | 15.9 % | 5.2 GB |
| 540×960 (native proxy) | 0.67 px | ~0.50 px | 13.2 % | 13.2 GB |

Constant signed bias of about +0.10 px in x in both runs: to investigate in checkpoint 3
(sub-pixel track refinement). `solve` now tracks at native proxy size up to 960×540.

### Environments (exact versions in `backends/<name>/requirements.lock.txt`)

| Env | Python | Key versions | Repo commit | Probe |
|---|---|---|---|---|
| niko | 3.12.3 | numpy 2.5, scipy 1.18, opencv-headless, jsonschema, huggingface-hub 2.0 | — | — |
| sam3 | 3.12.3 | torch 2.10.0+cu128, numpy 1.26.4 | sam3 @ 2345a4ad109a | 3-frame session OK |
| cotracker | 3.12.3 | torch 2.10.0+cu128 | co-tracker @ 82e02e802975 | tiny run OK |
| colmap | 3.12.3 | pycolmap-cuda12 4.2.0 | — | GPU SIFT on sm_120 OK |
| megasam | 3.12.3 | torch 2.10.0+cu128, numpy 1.26.4, no xformers | mega-sam @ a27b4e633c5c, base @ ee9ac6af512c | lietorch/DROID ext OK |
| da3 | 3.12.3 | torch 2.10.0+cu128, xformers 0.0.35 | Depth-Anything-3 @ 3d835ec1a580 | 3-view run OK |

Disk: VHDX 44 GB (envs 9.8 GB with torch hard-linked, checkpoints 13 GB, Blender 1.6 GB).

### Porting / fixes needed on Blackwell + current libraries (all scripted, re-runnable)

- **MegaSaM** (`backends/megasam/port.py`): setup.py hardcoded sm_70–86 → arch from
  `TORCH_CUDA_ARCH_LIST=12.0`; `x.type()` → `x.scalar_type()` in DISPATCH macros (41 places);
  `torch_scatter` → pure-PyTorch shim (`backends/megasam/shims`); Depth-Anything no longer
  downloads unused DINOv2 weights; UniDepth loads from `checkpoints/unidepth_v2_vitl14`;
  `wandb` import optional; **xformers removed** from this env (on sm_120 xformers 0.0.35 only
  has FA2 kernels, fp16/bf16 only, and DINOv2/UniDepth run fp32); UniDepthV2's
  `xformers.components.NystromAttention` (removed upstream) replaced by an exact stand-in of the
  path it takes (`niko_megasam/nystrom.py`: seq dim = heads ≤ 8 < 128 landmarks → dense SDPA).
  Needs `python3.12-dev` for the build (added to `00_system_root.sh`).
- **SAM 3** (`backends/sam3/port.py`): `start_session` passes `offload_state_to_cpu` that the
  3.1 multiplex `init_state()` does not accept → accepted (False only). `use_fa3=False`
  (FA3 is Hopper-only). Undeclared deps added: `pycocotools`, `psutil`.
- **DA3**: undeclared dep `addict` added.
- **COLMAP**: none — the pycolmap-cuda12 4.2.0 wheel runs GPU SIFT on sm_120, no source build.

### Reproduce from scratch (inside WSL, repo at `$NIKO_REPO`)

```
sudo bash scripts/wsl/00_system_root.sh          # or: wsl -d Ubuntu-24.04 -u root --exec bash ...
bash scripts/wsl/10_user.sh
bash scripts/wsl/20_blender.sh
UV_PROJECT_ENVIRONMENT=$NIKO_HOME/envs/niko uv sync
for b in cotracker colmap sam3 da3 megasam; do bash backends/$b/install.sh; done
HF_HUB_DISABLE_XET=1 uv run python scripts/fetch_checkpoints.py sam3.1_multiplex cotracker3_offline \
    da3_nested_giant_large_1.1 depth_anything_v1_vitl14 unidepth_v2_vitl14
uv run niko doctor
uv run pytest -q && uv run pytest -m smoke -s
uv run niko solve <clip or bench shot> -o <out>
```

### Known limits carried into checkpoints 2–3

- No refinement yet: candidates are scored unrefined (`solve.json: "refined": false`).
- Jitter in the select score is a raw second difference (includes real acceleration).
- DA3 runs up to 64 keyframes per call; longer clips leave the other frames invalid until
  chunked inference lands.
- peak VRAM is not measured for colmap/megasam (their GPU work runs outside torch in the adapter
  process); an nvidia-smi sampler is planned for the benchmark report.

## 2026-09-28 (afternoon) — WSL2 up, first envs green; downloads PAUSED

**Downloads paused at Nihad's request (work network).** Nothing is downloading now.

| Item | State |
|---|---|
| WSL 2.7.14, Ubuntu 24.04.5 on `D:\WSL\Ubuntu-24.04`, user `rudaw` | done |
| System packages + CUDA toolkit 12.8 (nvcc V12.8.93), GPU sm 12.0 visible | done (`scripts/wsl/00_system_root.sh`) |
| uv 0.12.19, `~/niko`, env file `~/.config/niko/env.sh` | done (`10_user.sh`) |
| Linux Blender 5.2.2 LTS (sha256 OK), Cycles GPU = **CUDA** (OptiX is not available in WSL) | done (`20_blender.sh`) |
| `niko` orchestrator env (Python 3.12) | done; all 37 tests pass in WSL (run as 30 core + 3 Blender + 2 ingest + 2 evaluate; marker render max 0.016 px) |
| `cotracker` env: torch 2.10.0+cu128, co-tracker @ 82e02e80, `scaled_offline.pth` (102 MB) | **GREEN** probe: sm_120 kernels, tiny run OK |
| `colmap` env: pycolmap-cuda12 4.2.0 | **GREEN** probe: has_cuda, GPU SIFT 2364/2415 keypoints on sm_120 (no source build needed) |
| `sam3` env: torch installed, sam3 @ 2345a4ad cloned | interrupted before `pip install -e sam3` |
| `da3` env: torch installed, DA3 @ 3d835ec1 cloned | interrupted before xformers / `pip install -e` |
| DA3NESTED-GIANT-LARGE-1.1 checkpoint 6.760 GB | done, sha256 8ebe871a022ed58d… |
| SAM 3.1 checkpoint (3.5 GB) | access ACCEPTED on HF; HF_TOKEN not yet set in WSL |
| MegaSaM env + priors (~3 GB) | not started |

Gotchas found:
- `wsl.exe -- cmd` re-parses the command through a shell (variables expand too early): use `wsl.exe --exec bash -lc '...'`.
- A sourced env file whose last line is `[ -f x ] && . x` returns 1 and kills `set -e` scripts: use `if … fi`.
- HF Xet downloads stalled after 134 MB; `HF_HUB_DISABLE_XET=1` (plain HTTP) worked.
- Full `git clone` of co-tracker was very slow: clones are now `--depth 1`.
- Smoke shot renders in WSL in 3.6 s (CUDA); identical scene_size to the Windows render (38.528 m).

## 2026-09-28 (later) — ground truth verified against rendered pixels; smoke shot generator

- WSL2 needs **Virtual Machine Platform**, which is disabled on this PC (`Win32_OptionalFeature`,
  no hypervisor present) → one admin command + one reboot. Nihad will do it after work
  (Adobe apps open; no reboot from our side).
- **Rendered-marker test** (`tests/test_render_markers.py`): tiny emissive spheres at known 3D
  points, rendered by Cycles (OptiX, RTX 5090, 256 spp), 16-bit linear PNG. Camera set with native
  Blender settings → `nikobpy.export_cameras` → `niko.blendercam` → projection vs. intensity centroids:

  | case | mean | max | bias (x, y) |
  |---|---|---|---|
  | 1920×1080, 35 mm, no shift | 0.005 px | 0.014 px | (−0.001, +0.001) |
  | 1280×720, shift 0.1 / −0.07 | 0.006 px | 0.015 px | (−0.000, −0.000) |
  | 1080×1920 portrait, AUTO fit | 0.006 px | 0.014 px | (+0.000, +0.001) |
  | 1600×900, VERTICAL fit, shifts | 0.007 px | 0.016 px | (−0.001, +0.000) |

  216 markers. The GT path (Blender → cameras.json) is exact to ~0.02 px, far below the
  0.5 px targets, and the pixel-origin convention is confirmed on real renders.
- **Synthetic generator, first version** (`blender/gen_shot.py`, `scene_lib.py`, `niko.bench.generate`):
  procedural textured yard (ground, 30 props, walls, sun, sky), orbit camera, GT exported by
  evaluating the camera per frame. Smoke spec `configs/shots/smoke_orbit.json`
  (30 frames, 960×540, 32 spp): **3.1 s render, 4.8 s total** on OptiX. GT: fx = 746.667 px
  (= 28/36 × 960), path length 6.72 m (= 35° arc × 11 m), scene_size 38.5 m.
  First render had low-contrast textures; materials now overlay mid-scale Voronoi cells and
  high-contrast fine noise.
- pytest: **33 passed**.

```
uv run pytest -q -s
uv run python -c "from niko.bench.generate import generate_shot; generate_shot('configs/shots/smoke_orbit.json', r'<out_dir>')"
```

Open points for checkpoint 2: repetitive brick ground may create ambiguous matches (keep one
shot like that on purpose, not all); "person"/"car" proxies must look enough like people/cars for
SAM 3 text prompts to fire — consider Blender's bundled human base meshes.

## 2026-09-28 — Checkpoint 1 started (blocked: WSL2 not installed)

### Machine as found

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| CPU / RAM | Intel Core Ultra 9 285K (24 threads), 127 GB, virtualization enabled in firmware |
| GPU | RTX 5090, 32 GB, driver 616.56, sm_120 (`nvidia-smi`) |
| Disk | one Samsung 9100 PRO 4 TB NVMe: C: 189 GB free, D: 687 GB free |
| WSL | **not installed** (`wsl --status`) |
| Blender (Windows) | 5.2.2 LTS at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` |
| Windows tools | Python 3.12, uv 0.12.5, git; no ffmpeg, no nvcc |

### Works (verified)

- Repo skeleton, docs (`docs/BRIEF.md`, `DESIGN.md`, `SCHEMA.md`, `LICENSES.md`, `schemas/cameras.schema.json`).
- `niko.geometry` (OpenCV ↔ Blender), `niko.sim3` (Umeyama), `niko.metrics`, `niko.camio`
  (cameras.json read/write/validate), `niko doctor`, CLI skeleton.
- **pytest: 32 passed** (Windows, Python 3.12.10, numpy 2.5.3, scipy 1.18.1, jsonschema 4.26.0).
  Includes the camera-convention test against real Blender 5.2.2:
  - Blender → OpenCV: max difference **3.78e-03 px** over 5 905 in-frame points, 60 random cameras
    (all sensor fits, shifts, pixel aspect, object scale, 100 %/50 % resolution).
  - OpenCV → Blender: max difference **2.24e-03 px** over 7 754 points, 40 random cameras.
  - The residual is Blender's float32 precision; a convention mistake would be ≥ 0.5 px.
- `niko doctor` runs; on Windows it correctly reports 16 failures (it is meant for WSL).

Reproduce (Windows, from the repo root):

```
$env:UV_PROJECT_ENVIRONMENT = "$env:TEMP\niko-venv-win"
uv sync --python 3.12
uv run pytest -q -s
```

### Findings to remember

- **Blender quirk:** with a render percentage that does not divide the resolution (e.g. 37 %),
  Blender renders an integer size (710×399) but `world_to_camera_view` keeps the unscaled 16:9
  frame — up to 8.6 px disagreement inside Blender itself. Our conversion follows the rendered
  integer size. Rule for the generator: always render at 100 %. To confirm against an actual
  render in checkpoint 2 (render markers at known 3D points, measure centroids).
- **GLOMAP is now part of COLMAP** (4.0, 2026-03-14; standalone repo archived 2026-03-09).
  Current release 4.2.0 (2026-08-31): `colmap global_mapper`, masks, ALIKED/LightGlue/LoMa via ONNX,
  pycolmap CUDA wheels for Linux. Baseline = COLMAP 4.2 incremental + global mapper.
- **MegaSaM upstream pins Python 3.10 / PyTorch 2.0.1 / CUDA 11.8** — no sm_120 support. Needs a
  port to torch 2.10 + cu128 with its DROID-SLAM CUDA extensions rebuilt for 12.0.
- **SAM 3.1** checkpoint `sam3.1_multiplex.pt` (3.5 GB) is gated on Hugging Face: access must be
  requested on the `facebook/sam3.1` page with the account that owns HF_TOKEN.
- xformers ≥ 0.0.34 targets torch ≥ 2.10 but an open upstream issue reports problems; checked at install.

### Blocked / next

1. **Nihad:** install WSL2 + Ubuntu 24.04 on D: (admin + reboot; commands in the chat message).
2. **Nihad:** request access to `facebook/sam3.1`; put the token in `~/.config/niko/secrets.sh` inside WSL.
3. Then: `scripts/wsl/00_system_root.sh` (apt + CUDA 12.8 toolkit), `10_user.sh` (uv, `$NIKO_HOME`),
   backend envs one by one with a probe each, checkpoints, Linux Blender 5.2, smoke test.

### Not yet written / not yet run

- Backend envs and adapters (sam3, cotracker, colmap, megasam, da3) — written when WSL exists so
  each one is tested as it is built.
- `scripts/wsl/*.sh` are written but **not yet run** (no WSL).
- Pipeline stages, smoke-test shot, generator, report.
