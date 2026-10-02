# File schemas

Every stage reads and writes plain files. This document is the contract; the code in
`src/niko/camio.py` and `docs/schemas/cameras.schema.json` enforces it.

## Conventions used everywhere

| Topic | Rule |
|---|---|
| Camera axes | OpenCV: +X right, +Y down, +Z forward (looking direction). |
| Extrinsics | world→camera: `x_cam = R @ X_world + t`. Camera centre `C = -Rᵀ t`. |
| Pixel coordinates | `corner` origin: (0, 0) is the top-left **corner** of the top-left pixel; the centre of pixel (col, row) is (col + 0.5, row + 0.5). u grows right, v grows down. |
| Intrinsics | `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`, in pixels of the full-resolution frame. Skew is always 0. |
| Distortion | OpenCV order `[k1, k2, p1, p2, k3]`, applied to normalised undistorted coordinates (`x_d = x (1 + k1 r² + k2 r⁴ + k3 r⁶) + tangential`). Radial-only models set `p1 = p2 = k3 = 0`. |
| Time | Frames, never milliseconds. `i` = 0-based index into the extracted frames, `frame` = the clip's own frame number (Blender timeline). |
| Units | Ground truth: metres. Solves: arbitrary scale unless stated in `conventions.units`. |
| Masks | uint8 PNG, **255 = excluded** (dynamic object / sky / water), 0 = static background. Adapters invert for tools that use the opposite rule (COLMAP: 0 = ignore). |

## `cameras.json`

```jsonc
{
  "schema": "niko.cameras/1",
  "producer": {                        // who made this file
    "method": "colmap_global",         // gt | colmap_incremental | colmap_global | megasam | da3 | refined:<method> | import:<blend> | ...
    "version": "niko 0.1.0",
    "created": "2026-09-28T10:00:00Z",
    "notes": ""
  },
  "clip": {
    "name": "shot_orbit_01",
    "width": 1920, "height": 1080,     // full-resolution pixels; K refers to this size
    "fps": 25.0,
    "frame_start": 1,                  // clip frame number of i = 0
    "n_frames": 150
  },
  "conventions": {
    "camera": "opencv",
    "extrinsics": "world_to_camera",
    "pixel_origin": "corner",
    "world_up": "+z",                  // "+z" for GT / aligned exports, "unknown" for raw solves
    "units": "meters"                  // "meters" | "arbitrary"
  },
  "intrinsics_mode": "shared",         // shared | per_frame_focal | per_frame (informational)
  "distortion_model": "opencv5",       // [k1, k2, p1, p2, k3]
  "rolling_shutter": null,             // or {"readout": 0.5, "direction": "top_to_bottom"}; readout = fraction of a frame
  "points": "points.ply",              // optional, path relative to this file
  "frames": [
    {
      "i": 0,
      "frame": 1,
      "valid": true,                   // false = method did not register this frame; K/R/t are then null
      "K": [[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]],
      "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
      "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      "t": [0.0, 0.0, 0.0]
    }
  ],
  "extra": {}                          // free-form per-method data (scene_size, timings, ...)
}
```

Semantic checks done by `camio.validate` on top of the JSON Schema:
`R` orthonormal with det +1 (tol 1e-5), `fx, fy > 0`, `K[2] == [0, 0, 1]`, `K[0][1] == 0`,
`K[1][0] == 0`, `len(frames) == clip.n_frames`, `frames[k].i == k`,
`frames[k].frame == frame_start + k`.

### Conversion to Blender

Blender cameras look down −Z with +Y up. The world frame is unchanged; only the camera axes flip:

```
R_c2w_blender = Rᵀ · diag(1, −1, −1)
matrix_world  = [[R_c2w_blender, C], [0, 0, 0, 1]],   C = −Rᵀ t
```

Intrinsics, with `sensor_fit = 'HORIZONTAL'` and a chosen `sensor_width` (default 36 mm):

```
lens    = fx · sensor_width / W
shift_x = (W/2 − cx) / W
shift_y = (cy − H/2) · (fx/fy) / W
pixel aspect x : y = 1 : fx/fy      (or fy/fx : 1 when fy > fx; Blender needs both ≥ 1)
```

The reverse direction (`geometry.blender_to_K`) implements Blender's own
`BKE_camera_params_compute_viewplane` for every `sensor_fit` (AUTO / HORIZONTAL / VERTICAL)
and pixel aspect, and is tested against Blender 5.2's `world_to_camera_view`.
Blender's movie-clip "Polynomial" distortion (k1, k2, k3) is the same radial model as OpenCV's.

## `points.ply`

Binary little-endian PLY, one vertex per 3D point, same world frame as `cameras.json`:
`float x, y, z; uchar red, green, blue;` and optionally `int track_id; float error_px`.

## Shot work directory (`$NIKO_HOME/runs/<run>/<shot>/`)

```
shot.json                     ingest metadata (below)
frames/000000.jpg …           full-res frames, JPG q95 (or .png)
proxy/000000.jpg …            proxy, height 1080 (not upscaled if the source is smaller)
masks/000000.png …            full-res uint8 masks, 255 = excluded
masks/masks.json              prompts, per-prompt coverage per frame, model + checkpoint
tracks/tracks.npz             point tracks (below)
candidates/<method>/          cameras.json, points.ply, result.json, log.txt
candidates/<method>+<mode>/   refined: cameras.json, points.ply, ba_input/ba_output.npz, refine.json
                              (modes ba, ba_k1k2, ba_zoom, tripod, tripod_zoom, ba_lens, ba_lens_k1k2)
sift/sift_tracks.npz          multi-view SIFT tracks from the COLMAP database (xy [T,N,2], vis [T,N])
selected/                     the answer: cameras.json, points.ply, errors.json, blender.json,
                              import_blender.py, niko_after_effects.jsx, locktest.mp4 (niko locktest)
solve.json                    every stage, every candidate's score, selection, solve_error, lens_check
```

### `solve.json` (fields the add-on reads)

```jsonc
{
  "selected": "colmap_global+ba",
  "solve_error": {"average_px": 0.32,        // mean held-out reprojection of observations < 3 px
                  "median_px": 0.21, "inlier_fraction": 0.99, "source": "sift",
                  "frames": 150, "n_frames": 150,
                  "worst_frame": {"frame": 150, "mean_px": 0.49}},
  "lens_check": {"uncertain": false,
                 "reasons": [],                // "depth_models" and / or "equally_good_fits"
                 "selected_focal_px": 1492.6,
                 "log_ratio": 0.007,           // |log(focal_learned / focal_solved)|, > 1 = "depth_models"
                 "learned_focal_px": {"megasam": 1497.6, "da3": 1502.5},
                 // candidates within 10 % of the best held-out reprojection (not for a tripod pick);
                 // lenses more than 10 % apart = "equally_good_fits"
                 "spread": {"equally_good": {"colmap_global+ba": 1492.6, "colmap_global": 1495.1},
                            "focal_px": [1492.6, 1495.1], "spread_pct": 0.17}},
  "known_lens": {"focal_mm": 28, "sensor_mm": 36, "focal_px": 1493.3},  // only with --focal-mm
  // breaks in tracking: fewer than 50 SIFT tracks tie the frames before to the frames after
  // (clip frame numbers); [] = none found, absent = too few SIFT tracks to tell
  "track_gaps": [{"from_frame": 72, "to_frame": 80, "tracks": 0}]
}
```

### `selected/errors.json`

Per frame of the selected camera: `median_px`, `mean_px` (observations < `inlier_px`), `n_obs`
(lists, `null` where the frame has no held-out observation), plus `frame_start`, `source`
(`sift` | `cotracker`), `inlier_px`.

### `selected/blender.json`

The values `import_blender.py` uses, for the add-on: `width`, `height`, `fps`, `fps_int`,
`frame_start`, `sensor_width`, `pixel_aspect_x/y`, `frames[]` (`frame`, `valid`, `matrix_world`
4x4 Blender camera in the solve's world, `lens` mm, `shift_x`, `shift_y`), `frames_dir`,
`first_frame_file`, `points`, and `world`: a 4x4 taking the solve's world to a levelled Blender
world (ground plane on Z = 0, first camera looking along +Y, median depth 10 units), applied by
the add-on as an empty that parents camera and points.

### `shot.json`

```jsonc
{
  "schema": "niko.shot/1",
  "source": "/path/to/clip.mp4",
  "width": 3840, "height": 2160, "fps": 23.976, "n_frames": 240, "frame_start": 1,
  "frame_format": "jpg", "jpg_quality": 95,
  "proxy": {"width": 1920, "height": 1080, "scale_x": 0.5, "scale_y": 0.5},  // proxy px = full px * scale
  "lens": {"focal_mm": null, "focal_35mm": null, "sensor_width_mm": null, "source": "ffprobe|exif|none"},
  "codec": "h264", "pix_fmt": "yuv420p", "rotation": 0
}
```

### `tracks/tracks.npz`

| Key | Shape / dtype | Meaning |
|---|---|---|
| `xy` | `[T, N, 2]` float32 | position in full-res pixels, corner origin |
| `vis` | `[T, N]` bool | visible (not occluded) |
| `conf` | `[T, N]` float32 | tracker confidence 0..1 |
| `query_frame` | `[N]` int32 | frame index where the track was seeded |
| `holdout` | `[N]` bool | reserved for held-out reprojection scoring, never used in BA |
| `vis_prob` | `[T, N]` float16 | raw visibility probability (`vis = vis_prob > 0.5`, inside image, outside masks) |
| `meta` | 0-d str (JSON) | backend, model, grid, resolution used, runtime |

### `candidates/<method>/camera_raw.npz` (written by a backend)

| Key | Shape / dtype | Meaning |
|---|---|---|
| `K` | `[T, 3, 3]` float64 | full-resolution intrinsics, corner origin |
| `dist` | `[T, 5]` float64 | OpenCV `[k1, k2, p1, p2, k3]` |
| `R`, `t` | `[T, 3, 3]`, `[T, 3]` float64 | world→camera, OpenCV axes |
| `valid` | `[T]` bool | frame registered by the method |
| `meta` | 0-d str (JSON) | method, sizes, intrinsics mode |

The orchestrator (`niko.pipeline.candidates`) checks it (finite values, proper rotations) and
writes the validated `cameras.json` next to it. Backends never write cameras.json themselves.

## Backend job protocol

`python -m niko_<backend> job.json` — `job.json`:

```jsonc
{"schema": "niko.job/1", "task": "masks|tracks|camera|probe",
 "shot_dir": "...", "out_dir": "...", "options": {...}}
```

The backend writes its outputs into `out_dir` and finally `out_dir/result.json`:

```jsonc
{"schema": "niko.result/1", "ok": true, "error": null, "runtime_s": 12.3,
 "peak_vram_mb": 8123.0, "versions": {"torch": "2.10.0+cu128", "repo_commit": "…"},
 "outputs": {"cameras": "cameras.json"}}
```
