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
telemetry.npz                 per-frame drone telemetry when the video has some (niko.telemetry, below)
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
                 "reasons": [],                // "depth_models", "equally_good_fits", "hardly_turns"
                 "max_turn_deg": 20.7,         // the camera's largest turn from the first frame;
                                               // < 1 deg (it only translates) = "hardly_turns"
                 "selected_focal_px": 1492.6,
                 "log_ratio": 0.007,           // |log(focal_learned / focal_solved)|, > 1 = "depth_models"
                 "learned_focal_px": {"megasam": 1497.6, "da3": 1502.5},
                 // candidates within 10 % of the best held-out reprojection (not for a tripod pick);
                 // lenses more than 10 % apart = "equally_good_fits"
                 "spread": {"equally_good": {"colmap_global+ba": 1492.6, "colmap_global": 1495.1},
                            "focal_px": [1492.6, 1495.1], "spread_pct": 0.17}},
  // a known lens: --focal-mm ("source": "given"), the lens the camera recorded ("source": "camera
  // metadata": Sony's 35 mm equivalent, constant through the clip, x diagonal px / 43.27, x the focus
  // factor for the median focus distance, "focus_m" / "focus_factor": thin lens, v / f) or the
  // camera's profile ("source": "camera profile", with "uncertainty_pct", "calibrated") or, when the raw
  // camera turns less than 1 deg, the maker's spec ("source": "maker's lens spec": DJI FC9113 / FC9313,
  // 24 mm equivalent, +-10 %); no lens_check then
  "known_lens": {"focal_mm": 28, "sensor_mm": 36, "focal_px": 1493.3, "source": "given"},
  "camera": {"id": "DJI FC9113", "name": "DJI Air3s (DJI FC9113)", "key": "...", "source": "telemetry"},
  "true_up": {"source": "gimbal", "up": [0.01, -0.99, 0.12]},  // up in the solve's world, when known
  // the sun at the recording (GPS place, container creation_time in UTC, north from the GPS track
  // or the gimbal's compass); "direction" points towards the sun in the solve's world
  "sun": {"direction": [0.1, -0.2, 0.97], "azimuth_deg": 288.9, "elevation_deg": 1.8,
          "utc": "2026-08-09T15:56:01+00:00", "north": [0.3, 0.1, 0.9], "heading_from": "gps",
          "lat": 37.14, "lon": 42.69},
  // breaks in tracking: fewer than 50 SIFT tracks tie the frames before to the frames after
  // (clip frame numbers); [] = none found, absent = too few SIFT tracks to tell
  "track_gaps": [{"from_frame": 72, "to_frame": 80, "tracks": 0}],
  // real-world size. From the drone's telemetry when it has some and it is reliable (niko.scale.
  // telemetry_scale: GPS fitted to the camera path, else the barometric altitude), else from the
  // two single-image metric depth models ("source": "depth_models", reliable when <= 50 % apart);
  // null for a tripod or when nothing measured it. blender.json is in metres only when "reliable".
  "metric_scale": {"metres_per_unit": 12.86, "reliable": true, "source": "gps", "uncertainty_pct": 1.0,
                   "telemetry": {"gps": {"metres_per_unit": 12.86, "fixes": 284, "extent_m": 143.2,
                                         "residual_median_m": 0.26, "halves_apart_pct": 0.75,
                                         "uncertainty_pct": 1.0, "reliable": true},
                                 "altitude": {"metres_per_unit": 12.62, "readings": 127, "change_m": 17.6, "...": 0},
                                 "gps_vs_altitude_pct": 1.84},
                   "depth_models": {"metres_per_unit": 7.65, "agree_pct": 243.9, "reliable": false,
                                    "models": {"unidepth_v2": {}, "da3_nested": {}}, "source": "depth_models"}}
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
world (up from the cameras' level right axes when the camera turns, else from the ground plane;
the ground on Z = 0; first camera looking along +Y), applied by the add-on as an empty
that parents camera and points (with the drone's gimbal, up is true gravity: `true_up`). `units`
says what one Blender unit is: `{"kind": "metres_gps" | "metres_altitude", "metres_per_unit",
"uncertainty_pct"}` from the drone's telemetry, `{"kind": "metres_estimated", "metres_per_unit",
"agree_pct"}` when the metric depth models agree (then 1 unit is about 1 m), else `{"kind":
"arbitrary"}` (median depth 10 units). `footage_frames`: true when Blender and After Effects show the engine's frames instead of the video
(variable frame rate, an upright phone video, interlaced). `sun`: as in solve.json (the add-on's "Add the real sun"
parents its lamp to the world empty, so the direction stays right after Set real size / Set ground). `median_depth` is the median camera-to-point depth
in the solve's units (times the world's scale for Blender units).

### `real_scale.json` (add-on, optional)

Written next to `selected/` by "Set real size" (two points or camera height) and "Set ground from
points": `adjust`, the whole change as a 4x4 (row-major) put in front of `world`; `factor` (size,
cumulative) and `how`; `ground` (e.g. "4 points, 0.3% off flat"). Files from 0.4.0 have only
`factor` and `how` (then `adjust` = factor x identity). "Reset size and ground" renames the file to
`real_scale.json.old`. When present it wins over `units`.

### `telemetry.npz`

Per-frame telemetry recorded in the clip (niko.telemetry.read_telemetry), one row per video frame,
NaN where a value is missing, held between readings:
- DJI drones and gimbals ("DJI meta" stream, `source` dji_djmd): `lat`, `lon` (degrees), `alt` (GPS,
  m), `rel_alt` (above take-off, m), `drone_pitch/roll/yaw`, `gimbal_pitch/roll/yaw` (degrees),
  `gimbal_q` [T,4] (w, x, y, z: camera body to NED), `zoom`; GPS at 10 Hz, altitude about 5 Hz.
- GoPro ("GoPro MET" GPMF stream, `gopro_gpmf`): `lat`, `lon`, `alt` from GPS5 / GPS9 (only 3D fixes
  with a dilution of precision <= 5), `gravity_cam` [T,3] (GRAV, when the camera writes it); `info`
  adds `lens_mode` (W / L / S / N / H), `fov_deg` (ZFOV), `stabilised`, `firmware`.
- Sony (XML sidecar, or the same XML inside an XAVC S MP4; MXF RDD 18 metadata, `sony`): `focal_mm`,
  `focal_35mm`, `focus_m` per frame (MXF; shot.json keeps `focus_m` as [min, median, max]);
  `info` adds `lens_model`, `gamma` (e.g. s-log3-cine) and `sensor_mm` (the imager's effective size).
`info` (JSON) always has `source`, `model`, `camera`, `width`, `height`, `fps`, `sensor_mm`.

### `$NIKO_HOME/camera_profiles.json`

Lenses measured by `niko calibrate <solve>` (a calibration clip: the camera turns 15 deg or more),
keyed by camera and video mode (`camera.key`): `{"name", "focal_px", "width", "uncertainty_pct",
"how", "date"}`. A profile within 1.5 % is the known lens of every later solve from that camera.

### Scene mesh (`niko mesh <solve> --quality fast|good|high`)

`selected/mesh.ply` (coloured surface, holes up to ~10 voxels filled), `selected/mesh_sim.ply`
(about 150k triangles, no non-manifold edges, bigger holes filled: for collisions and simulation),
`selected/dense_points.ply` (the fused stereo points), `selected/mesh_textured.obj` + `.mtl` +
`_albedo.png` (the surface simplified to 150k / 300k / 500k triangles for fast / good / high, unwrapped
with UVAtlas and textured from 16 / 24 / 36 of the stereo frames in a 2048 / 4096 px image), all in the
solve's world like `points.ply` (the OBJ too: import it without an axis change).
Stereo points on moving things (the solve's masks) and far away are removed before the surface.

### Inputs

Videos ffmpeg decodes (H.264 / H.265 8 and 10 bit, ProRes, XAVC / MXF, HDR HLG and PQ, phone videos
held upright: the rotation flag is applied and the solve is portrait) and image folders (.jpg .png
.tif .tiff incl. 16 bit, .exr incl. float with the sRGB curve, .dpx, .bmp; `--fps`). Camera RAW (.braw,
.r3d, .ari / .arx, .crm) stops at ingest with the way out: export from DaVinci Resolve as ProRes 422
HQ or an EXR / TIFF sequence.

### `shot.json`

```jsonc
{
  "schema": "niko.shot/1",
  "source": "/path/to/clip.mp4",
  "width": 3840, "height": 2160, "fps": 23.976, "n_frames": 240, "frame_start": 1,
  "frame_format": "jpg", "jpg_quality": 95,
  "proxy": {"width": 1920, "height": 1080, "scale_x": 0.5, "scale_y": 0.5},  // proxy px = full px * scale
  "lens": {"focal_mm": null, "focal_35mm": null, "sensor_width_mm": null, "source": "ffprobe|exif|none"},
  "codec": "h264", "pix_fmt": "yuv420p", "rotation": 0,
  // fps is measured from the frame spacing and set to the nearest standard rate (the container's own
  // numbers can be wrong: 1080i MXF says 12.5 for 25); a variable-rate clip (phones in low light) keeps
  // every recorded frame, one frame each, at its nominal rate; interlaced video is deinterlaced (yadif)
  "vfr": false, "fps_average": 23.976, "interlaced": false,
  // the drone telemetry's summary (null without): source, proto, model, camera, sensor_mm, gps, altitude, gimbal, zoom
  "telemetry": {"source": "dji_djmd", "model": "DJI Air3s", "camera": "DJI FC9113", "sensor_mm": [13.107, 7.372],
                "gps": true, "altitude": true, "gimbal": true, "zoom": [1.0, 1.0], "...": 0},
  // which camera made the clip (niko.camera.identify: telemetry, else make / model tags), or null
  "camera": {"id": "DJI FC9113", "name": "DJI Air3s (DJI FC9113)",
             "key": "DJI FC9113|3840x2160|sensor 13.107x7.372|zoom 1.00", "source": "telemetry"}
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
