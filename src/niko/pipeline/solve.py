"""`niko solve`: ingest -> masks -> tracks -> candidates -> (refine) -> select -> export.

Refinement (bundle adjustment against CoTracker3 tracks) is checkpoint 3; until then the
candidates go to auto-select unrefined, and solve.json says so.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from ..backend import run_backend
from ..camio import CameraTrack
from ..tracks import track_gaps
from ..evaluate import evaluate_against_gt
from .candidates import STRIDED, coverage, keyframe_stride, run_candidate
from .export import export_solve
from .ingest import ingest
from .select import MIN_SIFT_OBS, frame_errors, select

DEFAULT_METHODS = ("colmap_global", "colmap_incremental", "megasam", "da3")


def _bench_shot(clip: Path) -> dict | None:
    """A shot made by `niko bench generate` (frames/ + spec.json + gt/cameras.json)."""
    if clip.is_dir() and (clip / "spec.json").exists() and (clip / "frames").is_dir():
        return json.loads((clip / "spec.json").read_text())
    return None


BA_MODES = {  # name suffix -> (intrinsics, distortion, complexity)
    "ba": ("shared_focal", "none", 1),
    "ba_k1k2": ("shared_focal", "k1k2", 2),
    "ba_zoom": ("per_frame_focal", "none", 3),
}
TRIPOD_MODES = {"tripod": ("shared_focal", 0), "tripod_zoom": ("per_frame_focal", 1)}
# a known lens (user or clip metadata): the focal is fixed, only the camera path and optionally k1/k2 move
KNOWN_LENS_MODES = {"ba_lens": ("fixed", "none", 1), "ba_lens_k1k2": ("fixed", "k1k2", 2)}
# the learned depth models (MegaSaM, DA3) estimate the focal from image content; when they disagree
# with the solve by more than this (|log ratio|), the tracks did not constrain the focal:
# synthetic_v1 dolly_forward 2.41 (focal 86 % off), every other shot <= 0.66 (zoom shots included)
LENS_WARNING_LOG_RATIO = 1.0
# a camera that turns less than this only translates, and pure translation does not measure the lens
# (a longer lens and a deeper scene project the same): synthetic shots turning >= 1.09 deg (handheld
# shake is enough) have focal errors <= 1.5 %; dolly_forward (0.03 deg) 86 %; real DJI 0079 (0.18 deg,
# a gimbal holds the camera still) fits the footage, the gimbal and the GPS equally at 2662 and 2979 px
MIN_TURN_DEG = 1.0
# candidates whose held-out reprojection is within LENS_SPREAD_REPROJ of the best fit the footage
# equally well; when their lenses differ by more than LENS_SPREAD_WARN the footage did not measure
# the lens (scripts/dev/lens_spread.py, 2026-09-29). Synthetic shots whose pick is not a tripod
# solve: 18 of 21 spread <= 5.6 % (their picks' focal error <= 1.54 %); the other three are
# dolly_forward 1026 % (pick 86 % off), tele_orbit_short 58 % (pick 28 % off) and pan_low_parallax
# 36 % (pick right: it won on path jitter alone). Real clips: 01 2.6 %, 02 21.8 %, 03 14.4 % (its
# 4 s cut 23.6 %), 04 0.5 %, DJI 0148 5.5 %.
LENS_SPREAD_REPROJ = 0.10
LENS_SPREAD_WARN = 0.10
RAW_COMPLEXITY = 2
TRIPOD_GATE_PX = 1.0


def write_ply(path: Path, xyz: np.ndarray) -> None:
    """Binary PLY with x, y, z, red, green, blue (grey), same layout as the backends write."""
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz.T
    data["red"] = data["green"] = data["blue"] = 200
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def homography_residual_px(tracks: dict, gap: int = 15, min_points: int = 30) -> tuple[float, int]:
    """(see below) On keyframe-strided SIFT tracks the pairs are keyframes about `gap` frames apart."""
    frames = np.nonzero(tracks["vis"].any(1))[0]
    if len(frames) > 1 and np.median(np.diff(frames)) > 1:
        step = int(np.median(np.diff(frames)))
        sub = {"xy": tracks["xy"][frames], "vis": tracks["vis"][frames]}
        return _homography_residual_px(sub, max(1, round(gap / step)), min_points)
    return _homography_residual_px(tracks, gap, min_points)


def _homography_residual_px(tracks: dict, gap: int = 15, min_points: int = 30) -> tuple[float, int]:
    """Parallax from the tracks alone: median over frame pairs `gap` apart of the median transfer
    residual of a RANSAC homography. A rotating (and zooming) camera maps all points by one
    homography, so the residual stays at the matching noise; parallax raises it.
    Returns (residual px, number of pairs); (inf, 0) without enough shared points."""
    import cv2

    xy, vis = tracks["xy"], tracks["vis"]
    T = len(xy)
    gap = max(1, min(gap, T // 4))
    per_pair = []
    for t in range(0, T - gap, max(1, gap // 2)):
        m = vis[t] & vis[t + gap]
        if m.sum() < min_points:
            continue
        p0, p1 = xy[t, m].astype(np.float64), xy[t + gap, m].astype(np.float64)
        H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)
        if H is None:
            continue
        q = np.c_[p0, np.ones(len(p0))] @ H.T
        r = np.linalg.norm(q[:, :2] / q[:, 2:3] - p1, axis=1)
        per_pair.append(float(np.median(r)))
    return (float(np.median(per_pair)), len(per_pair)) if per_pair else (float("inf"), 0)


SIFT_SPLIT = "longest-8000"  # recorded on refined candidates; reuse needs the same train/held-out split


def load_sift(out_dir: Path, shot: dict, log) -> tuple[dict | None, dict | None]:
    """Multi-view SIFT tracks from the COLMAP candidates' database, split into (train, held out)."""
    from ..refine import RefineOptions
    from ..tracks import split_sift

    for m in ("colmap_global", "colmap_incremental"):
        db = out_dir / "candidates" / m / "colmap" / "database.db"
        if db.exists():
            r = run_backend("colmap", "sift_tracks", out_dir, out_dir / "sift",
                            {"db": str(db), "width": shot["width"], "height": shot["height"],
                             "proxy_width": shot["proxy"]["width"], "proxy_height": shot["proxy"]["height"],
                             "n_frames": shot["n_frames"]})
            train, held = split_sift(dict(np.load(out_dir / "sift" / "sift_tracks.npz")), RefineOptions().max_sift)
            log(f"[sift] tracks from {m}: {r['stats']['tracks']} tracks, {r['stats']['observations']} obs; "
                f"{train['vis'].shape[1]} for refinement, {held['vis'].shape[1]} held out "
                f"({int(held['vis'].sum())} obs)")
            return train, held
    return None, None


def _reuse_refined(folder: Path, key: str, options: dict, focal_px=None) -> CameraTrack | None:
    """A refined candidate from an earlier run, if it was made with exactly these options."""
    path = folder / "cameras.json"
    if not path.exists():
        return None
    trk = CameraTrack.load(path)
    ok = (trk.extra.get(key, {}).get("options") == options and trk.extra.get("sift_split") == SIFT_SPLIT
          and trk.extra.get("known_focal_px") == focal_px)
    return trk if ok else None


def refine_stage(cands: dict, tracks: dict, out_dir: Path, log, sift: dict | None = None,
                 sift_holdout: dict | None = None, top_k: int = 2, reuse: bool = False,
                 focal_px: float | None = None) -> dict:
    """Refine the best raw candidates (bundle adjustment in three intrinsics models, tripod solver).
    sift: the SIFT tracks refinement may use; sift_holdout: the ones kept for auto-select.
    reuse=True takes a refined candidate from disk when its recorded options match (selection work).
    focal_px: a known focal (pixels): bundle adjustment keeps it fixed on every candidate."""
    from dataclasses import asdict

    from ..refine import RefineOptions, copy_track, refine
    from ..tripod import TripodOptions, solve_tripod
    from .select import score_candidate, usable_sift_holdout

    held = usable_sift_holdout(sift_holdout)
    pre = {n: score_candidate(c, tracks, held) for n, c in cands.items()}
    # coverage counts the frames a method was given: keyframe-strided candidates are complete at 1/stride
    usable = [n for n in cands if coverage(cands[n]) >= 0.2 and np.isfinite(pre[n]["score_px"])]
    usable.sort(key=lambda n: (coverage(cands[n]) < 0.95, pre[n]["score_px"]))
    added = {}
    # every good start converges to the same optimum (orbit_yard: all four candidates -> 0.013 deg),
    # so the three camera models run on the best candidate and plain BA on the runner-up
    modes = KNOWN_LENS_MODES if focal_px else BA_MODES
    for rank, n in enumerate(usable[:top_k]):
        start = cands[n]
        if focal_px:
            start = copy_track(start)
            start.K[:, 0, 0] = start.K[:, 1, 1] = focal_px
        for suffix, (intr, dist, cx) in modes.items():
            if rank > 0 and suffix not in ("ba", "ba_lens"):
                continue
            name = f"{n}+{suffix}"
            ropts = RefineOptions(intrinsics=intr, distortion=dist)
            old = _reuse_refined(out_dir / "candidates" / name, "refine", asdict(ropts), focal_px) if reuse else None
            if old is not None:
                added[name] = old
                log(f"[refine] {name}: reused")
                continue
            try:  # one camera model failing must not cost the others
                trk, rep = refine(start, tracks, out_dir / "candidates" / name, ropts, sift=sift)
                f = trk.K[trk.valid, 0, 0]
                if not np.all(np.isfinite(f) & (f > 0)):  # real 4K clip 04, per-frame focal: frame 39 went <= 0
                    raise RuntimeError(f"focal diverged on {int((~(np.isfinite(f) & (f > 0))).sum())} frames")
                trk.extra["complexity"] = cx
                trk.extra["sift_split"] = SIFT_SPLIT
                if focal_px:
                    trk.extra["known_focal_px"] = focal_px
                write_ply(out_dir / "candidates" / name / "points.ply",
                          np.load(out_dir / "candidates" / name / "ba_output.npz")["X"])
                trk.points = "points.ply"
                trk.save(out_dir / "candidates" / name / "cameras.json")
            except Exception as e:
                log(f"[refine] {name} failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            added[name] = trk
            log(f"[refine] {name}: {int(trk.valid.sum())}/{trk.n_frames} frames, "
                f"{rep['rounds'][-1]['median_px']:.3f} px median, {rep['seconds']}s")
    # rotation-only models only when a homography explains the tracks. synthetic_v1 (SIFT, 15-frame
    # pairs): tripod 0.20, zoom_in 0.23, whip 0.39, 5 cm nodal pan 0.30 px; every shot with a
    # moving camera >= 0.72 px except distant drone_orbit 0.29 (tripod runs there and loses on score).
    h_px, h_pairs = homography_residual_px(sift if sift is not None else tracks)
    run_tripod = bool(usable) and h_px < TRIPOD_GATE_PX and not focal_px  # the tripod solver refines the focal
    log(f"[refine] homography residual {h_px:.3f} px over {h_pairs} pairs -> "
        f"rotation-only models {'on' if run_tripod else 'off'}")
    if run_tripod:
        base = usable[0]
        tr = sift if sift is not None else tracks
        start = cands[base]
        if coverage(start) > start.valid.mean() + 1e-9:  # keyframes only: every frame needs observations
            from ..refine import interpolate_gaps
            from ..tracks import split_tracks
            start = copy_track(start)
            interpolate_gaps(start)
            seg = split_tracks(tracks, 16)
            keep = ~seg["holdout"]
            parts = [(seg["xy"][:, keep], seg["vis"][:, keep])] + ([(sift["xy"], sift["vis"])] if sift is not None else [])
            tr = {"xy": np.concatenate([p[0] for p in parts], 1), "vis": np.concatenate([p[1] for p in parts], 1)}
        for suffix, (intr, cx) in TRIPOD_MODES.items():
            name = f"{base}+{suffix}"
            topts = TripodOptions(intrinsics=intr)
            old = _reuse_refined(out_dir / "candidates" / name, "tripod", asdict(topts)) if reuse else None
            if old is not None:
                added[name] = old
                log(f"[refine] {name}: reused")
                continue
            try:
                trk, rep = solve_tripod(start, tr, topts)
                f = trk.K[trk.valid, 0, 0]
                if not np.all(np.isfinite(f) & (f > 0)):
                    raise RuntimeError(f"focal diverged on {int((~(np.isfinite(f) & (f > 0))).sum())} frames")
                trk.extra["complexity"] = cx
                trk.extra["sift_split"] = SIFT_SPLIT
                (out_dir / "candidates" / name).mkdir(parents=True, exist_ok=True)
                trk.save(out_dir / "candidates" / name / "cameras.json")
            except Exception as e:
                log(f"[refine] {name} failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            added[name] = trk
            log(f"[refine] {name}: {rep['rounds'][-1]['median_px']:.3f} px median, {rep['seconds']}s")
    return added


def hardware_settings() -> dict:
    """Per-machine settings from $NIKO_HOME/hardware.json (written by installer/hardware_check.ps1:
    GPU memory decides the SAM 3 chunk and the CoTracker3 size). None found: this machine's defaults."""
    from ..paths import niko_home

    f = niko_home() / "hardware.json"
    try:
        return json.loads(f.read_text(encoding="utf-8-sig")).get("settings") or {}
    except (OSError, ValueError):
        return {}


def lens_spread(cands: dict, scores: dict, best: str) -> dict | None:
    """Focal range of the candidates that fit the footage as well as the best one (held-out
    reprojection within LENS_SPREAD_REPROJ, >= 95 % of their frames or keyframes). None for a tripod
    pick: its lens comes from the rotation alone, and 6-dof solves of a pure rotation are degenerate."""
    if (scores.get(best) or {}).get("rotation_only"):
        return None
    ok = {n: s for n, s in scores.items() if n in cands and np.isfinite(s.get("score_px", np.inf))
          and not s.get("rotation_only") and coverage(cands[n]) >= 0.95}
    ok.setdefault(best, scores[best])
    r0 = min(s["reproj_median_px"] for s in ok.values())
    near = sorted(n for n, s in ok.items() if s["reproj_median_px"] <= r0 * (1 + LENS_SPREAD_REPROJ))
    f = {n: float(np.median(cands[n].K[cands[n].valid, 0, 0])) for n in near}
    lo, hi = min(f.values()), max(f.values())
    return {"equally_good": {n: round(v, 1) for n, v in f.items()}, "focal_px": [round(lo, 1), round(hi, 1)],
            "spread_pct": round(100 * (hi / lo - 1), 2)}


def lens_check(cands: dict, best: str, scores: dict | None = None) -> dict | None:
    """Flag a solve whose lens the footage did not measure: the learned depth models disagree with
    its focal by more than a factor e^LENS_WARNING_LOG_RATIO (pure forward motion), or candidates
    that fit the footage equally well disagree on the lens by more than LENS_SPREAD_WARN (long lens,
    little parallax), or the camera turns less than MIN_TURN_DEG (it only translates)."""
    trk = cands[best]
    f = float(np.median(trk.K[trk.valid, 0, 0]))
    out, reasons = {"selected_focal_px": round(f, 1)}, []
    learned = {m: float(np.median(cands[m].K[cands[m].valid, 0, 0])) for m in ("megasam", "da3")
               if m in cands and cands[m].valid.sum() >= 3}
    if learned:
        ratio = max(abs(np.log(v / f)) for v in learned.values())
        out.update(log_ratio=round(float(ratio), 3), learned_focal_px={k: round(v, 1) for k, v in learned.items()})
        if ratio > LENS_WARNING_LOG_RATIO:
            reasons.append("depth_models")
    sp = lens_spread(cands, scores, best) if scores else None
    if sp:
        out["spread"] = sp
        if sp["spread_pct"] > 100 * LENS_SPREAD_WARN:
            reasons.append("equally_good_fits")
    v = np.nonzero(trk.valid)[0]
    if len(v) >= 2:
        R0 = trk.R_c2w[v[0]]
        turn = max(float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ trk.R_c2w[i]) - 1) / 2, -1, 1)))) for i in v)
        out["max_turn_deg"] = round(turn, 3)
        tripod = trk.extra.get("rotation_only") or bool((scores or {}).get(best, {}).get("rotation_only"))
        if turn < MIN_TURN_DEG and not tripod:
            reasons.append("hardly_turns")
    if not learned and not sp and "hardly_turns" not in reasons:
        return None
    return {"uncertain": bool(reasons), "reasons": reasons, **out}


def solve(clip: str | Path, out_dir: str | Path, methods=DEFAULT_METHODS, prompts=None,
          fps: float | None = None, frame_start: int = 1, track_size=None, log=print, refine: bool = True,
          reuse: bool = False, focal_mm: float | None = None, sensor_mm: float = 36.0,
          stride: int | None = None) -> dict:
    """focal_mm: a known lens, with sensor_mm its sensor width (36 = full frame / 35 mm equivalent).
    stride: keyframe step for COLMAP / MegaSaM (None: candidates.keyframe_stride)."""
    clip, out_dir = Path(clip).resolve(), Path(out_dir).resolve()
    from .cache import check_reuse, signature

    spec = _bench_shot(clip)
    src = clip / "frames" if spec else clip
    hw = hardware_settings()
    fingerprint = signature(src, {"methods": list(methods), "prompts": prompts, "fps": fps,
                                  "frame_start": frame_start, "track_size": track_size, "refine": refine,
                                  "focal_mm": focal_mm, "sensor_mm": sensor_mm, "stride": stride,
                                  "hardware": hw, "spec": spec})
    if reuse:
        check_reuse(out_dir, fingerprint)
    elif any((out_dir / p).exists() for p in ("shot.json", "solve.json", "candidates", "selected", "frames")):
        raise ValueError("This output folder already contains a solve. Choose a new folder, or use --reuse "
                         "with unchanged inputs. Existing results have been kept.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reuse.json").unlink(missing_ok=True)
    report: dict = {"schema": "niko.solve/1", "clip": str(clip), "stages": {}, "refined": False}
    t_all = time.time()

    def stage(name, fn):
        t0 = time.time()
        try:
            value = fn()
            report["stages"][name] = {"ok": True, "seconds": round(time.time() - t0, 2)}
            return value
        except Exception as e:
            report["stages"][name] = {"ok": False, "seconds": round(time.time() - t0, 2),
                                      "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
            log(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}")
            return None

    gt = CameraTrack.load(clip / "gt/cameras.json") if spec else None

    def reused(path: Path):
        """With reuse=True, an earlier stage's result.json that says ok is taken as is."""
        if reuse and path.exists():
            res = json.loads(path.read_text())
            if res.get("ok"):
                return res
        return None

    if reuse and (out_dir / "shot.json").exists():
        shot = json.loads((out_dir / "shot.json").read_text())
        report["stages"]["ingest"] = {"ok": True, "seconds": 0.0, "reused": True}
    else:
        shot = stage("ingest", lambda: ingest(src, out_dir, fps=(spec or {}).get("fps", fps),
                                              frame_start=(spec or {}).get("frame_start", frame_start)))
    if shot is None:
        return _finish(report, out_dir, t_all)
    log(f"[ingest] {shot['n_frames']} frames {shot['width']}x{shot['height']} @ {shot['fps']:g} fps")

    opts = {"prompts": prompts} if prompts else {}
    if hw.get("sam3_chunk_frames"):
        opts["chunk_frames"] = int(hw["sam3_chunk_frames"])
    if prompts is not None and not prompts:  # the user chose to ignore nothing: no masks step at all
        m = None
        report["stages"]["masks"] = {"ok": True, "seconds": 0.0, "skipped": "nothing to ignore"}
        log("[masks] off: nothing is ignored (every pixel is tracked, moving things included)")
    else:
        m = reused(out_dir / "jobs/masks/result.json") or stage(
            "masks", lambda: run_backend("sam3", "masks", out_dir, out_dir / "jobs/masks", opts))
    if m:
        log(f"[masks] {m['runtime_s']:.1f}s, excluded {100 * m['stats']['mean_excluded_fraction']:.1f}% "
            f"of pixels, objects {m['stats']['objects_per_prompt']}")

    # CoTracker3 at up to 960x540 pixels (0.67 px median vs 1.0 px at its default 512x384 on the
    # smoke shot); larger proxies are tracked at that area, aspect kept, sides multiples of 8.
    pw, ph = shot["proxy"]["width"], shot["proxy"]["height"]
    k = float(np.sqrt(float(hw.get("cotracker_max_area", 960 * 540)) / (pw * ph)))
    size = track_size or ([ph, pw] if k >= 1 else [int(round(ph * k / 8) * 8), int(round(pw * k / 8) * 8)])
    topt = {"query_every": 10, **({"model_size": size} if size else {})}
    t = reused(out_dir / "tracks/result.json")
    if t is not None and t.get("stats", {}).get("query_mode") != "model":
        t = None  # tracks from before the query-frame fix (exact queries) are recomputed
    t = t or stage("tracks", lambda: run_backend("cotracker", "tracks", out_dir, out_dir / "tracks", topt))
    if t:
        log(f"[tracks] {t['runtime_s']:.1f}s, {t['stats']['n_tracks']} tracks, "
            f"{t['stats']['mean_visible_per_frame']:.0f} visible/frame, peak VRAM {t['peak_vram_mb']:.0f} MB")

    cands = {}
    report["candidates"] = {}
    kstride = stride or keyframe_stride(shot["n_frames"], shot["fps"])
    report["keyframe_stride"] = kstride
    if kstride > 1:
        log(f"[candidates] COLMAP / MegaSaM on every {kstride}th frame "
            f"({-(-shot['n_frames'] // kstride)} of {shot['n_frames']}); refinement fills the rest")
    for method in methods:
        prev = reused(out_dir / "candidates" / method / "result.json")
        want = kstride if method in STRIDED else 1
        if prev is not None and int(prev.get("stats", {}).get("stride", 1) or 1) != want:
            prev = None  # made with another keyframe stride
        if prev is not None and (out_dir / "candidates" / method / "cameras.json").exists():
            r = (CameraTrack.load(out_dir / "candidates" / method / "cameras.json"), prev)
            report["stages"][f"candidate:{method}"] = {"ok": True, "seconds": 0.0, "reused": True}
        else:
            copts = {"stride": kstride} if method in STRIDED and kstride > 1 else None
            r = stage(f"candidate:{method}", lambda: run_candidate(method, out_dir, out_dir / "candidates", copts))
        if r:
            trk, res = r
            cands[method] = trk
            report["candidates"][method] = {
                "runtime_s": res["runtime_s"], "peak_vram_mb": res.get("peak_vram_mb"),
                "peak_vram_source": res.get("peak_vram_source", "torch"), "registered": int(trk.valid.sum()),
                "stats": res.get("stats")}
            log(f"[{method}] {res['runtime_s']:.1f}s, {int(trk.valid.sum())}/{trk.n_frames} frames")

    if not cands or t is None:
        log("[select] skipped: no candidates or no tracks")
        return _finish(report, out_dir, t_all)
    tracks = dict(np.load(out_dir / "tracks/tracks.npz"))
    sift, sift_holdout = stage("sift", lambda: load_sift(out_dir, shot, log)) or (None, None)
    if sift is not None and int(sift["vis"].sum()) + int(sift_holdout["vis"].sum()) >= MIN_SIFT_OBS:
        fs = int(shot.get("frame_start", 1))
        gaps = track_gaps(np.load(out_dir / "sift" / "sift_tracks.npz")["vis"])
        report["track_gaps"] = [{"from_frame": fs + g["last_before"], "to_frame": fs + g["first_after"],
                                 "tracks": g["tracks"]} for g in gaps]
        for g in report["track_gaps"]:
            log(f"[sift] tracking breaks between frames {g['from_frame']} and {g['to_frame']} ({g['tracks']} tracks "
                "cross; fast motion or blur): each side can be right and the turn between them off")
    if refine:
        focal_px = focal_mm / sensor_mm * shot["width"] if focal_mm else None
        if focal_px:
            report["known_lens"] = {"focal_mm": focal_mm, "sensor_mm": sensor_mm, "focal_px": focal_px,
                                    "source": "given"}
            log(f"[refine] known lens {focal_mm:g} mm on a {sensor_mm:g} mm sensor = {focal_px:.1f} px")
        else:
            from ..camera import profile_for
            prof = profile_for(shot.get("camera"))
            if prof and prof.get("width") == shot["width"]:
                focal_px = float(prof["focal_px"])
                report["known_lens"] = {"focal_px": focal_px, "source": "camera profile",
                                        "camera": shot["camera"]["name"], "uncertainty_pct": prof["uncertainty_pct"],
                                        "calibrated": prof.get("how")}
                log(f"[refine] known lens from the {shot['camera']['name']} profile: {focal_px:.1f} px "
                    f"(+-{prof['uncertainty_pct']} %, {prof.get('how')})")
        added = stage("refine", lambda: refine_stage(cands, tracks, out_dir, log, sift, sift_holdout, reuse=reuse,
                                                     focal_px=focal_px))
        if added:
            if focal_px:  # with a known lens only the candidates that use it are answers
                raw = {n: c for n, c in cands.items() if n not in added}
                report["raw_candidates_kept_for_evaluation"] = sorted(raw)
                for name in raw:
                    cands.pop(name)
            cands.update(added)
            report["refined"] = True
        elif focal_px:
            report["stages"]["refine"] = {"ok": False, "error": "No candidate could use the supplied lens."}
            return _finish(report, out_dir, t_all)
    best, sel = stage("select", lambda: select(cands, tracks, sift_holdout)) or (None, None)
    report["select"] = sel
    if sel:
        for name in sel["ranking"]:
            s = sel["scores"][name]
            log(f"[select] {name:<20} score {s['score_px']:.3f} px (reproj {s.get('reproj_median_px', float('nan')):.3f}, "
                f"jitter {s.get('jitter_px', float('nan')):.3f}, frames {100 * s['success_rate']:.0f}%)")
    report["selected"] = best
    report["camera"] = shot.get("camera")
    if best and not report.get("known_lens"):
        lc = lens_check(cands, best, sel["scores"])
        if lc:
            report["lens_check"] = lc
            if lc["uncertain"]:
                why = []
                if "depth_models" in lc["reasons"]:
                    why.append(f"depth models say {lc['learned_focal_px']}")
                if "equally_good_fits" in lc["reasons"]:
                    lo, hi = lc["spread"]["focal_px"]
                    why.append(f"solves that fit the footage as well range {lo:.0f}-{hi:.0f} px")
                if "hardly_turns" in lc["reasons"]:
                    why.append(f"the camera turns only {lc['max_turn_deg']:.2f} deg, and a camera that only moves "
                               "does not measure its lens (calibrate the camera once: niko calibrate)")
                log(f"[solve] lens uncertain: solved focal {lc['selected_focal_px']} px, {'; '.join(why)}; this "
                    "camera motion does not measure the lens: give the focal length (--focal-mm) if you know it")
    if best:
        s = sel["scores"][best]
        report["solve_error"] = {"average_px": s.get("reproj_mean_px"), "median_px": s["reproj_median_px"],
                                 "inlier_fraction": s["inlier_fraction"], "source": sel.get("reproj_source"),
                                 "frames": int(cands[best].valid.sum()), "n_frames": cands[best].n_frames}
        log(f"[solve] {best}: average tracking error {s.get('reproj_mean_px') or float('nan'):.3f} px "
            f"(median {s['reproj_median_px']:.3f} px, {100 * s['inlier_fraction']:.1f} % of held-out "
            f"{sel.get('reproj_source')} observations within 3 px), "
            f"{int(cands[best].valid.sum())}/{cands[best].n_frames} frames")
        trk = cands[best]
        metric = stage("metric_scale", lambda: _metric_scale(trk, out_dir, tracks))
        if metric:
            report["metric_scale"] = metric
            agree = metric.get("agree_pct")
            apart = f"the two depth models {agree:.0f} % apart" if agree is not None else "one depth model only"
            if metric.get("source") in ("gps", "altitude"):
                t = metric["telemetry"][metric["source"]]
                what = (f"GPS: {t['fixes']} fixes over {t['extent_m']:.0f} m" if metric["source"] == "gps"
                        else f"altitude: {t['change_m']:.1f} m of climb or descent")
                log(f"[scale] {metric['metres_per_unit']:.4g} m per solve unit from the drone's {what}, "
                    f"+-{metric['uncertainty_pct']:.1f} %: the Blender scene is in metres")
            elif metric.get("reliable"):
                log(f"[scale] about {metric['metres_per_unit']:.4g} m per solve unit ({apart}): the Blender scene is "
                    "in metres, approximately; set the exact size in Blender from a known distance")
            else:
                log(f"[scale] real size unknown ({apart}): set it in Blender from a known distance or the camera height")
        use = metric if metric and metric.get("reliable") else None
        from ..telemetry import for_solve, true_up
        up = true_up(trk, for_solve(out_dir))
        if up is not None:
            report["true_up"] = {"source": "gimbal", "up": [float(x) for x in up]}
            log("[export] level from the drone's gimbal (true gravity)")
        stage("export", lambda: export_solve(trk, out_dir / "candidates" / best / "points.ply", out_dir / "selected",
                                             out_dir / "frames", f"000000.{shot['frame_format']}", metric=use,
                                             up=up))
        from ..export_ae import export_after_effects
        stage("export_ae", lambda: export_after_effects(out_dir, log=log))
        fe = stage("frame_errors", lambda: frame_errors(trk, tracks, sift_holdout))
        if fe:
            (out_dir / "selected" / "errors.json").write_text(json.dumps(fe), encoding="utf-8")
            m = [(v, i) for i, v in enumerate(fe["mean_px"]) if v is not None]
            if m:
                v, i = max(m)
                report["solve_error"]["worst_frame"] = {"frame": trk.frame_start + i, "mean_px": v}
        log(f"[export] {best} -> {out_dir / 'selected'}")

    if gt is not None:
        report["gt_eval"] = {}
        gt_depth = gt.extra.get("median_depth")
        for name, trk in cands.items():
            s = (sel or {}).get("scores", {}).get(name, {})
            # static GT (tripod): express the estimate's own spread in GT units via median scene depth
            est_size = (gt.extra["scene_size"] * s["median_depth"] / gt_depth
                        if s.get("median_depth") and gt_depth else None)
            try:
                r = evaluate_against_gt(trk, gt, est_scene_size=est_size)
            except ValueError as e:
                r = {"failed": True, "error": str(e), "n_valid": int(trk.valid.sum()), "n_frames": trk.n_frames,
                     "success_rate": float(trk.valid.mean())}
            r["heldout_reproj_px"] = s.get("reproj_median_px")
            r["jitter_px"] = s.get("jitter_px")
            report["gt_eval"][name] = r
        for name, r in report["gt_eval"].items():
            flag = "  <- selected" if name == best else ""
            if r.get("failed"):
                log(f"[gt] {name:<20} failed ({r['n_valid']}/{r['n_frames']} frames){flag}")
            else:
                log(f"[gt] {name:<20} ATE {r['ate_pct']:.3f}%  rot {r['rot_err_deg_max']:.3f} deg max  "
                    f"focal {r['focal_err_pct_max']:.2f}% max  {'TARGETS MET' if r['meets_targets'] else 'missed'}{flag}")
    result = _finish(report, out_dir, t_all)
    if result["ok"]:
        (out_dir / "reuse.json").write_text(json.dumps(fingerprint, indent=1), encoding="utf-8")
    return result


def _metric_scale(trk: CameraTrack, out_dir: Path, tracks: dict) -> dict | None:
    """Metres per solve unit: from the drone's own telemetry (GPS, else altitude) when it has some
    and it is reliable, else approximately from the two depth models (niko.scale)."""
    from ..scale import estimate_metric_scale, telemetry_scale
    from ..telemetry import for_solve

    sift = out_dir / "sift" / "sift_tracks.npz"
    src = dict(np.load(sift)) if sift.exists() else tracks
    depth = estimate_metric_scale(trk, src["xy"], src["vis"], out_dir)
    if depth:
        depth["source"] = "depth_models"
    tel = for_solve(out_dir)
    ts = telemetry_scale(trk, tel) if tel else None
    if ts and ts["reliable"]:
        return {"metres_per_unit": ts["metres_per_unit"], "reliable": True, "source": ts["source"],
                "uncertainty_pct": ts["uncertainty_pct"], "telemetry": ts, "depth_models": depth}
    if ts:
        return {**(depth or {"reliable": False}), "telemetry": ts}
    return depth


def _finish(report: dict, out_dir: Path, t0: float) -> dict:
    report["total_seconds"] = round(time.time() - t0, 1)
    # Candidate methods are alternatives. A failed alternative does not invalidate the selected export.
    required = ("ingest", "select", "export", "frame_errors")
    failed = {k: v.get("error", "Stage failed") for k, v in report["stages"].items() if not v["ok"]}
    report["warnings"] = [{"stage": k, "message": v} for k, v in failed.items() if k not in required]
    report["ok"] = (report.get("selected") is not None
                    and all(report["stages"].get(k, {}).get("ok", False) for k in required))
    report["status"] = ("completed_with_warnings" if report["warnings"] else "completed") if report["ok"] else "failed"
    (out_dir / "solve.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report
