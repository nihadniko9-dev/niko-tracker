"""`python -m niko_cotracker job.json` - CoTracker3 offline point tracks -> tracks.npz.

Long clips are tracked in overlapping temporal windows and the tracks are chained: a track that
is still visible at the first frame of the next window is re-queried there at its predicted
position and keeps its id. This bounds VRAM (one window of frames x one chunk of queries on the
GPU; the whole clip at once needed 24.7 GB at 384x512 for 150 frames) and keeps every GPU call
short enough for the Windows display-driver watchdog (a whole 150-frame clip at 544x960 failed
with "CUDA driver error: device not ready").

Job options (all optional):
  model_size   [h, w]  resolution the model runs at (default: proxy size capped at 960x540 area)
  window       frames per window (default 48), overlap (default 8)
  grid, corners        new query points per query frame (grid per side, Shi-Tomasi corners)
  chunk        queries per model call (default 400)
  min_dist     new queries keep this many proxy px away from continuing tracks (default 8)
  holdout      fraction of tracks reserved for held-out scoring (default 0.1), seed

Coordinates: CoTracker works in pixel-centre-at-integer coordinates with align_corners=True
resizing; tracks.npz uses corner-origin full-resolution pixels (docs/SCHEMA.md).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from niko_backend_common import ResultWriter, read_job

CKPT = Path(os.environ.get("NIKO_HOME", Path.home() / "niko")) / "checkpoints/cotracker3_offline/scaled_offline.pth"


def load_frames(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("*.jpg")) or sorted(folder.glob("*.png"))
    return np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files])


def load_masks(shot_dir: Path, n: int, size_wh: tuple[int, int]) -> np.ndarray | None:
    """Dynamic masks resized to (w, h); True = excluded. None if the shot has no masks."""
    d = shot_dir / "masks"
    files = sorted(d.glob("*.png")) if d.is_dir() else []
    if not files:
        return None
    if len(files) != n:
        raise ValueError(f"{len(files)} masks for {n} frames")
    out = np.zeros((n, size_wh[1], size_wh[0]), bool)
    for i, f in enumerate(files):
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        out[i] = cv2.resize(m, size_wh, interpolation=cv2.INTER_NEAREST) > 127
    return out


def sample_queries(frame_rgb, excluded, grid, corners, rng) -> np.ndarray:
    """Grid + corner points on one frame, corner-origin coordinates of that frame."""
    h, w = frame_rgb.shape[:2]
    margin = 4
    xs = np.linspace(margin, w - margin, grid)
    ys = np.linspace(margin, h - margin, grid)
    pts = np.array([(x, y) for y in ys for x in xs], np.float64)
    pts += rng.uniform(-0.5, 0.5, pts.shape) * np.array([w / grid, h / grid]) * 0.5
    if corners > 0:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        allowed = np.full(gray.shape, 255, np.uint8)
        if excluded is not None:
            allowed[excluded] = 0
        c = cv2.goodFeaturesToTrack(gray, maxCorners=corners, qualityLevel=0.005,
                                    minDistance=max(4, min(w, h) // 80), mask=allowed, blockSize=5)
        if c is not None:
            pts = np.vstack([pts, c.reshape(-1, 2) + 0.5])  # OpenCV index -> corner origin
    pts[:, 0] = np.clip(pts[:, 0], 0.5, w - 0.5)
    pts[:, 1] = np.clip(pts[:, 1], 0.5, h - 0.5)
    if excluded is not None:
        keep = ~excluded[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        pts = pts[keep]
    return pts


def to_model(xy, src_wh, model_wh):
    """corner-origin pixels of a src_wh image -> CoTracker coordinates at model_wh."""
    sx = (model_wh[0] - 1) / (src_wh[0] - 1)
    sy = (model_wh[1] - 1) / (src_wh[1] - 1)
    return np.stack([(xy[..., 0] - 0.5) * sx, (xy[..., 1] - 0.5) * sy], -1)


def from_model(xy, src_wh, model_wh):
    sx = (src_wh[0] - 1) / (model_wh[0] - 1)
    sy = (src_wh[1] - 1) / (model_wh[1] - 1)
    return np.stack([xy[..., 0] * sx + 0.5, xy[..., 1] * sy + 0.5], -1)


def repo_commit() -> str:
    import cotracker

    src = Path(cotracker.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def default_model_size(pw, ph):
    k = np.sqrt(960 * 540 / (pw * ph))
    if k >= 1:
        # CoTracker3 needs both sides divisible by its stride (4): GoPro's 854x480 samples failed with
        # an assertion. Sizes that already divide (960x540, 640x360) stay exactly as they were.
        return [int(round(ph / 4) * 4), int(round(pw / 4) * 4)]
    return [int(round(ph * k / 8) * 8), int(round(pw * k / 8) * 8)]


def windows(T, size, overlap):
    if T <= size:
        return [(0, T)]
    stride = size - overlap
    starts = list(range(0, T - size, stride)) + [T - size]
    return [(s, s + size) for s in sorted(set(starts))]


def main() -> None:
    job = read_job(sys.argv[1])
    opt = job.get("options", {})
    shot_dir, out_dir = Path(job["shot_dir"]), Path(job["out_dir"])
    with ResultWriter(job, versions={"cotracker_commit": repo_commit()}) as res:
        from cotracker.predictor import CoTrackerPredictor

        shot = json.loads((shot_dir / "shot.json").read_text())
        W, H = shot["width"], shot["height"]
        frames = load_frames(shot_dir / "proxy")
        T, ph, pw = frames.shape[:3]
        model_hw = tuple(opt.get("model_size") or default_model_size(pw, ph))
        model_wh = (model_hw[1], model_hw[0])
        model = CoTrackerPredictor(checkpoint=str(CKPT), offline=True, window_len=60).model.cuda().eval()
        rng = np.random.default_rng(opt.get("seed", 0))
        masks = load_masks(shot_dir, T, (pw, ph))
        grid, corners = opt.get("grid", 20), opt.get("corners", 300)
        chunk, min_dist = opt.get("chunk", 400), opt.get("min_dist", 8)

        xy_cols, vis_cols, conf_cols, qframe = [], [], [], []  # per track: [T,2], [T], [T], int

        def new_track(t0):
            xy_cols.append(np.full((T, 2), np.nan, np.float32))
            vis_cols.append(np.zeros(T, np.float16))
            conf_cols.append(np.zeros(T, np.float16))
            qframe.append(t0)
            return len(qframe) - 1

        wins = windows(T, opt.get("window", 48), opt.get("overlap", 8))
        for a, b in wins:
            # continuing tracks: visible and confident at the window's first frame
            q_ids, q_pts = [], []
            for n in range(len(qframe)):
                if vis_cols[n][a] > 0.5 and conf_cols[n][a] > 0.5 and np.isfinite(xy_cols[n][a, 0]):
                    q_ids.append(n)
                    q_pts.append((a, *xy_cols[n][a]))
            # new tracks at the window start and middle, away from continuing ones
            for t in sorted({a, (a + b - 1) // 2} | ({T - 1} if b == T else set())):
                occ = np.zeros((ph, pw), np.uint8)
                for n in q_ids:
                    p = xy_cols[n][t]
                    if np.isfinite(p[0]) and t == a:
                        cv2.circle(occ, (int(p[0]), int(p[1])), min_dist, 1, -1)
                excl = occ.astype(bool)
                if masks is not None:
                    excl |= masks[t]
                for p in sample_queries(frames[t], excl, grid, corners, rng):
                    q_ids.append(new_track(t))
                    q_pts.append((t, *p))
            if not q_pts:
                continue
            q = np.array(q_pts, np.float64)
            video = torch.from_numpy(frames[a:b]).permute(0, 3, 1, 2).float().cuda()
            video = F.interpolate(video, model_hw, mode="bilinear", align_corners=True)[None]
            for s in range(0, len(q), chunk):
                qc = q[s:s + chunk]
                qm = np.column_stack([qc[:, 0] - a, to_model(qc[:, 1:], (pw, ph), model_wh)])
                with torch.no_grad():
                    tracks, vis, conf, _ = model(video=video, queries=torch.from_numpy(qm).float().cuda()[None],
                                                 iters=opt.get("iters", 6))
                xy = from_model(tracks[0].cpu().numpy(), (pw, ph), model_wh)  # [b-a, n, 2]
                vp, cp = vis[0].cpu().numpy(), conf[0].cpu().numpy()
                for j, n in enumerate(q_ids[s:s + chunk]):
                    xy_cols[n][a:b] = xy[:, j]
                    vis_cols[n][a:b] = vp[:, j]
                    conf_cols[n][a:b] = cp[:, j]
                    tq = int(qc[j, 0])
                    vis_cols[n][tq] = 1.0
                    # The model's own prediction is kept at the query frame too (not the exact query):
                    # its small constant offset (~+0.3 px measured vs ray-cast GT) is then the same in
                    # every frame, i.e. harmless, instead of making the query frame disagree with the rest.
                    if opt.get("exact_queries", False):
                        xy_cols[n][tq] = qc[j, 1:]
            del video
            torch.cuda.empty_cache()

        xy_proxy = np.stack(xy_cols, 1)  # [T, N, 2]
        vis_p = np.stack(vis_cols, 1).astype(np.float32)
        conf_p = np.stack(conf_cols, 1).astype(np.float32)
        vis = (vis_p > 0.5) & np.isfinite(xy_proxy[..., 0])
        inside = (xy_proxy[..., 0] > 0) & (xy_proxy[..., 0] < pw) & (xy_proxy[..., 1] > 0) & (xy_proxy[..., 1] < ph)
        vis &= inside
        n_masked = 0
        if masks is not None:
            xi = np.clip(np.nan_to_num(xy_proxy[..., 0]).astype(int), 0, pw - 1)
            yi = np.clip(np.nan_to_num(xy_proxy[..., 1]).astype(int), 0, ph - 1)
            hit = masks[np.arange(T)[:, None], yi, xi] & vis
            n_masked = int(hit.sum())
            vis &= ~hit

        xy_full = xy_proxy * np.array([W / pw, H / ph])
        N = xy_full.shape[1]
        holdout = rng.random(N) < opt.get("holdout", 0.1)
        length = vis.sum(0)
        meta = {"backend": "cotracker3_offline", "checkpoint": CKPT.name, "model_size": list(model_hw),
                "proxy_size": [pw, ph], "windows": wins, "n_tracks": int(N), "vis_threshold": 0.5,
                "masked_observations": n_masked}
        np.savez_compressed(out_dir / "tracks.npz", xy=xy_full.astype(np.float32), vis=vis,
                            conf=conf_p, vis_prob=vis_p.astype(np.float16),
                            query_frame=np.array(qframe, np.int32), holdout=holdout,
                            meta=np.array(json.dumps(meta)))
        res.outputs["tracks"] = "tracks.npz"
        res.stats.update({"n_tracks": int(N), "mean_visible_per_frame": float(vis.sum(1).mean()),
                          "median_track_length": float(np.median(length)), "max_track_length": int(length.max()),
                          "windows": len(wins), "model_size": list(model_hw), "masked_observations": n_masked,
                          "query_mode": "exact" if opt.get("exact_queries", False) else "model"})


if __name__ == "__main__":
    main()
