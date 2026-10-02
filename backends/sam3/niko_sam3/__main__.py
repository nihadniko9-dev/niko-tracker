"""`python -m niko_sam3 job.json` - SAM 3.1 multiplex text-prompted video masks -> masks/*.png.

Job options:
  prompts     list of text prompts (default: person, car, animal, sky, water)
  dilate_px   grow the union mask by this many proxy pixels (default 4), for blurred edges
Writes <shot_dir>/masks/000000.png ... (full resolution, uint8, 255 = excluded) and
<shot_dir>/masks/masks.json (per-prompt coverage per frame, object ids).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from niko_backend_common import ResultWriter, read_job

from .common import CKPT, build_predictor, masks_for_prompt

DEFAULT_PROMPTS = ["person", "car", "animal", "sky", "water"]


def repo_commit() -> str:
    import sam3

    try:
        src = Path(sam3.__file__).resolve().parents[1]
        return subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    job = read_job(sys.argv[1])
    opt = job.get("options", {})
    shot_dir = Path(job["shot_dir"])
    with ResultWriter(job, versions={"sam3_commit": repo_commit(), "checkpoint": CKPT.name}) as res:
        shot = json.loads((shot_dir / "shot.json").read_text())
        W, H, T = shot["width"], shot["height"], shot["n_frames"]
        proxy = shot_dir / "proxy"
        prompts = opt.get("prompts", DEFAULT_PROMPTS)

        # SAM 3 keeps a whole session's frames and memory on the GPU (26.8 GB reserved for
        # 150 x 1080p frames, 23.4 GB for 50), so long clips run in short chunks. Only the per-frame union of masks is kept, so object
        # identities across chunks do not matter; every chunk re-detects each prompt on its first frame.
        chunk = int(opt.get("chunk_frames", 30))  # 50 frames still reserved 23.4 GB at 1080p
        frames = sorted(proxy.glob("*.jpg"))
        if len(frames) != T:
            raise ValueError(f"{len(frames)} proxy frames, shot has {T}")
        proxy_hw = cv2.imread(str(frames[0])).shape[:2]
        # frame_stride k: SAM 3 sees every k-th frame; a frame in between gets the union of its two
        # neighbours' masks (covers the object at both ends of its move). For 60 fps footage.
        stride = max(1, int(opt.get("frame_stride", 1)))
        seen = list(range(0, T, stride))
        if seen[-1] != T - 1:
            seen.append(T - 1)
        predictor = build_predictor()
        union = None
        report = {"prompts": prompts, "checkpoint": CKPT.name, "chunk_frames": chunk, "frame_stride": stride,
                  "coverage": {p: [] for p in prompts}, "objects_per_chunk": {p: [] for p in prompts}}
        work = Path(job["out_dir"]) / "chunks"
        for c0 in range(0, len(seen), chunk):
            idx = seen[c0:c0 + chunk]
            cdir = work / f"{c0:06d}"
            cdir.mkdir(parents=True, exist_ok=True)
            for k, i in enumerate(idx):  # SAM 3 expects <int>.jpg names starting at 0
                link = cdir / f"{k:06d}.jpg"
                if not link.exists():
                    link.symlink_to(frames[i])
            session = predictor.handle_request(request=dict(
                type="start_session", resource_path=str(cdir), offload_video_to_cpu=True))["session_id"]
            for text in prompts:
                per_frame, ids, empty = masks_for_prompt(predictor, session, text, len(idx), hw=proxy_hw)
                if empty:
                    report.setdefault("frames_without_output", {}).setdefault(text, 0)
                    report["frames_without_output"][text] += empty
                for k, i in enumerate(idx):
                    m = per_frame[k]
                    frame_union = m.any(0) if m.ndim == 3 and len(m) else np.zeros(m.shape[-2:], bool)
                    if union is None:
                        union = np.zeros((T, *frame_union.shape), bool)
                    union[i] |= frame_union
                    report["coverage"][text].append(float(frame_union.mean()))
                report["objects_per_chunk"][text].append(len(ids))
            predictor.handle_request(request=dict(type="close_session", session_id=session))
            import torch

            torch.cuda.empty_cache()
        shutil.rmtree(work, ignore_errors=True)
        if stride > 1:
            for a, b in zip(seen[:-1], seen[1:]):
                for t in range(a + 1, b):
                    union[t] = union[a] | union[b]
        report["object_ids"] = {p: list(range(max(report["objects_per_chunk"][p], default=0))) for p in prompts}

        d = int(opt.get("dilate_px", 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)) if d > 0 else None
        out = shot_dir / "masks"
        out.mkdir(exist_ok=True)
        total = []
        for t in range(T):
            m = union[t].astype(np.uint8) * 255
            if kernel is not None:
                m = cv2.dilate(m, kernel)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / f"{t:06d}.png"), m)
            total.append(float((m > 127).mean()))
        report["coverage"]["union_dilated"] = total
        report["dilate_px_proxy"] = d
        (out / "masks.json").write_text(json.dumps(report, indent=1))
        res.outputs["masks"] = str(out)
        res.stats.update({"mean_excluded_fraction": float(np.mean(total)),
                          "objects_per_prompt": {k: len(v) for k, v in report["object_ids"].items()}})


if __name__ == "__main__":
    main()
