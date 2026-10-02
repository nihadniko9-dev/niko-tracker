"""Does SAM 3 on every k-th frame (neighbour union in between) mask what SAM 3 on every frame masks?

usage: python scripts/dev/mask_stride_check.py <solve_dir with masks/> <k> [prompts]
Runs the sam3 masks task with frame_stride k on a hard-linked copy of the shot (the solve's own
masks are not touched) and compares, per frame, with the existing full-rate masks:
  recall = masked-by-full pixels also masked by strided (what a missed mover would cost),
  extra  = strided-only pixels / strided pixels (texture thrown away for nothing).
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from niko.backend import run_backend


def main():
    src = Path(sys.argv[1])
    k = int(sys.argv[2])
    prompts = sys.argv[3].split(",") if len(sys.argv) > 3 else json.loads(
        (src / "masks" / "masks.json").read_text())["prompts"]
    tmp = Path(os.environ.get("NIKO_HOME", "/tmp")) / "tmp_mask_stride" / src.name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copyfile(src / "shot.json", tmp / "shot.json")
    os.symlink(src / "proxy", tmp / "proxy")
    t0 = time.time()
    run_backend("sam3", "masks", tmp, tmp / "jobs" / "masks", {"prompts": prompts, "frame_stride": k})
    secs = time.time() - t0
    full = sorted((src / "masks").glob("*.png"))
    rec, extra, worst = [], [], (1.0, None)
    for f in full:
        a = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127
        b = cv2.imread(str(tmp / "masks" / f.name), cv2.IMREAD_GRAYSCALE) > 127
        if a.sum():
            r = float((a & b).sum() / a.sum())
            rec.append(r)
            if r < worst[0]:
                worst = (r, f.name)
        if b.sum():
            extra.append(float((b & ~a).sum() / b.sum()))
    print(f"frame_stride {k}: {secs:.0f} s; recall median {np.median(rec):.4f}, 1st pct {np.percentile(rec, 1):.4f}, "
          f"worst {worst[0]:.4f} ({worst[1]}); extra median {np.median(extra):.4f}")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
