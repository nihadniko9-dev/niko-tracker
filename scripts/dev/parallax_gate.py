"""How much parallax does each benchmark shot have, measured from its SIFT tracks alone?

For frame pairs `gap` frames apart a RANSAC homography is fitted to the shared SIFT tracks; a
camera that only rotates (and zooms) maps every point by one homography, so the median transfer
residual stays at the matching noise, while real parallax pushes it up. Printed per shot next to
the ground-truth camera path extent / median depth, to set the gate for the rotation-only solver.

usage: python scripts/dev/parallax_gate.py <run_dir> [shot ...]
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from niko.backend import run_backend
from niko.camio import CameraTrack
from niko.pipeline.solve import homography_residual_px


def main():
    run = Path(sys.argv[1])
    shots = sys.argv[2:] or sorted(p.name for p in run.iterdir() if (p / "shot.json").exists())
    print(f"{'shot':<26}{'H resid px':>11}{'pairs':>7}{'gt path/depth':>15}")
    for name in shots:
        d = run / name
        shot = json.loads((d / "shot.json").read_text())
        db = d / "candidates" / "colmap_global" / "colmap" / "database.db"
        if not db.exists():
            print(f"{name:<26} no database")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            run_backend("colmap", "sift_tracks", d, Path(tmp),
                        {"db": str(db), "width": shot["width"], "height": shot["height"],
                         "proxy_width": shot["proxy"]["width"], "proxy_height": shot["proxy"]["height"]})
            sift = dict(np.load(Path(tmp) / "sift_tracks.npz"))
        res, pairs = homography_residual_px(sift)
        gt = CameraTrack.load(Path(shot.get("source", "")).parent / "gt/cameras.json") \
            if (Path(shot.get("source", "")).parent / "gt/cameras.json").exists() else None
        ratio = float("nan")
        if gt is not None and gt.extra.get("median_depth"):
            C = gt.centers[gt.valid]
            ratio = float(np.ptp(C, axis=0).max()) / gt.extra["median_depth"]
        print(f"{name:<26}{res:>11.3f}{pairs:>7}{ratio:>15.4f}")


if __name__ == "__main__":
    main()
