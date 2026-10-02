"""The levelled world's ground plane on finished solves: how high the cameras are above it, as a share
of the median camera-to-point depth (a plane through or above the cameras is not a ground). With
--gt <bench folder>, synthetic shots are also compared with the truth (ground at Z = 0): the tilt of
the found up direction and the true camera height as a share of the true median depth.

usage (niko env): python scripts/dev/ground_check.py [--gt ~/niko/bench/synthetic_v1] <solve_dir> ...
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.export_ae import world_alignment
from niko.pipeline.export import median_depth
from niko.plyio import read_ply_xyz


def main():
    args = sys.argv[1:]
    gt_root = None
    if args[:1] == ["--gt"]:
        gt_root, args = Path(args[1]), args[2:]
    for d in map(Path, args):
        sel = d / "selected"
        trk = CameraTrack.load(sel / "cameras.json")
        X = read_ply_xyz(sel / "points.ply") if (sel / "points.ply").exists() else None
        A, origin, how = world_alignment(trk, X, np.random.default_rng(0))
        up = -A[1]
        h = (trk.centers[trk.valid] - origin) @ up
        depth = median_depth(trk, X) if X is not None else None
        line = f"{d.name[:28]:28s} {how:13s}"
        line += f" height {100 * np.median(h) / depth:7.2f} % of depth" if depth else " height       -"
        g = gt_root / d.name if gt_root else None
        if g is not None and (g / "gt" / "cameras.json").exists():
            gt = CameraTrack.load(g / "gt" / "cameras.json")
            ok = trk.valid & gt.valid
            # the rotation from the camera orientations (camera centres alone are ambiguous on a line)
            U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", gt.R_c2w[ok], trk.R_c2w[ok]))
            Ra = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
            up_gt = Ra @ up
            tilt = np.degrees(np.arccos(np.clip(up_gt[2], -1, 1)))
            true_depth = json.loads((g / "scene.json").read_text())["median_depth"]
            line += f" | true {100 * np.median(gt.centers[gt.valid][:, 2]) / true_depth:6.2f} %, up tilt {tilt:5.1f} deg"
        print(line, flush=True)


if __name__ == "__main__":
    main()
