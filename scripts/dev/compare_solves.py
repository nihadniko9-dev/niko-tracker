"""How far apart are two solves of the same clip (no ground truth: e.g. keyframe stride vs every frame)?

usage: python scripts/dev/compare_solves.py <solve_dir_a> <solve_dir_b>
Prints, over the frames both solved: rotation difference after the best global rotation (deg),
camera path difference after Sim(3) alignment (% of the path length and of median point depth),
focal difference (%), and each solve's average held-out error.
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko import metrics
from niko.camio import CameraTrack
from niko.plyio import read_ply_xyz
from niko.sim3 import umeyama


def main():
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    A = CameraTrack.load(a_dir / "selected" / "cameras.json")
    B = CameraTrack.load(b_dir / "selected" / "cameras.json")
    v = A.valid & B.valid
    rot = metrics.rotation_errors_deg(A.R_c2w[v], B.R_c2w[v])
    Ca, Cb = A.centers[v], B.centers[v]
    sim = umeyama(Ca, Cb, with_scale=True)
    d = np.linalg.norm(sim.apply(Ca) - Cb, axis=1)
    path = float(np.sum(np.linalg.norm(np.diff(Cb, axis=0), axis=1)))
    X = read_ply_xyz(b_dir / "selected" / "points.ply")
    t = np.nonzero(v)[0][len(np.nonzero(v)[0]) // 2]
    depth = float(np.median((X @ B.R[t].T + B.t[t])[:, 2]))
    fa, fb = np.median(A.K[v, 0, 0]), np.median(B.K[v, 0, 0])
    ra = json.loads((a_dir / "solve.json").read_text()).get("solve_error", {})
    rb = json.loads((b_dir / "solve.json").read_text()).get("solve_error", {})
    print(f"frames compared {int(v.sum())} / {A.n_frames}")
    print(f"rotation difference: median {np.median(rot):.4f} deg, max {rot.max():.4f} deg")
    print(f"path difference: rms {100 * np.sqrt(np.mean(d ** 2)) / path:.3f} % of path length, "
          f"{100 * np.sqrt(np.mean(d ** 2)) / depth:.3f} % of median depth")
    print(f"focal: {fa:.1f} vs {fb:.1f} px ({100 * abs(fa - fb) / fb:.2f} %)")
    print(f"average error: {ra.get('average_px')} vs {rb.get('average_px')} px")


if __name__ == "__main__":
    main()
