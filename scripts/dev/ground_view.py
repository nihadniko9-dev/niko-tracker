"""Draw candidate "up" directions and ground-plane points onto a frame of a real solve, to judge them
by eye: vertical things in the picture (posts, walls, people) should follow the lines.
  green dots  inliers of the ground plane found from the points
  red lines   verticals along that plane's normal
  blue lines  verticals along the up given by the cameras' level right axes

usage (niko env): python scripts/dev/ground_view.py <solve_dir> <frame index> <out.jpg> [min height share]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from niko.camio import CameraTrack
from niko.pipeline.export import median_depth
from niko.plyio import read_ply_xyz

sys.path.insert(0, str(Path(__file__).parent))
from ground_variants import fit, up_from_right_axes  # noqa: E402


def project(trk, f, P):
    c = P @ trk.R[f].T + trk.t[f]
    z = c[:, 2]
    uv = np.column_stack([trk.K[f, 0, 0] * c[:, 0] / z + trk.K[f, 0, 2], trk.K[f, 1, 1] * c[:, 1] / z + trk.K[f, 1, 2]])
    return uv, z


def main():
    d, f, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    min_share = float(sys.argv[4]) if len(sys.argv) > 4 else None
    trk = CameraTrack.load(d / "selected" / "cameras.json")
    X = read_ply_xyz(d / "selected" / "points.ply")
    depth = median_depth(trk, X)
    n_x, up0, _ = up_from_right_axes(trk)
    r = fit(X, trk, depth, up0, 0.7, np.random.default_rng(0), None if min_share is None else min_share * depth)
    frames = sorted((d / "proxy").glob("*.jpg"))
    im = cv2.imread(str(frames[f]))
    sx, sy = im.shape[1] / trk.width, im.shape[0] / trk.height
    if r is not None:
        n, h, _ = r
        scale = float(np.median(np.linalg.norm(X - np.median(X, 0), axis=1)))
        C = trk.centers[f]
        c0 = X[np.abs((X - (C - h * n)) @ n) < 0.01 * scale]
        uv, z = project(trk, f, c0)
        for (u, v), zz in zip(uv, z):
            if zz > 0:
                cv2.circle(im, (int(u * sx), int(v * sy)), 3, (60, 220, 40), -1)
    rng = np.random.default_rng(1)
    uv, z = project(trk, f, X)
    vis = np.nonzero((z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < trk.width) & (uv[:, 1] > 0) & (uv[:, 1] < trk.height))[0]
    for i in rng.choice(vis, min(40, len(vis)), replace=False):
        for up, col in (([r[0]] if r is not None else []) and ((r[0], (40, 40, 230)),)) + ((n_x, (255, 120, 40)),):
            seg = np.stack([X[i], X[i] + 0.15 * z[i] * up])
            q, zz = project(trk, f, seg)
            if (zz > 0).all():
                a, b = (q * (sx, sy)).astype(int)
                cv2.line(im, tuple(map(int, a)), tuple(map(int, b)), col, 3)
    cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, "plane:", None if r is None else (round(float(100 * r[1] / depth), 1), "% of depth"),
          "angle plane vs right-axes up:", None if r is None else round(float(np.degrees(np.arccos(np.clip(r[0] @ n_x, -1, 1)))), 1))


if __name__ == "__main__":
    main()
