"""How good is a scene mesh? Against a benchmark scene's true geometry (scripts/dev/gt_scene_mesh.py),
after aligning the solve to the ground truth through the cameras (Umeyama on camera centres):

  accuracy   median and 90th-percentile distance from the mesh to the true surface
  precision  share of the mesh within tau of the true surface
  recall     share of the true surface the cameras see (in a frustum, within 3x the median depth)
             that has mesh within tau
  pieces     connected pieces of the mesh (floating junk shows up here)
tau = 1 % and 2 % of the median camera-to-scene depth.

usage (da3 env, PYTHONPATH=src): python scripts/dev/mesh_check.py <solve_dir> <gt cameras.json> <gt mesh.ply> <mesh.ply> ...
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from niko.camio import CameraTrack
from niko.sim3 import umeyama


def visible(P, gt: CameraTrack, frames, max_depth):
    keep = np.zeros(len(P), bool)
    for f in frames:
        c = P @ gt.R[f].T + gt.t[f]
        z = c[:, 2]
        ok = (z > 0) & (z < max_depth)
        u = gt.K[f, 0, 0] * c[:, 0] / np.where(ok, z, 1) + gt.K[f, 0, 2]
        v = gt.K[f, 1, 1] * c[:, 1] / np.where(ok, z, 1) + gt.K[f, 1, 2]
        keep |= ok & (u >= 0) & (u < gt.width) & (v >= 0) & (v < gt.height)
    return keep


def main():
    o3d.utility.random.seed(0)  # the same surface samples on every run
    solve_dir, gt_cams, gt_mesh = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    est = CameraTrack.load(solve_dir / "selected" / "cameras.json")
    gt = CameraTrack.load(gt_cams)
    ok = est.valid & gt.valid
    sim = umeyama(est.centers[ok], gt.centers[ok], with_scale=True)
    fr = np.nonzero(ok)[0][:: max(1, ok.sum() // 30)]
    G = o3d.io.read_triangle_mesh(gt_mesh)
    Pg = np.asarray(G.sample_points_uniformly(400_000).points)
    depth = np.median([np.median((Pg @ gt.R[f].T + gt.t[f])[:, 2][(Pg @ gt.R[f].T + gt.t[f])[:, 2] > 0])
                       for f in fr])
    Pg = Pg[visible(Pg, gt, fr, 3 * depth)]
    tree_g = cKDTree(Pg)
    print(f"true surface seen: {len(Pg):,} samples, median depth {depth:.3g} GT units")
    print(f"{'mesh':52s}{'tris':>10}{'pieces':>8}{'acc med':>9}{'acc 90%':>9}"
          f"{'prec 1%':>9}{'rec 1%':>8}{'prec 2%':>9}{'rec 2%':>8}")
    for path in sys.argv[4:]:
        M = o3d.io.read_triangle_mesh(path)
        V = np.asarray(M.vertices)
        M.vertices = o3d.utility.Vector3dVector(sim.apply(V))
        _, counts, _ = M.cluster_connected_triangles()
        Pm = np.asarray(M.sample_points_uniformly(200_000).points)
        d_mg, _ = tree_g.query(Pm, k=1, workers=-1)
        d_gm, _ = cKDTree(Pm).query(Pg, k=1, workers=-1)
        row = [f"{Path(path).parent.parent.name[-24:]}/{Path(path).name:<26s}", f"{len(M.triangles):>10,}",
               f"{len(counts):>8}", f"{100 * np.median(d_mg) / depth:>8.2f}%", f"{100 * np.percentile(d_mg, 90) / depth:>8.2f}%"]
        for tau in (0.01, 0.02):
            row += [f"{100 * np.mean(d_mg < tau * depth):>8.1f}%", f"{100 * np.mean(d_gm < tau * depth):>7.1f}%"]
        print("".join(row), flush=True)
    print("(distances as % of the median depth)")


if __name__ == "__main__":
    main()
