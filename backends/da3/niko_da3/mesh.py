"""Surface from fused multi-view-stereo points (task "mesh", da3 env: Open3D 0.20).

Input  <out_dir>/dense/fused.ply  (COLMAP stereo fusion: points, normals, colours; solve world)
Output <out_dir>/mesh.ply          (coloured triangle mesh, same world)

COLMAP's own Poisson mesher returned vertices at +-1e38 and NaN on the first real clip, so the
surface is built here: statistical outlier removal, voxel downsampling to ~1500 cells across the
scene, screened Poisson (Open3D), then only surface that has data behind it is kept: vertices of
the lowest-density 8 % and vertices further than 3 voxels from any point are removed (Poisson
closes a bubble around the scene otherwise, and the camera would sit inside it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run(job: dict, res) -> None:
    import open3d as o3d
    from scipy.spatial import cKDTree

    out = Path(job["out_dir"])
    opt = job.get("options", {})
    pcd = o3d.io.read_point_cloud(str(out / "dense" / "fused.ply"))
    n0 = len(pcd.points)
    if n0 < 1000:
        raise RuntimeError(f"only {n0} fused points")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    P = np.asarray(pcd.points)
    extent = float((np.percentile(P, 98, axis=0) - np.percentile(P, 2, axis=0)).max())
    voxel = extent / float(opt.get("cells", 1500))
    pcd = pcd.voxel_down_sample(voxel)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(opt.get("depth", 10)))
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, float(opt.get("density_quantile", 0.08))))
    V = np.asarray(mesh.vertices)
    d, _ = cKDTree(np.asarray(pcd.points)).query(V, k=1, workers=-1)
    mesh.remove_vertices_by_mask(~np.isfinite(d) | (d > 3 * voxel))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    target = int(opt.get("max_triangles", 1_500_000))
    if len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target)
    mesh.compute_vertex_normals()
    if not len(mesh.triangles):
        raise RuntimeError("no surface left after trimming")
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh, write_ascii=False, write_vertex_normals=True,
                               write_vertex_colors=True)
    V = np.asarray(mesh.vertices)
    res.outputs["mesh"] = "mesh.ply"
    res.stats.update({"fused_points": n0, "points_after_cleaning": int(len(pcd.points)), "voxel": voxel,
                      "vertices": int(len(V)), "triangles": int(len(mesh.triangles)),
                      "bbox_min": V.min(0).round(4).tolist(), "bbox_max": V.max(0).round(4).tolist()})
