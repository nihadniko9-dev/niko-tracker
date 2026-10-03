"""Surface from fused multi-view-stereo points (task "mesh", da3 env: Open3D 0.20).

Input  <out_dir>/dense/fused.ply   (COLMAP stereo fusion: points, normals, colours; solve world)
       <out_dir>/mesh_input.npz    (the stereo frames' cameras) and <out_dir>/masks/<frame>.png
       (255 = a moving thing, at the stereo image size) when the solve has masks
Output <out_dir>/mesh.ply          detailed coloured surface, same world
       <out_dir>/mesh_sim.ply      simplified, hole-filled copy for physics
Options: cells (voxels across the scene, 1500), depth (Poisson, 11), smooth (Taubin iterations, 5),
         density_quantile (0.06), trim_voxels (5), radius_outliers (0), min_piece (0.001),
         sim_triangles (150 000), max_triangles (2 000 000)

Steps, each against a fault seen on real clips:
  1. points on moving things: removed when they land on a mask in at least half of the stereo
     frames that see them (the images are no longer painted black: that left black blobs)
  2. far junk (sky, horizon): points further from every camera than 4 x the median distance
  3. statistical outliers (a radius-outlier pass is off by default: it removed the thinly seen
     parts of the scene, synthetic drone_orbit completeness 32 -> 19 %)
  4. screened Poisson; only surface with data behind it kept (lowest-density vertices and vertices
     further than 5 voxels from any point: Poisson otherwise closes a bubble around the camera)
  5. floating pieces under 0.1 % of the largest piece removed; Taubin smoothing (keeps the volume);
     holes up to 10 voxels across filled
"""

from __future__ import annotations

from pathlib import Path

import time

import numpy as np


def _project(K, dist, R, t, X):
    Xc = X @ R.T + t
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
    x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
    r2 = x * x + y * y
    s = 1 + r2 * (dist[0] + r2 * dist[1])
    return np.stack([K[0, 0] * x * s + K[0, 2], K[1, 1] * y * s + K[1, 2]], 1), z


def _on_moving_things(P, out: Path) -> np.ndarray:
    """[N] bool: the point lands on a mask in at least half of the stereo frames that see it."""
    import cv2

    inp = out / "mesh_input.npz"
    if not inp.exists() or not (out / "masks").exists():
        return np.zeros(len(P), bool)
    d = np.load(inp)
    W, H = (int(d["width"]), int(d["height"])) if "width" in d.files else (0, 0)
    seen = np.zeros(len(P), np.int32)
    hit = np.zeros(len(P), np.int32)
    for i, name in enumerate(d["names"]):
        mf = out / "masks" / (Path(str(name)).stem + ".png")
        if not mf.exists():
            continue
        mk = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE) > 127
        h, w = mk.shape
        uv, z = _project(d["K"][i], d["dist"][i], d["R"][i], d["t"][i], P)
        u, v = np.round(uv[:, 0] * w / (W or w)).astype(int), np.round(uv[:, 1] * h / (H or h)).astype(int)
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        seen += inside
        hit[inside] += mk[v[inside], u[inside]]
    return (seen > 0) & (hit * 2 >= seen)


def _far(P, out: Path) -> np.ndarray:
    """[N] bool: further from every stereo camera than 4 x the median nearest-camera distance."""
    inp = out / "mesh_input.npz"
    if not inp.exists():
        return np.zeros(len(P), bool)
    d = np.load(inp)
    C = -np.einsum("nji,nj->ni", d["R"], d["t"])  # camera centres
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(C).query(P, k=1, workers=-1)
    return dist > 4.0 * float(np.median(dist))


def run(job: dict, res) -> None:
    import open3d as o3d
    from scipy.spatial import cKDTree

    out = Path(job["out_dir"])
    opt = job.get("options", {})
    pcd = o3d.io.read_point_cloud(str(out / "dense" / "fused.ply"))
    n0 = len(pcd.points)
    if n0 < 1000:
        raise RuntimeError(f"only {n0} fused points")
    P = np.asarray(pcd.points)
    moving = _on_moving_things(P, out)
    far = _far(P, out) & ~moving
    pcd = pcd.select_by_index(np.nonzero(~(moving | far))[0])
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    P = np.asarray(pcd.points)
    extent = float((np.percentile(P, 98, axis=0) - np.percentile(P, 2, axis=0)).max())
    voxel = extent / float(opt.get("cells", 1500))
    pcd = pcd.voxel_down_sample(voxel)
    if int(opt.get("radius_outliers", 0)):  # off: it cut the thinly seen parts (drone_orbit recall 32 -> 19 %)
        pcd, _ = pcd.remove_radius_outlier(nb_points=4, radius=3 * voxel)
    if len(pcd.points) < 1000:
        raise RuntimeError(f"only {len(pcd.points)} points left after cleaning")
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(opt.get("depth", 11)))
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, float(opt.get("density_quantile", 0.06))))
    V = np.asarray(mesh.vertices)
    d, _ = cKDTree(np.asarray(pcd.points)).query(V, k=1, workers=-1)
    mesh.remove_vertices_by_mask(~np.isfinite(d) | (d > float(opt.get("trim_voxels", 5)) * voxel))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # floating pieces: under `min_piece` of the largest connected piece (0.1 %: at 1 % the thin
    # posts along the river in Nihad's DJI 0079 went with the junk)
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    small = (counts < max(50, float(opt.get("min_piece", 0.001)) * counts.max())
             if len(counts) else np.zeros(0, bool))
    mesh.remove_triangles_by_mask(small[labels])
    mesh.remove_unreferenced_vertices()
    if int(opt.get("smooth", 5)) > 0:
        colors = np.asarray(mesh.vertex_colors).copy()
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(opt.get("smooth", 5)))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh = _fill_holes(mesh, 10 * voxel)
    target = int(opt.get("max_triangles", 2_000_000))
    if len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target)
    mesh.compute_vertex_normals()
    if not len(mesh.triangles):
        raise RuntimeError("no surface left after cleaning")
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh, write_ascii=False, write_vertex_normals=True,
                               write_vertex_colors=True)

    # physics copy: fewer triangles, larger holes closed, no non-manifold edges
    sim = mesh.simplify_quadric_decimation(min(int(opt.get("sim_triangles", 150_000)), len(mesh.triangles)))
    sim.remove_duplicated_vertices()
    sim.remove_degenerate_triangles()
    sim.remove_non_manifold_edges()
    sim.remove_unreferenced_vertices()
    sim = _fill_holes(sim, 30 * voxel)
    sim.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "mesh_sim.ply"), sim, write_ascii=False, write_vertex_normals=True,
                               write_vertex_colors=True)

    V = np.asarray(mesh.vertices)
    res.outputs.update({"mesh": "mesh.ply", "mesh_sim": "mesh_sim.ply"})
    res.stats.update({"fused_points": n0, "removed_masked": int(moving.sum()), "removed_far": int(far.sum()),
                      "points_after_cleaning": int(len(pcd.points)), "voxel": voxel,
                      "components_removed": int(small.sum()), "vertices": int(len(V)),
                      "triangles": int(len(mesh.triangles)), "sim_triangles": int(len(sim.triangles)),
                      "bbox_min": V.min(0).round(4).tolist(), "bbox_max": V.max(0).round(4).tolist()})


def _fill_holes(mesh, hole_size: float):
    """Holes up to hole_size across closed (Open3D tensor API); the mesh as it was if that fails."""
    import open3d as o3d

    try:
        filled = o3d.t.geometry.TriangleMesh.from_legacy(mesh).fill_holes(hole_size=float(hole_size)).to_legacy()
    except Exception:  # noqa: BLE001 - a mesh without holes to fill, or an Open3D build without it
        return mesh
    if len(filled.triangles) < len(mesh.triangles):
        return mesh
    if mesh.has_vertex_colors():
        # Open3D 0.20's fill_holes returns garbage vertex colours (+-3.7e19): each vertex takes the
        # colour of the nearest vertex of the mesh before filling (the filled holes add no vertices)
        from scipy.spatial import cKDTree

        _, idx = cKDTree(np.asarray(mesh.vertices)).query(np.asarray(filled.vertices), k=1, workers=-1)
        filled.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[idx])
    return filled



def _manifold(mesh, rounds: int = 10):
    """Remove what makes a mesh non-manifold (UVAtlas refuses it): duplicates, degenerate triangles,
    non-manifold edges and vertices."""
    for _ in range(rounds):
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        bad = np.asarray(mesh.get_non_manifold_vertices())
        if len(bad):
            mesh.remove_vertices_by_index(bad.tolist())
        mesh.remove_unreferenced_vertices()
        if mesh.is_edge_manifold() and mesh.is_vertex_manifold():
            break
    return mesh


def texture(job: dict, res) -> None:
    """mesh_textured.obj (+ .mtl, _albedo.png): the surface simplified to `tex_tris` triangles, unwrapped
    (UVAtlas) and coloured by projecting `tex_views` of the stereo frames into one `tex_size` texture.
    Real clip 03: 200k triangles + a 4096 texture show what the 1.8M-triangle coloured mesh shows."""
    import open3d as o3d

    out = Path(job["out_dir"])
    opt = job.get("options", {})
    t0 = time.time()
    mesh = o3d.io.read_triangle_mesh(str(out / "mesh.ply"))
    tris = int(opt.get("tex_tris", 300_000))
    if len(mesh.triangles) > tris:
        mesh = mesh.simplify_quadric_decimation(tris)
    mesh = _manifold(mesh)
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    size = int(opt.get("tex_size", 4096))
    tm.compute_uvatlas(size=size, gutter=2.0)
    inp = np.load(out / "mesh_input.npz")
    names = list(inp["names"])
    k = min(int(opt.get("tex_views", 24)), len(names))
    pick = np.unique(np.round(np.linspace(0, len(names) - 1, k)).astype(int))
    imgs, Ks, Es = [], [], []
    for i in pick:
        imgs.append(o3d.t.io.read_image(str(out / "images" / str(names[i]))))
        Ks.append(o3d.core.Tensor(np.asarray(inp["K"][i], np.float64)))
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = inp["R"][i], inp["t"][i]
        Es.append(o3d.core.Tensor(E))
    tm.project_images_to_albedo(imgs, Ks, Es, size, True)
    o3d.t.io.write_triangle_mesh(str(out / "mesh_textured.obj"), tm)
    res.outputs.update({"mesh_textured": "mesh_textured.obj"})
    res.stats.update({"triangles": int(len(mesh.triangles)), "vertices": int(len(mesh.vertices)),
                      "views": int(len(pick)), "texture_size": size, "seconds": round(time.time() - t0, 1)})
