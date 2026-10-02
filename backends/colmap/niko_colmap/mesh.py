"""Scene mesh from a solved shot (task "mesh"): COLMAP multi-view stereo on the solve's own cameras.

Input  <out_dir>/mesh_input.npz  (written by niko.mesh):
  names [F] str (files in <out_dir>/images, moving objects already painted black),
  K [F,3,3], dist [F,5] (OpenCV, for those images), R [F,3,3], t [F,3] (world -> camera),
  X [P,3] solve points (only to give PatchMatch its depth range per image)
Options: max_image_size (1600), num_iterations (5), min_num_pixels (4)
Output <out_dir>/dense/fused.ply (coloured points with normals, the solve's world coordinates);
       the surface is made from it by the da3 env's "mesh" task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap


def _project(K, dist, R, t, X):
    Xc = X @ R.T + t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = x * x + y * y
    s = 1 + r2 * (dist[0] + r2 * dist[1])
    return np.stack([K[0, 0] * x * s + K[0, 2], K[1, 1] * y * s + K[1, 2]], 1), z


def run(job: dict, res) -> None:
    out = Path(job["out_dir"])
    opt = job.get("options", {})
    d = np.load(out / "mesh_input.npz")
    names, K, dist, R, t, X = [str(n) for n in d["names"]], d["K"], d["dist"], d["R"], d["t"], d["X"]
    W, H = int(opt["width"]), int(opt["height"])  # of the images in out/images
    F = len(names)
    rng = np.random.default_rng(0)

    rec = pycolmap.Reconstruction()
    obs = {}  # point index -> [(image_id, point2D index)]
    for i in range(F):
        cam = pycolmap.Camera(model="OPENCV", width=W, height=H, camera_id=i + 1,
                              params=[K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2],
                                      dist[i, 0], dist[i, 1], dist[i, 2], dist[i, 3]])
        rec.add_camera_with_trivial_rig(cam)
        uv, z = _project(K[i], dist[i], R[i], t[i], X)
        inside = np.nonzero((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))[0]
        if len(inside) > 3000:
            inside = rng.choice(inside, 3000, replace=False)
        img = pycolmap.Image(name=names[i], keypoints=uv[inside], camera_id=i + 1, image_id=i + 1)
        rec.add_image_with_trivial_frame(img, pycolmap.Rigid3d(pycolmap.Rotation3d(R[i]), t[i]))
        for k, p in enumerate(inside):
            obs.setdefault(int(p), []).append((i + 1, k))
    n_pts = 0
    for p, els in obs.items():
        if len(els) >= 2:
            rec.add_point3D(X[p], pycolmap.Track([pycolmap.TrackElement(a, b) for a, b in els]))
            n_pts += 1
    sparse = out / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    rec.write(sparse)

    dense = out / "dense"
    pycolmap.undistort_images(dense, sparse, out / "images", output_type="COLMAP")
    pm = pycolmap.PatchMatchOptions()
    pm.max_image_size = int(opt.get("max_image_size", 1600))
    pm.num_iterations = int(opt.get("num_iterations", 5))
    pm.geom_consistency = True
    pycolmap.patch_match_stereo(dense, options=pm)
    maps = dense / "stereo" / "depth_maps"
    relaxed = False
    if not any(maps.glob("*.geometric.bin")):
        # real clip 02 (telephoto, the camera barely moves): "no source images" for every frame, the
        # views are less than 1 degree apart. One try with small angles; noisier depth, still depth.
        pm.min_triangulation_angle = 0.2
        pm.filter_min_triangulation_angle = 0.5
        pycolmap.patch_match_stereo(dense, options=pm)
        relaxed = True
    if not any(maps.glob("*.geometric.bin")):
        raise RuntimeError("the camera moves too little for 3D: no two frames see the scene from angles far "
                           "enough apart (a tripod, a slow push with a long lens)")
    fo = pycolmap.StereoFusionOptions()
    fo.max_image_size = pm.max_image_size
    fo.min_num_pixels = int(opt.get("min_num_pixels", 4))
    fused = pycolmap.stereo_fusion(dense / "fused.ply", dense, options=fo, output_type="PLY")
    n_fused = fused.num_points3D() if hasattr(fused, "num_points3D") else len(fused.points3D)
    if n_fused < 1000:
        raise RuntimeError(f"multi-view stereo fused only {n_fused} points: the camera moves too little (or the "
                           "scene has too little texture) for 3D. The camera track itself is not affected")
    # the surface is built in the da3 env (Open3D): COLMAP's Poisson mesher returned vertices at
    # +-1e38 and NaN on the first real clip (niko_da3/mesh.py)
    res.outputs.update({"fused": "dense/fused.ply"})
    res.stats.update({"images": F, "sparse_points": n_pts, "fused_points": int(n_fused), "relaxed_angles": relaxed})
