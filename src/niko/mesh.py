"""`niko mesh`: an editable 3D mesh of the shot's static scene, from the solved cameras.

COLMAP multi-view stereo (backends/colmap/niko_colmap/mesh.py) runs on frames spread over the
shot with the solve's own cameras, so the mesh lines up with the camera and the footage exactly.
The surface (backends/da3/niko_da3/mesh.py, Open3D) is then cleaned for use: points on moving
things (the SAM 3 masks) and far junk removed, floating bits dropped, small holes filled, a light
smoothing. Two files, in the solve world like cameras.json (the Blender scene puts them in metres
when the solve has a real size):
  selected/mesh.ply      detailed surface with vertex colours
  selected/mesh_sim.ply  simplified, hole-filled copy for physics (colliders, simulations)
  selected/dense_points.ply  the fused stereo points
Quality: fast / good / high = how many frames and how large they are (time grows with both).
Needs parallax: a rotation-only (tripod) solve sees every point from one place, no 3D.

Up to 0.3.9 the masked pixels were painted black before stereo: no surface on moving things, but
black blobs along their edges and in the vertex colours. Now the images stay as they are and fused
points that fall on a mask in most of the frames that see them are removed instead.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .backend import run_backend
from .camio import CameraTrack
from .plyio import read_ply_xyz

# frames used, their longest side in pixels, surface detail (voxels across the scene), Poisson depth
# (synthetic drone_orbit against its true geometry, scripts/dev/mesh_check.py: finer voxels trimmed
# away thinly seen surface; detail comes from the frames and Poisson depth, 1500 cells keep coverage)
QUALITY = {
    "fast": {"images": 40, "size": 1280, "cells": 1200, "depth": 10, "smooth": 3,
             "tex_tris": 150_000, "tex_views": 16, "tex_size": 2048},
    "good": {"images": 60, "size": 1600, "cells": 1500, "depth": 11, "smooth": 5,
             "tex_tris": 300_000, "tex_views": 24, "tex_size": 4096},
    "high": {"images": 90, "size": 2400, "cells": 1500, "depth": 11, "smooth": 5,
             "tex_tris": 500_000, "tex_views": 36, "tex_size": 4096},
}
TEXTURED = ("mesh_textured.obj", "mesh_textured.mtl", "mesh_textured_albedo.png")


def build_mesh(solve_dir: str | Path, quality: str = "good", max_images: int | None = None,
               max_image_size: int | None = None, log=print, reuse_stereo: bool = False) -> dict:
    """reuse_stereo: keep an earlier run's fused stereo points and redo only the surface."""
    t0 = time.time()
    q = dict(QUALITY[quality])
    if max_images:
        q["images"] = max_images
    if max_image_size:
        q["size"] = max_image_size
    solve_dir = Path(solve_dir)
    sel = solve_dir / "selected"
    trk = CameraTrack.load(sel / "cameras.json")
    shot = json.loads((solve_dir / "shot.json").read_text())
    if trk.extra.get("rotation_only"):
        raise RuntimeError("the camera only rotates (tripod): every point is seen from one place, no 3D to mesh")
    if not (sel / "points.ply").exists():
        raise RuntimeError("the solve has no 3D points")
    surface_opts = {k: q[k] for k in ("cells", "depth", "smooth", "tex_tris", "tex_views", "tex_size")}
    work = solve_dir / "mesh"
    if reuse_stereo and (work / "dense" / "fused.ply").exists():
        return _surface(solve_dir, work, sel, surface_opts, log, t0)
    X = read_ply_xyz(sel / "points.ply")
    v = np.nonzero(trk.valid)[0]
    frames = v[np.unique(np.round(np.linspace(0, len(v) - 1, min(q["images"], len(v)))).astype(int))]
    if work.exists():
        shutil.rmtree(work)
    (work / "images").mkdir(parents=True)
    (work / "masks").mkdir()
    # full-resolution frames when the solve kept them (4K detail), else the proxy
    ext = shot.get("frame_format", "jpg")
    full = solve_dir / "frames"
    use_full = (full / f"{int(frames[0]):06d}.{ext}").exists()
    src_dir, iw, ih = (full, trk.width, trk.height) if use_full else (
        solve_dir / "proxy", shot["proxy"]["width"], shot["proxy"]["height"])
    S = np.diag([iw / trk.width, ih / trk.height, 1.0])
    names, masked = [], []
    for i in frames:
        name = f"{int(i):06d}.jpg"
        img = cv2.imread(str(src_dir / f"{int(i):06d}.{ext if use_full else 'jpg'}"))
        cv2.imwrite(str(work / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        m = solve_dir / "masks" / f"{int(i):06d}.png"
        if m.exists():
            mk = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            mk = cv2.resize(mk, (iw, ih), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(work / "masks" / f"{int(i):06d}.png"), mk)
            masked.append(float((mk > 127).mean()))
        names.append(name)
    np.savez(work / "mesh_input.npz", names=np.array(names), K=np.einsum("ij,njk->nik", S, trk.K[frames]),
             dist=trk.dist[frames], R=trk.R[frames], t=trk.t[frames], X=X, width=iw, height=ih)
    log(f"[mesh] {quality} quality: multi-view stereo on {len(frames)} frames ({iw}x{ih}, at most "
        f"{q['size']} px), {100 * np.mean(masked) if masked else 0:.0f} % of each frame masked on average")
    res = run_backend("colmap", "mesh", solve_dir, work, {"width": iw, "height": ih, "max_image_size": q["size"]})
    out = _surface(solve_dir, work, sel, surface_opts, log, t0)
    out.update({"frames": len(frames), "image_size": [iw, ih], "quality": quality,
                "relaxed_angles": res["stats"].get("relaxed_angles")})
    return out


def texture(solve_dir: Path, work: Path, sel: Path, opts: dict, log) -> str | None:
    """The textured copy (extra: a failure leaves the coloured mesh in place). Open3D's UVAtlas crashes on
    some meshes at 300k triangles (real clip 04: segfault, also with partitions or slivers removed; fine
    at 100k), so it is tried again with fewer triangles."""
    tris = int(opts.get("tex_tris", 300_000))
    last = ""
    for attempt in range(3):
        try:
            tex = run_backend("da3", "texture", solve_dir, work, {**opts, "tex_tris": tris})["stats"]
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:120]}"
            tris = int(tris * 0.6)
            continue
        for name in TEXTURED:
            shutil.copyfile(work / name, sel / name)
        log(f"[mesh] textured copy: {tex['triangles']:,} triangles, a {tex['texture_size']} px texture from "
            f"{tex['views']} frames ({tex['seconds']} s)")
        return str(sel / TEXTURED[0])
    log(f"[mesh] no textured copy ({last})")
    return None


def _surface(solve_dir: Path, work: Path, sel: Path, opts: dict, log, t0: float) -> dict:
    surf = run_backend("da3", "mesh", solve_dir, work, opts)
    st = surf["stats"]
    log(f"[mesh] {st['fused_points']:,} fused points: {st.get('removed_masked', 0):,} on moving things and "
        f"{st.get('removed_far', 0):,} far away removed; surface {st['vertices']:,} vertices, "
        f"{st['triangles']:,} triangles ({st.get('components_removed', 0)} floating pieces removed); "
        f"simulation copy {st.get('sim_triangles', 0):,} triangles")
    shutil.copyfile(work / "mesh.ply", sel / "mesh.ply")
    if (work / "mesh_sim.ply").exists():
        shutil.copyfile(work / "mesh_sim.ply", sel / "mesh_sim.ply")
    shutil.copyfile(work / "dense" / "fused.ply", sel / "dense_points.ply")
    out = {"mesh": str(sel / "mesh.ply"), "mesh_sim": str(sel / "mesh_sim.ply"),
           "dense_points": str(sel / "dense_points.ply"), "fused_points": st.get("fused_points"),
           "vertices": st["vertices"], "triangles": st["triangles"], "seconds": round(time.time() - t0, 1)}
    tex = texture(solve_dir, work, sel, opts, log)
    if tex:
        out["textured"] = tex
    out["seconds"] = round(time.time() - t0, 1)
    log(f"[mesh] -> {out['mesh']} in {out['seconds']} s")
    return out
