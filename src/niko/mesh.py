"""`niko mesh`: an editable 3D mesh of the shot's static scene, from the solved cameras.

COLMAP multi-view stereo (backends/colmap/niko_colmap/mesh.py) runs on up to `max_images` frames
spread over the shot, using the solve's own cameras (so the mesh lines up with the camera and the
footage exactly), with moving things (the SAM 3 masks) painted black so they leave no surface.
Output (solve world, like cameras.json and points.ply): selected/mesh.ply (Poisson mesh, vertex
colours) and selected/dense_points.ply (fused MVS points).
Needs parallax: a rotation-only (tripod) solve sees every point from one place, no 3D.
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


def build_mesh(solve_dir: str | Path, max_images: int = 40, max_image_size: int = 1280, log=print,
               reuse_stereo: bool = False) -> dict:
    """reuse_stereo: keep an earlier run's fused stereo points and redo only the surface."""
    t0 = time.time()
    solve_dir = Path(solve_dir)
    sel = solve_dir / "selected"
    trk = CameraTrack.load(sel / "cameras.json")
    shot = json.loads((solve_dir / "shot.json").read_text())
    if trk.extra.get("rotation_only"):
        raise RuntimeError("the camera only rotates (tripod): every point is seen from one place, no 3D to mesh")
    if not (sel / "points.ply").exists():
        raise RuntimeError("the solve has no 3D points")
    X = read_ply_xyz(sel / "points.ply")
    v = np.nonzero(trk.valid)[0]
    frames = v[np.unique(np.round(np.linspace(0, len(v) - 1, min(max_images, len(v)))).astype(int))]

    work = solve_dir / "mesh"
    if reuse_stereo and (work / "dense" / "fused.ply").exists():
        return _surface_only(solve_dir, work, sel, log, t0)
    if work.exists():
        shutil.rmtree(work)
    (work / "images").mkdir(parents=True)
    pw, ph = shot["proxy"]["width"], shot["proxy"]["height"]
    S = np.diag([pw / trk.width, ph / trk.height, 1.0])
    names, masked = [], []
    for i in frames:
        img = cv2.imread(str(solve_dir / "proxy" / f"{i:06d}.jpg"))
        m = solve_dir / "masks" / f"{i:06d}.png"
        if m.exists():
            mk = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            mk = cv2.resize(mk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            img[mk] = 0  # moving things: black, textureless, so stereo finds no surface there
            masked.append(float(mk.mean()))
        name = f"{i:06d}.jpg"
        cv2.imwrite(str(work / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names.append(name)
    np.savez(work / "mesh_input.npz", names=np.array(names), K=np.einsum("ij,njk->nik", S, trk.K[frames]),
             dist=trk.dist[frames], R=trk.R[frames], t=trk.t[frames], X=X)
    log(f"[mesh] multi-view stereo on {len(frames)} frames ({pw}x{ph}), "
        f"{100 * np.mean(masked) if masked else 0:.0f} % masked on average")
    res = run_backend("colmap", "mesh", solve_dir, work, {"width": pw, "height": ph, "max_image_size": max_image_size})
    surf = run_backend("da3", "mesh", solve_dir, work, {})
    res["stats"].setdefault("fused_points", surf["stats"].get("fused_points"))
    log(f"[mesh] surface: {surf['stats']['vertices']:,} vertices, {surf['stats']['triangles']:,} triangles")
    shutil.copyfile(work / "mesh.ply", sel / "mesh.ply")
    shutil.copyfile(work / "dense" / "fused.ply", sel / "dense_points.ply")
    out = {"mesh": str(sel / "mesh.ply"), "dense_points": str(sel / "dense_points.ply"),
           "frames": len(frames), "fused_points": res["stats"].get("fused_points"),
           "seconds": round(time.time() - t0, 1)}
    log(f"[mesh] {out['fused_points']} fused points -> {out['mesh']} in {out['seconds']} s")
    return out


def _surface_only(solve_dir: Path, work: Path, sel: Path, log, t0: float) -> dict:
    surf = run_backend("da3", "mesh", solve_dir, work, {})
    log(f"[mesh] surface: {surf['stats']['vertices']:,} vertices, {surf['stats']['triangles']:,} triangles")
    shutil.copyfile(work / "mesh.ply", sel / "mesh.ply")
    shutil.copyfile(work / "dense" / "fused.ply", sel / "dense_points.ply")
    return {"mesh": str(sel / "mesh.ply"), "dense_points": str(sel / "dense_points.ply"),
            "fused_points": surf["stats"].get("fused_points"), "seconds": round(time.time() - t0, 1)}
