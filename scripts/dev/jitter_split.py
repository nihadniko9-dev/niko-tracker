"""Where does a candidate's path jitter come from, and how much of it can the image show?

usage: python scripts/dev/jitter_split.py <run_dir> <shot> [candidate ...]
Splits the selector's jitter (niko.pipeline.select: rotation * f + translation * f / depth) into
rotation, sideways translation and translation along the view axis (camera frame). Sideways moves
shift the image by f * d / depth, but a move along the view axis only scales the image about its
centre: r * d / depth for a point r px from the centre, i.e. rms_r * d / depth over a uniformly
covered frame (rms_r = sqrt((W^2 + H^2) / 12)). With a long lens f is several times rms_r, so the
current formula overstates axial jitter by f / rms_r (5333 / 636 = 8.4x at 100 mm on 1080p).
"img jit": what a composite would show - held-out SIFT points triangulated with the candidate,
projected into every frame, rms second difference of their image positions (px).
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko import metrics
from niko.camio import CameraTrack
from niko.refine import RefineOptions
from niko.tracks import split_sift
from niko.triangulate import triangulate_tracks


import os
PER_FRAME = bool(os.environ.get("PER_FRAME"))


def image_jitter(trk: CameraTrack, held: dict, max_points: int = 2000) -> float:
    vis = held["vis"] & trk.valid[:, None]
    X, good = triangulate_tracks(trk, held["xy"].astype(float), vis)
    X = X[good]
    X = X[np.random.default_rng(0).permutation(len(X))[:max_points]]
    frames = np.nonzero(trk.valid)[0]  # consecutive valid frames, like metrics.jitter
    def uv(t):
        c = np.einsum("ij,nj->ni", trk.R[t], X) + trk.t[t]
        return trk.K[t, 0, 0] * c[:, :2] / c[:, 2:3], c[:, 2] > 0
    acc, per = [], []
    for i in range(1, len(frames) - 1):
        (a, za), (b, zb), (c, zc) = uv(frames[i - 1]), uv(frames[i]), uv(frames[i + 1])
        m = za & zb & zc
        acc.append(np.linalg.norm(a[m] - 2 * b[m] + c[m], axis=1))
        per.append(float(np.sqrt(np.mean(acc[-1] ** 2))) if m.any() else np.nan)
    if not acc:
        return float("nan")
    if PER_FRAME:
        print("   per-frame:", np.round(per, 3).tolist())
    a = np.concatenate(acc)
    return float(np.sqrt(np.mean(a ** 2)))


def split(trk: CameraTrack, depth: float) -> dict:
    ok = trk.valid
    C, Rc2w = trk.centers[ok], trk.R_c2w[ok]
    f = float(np.median(trk.K[ok, 0, 0]))
    acc = C[2:] - 2.0 * C[1:-1] + C[:-2]
    local = np.einsum("nji,nj->ni", Rc2w[1:-1], acc)  # world -> camera frame of the middle frame
    lat = np.linalg.norm(local[:, :2], axis=1) / depth
    ax = np.abs(local[:, 2]) / depth
    j = metrics.jitter(C, Rc2w, scale=depth)
    rms_r = np.hypot(trk.width, trk.height) / np.sqrt(12.0)
    rot_px = f * np.radians(j["rot_deg_rms"])
    lat_px = f * float(np.sqrt(np.mean(lat ** 2)))
    ax_px_now = f * float(np.sqrt(np.mean(ax ** 2)))
    ax_px_img = rms_r * float(np.sqrt(np.mean(ax ** 2)))
    return {"f": f, "rot_px": rot_px, "lateral_px": lat_px, "axial_px_as_now": ax_px_now,
            "axial_px_in_image": ax_px_img,
            "jitter_now": rot_px + f * j["trans_pct_rms"] / 100.0,
            "jitter_image": rot_px + float(np.hypot(lat_px, ax_px_img))}


def main():
    run, shot = Path(sys.argv[1]), sys.argv[2]
    d = run / shot
    names = sys.argv[3:] or sorted(p.name for p in (d / "candidates").iterdir())
    scores = json.loads((d / "solve.json").read_text()).get("select", {}).get("scores", {})
    held = split_sift(dict(np.load(d / "sift/sift_tracks.npz")), RefineOptions().max_sift)[1]
    spec = json.loads((d / "shot.json").read_text())
    gt_path = Path(spec.get("source", "")).parent / "gt/cameras.json"
    if gt_path.exists():
        print(f"{'GT':28s}{'':>55}{image_jitter(CameraTrack.load(gt_path), held):>9.3f}")
    print(f"{'candidate':28s}{'f px':>8}{'rot':>7}{'side':>7}{'axial':>7}{'ax img':>8}{'jit now':>9}{'jit img':>9}"
          f"{'img jit':>9}")
    for name in names:
        p = d / "candidates" / name / "cameras.json"
        if not p.exists():
            continue
        trk = CameraTrack.load(p)
        depth = (scores.get(name) or {}).get("median_depth")  # the selector's own, from held-out points
        if not depth:
            print(f"{name:28s} no score")
            continue
        s = split(trk, abs(depth))
        print(f"{name:28s}{s['f']:>8.0f}{s['rot_px']:>7.3f}{s['lateral_px']:>7.3f}{s['axial_px_as_now']:>7.3f}"
              f"{s['axial_px_in_image']:>8.3f}{s['jitter_now']:>9.3f}{s['jitter_image']:>9.3f}"
              f"{image_jitter(trk, held):>9.3f}", flush=True)


if __name__ == "__main__":
    main()
