"""Measure a camera's lens from a finished solve: bundle adjustment with the focal held at each value of
a sweep, then for each value the held-out reprojection error (does the footage prefer it?) and, for
a DJI clip, how well the solved camera rotation matches the gimbal's (the gimbal measures the turn
independently of the lens). Work goes to a scratch folder; the solve itself is not changed.

usage (niko env): python scripts/dev/lens_sweep.py <solve_dir> <scratch_dir> <focal_px> [<focal_px> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.pipeline.select import score_candidate, usable_sift_holdout
from niko.pipeline.solve import load_sift
from niko.refine import RefineOptions, copy_track, refine
from niko.scale import telemetry_scale
from niko.telemetry import read_dji

sys.path.insert(0, str(Path(__file__).parent))
from gps_check import NED_TO_ENU, P_CAM_BODY, angle, quat_R  # noqa: E402


def gimbal_fit(trk, Rg):
    """(median orientation residual after one fixed rotation, solve turn / gimbal turn)."""
    v = np.nonzero(trk.valid[:len(Rg)])[0]
    U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", Rg[v], trk.R_c2w[v]))
    Ra = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    res = np.array([angle(Ra @ trk.R_c2w[i], Rg[i]) for i in v])
    a, b = v[0], v[-1]
    tg = angle(np.eye(3), Rg[a].T @ Rg[b])
    ts = angle(np.eye(3), trk.R_c2w[a].T @ trk.R_c2w[b])
    return float(np.median(res)), float(np.percentile(res, 90)), ts / tg if tg > 1 else float("nan")


def main():
    d, scratch = Path(sys.argv[1]), Path(sys.argv[2])
    focals = [float(x) for x in sys.argv[3:]]
    shot = json.loads((d / "shot.json").read_text())
    rep = json.loads((d / "solve.json").read_text())
    best = rep["selected"]
    start0 = CameraTrack.load(d / "candidates" / best / "cameras.json")
    tracks = dict(np.load(d / "tracks" / "tracks.npz"))
    sift, held = load_sift(d, shot, lambda *a: None)
    held_u = usable_sift_holdout(held)
    tel = read_dji(shot["source"])
    Rg = None
    if tel and tel.get("gimbal_q") is not None:
        Rg = np.array([NED_TO_ENU @ quat_R(q) @ P_CAM_BODY for q in tel["gimbal_q"]])
    dist = "k1k2" if "k1k2" in best else "none"
    f0 = float(np.median(start0.K[start0.valid, 0, 0]))
    print(f"{d.name}: selected {best} (focal {f0:.1f} px), sweep with distortion {dist}")
    print(f"{'focal px':>9} {'score px':>9} {'reproj med':>11} {'inliers':>8} {'gimbal med':>11} {'gimbal 90%':>11} "
          f"{'turn ratio':>11} {'GPS res m':>10} {'GPS halves':>11} {'alt vs GPS':>11}")
    for f in focals:
        start = copy_track(start0)
        start.K[:, 0, 0] = start.K[:, 1, 1] = f
        work = scratch / f"f{int(round(f))}"
        trk, _ = refine(start, tracks, work, RefineOptions(intrinsics="fixed", distortion=dist), sift=sift)
        s = score_candidate(trk, tracks, held_u)
        g = gimbal_fit(trk, Rg) if Rg is not None else (np.nan, np.nan, np.nan)
        ts = telemetry_scale(trk, tel) if tel else None
        gps = (ts or {}).get("gps") or {}
        print(f"{f:9.1f} {s['score_px']:9.4f} {s.get('reproj_median_px', np.nan):11.4f} "
              f"{100 * s.get('inlier_fraction', np.nan):7.2f}% {g[0]:10.3f}° {g[1]:10.3f}° {g[2]:11.4f} "
              f"{gps.get('residual_median_m', np.nan):10.3f} {gps.get('halves_apart_pct') or np.nan:10.2f}% "
              f"{(ts or {}).get('gps_vs_altitude_pct') or np.nan:10.2f}%", flush=True)


if __name__ == "__main__":
    main()
