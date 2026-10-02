"""A DJI solve against the drone's own telemetry (niko.telemetry): the camera path against GPS in
metres (scale, residuals), the take-off-relative altitude against the solve's height changes, the
camera orientation against the gimbal attitude, and the levelled world's up against true up.

usage (niko env): python scripts/dev/gps_check.py <solve_dir> [video]
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.export_ae import world_alignment
from niko.plyio import read_ply_xyz
from niko.sim3 import umeyama
from niko.telemetry import enu, read_dji, updates

# camera (OpenCV: x right, y down, z forward) -> DJI body (x forward, y right, z down)
P_CAM_BODY = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
NED_TO_ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)


def quat_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angle(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def main():
    d = Path(sys.argv[1])
    shot = json.loads((d / "shot.json").read_text())
    video = sys.argv[2] if len(sys.argv) > 2 else shot["source"]
    tel = read_dji(video)
    trk = CameraTrack.load(d / "selected" / "cameras.json")
    n = min(trk.n_frames, tel["n"])
    print(f"{d.name}: {tel['model']} {tel['camera']}, {tel['n']} telemetry packets, {trk.n_frames} frames")

    # 1. orientation: gimbal attitude vs the solve's cameras (rotation that best maps one to the other)
    R_g = np.array([NED_TO_ENU @ quat_R(q) @ P_CAM_BODY for q in tel["gimbal_q"][:n]])
    v = np.nonzero(trk.valid[:n])[0]
    U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", R_g[v], trk.R_c2w[v]))
    Ra = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt  # solve world -> ENU
    res = np.array([angle(Ra @ trk.R_c2w[i], R_g[i]) for i in v])
    print(f"  orientation: solve vs gimbal after one fixed rotation: median {np.median(res):.2f} deg, "
          f"90% {np.percentile(res, 90):.2f}, max {res.max():.2f}")
    X = read_ply_xyz(d / "selected" / "points.ply") if (d / "selected" / "points.ply").exists() else None
    A, origin, how = world_alignment(trk, X, np.random.default_rng(0))
    up = -A[1]
    print(f"  levelled up ({how}) vs true up from the gimbal: {np.degrees(np.arccos(np.clip((Ra @ up)[2], -1, 1))):.2f} deg")

    # 2. position: GPS (and altitude) vs the solve's camera centres
    alt = tel["alt"] if tel["alt"] is not None else None
    if tel["lat"] is not None:
        u = updates(tel["lat"][:n])
        u = u[trk.valid[u]]
        E = enu(tel["lat"][u], tel["lon"][u], tel["alt"][u])
        C = trk.centers[u]
        sim = umeyama(C, E, with_scale=True)
        r = np.linalg.norm(sim.apply(C) - E, axis=1)
        ext = np.linalg.norm(np.ptp(E, 0))
        print(f"  GPS: {len(u)} fixes, path extent {ext:.1f} m; best fit {sim.s:.4f} m per solve unit, "
              f"residual median {np.median(r):.2f} m, 90% {np.percentile(r, 90):.2f} m")
        # scale with the rotation fixed by the gimbal (only scale and offset fitted)
        Cr = C @ Ra.T
        a0, b0 = Cr.mean(0), E.mean(0)
        s2 = float(np.sum((Cr - a0) * (E - b0)) / np.sum((Cr - a0) ** 2))
        r2 = np.linalg.norm(s2 * (Cr - a0) + b0 - E, axis=1)
        print(f"  GPS with the gimbal's rotation: {s2:.4f} m per unit, residual median {np.median(r2):.2f} m")
        half = len(u) // 2
        for name, part in (("first half", slice(0, half)), ("second half", slice(half, None))):
            s_p = umeyama(C[part], E[part], with_scale=True).s
            print(f"    {name}: {s_p:.4f} m per unit")
    if tel["rel_alt"] is not None:
        ua = updates(tel["rel_alt"][:n])
        ua = ua[trk.valid[ua]]
        h = tel["rel_alt"][ua]
        z = (trk.centers[ua] @ Ra.T)[:, 2]
        if np.ptp(h) > 1.0:
            s3 = float(np.polyfit(z, h, 1)[0])
            print(f"  altitude: {np.ptp(h):.1f} m change over {len(ua)} readings -> {s3:.4f} m per unit (vertical only)")
        else:
            print(f"  altitude: changes only {np.ptp(h):.1f} m (no vertical scale)")
    metric = json.loads((d / "solve.json").read_text()).get("metric_scale") or {}
    print(f"  depth-model estimate was {metric.get('metres_per_unit')} m per unit (models {metric.get('agree_pct')} % apart)")


if __name__ == "__main__":
    main()
