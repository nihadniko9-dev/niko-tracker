"""After Effects export: a .jsx that rebuilds the solve in a comp (File > Scripts > Run Script File).

Conventions measured in After Effects 26.5 (scripts/dev/ae_probe.jsx, 2026-09-29):
- 3D axes: X right, Y down, Z away from the viewer - the same axes as an OpenCV camera.
- Orientation (x, y, z) in degrees is the matrix Rx(x) Ry(y) Rz(z) (standard right-handed
  rotations); its columns are the layer's local axes in world space. Max error 5e-15.
- A camera with auto-orient off looks along its local +Z with local +Y down and projects
  u = W/2 + zoom * x/z, v = H/2 + zoom * y/z (comp pixels, top-left origin): an OpenCV pinhole
  with f = zoom and the principal point at the comp centre. Matched to 1e-6 px.
So the AE camera matrix is A @ R_c2w and its position s * A @ (C - origin), where A turns the
solve's world so that up is AE's -Y (ground plane normal, else the first camera's up) and s puts
the scene at a comfortable pixel scale (median point depth = zoom).

Built: comp (clip size, fps, duration), footage, "Niko camera" (position, orientation, zoom
keyed on every frame), 3D nulls on well-tracked points, a guide ground grid lying on y = 0.
Lens distortion is not modelled by an AE camera: solves with k1/k2 get a note in the script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camio import CameraTrack

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mxf", ".avi", ".mkv", ".webm", ".mts", ".m2ts"}


def wsl_windows_path():
    """Inside WSL: a function mapping Linux paths to the Windows paths After Effects needs."""
    import shutil
    import subprocess

    if not shutil.which("wslpath"):
        return None

    def conv(p) -> str:
        r = subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else str(p)
    return conv


def euler_xyz(M: np.ndarray) -> np.ndarray:
    """Degrees (x, y, z) with M = Rx(x) Ry(y) Rz(z) (After Effects orientation)."""
    y = np.arcsin(np.clip(M[0, 2], -1.0, 1.0))
    if abs(M[0, 2]) < 1 - 1e-9:
        x = np.arctan2(-M[1, 2], M[2, 2])
        z = np.arctan2(-M[0, 1], M[0, 0])
    else:  # gimbal lock: put everything in x
        x = np.arctan2(M[2, 1], M[1, 1])
        z = 0.0
    return np.degrees([x, y, z])


def rot_xyz(e_deg) -> np.ndarray:
    x, y, z = np.radians(e_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _unwrap(e: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(e), axis=0))


def world_alignment(trk: CameraTrack, X: np.ndarray | None, rng) -> tuple[np.ndarray, np.ndarray, str]:
    """(A, origin, how): rotation taking solve-world vectors to AE-world and the solve-world point
    that becomes AE's origin. Up -> -Y, the first camera's heading -> +Z."""
    from .locktest import fit_ground

    v = np.nonzero(trk.valid)[0]
    Rc = trk.R_c2w[v]
    up = -Rc[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    origin, how = trk.centers[v].mean(0), "cameras"
    if X is not None and len(X) >= 50:
        g = fit_ground(X, up, rng)
        if g is not None:
            n, d, inl = g
            up = n
            c = np.median(X[inl], 0)
            origin, how = c - (c @ n + d) * n, "ground plane"
    fwd = Rc[0][:, 2] - (Rc[0][:, 2] @ up) * up  # first camera's view direction, flattened
    if np.linalg.norm(fwd) < 1e-6:
        fwd = Rc[0][:, 0] - (Rc[0][:, 0] @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(-up, fwd)  # AE: X = Y x Z with Y = -up (down), Z = fwd
    A = np.stack([right, -up, fwd])  # rows: AE axes expressed in the solve world
    return A, origin, how


def _pick_points(trk: CameraTrack, X: np.ndarray, vis: np.ndarray | None, k: int) -> np.ndarray:
    """Up to k points spread over the middle frame, long tracks first."""
    from .geometry import project

    t = np.nonzero(trk.valid)[0]
    t = t[len(t) // 2]
    uv, z = project(trk.K[t], trk.R[t], trk.t[t], X)
    inside = (z > 0) & (uv[:, 0] > 0.05 * trk.width) & (uv[:, 0] < 0.95 * trk.width) \
        & (uv[:, 1] > 0.05 * trk.height) & (uv[:, 1] < 0.95 * trk.height)
    length = vis.sum(0) if vis is not None else np.zeros(len(X))
    order = np.argsort(-length)
    cells, pick = set(), []
    for i in order:
        if not inside[i]:
            continue
        c = (int(uv[i, 0] * 6 / trk.width), int(uv[i, 1] * 4 / trk.height))
        if c in cells:
            continue
        cells.add(c)
        pick.append(i)
        if len(pick) >= k:
            break
    return np.array(pick, int)


JSX = r'''// Niko Tracker - After Effects import
// Niko Tracker Engine - author: Nihad Jihad ("Niko"). Generated file.
// Run in After Effects: File > Scripts > Run Script File... and pick this file.
// __NOTE__
(function () {
    var D = __DATA__;
    __PROBE_GUARD__
    app.beginUndoGroup("Niko Tracker import");
    var fps = D.fps, n = D.frames.length;
    var comp = app.project.items.addComp(D.name, D.width, D.height, 1, n / fps, fps);
    comp.displayStartFrame = D.frame_start;
    var footage = null;
    try {
        var io = new ImportOptions(new File(D.footage));
        if (D.sequence) { io.sequence = true; io.forceAlphabetical = true; }
        footage = app.project.importFile(io);
        if (D.sequence) { footage.mainSource.conformFrameRate = fps; }
        var fl = comp.layers.add(footage);
        fl.startTime = 0;
    } catch (e) {
        if (!D.probe) alert("Niko Tracker: footage not found at\n" + D.footage + "\nThe camera is built anyway; add the footage yourself.");
    }
    var grid = comp.layers.addSolid([0.3, 0.8, 1.0], "Niko ground (guide)", D.grid_size, D.grid_size, 1);
    grid.threeDLayer = true;
    grid.guideLayer = true;
    grid.transform.orientation.setValue([90, 0, 0]);
    grid.transform.position.setValue([0, 0, 0]);
    grid.transform.opacity.setValue(35);
    try {
        var g = grid.Effects.addProperty("ADBE Grid");
        g.property("ADBE Grid-0002").setValue([D.grid_size / 2 + D.grid_size / 10, D.grid_size / 2]);
    } catch (e2) {}
    for (var k = 0; k < D.nulls.length; k++) {
        var nl = comp.layers.addNull();
        nl.name = D.nulls[k].name;
        nl.threeDLayer = true;
        nl.transform.anchorPoint.setValue([0, 0, 0]);
        nl.transform.position.setValue(D.nulls[k].position);
        nl.transform.scale.setValue([D.null_scale, D.null_scale, D.null_scale]);
    }
    var cam = comp.layers.addCamera("Niko camera", [D.width / 2, D.height / 2]);
    cam.autoOrient = AutoOrientType.NO_AUTO_ORIENT;
    var times = [], pos = [], ori = [], zoom = [];
    for (var i = 0; i < n; i++) {
        var f = D.frames[i];
        if (!f) continue;
        times.push(i / fps); pos.push(f[0]); ori.push(f[1]); zoom.push(f[2]);
    }
    var tr = cam.transform;
    tr.position.setValuesAtTimes(times, pos);
    tr.orientation.setValuesAtTimes(times, ori);
    cam.property("ADBE Camera Options Group").property("ADBE Camera Zoom").setValuesAtTimes(times, zoom);
    for (var j = 1; j <= tr.position.numKeys; j++) {
        tr.position.setInterpolationTypeAtKey(j, KeyframeInterpolationType.LINEAR);
    }
    __PROBE__
    comp.openInViewer();
    app.endUndoGroup();
})();
'''

# the probe closes the project unsaved and quits After Effects: never inside someone's work. If AE is
# already open with a saved or edited project, it leaves everything untouched and only reports that.
PROBE_GUARD = r'''if (D.probe && (app.project.file !== null || app.project.dirty || app.project.numItems > 0)) {
        var G = new File(D.probe.out); G.encoding = "UTF-8"; G.open("w");
        G.write('{"error":"After Effects already has a project open; nothing was changed"}'); G.close();
        return;
    }'''

PROBE = r'''
    // verification only: comp positions of the nulls at a few frames, written next to this file
    var host = comp.layers.addNull(); host.name = "probe";
    var P = D.probe, out = [];
    for (var a = 0; a < D.nulls.length; a++) {
        var fx = host.Effects.addProperty("ADBE Point Control");
        fx.property(1).expression = 'var L = thisComp.layer("' + D.nulls[a].name + '"); L.toComp(L.transform.anchorPoint)';
    }
    for (var b = 0; b < P.frames.length; b++) {
        var row = [];
        for (var a2 = 0; a2 < D.nulls.length; a2++) {
            var v = host.Effects.property(a2 + 1).property(1).valueAtTime(P.frames[b] / fps, false);
            row.push("[" + v[0] + "," + v[1] + "]");
        }
        out.push("[" + row.join(",") + "]");
    }
    var F = new File(P.out); F.encoding = "UTF-8"; F.open("w");
    F.write('{"frames":[' + P.frames.join(",") + '],"footage_ok":' + (footage ? "true" : "false") +
            ',"uv":[' + out.join(",") + ']}'); F.close();
    app.endUndoGroup();
    app.project.close(CloseOptions.DO_NOT_SAVE_CHANGES);
    app.quit();
    return;
'''


def undistort_maps(K: np.ndarray, dist: np.ndarray, W: int, H: int):
    """cv2.remap maps (source x, y per output pixel) that remove the distortion, keeping K.
    K is corner-origin (ours); OpenCV's maps use integer pixel centres, hence the half pixel."""
    import cv2

    Kc = np.asarray(K, np.float64).copy()
    Kc[0, 2] -= 0.5
    Kc[1, 2] -= 0.5
    return cv2.initUndistortRectifyMap(Kc, np.asarray(dist, np.float64), None, Kc, (W, H), cv2.CV_32FC1)


def write_undistorted(solve_dir: Path, trk: CameraTrack, shot: dict, log=print) -> Path:
    """selected/undistorted/000000.jpg ...: the frames with the solve's lens distortion removed, same
    size and same K, so a pinhole camera (After Effects) matches them exactly. Our K is corner-origin
    (pixel centre at +0.5), OpenCV's maps use integer pixel centres: shifted by half a pixel here.
    Frames the solve did not register are copied as they are. Returns the first file."""
    import cv2

    out = solve_dir / "selected" / "undistorted"
    first = out / "000000.jpg"
    n = trk.n_frames
    if out.exists() and len(list(out.glob("*.jpg"))) == n:
        return first
    out.mkdir(parents=True, exist_ok=True)
    ext = shot["frame_format"]
    maps = {}
    for i in range(n):
        img = cv2.imread(str(solve_dir / "frames" / f"{i:06d}.{ext}"))
        if trk.valid[i]:
            key = tuple(np.round(np.r_[trk.K[i].ravel(), trk.dist[i]], 6))
            if key not in maps:
                maps = {key: undistort_maps(trk.K[i], trk.dist[i], trk.width, trk.height)}
            mx, my = maps[key]
            img = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(out / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    log(f"[ae] {n} undistorted frames -> {out}")
    return first


def export_after_effects(solve_dir: str | Path, out: str | Path | None = None, n_nulls: int = 24,
                         windows_path=None, probe: dict | None = None, seed: int = 0, log=print,
                         undistort: bool = True) -> dict:
    """Write <solve_dir>/selected/niko_after_effects.jsx. `windows_path` maps a Linux path to the
    path After Effects sees (WSL: wslpath -w); `probe` = {"frames": [...], "out": path} adds the
    verification block (used by scripts/dev/ae_check.py only)."""
    from .locktest import _held_out_tracks
    from .pipeline.select import is_rotation_only
    from .triangulate import triangulate_tracks

    solve_dir = Path(solve_dir)
    sel = solve_dir / "selected"
    trk = CameraTrack.load(sel / "cameras.json")
    shot = json.loads((solve_dir / "shot.json").read_text())
    rng = np.random.default_rng(seed)
    v = np.nonzero(trk.valid)[0]
    rot_only = is_rotation_only(trk)

    xy, vis, _ = _held_out_tracks(solve_dir, trk)
    vis = vis & trk.valid[:, None]
    if rot_only:
        from .locktest import _directions
        X, ok = _directions(trk, xy, vis, 1.0)
    else:
        X, ok = triangulate_tracks(trk, xy, vis)
    ok &= vis.sum(0) >= 3
    X, vis = X[ok], vis[:, ok]

    A, origin, how = world_alignment(trk, None if rot_only else X, rng)
    f_med = float(np.median(trk.K[v, 0, 0]))
    if rot_only:
        s, origin = 1.0, trk.centers[v].mean(0)
        dist_px = 3.0 * f_med  # nulls on a sphere around the camera
        Xae = (X - origin) @ A.T * dist_px
    else:
        depth = np.median([np.median((X @ trk.R[t].T + trk.t[t])[:, 2]) for t in v[:: max(1, len(v) // 10)]])
        s = f_med / float(depth)
        Xae = (X - origin) @ A.T * s
    C = (trk.centers - origin) @ A.T * s
    M = np.einsum("ij,njk->nik", A, trk.R_c2w)
    E = np.full((trk.n_frames, 3), np.nan)
    E[v] = _unwrap(np.array([euler_xyz(M[t]) for t in v]))
    frames = []
    for t in range(trk.n_frames):
        if not trk.valid[t]:
            frames.append(None)
            continue
        frames.append([np.round(C[t], 4).tolist(), np.round(E[t], 5).tolist(), round(float(trk.K[t, 0, 0]), 4)])

    pick = _pick_points(trk, X, vis, n_nulls)
    nulls = [{"name": f"Track {i + 1:02d}", "position": np.round(Xae[j], 4).tolist()} for i, j in enumerate(pick)]
    spread = float(np.percentile(np.linalg.norm(Xae - np.median(Xae, 0), axis=1), 75)) if len(Xae) else 1000.0
    notes = []
    cx, cy = np.median(trk.K[v, 0, 2]), np.median(trk.K[v, 1, 2])
    if abs(cx - trk.width / 2) > 0.5 or abs(cy - trk.height / 2) > 0.5:
        notes.append(f"principal point ({cx:.1f}, {cy:.1f}) is not the frame centre; an AE camera cannot shift it")
    k1, k2 = float(np.median(trk.dist[v, 0])), float(np.median(trk.dist[v, 1]))
    r_px = float(np.hypot(trk.width, trk.height)) / 2  # frame corner
    r = r_px / f_med
    corner_px = abs(k1 * r ** 2 + k2 * r ** 4) * r_px
    undistorted = None
    if corner_px > 0.5 * max(1.0, trk.width / 1920.0):  # HD pixels
        if undistort:
            undistorted = write_undistorted(solve_dir, trk, shot, log)
            notes.append(f"footage undistorted for the AE pinhole camera (k1 {k1:.4f}, k2 {k2:.4f}: "
                         f"{corner_px:.1f} px at the frame corners): {undistorted.parent.name}/")
        else:
            notes.append(f"the solve has lens distortion (k1 {k1:.4f}, k2 {k2:.4f}: {corner_px:.1f} px at the "
                         "frame corners); the AE camera is a pinhole, so undistort the footage or expect drift "
                         "towards the frame edges")
    if abs(np.median(trk.K[v, 0, 0]) - np.median(trk.K[v, 1, 1])) > 0.5:
        notes.append("fx and fy differ (non-square pixels); zoom uses fx")

    src = Path(shot.get("source", ""))
    if undistorted is not None:
        footage, sequence = undistorted, True
    elif src.suffix.lower() in VIDEO_EXT and src.exists():
        footage, sequence = src, False
    else:
        footage, sequence = solve_dir / "frames" / f"000000.{shot['frame_format']}", True
    wp = windows_path or wsl_windows_path() or (lambda p: str(p))
    data = {"name": f"Niko {trk.name or solve_dir.name}", "width": trk.width, "height": trk.height,
            "fps": trk.fps, "frame_start": trk.frame_start, "frames": frames,
            "footage": wp(footage).replace("\\", "/"), "sequence": sequence, "nulls": nulls,
            "null_scale": 20, "grid_size": int(max(2000, round(4 * spread, -2)))}
    if probe:
        data["probe"] = probe
    jsx = JSX.replace("__DATA__", json.dumps(data)).replace(
        "__NOTE__", "; ".join(notes) if notes else f"world aligned to the {how}; 1 unit = {s:.4g} px")
    jsx = jsx.replace("__PROBE__", PROBE if probe else "").replace("__PROBE_GUARD__", PROBE_GUARD if probe else "")
    out = Path(out) if out else sel / "niko_after_effects.jsx"
    out.write_text(jsx, encoding="utf-8")
    log(f"[ae] {out}: {len(v)} camera keys, {len(nulls)} nulls, aligned to the {how}"
        + (f"; notes: {'; '.join(notes)}" if notes else ""))
    return {"jsx": str(out), "nulls": nulls, "A": A, "origin": origin, "scale": s, "rotation_only": rot_only,
            "X_solve": X[pick], "notes": notes}
