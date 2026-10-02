"""Reading a finished solve folder: blender.json (camera values computed by the engine), solve.json,
errors.json (per-frame held-out error) and points.ply."""

import json
import os

import numpy as np

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "short": "<i2", "ushort": "<u2", "int": "<i4", "int32": "<i4",
              "uint": "<u4", "uint32": "<u4"}


def read_ply_xyz(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        raw = fh.read()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    n, props, in_vertex, fmt = 0, [], False, None
    for line in raw[:end].decode("ascii").splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "format":
            fmt = w[1]
        elif w[0] == "element":
            in_vertex = w[1] == "vertex"
            if in_vertex:
                n = int(w[2])
        elif w[0] == "property" and in_vertex:
            props.append((w[2], _PLY_TYPES[w[1]]))
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt}")
    d = np.frombuffer(raw, dtype=np.dtype(props), count=n, offset=end)
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32)
    return xyz[np.all(np.isfinite(xyz), 1)]


def selected_dir(folder: str) -> str:
    """A solve folder (engine layout: <folder>/selected/...) or a flat copy of selected/ (with
    blender.json and solve.json side by side, e.g. reports/test_shots/01)."""
    if os.path.exists(os.path.join(folder, "blender.json")):
        return folder
    return os.path.join(folder, "selected")


REAL_SCALE = "real_scale.json"


def read_real_scale(folder: str) -> dict:
    """What the user set in Blender, {} when nothing was: "factor" (size, multiplies the engine's
    world) and "how"; "ground" when the floor was set from picked points; "adjust", the whole change
    as a 4x4 (row-major) applied in front of the engine's world (files from 0.4.0 have only the
    factor)."""
    try:
        with open(os.path.join(selected_dir(folder), REAL_SCALE), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def user_adjust(folder: str) -> list | None:
    """The user's change in front of the engine's world as a 4x4 (row-major), or None."""
    r = read_real_scale(folder)
    if r.get("adjust"):
        return r["adjust"]
    if "factor" in r:
        k = float(r["factor"])
        return [[k, 0, 0, 0], [0, k, 0, 0], [0, 0, k, 0], [0, 0, 0, 1]]
    return None


def write_real_scale(folder: str, factor: float | None = None, how: str | None = None,
                     adjust: list | None = None, ground: str | None = None) -> None:
    """Update the saved size / ground; fields left None keep their saved value."""
    import time

    data = read_real_scale(folder)
    for key, val in (("factor", factor), ("how", how), ("adjust", adjust), ("ground", ground)):
        if val is not None:
            data[key] = val
    data["set"] = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(selected_dir(folder), REAL_SCALE), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)


def clear_real_scale(folder: str) -> None:
    """Back to the engine's size and ground; the file is renamed to .old, not deleted."""
    path = os.path.join(selected_dir(folder), REAL_SCALE)
    if os.path.exists(path):
        os.replace(path, path + ".old")


class Solve:
    """Everything the add-on shows about one solve."""

    def __init__(self, folder: str):
        self.folder = folder
        sel = selected_dir(folder)
        with open(os.path.join(sel, "blender.json"), encoding="utf-8") as fh:
            self.blender = json.load(fh)
        self.report = {}
        p = os.path.join(folder, "solve.json")
        if not os.path.exists(p):
            p = os.path.join(sel, "solve.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                self.report = json.load(fh)
        self.errors = None
        p = os.path.join(sel, "errors.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                self.errors = json.load(fh)
        p = os.path.join(sel, "points.ply")
        self.points = read_ply_xyz(p) if os.path.exists(p) else np.zeros((0, 3), np.float32)
        self.selected = self.report.get("selected") or self.blender.get("method", "")

    def source_clip(self):
        """The clip the engine solved, as Windows sees it (solve.json 'clip': /mnt/d/... -> D:\\...)."""
        c = self.report.get("clip") or ""
        if c and not os.path.isabs(c):
            # Older CLI runs recorded a relative clip; ingest already saved its canonical source.
            try:
                with open(os.path.join(self.folder, "shot.json"), encoding="utf-8") as fh:
                    c = json.load(fh).get("source") or c
            except (OSError, ValueError):
                pass
        if c.startswith("/mnt/") and len(c) > 6 and c[6] == "/":
            c = f"{c[5].upper()}:\\" + c[7:].replace("/", "\\")
        return os.path.abspath(c) if c and os.path.isfile(c) else None

    def size_text(self) -> str:
        """What one Blender unit means in this scene."""
        real = read_real_scale(self.folder)
        if real.get("how"):
            return f"Real size: {real['how']}"
        units = self.blender.get("units") or {}
        if units.get("kind") in ("metres_gps", "metres_altitude"):
            src = "GPS" if units["kind"] == "metres_gps" else "altitude"
            unc = units.get("uncertainty_pct")
            return f"Metres, from the drone's {src}" + (f" (±{unc:.0f}%)" if unc is not None else "")
        if units.get("kind") == "metres_estimated":
            apart = units.get("agree_pct")
            return ("Size: about metres (estimated" + (f", models {apart:.0f}% apart)" if apart is not None else ")"))
        if self.report.get("metric_scale"):
            return "Size unknown: the depth models disagree. Set it with a known distance or the camera height."
        return "Size unknown: set it with a known distance or the camera height."

    def ground_text(self) -> str:
        """Where the floor (Z = 0) comes from."""
        real = read_real_scale(self.folder)
        if real.get("ground"):
            return f"Ground: set by you ({real['ground']})"
        level = " Level from the drone's gimbal." if self.report.get("true_up") else ""
        if self.blender.get("world_how") == "ground plane":
            return "Ground: found automatically." + level + " Wrong? Pick 3+ floor points and set it."
        return "Ground: not found (no clear floor). Pick 3+ floor points and set it."

    @property
    def average_px(self):
        return (self.report.get("solve_error") or {}).get("average_px")

    def guidance(self):
        """Actionable checks, independent of a deceptively low average pixel error."""
        messages = []
        lc = self.report.get("lens_check") or {}
        if "hardly_turns" in lc.get("reasons", []):
            messages.append("Lens not measured: the camera hardly turns (it only moves), so depth may be off. "
                            "Calibrate this camera once, or enter its lens in Advanced settings.")
        elif lc.get("uncertain"):
            messages.append("Lens uncertain: enter a known lens in Advanced settings, or check object sliding carefully.")
        if self.report.get("track_gaps"):
            messages.append("Tracking broke: inspect both sides of each marked gap before using the camera.")
        done, total = self.frames_solved
        if done < total:
            messages.append(f"Only {done} of {total} frames solved: check the missing parts.")
        err = self.report.get("solve_error") or {}
        fraction = err.get("inlier_fraction")
        if fraction is not None and fraction < 0.9:
            messages.append("Many tracked observations disagree: inspect the lock-test video.")
        if self.average_px is None:
            messages.append("No measured tracking error: inspect the result before using it.")
        elif self.average_px / hd_scale(self.blender.get("width")) >= 1:
            messages.append("Tracking error is high: inspect the worst frame and the lock-test video.")
        failed = [k for k, v in self.report.get("stages", {}).items()
                  if not v.get("ok") and not k.startswith("candidate:")]
        if failed:
            messages.append("A processing step failed. Enable Advanced settings to see the details.")
        return messages

    @property
    def frames_solved(self):
        f = self.blender["frames"]
        return sum(1 for x in f if x["valid"]), len(f)

    @property
    def lens_mm(self):
        lens = [x["lens"] for x in self.blender["frames"] if x["valid"]]
        return (min(lens), max(lens)) if lens else (None, None)

    @property
    def camera_kind(self):
        name = self.selected or ""
        if "tripod_zoom" in name:
            return "Tripod, zooming"
        if "tripod" in name:
            return "Tripod"
        if "ba_zoom" in name:
            return "Moving, zooming"
        return "Moving"

    def frame_error(self, frame: int):
        if not self.errors:
            return None
        i = frame - self.errors.get("frame_start", 1)
        m = self.errors.get("mean_px", [])
        return m[i] if 0 <= i < len(m) else None

    def worst_frame(self):
        if not self.errors:
            return None, None
        best = None
        for i, v in enumerate(self.errors.get("mean_px", [])):
            if v is not None and (best is None or v > best[1]):
                best = (i, v)
        if best is None:
            return None, None
        return self.errors.get("frame_start", 1) + best[0], best[1]


def hd_scale(width) -> float:
    """Pixels of this clip per HD pixel (>= 1): thresholds are set in HD pixels, so a 4K solve with
    the same angular accuracy gets the same rating."""
    return max(1.0, (width or 1920) / 1920.0)


def rating(px, width=1920):
    """(label, icon, rgba) for an average error in the clip's own pixels."""
    if px is None:
        return "No result", "QUESTION", (0.7, 0.7, 0.7, 1.0)
    v = px / hd_scale(width)
    if v < 0.5:
        return "Excellent", "CHECKMARK", (0.45, 0.80, 0.25, 1.0)
    if v < 1.0:
        return "Good", "INFO", (0.95, 0.65, 0.20, 1.0)
    return "Check it", "ERROR", (0.90, 0.30, 0.30, 1.0)


_CACHE: dict = {}


def get(folder: str, refresh: bool = False):
    """Cached Solve for a folder, re-checked on disk at most every 2 s (draw callbacks call this on
    every redraw, and the folder is often on the WSL file system); refresh=True checks now."""
    import time

    if not folder:
        return None
    now = time.monotonic()
    c = _CACHE.get(folder)
    if c is not None and not refresh and now - c[2] < 2.0:
        return c[1]
    try:
        key = os.path.getmtime(os.path.join(selected_dir(folder), "blender.json"))
    except OSError:
        _CACHE[folder] = (None, None, now)
        return None
    # the engine writes solve.json (average error, lens check) and errors.json after blender.json:
    # a solve read in between must be read again once they appear or change
    extra = []
    for name in ("solve.json", os.path.join(selected_dir(folder), "errors.json")):
        try:
            extra.append(os.path.getmtime(os.path.join(folder, name)))
        except OSError:
            extra.append(None)
    key = (key, *extra)
    if c is None or c[0] != key or c[1] is None:
        try:
            _CACHE[folder] = (key, Solve(folder), now)
        except (OSError, ValueError, KeyError):
            _CACHE[folder] = (None, None, now)
            return None
    else:
        _CACHE[folder] = (c[0], c[1], now)
    return _CACHE[folder][1]
