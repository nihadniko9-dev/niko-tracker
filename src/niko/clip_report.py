"""`niko report`: one self-contained HTML page about solved real clips (reads well on a phone).

Per clip: a lock-test frame, the average tracking error (and in HD pixels for bigger footage)
with its rating, the error of every frame as a bar strip, lens, camera, what was picked, lens
check, keyframe stride, masks, time per stage, scene mesh. Images are embedded (base64), so the
file can be sent on its own.
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
from pathlib import Path

GOOD, WARN, BAD = "#3f9f3f", "#d08a1e", "#cc3b3b"


def _thumb(solve_dir: Path, width: int = 720) -> str:
    """A frame from the middle of the lock test (or of the footage), as a data URI."""
    sel = solve_dir / "selected"
    src = sel / "locktest.mp4"
    if src.exists():
        n = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(src)],
                               capture_output=True, text=True).stdout.strip() or 1)
        cmd = ["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"select=eq(n\\,{n // 2}),scale={width}:-2",
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"]
    else:
        frames = sorted((solve_dir / "frames").glob("*.jpg"))
        if not frames:
            return ""
        cmd = ["ffmpeg", "-v", "error", "-i", str(frames[len(frames) // 2]), "-vf", f"scale={width}:-2",
               "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"]
    data = subprocess.run(cmd, capture_output=True).stdout
    return "data:image/jpeg;base64," + base64.b64encode(data).decode() if data else ""


def _rating(px_hd):
    if px_hd is None:
        return "No result", "#888"
    if px_hd < 0.5:
        return "Excellent", GOOD
    if px_hd < 1.0:
        return "Good", WARN
    return "Check it", BAD


def _strip(errors: dict | None, k: float, width: int = 680, height: int = 56) -> str:
    """Per-frame error as an SVG bar strip (HD pixels), dashed line at 0.5 px."""
    if not errors:
        return ""
    vals = errors.get("mean_px", [])
    n = len(vals)
    if not n:
        return ""
    top = 1.5
    bw = width / n
    bars = []
    for i, v in enumerate(vals):
        if v is None:
            continue
        hd = v / k
        h = min(height, hd / top * height)
        col = GOOD if hd < 0.5 else (WARN if hd < 1.0 else BAD)
        bars.append(f'<rect x="{i * bw:.2f}" y="{height - h:.2f}" width="{max(bw * 0.8, 0.6):.2f}" '
                    f'height="{h:.2f}" fill="{col}"><title>frame {errors.get("frame_start", 1) + i}: '
                    f'{v:.2f} px</title></rect>')
    y = height - 0.5 / top * height
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
            f'aria-label="error per frame" preserveAspectRatio="none">{"".join(bars)}'
            f'<line x1="0" x2="{width}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/></svg>')


def _runtime(solve_dir: Path, rep: dict) -> list[tuple[str, float]]:
    out = []
    for name, rel in (("Masks", "jobs/masks/result.json"), ("Tracks", "tracks/result.json")):
        f = solve_dir / rel
        if f.exists():
            out.append((name, float(json.loads(f.read_text()).get("runtime_s") or 0)))
    cams = 0.0
    for m in ("colmap_global", "colmap_incremental", "megasam", "da3"):
        f = solve_dir / "candidates" / m / "result.json"
        if f.exists():
            cams += float(json.loads(f.read_text()).get("runtime_s") or 0)
    out.append(("Camera candidates", cams))
    out.append(("Refine + pick", float(rep["stages"].get("refine", {}).get("seconds", 0))
                + float(rep["stages"].get("select", {}).get("seconds", 0))))
    return out


def _mesh_info(solve_dir: Path) -> str:
    f = solve_dir / "mesh" / "result.json"
    if not f.exists():
        return "not built"
    r = json.loads(f.read_text())
    if not r.get("ok"):
        err = (r.get("error") or "").strip().splitlines()
        return "not possible: " + html.escape((err[-1] if err else "failed").split(": ", 1)[-1][:160])
    return f"{r['stats'].get('fused_points', 0):,} stereo points"


def clip_card(solve_dir: Path) -> str:
    from .camio import CameraTrack

    rep = json.loads((solve_dir / "solve.json").read_text())
    trk = CameraTrack.load(solve_dir / "selected" / "cameras.json")
    errors_f = solve_dir / "selected" / "errors.json"
    errors = json.loads(errors_f.read_text()) if errors_f.exists() else None
    se = rep.get("solve_error") or {}
    avg = se.get("average_px")
    k = max(1.0, trk.width / 1920.0)
    label, col = _rating(avg / k if avg is not None else None)
    f_px = float(sorted(trk.K[trk.valid, 0, 0])[int(trk.valid.sum()) // 2])
    mm = f_px / trk.width * 36.0
    lc = rep.get("lens_check") or {}
    sp = lc.get("spread") or {}
    if lc.get("uncertain") and "equally_good_fits" in (lc.get("reasons") or []):
        lo, hi = (v / trk.width * 36.0 for v in sp["focal_px"])
        lens_note = f"uncertain: solves that fit just as well range {lo:.0f} - {hi:.0f} mm; give the focal length"
    elif lc.get("uncertain"):
        lens_note = "uncertain: the depth models disagree; give the focal length"
    elif sp:
        lens_note = f"measured: equally good solves agree within {sp['spread_pct']:.1f} %"
    else:
        lens_note = "agreed by the depth models"
    masks = solve_dir / "jobs/masks/result.json"
    mstats = json.loads(masks.read_text()).get("stats", {}) if masks.exists() else {}
    objs = ", ".join(f"{n} {p}" for p, n in (mstats.get("objects_per_prompt") or {}).items() if n)
    worst = se.get("worst_frame") or {}
    rt = _runtime(solve_dir, rep)
    rows = [
        ("Frames solved", f"{int(trk.valid.sum())} / {trk.n_frames}"),
        ("Footage", f"{trk.width}x{trk.height} at {trk.fps:.2f} fps"),
        ("Lens", f"{mm:.0f} mm full-frame equivalent ({f_px:.0f} px), {lens_note}"),
        ("Camera picked", html.escape(rep.get("selected") or "")),
        ("Worst frame", f"{worst.get('frame')} ({worst.get('mean_px', 0):.2f} px)" if worst else "-"),
        ("Tracking breaks", ", ".join(f"frames {g['from_frame']}-{g['to_frame']}" for g in rep["track_gaps"])
         + " (fast move or blur: the turn across can be off by a few degrees)" if rep.get("track_gaps") else
         "none" if "track_gaps" in rep else "-"),
        ("Checked on", f"{100 * se.get('inlier_fraction', 0):.1f} % of held-out {se.get('source', '')} "
                       f"observations within 3 px"),
        ("Keyframes", f"every {rep.get('keyframe_stride') or 1}"),
        ("Ignored", f"{100 * mstats.get('mean_excluded_fraction', 0):.0f} % of each frame ({objs})"
         if mstats else "-"),
        ("Scene mesh", _mesh_info(solve_dir)),
        ("Time", ", ".join(f"{n} {s / 60:.1f} min" for n, s in rt)),
    ]
    img = _thumb(solve_dir)
    hd = f" <span class='muted'>({avg / k:.2f} px in HD)</span>" if avg is not None and k > 1 else ""
    return f"""
<section class="card">
  <h2>{html.escape(Path(rep.get('clip', solve_dir.name)).name)}</h2>
  {f'<img src="{img}" alt="lock test frame">' if img else ''}
  <div class="big"><span class="num" style="color:{col}">{avg:.2f} px</span>
    <span class="badge" style="border-color:{col};color:{col}">{label}</span>{hd}</div>
  <div class="muted small">average tracking error, on tracks the solver never saw</div>
  <div class="strip">{_strip(errors, k)}<div class="muted small">error per frame (green under 0.5 px,
    amber under 1 px, red above; dashed line 0.5 px)</div></div>
  <table>{''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in rows)}</table>
</section>"""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Niko Tracker clips</title>
<style>
:root {{ --bg:#f6f5f2; --card:#fff; --ink:#1d1d1b; --muted:#6b6a66; --line:#e3e1db; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1b1b1a; --card:#242423; --ink:#ecebe7; --muted:#a3a19b; --line:#3a3a38; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:760px; margin:0 auto; padding:16px; }}
h1 {{ font-size:20px; font-weight:600; margin:8px 0 2px; }}
h2 {{ font-size:17px; font-weight:600; margin:0 0 10px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin:14px 0; }}
img {{ width:100%; border-radius:8px; display:block; margin-bottom:12px; }}
.big {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.num {{ font-size:30px; font-weight:600; }}
.badge {{ border:1px solid; border-radius:999px; padding:1px 10px; font-size:13px; }}
.muted {{ color:var(--muted); }} .small {{ font-size:12.5px; }}
.strip {{ margin:12px 0; }} .ref {{ stroke:var(--muted); stroke-dasharray:4 4; stroke-width:1; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ text-align:left; font-weight:500; color:var(--muted); padding:5px 10px 5px 0; vertical-align:top; width:34%; }}
td {{ padding:5px 0; border-top:1px solid var(--line); }} th {{ border-top:1px solid var(--line); }}
footer {{ color:var(--muted); font-size:12px; margin:18px 0; }}
</style></head><body><main>
<h1>Niko Tracker - {title}</h1>
<div class="muted small">{subtitle}</div>
{cards}
<footer>Niko Tracker Engine - Nihad Jihad. Every number measured by the engine on these clips.</footer>
</main></body></html>"""


def clip_report(solve_dirs, out: str | Path, title: str = "your clips", subtitle: str = "") -> Path:
    cards = "".join(clip_card(Path(d)) for d in solve_dirs)
    out = Path(out)
    out.write_text(PAGE.format(title=html.escape(title), subtitle=html.escape(subtitle), cards=cards),
                   encoding="utf-8")
    return out
