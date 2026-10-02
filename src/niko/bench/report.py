"""`niko bench report`: HTML + CSV from a bench run (runs/<run>/<shot>/solve.json).

Metrics per shot x method (docs: niko.metrics, niko.pipeline.select):
success rate, ATE after Sim(3) (% of scene size), relative drift (RPE over 10 frames),
rotation error (deg), focal error (%), held-out reprojection (px), jitter (px), runtime, peak VRAM.
Targets on synthetic shots: rotation < 0.2 deg, focal < 2 %, ATE < 1 %, every frame solved.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..camio import CameraTrack
from ..metrics import align_sim3_trajectory

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

COLUMNS = ["shot", "tags", "method", "selected", "success_rate", "ate_pct", "ate_max_pct", "rpe10_trans_pct",
           "rpe10_rot_deg", "rot_err_deg_mean", "rot_err_deg_max", "focal_err_pct_mean", "focal_err_pct_max",
           "heldout_reproj_px", "jitter_px", "runtime_s", "peak_vram_mb", "meets_targets", "ate_mode", "error"]


def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


SELECTED = "niko_selected"   # the engine's answer: whatever auto-select picked (raw or refined)
ORACLE = "oracle_best"       # best candidate by ground truth (rotation max) - measures the selector


def collect(run_dir: Path) -> tuple[list[dict], list[str], dict]:
    """One row per shot x candidate (all go to the CSV). HTML tables show the raw methods plus
    the two pseudo-methods SELECTED and ORACLE; refined variants are named '<method>+<mode>'."""
    rows, methods, shots = [], [], {}
    for sj in sorted(run_dir.glob("*/solve.json")):
        rep = json.loads(sj.read_text())
        shot = sj.parent.name
        spec_path = Path(rep["clip"]) / "spec.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        shots[shot] = {"report": rep, "spec": spec, "dir": sj.parent}
        shot_rows = []
        for method, ev in (rep.get("gt_eval") or {}).items():
            if "+" not in method and method not in methods:
                methods.append(method)
            cand = rep.get("candidates", {}).get(method, {})
            row = {"shot": shot, "tags": " ".join(spec.get("tags", [])), "method": method,
                   "selected": rep.get("selected") == method, "runtime_s": cand.get("runtime_s"),
                   "peak_vram_mb": cand.get("peak_vram_mb")}
            for k in COLUMNS:
                if k not in row:
                    row[k] = ev.get(k)
            rows.append(row)
            shot_rows.append(row)
        sel = next((r for r in shot_rows if r["selected"]), None)
        if sel is not None:
            rows.append({**sel, "method": SELECTED, "selected": True, "error": f"picked {sel['method']}",
                         "runtime_s": rep.get("total_seconds")})
        ok = [r for r in shot_rows if r.get("rot_err_deg_max") is not None and (r.get("success_rate") or 0) >= 0.95]
        if ok:
            best = min(ok, key=lambda r: r["rot_err_deg_max"])
            rows.append({**best, "method": ORACLE, "selected": False, "error": f"best is {best['method']}"})
        for stage, st in rep.get("stages", {}).items():  # candidates that crashed
            if stage.startswith("candidate:") and not st["ok"]:
                m = stage.split(":", 1)[1]
                if m not in methods:
                    methods.append(m)
                rows.append({"shot": shot, "tags": " ".join(spec.get("tags", [])), "method": m, "selected": False,
                             "success_rate": 0.0, "meets_targets": False, "error": st.get("error")})
    if any(r["method"] == SELECTED for r in rows):
        methods += [SELECTED, ORACLE]
    return rows, methods, shots


def summarize(rows: list[dict], methods: list[str]) -> list[dict]:
    out = []
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        solved = [x for x in r if (x.get("success_rate") or 0) >= 0.95 and x.get("ate_pct") is not None]

        def med(key):
            vals = [x[key] for x in solved if isinstance(x.get(key), (int, float)) and np.isfinite(x[key])]
            return statistics.median(vals) if vals else None

        out.append({"method": m, "shots": len(r), "solved": len(solved),
                    "targets_met": sum(bool(x.get("meets_targets")) for x in r),
                    "selected": sum(bool(x.get("selected")) for x in r),
                    "ate_pct": med("ate_pct"), "rot_err_deg_max": med("rot_err_deg_max"),
                    "focal_err_pct_max": med("focal_err_pct_max"), "heldout_reproj_px": med("heldout_reproj_px"),
                    "rpe10_trans_pct": med("rpe10_trans_pct"),
                    "runtime_s": sum(x.get("runtime_s") or 0 for x in r),
                    "peak_vram_mb": max((x.get("peak_vram_mb") or 0 for x in r), default=0)})
    return out


def _thumb(shot_dir: Path, width=280) -> str:
    frames = sorted((shot_dir / "proxy").glob("*.jpg")) or sorted((shot_dir / "frames").glob("*.*"))
    if not frames:
        return ""
    img = cv2.imread(str(frames[len(frames) // 2]))
    h = int(img.shape[0] * width / img.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    return f'<img class="thumb" alt="middle frame" src="data:image/jpeg;base64,{base64.b64encode(buf).decode()}">'


def _trajectory_svg(shot_dir: Path, rep: dict, methods: list[str], w=280, h=190) -> str:
    """Top-down (world XY) camera path: GT and every method after Sim(3) alignment."""
    gt_path = Path(rep["clip"]) / "gt" / "cameras.json"
    if not gt_path.exists():
        return ""
    gt = CameraTrack.load(gt_path)
    paths = {"GT": gt.centers[:, :2]}
    for m in methods:
        folder = rep.get("selected") if m == SELECTED else m
        if not folder:
            continue
        p = shot_dir / "candidates" / folder / "cameras.json"
        if not p.exists():
            continue
        est = CameraTrack.load(p)
        ok = est.valid & gt.valid
        if ok.sum() < 3:
            continue
        try:
            sim, _ = align_sim3_trajectory(est.centers[ok], gt.centers[ok], est.R_c2w[ok], gt.R_c2w[ok])
        except ValueError:
            continue
        xy = np.full((gt.n_frames, 2), np.nan)
        xy[ok] = sim.apply(est.centers[ok])[:, :2]
        paths[m] = xy
    allxy = np.concatenate([p[np.isfinite(p[:, 0])] for p in paths.values()])
    lo, hi = allxy.min(0), allxy.max(0)
    span = float(max(hi - lo)) or 1.0
    if float(max(gt.centers[:, :2].max(0) - gt.centers[:, :2].min(0))) < 1e-6:
        return '<p class="muted small">Camera does not move (tripod): no path to draw.</p>'
    pad = 12
    s = min((w - 2 * pad), (h - 2 * pad)) / span
    cx, cy = (lo + hi) / 2

    def pts(xy):
        segs, cur = [], []
        for x, y in xy:
            if not np.isfinite(x):
                if cur:
                    segs.append(cur)
                cur = []
                continue
            cur.append(f"{w / 2 + (x - cx) * s:.1f},{h / 2 - (y - cy) * s:.1f}")
        if cur:
            segs.append(cur)
        return segs

    lines = []
    for name, xy in paths.items():
        cls = "gt" if name == "GT" else f"s{methods.index(name) + 1}"
        for seg in pts(xy):
            lines.append(f'<polyline class="traj {cls}" points="{" ".join(seg)}"><title>{html.escape(name)}</title></polyline>')
        first = pts(xy)
        if first and name == "GT":
            x0, y0 = first[0][0].split(",")
            lines.append(f'<circle class="start" cx="{x0}" cy="{y0}" r="4"><title>GT start</title></circle>')
    scale_m = 10 ** np.floor(np.log10(span / 3)) if span > 0 else 1
    bar = scale_m * s
    lines.append(f'<line class="scalebar" x1="{pad}" y1="{h - 6}" x2="{pad + bar:.1f}" y2="{h - 6}"/>'
                 f'<text class="axis" x="{pad + bar + 4:.1f}" y="{h - 3}">{scale_m:g} m</text>')
    return (f'<svg class="traj-svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="top-down camera path">{"".join(lines)}</svg>')


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in COLUMNS})


CSS = """
:root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --border:rgba(11,11,11,.10); --good:#0ca30c; --good-ink:#006300; --bad:#d03b3b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
  color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
  --border:rgba(255,255,255,.10); --good-ink:#0ca30c;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --border:rgba(255,255,255,.10); --good-ink:#0ca30c;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 10px; }
.muted { color: var(--muted); } .small { font-size: 12px; } .ink2 { color: var(--ink2); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; font-size: 12px; } td:first-child, th:first-child { text-align: left; }
.scroll { overflow-x: auto; }
.pass { color: var(--good-ink); font-weight: 600; } .fail { color: var(--bad); font-weight: 600; }
.swatch { display:inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
.legend span { margin-right: 14px; white-space: nowrap; }
.shots { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
.thumb { width: 100%; height: auto; border-radius: 6px; display: block; }
.traj-svg { width: 100%; height: auto; background: var(--surface); }
.traj { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.traj.gt { stroke: var(--muted); stroke-width: 4; opacity: .55; }
.traj.s1 { stroke: var(--s1); } .traj.s2 { stroke: var(--s2); } .traj.s3 { stroke: var(--s3); } .traj.s4 { stroke: var(--s4); }
.traj.s5 { stroke: var(--s5); } .traj.s6 { stroke: var(--s6); } .traj.s7 { stroke: var(--s7); } .traj.s8 { stroke: var(--s8); }
.start { fill: var(--surface); stroke: var(--ink2); stroke-width: 2; }
.scalebar { stroke: var(--ink2); stroke-width: 2; } .axis { fill: var(--muted); font-size: 10px; }
.kv { display: grid; grid-template-columns: auto auto; gap: 2px 10px; font-size: 12px; }
.kv div:nth-child(odd) { color: var(--ink2); }
"""


def _cell_status(ok):
    return '<span class="pass">✓ met</span>' if ok else '<span class="fail">✗ missed</span>'


def write_html(run_dir: Path, rows, methods, shots, summary, path: Path) -> None:
    meta = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    legend = "".join(f'<span><i class="swatch" style="background:var(--s{i + 1})"></i>{html.escape(m)}</span>'
                     for i, m in enumerate(methods))
    legend = f'<span><i class="swatch" style="background:var(--muted);height:4px"></i>ground truth</span>' + legend

    srows = "".join(
        f"<tr><td><i class='swatch' style='background:var(--s{i + 1})'></i>{html.escape(s['method'])}</td>"
        f"<td>{s['solved']}/{s['shots']}</td><td>{s['targets_met']}/{s['shots']}</td><td>{s['selected']}</td>"
        f"<td>{_fmt(s['ate_pct'])}</td><td>{_fmt(s['rot_err_deg_max'])}</td><td>{_fmt(s['focal_err_pct_max'], 2)}</td>"
        f"<td>{_fmt(s['heldout_reproj_px'])}</td><td>{_fmt(s['rpe10_trans_pct'])}</td>"
        f"<td>{_fmt(s['runtime_s'], 0)}</td><td>{_fmt(s['peak_vram_mb'], 0)}</td></tr>"
        for i, s in enumerate(summary))

    by = {(r["shot"], r["method"]): r for r in rows}
    head = "".join(f"<th colspan='5'>{html.escape(m)}</th>" for m in methods)
    sub = "".join("<th>ATE %</th><th>rot max °</th><th>focal max %</th><th>reproj px</th><th>targets</th>"
                  for _ in methods)
    body = ""
    for shot, info in shots.items():
        tags = html.escape(" ".join(info["spec"].get("tags", [])))
        cells = ""
        for m in methods:
            r = by.get((shot, m))
            if r is None:
                cells += "<td colspan='5' class='muted'>not run</td>"
            elif r.get("error") or r.get("ate_pct") is None:
                sr = r.get("success_rate")
                cells += (f"<td colspan='4' class='fail'>✗ failed{'' if sr is None else f' ({sr * 100:.0f}% frames)'}</td>"
                          f"<td>{_cell_status(False)}</td>")
            else:
                star = " ★" if r.get("selected") else ""
                cells += (f"<td>{_fmt(r['ate_pct'])}{star}</td><td>{_fmt(r['rot_err_deg_max'])}</td>"
                          f"<td>{_fmt(r['focal_err_pct_max'], 2)}</td><td>{_fmt(r.get('heldout_reproj_px'))}</td>"
                          f"<td>{_cell_status(bool(r.get('meets_targets')))}</td>")
        body += f"<tr><td>{html.escape(shot)}<div class='muted small'>{tags}</div></td>{cells}</tr>"

    cards = ""
    for shot, info in shots.items():
        rep, spec = info["report"], info["spec"]
        kv = [("selected", rep.get("selected") or "–"), ("frames", spec.get("n_frames", "–")),
              ("scene size", f"{json.loads((Path(rep['clip']) / 'scene.json').read_text()).get('scene_size', 0):.1f} m"
               if (Path(rep["clip"]) / "scene.json").exists() else "–"),
              ("solve time", f"{rep.get('total_seconds', 0):.0f} s")]
        for key in ("motion_blur", "rolling_shutter", "distortion", "noise", "movers"):
            if spec.get(key):
                kv.append((key.replace("_", " "), html.escape(json.dumps(spec[key]))[:60]))
        cards += (f"<div class='card'><strong>{html.escape(shot)}</strong>"
                  f"<div class='muted small'>{html.escape(' · '.join(spec.get('tags', [])))}</div>"
                  f"{_thumb(info['dir'])}{_trajectory_svg(info['dir'], rep, methods)}"
                  f"<div class='kv'>{''.join(f'<div>{k}</div><div>{v}</div>' for k, v in kv)}</div></div>")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Niko bench report</title>
<style>{CSS}</style></head><body><main>
<h1>Niko Tracker — benchmark report</h1>
<div class="muted">Run <b>{html.escape(run_dir.name)}</b> · set {html.escape(str(meta.get('set', '')))} ·
generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Niko Tracker Engine, author Nihad Jihad (Niko)</div>
<p class="ink2">Targets on synthetic shots: rotation error &lt; 0.2°, focal error &lt; 2 %, ATE &lt; 1 % of scene size,
every frame solved. ATE after Sim(3) alignment; rotation after the best global rotation; reprojection is the
median over held-out CoTracker3 tracks triangulated with each method's own cameras (no ground truth).
Candidates are <b>unrefined</b> in this run (refinement is checkpoint 3). ★ = picked by auto-select.</p>
<h2>Summary per method</h2><div class="card scroll"><table>
<tr><th>method</th><th>solved</th><th>targets met</th><th>auto-selected</th><th>median ATE %</th>
<th>median rot max °</th><th>median focal max %</th><th>median reproj px</th><th>median drift % /10 fr</th>
<th>total runtime s</th><th>max VRAM MB</th></tr>{srows}</table>
<p class="muted small">Medians over solved shots (≥ 95 % of frames). VRAM for COLMAP is the whole-GPU delta seen by nvidia-smi (approximate).</p></div>
<h2>Per shot</h2><div class="card scroll"><table><tr><th rowspan="2">shot</th>{head}</tr><tr>{sub}</tr>{body}</table></div>
<h2>Camera paths (top-down, aligned to ground truth)</h2><div class="legend small">{legend}</div><br>
<div class="shots">{cards}</div>
<p class="muted small">All numbers are also in metrics.csv next to this file.</p>
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")


def make_report(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    rows, methods, shots = collect(run_dir)
    if not rows:
        raise RuntimeError(f"no solve.json with ground-truth evaluation under {run_dir}")
    summary = summarize(rows, methods)
    write_csv(rows, run_dir / "metrics.csv")
    write_html(run_dir, rows, methods, shots, summary, run_dir / "report.html")
    return {"html": str(run_dir / "report.html"), "csv": str(run_dir / "metrics.csv"), "summary": summary}
