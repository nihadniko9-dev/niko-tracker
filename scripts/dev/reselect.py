"""Re-run auto-select on a finished benchmark run with each jitter measure and compare the picks
against ground truth (no solving: candidates, tracks and SIFT tracks are read from the run).

usage: python scripts/dev/reselect.py <run_dir> [shot ...] [--all]  (--all: every candidate of every shot)
Prints, per shot, the pick with the selector's "path" jitter (checkpoint 3) and with "screen"
jitter (screen_jitter_px below, swapped into the score) and the GT figures of each pick.

Result 2026-09-29 (PROGRESS.md): "screen" is the truer measure of visible shake but picks worse -
zoom_in goes to a dolly solution 72 % off, telephoto 1 / 3 instead of 2 / 3. The path measure's
double counting of a turn and a sideways move that cancel on screen is what marks those paths as
implausible, so the selector keeps it. An earlier variant on the held-out points themselves
(their image acceleration) was worse still: near points blew up car_follow (1467 px) and it took
distortion_barrel to a 31 % focal error.
"""

import json
import sys
from pathlib import Path

import numpy as np

from niko.camio import CameraTrack
from niko.pipeline import select as sel
from niko.refine import RefineOptions
from niko.tracks import split_sift

MODES = ("path", "screen")
_score = sel.score_candidate


def screen_jitter_px(trk: CameraTrack, depth: float, spread: float = 0.6) -> float | None:
    """Path jitter as seen on screen: a 3 x 3 grid of virtual points at the median scene depth,
    fixed in the world at each solved frame, projected into it and its two neighbours (consecutive
    solved frames); rms second difference of their image positions (px). Unlike the "path"
    measure, a turn and a sideways move that cancel on screen (a long lens trades one for the
    other almost freely) count as nothing, and a move along the view axis only as the image
    scaling it causes. Virtual points, not the held-out ones: near points and each candidate's own
    triangulation would make the figure depend on the scene instead of the path."""
    frames = np.nonzero(trk.valid)[0]
    if len(frames) < 3:
        return None
    R, t, f = trk.R[frames], trk.t[frames], trk.K[frames, 0, 0]
    gx, gy = np.meshgrid([-spread, 0.0, spread], [-spread, 0.0, spread])
    P = np.stack([gx.ravel() * trk.width / 2, gy.ravel() * trk.height / 2], 1)  # px from the centre
    acc = []
    for i in range(1, len(frames) - 1):
        Pc = depth * np.column_stack([P / f[i], np.ones(len(P))])
        Xw = (Pc - t[i]) @ R[i]  # R^T (Pc - t)
        uv = []
        for k in (i - 1, i, i + 1):
            c = Xw @ R[k].T + t[k]
            if (c[:, 2] <= 0).any():
                break
            uv.append(f[k] * c[:, :2] / c[:, 2:3])
        if len(uv) == 3:
            acc.append(np.linalg.norm(uv[2] - 2.0 * uv[1] + uv[0], axis=1))
    if not acc:
        return None
    return float(np.sqrt(np.mean(np.concatenate(acc) ** 2)))


def _score_screen(trk, tracks, sift_holdout=None):
    s = _score(trk, tracks, sift_holdout)
    if not np.isfinite(s.get("score_px", np.inf)):
        return s
    j = screen_jitter_px(trk, max(s["median_depth"], 1e-9))
    s["jitter_path_px"], s["jitter_screen_px"] = s["jitter_px"], j
    if j is not None:
        s["jitter_px"], s["score_px"] = j, s["reproj_median_px"] + 0.5 * j
    return s


def gt_line(g: dict | None) -> str:
    if not g or g.get("failed"):
        return "no GT"
    return (f"rot {g['rot_err_deg_max']:.3f} deg  focal {g['focal_err_pct_max']:.2f} %  "
            f"{'MET' if g.get('meets_targets') else 'missed'}")


def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    show_all = "--all" in sys.argv
    run = Path(args[0])
    shots = args[1:] or sorted(p.parent.name for p in run.glob("*/solve.json"))
    met = {m: 0 for m in MODES}
    changed = []
    for shot in shots:
        d = run / shot
        solve = json.loads((d / "solve.json").read_text())
        names = list((solve.get("select") or {}).get("scores", {}))
        cands = {n: CameraTrack.load(d / "candidates" / n / "cameras.json") for n in names
                 if (d / "candidates" / n / "cameras.json").exists()}
        if not cands:
            print(f"{shot}: no candidates saved")
            continue
        tracks = dict(np.load(d / "tracks" / "tracks.npz"))
        held = None
        if (d / "sift" / "sift_tracks.npz").exists():
            held = split_sift(dict(np.load(d / "sift" / "sift_tracks.npz")), RefineOptions().max_sift)[1]
        gt = solve.get("gt_eval") or {}
        picks = {}
        for m in MODES:
            sel.score_candidate = _score_screen if m == "screen" else _score
            best, info = sel.select(cands, tracks, held)
            picks[m] = (best, info["scores"])
            met[m] += bool((gt.get(best) or {}).get("meets_targets"))
        flag = "" if picks["path"][0] == picks["screen"][0] else "   <-- CHANGED"
        if flag:
            changed.append(shot)
        print(f"{shot}{flag}  (stored pick: {solve.get('selected')})")
        for m in MODES:
            best, sc = picks[m]
            s = sc.get(best, {})
            print(f"  {m:5s}: {best:28s} score {s.get('score_px', float('nan')):.3f}  reproj "
                  f"{s.get('reproj_median_px', float('nan')):.3f}  jitter {s.get('jitter_px', float('nan')):.3f}  "
                  f"| {gt_line(gt.get(best))}")
        if flag or show_all:
            for n, s in sorted(picks["screen"][1].items(), key=lambda kv: kv[1].get("score_px", 1e9)):
                ji = s.get("jitter_screen_px")
                print(f"      {n:28s} reproj {s.get('reproj_median_px', float('nan')):.3f}  path jitter "
                      f"{s.get('jitter_path_px', float('nan')):.3f}  screen jitter "
                      f"{ji if ji is None else round(ji, 3)}  f {np.median(cands[n].K[cands[n].valid, 0, 0]):.0f}"
                      f"  | {gt_line(gt.get(n))}")
        sys.stdout.flush()
    print(f"\ntargets met: path {met['path']} / {len(shots)}, screen {met['screen']} / {len(shots)}; "
          f"changed picks: {', '.join(changed) or 'none'}")


if __name__ == "__main__":
    main()
