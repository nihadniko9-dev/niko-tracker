"""bench report on a tiny fake run: two methods, one shot, one failed candidate."""

import csv
import json

import cv2
import numpy as np

from niko.bench.report import make_report
from niko.evaluate import evaluate_against_gt

from test_evaluate import make_gt


def test_report_from_fake_run(tmp_path):
    shot_src = tmp_path / "bench" / "shotA"
    (shot_src / "gt").mkdir(parents=True)
    gt = make_gt()
    gt.save(shot_src / "gt" / "cameras.json")
    (shot_src / "spec.json").write_text(json.dumps({"n_frames": gt.n_frames, "tags": ["orbit", "easy"]}))
    (shot_src / "scene.json").write_text(json.dumps({"scene_size": 20.0}))

    run = tmp_path / "runs" / "fake"
    shot = run / "shotA"
    (shot / "proxy").mkdir(parents=True)
    cv2.imwrite(str(shot / "proxy" / "000000.jpg"), np.full((90, 160, 3), 120, np.uint8))
    est = make_gt()
    est.K[:, 0, 0] *= 1.05  # 5 % focal error -> targets missed
    (shot / "candidates" / "good").mkdir(parents=True)
    (shot / "candidates" / "bad_focal").mkdir(parents=True)
    gt.save(shot / "candidates" / "good" / "cameras.json")
    est.save(shot / "candidates" / "bad_focal" / "cameras.json")
    ev_good = evaluate_against_gt(gt, gt)
    ev_bad = evaluate_against_gt(est, gt)
    rep = {"clip": str(shot_src), "selected": "good", "total_seconds": 12.0,
           "gt_eval": {"good": {**ev_good, "heldout_reproj_px": 0.4}, "bad_focal": {**ev_bad, "heldout_reproj_px": 0.9}},
           "candidates": {"good": {"runtime_s": 3.0, "peak_vram_mb": 100.0},
                          "bad_focal": {"runtime_s": 4.0, "peak_vram_mb": 200.0}},
           "stages": {"candidate:crashy": {"ok": False, "error": "RuntimeError: boom"}}}
    (shot / "solve.json").write_text(json.dumps(rep, default=str))

    out = make_report(run)
    html = (run / "report.html").read_text()
    assert "good" in html and "bad_focal" in html and "crashy" in html
    assert "✓ met" in html and "✗ missed" in html and "✗ failed" in html
    assert "<polyline" in html  # trajectory plot drawn
    rows = list(csv.DictReader((run / "metrics.csv").open()))
    assert {r["method"] for r in rows} == {"good", "bad_focal", "crashy", "niko_selected", "oracle_best"}
    sel = next(r for r in rows if r["method"] == "niko_selected")
    assert sel["error"] == "picked good" and sel["meets_targets"] == "True"
    summary = {s["method"]: s for s in out["summary"]}
    assert summary["good"]["targets_met"] == 1 and summary["bad_focal"]["targets_met"] == 0
    assert abs(summary["bad_focal"]["focal_err_pct_max"] - 5.0) < 1e-6
