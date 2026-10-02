import json

from niko.clip_report import clip_report

from test_camio import make_track


def test_clip_report_page(rng, tmp_path):
    d = tmp_path / "solve"
    (d / "selected").mkdir(parents=True)
    trk = make_track(rng)
    trk.save(d / "selected" / "cameras.json")
    (d / "selected" / "errors.json").write_text(json.dumps(
        {"frame_start": 1001, "mean_px": [0.3, 0.7, None, 1.4] + [0.2] * 8}))
    (d / "solve.json").write_text(json.dumps({
        "clip": "/mnt/d/clips/street 01.mp4", "selected": "colmap_global+ba", "keyframe_stride": 3,
        "stages": {"refine": {"seconds": 60}, "select": {"seconds": 5}},
        "solve_error": {"average_px": 0.42, "inlier_fraction": 0.99, "source": "sift",
                        "worst_frame": {"frame": 1004, "mean_px": 1.4}},
        "lens_check": {"uncertain": True}}))
    out = clip_report([d], tmp_path / "page.html", title="test <clips>")
    page = out.read_text(encoding="utf-8")
    assert "street 01.mp4" in page and "0.42 px" in page and "Excellent" in page
    assert "uncertain" in page and "every 3" in page and "1004" in page
    assert page.count("<rect") == 11  # one bar per measured frame, none for the missing one
    assert "test &lt;clips&gt;" in page  # escaped
