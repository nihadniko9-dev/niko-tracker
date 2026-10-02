"""Lens check: equally good candidates that disagree on the lens make the solve's lens uncertain."""

import numpy as np

from niko.camio import CameraTrack
from niko.pipeline.solve import lens_check, lens_spread


def cam(f: float, n: int = 20, stride: int = 1) -> CameraTrack:
    c = CameraTrack.empty("x", 1920, 1080, 25.0, 1, n)
    c.K[:] = [[f, 0, 960], [0, f, 540], [0, 0, 1]]
    c.R[:] = np.eye(3)
    c.t[:] = 0
    c.dist[:] = 0
    c.valid[::stride] = True
    if stride > 1:
        c.extra["backend_meta"] = {"stride": stride}
    return c


def score(reproj: float, rot_only: bool = False, success: float = 1.0) -> dict:
    return {"reproj_median_px": reproj, "score_px": reproj + 0.1, "rotation_only": rot_only, "success_rate": success}


def test_close_lenses_are_not_flagged():
    cands = {"a+ba": cam(1500), "a": cam(1510), "b": cam(1490)}
    scores = {"a+ba": score(0.20), "a": score(0.21), "b": score(0.21)}
    lc = lens_check(cands, "a+ba", scores)
    assert not lc["uncertain"]
    assert lc["spread"]["focal_px"] == [1490.0, 1510.0]


def test_equally_good_fits_far_apart_flag_the_lens():
    cands = {"a+ba": cam(5146), "a": cam(3823), "b+ba": cam(4628)}
    scores = {"a+ba": score(0.200), "a": score(0.198), "b+ba": score(0.193)}
    lc = lens_check(cands, "a", scores)
    assert lc["uncertain"] and lc["reasons"] == ["equally_good_fits"]
    assert lc["spread"]["spread_pct"] > 30


def test_worse_fits_do_not_count():
    cands = {"a+ba": cam(1500), "wrong": cam(2500)}
    scores = {"a+ba": score(0.20), "wrong": score(0.25)}  # 25 % worse held-out reprojection
    assert lens_spread(cands, scores, "a+ba")["focal_px"] == [1500.0, 1500.0]


def test_strided_raw_candidates_count_by_keyframes():
    cands = {"a+ba": cam(1500), "raw": cam(2000, stride=2)}  # raw solved every keyframe, half the frames
    scores = {"a+ba": score(0.20), "raw": score(0.20, success=0.5)}
    assert lens_spread(cands, scores, "a+ba")["focal_px"] == [1500.0, 2000.0]


def test_tripod_pick_is_not_checked_by_spread():
    cands = {"a+tripod": cam(1500), "a+ba": cam(2200)}
    scores = {"a+tripod": score(0.15, rot_only=True), "a+ba": score(0.15)}
    assert lens_spread(cands, scores, "a+tripod") is None
    assert lens_check(cands, "a+tripod", scores) is None  # no depth models either


def test_depth_models_still_flag_forward_motion():
    cands = {"a+ba": cam(200), "megasam": cam(1500), "da3": cam(1400)}
    scores = {"a+ba": score(0.19), "megasam": score(0.30), "da3": score(0.9)}
    lc = lens_check(cands, "a+ba", scores)
    assert "depth_models" in lc["reasons"] and lc["uncertain"]
