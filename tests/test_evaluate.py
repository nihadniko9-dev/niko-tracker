import numpy as np
import pytest

from niko.camio import CameraTrack
from niko.evaluate import evaluate_against_gt
from niko.sim3 import Sim3

from conftest import random_rotation
from test_metrics import orbit


def make_gt(n=40):
    C, Rc = orbit(n=n)
    gt = CameraTrack.empty("gt", 1920, 1080, 25.0, 1, n)
    for i in range(n):
        gt.R[i] = Rc[i].T
        gt.t[i] = -Rc[i].T @ C[i]
        gt.K[i] = [[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]]
        gt.dist[i] = 0
    gt.valid[:] = True
    gt.extra = {"scene_size": 20.0, "median_depth": 10.0}
    return gt


def test_gt_against_itself_and_a_similarity(rng):
    gt = make_gt()
    r = evaluate_against_gt(gt, gt)
    assert r["ate_pct"] < 1e-9 and r["rot_err_deg_max"] < 1e-6 and r["meets_targets"]

    T = Sim3(0.2, random_rotation(rng), rng.normal(size=3))
    est = make_gt()
    for i in range(est.n_frames):  # map the world by T: C' = T(C), R_c2w' = T.R R_c2w
        C = T.apply(gt.centers[i][None])[0]
        Rc2w = T.R @ gt.R_c2w[i]
        est.R[i] = Rc2w.T
        est.t[i] = -Rc2w.T @ C
    r = evaluate_against_gt(est, gt)
    assert r["ate_pct"] < 1e-8 and r["rot_err_deg_max"] < 1e-6
    assert r["sim3_scale"] == pytest.approx(5.0)
    assert r["rpe10_trans_pct"] < 1e-8


def test_missing_frames_and_focal():
    gt = make_gt()
    est = make_gt()
    est.valid[5] = False
    est.K[:, 0, 0] = 1530.0
    r = evaluate_against_gt(est, gt)
    assert r["success_rate"] == pytest.approx(39 / 40)
    assert r["focal_err_pct_max"] == pytest.approx(2.0)
    assert not r["meets_targets"]
    assert "rpe1_rot_deg" not in r  # drift is only measured on fully valid tracks
