import numpy as np
import pytest

from niko import metrics

from conftest import axis_angle


def test_rotation_only_estimate_of_a_slightly_moving_camera():
    n = 30
    Rc = np.stack([axis_angle([0, 0, 1], 1.5 * i) for i in range(n)])
    C_gt = np.column_stack([0.05 * np.cos(np.radians(1.5 * np.arange(n))),
                            0.05 * np.sin(np.radians(1.5 * np.arange(n))), np.zeros(n)])  # 5 cm nodal offset
    C_est = np.zeros((n, 3))  # tripod solver: one shared centre
    r = metrics.ate(C_est, C_gt, scene_size=50.0, R_est_c2w=Rc, R_gt_c2w=Rc)
    assert r["mode"] == "est_static"
    # the estimate sits at the GT centroid; its error is the GT's own small motion
    expected = 100 * np.sqrt(np.mean(np.sum((C_gt - C_gt.mean(0)) ** 2, axis=1))) / 50.0
    assert r["rmse_pct"] == pytest.approx(expected, rel=1e-9)
