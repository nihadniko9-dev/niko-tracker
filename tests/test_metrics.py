import numpy as np
import pytest

from niko import metrics
from niko.geometry import project
from niko.sim3 import Sim3

from conftest import axis_angle, random_rotation


def orbit(n=120, radius=10.0):
    """Cameras on a circle looking at the origin: camera-to-world rotations and centres."""
    ang = np.linspace(0, np.pi, n)
    C = np.column_stack([radius * np.cos(ang), radius * np.sin(ang), 2.0 + 0.5 * np.sin(3 * ang)])
    R_c2w = []
    for c in C:
        z = -c / np.linalg.norm(c)  # OpenCV forward
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_c2w.append(np.column_stack([x, y, z]))
    return C, np.stack(R_c2w)


def transform(C, R_c2w, T: Sim3):
    return T.apply(C), np.einsum("ij,njk->nik", T.R, R_c2w)


def test_ate_zero_under_any_similarity(rng):
    C, Rc = orbit()
    T = Sim3(0.37, random_rotation(rng), rng.normal(size=3))
    C_est, R_est = transform(C, Rc, T.inverse())
    r = metrics.ate(C_est, C, scene_size=20.0, R_est_c2w=R_est, R_gt_c2w=Rc)
    assert r["mode"] == "umeyama"
    assert r["rmse_pct"] < 1e-9
    assert r["scale"] == pytest.approx(0.37, rel=1e-9)
    assert metrics.rotation_errors_deg(R_est, Rc).max() < 1e-6


def test_ate_noise_level_and_units(rng):
    C, Rc = orbit(n=400)
    noisy = C + rng.normal(scale=0.05, size=C.shape)
    r1 = metrics.ate(noisy, C, scene_size=20.0)
    expected = 100 * 0.05 * np.sqrt(3) / 20.0  # RMS 3D noise as % of scene size
    assert r1["rmse_pct"] == pytest.approx(expected, rel=0.1)
    r2 = metrics.ate(noisy, C, scene_size=40.0)
    assert r2["rmse_pct"] == pytest.approx(r1["rmse_pct"] / 2, rel=1e-9)


def test_collinear_path_uses_orientations(rng):
    n = 60
    C = np.outer(np.linspace(0, 5, n), [1.0, 0.0, 0.0])
    Rc = np.stack([axis_angle([0, 0, 1], 0.2 * i) for i in range(n)])
    T = Sim3(4.0, random_rotation(rng), rng.normal(size=3))
    C_est, R_est = transform(C, Rc, T)
    r = metrics.ate(C_est, C, 10.0, R_est, Rc)
    assert r["mode"] == "collinear"
    assert r["rmse_pct"] < 1e-9
    assert r["scale"] == pytest.approx(0.25, rel=1e-9)


def test_static_tripod(rng):
    n = 50
    C = np.zeros((n, 3))
    Rc = np.stack([axis_angle([0, 1, 0], 0.5 * i) for i in range(n)])
    r = metrics.ate(np.full((n, 3), 7.0), C, 10.0, Rc, Rc, est_scene_size=3.0)
    assert r["mode"] == "static" and r["rmse_pct"] == 0.0
    wobble = np.full((n, 3), 7.0)
    wobble[:, 0] += 0.03 * np.where(np.arange(n) % 2, 1, -1)
    r = metrics.ate(wobble, C, 10.0, Rc, Rc, est_scene_size=3.0)
    assert r["rmse_pct"] == pytest.approx(1.0)


def test_rotation_error_detects_one_bad_frame():
    C, Rc = orbit(n=200)
    bad = Rc.copy()
    bad[100] = bad[100] @ axis_angle([1, 0.2, 0], 0.5)
    err = metrics.rotation_errors_deg(bad, Rc)
    assert err[100] == pytest.approx(0.5, abs=0.01)
    assert np.delete(err, 100).max() < 0.01


def test_rpe_zero_for_similarity(rng):
    C, Rc = orbit()
    T = Sim3(3.0, random_rotation(rng), rng.normal(size=3))
    C_est, R_est = transform(C, Rc, T)
    for delta in (1, 10):
        r = metrics.rpe(C_est, R_est, C, Rc, scene_size=20.0, scale=1 / 3.0, delta=delta)
        assert r["rot_deg_rmse"] < 1e-6 and r["trans_pct_rmse"] < 1e-9


def test_focal_error():
    assert np.allclose(metrics.focal_error_pct([1020.0, 980.0], [1000.0, 1000.0]), [2.0, 2.0])


def test_jitter():
    n = 100
    C = np.outer(np.arange(n), [0.1, 0.0, 0.05])  # constant velocity
    Rc = np.stack([axis_angle([0, 0, 1], 0.3 * i) for i in range(n)])  # constant angular velocity
    j = metrics.jitter(C, Rc, scale=10.0)
    assert j["trans_pct_rms"] < 1e-9 and j["rot_deg_rms"] < 1e-6
    C2 = C.copy()
    C2[50, 1] += 0.1
    assert metrics.jitter(C2, Rc, scale=10.0)["trans_pct_rms"] > 0.1


def test_reprojection_errors():
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 400], [0, 0, 1]])
    R, t = np.eye(3), np.zeros(3)
    X = np.array([[0.1, 0.2, 5.0], [0.0, 0.0, -5.0]])
    uv, _ = project(K, R, t, X)
    err = metrics.reprojection_errors(K, R, t, X, uv + [[0.3, 0.4], [0.0, 0.0]])
    assert err[0] == pytest.approx(0.5)
    assert np.isinf(err[1])  # behind the camera


def test_success_rate():
    assert metrics.success_rate(np.array([1, 1, 0, 1], bool)) == 0.75
