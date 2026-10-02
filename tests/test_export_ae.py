"""After Effects mapping. The AE conventions themselves were measured in AE 26.5
(scripts/dev/ae_check.py: AE comp positions vs niko projection, max 0.0004 px on orbit_yard,
zoom_in and distortion_barrel); these tests pin the maths that relies on them."""

import numpy as np

from niko.export_ae import euler_xyz, rot_xyz, undistort_maps
from niko.geometry import project


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_euler_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        M = _random_rotation(rng)
        np.testing.assert_allclose(rot_xyz(euler_xyz(M)), M, atol=1e-12)


def test_measured_ae_orientation_matrix():
    # After Effects 26.5, orientation (10, 20, 30): toWorldVec of the local X / Y / Z axes
    axes = np.array([[0.81379768134937, 0.54383814248233, -0.20487412870286],
                     [-0.46984631039295, 0.8231729446455, 0.31879577759717],
                     [0.34202014332567, -0.16317591116653, 0.92541657839832]]).T
    np.testing.assert_allclose(rot_xyz([10, 20, 30]), axes, atol=1e-12)


def test_ae_camera_equals_opencv_projection():
    """AE camera = (A R_c2w, s A (C - o), zoom f) projects like the OpenCV camera (f, centre pp)."""
    rng = np.random.default_rng(1)
    W, H, f = 1920, 1080, 1500.0
    for _ in range(20):
        R = _random_rotation(rng)            # world -> camera (OpenCV)
        C = rng.normal(size=3) * 5
        X = C + R.T @ np.array([0.3, -0.2, 6.0]) + rng.normal(size=(30, 3)) * 0.5  # in front
        A, o, s = _random_rotation(rng), rng.normal(size=3), 37.0
        Xc = (X - C) @ R.T
        uv_cv = np.stack([W / 2 + f * Xc[:, 0] / Xc[:, 2], H / 2 + f * Xc[:, 1] / Xc[:, 2]], 1)
        M = A @ R.T
        E = euler_xyz(M)
        P = s * A @ (C - o)
        Q = s * (X - o) @ A.T
        Yc = (Q - P) @ rot_xyz(E)            # M^T (Q - P) with AE's measured orientation matrix
        uv_ae = np.stack([W / 2 + f * Yc[:, 0] / Yc[:, 2], H / 2 + f * Yc[:, 1] / Yc[:, 2]], 1)
        np.testing.assert_allclose(uv_ae, uv_cv, atol=1e-8)


def test_undistort_maps_match_our_distortion_model():
    """Output pixel (i, j) of the undistorted frame shows the scene point that our projection puts
    at (i + .5, j + .5) without distortion; the map must fetch it where our model distorts it to."""
    W, H = 1920, 1080
    K = np.array([[1400.0, 0, 961.3], [0, 1400.0, 538.2], [0, 0, 1]])
    dist = np.array([-0.21, 0.35, 0.0, 0.0, 0.0])
    mx, my = undistort_maps(K, dist, W, H)
    worst = 0.0
    for i, j in [(0, 0), (1919, 1079), (960, 540), (100, 900), (1800, 60), (500, 300)]:
        ray = np.linalg.solve(K, [i + 0.5, j + 0.5, 1.0])
        uv_d, _ = project(K, np.eye(3), np.zeros(3), (5 * ray)[None], dist)
        src = np.array([mx[j, i], my[j, i]]) + 0.5  # back to corner-origin
        worst = max(worst, float(np.linalg.norm(src - uv_d[0])))
    assert worst < 1e-3, worst


def test_probe_never_runs_inside_an_open_project():
    """The verification probe closes the project unsaved and quits AE; the guard comes first."""
    from niko import export_ae as ea
    assert "app.quit()" in ea.PROBE
    assert ea.JSX.index("__PROBE_GUARD__") < ea.JSX.index("app.beginUndoGroup")
    assert "app.project.dirty" in ea.PROBE_GUARD and "app.project.file !== null" in ea.PROBE_GUARD
    assert "return;" in ea.PROBE_GUARD
