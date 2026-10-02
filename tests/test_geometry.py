import numpy as np
import pytest

from niko.geometry import (
    K_to_blender,
    blender_matrix_world_to_opencv,
    blender_to_K,
    camera_center,
    opencv_to_blender_matrix_world,
    project,
    rotation_angle_deg,
)

from conftest import axis_angle, random_rotation


def test_extrinsics_roundtrip(rng):
    for _ in range(200):
        R = random_rotation(rng)
        t = rng.normal(scale=10.0, size=3)
        M = opencv_to_blender_matrix_world(R, t)
        R2, t2 = blender_matrix_world_to_opencv(M)
        assert np.abs(R2 - R).max() < 1e-12
        assert np.abs(t2 - t).max() < 1e-10
        # Blender's camera location is the OpenCV camera centre
        assert np.abs(M[:3, 3] - camera_center(R, t)).max() < 1e-10


def test_object_scale_is_ignored(rng):
    R = random_rotation(rng)
    t = rng.normal(size=3)
    M = opencv_to_blender_matrix_world(R, t)
    M[:3, :3] = M[:3, :3] @ np.diag([2.0, 0.5, 3.0])
    R2, t2 = blender_matrix_world_to_opencv(M)
    assert np.abs(R2 - R).max() < 1e-12
    assert np.abs(t2 - t).max() < 1e-10


def test_axis_meaning():
    # OpenCV camera at the origin with R = I looks along world +Z, image-down = world +Y.
    M = opencv_to_blender_matrix_world(np.eye(3), np.zeros(3))
    view_dir = M[:3, :3] @ np.array([0.0, 0.0, -1.0])  # Blender cameras look down local -Z
    up_dir = M[:3, :3] @ np.array([0.0, 1.0, 0.0])  # Blender local +Y is image-up
    assert np.allclose(view_dir, [0, 0, 1])
    assert np.allclose(up_dir, [0, -1, 0])


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920), (1000, 1000), (4096, 1716)])
def test_intrinsics_roundtrip(rng, width, height):
    for _ in range(100):
        fx = rng.uniform(300, 6000)
        fy = fx * rng.choice([1.0, rng.uniform(0.8, 1.25)])
        cx = width / 2 + rng.uniform(-0.1, 0.1) * width
        cy = height / 2 + rng.uniform(-0.1, 0.1) * height
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        b = K_to_blender(K, width, height, sensor_width=rng.uniform(6, 50))
        assert b["pixel_aspect_x"] >= 1.0 and b["pixel_aspect_y"] >= 1.0
        K2 = blender_to_K(b["lens"], b["sensor_width"], b["sensor_height"], b["sensor_fit"],
                          b["shift_x"], b["shift_y"], width, height,
                          b["pixel_aspect_x"], b["pixel_aspect_y"])
        assert np.abs(K2 - K).max() < 1e-9


def test_blender_default_camera():
    # Blender default: 50 mm lens, 36 mm sensor, AUTO fit, 1920x1080 -> fx = 50/36*1920.
    K = blender_to_K(50.0, 36.0, 24.0, "AUTO", 0.0, 0.0, 1920, 1080)
    assert K[0, 0] == pytest.approx(50.0 / 36.0 * 1920)
    assert K[1, 1] == pytest.approx(K[0, 0])
    assert K[0, 2] == 960.0 and K[1, 2] == 540.0
    # Portrait with AUTO fit: the sensor width spans the (larger) height.
    Kp = blender_to_K(50.0, 36.0, 24.0, "AUTO", 0.0, 0.0, 1080, 1920)
    assert Kp[0, 0] == pytest.approx(50.0 / 36.0 * 1920)


def test_project_pinhole_and_distortion():
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    R, t = np.eye(3), np.zeros(3)
    X = np.array([[0.0, 0.0, 5.0], [1.0, -0.5, 10.0]])
    uv, z = project(K, R, t, X)
    assert np.allclose(uv[0], [960, 540]) and np.allclose(z, [5, 10])
    assert np.allclose(uv[1], [960 + 100, 540 - 50])
    uv_d, _ = project(K, R, t, X, dist=[0.1, 0.0, 0.0, 0.0, 0.0])
    r2 = 0.1**2 + 0.05**2
    assert np.allclose(uv_d[1], [960 + 100 * (1 + 0.1 * r2), 540 - 50 * (1 + 0.1 * r2)])
    assert np.allclose(uv_d[0], uv[0])


def test_rotation_angle():
    R = axis_angle([0.3, -1.0, 0.2], 17.0)
    assert rotation_angle_deg(np.eye(3), R) == pytest.approx(17.0)
    assert rotation_angle_deg(R, R) == pytest.approx(0.0, abs=1e-6)
