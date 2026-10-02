"""The levelled world (export_ae.world_alignment): up from the cameras' level right axes when the
camera turns, a ground that lies below the cameras, no ground at all rather than a wrong one."""

import numpy as np

from niko.camio import CameraTrack
from niko.export_ae import level_up, world_alignment


def _track(centers, yaw_deg, pitch_deg, roll_noise_deg=0.0, seed=0, W=1920, H=1080, f=1500.0):
    """Cameras in a Z-up world: yaw about Z (0 = looking along +X), pitch down positive."""
    rng = np.random.default_rng(seed)
    n = len(centers)
    trk = CameraTrack.empty("level", W, H, 25.0, 1, n)
    for i in range(n):
        ps, pp = np.radians(yaw_deg[i]), np.radians(pitch_deg[i])
        fwd = np.array([np.cos(pp) * np.cos(ps), np.cos(pp) * np.sin(ps), -np.sin(pp)])
        right = np.array([np.sin(ps), -np.cos(ps), 0.0])
        a = np.radians(rng.normal(0, roll_noise_deg)) if roll_noise_deg else 0.0
        right = np.cos(a) * right + np.sin(a) * np.cross(fwd, right)  # roll about the view axis
        down = np.cross(fwd, right)
        Rc2w = np.column_stack([right, down, fwd])
        trk.R[i] = Rc2w.T
        trk.t[i] = -Rc2w.T @ centers[i]
        trk.K[i] = [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]
        trk.dist[i] = 0
    trk.valid[:] = True
    return trk


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def test_up_from_turning_drone_pitched_far_down():
    # a drone 100 m up, 55 deg down, drifting 8 deg in heading: the cameras' own up is 55 deg off
    n = 60
    C = np.column_stack([np.linspace(-5, 5, n), np.full(n, -60.0), np.full(n, 100.0)])
    trk = _track(C, np.linspace(86, 94, n), np.full(n, 55.0), roll_noise_deg=0.05)
    up, level = level_up(trk)
    assert level and _angle(up, [0, 0, 1]) < 0.5
    rng = np.random.default_rng(1)
    ground = np.column_stack([rng.uniform(-60, 60, 3000), rng.uniform(-30, 90, 3000), rng.normal(0, 0.05, 3000)])
    roofs = np.column_stack([rng.uniform(-20, 20, 2500), rng.uniform(0, 40, 2500), 8 + rng.uniform(-6, 6, 2500)])
    A, origin, how = world_alignment(trk, np.vstack([ground, roofs]), np.random.default_rng(0))
    assert how == "ground plane"
    assert _angle(-A[1], [0, 0, 1]) < 0.5      # AE's -Y is up
    assert abs(origin[2]) < 0.3                 # the origin sits on the ground


def test_plane_through_the_cameras_is_not_the_ground():
    # a dolly with no turning (up from the cameras themselves); most points lie on a level band at
    # the camera's height (what a telephoto shot can triangulate), the real floor 1.5 m below
    n = 40
    C = np.column_stack([np.linspace(0, 2, n), np.zeros(n), np.full(n, 1.5)])
    trk = _track(C, np.full(n, 90.0), np.full(n, 2.0))
    assert not level_up(trk)[1]
    rng = np.random.default_rng(2)
    band = np.column_stack([rng.uniform(-6, 8, 4000), rng.uniform(4, 12, 4000), 1.5 + rng.normal(0, 0.01, 4000)])
    floor = np.column_stack([rng.uniform(-4, 6, 1200), rng.uniform(3, 10, 1200), rng.normal(0, 0.01, 1200)])
    _, origin, how = world_alignment(trk, np.vstack([band, floor]), np.random.default_rng(0))
    assert how == "ground plane" and abs(origin[2]) < 0.05
    # only the band: no ground rather than a ground through the camera
    _, _, how = world_alignment(trk, band, np.random.default_rng(0))
    assert how == "cameras"


def test_rolling_camera_without_turning_is_not_level():
    # the right axes swing about the view axis (roll), not about up: no level estimate
    n = 50
    C = np.column_stack([np.zeros(n), np.linspace(0, 1, n), np.full(n, 1.6)])
    trk = _track(C, np.full(n, 0.0), np.full(n, 0.0), roll_noise_deg=6.0)
    up, level = level_up(trk)
    assert not level and _angle(up, [0, 0, 1]) < 3
