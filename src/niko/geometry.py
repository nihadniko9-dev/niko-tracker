"""Camera geometry: projection and OpenCV <-> Blender conversion.

Conventions (docs/SCHEMA.md):
- OpenCV camera: +X right, +Y down, +Z forward. Extrinsics are world->camera,
  x_cam = R @ X + t, camera centre C = -R.T @ t.
- Blender camera: +X right, +Y up, looks down -Z. The world frame is shared.
- Pixels use the "corner" origin: (0, 0) is the top-left corner of the image and
  the centre of pixel (col, row) is (col + 0.5, row + 0.5). Blender and COLMAP
  use the same origin, so no half-pixel shift appears in these conversions.
"""

from __future__ import annotations

import numpy as np

# Maps OpenCV camera axes to Blender camera axes (and back: it is its own inverse).
FLIP_YZ = np.diag([1.0, -1.0, -1.0])


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    """Closest proper rotation to a 3x3 matrix (polar decomposition via SVD)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotations, in degrees. Works on [..., 3, 3] stacks."""
    Rrel = np.einsum("...ji,...jk->...ik", Ra, Rb)  # Ra^T @ Rb
    cos = (np.trace(Rrel, axis1=-2, axis2=-1) - 1.0) / 2.0
    # atan2 keeps full precision for tiny angles, where arccos(cos) does not.
    vee = np.stack([
        Rrel[..., 2, 1] - Rrel[..., 1, 2],
        Rrel[..., 0, 2] - Rrel[..., 2, 0],
        Rrel[..., 1, 0] - Rrel[..., 0, 1],
    ], axis=-1)
    sin = 0.5 * np.linalg.norm(vee, axis=-1)
    return np.degrees(np.arctan2(sin, cos))


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """C = -R^T t, vectorised over leading dimensions."""
    return -np.einsum("...ji,...j->...i", R, t)


def project(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    dist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points X [N, 3] to pixels [N, 2] (corner origin).

    Returns (uv, depth). `dist` is OpenCV order [k1, k2, p1, p2, k3].
    """
    Xc = np.asarray(X, dtype=np.float64) @ np.asarray(R).T + np.asarray(t)
    z = Xc[:, 2]
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    if dist is not None and np.any(np.asarray(dist) != 0):
        k1, k2, p1, p2, k3 = np.asarray(dist, dtype=np.float64)
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x, y = xd, yd
    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=-1), z


# --------------------------------------------------------------------------- #
# Extrinsics
# --------------------------------------------------------------------------- #


def opencv_to_blender_matrix_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """OpenCV world->camera (R, t) to a Blender camera matrix_world (4x4)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = R.T @ FLIP_YZ
    M[:3, 3] = -R.T @ t
    return M


def blender_matrix_world_to_opencv(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blender camera matrix_world (4x4, may carry object scale) to OpenCV (R, t).

    Object scale does not change what a Blender camera sees, so it is removed
    by taking the nearest rotation of the 3x3 block.
    """
    M = np.asarray(M, dtype=np.float64)
    R_c2w_blender = nearest_rotation(M[:3, :3])
    C = M[:3, 3]
    R = (R_c2w_blender @ FLIP_YZ).T
    t = -R @ C
    return R, t


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #


def _blender_fit(sensor_fit: str, width: float, height: float, aspx: float, aspy: float) -> str:
    """Resolved fit, as BKE_camera_sensor_fit()."""
    if sensor_fit == "AUTO":
        return "HORIZONTAL" if aspx * width >= aspy * height else "VERTICAL"
    if sensor_fit not in ("HORIZONTAL", "VERTICAL"):
        raise ValueError(f"unknown sensor_fit {sensor_fit!r}")
    return sensor_fit


def blender_to_K(
    lens: float,
    sensor_width: float,
    sensor_height: float,
    sensor_fit: str,
    shift_x: float,
    shift_y: float,
    width: int,
    height: int,
    pixel_aspect_x: float = 1.0,
    pixel_aspect_y: float = 1.0,
) -> np.ndarray:
    """Blender perspective camera settings to a pixel intrinsics matrix K.

    Follows BKE_camera_params_compute_viewplane(): the view plane is measured in
    x-pixel units; one y pixel spans ycor = aspy / aspx of them. `width`/`height`
    are the effective render size (resolution * percentage / 100).
    """
    ycor = pixel_aspect_y / pixel_aspect_x
    fit = _blender_fit(sensor_fit, width, height, pixel_aspect_x, pixel_aspect_y)
    # BKE_camera_sensor_size(): only an explicit VERTICAL fit uses sensor_height.
    sensor_size = sensor_height if sensor_fit == "VERTICAL" else sensor_width
    viewfac = width if fit == "HORIZONTAL" else ycor * height

    fx = lens * viewfac / sensor_size
    fy = fx / ycor
    cx = width / 2.0 - shift_x * viewfac
    cy = height / 2.0 + shift_y * viewfac / ycor
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def K_to_blender(K: np.ndarray, width: int, height: int, sensor_width: float = 36.0) -> dict:
    """Pixel intrinsics to Blender camera + render settings (sensor_fit HORIZONTAL).

    Blender requires both pixel aspect values to be >= 1, so the larger one
    carries the ratio fx / fy.
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if abs(float(K[0, 1])) > 1e-9:
        raise ValueError("Blender cameras cannot represent skew")
    ycor = fx / fy
    if ycor >= 1.0:
        aspx, aspy = 1.0, ycor
    else:
        aspx, aspy = 1.0 / ycor, 1.0
    viewfac = float(width)
    return {
        "lens": fx * sensor_width / viewfac,
        "sensor_width": float(sensor_width),
        "sensor_height": float(sensor_width) * height / width,
        "sensor_fit": "HORIZONTAL",
        "shift_x": (width / 2.0 - cx) / viewfac,
        "shift_y": (cy - height / 2.0) * ycor / viewfac,
        "pixel_aspect_x": aspx,
        "pixel_aspect_y": aspy,
        "resolution_x": int(width),
        "resolution_y": int(height),
    }
