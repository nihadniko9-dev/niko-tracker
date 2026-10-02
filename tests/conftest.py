import os
from pathlib import Path

import numpy as np
import pytest


def pytest_configure(config):
    """Test folders under $NIKO_HOME/pytest_tmp instead of /tmp: WSL clears /tmp when the distro
    goes idle, which took the log of a failed smoke run with it (2026-10-02). Kept until the next run."""
    if not config.option.basetemp and os.environ.get("NIKO_HOME"):
        config.option.basetemp = str(Path(os.environ["NIKO_HOME"]) / "pytest_tmp")


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle(axis, deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
