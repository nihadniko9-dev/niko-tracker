import numpy as np
import pytest

from niko.sim3 import DegenerateAlignment, Sim3, align_rotations, is_collinear, umeyama

from conftest import random_rotation


def test_recovers_exact_sim3(rng):
    for _ in range(50):
        src = rng.normal(size=(30, 3))
        gt = Sim3(float(rng.uniform(0.01, 100)), random_rotation(rng), rng.normal(scale=50, size=3))
        est = umeyama(src, gt.apply(src))
        assert est.s == pytest.approx(gt.s, rel=1e-10)
        assert np.abs(est.R - gt.R).max() < 1e-10
        assert np.abs(est.t - gt.t).max() < 1e-8 * max(1.0, np.abs(gt.t).max())


def test_noisy_sim3(rng):
    src = rng.uniform(-10, 10, size=(500, 3))
    gt = Sim3(2.5, random_rotation(rng), np.array([1.0, -2.0, 3.0]))
    dst = gt.apply(src) + rng.normal(scale=0.01, size=src.shape)
    est = umeyama(src, dst)
    assert est.s == pytest.approx(2.5, rel=1e-3)
    assert np.abs(est.R - gt.R).max() < 1e-3


def test_without_scale(rng):
    src = rng.normal(size=(20, 3))
    gt = Sim3(1.0, random_rotation(rng), rng.normal(size=3))
    est = umeyama(src, gt.apply(src), with_scale=False)
    assert est.s == 1.0
    assert np.abs(est.R - gt.R).max() < 1e-10


def test_never_returns_reflection(rng):
    src = rng.normal(size=(40, 3))
    dst = src * np.array([1.0, 1.0, -1.0])  # a mirror image: best proper rotation, det = +1
    est = umeyama(src, dst)
    assert np.linalg.det(est.R) == pytest.approx(1.0)


def test_inverse(rng):
    T = Sim3(3.0, random_rotation(rng), rng.normal(size=3))
    X = rng.normal(size=(10, 3))
    assert np.allclose(T.inverse().apply(T.apply(X)), X)


def test_degenerate_inputs():
    with pytest.raises(DegenerateAlignment):
        umeyama(np.zeros((5, 3)), np.ones((5, 3)))
    with pytest.raises(DegenerateAlignment):
        umeyama(np.eye(3)[:2], np.eye(3)[:2])


def test_align_rotations(rng):
    A = random_rotation(rng)
    Rs = np.stack([random_rotation(rng) for _ in range(20)])
    got = align_rotations(Rs, np.einsum("ij,njk->nik", A, Rs))
    assert np.abs(got - A).max() < 1e-10


def test_collinear_detection(rng):
    line = np.outer(np.linspace(0, 10, 50), [1.0, 2.0, 0.5]) + 3.0
    assert is_collinear(line)
    assert is_collinear(np.ones((10, 3)))
    assert not is_collinear(rng.normal(size=(10, 3)))
