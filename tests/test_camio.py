import copy
import json

import numpy as np
import pytest

from niko.camio import CameraFileError, CameraTrack, validate

from conftest import random_rotation


def make_track(rng, n=12) -> CameraTrack:
    trk = CameraTrack.empty("unit_test", 1920, 1080, 25.0, 1001, n, name="shot")
    for i in range(n):
        trk.K[i] = [[1500.0 + i, 0, 960.5], [0, 1500.0 + i, 540.25], [0, 0, 1]]
        trk.dist[i] = [0.01, -0.002, 0, 0, 0]
        trk.R[i] = random_rotation(rng)
        trk.t[i] = rng.normal(size=3)
        trk.valid[i] = True
    trk.valid[5] = False
    trk.K[5] = trk.dist[5] = trk.R[5] = trk.t[5] = np.nan
    trk.extra = {"scene_size": 12.5}
    return trk


def test_roundtrip(rng, tmp_path):
    trk = make_track(rng)
    p = tmp_path / "cameras.json"
    trk.save(p)
    back = CameraTrack.load(p)
    assert back.n_frames == trk.n_frames and back.frame_start == 1001
    assert np.array_equal(back.valid, trk.valid)
    ok = trk.valid
    for a in ("K", "dist", "R", "t"):
        assert np.array_equal(getattr(back, a)[ok], getattr(trk, a)[ok]), a
        assert np.all(np.isnan(getattr(back, a)[~ok]))
    assert back.extra == {"scene_size": 12.5}
    d = json.loads(p.read_text())
    assert d["frames"][5] == {"i": 5, "frame": 1006, "valid": False,
                              "K": None, "dist": None, "R": None, "t": None}


def _bad(d, fn):
    d = copy.deepcopy(d)
    fn(d)
    with pytest.raises(CameraFileError):
        validate(d)


def test_validation_rejects_bad_files(rng):
    d = make_track(rng).to_dict()
    validate(d)
    _bad(d, lambda d: d["frames"][0].__setitem__("R", [[2, 0, 0], [0, 1, 0], [0, 0, 1]]))
    _bad(d, lambda d: d["frames"][0].__setitem__("R", [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]))  # det -1
    _bad(d, lambda d: d["frames"][0]["K"][0].__setitem__(1, 3.0))  # skew
    _bad(d, lambda d: d["frames"][0]["K"][0].__setitem__(0, -10.0))
    _bad(d, lambda d: d["frames"][1].__setitem__("frame", 7))
    _bad(d, lambda d: d["frames"].pop())
    _bad(d, lambda d: d.pop("conventions"))
    _bad(d, lambda d: d["conventions"].__setitem__("camera", "blender"))
    _bad(d, lambda d: d["frames"][5].__setitem__("R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))  # invalid frame with data
    _bad(d, lambda d: d["frames"][0].__setitem__("t", [0.0, float("nan"), 0.0]))
