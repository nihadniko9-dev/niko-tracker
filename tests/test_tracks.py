import numpy as np

from niko.tracks import split_sift, split_tracks, track_gaps


def test_split_tracks_segments_and_bookkeeping():
    T, N = 40, 3
    xy = np.random.default_rng(0).normal(size=(T, N, 2)).astype(np.float32)
    vis = np.zeros((T, N), bool)
    vis[0:40, 0] = True      # long track from frame 0 -> 3 segments of 16, 16, 8
    vis[10:14, 1] = True     # short track queried at 10 -> 1 segment
    vis[20:21, 2] = True     # single observation -> dropped (needs >= 2)
    tr = {"xy": xy, "vis": vis, "conf": np.ones((T, N), np.float32),
          "query_frame": np.array([0, 10, 20]), "holdout": np.array([False, True, False])}
    s = split_tracks(tr, max_len=16)
    assert s["vis"].shape == (T, 4)
    assert sorted(s["vis"].sum(0).tolist()) == [4, 8, 16, 16]
    assert set(s["parent"].tolist()) == {0, 1}
    assert s["holdout"][s["parent"] == 1].all() and not s["holdout"][s["parent"] == 0].any()
    for c in range(4):  # every observation copied unchanged
        t = np.nonzero(s["vis"][:, c])[0]
        n = s["parent"][c]
        assert np.array_equal(s["xy"][t, c], xy[t, n])
        assert (t.max() - t.min()) < 16


def test_split_sift_longest_train_rest_held_out():
    T, N = 30, 6
    vis = np.zeros((T, N), bool)
    for n, length in enumerate([30, 2, 10, 20, 5, 3]):  # track 1 (2 views) is too short for either set
        vis[:length, n] = True
    xy = np.arange(T * N * 2, dtype=np.float32).reshape(T, N, 2)
    train, held = split_sift({"xy": xy, "vis": vis}, n_train=2)
    assert train["vis"].sum(0).tolist() == [30, 20]          # the two longest, in original order
    assert sorted(held["vis"].sum(0).tolist()) == [3, 5, 10]  # the rest with >= 3 views
    np.testing.assert_array_equal(train["xy"][:, 1], xy[:, 3])


def _overlapping_tracks(T, spans, per_frame=200, length=8):
    """Tracks of `length` frames starting every frame inside each (first, last) span."""
    cols = []
    for a, b in spans:
        for s in range(a, b - length + 2):
            col = np.zeros((T, per_frame), bool)
            col[s:s + length] = True
            cols.append(col)
    return np.concatenate(cols, 1)


def test_track_gaps_finds_a_whip_and_nothing_else():
    T = 60
    whole = _overlapping_tracks(T, [(0, 59)])
    assert track_gaps(whole) == []
    # frames 25-34 untracked (blurred whip, no observations), halves tied by nothing
    broken = _overlapping_tracks(T, [(0, 24), (35, 59)])
    assert track_gaps(broken) == [{"last_before": 24, "first_after": 35, "tracks": 0}]
    # a few stray matches across the whip do not tie the halves
    stray = broken.copy()
    stray[20:40, :10] = True
    gaps = track_gaps(stray)
    assert len(gaps) == 1 and gaps[0]["tracks"] == 10


def test_track_gaps_on_keyframes():
    T = 120
    vis = _overlapping_tracks(T, [(0, 119)], length=20)
    vis[1::4] = False  # a strided solve: observations on every 4th frame only
    vis[2::4] = False
    vis[3::4] = False
    assert track_gaps(vis) == []
