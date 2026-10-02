"""Where does tracking break? For every frame boundary t | t+1, count the multi-view SIFT tracks
seen on both sides within a few frames (t-w+1 .. t and t+1 .. t+w): the tracks that tie the two
sides of the cut together. A whip pan with motion blur leaves (almost) none.

usage: python scripts/dev/track_gaps.py <solve.json | run_dir> ... [--window 3]
Prints, per solve, the weakest boundaries (frame numbers 1-based like the UI).
"""

import json
import sys
from pathlib import Path

import numpy as np


def crossing_counts(vis: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Over the frames that have SIFT observations at all (keyframes of a strided solve, frames
    COLMAP registered): for each boundary between consecutive such frames, the tracks seen in the
    `window` frames before it AND the `window` after. Returns (rows, counts[len(rows) - 1])."""
    rows = np.nonzero(vis.any(1))[0]
    v = vis[rows]
    out = np.zeros(len(rows) - 1, int)
    for i in range(len(rows) - 1):
        before = v[max(0, i - window + 1): i + 1].any(0)
        after = v[i + 1: i + 1 + window].any(0)
        out[i] = int((before & after).sum())
    return rows, out


def main():
    args = sys.argv[1:]
    window = 3
    if "--window" in args:
        i = args.index("--window")
        window = int(args[i + 1])
        del args[i:i + 2]
    paths = []
    for a in map(Path, args):
        paths += sorted(a.glob("*/solve.json")) if a.is_dir() else [a]
    print(f"{'solve':34s}{'frames':>7}{'median':>8}{'weakest boundaries (frame|frame: tracks)':>44}")
    for p in paths:
        npz = p.parent / "sift" / "sift_tracks.npz"
        if not npz.exists():
            print(f"{p.parent.name[:34]:34s}  no SIFT tracks")
            continue
        vis = np.load(npz)["vis"]
        rows, c = crossing_counts(vis, window)
        order = np.argsort(c)[:4]
        weak = "  ".join(f"{rows[i] + 1}|{rows[i + 1] + 1}: {c[i]}" for i in sorted(order))
        print(f"{p.parent.name[:34]:34s}{vis.shape[0]:>7}{int(np.median(c)):>8}   min {c.min():>5}   {weak}")


if __name__ == "__main__":
    main()
