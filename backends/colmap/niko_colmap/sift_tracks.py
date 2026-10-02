"""Multi-view SIFT tracks from a COLMAP database (task "sift_tracks").

Verified inlier matches (two-view geometries) are joined into tracks with connected components
over (image, keypoint) nodes; a component that holds two keypoints of the same image is
inconsistent and dropped. Unlike CoTracker tracks these do not drift, and they link frames far
apart (sequential + quadratic pairs), which is what keeps a bundle adjustment from bending.

Options: db (path), width/height (full res), proxy_width/proxy_height (the images COLMAP saw),
         min_views (default 3), max_tracks (default 20000)
Output <out_dir>/sift_tracks.npz: xy [T, N, 2] float32 (full-res, corner origin), vis [T, N] bool
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def run(job: dict, res) -> None:
    opt = job["options"]
    out_dir = Path(job["out_dir"])
    db = pycolmap.Database.open(opt["db"]) if hasattr(pycolmap.Database, "open") else pycolmap.Database(opt["db"])
    images = {im.image_id: im.name for im in db.read_all_images()}
    frame_of = {iid: int(Path(name).stem) for iid, name in images.items()}
    T = int(opt.get("n_frames") or max(frame_of.values()) + 1)  # keyframes may end before the shot
    kps = {iid: np.asarray(db.read_keypoints(iid))[:, :2] for iid in images}
    offset, n = {}, 0
    for iid in sorted(images):
        offset[iid] = n
        n += len(kps[iid])
    rows, cols = [], []
    pairs = db.read_two_view_geometries()
    pair_ids, geoms = (pairs if isinstance(pairs, tuple) else (list(pairs.keys()), list(pairs.values())))
    for pid, g in zip(pair_ids, geoms):
        m = np.asarray(g.inlier_matches)
        if len(m) == 0:
            continue
        i1, i2 = pycolmap.pair_id_to_image_pair(pid) if not isinstance(pid, tuple) else pid
        rows.append(offset[i1] + m[:, 0])
        cols.append(offset[i2] + m[:, 1])
    db.close()
    if not rows:
        raise RuntimeError("no verified matches in the database")
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n))
    ncomp, label = connected_components(graph, directed=False)
    node_img = np.concatenate([np.full(len(kps[iid]), frame_of[iid]) for iid in sorted(images)])
    node_xy = np.concatenate([kps[iid] for iid in sorted(images)])
    matched = np.zeros(n, bool)
    matched[r] = matched[c] = True
    lab, img = label[matched], node_img[matched]
    xy = node_xy[matched]
    # consistency: one keypoint per image per track
    key = lab.astype(np.int64) * T + img
    uniq, cnt = np.unique(key, return_counts=True)
    bad_labels = np.unique(uniq[cnt > 1] // T)
    ok = ~np.isin(lab, bad_labels)
    lab, img, xy = lab[ok], img[ok], xy[ok]
    views = np.bincount(lab, minlength=ncomp)
    good = np.nonzero(views >= int(opt.get("min_views", 3)))[0]
    if len(good) > int(opt.get("max_tracks", 20000)):
        good = good[np.argsort(views[good])[::-1][: int(opt.get("max_tracks", 20000))]]
    col = -np.ones(ncomp, int)
    col[good] = np.arange(len(good))
    keep = col[lab] >= 0
    lab, img, xy = col[lab[keep]], img[keep], xy[keep]
    sx = opt["width"] / opt["proxy_width"]
    sy = opt["height"] / opt["proxy_height"]
    out_xy = np.full((T, len(good), 2), np.nan, np.float32)
    out_vis = np.zeros((T, len(good)), bool)
    out_xy[img, lab] = xy * np.array([sx, sy])  # COLMAP keypoints are corner-origin already
    out_vis[img, lab] = True
    np.savez_compressed(out_dir / "sift_tracks.npz", xy=out_xy, vis=out_vis)
    res.outputs["sift_tracks"] = "sift_tracks.npz"
    res.stats.update({"tracks": int(len(good)), "observations": int(out_vis.sum()),
                      "median_views": float(np.median(views[good])) if len(good) else 0,
                      "inconsistent_components": int(len(bad_labels)), "verified_pairs": len(rows)})
