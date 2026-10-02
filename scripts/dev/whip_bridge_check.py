"""Can direct image alignment measure the camera turn between motion-blurred frames where no
feature survives (whip_pan_blur)? Feature-free: phase correlation for a coarse shift, then ECC
(cv2.findTransformECC, homography) on the grey images; the homography of a turning camera is
K R K^-1, so R = K^-1 H K projected onto a rotation. Compared with the GT turn of each pair.

"rot": the same with a 3-parameter rotation warp (K R K^-1) fitted directly to the pixels,
coarse to fine (scipy least squares on cv2.remap residuals) from the phase-correlation start.

usage: python scripts/dev/whip_bridge_check.py <bench_shot_dir> [first last] [scale]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from niko.camio import CameraTrack

SEARCH = True
PASSES = int(__import__('os').environ.get('PASSES', 1))


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else U @ np.diag([1, 1, -1]) @ Vt


def grey(path: Path, scale: float) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def pair_turn(a: np.ndarray, b: np.ndarray, K: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Rotation taking camera a's view to camera b's (R_b R_a^T), and the ECC correlation."""
    small = 0.25
    sa = cv2.resize(a, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
    sb = cv2.resize(b, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow(sa.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(sa, sb, win)
    H = np.array([[1, 0, dx / small], [0, 1, dy / small], [0, 0, 1]], np.float32)
    try:
        cc, H = cv2.findTransformECC(a, b, H, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), None, 5)
    except cv2.error:
        return None, float("nan")
    M = np.linalg.inv(K) @ H.astype(np.float64) @ K
    M /= np.cbrt(np.linalg.det(M))
    return nearest_rotation(M), float(cc)


def turn_from_shift(K: np.ndarray, shape, dx: float, dy: float) -> np.ndarray:
    """Smallest rotation moving the image centre's ray to the ray (dx, dy) px further."""
    h, w = shape
    Ki = np.linalg.inv(K)
    d0 = Ki @ [w / 2, h / 2, 1.0]
    d1 = Ki @ [w / 2 + dx, h / 2 + dy, 1.0]
    d0, d1 = d0 / np.linalg.norm(d0), d1 / np.linalg.norm(d1)
    ax = np.cross(d0, d1)
    n = np.linalg.norm(ax)
    return np.zeros(3) if n < 1e-12 else ax / n * np.arctan2(n, d0 @ d1)


def rot_residuals(w, a, b, K, step):
    h, wd = a.shape
    ys, xs = np.mgrid[0:h:step, 0:wd:step].astype(np.float64)
    x = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    H = K @ Rotation.from_rotvec(w).as_matrix() @ np.linalg.inv(K)
    q = H @ x
    u, v = (q[0] / q[2]).astype(np.float32), (q[1] / q[2]).astype(np.float32)
    bw = cv2.remap(b, u.reshape(xs.shape), v.reshape(xs.shape), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0).ravel()
    av = a[::step, ::step].ravel()
    r = bw - av
    r[bw < 0] = 0.0  # outside b
    return r


def pair_turn_rot(a: np.ndarray, b: np.ndarray, K: np.ndarray, w0: np.ndarray | None = None,
                  spans=((8, (0.5, 1.8)), (4, (0.9, 1.1)), (2, (0.95, 1.05)), (1, (0.98, 1.02)))) -> np.ndarray:
    small = 0.25
    sa = cv2.resize(a, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
    sb = cv2.resize(b, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow(sa.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(sa, sb, win)
    w = turn_from_shift(K, a.shape, dx / small, dy / small) if w0 is None else np.asarray(w0, float)
    # phase correlation under-reads big shifts (its window weights the centre) and the blurred
    # cost has local dips: at every level, search the turn's size along its axis first
    for lv, span in spans:
        S = np.diag([1 / lv, 1 / lv, 1.0])
        al = cv2.resize(a, None, fx=1 / lv, fy=1 / lv, interpolation=cv2.INTER_AREA) if lv > 1 else a
        bl = cv2.resize(b, None, fx=1 / lv, fy=1 / lv, interpolation=cv2.INTER_AREA) if lv > 1 else b
        step = 1 if lv >= 4 else 2
        if SEARCH and np.linalg.norm(w) > 1e-6:
            ks = np.linspace(*span, 41)
            c = [np.mean(rot_residuals(w * k, al, bl, S @ K, step) ** 2) for k in ks]
            w = w * ks[int(np.argmin(c))]
        w = least_squares(rot_residuals, w, args=(al, bl, S @ K, step), loss="soft_l1", f_scale=0.05,
                          x_scale=1e-3, diff_step=1e-4).x
    return Rotation.from_rotvec(w).as_matrix()


def rot_blur(img: np.ndarray, K: np.ndarray, w: np.ndarray, shutter: float, max_n: int = 64) -> np.ndarray:
    """Motion blur of a camera turning by w (rotation vector, per frame) with the shutter open for
    `shutter` of a frame: the mean of the image turned through [-shutter/2, shutter/2] * w."""
    width_px = np.linalg.norm(w) * shutter * K[0, 0]
    n = int(min(max_n, max(1, np.ceil(width_px))))
    if n == 1:
        return img
    Ki = np.linalg.inv(K)
    h, wd = img.shape
    acc = np.zeros_like(img)
    for u in np.linspace(-0.5, 0.5, n):
        H = K @ Rotation.from_rotvec(u * shutter * w).as_matrix() @ Ki
        acc += cv2.warpPerspective(img, H, (wd, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


def path_blur(img: np.ndarray, K: np.ndarray, axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Motion blur as the mean of the image turned about `axis` by each sub-frame angle (rad,
    relative to the frame's own pose): uneven spacing makes the lopsided blur of a changing speed."""
    Ki = np.linalg.inv(K)
    h, wd = img.shape
    acc = np.zeros_like(img)
    for phi in angles:
        H = K @ Rotation.from_rotvec(phi * axis).as_matrix() @ Ki
        acc += cv2.warpPerspective(img, H, (wd, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / len(angles)


def sub_frame_angles(w: dict, first: int, last: int, shutter: float, f_px: float, max_n: int = 64):
    """Per frame: the turn axis and the angles of the camera across its shutter, from a monotone
    spline (PCHIP) through the cumulative turn of the pair estimates."""
    from scipy.interpolate import PchipInterpolator
    ts = np.arange(first, last + 1)
    mags = np.array([np.linalg.norm(w[t]) for t in range(first, last)])
    cum = np.concatenate([[0.0], np.cumsum(mags)])
    spline = PchipInterpolator(ts, cum)
    out = {}
    for i, t in enumerate(ts):
        nb = [w[p] for p in (t - 1, t) if p in w]
        m = np.sum(nb, axis=0)
        axis = m / max(np.linalg.norm(m), 1e-12)
        lo, hi = max(first, t - shutter / 2), min(last, t + shutter / 2)
        span = abs(float(spline(hi) - spline(lo)))
        n = int(min(max_n, max(1, np.ceil(span * f_px))))
        u = np.linspace(lo, hi, n)
        out[t] = (axis, spline(u) - spline(t))
    return out


def main():
    shot = Path(sys.argv[1])
    first, last = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (60, 92)
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    shutter = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    gt = CameraTrack.load(shot / "gt" / "cameras.json")
    frames = sorted((shot / "frames").glob("*.png"))
    S = np.diag([scale, scale, 1.0])
    imgs = {t: grey(frames[t], scale) for t in range(first, last + 1)}
    pairs = list(range(first, last))
    gts = {t: gt.R[t + 1] @ gt.R[t].T for t in pairs}
    # pass 1: plain rotation warp
    w1 = {t: Rotation.from_matrix(pair_turn_rot(imgs[t], imgs[t + 1], S @ gt.K[t])).as_rotvec() for t in pairs}
    w2 = w1
    for it in range(PASSES):
        sub = sub_frame_angles(w2, first, last, shutter, float((S @ gt.K[first])[0, 0]))
        # give each image the other's blur, then align from the previous estimate
        prev, w2 = w2, {}
        for t in pairs:
            K = S @ gt.K[t]
            A = path_blur(imgs[t], K, *sub[t + 1])
            B = path_blur(imgs[t + 1], K, *sub[t])
            w2[t] = Rotation.from_matrix(pair_turn_rot(A, B, K, prev[t], spans=((4, (0.9, 1.1)), (2, (0.95, 1.05)),
                                                                                (1, (0.98, 1.02))))).as_rotvec()
        ch = np.eye(3)
        for t in pairs:
            ch = Rotation.from_rotvec(w2[t]).as_matrix() @ ch
        cg = np.eye(3)
        for t in pairs:
            cg = gts[t] @ cg
        print(f"  blur-equalised pass {it + 1}: chained error {rot_angle_deg(ch @ cg.T):.4f} deg", flush=True)
    print(f"{'pair':>9}{'GT turn':>9}{'pass1 err':>11}{'blur-eq err':>13}   (along-axis, deg)")
    chain = {"gt": np.eye(3), "p1": np.eye(3), "p2": np.eye(3)}
    for t in pairs:
        wg = Rotation.from_matrix(gts[t]).as_rotvec()
        ug = wg / max(np.linalg.norm(wg), 1e-12)
        e1 = np.degrees(w1[t] @ ug - np.linalg.norm(wg))
        e2 = np.degrees(w2[t] @ ug - np.linalg.norm(wg))
        chain["gt"] = gts[t] @ chain["gt"]
        chain["p1"] = Rotation.from_rotvec(w1[t]).as_matrix() @ chain["p1"]
        chain["p2"] = Rotation.from_rotvec(w2[t]).as_matrix() @ chain["p2"]
        print(f"{t:>4}-{t + 1:<4}{np.degrees(np.linalg.norm(wg)):>9.3f}{e1:>+11.4f}{e2:>+13.4f}", flush=True)
    print(f"chained {first}-{last}: GT {rot_angle_deg(chain['gt']):.3f} deg; error pass 1 "
          f"{rot_angle_deg(chain['p1'] @ chain['gt'].T):.4f}, blur-equalised {rot_angle_deg(chain['p2'] @ chain['gt'].T):.4f} deg")


if __name__ == "__main__":
    main()
