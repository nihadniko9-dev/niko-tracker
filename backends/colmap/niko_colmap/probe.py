"""`python -m niko_colmap.probe`: pycolmap imports, has CUDA, and runs GPU SIFT on a real image."""

import tempfile
from pathlib import Path

import numpy as np

from niko_backend_common import print_probe


def main():
    import pycolmap

    info = {"pycolmap": pycolmap.__version__, "has_cuda": bool(getattr(pycolmap, "has_cuda", False))}
    try:
        import cv2

        rng = np.random.default_rng(0)
        img = cv2.resize(rng.integers(0, 255, (60, 80), dtype=np.uint8), (640, 480), interpolation=cv2.INTER_CUBIC)
        with tempfile.TemporaryDirectory() as d:
            imgs = Path(d) / "images"
            imgs.mkdir()
            cv2.imwrite(str(imgs / "a.png"), img)
            cv2.imwrite(str(imgs / "b.png"), np.roll(img, 7, axis=1))
            db = Path(d) / "db.db"
            pycolmap.extract_features(db, imgs, device=pycolmap.Device.cuda)
            database = pycolmap.Database.open(db) if hasattr(pycolmap.Database, "open") else pycolmap.Database(db)
            n = [database.num_keypoints_for_image(i.image_id) for i in database.read_all_images()]
            database.close()
        info["gpu_sift_keypoints"] = n
        info["ok"] = info["has_cuda"] and min(n) > 50
    except Exception as e:  # report, don't hide
        info["ok"] = False
        info["error"] = f"{type(e).__name__}: {e}"
    print_probe(info)


if __name__ == "__main__":
    main()
