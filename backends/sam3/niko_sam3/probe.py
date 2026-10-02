"""`python -m niko_sam3.probe`: CUDA, sam3 imports, the SAM 3.1 checkpoint runs on a 3-frame clip."""

import tempfile
from pathlib import Path

import numpy as np

from niko_backend_common import print_probe, probe_torch_cuda

from .common import CKPT, build_predictor, masks_for_prompt


def main():
    info = probe_torch_cuda()
    if not info.get("ok"):
        print_probe(info)
    import sam3  # noqa: F401

    info["checkpoint"] = CKPT.is_file()
    if info["checkpoint"]:
        import cv2

        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            base = cv2.resize(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), (320, 240))
            for k in range(3):
                cv2.imwrite(str(Path(d) / f"{k:06d}.jpg"), np.roll(base, 4 * k, axis=1))
            predictor = build_predictor()
            sid = predictor.handle_request(request=dict(type="start_session", resource_path=d))["session_id"]
            per_frame, _, _ = masks_for_prompt(predictor, sid, "person", 3)
            predictor.handle_request(request=dict(type="close_session", session_id=sid))
        info["tiny_run"] = sorted(per_frame) == [0, 1, 2]
    info["ok"] = bool(info.get("ok") and info["checkpoint"] and info.get("tiny_run"))
    print_probe(info)


if __name__ == "__main__":
    main()
