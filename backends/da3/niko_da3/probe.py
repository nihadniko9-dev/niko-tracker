"""`python -m niko_da3.probe`: CUDA, xformers, the DA3 checkpoint loads and predicts cameras for 3 views."""

import numpy as np

from niko_backend_common import print_probe, probe_torch_cuda

from .common import CKPT_DIR, load_model


def main():
    info = probe_torch_cuda()
    if not info.get("ok"):
        print_probe(info)
    try:
        import xformers

        info["xformers"] = xformers.__version__
    except ImportError:
        info["xformers"] = None
    info["checkpoint"] = (CKPT_DIR / "model.safetensors").is_file()
    if info["checkpoint"]:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8).repeat(8, 0).repeat(8, 1)
        views = [np.roll(base, 6 * k, axis=1) for k in range(3)]
        model = load_model()
        pred = model.inference(views, process_res=336)
        info["extrinsics_shape"] = list(np.asarray(pred.extrinsics).shape)
        info["intrinsics_shape"] = list(np.asarray(pred.intrinsics).shape)
        info["tiny_run"] = info["extrinsics_shape"][0] == 3
    info["ok"] = bool(info.get("ok") and info["checkpoint"] and info.get("tiny_run"))
    print_probe(info)


if __name__ == "__main__":
    main()
