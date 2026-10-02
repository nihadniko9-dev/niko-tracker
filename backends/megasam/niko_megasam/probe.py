"""`python -m niko_megasam.probe`: CUDA, the sm_120 DROID/lietorch extensions run, weights present."""

from niko_backend_common import print_probe, probe_torch_cuda

from .common import DA_CKPT, UNIDEPTH_DIR, WEIGHTS


def main():
    info = probe_torch_cuda()
    if not info.get("ok"):
        print_probe(info)
    import torch

    try:
        import droid_backends  # noqa: F401
        from lietorch import SE3

        x = torch.randn(4, 6, device="cuda", dtype=torch.float32) * 0.1
        T = SE3.exp(x)
        back = T.log()
        info["lietorch_exp_log_err"] = float((back - x).abs().max())
        info["extensions"] = info["lietorch_exp_log_err"] < 1e-4
    except Exception as e:
        info["extensions"] = False
        info["error"] = f"{type(e).__name__}: {e}"
    from torch_scatter import scatter_mean

    s = scatter_mean(torch.tensor([[1.0, 3.0, 5.0]], device="cuda"), torch.tensor([0, 0, 1], device="cuda"), dim=1)
    info["scatter_shim"] = s.tolist() == [[2.0, 5.0]]
    info["weights"] = {"megasam": WEIGHTS.is_file(), "depth_anything": DA_CKPT.is_file(),
                       "unidepth": (UNIDEPTH_DIR / "model.safetensors").is_file()}
    info["ok"] = bool(info.get("ok") and info["extensions"] and info["scatter_shim"] and all(info["weights"].values()))
    print_probe(info)


if __name__ == "__main__":
    main()
