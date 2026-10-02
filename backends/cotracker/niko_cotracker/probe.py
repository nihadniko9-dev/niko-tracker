"""`python -m niko_cotracker.probe`: CUDA works, cotracker imports, the checkpoint runs on a tiny video."""

import os
from pathlib import Path

from niko_backend_common import print_probe, probe_torch_cuda

CKPT = Path(os.environ.get("NIKO_HOME", Path.home() / "niko")) / "checkpoints/cotracker3_offline/scaled_offline.pth"


def main():
    info = probe_torch_cuda()
    if not info.get("ok"):
        print_probe(info)
    import torch
    from cotracker.predictor import CoTrackerPredictor

    info["checkpoint"] = CKPT.is_file()
    if CKPT.is_file():
        model = CoTrackerPredictor(checkpoint=str(CKPT), offline=True, window_len=60).cuda().eval()
        video = torch.rand(1, 8, 3, 96, 128, device="cuda") * 255
        with torch.no_grad():
            tracks, vis = model(video, grid_size=4)
        info["tiny_run"] = list(tracks.shape) == [1, 8, 16, 2]
    info["ok"] = bool(info.get("ok") and info["checkpoint"] and info.get("tiny_run"))
    print_probe(info)


if __name__ == "__main__":
    main()
