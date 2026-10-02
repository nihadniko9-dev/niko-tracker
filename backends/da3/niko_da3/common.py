import os
from pathlib import Path

CKPT_DIR = Path(os.environ.get("NIKO_HOME", Path.home() / "niko")) / "checkpoints/da3_nested_giant_large_1.1"


def load_model():
    import torch
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(str(CKPT_DIR))
    return model.to(device=torch.device("cuda")).eval()
