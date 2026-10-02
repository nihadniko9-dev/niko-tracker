import os
from pathlib import Path

HOME = Path(os.environ.get("NIKO_HOME", Path.home() / "niko"))
SRC = HOME / "third_party/mega-sam"
WEIGHTS = SRC / "checkpoints/megasam_final.pth"
DA_CKPT = HOME / "checkpoints/depth_anything_v1_vitl14/checkpoints/depth_anything_vitl14.pth"
UNIDEPTH_DIR = HOME / "checkpoints/unidepth_v2_vitl14"
