"""Port MegaSaM (upstream: py3.10 / torch 2.0.1 / CUDA 11.8) to torch 2.10 + CUDA 12.8 + sm_120.

    python port.py <mega-sam checkout>

Idempotent, prints every change. Edits only the checkout in $NIKO_HOME/third_party.
"""

import re
import sys
from pathlib import Path


def edit(path: Path, pattern: str, repl: str, why: str, count_expected: int | None = None) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, repl, text)
    if n:
        path.write_text(new)
    print(f"[port] {path.relative_to(ROOT)}: {n} change(s) - {why}")
    if count_expected is not None and n not in (0, count_expected):
        raise SystemExit(f"unexpected number of matches in {path}: {n}")


ROOT = Path(sys.argv[1]).resolve()
base = ROOT / "base"

# 1. Let TORCH_CUDA_ARCH_LIST (12.0) pick the GPU arch instead of the hardcoded sm_70..sm_86.
edit(base / "setup.py", r"\n\s*(?:# )?'-gencode=arch=compute_\d+,code=sm_\d+',", "",
     "drop hardcoded -gencode (sm_70-86); arch comes from TORCH_CUDA_ARCH_LIST=12.0")

# 2. Deprecated Tensor::type() in AT_DISPATCH macros -> scalar_type().
for f in list((base / "src").glob("*.c*")) + list((base / "thirdparty/lietorch/lietorch/src").glob("*.c*")):
    # covers AT_DISPATCH_*(x.type(), ...) and lietorch's DISPATCH_GROUP_AND_FLOATING_TYPES(g, x.type(), ...)
    edit(f, r"(\b\w*DISPATCH\w*\(\s*(?:[\w\.]+,\s*)?)([\w\.\[\]]+)\.type\(\)", r"\1\2.scalar_type()",
         "DISPATCH(x.type()) -> DISPATCH(x.scalar_type())")

# 3. Depth-Anything v1: the checkpoint holds the full model (loaded strict=True), so don't
#    download DINOv2 pretrained weights (~1.2 GB) through torch.hub.
edit(ROOT / "Depth-Anything/depth_anything/dpt.py",
     r"torch\.hub\.load\(\s*'facebookresearch/dinov2', 'dinov2_\{:\}14'\.format\(encoder\)\s*\)",
     "torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder), pretrained=False)",
     "DINOv2 backbone code only, weights come from depth_anything_vitl14.pth")

# 3b. UniDepth weights from our checkpoint folder (same pinned revision, downloaded once).
edit(ROOT / "UniDepth/scripts/demo_mega-sam.py",
     r'UniDepthV2\.from_pretrained\("lpiccinelli/unidepth-v2-vitl14"',
     'UniDepthV2.from_pretrained(os.environ.get("NIKO_UNIDEPTH_DIR", "lpiccinelli/unidepth-v2-vitl14")',
     "load UniDepth from $NIKO_UNIDEPTH_DIR when set")

# 3c. UniDepth's visualization module imports wandb (training logs only).
edit(ROOT / "UniDepth/unidepth/utils/visualization.py", r"(?m)^import wandb\n",
     "try:\n    import wandb\nexcept ImportError:  # only used for training logs\n    wandb = None\n",
     "optional wandb import")

# 4. UniDepthV2's depth head uses xformers.components NystromAttention, removed from recent
#    xformers (and xformers is not installed here). Use the exact re-implementation of the
#    path UniDepthV2 takes (niko_megasam/nystrom.py).
nys = ROOT / "UniDepth/unidepth/layers/nystrom_attention.py"
fallback = ("except ImportError:  # removed from recent xformers: exact stand-in, see niko_megasam/nystrom.py\n"
            "    from niko_megasam.nystrom import NystromAttention\n")
edit(nys, r"(?m)^from xformers\.components\.attention import NystromAttention\n",
     "try:\n    from xformers.components.attention import NystromAttention\n" + fallback,
     "xformers NystromAttention -> niko_megasam.nystrom stand-in")
edit(nys, r"except ImportError:  # removed in recent xformers; unused by UniDepthV2\n    NystromAttention = None\n",
     fallback, "upgrade the earlier None fallback to the stand-in")
