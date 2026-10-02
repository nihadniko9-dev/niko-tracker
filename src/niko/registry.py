"""Single source of truth for backends and checkpoints.

Used by `niko doctor`, scripts/fetch_checkpoints.py and docs/LICENSES.md.
`size_bytes` is the expected download size (None = not yet verified).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    key: str
    backend: str
    source: str  # "hf:<repo>" (model) or "hf-space:<repo>"
    files: tuple[str, ...]
    size_bytes: int | None
    license: str
    gated: bool = False
    revision: str | None = None


@dataclass(frozen=True)
class Backend:
    name: str
    python: str
    torch: str | None
    upstream: str
    license: str


TORCH = "2.10.0+cu128"

BACKENDS = {
    "sam3": Backend("sam3", "3.12", TORCH, "https://github.com/facebookresearch/sam3", "SAM License"),
    "cotracker": Backend("cotracker", "3.12", TORCH, "https://github.com/facebookresearch/co-tracker",
                         "CC-BY-NC 4.0"),
    "colmap": Backend("colmap", "3.12", None, "https://github.com/colmap/colmap", "BSD-3-Clause"),
    "megasam": Backend("megasam", "3.11", TORCH, "https://github.com/mega-sam/mega-sam", "Apache-2.0"),
    "da3": Backend("da3", "3.12", TORCH, "https://github.com/ByteDance-Seed/Depth-Anything-3", "Apache-2.0"),
}

GB = 1_000_000_000

CHECKPOINTS = [
    Checkpoint("sam3.1_multiplex", "sam3", "hf:facebook/sam3.1", ("sam3.1_multiplex.pt",),
               int(3.5 * GB), "SAM License", gated=True),
    Checkpoint("cotracker3_offline", "cotracker", "hf:facebook/cotracker3", ("scaled_offline.pth",),
               None, "CC-BY-NC 4.0"),
    Checkpoint("da3_nested_giant_large_1.1", "da3", "hf:depth-anything/DA3NESTED-GIANT-LARGE-1.1",
               ("model.safetensors", "config.json"), int(6.76 * GB), "CC-BY-NC 4.0"),
    # MegaSaM priors (megasam_final.pth itself ships inside the mega-sam repo, 20.8 MB)
    Checkpoint("depth_anything_v1_vitl14", "megasam", "hf-space:LiheYoung/Depth-Anything",
               ("checkpoints/depth_anything_vitl14.pth",), 1_341_401_882, "CC-BY-NC 4.0 (to verify)"),
    Checkpoint("unidepth_v2_vitl14", "megasam", "hf:lpiccinelli/unidepth-v2-vitl14",
               ("config.json", "model.safetensors"), 1_452_917_937, "CC-BY-NC 4.0 (to verify)",
               revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4"),  # revision MegaSaM's demo pins
]
