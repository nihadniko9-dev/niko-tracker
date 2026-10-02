"""Minimal pure-PyTorch replacement for the two torch_scatter functions MegaSaM uses.

Avoids building the PyG extension for torch 2.10 / sm_120. Semantics follow
torch_scatter: `index` is broadcast along `dim`, missing slots are 0.
"""

import torch


def _broadcast(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % src.dim()
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(shape)
    return index.expand_as(src)


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    dim = dim % src.dim()
    idx = _broadcast(index, src, dim)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    if out is None:
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, idx, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    total = scatter_sum(src, index, dim, out, dim_size)
    ones = torch.ones_like(src)
    count = scatter_sum(ones, index, dim, None, total.shape[dim % src.dim()])
    return total / count.clamp(min=1)
