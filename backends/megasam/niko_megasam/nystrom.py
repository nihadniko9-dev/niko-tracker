"""Stand-in for xformers.components.attention.NystromAttention (removed from recent xformers).

Reproduces xformers v0.0.24 behaviour (nystrom.py + core.py, BSD-3-Clause, Meta) for the
path UniDepthV2 exercises: UniDepth passes 4-D tensors (b, n, heads, d), and xformers takes
seq_len = k.size(-2) = heads (<= 8), which is always < num_landmarks (128). xformers then runs
dense scaled-dot-product attention over the last two dims with no mask:

    softmax((q / sqrt(d)) @ k^T, dim=-1) @ v,  followed by Dropout (a no-op in eval).

The landmark (true Nystrom) branch and key padding masks are not reproduced; they raise
instead of silently computing something different.
"""

import math

import torch
import torch.nn as nn


class NystromAttention(nn.Module):
    def __init__(self, dropout: float, num_heads: int, num_landmarks: int = 64, causal: bool = False,
                 conv_kernel_size=None, **kwargs):
        super().__init__()
        if causal or conv_kernel_size is not None or kwargs.get("v_skip_connection") is not None:
            raise NotImplementedError("only the configuration UniDepthV2 uses is reproduced")
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None, *args, **kwargs):
        if key_padding_mask is not None:
            raise NotImplementedError("key_padding_mask is not reproduced")
        seq_len = k.size(-2)
        if self.num_landmarks < seq_len:
            raise NotImplementedError("landmark branch is not reproduced (not reached by UniDepthV2)")
        att = (q / math.sqrt(k.size(-1))) @ k.transpose(-2, -1)
        att = torch.softmax(att, dim=att.ndim - 1)
        return self.attn_drop(att @ v)
