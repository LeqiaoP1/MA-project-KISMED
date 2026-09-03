"""Output adapters: turn encoder/decoder tokens into task predictions.

Adapted from ``tmp/MultiMAE/multimae/output_adapters.py``.

The scaffold ships a token-level ``SpatialOutputAdapter`` that reconstructs one
per-token prediction (used with the masked MSE/L1 losses in ``criterion.py``).
For pixel-space / segmentation heads (``DPTOutputAdapter``, ``ConvNeXtAdapter``,
``SegmenterMaskTransformerAdapter``, ``LinearOutputAdapter``) port the full
versions from ``tmp/MultiMAE/multimae/output_adapters.py``.
"""
from typing import Optional

import torch
import torch.nn as nn

__all__ = ['SpatialOutputAdapter']


class SpatialOutputAdapter(nn.Module):
    """Per-token output head for masked reconstruction.

    :param num_channels: number of output channels per token
        (e.g. 3 for RGB reconstruction, num_classes for per-token logits)
    :param dim_tokens_enc: encoder token dim (set by ``init``)
    :param task: task/modality name, used only for bookkeeping
    """

    def __init__(self, num_channels: int, task: Optional[str] = None,
                 dim_tokens_enc: Optional[int] = None, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.task = task
        self.dim_tokens_enc = dim_tokens_enc
        self.head = None   # built in init(dim_tokens_enc)

    def init(self, dim_tokens_enc: int):
        self.dim_tokens_enc = dim_tokens_enc
        # token-level projection: [B, N, D] -> [B, N, num_channels]
        self.head = nn.Linear(dim_tokens_enc, self.num_channels)

    def forward(self, x):
        # x: [B, N, D] -> [B, N, num_channels]
        return self.head(x)
