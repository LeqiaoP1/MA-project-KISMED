"""Input adapters: turn raw modality inputs into patch tokens.

Adapted from ``tmp/MultiMAE/multimae/input_adapters.py``. The pattern is:
every modality has an adapter producing a ``[B, N, D]`` token sequence that
feeds the shared transformer encoder. MultiMAE composes several of these in a
``DOMAIN_CONF``-style dict (see ``tmp/MultiMAE/run_pretraining_multimae.py``).
"""
from typing import Optional

import torch
import torch.nn as nn

__all__ = ['PatchedInputAdapter', 'SemSegInputAdapter', 'SignalInputAdapter']


class PatchedInputAdapter(nn.Module):
    """Patch-embed a dense 2D modality (RGB / depth / normals ...).

    Parameters mirror MultiMAE:
    :param num_channels: channels of the input (e.g. 3 for RGB, 1 for depth)
    :param stride_level: spatial subsampling already applied by the dataset
    :param patch_size_full: patch size at full resolution
    :param dim_tokens: token embedding dim (set later by ``init``)
    """

    def __init__(self, num_channels: int, stride_level: int = 1,
                 patch_size_full: int = 16,
                 dim_tokens: Optional[int] = None,
                 interpolate_pos_encoding: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = patch_size_full
        self.dim_tokens = dim_tokens
        self.interpolate_pos_encoding = interpolate_pos_encoding

        self.patch_embed = None        # built in init(dim_tokens)

    def init(self, dim_tokens: int):
        """Build parameters once the encoder token dim is known."""
        self.dim_tokens = dim_tokens
        kernel = self.patch_size_full * self.stride_level
        # A single conv doubles as patchify + linear projection.
        self.patch_embed = nn.Conv2d(
            self.num_channels, dim_tokens,
            kernel_size=kernel, stride=kernel)

    def forward(self, x):
        # x: [B, C, H, W] -> tokens [B, N, D]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SemSegInputAdapter(nn.Module):
    """Input adapter for per-pixel class maps (semantic segmentation).

    Simplified port of MultiMAE's ``SemSegInputAdapter``: class IDs are mapped
    to learned class embeddings, then patch-embedded into tokens.

    NOTE: MultiMAE upsamples low-res maps first and uses a separate class-embed
    conv with ``stride_level=4``; port the exact recipe from
    ``tmp/MultiMAE/multimae/input_adapters.py`` if you need that behaviour.
    """

    def __init__(self, num_classes: int, dim_class_emb: int = 64,
                 stride_level: int = 1, patch_size_full: int = 16,
                 dim_tokens: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        self.dim_class_emb = dim_class_emb
        self.stride_level = stride_level
        self.patch_size_full = patch_size_full
        self.dim_tokens = dim_tokens

        self.class_emb = None    # 1x1 conv: num_classes -> dim_class_emb
        self.patch_embed = None  # built in init(dim_tokens)

    def init(self, dim_tokens: int):
        self.dim_tokens = dim_tokens
        self.class_emb = nn.Conv2d(self.num_classes, self.dim_class_emb, 1)
        kernel = self.patch_size_full * self.stride_level
        self.patch_embed = nn.Conv2d(
            self.dim_class_emb, dim_tokens, kernel_size=kernel, stride=kernel)

    def forward(self, x):
        # x: [B, num_classes, H, W] one-hot class maps (or [B, 1, H, W] with
        #    class ids converted upstream) -> tokens [B, N, D]
        x = self.class_emb(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SignalInputAdapter(nn.Module):
    """Patch-embed a 1D physiological signal stream (BVP / RESP / EDA).

    Mirrors ``PatchedInputAdapter`` but along the time axis. A 1D convolution
    doubles as segmentation-into-windows + linear projection:
    [B, C, T] -> tokens [B, N, D], one token per ``kernel_size`` samples.
    """

    def __init__(self, num_channels: int = 1, kernel_size: int = 32,
                 stride: Optional[int] = None, dim_tokens: Optional[int] = None):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.dim_tokens = dim_tokens
        self.patch_embed = None   # built in init(dim_tokens)

    def init(self, dim_tokens: int):
        self.dim_tokens = dim_tokens
        self.patch_embed = nn.Conv1d(
            self.num_channels, dim_tokens,
            kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # x: [B, C, T] -> tokens [B, N, D]
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        return x
