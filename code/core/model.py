"""Architecture: a patch-based Vision Transformer usable for classification
and as the shared encoder for masked pre-training.

This is a compact, self-contained starting point inspired by
``tmp/videomae/modeling_finetune.py`` (2D adaptation) and MultiMAE's
``multimae.py``. Extend it for your thesis as needed:

- multimodal / multi-task encoder with input/output adapters  -> port MultiMAE
  ``MultiMAE``/``MultiViT`` (``tmp/MultiMAE/multimae/multimae.py``)
- video (3D tubelet) patch embedding, masking + decoder        -> port VideoMAE
  ``modeling_pretrain.py`` / ``modeling_finetune.py``

Register any new variant with the ``@register_model`` decorator so it can be
built by name from YAML::

    @register_model
    def project_vit_small_patch16_224(**kwargs):
        return ProjectViT(embed_dim=384, depth=12, num_heads=6, ...)
"""
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .blocks import Block, trunc_normal_
from .registry import register_model

__all__ = ['ProjectViT',
           'project_vit_small_patch16_224',
           'project_vit_base_patch16_224',
           'project_vit_large_patch16_224',
           'project_vit_huge_patch16_224']


class ProjectViT(nn.Module):
    """Vision Transformer.

    :param in_chans: input channels
    :param num_classes: number of output classes (0 -> feature-only, no head)
    :param output_len: if > 0, build a regression head that maps the [CLS]
        token to a 1D sequence of ``output_len`` samples (Stage-3 waveform
        head baseline). Use a lightweight conv decoder instead for better
        temporal resolution (see docs/ImplementationPlan.md Stage 3).
    :param embed_dim / depth / num_heads / mlp_ratio: transformer geometry
    :param patch_size / input_size: patchify geometry (2D)
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 1000,
                 output_len: int = 0,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0., norm_layer=nn.LayerNorm,
                 patch_size: int = 16, input_size: int = 224):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.output_len = output_len
        self.depth = depth
        self.input_size = input_size

        grid = input_size // patch_size
        self.grid_size = grid
        self.num_patches = grid * grid

        # patchify + linear projection (single conv)
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if num_classes <= 0 and output_len > 0:
            # waveform regression head (Stage 3): [CLS] -> sequence of samples
            self.head = nn.Linear(embed_dim, output_len)

        trunc_normal_(self.cls_token, std=.02)
        self._init_pos_embed()
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    def _init_pos_embed(self):
        from utils.pos_embed import get_2d_sincos_pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches ** 0.5),
            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return self.depth

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # ------------------------------------------------------------------ #
    def forward_features(self, x):
        x = self.patch_embed(x)                       # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)              # [B, N, D]

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)         # [B, N+1, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, 0]                                   # take [CLS] token
        x = self.head(x)
        return x

    def forward_waveform(self, x):
        """Regression path: returns a 1D waveform [B, output_len].

        NOTE: a single [CLS]-token linear head is a *baseline*; prefer a
        lightweight decoder over all patch tokens for fine-grained signals
        (see docs/ImplementationPlan.md Stage 3).
        """
        return self.forward(x)


# --------------------------------------------------------------------------- #
# Registered entrypoints (timm-style naming)
# --------------------------------------------------------------------------- #
@register_model
def project_vit_small_patch16_224(**kwargs):
    return ProjectViT(embed_dim=384, depth=12, num_heads=6, **kwargs)


@register_model
def project_vit_base_patch16_224(**kwargs):
    return ProjectViT(embed_dim=768, depth=12, num_heads=12, **kwargs)


@register_model
def project_vit_large_patch16_224(**kwargs):
    return ProjectViT(embed_dim=1024, depth=24, num_heads=16, **kwargs)


@register_model
def project_vit_huge_patch16_224(**kwargs):
    return ProjectViT(embed_dim=1280, depth=32, num_heads=16, **kwargs)
