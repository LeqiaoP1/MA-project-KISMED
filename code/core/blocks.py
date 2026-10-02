"""Transformer building blocks.

Adapted from ``tmp/MultiMAE/multimae/multimae_utils.py`` and the MAE /
timm codebases (BSD-3). See `core/model.py` for how these are assembled.
"""
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                      # torch >= 2.0; fused, memory-efficient attention
    from torch.nn.functional import scaled_dot_product_attention as _sdpa
except ImportError:                                # pragma: no cover
    _sdpa = None

__all__ = [
    'trunc_normal_', 'drop_path', 'DropPath', 'Mlp', 'Attention', 'Block',
    'CrossAttention', 'DecoderBlock',
]


# --------------------------------------------------------------------------- #
# init helpers
# --------------------------------------------------------------------------- #
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.', stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# --------------------------------------------------------------------------- #
# stochastic depth
# --------------------------------------------------------------------------- #
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)   # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()   # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --------------------------------------------------------------------------- #
# FFN + attention
# --------------------------------------------------------------------------- #
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, usually ``act(GELU)``."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with a fused qkv projection."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each [B, num_heads, N, head_dim]

        # Fused (memory-efficient) attention. Mathematically IDENTICAL to
        # ``(q @ k.T * scale).softmax(-1) @ v`` (SDPA applies the same
        # 1/sqrt(head_dim) scale), but it never materialises the
        # [B, heads, N, N] score matrix. That matrix is what makes the dense
        # 224 px geometry impossible: the MultiMAE decoder concatenates the
        # encoded visible tokens with ``mask_token`` up to the FULL n_visual
        # (9800 tokens at 224 px), i.e. 2 x 12 x 9800^2 x 4 B ~= 9 GB per
        # block per sample on the naive path. Falls back to the naive form on a
        # torch without SDPA.
        if _sdpa is not None:
            x = _sdpa(q, k, v,
                      dropout_p=self.attn_drop.p if self.training else 0.0)
        else:                                          # pragma: no cover
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block: LayerNorm -> Attention -> LayerNorm -> MLP.

    Port the full CrossAttention / DecoderBlock variants from
    ``tmp/MultiMAE/multimae/multimae_utils.py`` if your decoder needs them.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, shall we disable during eval?
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from ``x``, keys/values from ``context``.

    Ported from ``tmp/MultiMAE/multimae/multimae_utils.py`` (``CrossAttention``).
    This is the piece the scaffold's :class:`Block` docstring told us to port
    "if your decoder needs them": it lets a target-driven query sequence attend
    to a DIFFERENT sequence (here, learned response-time queries attending to the
    whole visual encoder output), which plain self-attention cannot express.

    Uses fused SDPA when available, exactly like :class:`Attention`, so the
    ``[B, heads, N, M]`` score matrix is never materialised on the fast path.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]   # each [B, num_heads, M, head_dim]

        if _sdpa is not None:
            x = _sdpa(q, k, v,
                      dropout_p=self.attn_drop.p if self.training else 0.0)
        else:                                          # pragma: no cover
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    """Self-attention + cross-attention + MLP (MultiMAE ``DecoderBlock``).

    ``x`` is the query sequence (the decoder's own tokens); ``context`` is the
    encoder output it cross-attends to. Ported from
    ``tmp/MultiMAE/multimae/multimae_utils.py`` (``DecoderBlock``) so the
    Asymmetric Cross-MAE RESP decoder can query the FULL visual encoder output
    instead of only its own stream's visible tokens.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, context):
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        x = x + self.drop_path(
            self.cross_attn(self.query_norm(x), self.context_norm(context)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
