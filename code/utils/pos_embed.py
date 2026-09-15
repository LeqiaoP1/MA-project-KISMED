"""2D/3D sinusoidal position embeddings for ViTs.

Standard MAE implementation (also present in
``tmp/MultiMAE/utils/pos_embed.py`` and inside
``tmp/MultiMAE/multimae/multimae_utils.py``) plus the 3-D (tubelet) variant
from VideoMAE (``tmp/videomae/util/pos_embed.py``).
"""
import math

import numpy as np
import torch

__all__ = ['get_1d_sincos_pos_embed', 'get_2d_sincos_pos_embed',
           'get_3d_sincos_pos_embed', 'interpolate_pos_embed']


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Return a [num_patches (+1), embed_dim] 2D sincos position embedding."""
    assert embed_dim % 2 == 0
    grid_h = grid_w = grid_size
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)   # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h.shape[0], grid_w.shape[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, length):
    """Return a [length, embed_dim] 1-D sincos position embedding (time axis)."""
    pos = np.arange(length, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])   # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])   # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)   # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """pos: (H*W,) positions; returns (H*W, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)   # (M, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Return a [Gt*Gh*Gw (+1), embed_dim] 3-D (t, h, w) sincos embedding.

    VideoMAE convention: the token order is ``(t, h, w)`` with ``w`` fastest --
    exactly the order a ``Conv3d`` tubelet embed produces when it is flattened
    with ``flatten(2)`` -- and the embedding dims are split as
    ``[time | height | width]`` with the TIME block first (so a 1-D temporal
    embedding can be placed in the same dims, see
    ``core.multimae.MultiModalMAE._init_pos_embeds``).

    :param grid_size: ``(Gt, Gh, Gw)`` = (temporal, height, width) token grid
    """
    grid_t, grid_h, grid_w = grid_size
    gt, gh, gw = np.meshgrid(np.arange(grid_t, dtype=np.float32),
                             np.arange(grid_h, dtype=np.float32),
                             np.arange(grid_w, dtype=np.float32),
                             indexing='ij')          # each (Gt, Gh, Gw)
    grid = np.stack([gt, gh, gw], axis=0)            # (3, Gt, Gh, Gw), w fastest
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """``grid``: (3, Gt, Gh, Gw) with the (t, h, w) coordinate values."""
    assert grid.shape[0] == 3, 'grid must be (3, Gt, Gh, Gw)'
    # each sub-block must stay EVEN (get_1d_sincos_pos_embed_from_grid splits it
    # into sin|cos halves), hence 2*(D//6) and not D//3: for D=1024 the plain
    # third 341 is odd and the 1-D helper would assert.
    d_t = d_h = 2 * (embed_dim // 6)
    d_w = embed_dim - 2 * d_t

    emb_t = get_1d_sincos_pos_embed_from_grid(d_t, grid[0].reshape(-1))
    emb_h = get_1d_sincos_pos_embed_from_grid(d_h, grid[1].reshape(-1))
    emb_w = get_1d_sincos_pos_embed_from_grid(d_w, grid[2].reshape(-1))
    return np.concatenate([emb_t, emb_h, emb_w], axis=1)               # (N, D)


def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate a checkpoint's pos_embed to the model's grid."""
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print(f'Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}')
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
