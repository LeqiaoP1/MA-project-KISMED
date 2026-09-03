"""Masking generators for masked pre-training.

Port of VideoMAE ``tmp/videomae/masking_generator.py``. Both generators return
a flat ``torch.LongTensor`` of length ``num_patches`` where ``1`` marks a token
to be *masked* (hidden) during pre-training.
"""
from typing import Dict, Iterable

import random

import numpy as np
import torch

__all__ = ['RandomMaskingGenerator', 'TubeMaskingGenerator',
           'MultiModalMaskingGenerator']


class RandomMaskingGenerator:
    """Random masking over flattened patches.

    :param input_size: spatial/temporal token grid, e.g. ``(14, 14)`` for an
        image or ``(T//tubelet, H/p, W/p)`` for video. An int is treated as a
        1-D grid of that many patches.
    :param mask_ratio: fraction of patches to mask.
    """

    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, (list, tuple)):
            input_size = [input_size]
        self.input_size = input_size
        self.num_patches = int(np.prod(input_size))
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        return (f'{self.__class__.__name__}(input_size={self.input_size}, '
                f'mask_ratio={self.num_mask / self.num_patches:.2f})')

    def __call__(self):
        mask = [0] * self.num_patches
        for idx in random.sample(range(self.num_patches), self.num_mask):
            mask[idx] = 1
        return torch.tensor(mask, dtype=torch.long)


class TubeMaskingGenerator:
    """Tube masking: mask the *same* spatial patches across all frames.

    Standard VideoMAE pre-training strategy.
    :param input_size: ``(T, H, W)`` token grid (T = frames // tubelet).
    :param mask_ratio: fraction of spatial patches to mask in every frame.
    """

    def __init__(self, input_size, mask_ratio):
        self.height, self.width, self.depth = input_size[0], input_size[1], input_size[2]
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.num_patches_per_frame * self.depth
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)

    def __repr__(self):
        return (f'{self.__class__.__name__}(input_size={[self.depth, self.height, self.width]}, '
                f'mask_ratio={self.num_masks_per_frame / self.num_patches_per_frame:.2f})')

    def __call__(self):
        mask_per_frame = [0] * self.num_patches_per_frame
        for idx in random.sample(range(self.num_patches_per_frame),
                                 self.num_masks_per_frame):
            mask_per_frame[idx] = 1
        mask = mask_per_frame * self.depth   # replicate across frames
        return torch.tensor(mask, dtype=torch.long)


class MultiModalMaskingGenerator:
    """Per-stream masking for multimodal masked pre-training (Stage 2).

    Composes one base generator per input stream, enabling the *asymmetric*
    masking of the implementation plan: visual streams (RGB / TIR) masked at
    50--75%, physiological 1D streams (BVP / RESP) masked at 90%+.

    Example::

        gen = MultiModalMaskingGenerator({
            'rgb':   TubeMaskingGenerator((T, H, W), 0.75),
            'tir':   TubeMaskingGenerator((T, H, W), 0.50),
            'bvp':   RandomMaskingGenerator(N_BVP_TOKENS, 0.90),
            'resp':  RandomMaskingGenerator(N_RESP_TOKENS, 0.95),
        })
        masks = gen()   # -> {'rgb': ..., 'tir': ..., 'bvp': ..., 'resp': ...}

    :param stream_generators: dict mapping stream name -> masking generator
        (any object with a no-arg ``__call__`` returning a token mask).
    """

    def __init__(self, stream_generators: Dict[str, object]):
        assert stream_generators, 'Provide at least one stream generator.'
        self.stream_generators = stream_generators

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'streams={list(self.stream_generators)})')

    def __call__(self):
        return {name: gen() for name, gen in self.stream_generators.items()}
