"""Masked losses used during (multi-task) masked pre-training.

Adapted from ``tmp/MultiMAE/multimae/criterion.py``. Contract used by the
scaffold engines:

    loss_fn(pred: [B, N, D], target: [B, N, D] | [B, N] labels, mask: [B, N])

where ``mask == 1`` marks tokens whose loss is kept (i.e. the *masked* /
predicted tokens). ``N`` includes padding so callers must build targets with
the same token grid and pass a mask covering the real (non-padded) tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MaskedMSELoss', 'MaskedL1Loss', 'MaskedCrossEntropyLoss']

_EPS = 1e-6


class MaskedMSELoss(nn.Module):
    """Masked mean-squared-error over per-token predictions."""

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.norm_pix = norm_pix

    def forward(self, pred, target, mask):
        # pred/target: [B, N, D]; mask: [B, N] in {0, 1}
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                       # per-token
        loss = (loss * mask).sum() / (mask.sum() + _EPS)
        return loss


class MaskedL1Loss(nn.Module):
    """Masked mean-absolute-error over per-token predictions."""

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.norm_pix = norm_pix

    def forward(self, pred, target, mask):
        loss = (pred - target).abs()
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + _EPS)
        return loss


class MaskedCrossEntropyLoss(nn.Module):
    """Masked cross-entropy over per-token class logits.

    ``pred`` is [B, N, num_classes], ``target`` is [B, N] class indices and
    ``mask`` is [B, N]. Padded (mask == 0) tokens are ignored.
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, label_smoothing: float = 0.0):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.label_smoothing = label_smoothing

    def forward(self, pred, target, mask):
        B, N, C = pred.shape
        pred = pred.reshape(B * N, C)
        target = target.reshape(B * N)
        mask = mask.reshape(B * N).float()
        loss = F.cross_entropy(
            pred, target, reduction='none', label_smoothing=self.label_smoothing)
        loss = (loss * mask).sum() / (mask.sum() + _EPS)
        return loss
