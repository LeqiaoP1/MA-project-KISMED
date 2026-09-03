"""Stage-3 waveform supervision losses (Spatio-Temporal-Spectral joint loss).

Implements the unified loss from the implementation plan
(``docs/ImplementationPlan.md``):

    L_joint = alpha * L_time + beta * L_Pearson + gamma * L_MR-STFT

* ``L_time``      -- point-wise amplitude error (L1 over time samples)
* ``L_Pearson``   -- negative Pearson correlation (temporal phase-locking)
* ``L_MR-STFT``   -- multi-resolution spectral error (FFT windows 64 / 128 / 256)

All losses operate on 1D waveforms of shape ``[B, T]`` or ``[B, 1, T]``
(a leading channel axis is squeezed). Targets/predictions should be
z-normalised / detrended upstream if scale invariance is desired.
"""
from typing import List

import torch
import torch.nn as nn

__all__ = ['PearsonLoss', 'MultiResolutionSTFTLoss', 'WaveformJointLoss']

_EPS = 1e-8


def _to_1d(x: torch.Tensor) -> torch.Tensor:
    """Squeeze an optional channel axis: [B, 1, T] -> [B, T]."""
    if x.dim() == 3:
        x = x[:, 0]
    return x


class PearsonLoss(nn.Module):
    """Negative Pearson correlation between predicted and target waveforms."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = _to_1d(pred)
        t = _to_1d(target)
        p = p - p.mean(dim=-1, keepdim=True)
        t = t - t.mean(dim=-1, keepdim=True)
        denom = torch.norm(p, dim=-1) * torch.norm(t, dim=-1)
        r = (p * t).sum(dim=-1) / (denom + _EPS)
        return (1.0 - r).mean()


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution spectral loss over several FFT window sizes.

    For every window in ``fft_sizes`` a short-time Fourier transform is taken
    and the error between magnitudes is penalised with both spectral
    convergence (scale-invariant) and log-magnitude L1 terms.
    """

    def __init__(self, fft_sizes: List[int] = (64, 128, 256),
                 hop_ratio: float = 0.25):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_ratio = hop_ratio

    def _magnitude(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        hop = max(1, int(n_fft * self.hop_ratio))
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, return_complex=True)
        return stft.abs()   # [B, n_fft // 2 + 1, frames]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = _to_1d(pred)
        t = _to_1d(target)
        total = torch.tensor(0.0, device=p.device, dtype=p.dtype)
        for n_fft in self.fft_sizes:
            p_mag = self._magnitude(p, n_fft)
            t_mag = self._magnitude(t, n_fft)
            # spectral convergence term
            sc = (torch.norm(p_mag - t_mag, dim=(-1, -2))
                  / (torch.norm(t_mag, dim=(-1, -2)) + _EPS)).mean()
            # log-magnitude L1 term
            lm = torch.mean(torch.abs(
                torch.log(p_mag + 1e-6) - torch.log(t_mag + 1e-6)))
            total = total + sc + lm
        return total / len(self.fft_sizes)


class WaveformJointLoss(nn.Module):
    """Combined temporal + Pearson + multi-resolution STFT loss.

    :param alpha: weight of the L1 time-domain loss
    :param beta: weight of the negative Pearson correlation
    :param gamma: weight of the multi-resolution STFT loss
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 gamma: float = 1.0, fft_sizes=(64, 128, 256)):
        super().__init__()
        self.register_buffer('_l1_w', torch.tensor(alpha))
        self.register_buffer('_pearson_w', torch.tensor(beta))
        self.register_buffer('_stft_w', torch.tensor(gamma))
        self.l1 = nn.L1Loss()
        self.pearson = PearsonLoss()
        self.stft = MultiResolutionSTFTLoss(fft_sizes=fft_sizes)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask=None) -> torch.Tensor:
        # ``mask`` accepted for API parity with core.criterion losses (unused).
        p = _to_1d(pred)
        t = _to_1d(target)
        loss = (self._l1_w * self.l1(p, t)
                + self._pearson_w * self.pearson(p, t)
                + self._stft_w * self.stft(p, t))
        return loss

    def extra_repr(self):
        return (f'alpha={float(self._l1_w)}, beta={float(self._pearson_w)}, '
                f'gamma={float(self._stft_w)}')
