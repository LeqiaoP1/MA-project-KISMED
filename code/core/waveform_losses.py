"""Stage-3 waveform supervision losses (Spatio-Temporal-Spectral joint loss).

Implements the unified loss of the Stage-3 design
(``code/notes/design/SolutionC_EarlyTokenMixing_stage3.md``; plan entry in
``code/notes/plan/experiments/RGB_BP_plan.md`` §4):

    L_joint = alpha * L_time + beta * L_Pearson + gamma * L_MR-STFT

* ``L_time``      -- HUBER (``torch.nn.SmoothL1Loss``) on the POINT-WISE error
                     ``pred - target``. ``huber_beta`` sets the transition
                     point in target-std units (quadratic inside the band,
                     linear beyond). It is shape-, polarity- and lag-sensitive
                     and NOT mean-invariant, so a constant phase offset IS
                     charged -- it is the only term that sees POLARITY, i.e. an
                     inverted waveform costs the full error.
* ``L_Pearson``   -- negative Pearson correlation (temporal phase-locking)
* ``L_MR-STFT``   -- multi-resolution spectral error (FFT windows 64 / 128 / 256)

All losses operate on 1D waveforms of shape ``[B, T]`` or ``[B, 1, T]``
(a leading channel axis is squeezed). ``L_time`` is the only SCALE-DEPENDENT term
(it owns the amplitude); ``L_Pearson`` is scale- and shift-invariant, and
``L_MR-STFT`` compares magnitude spectra. The dataset z-scores every target with
its own mean/std (``data.tir_resp_dataset.TARGET_NORM``), so the label has ZERO
mean and UNIT std and ``huber_beta = 1.0`` sits at ~1 std of the target.
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

    DEGENERATE-TARGET GUARD. The spectral-convergence term is
    ``||P - T|| / (||T|| + eps)``, i.e. SCALE-INVARIANT in the target, so it is
    **undefined when the target has no spectral energy at all**: a CONSTANT
    target (a disconnected/railed sensor channel, a flat baseline window)
    becomes exactly zero after the usual per-clip z-scoring upstream, and the
    ratio then explodes to ``||P|| / eps``. Measured on a real railed
    respiration clip (flat at ``-10.0000 V``) vs a normal one, same prediction::

        normal target   ||T|| 343 / 455 / 613   sc 1.21 / 1.24 / 1.29
        flat target     ||T||   0 /   0 /   0   sc 2.8e10 / 4.0e10 / 5.9e10

    ONE such sample in a batch of 4 produced ``4.3e10`` of loss and a pre-clip
    ``grad_norm`` of ~2.5e9, which is fatal
    without gradient clipping. Samples whose target has (near-)zero energy are
    therefore EXCLUDED from the term, and when none qualify the term contributes
    0 instead of a garbage value. For any batch in which every sample is valid
    -- the normal case -- the returned value is BIT-IDENTICAL to the unguarded
    formula, so no existing result changes.

    :param target_norm_floor: a sample counts as "no energy" when
        ``||T|| <= target_norm_floor * max_batch(||T||)``, i.e. relative to the
        batch's own energy scale (1e-6 sits ~6 orders of magnitude below the
        spread seen in practice).
    """

    def __init__(self, fft_sizes: List[int] = (64, 128, 256),
                 hop_ratio: float = 0.25, target_norm_floor: float = 1e-6):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_ratio = hop_ratio
        self.target_norm_floor = float(target_norm_floor)
        self._warned_degenerate = False

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
        n_skipped = 0
        for n_fft in self.fft_sizes:
            p_mag = self._magnitude(p, n_fft)
            t_mag = self._magnitude(t, n_fft)
            t_norm = torch.norm(t_mag, dim=(-1, -2))
            # a SCALE-INVARIANT ratio needs a non-zero reference; the floor is
            # relative to the batch, so it works on any input scale
            valid = t_norm > self.target_norm_floor * float(t_norm.detach().max())
            if not bool(valid.any()):
                n_skipped += int(t_norm.numel())
                continue
            if bool(valid.all()):
                # ---- unguarded path: kept byte-for-byte for the normal case --
                sc = (torch.norm(p_mag - t_mag, dim=(-1, -2))
                      / (t_norm + _EPS)).mean()
                lm = torch.mean(torch.abs(
                    torch.log(p_mag + 1e-6) - torch.log(t_mag + 1e-6)))
            else:
                n_skipped += int((~valid).sum())
                # spectral convergence term
                sc = (torch.norm(p_mag - t_mag, dim=(-1, -2))[valid]
                      / (t_norm[valid] + _EPS)).mean()
                # log-magnitude L1 term (also meaningless against a zero target)
                lm = torch.mean(torch.abs(
                    torch.log(p_mag[valid] + 1e-6)
                    - torch.log(t_mag[valid] + 1e-6)))
            total = total + sc + lm
        if n_skipped and not self._warned_degenerate:
            self._warned_degenerate = True
            # NOTE: message text is frozen so runtime output is byte-identical;
            # the current pointer is eval/tir_resp/resp_rail_touch_filter_results.md.
            print(f'[stft] WARNING: {n_skipped} sample-window(s) with a '
                  f'(near-)constant target excluded from the MR-STFT term '
                  f'(||T|| ~ 0 makes the scale-invariant ratio undefined; a '
                  f'railed/dead sensor channel, or a flat baseline window). '
                  f'Check the target data -- see '
                  f'TirROI_Resp_plan.md §7.5.')
        return total / len(self.fft_sizes)


class WaveformJointLoss(nn.Module):
    """Combined point-wise Huber + Pearson + multi-resolution STFT loss.

    :param alpha: weight of the ``L_time`` term
    :param beta: weight of the negative Pearson correlation
    :param gamma: weight of the multi-resolution STFT loss
    :param huber_beta: the ``SmoothL1Loss`` transition point, in TARGET-STD
        units. The label is z-scored per clip, so its std is ~1 and the default
        1.0 is ~1 std: quadratic for the bulk of the error, linear for outliers.
        Lower it for a heavier (L1-like) charge on large errors.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 gamma: float = 1.0, fft_sizes=(64, 128, 256),
                 huber_beta: float = 1.0):
        super().__init__()
        self.register_buffer('_time_w', torch.tensor(alpha))
        self.register_buffer('_pearson_w', torch.tensor(beta))
        self.register_buffer('_stft_w', torch.tensor(gamma))
        self.huber_beta = float(huber_beta)
        if self.huber_beta <= 0:
            raise ValueError(
                'WaveformJointLoss: huber_beta must be > 0, got '
                f'{self.huber_beta}')
        self.time = nn.SmoothL1Loss(beta=self.huber_beta, reduction='mean')
        self.pearson = PearsonLoss()
        self.stft = MultiResolutionSTFTLoss(fft_sizes=fft_sizes)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask=None) -> torch.Tensor:
        # ``mask`` accepted for API parity with core.criterion losses (unused).
        p = _to_1d(pred)
        t = _to_1d(target)
        loss = (self._time_w * self.time(p, t)
                + self._pearson_w * self.pearson(p, t)
                + self._stft_w * self.stft(p, t))
        return loss

    def extra_repr(self):
        return (f'alpha={float(self._time_w)}, '
                f'beta={float(self._pearson_w)}, '
                f'gamma={float(self._stft_w)}, huber_beta={self.huber_beta}')
