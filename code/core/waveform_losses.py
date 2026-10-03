"""Stage-3 waveform supervision losses (Spatio-Temporal-Spectral joint loss).

Implements the unified loss from the implementation plan
(``docs/ImplementationPlan.md``):

    L_joint = alpha * L_time + beta * L_Pearson + gamma * L_MR-STFT

* ``L_time``      -- STANDARD-DEVIATION (amplitude) error between the predicted
                     and target waveforms, ``|std(pred) - std(target)|``.
                     It REPLACES the original point-wise L1 term (2026-09-28) so
                     that a pure pulse-transit-time lag is no longer charged, and
                     so that an amplitude error is penalised ~19 % harder. It does
                     NOT change the gradient balance (both terms measure
                     ``1/sqrt(N)``) and it is blind to shape and polarity -- read
                     the measured caveats in :class:`StdLoss` before relying on it.
* ``L_Pearson``   -- negative Pearson correlation (temporal phase-locking)
* ``L_MR-STFT``   -- multi-resolution spectral error (FFT windows 64 / 128 / 256)

All losses operate on 1D waveforms of shape ``[B, T]`` or ``[B, 1, T]``
(a leading channel axis is squeezed). ``L_time`` is the only SCALE-DEPENDENT term
(it owns the amplitude); ``L_Pearson`` is scale- and shift-invariant, and
``L_MR-STFT`` compares magnitude spectra. With the Stage-3 per-clip
``signal_norm: zscore`` target the target std is ~1.0 by construction, so
``L_time`` reduces to ``|std(pred) - 1|``.
"""
from typing import List

import torch
import torch.nn as nn

__all__ = ['PearsonLoss', 'StdLoss', 'MultiResolutionSTFTLoss',
           'WaveformJointLoss']

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


def _std(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Per-sample POPULATION standard deviation, with ``eps`` under the sqrt.

    ``torch.std(constant, unbiased=False)`` is 0 and its backward is ``0 / 0``,
    i.e. **NaN** -- and a collapsing model is precisely the case that walks
    toward a constant output. Adding ``eps`` INSIDE the sqrt keeps the forward
    value (``sqrt(var + 1e-8)`` differs from ``sqrt(var)`` by ~5e-9 at var ~ 1)
    and makes the backward well-defined, so a truly constant input yields a ZERO
    gradient instead of a NaN.

    ``unbiased=False`` (divide by N) matches the target normalisation used
    upstream: ``data.rgb_roi_dataset.RGBRoiFinetuneDataset._normalize_target``
    and ``data.paired_dataset`` both divide by the numpy population std, so a
    ``signal_norm: zscore`` target measures std ~ 1.0 here.
    """
    d = x - x.mean(dim=-1, keepdim=True)
    return torch.sqrt((d * d).mean(dim=-1) + eps)


class StdLoss(nn.Module):
    """Standard-deviation (amplitude) loss -- the Stage-3 ``L_time`` term.

    ``|std(pred) - std(target)|``, averaged over the batch. It constrains the
    AMPLITUDE only; phase and shape are left to :class:`PearsonLoss` (global
    phase) and :class:`MultiResolutionSTFTLoss` (magnitude spectrum).

    WHY REPLACE L1 -- what the swap DOES change (all measured 2026-09-28,
    1.2 Hz BP-like z-scored target, N = 496 @ 100 Hz):

    * It stops charging for a point-wise TIME SHIFT, which is the one error the
      model cannot remove: the visual pulse lags the pressure trace by the pulse
      transit time (``code/refined_RGB_BP_study.md`` §8). A PERFECT waveform delayed
      by 0.20 s (~1/4 cardiac cycle at 1.2 Hz) cost the old L1 ``1.2868`` and now
      costs ``0.0000`` -- L1 was pushing the optimiser toward a compromise shape
      for an error it could never fix.
    * It penalises an amplitude error ~19 % harder across the board: 10x too
      small ``0.8991`` vs ``0.7545``, 3x too big ``1.9980`` vs ``1.6767``. That
      is the axis the collapsed branch actually got wrong (``pred_std`` 1.7 % of
      target, ``ab-attribution.md``).
    * It degrades sanely on a flat target (``0.0000`` for a flat prediction),
      where :class:`PearsonLoss` adds a CONSTANT 1.0 with no gradient.

    *** WHAT THE SWAP DOES *NOT* DO -- corrected by measurement ***
    The original rationale ("a std term's gradient grows like ``1/std(pred)``, so
    it fights the collapse harder than L1") is WRONG. Under a PROPORTIONAL
    collapse ``pred = c * target`` the numerator ``pred - mean`` shrinks in step
    with the denominator ``std(pred)``, so the ``1/std`` cancels. MEASURED: the
    gradient norm is ``1/sqrt(N) = 0.0449`` for BOTH terms at every ``c``, i.e.
    an IDENTICAL **1.81 %** share of the composite gradient norm. The swap does
    NOT rebalance alpha against gamma -- that still needs an explicit ``alpha``
    (or a re-tuned ``clip_grad``), since gamma holds ~63-84 % of the norm.

    *** THE COST ***
    The term is now blind to SHAPE and POLARITY. Measured against the same
    target: white noise with the right standard deviation costs ``0.0007`` (the
    old L1 charged ``1.125``), an inverted waveform ``0.0000``, a 0.20 s lag
    ``0.0000``. Rejecting a wrong-SHAPED prediction is now the job of ``beta``
    alone, and Pearson is scale-invariant and cosine-flat near r = 1.

    :param reduction: ``'l1'`` (default, ``|d|``) or ``'l2'`` (``d ** 2``). With
        the per-clip ``signal_norm: zscore`` target (measures std 0.999) both
        reduce to ``|std(pred) - 1|``.
    """

    def __init__(self, reduction: str = 'l1'):
        super().__init__()
        if reduction not in ('l1', 'l2'):
            raise ValueError(f"reduction must be 'l1' or 'l2'; got {reduction!r}")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = _std(_to_1d(pred)) - _std(_to_1d(target))
        return d.abs().mean() if self.reduction == 'l1' else (d * d).mean()

    def extra_repr(self):
        return f"reduction='{self.reduction}'"


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
    ``grad_norm`` of ~2.5e9 (``TirROI_Resp_plan.md`` §7.5), which is fatal
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
            print(f'[stft] WARNING: {n_skipped} sample-window(s) with a '
                  f'(near-)constant target excluded from the MR-STFT term '
                  f'(||T|| ~ 0 makes the scale-invariant ratio undefined; a '
                  f'railed/dead sensor channel, or a flat baseline window). '
                  f'Check the target data -- see '
                  f'TirROI_Resp_plan.md §7.5.')
        return total / len(self.fft_sizes)


class WaveformJointLoss(nn.Module):
    """Combined standard-deviation + Pearson + multi-resolution STFT loss.

    :param alpha: weight of the standard-deviation (amplitude) time loss
    :param beta: weight of the negative Pearson correlation
    :param gamma: weight of the multi-resolution STFT loss
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 gamma: float = 1.0, fft_sizes=(64, 128, 256)):
        super().__init__()
        self.register_buffer('_std_w', torch.tensor(alpha))
        self.register_buffer('_pearson_w', torch.tensor(beta))
        self.register_buffer('_stft_w', torch.tensor(gamma))
        self.std = StdLoss()
        self.pearson = PearsonLoss()
        self.stft = MultiResolutionSTFTLoss(fft_sizes=fft_sizes)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask=None) -> torch.Tensor:
        # ``mask`` accepted for API parity with core.criterion losses (unused).
        p = _to_1d(pred)
        t = _to_1d(target)
        loss = (self._std_w * self.std(p, t)
                + self._pearson_w * self.pearson(p, t)
                + self._stft_w * self.stft(p, t))
        return loss

    def extra_repr(self):
        return (f'alpha={float(self._std_w)}, beta={float(self._pearson_w)}, '
                f'gamma={float(self._stft_w)}')
