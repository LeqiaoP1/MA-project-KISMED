"""Stage-2 composite respiration waveform objective (ADR-008 §2.2).

    L_resp = alpha * L_MSE + beta * L_Pearson + gamma * L_spectral

Why a separate module (plan decision **D2**)
--------------------------------------------
``core/waveform_losses.py`` is the **Stage-3** objective: its ``PearsonLoss``
works on unmasked ``[B, T]`` waveforms, puts ``eps`` *outside* both norms, has no
degenerate-variance floor, and its ``alpha/beta/gamma`` already name
Huber/Pearson/MR-STFT. Reusing it here would silently change Stage-3 semantics
(forbidden by ``.github/instructions/research-python.instructions.md``) and would
give one symbol two meanings. This module owns the Stage-2 names only.

Masking semantics (§2.4 contracts 2 and 3)
------------------------------------------
In a masked autoencoder only ``mask == 1`` positions of ``pred`` are decoded; the
rest carry placeholder zeros. So:

* ``L_Pearson`` is computed **only** over the gathered masked tokens
  (``[B, n_masked * k]``), never over the zero-scattered tensor;
* ``L_spectral`` is computed on the full-clip length-``L`` waveform formed by
  splicing **detached** ground truth into the visible (``mask == 0``) positions,
  so a DFT sees a uniformly sampled grid with no rectangular-window harmonics
  and gradients flow exclusively through the masked prediction tokens.

D1 — the MSE term is passed IN
------------------------------
``CompositeRespLoss.forward`` takes the already-computed ``MaskedMSELoss`` scalar
from the caller instead of recomputing it from gathered tensors. Two reasons:
the neutral defaults must reproduce the previous loss **bit-for-bit** (a gathered
re-reduction changes the summation order), and ``mse_resp`` must stay the *same
number* in every Stage-2 arm so the ADR-007 comparator remains comparable
(ADR-008 §2.7).

FP32 (§2.4 contract 4)
----------------------
Every term casts to float32 on entry and stays there. Under ``autocast`` a
complex half-precision rFFT fails on some backends, and the correlation /
normalisation ratios lose the precision the degenerate guards rely on.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

__all__ = ['MaskedPearsonLoss', 'BandLimitedSpectralLoss', 'CompositeRespLoss']

#: stabiliser inside EACH square root of the correlation denominator (§2.2.2)
_EPS_NORM = 1e-8
#: degenerate-target variance floor, in squared target units (§2.2.2)
_VAR_FLOOR = 1e-6
#: degenerate-target band-energy floor (§2.2.3)
_ENERGY_FLOOR = 1e-8


class MaskedPearsonLoss(nn.Module):
    """Masked differentiable negative Pearson correlation (ADR-008 §2.2.2).

    ``L = mean_{b in V} (1 - r_b)`` where ``r_b`` is the Pearson correlation of
    one sample's masked prediction and masked target, and ``V`` is the set of
    samples whose target variance exceeds ``var_floor``.

    :param eps: stabiliser placed inside **each** square root of the
        denominator, so a zero-variance prediction yields a finite ``r = 0``
        (loss ``1.0``) with a finite gradient instead of ``0/0``.
    :param var_floor: a sample whose centred target energy is below this is
        EXCLUDED (a railed / disconnected sensor channel becomes exactly zero
        after the per-clip z-score, and ``r`` would be meaningless). When no
        sample qualifies the term is an explicit ``0.0``.
    """

    def __init__(self, eps: float = _EPS_NORM, var_floor: float = _VAR_FLOOR):
        super().__init__()
        self.eps = float(eps)
        self.var_floor = float(var_floor)
        if self.eps <= 0:
            raise ValueError(
                f'MaskedPearsonLoss: eps must be > 0, got {self.eps}')
        if self.var_floor < 0:
            raise ValueError(
                f'MaskedPearsonLoss: var_floor must be >= 0, got '
                f'{self.var_floor}')

    def forward(self, pred_masked: torch.Tensor,
                target_masked: torch.Tensor) -> torch.Tensor:
        """``pred_masked`` / ``target_masked``: ``[B, n_masked * k]`` (masked
        positions only, already gathered and flattened)."""
        p = pred_masked.float()
        t = target_masked.float()
        if p.dim() != 2 or t.dim() != 2:
            raise ValueError(
                f'MaskedPearsonLoss: expected [B, M_pts] inputs; got '
                f'{tuple(p.shape)} / {tuple(t.shape)}.')
        if p.shape != t.shape:
            raise ValueError(
                f'MaskedPearsonLoss: shape mismatch {tuple(p.shape)} vs '
                f'{tuple(t.shape)}.')
        zero = p.new_zeros(())
        if p.shape[-1] < 2:
            # a single sample cannot have a correlation (§2.2.2)
            return zero

        pc = p - p.mean(dim=-1, keepdim=True)
        tc = t - t.mean(dim=-1, keepdim=True)
        numerator = (pc * tc).sum(dim=-1)
        # eps INSIDE each root: a flat prediction gives sqrt(eps), not 0
        denominator = (torch.sqrt((pc * pc).sum(dim=-1) + self.eps)
                       * torch.sqrt((tc * tc).sum(dim=-1) + self.eps))
        r = numerator / denominator

        valid = (tc * tc).sum(dim=-1) > self.var_floor
        if not bool(valid.any()):
            return zero
        loss = (1.0 - r[valid]).mean()
        return loss

    def extra_repr(self) -> str:
        return f'eps={self.eps:g}, var_floor={self.var_floor:g}'


class BandLimitedSpectralLoss(nn.Module):
    """Band-limited respiratory spectral power loss (ADR-008 §2.2.3).

    Per-clip DC removal -> Hann window -> zero-padded real FFT -> band slice ->
    unit-sum power distributions -> **sum** of ``|P_hat - P_target|`` over the
    respiratory bins (total variation, bounded in ``[0, 2]``), averaged over the
    samples whose target carries band energy.

    The sum (not a mean over bins) keeps the loss invariant to the zero-padding
    factor and to the number of bins inside the passband.

    :param fs: sampling rate of the waveform (Hz) — declared, never inferred.
    :param seq_len: the full clip length ``L = n_signal * sig_kernel``. Passed
        explicitly so this module never hardcodes a geometry.
    :param n_fft: ``0`` = auto (``4 * 2**ceil(log2 L)``, i.e. 32768 at
        ``L = 8000``); otherwise must be ``>= seq_len``.
    :param f_min: lower passband edge (Hz), inclusive.
    :param f_max: upper passband edge (Hz), inclusive.
    :param eps: normalisation stabiliser for the unit-sum distributions.
    :param energy_floor: a sample whose band energy is at or below this is
        EXCLUDED (a constant target has no spectrum); ``0.0`` when none qualify.
    """

    def __init__(self, fs: float, seq_len: int, n_fft: int = 0,
                 f_min: float = 0.10, f_max: float = 0.60,
                 eps: float = _EPS_NORM, energy_floor: float = _ENERGY_FLOOR):
        super().__init__()
        self.fs = float(fs)
        self.seq_len = int(seq_len)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.eps = float(eps)
        self.energy_floor = float(energy_floor)

        if self.seq_len < 2:
            raise ValueError(
                f'BandLimitedSpectralLoss: seq_len must be >= 2, got '
                f'{self.seq_len}.')
        if not (0.0 < self.f_min < self.f_max <= self.fs / 2.0):
            raise ValueError(
                f'BandLimitedSpectralLoss: need 0 < f_min < f_max <= fs/2; got '
                f'f_min={self.f_min}, f_max={self.f_max}, fs={self.fs} '
                f'(fs/2 = {self.fs / 2.0}).')
        if self.eps <= 0:
            raise ValueError(
                f'BandLimitedSpectralLoss: eps must be > 0, got {self.eps}')
        if self.energy_floor < 0:
            raise ValueError(
                'BandLimitedSpectralLoss: energy_floor must be >= 0, got '
                f'{self.energy_floor}')

        requested = int(n_fft)
        if requested < 0:
            raise ValueError(
                f'BandLimitedSpectralLoss: n_fft must be >= 0 (0 = auto), got '
                f'{requested}.')
        if requested == 0:
            resolved = 4 * (2 ** int(math.ceil(math.log2(self.seq_len))))
        else:
            if requested < self.seq_len:
                raise ValueError(
                    f'BandLimitedSpectralLoss: n_fft={requested} is shorter '
                    f'than the clip (seq_len={self.seq_len}); the rFFT would '
                    f'wrap samples instead of resolving them.')
            resolved = requested
        self.n_fft_resolved = int(resolved)

        freqs = (torch.arange(self.n_fft_resolved // 2 + 1,
                              dtype=torch.float64)
                 * (self.fs / float(self.n_fft_resolved)))
        bins = torch.nonzero((freqs >= self.f_min) & (freqs <= self.f_max),
                             as_tuple=False).flatten()
        if bins.numel() < 1:
            raise ValueError(
                f'BandLimitedSpectralLoss: no FFT bin lies inside '
                f'[{self.f_min}, {self.f_max}] Hz with n_fft='
                f'{self.n_fft_resolved} at fs={self.fs} '
                f'(df={self.fs / self.n_fft_resolved:.6f} Hz).')
        # derived from the constructor arguments, so NOT persisted in the
        # state dict: a checkpoint must not carry geometry it cannot validate.
        self.register_buffer('band_bins', bins.to(torch.long), persistent=False)
        self.register_buffer(
            'window',
            torch.hann_window(self.seq_len, periodic=True, dtype=torch.float32),
            persistent=False)

    @property
    def band_bin_list(self):
        """The resolved passband bin indices, as Python ints."""
        return [int(i) for i in self.band_bins.tolist()]

    def forward(self, pred_full: torch.Tensor,
                target_full: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` spliced prediction and target (both already FP32-cast)."""
        p = pred_full.float()
        t = target_full.float()
        if p.dim() != 2 or p.shape != t.shape:
            raise ValueError(
                f'BandLimitedSpectralLoss: expected matching [B, L] inputs; '
                f'got {tuple(p.shape)} / {tuple(t.shape)}.')
        zero = p.new_zeros(())

        # DC removal BEFORE windowing, else the DC lobe leaks into the lowest
        # respiratory bins (§2.2.3 step 1)
        p = p - p.mean(dim=-1, keepdim=True)
        t = t - t.mean(dim=-1, keepdim=True)
        w = self.window.to(device=p.device, dtype=p.dtype)
        p_spec = torch.fft.rfft(p * w, n=self.n_fft_resolved)
        t_spec = torch.fft.rfft(t * w, n=self.n_fft_resolved)
        p_pow = p_spec.real ** 2 + p_spec.imag ** 2
        t_pow = t_spec.real ** 2 + t_spec.imag ** 2
        p_pow = p_pow.index_select(-1, self.band_bins.to(p.device))
        t_pow = t_pow.index_select(-1, self.band_bins.to(p.device))

        t_energy = t_pow.sum(dim=-1)
        valid = t_energy > self.energy_floor
        if not bool(valid.any()):
            return zero
        p_dist = p_pow / (p_pow.sum(dim=-1, keepdim=True) + self.eps)
        t_dist = t_pow / (t_energy.unsqueeze(-1) + self.eps)
        loss = (p_dist - t_dist).abs().sum(dim=-1)
        return loss[valid].mean()

    def extra_repr(self) -> str:
        return (f'fs={self.fs:g}, L={self.seq_len}, n_fft={self.n_fft_resolved}, '
                f'band=[{self.f_min:g}, {self.f_max:g}] Hz, '
                f'bins={len(self.band_bin_list)}')


class CompositeRespLoss(nn.Module):
    """``alpha * L_MSE + beta * L_Pearson + gamma * L_spectral`` (ADR-008 §2.2).

    A disabled term (weight ``0.0``) is neither evaluated nor constructed:
    ``beta = 0`` costs nothing, ``gamma = 0`` never builds an FFT. The returned
    ``pearson`` / ``spectral`` values are **unweighted** (ADR-008 §2.7 asks for
    the raw terms), while ``total`` applies the weights.

    :param mse_weight: ``alpha``.
    :param pearson_weight: ``beta``.
    :param spectral_weight: ``gamma``.
    """

    def __init__(self, *, fs: float, sig_kernel: int, n_signal: int,
                 mse_weight: float = 1.0, pearson_weight: float = 0.0,
                 spectral_weight: float = 0.0, spectral_n_fft: int = 0,
                 spectral_f_min: float = 0.10, spectral_f_max: float = 0.60,
                 spectral_eps: float = _EPS_NORM,
                 spectral_energy_floor: float = _ENERGY_FLOOR,
                 pearson_eps: float = _EPS_NORM,
                 pearson_var_floor: float = _VAR_FLOOR):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.pearson_weight = float(pearson_weight)
        self.spectral_weight = float(spectral_weight)
        for name, value in (('mse_weight', self.mse_weight),
                            ('pearson_weight', self.pearson_weight),
                            ('spectral_weight', self.spectral_weight)):
            if value < 0:
                raise ValueError(
                    f'CompositeRespLoss: {name} must be >= 0, got {value}')

        self.sig_kernel = int(sig_kernel)
        self.n_signal = int(n_signal)
        self.seq_len = self.sig_kernel * self.n_signal
        if self.sig_kernel < 1 or self.n_signal < 1:
            raise ValueError(
                f'CompositeRespLoss: sig_kernel={self.sig_kernel} and '
                f'n_signal={self.n_signal} must both be >= 1.')

        self.pearson: Optional[MaskedPearsonLoss] = (
            MaskedPearsonLoss(eps=pearson_eps, var_floor=pearson_var_floor)
            if self.pearson_weight > 0.0 else None)
        self.spectral: Optional[BandLimitedSpectralLoss] = (
            BandLimitedSpectralLoss(
                fs=fs, seq_len=self.seq_len, n_fft=spectral_n_fft,
                f_min=spectral_f_min, f_max=spectral_f_max, eps=spectral_eps,
                energy_floor=spectral_energy_floor)
            if self.spectral_weight > 0.0 else None)

    def forward(self, mse_term: torch.Tensor,
                pred_masked: Optional[torch.Tensor],
                target_masked: Optional[torch.Tensor],
                pred_full: Optional[torch.Tensor],
                target_full: Optional[torch.Tensor],
                n_masked: int) -> Dict[str, torch.Tensor]:
        """One step's composite value.

        :param mse_term: the caller's ``MaskedMSELoss(pred, target, mask)``
            scalar, computed on the FULL tensors (decision **D1**).
        :param pred_masked: ``[B, n_masked, k]`` decoded masked tokens.
        :param target_masked: ``[B, n_masked, k]`` the matching targets.
        :param pred_full: ``[B, L]`` spliced waveform (detached targets at the
            visible positions).
        :param target_full: ``[B, L]`` the full target clip.
        :param n_masked: masked token count of the step.
        :returns: ``{'total', 'mse', 'pearson', 'spectral'}``; ``mse`` is the
            UNWEIGHTED caller value, the other two are unweighted terms.
        """
        zero = mse_term.new_zeros(())
        if int(n_masked) == 0:
            return {'total': zero, 'mse': zero, 'pearson': zero,
                    'spectral': zero}

        mse = mse_term

        if self.pearson is None:
            pearson = zero
        else:
            if pred_masked is None or target_masked is None:
                raise ValueError(
                    'CompositeRespLoss: pearson_weight > 0 needs the masked '
                    'prediction/target tensors ([B, n_masked, k]).')
            pearson = self.pearson(
                pred_masked.reshape(pred_masked.shape[0], -1),
                target_masked.reshape(target_masked.shape[0], -1))

        if self.spectral is None:
            spectral = zero
        else:
            if pred_full is None or target_full is None:
                raise ValueError(
                    'CompositeRespLoss: spectral_weight > 0 needs the spliced '
                    '[B, L] prediction and the full [B, L] target.')
            spectral = self.spectral(pred_full, target_full)

        total = (self.mse_weight * mse
                 + self.pearson_weight * pearson
                 + self.spectral_weight * spectral)
        return {'total': total, 'mse': mse, 'pearson': pearson,
                'spectral': spectral}

    def extra_repr(self) -> str:
        return (f'alpha={self.mse_weight:g}, beta={self.pearson_weight:g}, '
                f'gamma={self.spectral_weight:g}, L={self.seq_len}')
