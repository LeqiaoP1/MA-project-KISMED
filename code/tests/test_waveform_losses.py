"""The Stage-3 ``L_time`` term: ``std`` (default) vs ``smoothl1``.

``core/waveform_losses.WaveformJointLoss`` owns three terms:

    L = alpha * L_time + beta * L_Pearson + gamma * L_MR-STFT

Historically ``L_time`` was a point-wise L1 on ``pred - target``; on 2026-09-28 it
was replaced by :class:`StdLoss` (``|std(pred) - std(target)|``) because the
point-wise form charged the optimiser for a constant PHASE OFFSET it can never
remove (a perfect waveform delayed by 0.20 s cost L1 ``1.2868`` and StdLoss
``0.0000``). The cost of that swap is that L_time became blind to SHAPE and
POLARITY, which is exactly what the Stage-3 respiration figures show being
missed.

``time_loss='smoothl1'`` buys that sensitivity back (at the price of re-charging
the phase offset), so these tests pin BOTH halves of the trade:

* the default is unchanged and numerically IDENTICAL to the old formula, so no
  existing run silently changes value;
* the two terms really do behave differently where it matters (a time shift, an
  inverted/noise-shaped waveform, a non-zero target mean) -- otherwise the A/B
  would be a no-op;
* the degenerate-input and gradient paths stay finite.
"""
import numpy as np
import pytest
import torch

from core.waveform_losses import (MultiResolutionSTFTLoss, PearsonLoss, StdLoss,
                                  WaveformJointLoss)

FFT_SIZES = (64, 128, 256)
T = 256


def _tone(std: float = 0.85, n: int = T, fs: float = 100.0, freq: float = 0.3,
          mean: float = 0.0, shift: int = 0, invert: bool = False,
          seed: int = 0) -> torch.Tensor:
    """A [B, T] batch shaped like a Stage-3 session-z-scored respiration target.

    ``std`` defaults to the ~0.85 measured on the real run, so ``huber_beta=1.0``
    sits at ~1 std as the docstring claims. ``shift`` is applied as an EXACT phase
    offset (``+2*pi*f*shift/fs``) rather than a roll, so a shifted signal keeps
    its standard deviation bit-for-bit -- which is what makes it a fair test of
    StdLoss's lag-invariance.
    """
    t = torch.arange(n, dtype=torch.float64) / fs
    phase = 2 * torch.pi * freq * (shift / fs)
    if invert:
        phase = phase + torch.pi
    sig = torch.sin(2 * torch.pi * freq * t + phase)
    sig = sig * (std / sig.std())
    sig = sig + mean
    return sig.unsqueeze(0).repeat(2, 1)[:, :n]


# --------------------------------------------------------------------------- #
# the default must not change any existing run
# --------------------------------------------------------------------------- #
def test_default_time_loss_is_std_and_repr_says_so():
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    assert crit.time_loss == 'std'
    assert isinstance(crit.time, StdLoss)
    assert "time_loss='std'" in repr(crit)


def test_default_forward_is_bit_identical_to_the_original_formula():
    """alpha*Std + beta*Pearson + gamma*MR-STFT, recomputed by hand.

    Guards the whole point of making the term selectable: the default path must
    not drift numerically.
    """
    torch.manual_seed(0)
    p = _tone(shift=3)
    t = _tone()
    crit = WaveformJointLoss(alpha=1.0, beta=1.0, gamma=1.0, fft_sizes=FFT_SIZES)
    expected = (1.0 * StdLoss()(p, t)
                + 1.0 * PearsonLoss()(p, t)
                + 1.0 * MultiResolutionSTFTLoss(fft_sizes=FFT_SIZES)(p, t))
    assert torch.equal(crit(p, t), expected)


def test_weights_still_scale_their_own_terms():
    p, t = _tone(std=0.5), _tone()          # a pure AMPLITUDE error
    base = WaveformJointLoss(fft_sizes=FFT_SIZES)(p, t)
    no_time = WaveformJointLoss(alpha=0.0, fft_sizes=FFT_SIZES)(p, t)
    assert not torch.isclose(base, no_time)
    assert torch.isclose(no_time,
                         PearsonLoss()(p, t)
                         + MultiResolutionSTFTLoss(fft_sizes=FFT_SIZES)(p, t))


# --------------------------------------------------------------------------- #
# selection + validation
# --------------------------------------------------------------------------- #
def test_smoothl1_is_selected_and_reported():
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='smoothl1',
                             huber_beta=0.5)
    assert crit.time_loss == 'smoothl1'
    assert isinstance(crit.time, torch.nn.SmoothL1Loss)
    assert crit.huber_beta == 0.5
    assert 'huber_beta=0.5' in repr(crit)


@pytest.mark.parametrize('alias', ['huber', 'smoothl1', 'SmoothL1', ' smoothl1 '])
def test_aliases_and_case_normalise_to_smoothl1(alias):
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss=alias)
    assert crit.time_loss == 'smoothl1'


def test_unknown_time_loss_is_a_hard_error_naming_the_choices():
    with pytest.raises(ValueError) as e:
        WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='l1')
    msg = str(e.value)
    assert 'std' in msg and 'smoothl1' in msg and "'l1'" in msg


def test_non_positive_huber_beta_is_rejected():
    with pytest.raises(ValueError) as e:
        WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='smoothl1',
                          huber_beta=0.0)
    assert 'huber_beta' in str(e.value)


def test_huber_beta_is_ignored_by_the_std_path():
    """A bad beta must not break a std run (it is simply not read)."""
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES, huber_beta=0.0)
    assert isinstance(crit.time, StdLoss)


# --------------------------------------------------------------------------- #
# the actual behavioural difference (this is why the A/B is worth running)
# --------------------------------------------------------------------------- #
def test_smoothl1_sees_the_shape_and_polarity_that_std_is_blind_to():
    """MEASURED: inverted -> std 0.000000 / smoothl1 1.126436;
    white noise at the same std -> std 0.032885 / smoothl1 0.585573."""
    t = _tone()
    std_crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    hub_crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='smoothl1')

    # an INVERTED waveform has the SAME std, so StdLoss cannot see it at all
    inverted = _tone(invert=True)
    assert std_crit.time(inverted, t) < 1e-6
    assert hub_crit.time(inverted, t) > 0.5

    # white noise of the right amplitude likewise
    torch.manual_seed(1)
    noise = torch.randn_like(t)
    noise = noise * (t.std() / noise.std())
    assert std_crit.time(noise, t) < 0.05
    assert hub_crit.time(noise, t) > 0.3


def test_the_two_terms_win_on_DIFFERENT_error_types():
    """MEASURED at huber_beta=1.0 (i.e. ~1 target std, so the QUADRATIC regime):
    a pure amplitude error is caught ~5x harder by std (0.349316 vs 0.065442),
    while polarity/noise are caught only by smoothl1 (1.126436 / 0.585573 vs
    ~0). So the A/B is NOT a one-sided upgrade -- smoothl1 buys shape and
    polarity at the cost of amplitude sensitivity, which is why beta (and
    possibly alpha) matter.
    """
    t = _tone()
    std_crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    hub_crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='smoothl1')

    amp = _tone(std=0.5)                      # amplitude error only
    assert std_crit.time(amp, t) > 3 * hub_crit.time(amp, t)

    inv = _tone(invert=True)                  # polarity error only
    assert hub_crit.time(inv, t) > 100 * std_crit.time(inv, t)


def test_smoothl1_reintroduces_the_phase_offset_charge_std_removed():
    """The documented COST of the swap, pinned so it cannot regress silently.

    MEASURED: a +0.25 s (quarter-cycle at 0.3 Hz) lag costs std 0.000000 and
    smoothl1 0.064684. The smoothl1 figure is modest because beta=1.0 puts it in
    the QUADRATIC region for this error size -- it is a real charge, not a large
    one; the point is that it is nonzero where std is bit-exactly zero.
    """
    t = _tone(freq=0.3, fs=100.0)              # 0.3 Hz
    shifted = _tone(freq=0.3, fs=100.0, shift=25)   # +0.25 s = 1/4 period
    std_crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    hub_crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='smoothl1')
    assert std_crit.time(shifted, t) < 1e-6, 'std must stay lag-invariant'
    assert hub_crit.time(shifted, t) > 0.01


def test_smoothl1_is_not_mean_invariant_but_std_is():
    t = _tone(mean=0.0)
    offset = _tone(mean=0.9)                   # a session-norm per-clip pedestal
    std_crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    hub_crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='smoothl1')
    assert std_crit.time(offset, t) < 1e-6
    assert hub_crit.time(offset, t) > 0.1


def test_huber_beta_controls_how_much_outliers_are_charged():
    p, t = _tone(shift=25), _tone()
    small = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='smoothl1',
                              huber_beta=0.1)
    large = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss='smoothl1',
                              huber_beta=10.0)
    # beta -> larger means more of the error sits in the quadratic region.
    # Both are valid losses; the point is only that beta is actually plumbed.
    assert torch.isfinite(small.time(p, t)) and torch.isfinite(large.time(p, t))
    assert not torch.isclose(small.time(p, t), large.time(p, t))


def test_perfect_match_is_zero_for_both_terms():
    t = _tone()
    assert WaveformJointLoss(fft_sizes=FFT_SIZES).time(t, t) < 1e-9
    assert WaveformJointLoss(fft_sizes=FFT_SIZES,
                             time_loss='smoothl1').time(t, t) == 0.0


# --------------------------------------------------------------------------- #
# robustness: degenerate inputs and gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('time_loss', ['std', 'smoothl1'])
def test_gradients_are_finite_and_nonzero(time_loss):
    p = _tone(shift=7).clone().requires_grad_(True)
    t = _tone()
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss=time_loss)
    crit(p, t).backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


@pytest.mark.parametrize('time_loss', ['std', 'smoothl1'])
def test_constant_prediction_does_not_nan(time_loss):
    """A collapsing model is the realistic failure; StdLoss needed an eps for
    this (``std(constant)`` is 0 and its backward is 0/0)."""
    t = _tone()
    p = torch.full_like(t, 0.3).clone().requires_grad_(True)
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss=time_loss)
    loss = crit(p, t)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize('time_loss', ['std', 'smoothl1'])
def test_channel_axis_is_squeezed_for_both(time_loss):
    """core.criterion-style [B, 1, T] input must match the [B, T] result."""
    t = _tone()
    p = _tone(shift=5)
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES, time_loss=time_loss)
    assert torch.equal(crit(p.unsqueeze(1), t.unsqueeze(1)), crit(p, t))
