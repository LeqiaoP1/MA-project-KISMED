"""The Stage-3 ``L_time`` term: point-wise Huber (``SmoothL1``).

``core/waveform_losses.WaveformJointLoss`` owns three terms:

    L = alpha * L_time + beta * L_Pearson + gamma * L_MR-STFT

``L_time`` is ``torch.nn.SmoothL1Loss`` on the POINT-WISE error ``pred -
target``, and ``huber_beta`` sets the quadratic -> linear transition point in
TARGET-STD units. The dataset z-scores every clip, so the label has zero mean
and unit std and ``huber_beta = 1.0`` sits at ~1 std of the target.

These tests pin the properties that make a point-wise term the right choice for
a waveform, plus the cost that comes with it:

* it sees POLARITY and SHAPE, which an amplitude-only statistic cannot -- an
  inverted waveform has the same std, mean and magnitude spectrum, so anything
  blind to the point-wise difference would call it a perfect reconstruction;
* it is NOT mean-invariant and NOT lag-invariant. That is the documented cost of
  a point-wise term (a constant phase offset is charged, not forgiven) and it is
  pinned here so it cannot drift silently;
* ``huber_beta`` is really plumbed through (it changes the value on a large
  error), and a non-positive beta is a hard error;
* the degenerate-input and gradient paths stay finite -- a collapsing model is
  the realistic Stage-3 failure, so a constant prediction must not produce NaN.
"""
import pytest
import torch

from core.waveform_losses import (MultiResolutionSTFTLoss, PearsonLoss,
                                  WaveformJointLoss)

FFT_SIZES = (64, 128, 256)
T = 256


def _tone(std: float = 0.85, n: int = T, fs: float = 100.0, freq: float = 0.3,
          mean: float = 0.0, shift: int = 0, invert: bool = False,
          seed: int = 0) -> torch.Tensor:
    """A [B, T] batch shaped like a Stage-3 respiration target.

    ``std`` defaults to 0.85 -- deliberately NOT the unit std of a real label
    (the dataset z-scores every clip), so the term is exercised across a
    non-degenerate spread and ``huber_beta=1.0`` sits at ~1.2 std. ``shift`` is
    applied as an EXACT phase offset (``+2*pi*f*shift/fs``) rather than a roll,
    so a shifted signal keeps its standard deviation bit-for-bit -- which keeps
    the lag test independent of amplitude.
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
# the term itself, and that the weights compose as documented
# --------------------------------------------------------------------------- #
def test_l_time_is_a_point_wise_huber_and_repr_says_so():
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    assert isinstance(crit.time, torch.nn.SmoothL1Loss)
    assert crit.huber_beta == 1.0
    assert 'huber_beta=1.0' in repr(crit)


def test_forward_is_the_weighted_sum_of_its_three_terms():
    """alpha*Huber + beta*Pearson + gamma*MR-STFT, recomputed by hand."""
    torch.manual_seed(0)
    p = _tone(shift=3)
    t = _tone()
    crit = WaveformJointLoss(alpha=1.0, beta=1.0, gamma=1.0, fft_sizes=FFT_SIZES)
    expected = (1.0 * torch.nn.SmoothL1Loss(beta=1.0)(p, t)
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


def test_non_positive_huber_beta_is_rejected():
    with pytest.raises(ValueError) as e:
        WaveformJointLoss(fft_sizes=FFT_SIZES, huber_beta=0.0)
    assert 'huber_beta' in str(e.value)


# --------------------------------------------------------------------------- #
# what a point-wise term buys: polarity and shape
# --------------------------------------------------------------------------- #
def test_a_polarity_error_is_charged_in_full():
    """An INVERTED waveform keeps the same std, mean and magnitude spectrum, so
    an amplitude-only statistic cannot see it at all. This is the whole reason
    ``L_time`` is point-wise."""
    t = _tone()
    inverted = _tone(invert=True)
    assert torch.isclose(t.std(), inverted.std(), rtol=1e-6)
    assert WaveformJointLoss(fft_sizes=FFT_SIZES).time(inverted, t) > 0.5


def test_noise_of_the_right_amplitude_is_charged():
    """A wrong SHAPE at the correct amplitude must not pass."""
    t = _tone()
    torch.manual_seed(1)
    noise = torch.randn_like(t)
    noise = noise * (t.std() / noise.std())
    assert WaveformJointLoss(fft_sizes=FFT_SIZES).time(noise, t) > 0.3


# --------------------------------------------------------------------------- #
# ... and what it costs: no mean- or lag-invariance
# --------------------------------------------------------------------------- #
def test_a_pure_lag_is_charged_not_ignored():
    """MEASURED: a +0.25 s lag (a quarter period at 0.3 Hz) costs 0.064684.

    The charge is modest because ``beta=1.0`` puts this error size in the
    QUADRATIC region; the point is that it is NON-ZERO. A statistic that only
    looks at amplitude would report exactly zero here.
    """
    t = _tone(freq=0.3, fs=100.0)
    shifted = _tone(freq=0.3, fs=100.0, shift=25)   # +0.25 s = 1/4 period
    assert WaveformJointLoss(fft_sizes=FFT_SIZES).time(shifted, t) > 0.01


def test_a_mean_pedestal_is_charged():
    """MEASURED: a +0.9 pedestal costs 0.405000 -- the term is not
    mean-invariant."""
    t = _tone(mean=0.0)
    offset = _tone(mean=0.9)
    assert WaveformJointLoss(fft_sizes=FFT_SIZES).time(offset, t) > 0.1


def test_huber_beta_controls_how_much_outliers_are_charged():
    """beta -> larger means more of the error sits in the QUADRATIC region, so
    the same error is charged more. Both are valid losses; the assertion is
    only that ``huber_beta`` is actually plumbed."""
    p, t = _tone(shift=25), _tone()
    small = WaveformJointLoss(fft_sizes=FFT_SIZES, huber_beta=0.1)
    large = WaveformJointLoss(fft_sizes=FFT_SIZES, huber_beta=10.0)
    assert torch.isfinite(small.time(p, t))
    assert torch.isfinite(large.time(p, t))
    assert not torch.isclose(small.time(p, t), large.time(p, t))


def test_perfect_match_is_exactly_zero():
    t = _tone()
    assert WaveformJointLoss(fft_sizes=FFT_SIZES).time(t, t) == 0.0


# --------------------------------------------------------------------------- #
# robustness: degenerate inputs and gradients
# --------------------------------------------------------------------------- #
def test_gradients_are_finite_and_nonzero():
    p = _tone(shift=7).clone().requires_grad_(True)
    t = _tone()
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    crit(p, t).backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


def test_constant_prediction_does_not_nan():
    """A collapsing model is the realistic Stage-3 failure."""
    t = _tone()
    p = torch.full_like(t, 0.3).clone().requires_grad_(True)
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    loss = crit(p, t)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(p.grad).all()


def test_channel_axis_is_squeezed():
    """core.criterion-style [B, 1, T] input must match the [B, T] result."""
    t = _tone()
    p = _tone(shift=5)
    crit = WaveformJointLoss(fft_sizes=FFT_SIZES)
    assert torch.equal(crit(p.unsqueeze(1), t.unsqueeze(1)), crit(p, t))


# --------------------------------------------------------------------------- #
# resume provenance: the objective must not change silently mid-run
# --------------------------------------------------------------------------- #
def test_resume_with_a_different_huber_beta_is_reported(capsys):
    from types import SimpleNamespace

    from utils.checkpoint import check_resume_loss

    verdict = check_resume_loss(SimpleNamespace(huber_beta=2.0),
                                SimpleNamespace(huber_beta=1.0))
    assert verdict['match'] is False
    assert verdict['checkpoint_huber_beta'] == 1.0
    assert verdict['run_huber_beta'] == 2.0
    out = capsys.readouterr().out
    assert 'WARNING' in out and '1.0' in out and '2.0' in out


def test_resume_with_the_same_huber_beta_is_silent(capsys):
    from types import SimpleNamespace

    from utils.checkpoint import check_resume_loss

    verdict = check_resume_loss(SimpleNamespace(huber_beta=1.0),
                                SimpleNamespace(huber_beta=1.0))
    assert verdict['match'] is True
    assert 'WARNING' not in capsys.readouterr().out


def test_resume_check_ignores_runners_without_a_huber_beta(capsys):
    """Stage 1/2 have no ``huber_beta``; the guard must stay out of their way."""
    from types import SimpleNamespace

    from utils.checkpoint import check_resume_loss

    assert check_resume_loss(SimpleNamespace(), SimpleNamespace()) == {}
    assert check_resume_loss(SimpleNamespace(huber_beta=1.0), None) == {}
    assert capsys.readouterr().out == ''


def test_resume_check_ignores_a_checkpoint_without_the_key(capsys):
    from types import SimpleNamespace

    from utils.checkpoint import check_resume_loss

    assert check_resume_loss(SimpleNamespace(huber_beta=1.0),
                             SimpleNamespace()) == {}
    assert capsys.readouterr().out == ''
