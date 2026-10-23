"""The AMPLITUDE metric in ``evaluation.metrics``.

``amplitude_metrics`` compares the in-band RMS of a prediction against the
reference **after** the caller has mapped both into the same space (for the
Stage-3 respiration run that mapping is the per-session ``affine_calibrate``, so
the number is oracle-calibrated). These tests pin the three properties the
pooled k-fold report relies on:

1. an exact prediction gives a ratio of exactly 1 and zero relative error;
2. a rescaled prediction reports the scale it was rescaled by;
3. drift / out-of-band noise does NOT pass for breathing depth, and a reference
   with no in-band energy is skipped instead of dividing by ~0.
"""
import numpy as np
import pytest

from evaluation import metrics as M

FS = 100.0
BAND = (0.1, 0.6)


def _tone(freq, n=1200, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


def test_identical_series_has_unit_ratio_and_zero_error():
    # n=1000 at fs=100 puts 0.3 Hz exactly on an FFT bin, so the brick-wall
    # band-pass keeps the whole tone and the RMS is analytic.
    x = _tone(0.3, n=1000, amp=2.5)
    r = M.amplitude_metrics(x, x, FS, BAND)
    assert r['n'] == 1
    assert r['amp_ratio'] == pytest.approx(1.0, rel=1e-6)
    assert r['amp_rel_err'] == pytest.approx(0.0, abs=1e-6)
    # peak 2.5 -> RMS 2.5/sqrt(2)
    assert r['amp_ref'] == pytest.approx(2.5 / np.sqrt(2.0), rel=1e-6)


@pytest.mark.parametrize('scale', [0.5, 2.0])
def test_rescaled_prediction_reports_its_scale(scale):
    ref = _tone(0.3, amp=1.0)
    r = M.amplitude_metrics(ref * scale, ref, FS, BAND)
    assert r['amp_ratio'] == pytest.approx(scale, rel=1e-6)
    assert r['amp_rel_err'] == pytest.approx(abs(scale - 1.0), rel=1e-6)


def test_inverted_prediction_is_a_perfect_amplitude_match():
    """Amplitude is polarity-blind by design (Pearson carries the sign)."""
    ref = _tone(0.3)
    r = M.amplitude_metrics(-ref, ref, FS, BAND)
    assert r['amp_ratio'] == pytest.approx(1.0, rel=1e-6)


def test_drift_and_hf_noise_do_not_count_as_amplitude():
    n = 1200
    ref = _tone(0.3, n=n, amp=1.0)
    # both contaminants sit EXACTLY on FFT bins (1/12 Hz and 8 Hz at fs=100,
    # n=1200), so the brick-wall band-pass removes them with no leakage
    drift = 20.0 * _tone(1.0 / 12.0, n=n, amp=1.0)
    hf = 4.0 * _tone(8.0, n=n, amp=1.0)
    r = M.amplitude_metrics(ref + drift + hf, ref, FS, BAND)
    assert r['amp_ratio'] == pytest.approx(1.0, abs=1e-6)


def test_reference_with_no_in_band_energy_is_skipped():
    """A drifting / dead reference has no breathing depth to compare."""
    flat = np.full(1200, 3.0)
    r = M.amplitude_metrics(np.zeros(1200), flat, FS, BAND)
    assert r['n'] == 0
    assert np.isnan(r['amp_ratio']) and np.isnan(r['amp_rel_err'])


def test_macro_averages_over_rows_and_skips_the_dead_one():
    good = _tone(0.3, amp=1.0)
    pred = np.stack([good, 2.0 * good, np.zeros_like(good)])
    ref = np.stack([good, good, np.full_like(good, 5.0)])
    r = M.amplitude_metrics(pred, ref, FS, BAND)
    assert r['n'] == 2                     # the constant row is dropped
    assert r['amp_ratio'] == pytest.approx(1.5, rel=1e-6)
    assert r['amp_rel_err'] == pytest.approx(0.5, rel=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match='Shape mismatch'):
        M.amplitude_metrics(np.zeros((2, 10)), np.zeros((3, 10)))


def test_exported_from_the_package():
    from evaluation import amplitude_metrics
    assert amplitude_metrics is M.amplitude_metrics
