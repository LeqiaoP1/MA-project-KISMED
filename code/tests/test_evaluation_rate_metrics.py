"""The derived / drift-robust Tier-2 metric helpers in ``evaluation.metrics``.

Four families, all pure NumPy and CPU-only:

* ``bandpass_fft`` / ``band_limited_pearson`` -- the brick-wall FFT band-pass.
  Its whole point is being phase-LINEAR and exactly zero outside the band,
  unlike the Butterworth ``smooth_band`` (ill-conditioned for a narrow low band,
  see ``test_evaluation_breakdown``). It is what makes a long, drift-dominated
  window scorable at all.
* ``single_tone_template`` -- the "beat the template" baseline: a model that
  knows only the breathing rate must not be beaten by a pure tone.
* ``bootstrap_ci`` -- the ACROSS-SUBJECT uncertainty (subjects, not clips).
* ``estimate_rate_bpm`` / ``rate_summary`` / ``rate_metrics`` -- the derived
  vital sign, in BPM.

``test_evaluation_breakdown.py`` covers ``aggregate_metrics`` /
``breakdown_by_label`` / ``smooth_band`` / ``_figure_metrics``; this file covers
the rest.
"""
import numpy as np
import pytest

from evaluation import metrics as M

FS = 100.0


def _tone(freq, n=800, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


# --------------------------------------------------------------------------- #
# bandpass_fft
# --------------------------------------------------------------------------- #
def test_bandpass_fft_passes_a_tone_in_the_band_unchanged():
    x = _tone(0.25)
    y = M.bandpass_fft(x, FS, (0.16, 0.4))
    assert np.std(y) == pytest.approx(np.std(x), rel=1e-6)
    assert float(np.corrcoef(x, y)[0, 1]) == pytest.approx(1.0, abs=1e-6)


def test_bandpass_fft_removes_dc_and_out_of_band_content():
    x = _tone(0.25) + 5.0 + _tone(5.0, amp=3.0)
    y = M.bandpass_fft(x, FS, (0.16, 0.4))
    assert abs(float(y.mean())) < 1e-12
    # only the 0.25 Hz component survives
    assert np.std(y) == pytest.approx(np.std(_tone(0.25)), rel=1e-3)


def test_bandpass_fft_is_zero_phase():
    """Linear phase -> the peak cross-correlation stays at lag 0."""
    x = _tone(0.25)
    y = M.bandpass_fft(x, FS, (0.16, 0.4))
    lags = np.arange(-20, 21)
    xc = [float(np.dot(x, np.roll(y, k))) for k in lags]
    assert lags[int(np.argmax(xc))] == 0


def test_bandpass_fft_returns_zeros_when_the_band_holds_no_bin():
    # 800 samples at 100 Hz -> 0.125 Hz bins: [0.63, 0.70) falls BETWEEN two bins
    assert not M.bandpass_fft(_tone(0.25), FS, (0.63, 0.70)).any()


def test_bandpass_fft_is_a_noop_for_none_or_a_very_short_row():
    x = _tone(0.25)
    assert np.array_equal(M.bandpass_fft(x, FS, None), x)
    short = np.ones(3)
    assert np.array_equal(M.bandpass_fft(short, FS, (0.1, 0.6)), short)


def test_bandpass_fft_handles_a_batch_with_the_last_axis_as_time():
    x = np.stack([_tone(0.25), _tone(0.3)])
    assert M.bandpass_fft(x, FS, (0.16, 0.4)).shape == x.shape
    assert np.allclose(M.bandpass_fft(x, FS, (0.16, 0.4))[0],
                       M.bandpass_fft(x[0], FS, (0.16, 0.4)))


# --------------------------------------------------------------------------- #
# band_limited_pearson
# --------------------------------------------------------------------------- #
def test_band_limited_pearson_ignores_a_dominant_slow_drift():
    """The point of the metric: a drift-dominated raw Pearson is unusable.

    The reference is the tone alone, so the prediction's slow (0.05 Hz)
    component -- far below the respiratory band -- is pure nuisance that the
    band-pass removes exactly. A 40 s window is used so 0.05 Hz falls on its own
    FFT bin instead of leaking across the band.
    """
    n = 4000
    t = np.arange(n) / FS
    tone = np.sin(2 * np.pi * 0.25 * t)
    drift = 5.0 * np.sin(2 * np.pi * 0.05 * t)
    a = (tone + drift)[None, :]
    b = tone[None, :]

    raw = M.time_domain_metrics(a, b)['pearson']
    band = M.band_limited_pearson(a, b, FS, (0.16, 0.4))

    assert band == pytest.approx(1.0, abs=1e-6)
    assert raw < 0.5, 'the drift must actually deflate the raw correlation'
    assert band > raw


def test_band_limited_pearson_is_1_for_a_scaled_and_shifted_copy():
    a = _tone(0.25)[None, :]
    assert M.band_limited_pearson(a, 3.0 * a + 7.0, FS, (0.16, 0.4)) == \
        pytest.approx(1.0, abs=1e-6)


def test_band_limited_pearson_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match='Shape mismatch'):
        M.band_limited_pearson(np.zeros((1, 100)), np.zeros((1, 99)))


# --------------------------------------------------------------------------- #
# single_tone_template
# --------------------------------------------------------------------------- #
def test_single_tone_template_recovers_the_peak_frequency_and_matches_it():
    ref = _tone(0.25)[None, :]
    tmpl, freq = M.single_tone_template(ref, FS, (0.1, 0.6))
    assert freq == pytest.approx(0.25, abs=1e-3)
    assert tmpl.shape == ref.shape
    assert M.band_limited_pearson(tmpl, ref, FS, (0.1, 0.6)) == \
        pytest.approx(1.0, abs=1e-3)


def test_single_tone_template_is_a_weak_baseline_for_noise():
    """Noise has no single tone, so the template must NOT explain it."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(1, 4000))
    tmpl, _ = M.single_tone_template(ref, FS, (0.1, 0.6))
    assert M.band_limited_pearson(tmpl, ref, FS, (0.1, 0.6)) < 0.5


def test_single_tone_template_handles_rows_shorter_than_four_samples():
    tmpl, freq = M.single_tone_template(np.ones((1, 3)), FS, (0.1, 0.6))
    assert tmpl.shape == (1, 3)
    assert np.isnan(freq)


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_the_point_and_is_seeded():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = M.bootstrap_ci(vals, n_boot=500, seed=0)
    b = M.bootstrap_ci(vals, n_boot=500, seed=0)
    assert a == b                                     # deterministic
    assert a['n'] == 5
    assert a['point'] == pytest.approx(3.0)
    assert a['lo'] <= a['point'] <= a['hi']


def test_bootstrap_ci_collapses_for_a_single_value():
    out = M.bootstrap_ci([2.0])
    assert out == {'point': 2.0, 'lo': 2.0, 'hi': 2.0, 'n': 1}


def test_bootstrap_ci_drops_nans_and_reports_n_zero_when_empty():
    out = M.bootstrap_ci([np.nan, np.inf, 4.0])
    assert out['n'] == 1 and out['point'] == pytest.approx(4.0)
    empty = M.bootstrap_ci([np.nan])
    assert empty['n'] == 0 and np.isnan(empty['point'])


# --------------------------------------------------------------------------- #
# estimate_rate_bpm
# --------------------------------------------------------------------------- #
def test_estimate_rate_bpm_reads_an_on_bin_frequency_exactly():
    # 800 samples at 100 Hz -> 0.125 Hz bins, so 0.25 Hz is bin 2
    assert M.estimate_rate_bpm(_tone(0.25)[None, :], FS)[0] == pytest.approx(15.0)


def test_estimate_rate_bpm_interpolates_between_bins():
    est = M.estimate_rate_bpm(_tone(0.30)[None, :], FS)[0] / 60.0
    assert 0.25 < est < 0.375, 'must land between the neighbouring bins'
    assert abs(est - 0.30) < abs(0.25 - 0.30), 'and be better than the bin centre'


def test_estimate_rate_bpm_is_nan_without_in_band_energy():
    """A flat / dead / railed row has NO rate -- it must not report a bin."""
    assert np.isnan(M.estimate_rate_bpm(np.zeros((1, 800)), FS)[0])
    assert np.isnan(M.estimate_rate_bpm(np.full((1, 800), -10.0), FS)[0])


def test_estimate_rate_bpm_is_nan_when_the_band_holds_no_bins():
    # 10 samples at 100 Hz -> 10 Hz bins: nothing lands in 0.1-0.6 Hz
    assert np.isnan(M.estimate_rate_bpm(_tone(0.25, n=10)[None, :], FS)[0])


def test_estimate_rate_bpm_is_nan_for_a_row_shorter_than_four_samples():
    assert np.isnan(M.estimate_rate_bpm(np.ones(3), FS)[0])


def test_estimate_rate_bpm_ignores_a_dominant_out_of_band_component():
    """The in-band tone sets the rate even under heavy 5 Hz interference."""
    x = _tone(0.25) + _tone(5.0, amp=10.0)
    assert M.estimate_rate_bpm(x[None, :], FS)[0] == pytest.approx(15.0, abs=0.5)


# --------------------------------------------------------------------------- #
# rate_summary / rate_metrics
# --------------------------------------------------------------------------- #
def test_rate_summary_is_perfect_for_identical_rates():
    out = M.rate_summary([15.0, 18.0], [15.0, 18.0])
    assert out['n'] == 2
    assert out['mae_bpm'] == pytest.approx(0.0)
    assert out['bias_bpm'] == pytest.approx(0.0)
    assert out['within_10pct'] == pytest.approx(1.0)


def test_rate_summary_flags_a_20_percent_error():
    out = M.rate_summary([18.0], [15.0])
    assert out['mae_bpm'] == pytest.approx(3.0)
    assert out['rmse_bpm'] == pytest.approx(3.0)
    assert out['bias_bpm'] == pytest.approx(3.0)          # signed
    assert out['within_10pct'] == pytest.approx(0.0)      # 3/15 = 20 %


def test_rate_summary_tolerance_key_follows_tol():
    assert 'within_20pct' in M.rate_summary([18.0], [15.0], tol=0.20)
    assert M.rate_summary([18.0], [15.0], tol=0.20)['within_20pct'] == \
        pytest.approx(1.0)


def test_rate_summary_drops_a_degenerate_reference():
    out = M.rate_summary([15.0, 18.0], [0.0, np.nan])
    assert out['n'] == 0 and np.isnan(out['mae_bpm'])


def test_rate_summary_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match='Shape mismatch'):
        M.rate_summary([15.0], [15.0, 18.0])


def test_rate_metrics_end_to_end_on_a_known_rate():
    ref = _tone(0.25, n=4000)
    pred = _tone(0.25, n=4000, phase=0.7)      # same rate, different phase
    out = M.rate_metrics(pred, ref, FS, (0.1, 0.6))
    assert out['n'] >= 1
    assert out['mae_bpm'] < 1.0
    assert out['within_10pct'] == pytest.approx(1.0)
