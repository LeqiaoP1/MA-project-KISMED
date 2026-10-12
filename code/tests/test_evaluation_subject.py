"""Subject-level evaluation primitives (2026-10-12).

Covers the pieces that make the TESTED-SUBJECT evaluation work:

* ``bandpass_fft`` / ``band_limited_pearson`` -- the drift-robust headline;
* ``single_tone_template`` -- the trivial baseline a reconstruction must beat;
* ``bootstrap_ci`` -- across-subject uncertainty;
* ``subject_of`` / ``aggregate_subjects`` -- the per-subject rollup + CI.
"""
import numpy as np
import pytest

from evaluation.metrics import (band_limited_pearson, bandpass_fft,
                                bootstrap_ci, estimate_rate_bpm, rate_metrics,
                                rate_summary, single_tone_template,
                                time_domain_metrics)
from runners.run_evaluate_subject import aggregate_subjects, subject_of


# --------------------------------------------------------------------------- #
# bandpass_fft / band_limited_pearson
# --------------------------------------------------------------------------- #
def test_bandpass_fft_keeps_inband_rejects_outband():
    fs, n = 100.0, 2000
    t = np.arange(n) / fs
    y_in = bandpass_fft(np.sin(2 * np.pi * 0.3 * t), fs, (0.1, 0.6))
    y_out = bandpass_fft(np.sin(2 * np.pi * 5.0 * t), fs, (0.1, 0.6))
    assert y_in.shape == (n,)
    assert np.max(np.abs(y_in)) > 0.9          # in-band tone survives
    assert np.max(np.abs(y_out)) < 0.05        # out-of-band tone removed


def test_bandpass_fft_none_is_noop_and_2d():
    x = np.linspace(0.0, 1.0, 40)
    assert np.allclose(bandpass_fft(x, 100.0, None), x)
    x2 = np.random.default_rng(0).standard_normal((3, 500))
    assert bandpass_fft(x2, 100.0, (0.1, 0.6)).shape == x2.shape


def test_band_limited_pearson_is_drift_robust():
    fs, n = 100.0, 2000                       # 20 s = exactly 1 cycle at 0.05 Hz
    t = np.arange(n) / fs
    target = 0.5 * np.sin(2 * np.pi * 0.3 * t + 0.4)
    # a LARGE drift BELOW the band, bin-aligned (bin 1) so it has no spectral
    # leakage into 0.1-0.6 Hz and band-limiting removes it exactly.
    drift = 20.0 * np.sin(2 * np.pi * 0.05 * t)
    pred = np.sin(2 * np.pi * 0.3 * t) + drift
    raw = time_domain_metrics(pred, target)['pearson']
    bl = band_limited_pearson(pred, target, fs, (0.1, 0.6))
    assert abs(raw) < 0.5                      # the 20x drift dominates the raw r
    assert bl > 0.9                            # cos(0.4) ~ 0.92 once drift is gone


# --------------------------------------------------------------------------- #
# single_tone_template
# --------------------------------------------------------------------------- #
def test_single_tone_template_tracks_a_clean_tone():
    fs, n = 100.0, 2000
    t = np.arange(n) / fs
    target = np.atleast_2d(np.sin(2 * np.pi * 0.3 * t))
    tmpl, f = single_tone_template(target, fs, (0.1, 0.6))
    assert tmpl.shape == target.shape
    r = band_limited_pearson(tmpl, target, fs, (0.1, 0.6))
    assert r > 0.95
    assert 0.25 <= f <= 0.35


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_the_median():
    ci = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_boot=500, seed=0)
    assert ci['n'] == 5
    assert ci['point'] == pytest.approx(0.3)
    assert ci['lo'] <= ci['point'] <= ci['hi']


def test_bootstrap_ci_single_and_empty():
    one = bootstrap_ci([0.7])
    assert one == {'point': 0.7, 'lo': 0.7, 'hi': 0.7, 'n': 1}
    empty = bootstrap_ci([])
    assert empty['n'] == 0 and np.isnan(empty['point'])
    assert np.isnan(bootstrap_ci([np.nan, np.nan])['point'])


# --------------------------------------------------------------------------- #
# subject rollup
# --------------------------------------------------------------------------- #
def test_subject_of():
    assert subject_of('F001_T7') == 'F001'
    assert subject_of('F010_T10') == 'F010'
    assert subject_of('F001') == 'F001'


def test_aggregate_subjects_headline_and_beats():
    quality = {'F001_T1': {'pearson_band': 0.5, 'template_pearson_band': 0.6},
               'F001_T2': {'pearson_band': 0.4, 'template_pearson_band': 0.6},
               'F002_T1': {'pearson_band': 0.9, 'template_pearson_band': 0.6}}
    subj = {n: subject_of(n) for n in quality}
    task = {n: n.rsplit('_', 1)[1] for n in quality}
    levels = {n: [] for n in quality}
    agg = aggregate_subjects(quality, subj, task, levels,
                             keys=('pearson_band', 'template_pearson_band'))
    assert set(agg['per_subject']) == {'F001', 'F002'}
    assert agg['per_subject']['F001']['n'] == 2
    assert agg['per_subject']['F001']['pearson_band_median'] == pytest.approx(0.45)
    assert set(agg['per_task']) == {'T1', 'T2'}
    assert agg['overall']['pearson_band']['n'] == 2
    # F001 (0.45) < template (0.6); F002 (0.9) > template (0.6) -> 1 of 2.
    assert agg['overall']['beat_template_frac_subjects'] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# derived vital sign: respiration RATE (BPM)
# --------------------------------------------------------------------------- #
def test_estimate_rate_bpm_recovers_known_rates():
    fs, n = 100.0, 3000                                  # 30 s window
    t = np.arange(n) / fs
    for bpm in (12.0, 15.0, 18.0, 24.0):                 # 0.25 Hz is NOT bin-aligned
        x = np.sin(2 * np.pi * (bpm / 60.0) * t)
        est = estimate_rate_bpm(np.atleast_2d(x), fs, (0.1, 0.6))[0]
        assert abs(est - bpm) < 0.6, (bpm, est)


def test_rate_summary_mae_and_within_10pct():
    pred = np.array([12.0, 18.0, 24.0, 11.0])
    target = np.array([12.0, 20.0, 30.0, 10.0])
    out = rate_summary(pred, target, tol=0.10)
    assert out['n'] == 4
    assert out['mae_bpm'] == pytest.approx(2.25)         # (0+2+6+1)/4
    assert out['bias_bpm'] == pytest.approx(-1.75)       # (0-2-6+1)/4
    assert out['within_10pct'] == pytest.approx(0.75)    # 3 of 4 (<= boundary)


def test_rate_summary_drops_degenerate_and_handles_empty():
    out = rate_summary([np.nan, 12.0], [10.0, np.nan], tol=0.10)
    assert out['n'] == 0 and np.isnan(out['mae_bpm'])
    empty = rate_summary([], [])
    assert empty['n'] == 0 and 'within_10pct' in empty


def test_rate_metrics_end_to_end_on_a_clean_tone():
    fs, n = 100.0, 3000
    t = np.arange(n) / fs
    target = np.sin(2 * np.pi * 0.3 * t)                 # 18 BPM
    pred = np.sin(2 * np.pi * 0.3 * t + 0.1)             # same rate, phase only
    out = rate_metrics(np.atleast_2d(pred), np.atleast_2d(target), fs, (0.1, 0.6))
    assert out['n'] == 1
    assert out['mae_bpm'] < 0.6
    assert out['within_10pct'] == 1.0
