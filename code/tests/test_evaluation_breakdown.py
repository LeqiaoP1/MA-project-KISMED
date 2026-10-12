"""Evaluation helpers added / fixed on 2026-10-12.

Covers three pieces:

* ``evaluation.metrics.aggregate_metrics`` / ``breakdown_by_label`` -- the
  per-task / per-distortion-group breakdown of the session metrics;
* ``runners.run_evaluate_session.smooth_band`` -- the SOS rewrite that fixes the
  ill-conditioned ``(b, a)`` recursion on the narrow respiratory band;
* ``runners.run_evaluate_session._figure_metrics`` / ``_session_quality`` -- the
  figure title (and the group breakdown) must use SPACE-CONSISTENT numbers
  (pearson + volts-calibrated mae/rmse), NOT the cross-space session mae/rmse
  that merely track the reference DC offset.
"""
import numpy as np
import pytest

from evaluation.metrics import aggregate_metrics, breakdown_by_label
from runners.run_evaluate_session import (_figure_metrics, _session_quality,
                                          _task_of_name, smooth_band)


# --------------------------------------------------------------------------- #
# aggregate_metrics
# --------------------------------------------------------------------------- #
def test_aggregate_metrics_median_iqr_mean_n():
    recs = [{'pearson': 0.1}, {'pearson': 0.3}, {'pearson': 0.5}]
    out = aggregate_metrics(recs, ['pearson'])
    assert out['n'] == 3
    assert out['pearson_median'] == pytest.approx(0.3)
    assert out['pearson_mean'] == pytest.approx(0.3)
    assert out['pearson_iqr'] == pytest.approx(0.2)      # 75th - 25th


def test_aggregate_metrics_drops_nan_per_key():
    recs = [{'pearson': 0.4}, {'pearson': float('nan')}, {'pearson': 0.6},
            {'pearson': 0.8, 'psd_mae': float('nan')}]
    out = aggregate_metrics(recs, ['pearson', 'psd_mae'])
    assert out['n'] == 4
    assert out['pearson_median'] == pytest.approx(0.6)   # NaN dropped
    assert np.isnan(out['psd_mae_median'])               # no finite value at all


def test_aggregate_metrics_without_mean():
    out = aggregate_metrics([{'x': 1.0}], ['x'], with_mean=False)
    assert 'x_mean' not in out
    out_nan = aggregate_metrics([{'x': float('nan')}], ['x'], with_mean=False)
    assert np.isnan(out_nan['x_median']) and np.isnan(out_nan['x_iqr'])


# --------------------------------------------------------------------------- #
# breakdown_by_label
# --------------------------------------------------------------------------- #
def test_breakdown_single_label():
    quality = {'F001_T3': {'pearson': 0.2}, 'F002_T3': {'pearson': 0.4},
               'F003_T7': {'pearson': 0.9}}
    label = {'F001_T3': 'T3', 'F002_T3': 'T3', 'F003_T7': 'T7'}
    out = breakdown_by_label(quality, label, ['pearson'])
    assert set(out) == {'T3', 'T7'}
    assert out['T3']['n'] == 2
    assert out['T3']['pearson_median'] == pytest.approx(0.3)
    assert out['T7']['pearson_median'] == pytest.approx(0.9)


def test_breakdown_multi_label_and_unmapped_skipped():
    quality = {'s1': {'pearson': 0.5}, 's2': {'pearson': 0.7},
               's3': {'pearson': 1.0}}
    labels = {'s1': ['low', 'all'], 's2': ['high']}   # s3 has no mapping
    out = breakdown_by_label(quality, labels, ['pearson'])
    assert out['low']['n'] == 1 and out['high']['n'] == 1
    assert out['all']['pearson_median'] == pytest.approx(0.5)
    # s1 is counted in BOTH its labels; s2 in one -> 3 memberships total.
    assert sum(v['n'] for v in out.values()) == 3
    # s3 (the only 1.0) is unmapped, so no bucket may contain it.
    assert max(v['pearson_median'] for v in out.values()) == pytest.approx(0.7)


def test_breakdown_empty_input():
    assert breakdown_by_label({}, {}, ['pearson']) == {}


# --------------------------------------------------------------------------- #
# smooth_band (SOS regression)
# --------------------------------------------------------------------------- #
def test_smooth_band_narrow_resp_band_is_stable():
    # regression for the (b, a) blow-up: a UNIT 0.3 Hz tone through the narrow
    # resp band must stay O(1), not ~1e8 (measured 2026-10-10).
    fs, n = 100.0, 800
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.3 * t)
    y = smooth_band(x, fs, (0.16, 0.4))
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) < 5.0


def test_smooth_band_none_is_noop():
    x = np.linspace(0.0, 1.0, 50)
    assert np.allclose(smooth_band(x, 100.0, None), x)


def test_smooth_band_lowpass_and_2d_and_short():
    fs, n = 100.0, 400
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    y = smooth_band(x, fs, (0.7,))                 # low-pass keeps the 0.3 Hz
    assert np.max(np.abs(y)) < 5.0

    x2 = np.random.default_rng(0).standard_normal((3, 300))
    y2 = smooth_band(x2, fs, (0.1, 0.6))           # [N, T], axis=-1
    assert y2.shape == x2.shape and np.isfinite(y2).all()

    short = np.array([1.0, 2.0])                   # < 3 samples -> unchanged
    assert np.allclose(smooth_band(short, fs, (0.1, 0.6)), short)


# --------------------------------------------------------------------------- #
# space-consistent figure / group metrics
# --------------------------------------------------------------------------- #
def test_figure_metrics_excludes_crossspace_mae():
    rec = {'session_tier1': {'pearson': 0.5, 'mae': 4.5, 'rmse': 6.0},
           'session_tier1_affine': {'mae': 0.2, 'rmse': 0.3, 'pearson': 0.5},
           'session_tier2': {'psd_mae': 0.15}}
    m = _figure_metrics(rec)
    assert m['pearson'] == pytest.approx(0.5)
    assert m['mae_aff_V'] == pytest.approx(0.2)
    assert m['rmse_aff_V'] == pytest.approx(0.3)
    assert m['psd_mae'] == pytest.approx(0.15)
    assert 'mae' not in m and 'rmse' not in m       # cross-space excluded


def test_session_quality_flattens_and_drops_crossspace():
    rec = {'session_tier1': {'pearson': 0.5, 'mae': 4.5},
           'session_tier1_affine': {'mae': 0.2, 'rmse': 0.3},
           'session_tier2': {'psd_mae': 0.15, 'dominant_freq_error_hz': 0.02}}
    q = _session_quality(rec)
    assert q['pearson'] == pytest.approx(0.5)
    assert q['mae_affine'] == pytest.approx(0.2)
    assert q['rmse_affine'] == pytest.approx(0.3)
    assert q['psd_mae'] == pytest.approx(0.15)
    assert q['dominant_freq_error_hz'] == pytest.approx(0.02)
    assert 'mae' not in q and 'rmse' not in q


def test_task_of_name():
    assert _task_of_name('F001_T7') == 'T7'
    assert _task_of_name('F010_t10') == 'T10'
    assert _task_of_name('no_task_here') == ''
