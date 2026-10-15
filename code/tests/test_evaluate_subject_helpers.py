"""Pure helpers of ``runners.run_evaluate_subject`` (the subject-level report).

The subject runner is the one that answers "does the model work on a PERSON"
rather than on a clip: it scores each session, reduces to one number per
subject, and reports the across-SUBJECT uncertainty. Everything tested here is
CPU-only and free of file I/O -- ``_evaluate_session`` (which assembles a
session) is covered by ``test_assemble_session`` instead.
"""
import numpy as np
import pytest

from runners.run_evaluate_subject import (_AGG_KEYS, _figure_metrics, _flat,
                                          _median_session, _window_rates,
                                          aggregate_subjects, subject_of)

FS = 100.0
BAND = (0.16, 0.4)


# --------------------------------------------------------------------------- #
# subject_of
# --------------------------------------------------------------------------- #
def test_subject_of_splits_a_subject_task_name():
    assert subject_of('F001_T7') == 'F001'
    assert subject_of('M058_T10') == 'M058'


def test_subject_of_falls_back_to_the_whole_name():
    assert subject_of('F001') == 'F001'
    assert subject_of('') == ''


# --------------------------------------------------------------------------- #
# _median_session
# --------------------------------------------------------------------------- #
def test_median_session_picks_the_middle_by_value():
    pairs = [('a', 0.1), ('b', 0.5), ('c', 0.9)]
    assert _median_session(pairs) == 'b'


def test_median_session_is_order_independent():
    pairs = [('c', 0.9), ('a', 0.1), ('b', 0.5)]
    assert _median_session(pairs) == 'b'


def test_median_session_treats_nan_as_the_lowest():
    pairs = [('a', np.nan), ('b', 0.5), ('c', 0.9)]
    assert _median_session(pairs) == 'b'


# --------------------------------------------------------------------------- #
# _flat -- the grouping view
# --------------------------------------------------------------------------- #
def test_flat_keeps_only_space_consistent_finite_metrics():
    rec = {
        'tier1': {'pearson': 0.5},
        'tier2': {'psd_mae': 0.1, 'dominant_freq_error_hz': np.nan,
                  'hf_power_rel': 3e-4},
        'affine': {'mae': 1.0, 'rmse': 1.5},
        'rate': {'mae_bpm': 2.0, 'within_10pct': 0.8},
    }
    out = _flat(rec)
    assert out['pearson'] == pytest.approx(0.5)
    assert out['psd_mae'] == pytest.approx(0.1)
    assert out['hf_power_rel'] == pytest.approx(3e-4)
    assert out['mae_affine'] == pytest.approx(1.0)
    assert out['rmse_affine'] == pytest.approx(1.5)
    assert out['mae_bpm'] == pytest.approx(2.0)
    assert out['within_10pct'] == pytest.approx(0.8)
    assert 'dominant_freq_error_hz' not in out, 'NaN must be dropped'
    # the RAW cross-space mae/rmse must never be aggregated
    assert 'mae' not in out and 'rmse' not in out


def test_flat_survives_empty_and_missing_blocks():
    assert _flat({}) == {}
    assert _flat({'tier1': None, 'tier2': {}, 'affine': None}) == {}


# --------------------------------------------------------------------------- #
# _figure_metrics
# --------------------------------------------------------------------------- #
def test_figure_metrics_prefers_band_limited_pearson_and_volts_calibrated_mae():
    rec = {'pearson_band': 0.71, 'tier1': {'pearson': 0.4, 'mae': 99.0},
           'tier2': {'psd_mae': 0.2}, 'affine': {'mae': 1.1, 'rmse': 1.6}}
    out = _figure_metrics(rec)
    assert out['pearson_band'] == pytest.approx(0.71)
    assert out['pearson'] == pytest.approx(0.4)
    assert out['mae_aff_V'] == pytest.approx(1.1)
    assert out['rmse_aff_V'] == pytest.approx(1.6)
    assert out['psd_mae'] == pytest.approx(0.2)
    assert 'mae' not in out, 'the cross-space tier1 mae must not be titled'


def test_figure_metrics_omits_non_finite_entries():
    rec = {'pearson_band': np.nan, 'tier1': {'pearson': 0.4},
           'tier2': {}, 'affine': {}}
    assert set(_figure_metrics(rec)) == {'pearson'}


# --------------------------------------------------------------------------- #
# aggregate_subjects
# --------------------------------------------------------------------------- #
def _fixture():
    quality = {
        'S1_T1': {'pearson_band': 0.5, 'template_pearson_band': 0.6},
        'S1_T2': {'pearson_band': 0.7, 'template_pearson_band': 0.6},
        'S2_T1': {'pearson_band': 0.9, 'template_pearson_band': 0.6},
    }
    subj = {'S1_T1': 'S1', 'S1_T2': 'S1', 'S2_T1': 'S2'}
    task = {'S1_T1': 'T1', 'S1_T2': 'T2', 'S2_T1': 'T1'}
    levels = {'S1_T1': ['high'], 'S1_T2': ['high', 'moderate'], 'S2_T1': ['high']}
    return quality, subj, task, levels


def test_aggregate_subjects_reduces_sessions_to_subjects():
    quality, subj, task, levels = _fixture()
    agg = aggregate_subjects(quality, subj, task, levels, _AGG_KEYS)
    assert set(agg) == {'per_subject', 'per_task', 'per_group', 'overall'}
    assert agg['per_subject']['S1']['n'] == 2
    assert agg['per_subject']['S1']['pearson_band_median'] == pytest.approx(0.6)
    assert agg['per_subject']['S2']['pearson_band_median'] == pytest.approx(0.9)


def test_aggregate_subjects_counts_a_session_in_every_group_it_belongs_to():
    quality, subj, task, levels = _fixture()
    agg = aggregate_subjects(quality, subj, task, levels, _AGG_KEYS)
    assert agg['per_group']['high']['n'] == 3
    assert agg['per_group']['moderate']['n'] == 1
    assert agg['per_task']['T1']['n'] == 2


def test_aggregate_subjects_overall_carries_the_across_subject_ci():
    quality, subj, task, levels = _fixture()
    agg = aggregate_subjects(quality, subj, task, levels, _AGG_KEYS)
    ov = agg['overall']
    assert ov['pearson_band']['n'] == 2, 'one observation per SUBJECT'
    assert ov['pearson_band']['point'] == pytest.approx(0.75)
    assert ov['pearson_band']['lo'] <= ov['pearson_band']['point']
    assert ov['pearson_band']['hi'] >= ov['pearson_band']['point']
    # S1's subject median TIES the 0.6 template and a tie is NOT a win, so only
    # S2 beats it -> 1 of 2 subjects.
    assert ov['beat_template_frac_subjects'] == pytest.approx(0.5)


def test_aggregate_subjects_reports_zero_when_the_template_wins():
    quality = {'S1_T1': {'pearson_band': 0.1, 'template_pearson_band': 0.9}}
    agg = aggregate_subjects(quality, {'S1_T1': 'S1'}, {'S1_T1': 'T1'},
                             {'S1_T1': ['low']}, _AGG_KEYS)
    assert agg['overall']['beat_template_frac_subjects'] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _window_rates
# --------------------------------------------------------------------------- #
def test_window_rates_splits_a_session_into_fixed_windows():
    """A 4 s window puts 0.25 Hz exactly ON an FFT bin at 100 Hz.

    (A 10 s window -> 0.1 Hz bins would straddle it and the parabolic sub-bin
    fit undershoots by ~10 %, which is the estimator's known short-window bias.)
    """
    t = np.arange(4000) / FS                     # 40 s
    sig = np.sin(2 * np.pi * 0.25 * t)           # 15 BPM throughout
    pw, tw = _window_rates(sig, sig, FS, BAND, 4.0)

    assert pw.shape == tw.shape == (10,)
    np.testing.assert_allclose(pw, tw)
    np.testing.assert_allclose(pw, 15.0, atol=1e-6)


def test_window_rates_scores_a_short_session_as_one_window():
    t = np.arange(300) / FS                      # 3 s < one 4 s window
    sig = np.sin(2 * np.pi * 0.25 * t)
    pw, tw = _window_rates(sig, sig, FS, BAND, 4.0)
    assert pw.shape == (1,) and tw.shape == (1,)


def test_window_rates_returns_nothing_for_a_signal_shorter_than_four_samples():
    pw, tw = _window_rates(np.ones(3), np.ones(3), FS, BAND, 4.0)
    assert pw.size == 0 and tw.size == 0
