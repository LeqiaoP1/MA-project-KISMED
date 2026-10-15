"""Session assembly: coverage masking (fix 2026-10-15).

``evaluation.assemble.assemble_session`` used to ZERO-FILL every sample no clip
covered, and to divide by ``max(wsum, eps)``. Two consequences, both measured on
the real Stage-3 run:

* a session whose first SURVIVING clip starts after ``t = 0`` (windows are
  dropped for a sentinel landmark line, a rail touch or a flat window) began
  with a flat-zero head. 33 of 273 val sessions had one, the worst padding
  82.6 % of its own timeline, and the artificial zeros moved one session's
  Pearson by 0.18;
* at the outermost sample of a LONE window the Hann weight is exactly 0, so
  ``acc / max(wsum, eps)`` returned 0 there too -- punching a fake sample into
  the first and last position of every single-clip session.

The assembler now returns a COVERAGE MASK and ``nan`` outside it, and the two
session runners slice to the covered samples before scoring.
"""
import argparse
import json
import os

import numpy as np
import pytest

from evaluation.assemble import assemble_session
from runners.run_evaluate_session import main as session_main


# --------------------------------------------------------------------------- #
# coverage of the timeline
# --------------------------------------------------------------------------- #
def test_uncovered_prefix_is_nan_not_zero():
    """A lone clip starting at t=8 s must leave 0-8 s EMPTY, not flat-zero.

    This is the M030_T4 shape: one clip, ``t_start`` 8.0 s, reference 16 s.
    """
    fs = 100.0
    clip = np.linspace(-1.0, 1.0, 800)            # 8 s at 100 Hz
    t, y, cov = assemble_session(clip[None, :], [800], fs)

    assert t.size == 1600 and y.size == 1600 and cov.size == 1600
    assert not cov[:800].any(), 'uncovered prefix must not be marked covered'
    assert cov[800:].all()
    assert np.isnan(y[:800]).all(), 'uncovered samples must be nan, not 0'
    assert np.isfinite(y[800:]).all()


def test_coverage_is_full_when_the_clip_starts_at_zero():
    fs = 100.0
    clip = np.ones(800)
    _, y, cov = assemble_session(clip[None, :], [0], fs)
    assert cov.all()
    assert np.isfinite(y).all()


def test_empty_input_returns_empty_with_a_bool_mask():
    t, y, cov = assemble_session(np.zeros((0, 800)), np.zeros(0), 100.0)
    assert t.size == 0 and y.size == 0 and cov.size == 0
    assert cov.dtype == bool


def test_offsets_count_mismatch_raises():
    with pytest.raises(ValueError, match='2 predictions but 1 offsets'):
        assemble_session(np.zeros((2, 10)), np.zeros(1, dtype=int), 100.0)


# --------------------------------------------------------------------------- #
# the Hann taper at a lone window's edges
# --------------------------------------------------------------------------- #
def test_single_clip_edges_are_not_zeroed():
    """``hanning(L)[0] == [-1] == 0`` must not punch a 0 into the clip's ends."""
    fs = 100.0
    clip = np.linspace(-2.0, 3.0, 800)            # starts and ends non-zero
    _, y, cov = assemble_session(clip[None, :], [0], fs)

    assert cov.all()
    # weight-sum normalisation is exact where w > 0; the two zero-weight edge
    # samples fall back to the unweighted value.
    np.testing.assert_allclose(y, clip)
    assert y[0] == pytest.approx(-2.0)
    assert y[-1] == pytest.approx(3.0)


def test_overlapping_identical_clips_cross_fade_is_a_no_op():
    """Where two windows carry the SAME waveform the weight sum must cancel.

    The shift is one full PERIOD (400 samples = 0.25 Hz at 100 Hz), so the two
    windows agree sample-by-sample in the overlap; with any other shift they
    would legitimately disagree and the cross-fade would average them.
    """
    fs = 100.0
    period = 400
    clip = np.sin(2 * np.pi * np.arange(800) / period) + 5.0
    _, y, cov = assemble_session(np.stack([clip, clip]), [0, 400], fs)

    assert cov.all() and y.size == 1200
    np.testing.assert_allclose(y[:400], clip[:400], atol=1e-12)
    np.testing.assert_allclose(y[400:800], clip[400:800], atol=1e-12)
    np.testing.assert_allclose(y[800:], clip[:400], atol=1e-12)


def test_ones_window_is_a_plain_mean_in_the_overlap():
    fs = 100.0
    a, b = np.full(100, 1.0), np.full(100, 5.0)   # L=100, shift 50
    _, y, cov = assemble_session(np.stack([a, b]), [0, 50], fs, window='ones')
    assert cov.all()
    np.testing.assert_allclose(y[:50], 1.0)
    np.testing.assert_allclose(y[50:100], 3.0)    # mean(1, 5)
    np.testing.assert_allclose(y[100:], 5.0)


# --------------------------------------------------------------------------- #
# why it matters: the padding is not harmless
# --------------------------------------------------------------------------- #
def test_zero_padding_biases_pearson():
    """Reproduces the old behaviour and shows the bias it introduced."""
    fs = 100.0
    ref = np.sin(2 * np.pi * 0.3 * np.arange(1600) / fs)
    pred = ref[800:1600]                          # perfect on the covered half
    _, y, cov = assemble_session(pred[None, :], [800], fs)

    def pearson(p, r):
        p, r = p - p.mean(), r - r.mean()
        return float((p * r).sum() / np.sqrt((p * p).sum() * (r * r).sum()))

    padded = np.nan_to_num(y)                     # the PRE-FIX output
    honest = pearson(y[cov], ref[cov])
    biased = pearson(padded, ref)

    assert honest == pytest.approx(1.0, abs=1e-9)
    assert biased < honest - 0.1, 'zero padding must visibly deflate the score'


# --------------------------------------------------------------------------- #
# end to end through the session runner
# --------------------------------------------------------------------------- #
def _write_one_clip_run(tmp_path, t_start, seconds=16.0, fs=100.0):
    n_ref = int(seconds * fs)
    ref = np.sin(2 * np.pi * 0.3 * np.arange(n_ref) / fs)
    sig = tmp_path / 'Resp_Volts.txt'
    np.savetxt(sig, np.repeat(ref, 10))           # 1000 Hz native

    start = int(round(t_start * fs))
    pred = ref[start:start + 800]
    np.save(tmp_path / 'preds.npy', pred[None, :])
    np.save(tmp_path / 'targets.npy', pred[None, :])
    with open(tmp_path / 'entries.json', 'w') as fh:
        json.dump([{'session': 'S001_T1', 't_start': float(t_start),
                    'signals_file': str(sig), 'task': 'T1',
                    'task_level': ['high']}], fh)
    return ref, start


def _run(tmp_path):
    args = argparse.Namespace(
        pred_dir=str(tmp_path), pred_path='', target_path='', entries_path='',
        waveform='resp', fs=100.0, tier='1', band='', smooth='',
        out=str(tmp_path / 'm.json'), fig_dir=str(tmp_path / 'fig'),
        zoom_s=20.0, max_sessions=0, window_s=30.0, raw_fs=1000.0)
    session_main(args)
    with open(tmp_path / 'm.json') as fh:
        return json.load(fh)['sessions']['S001_T1']


def test_runner_scores_only_the_covered_span(tmp_path):
    """The e2e M030_T4 shape: 8 s of padding must not enter any metric."""
    _write_one_clip_run(tmp_path, t_start=8.0)
    rec = _run(tmp_path)

    assert rec['coverage'] == pytest.approx(0.5)
    assert rec['scored_samples'] == 800
    assert rec['duration_s'] == pytest.approx(15.99, abs=0.02)
    assert rec['t_scored'] == pytest.approx([8.0, 15.99], abs=0.02)
    # the prediction IS the reference over the covered span
    assert rec['session_tier1']['pearson'] == pytest.approx(1.0, abs=1e-6)
    assert rec['session_tier1']['mae'] == pytest.approx(0.0, abs=1e-9)
    assert os.path.isfile(rec['figure'])


def test_runner_reports_full_coverage_for_a_zero_start(tmp_path):
    _write_one_clip_run(tmp_path, t_start=0.0)
    rec = _run(tmp_path)
    assert rec['coverage'] == pytest.approx(1.0)
    assert rec['scored_samples'] == 800
    assert rec['session_tier1']['pearson'] == pytest.approx(1.0, abs=1e-6)
