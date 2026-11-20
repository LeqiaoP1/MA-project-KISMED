"""Consumer window-contract tests for the BP artefact (ADR-004 Criterion 7, C3/C4).

The window rule is written once, in ``data.bp_windows``. These tests pin it
against a brute-force index walk and pin the difference between the two ways a
``segment_of`` call can fail: a caller asking the wrong question (``ValueError``)
versus a shard whose mask and segment table disagree (``AssertionError``).
"""
import numpy as np
import pytest

import bp_synth as bs
from data import bp_artefact as bpa
from data import bp_preprocess as bpp
from data import bp_windows as bpw

FS = 1000


def _shard(n_segments=2, seg_len=5000, gap=100):
    """A real payload with ``n_segments`` valid 5 s segments separated by gaps."""
    parts = []
    for i in range(n_segments):
        parts.append(bs.sine(seg_len, level=70.0 + 40.0 * i))
        if i + 1 < n_segments:
            parts.append(bs.constant(gap, 20.0))
    out = bpp.preprocess_session(bs.concat(*parts))
    assert out['n_segments'] == n_segments
    return out


# --------------------------------------------------------------------------- #
# window arithmetic (G9, G14)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('t_start,clip_s', [
    (0.0, 1.0), (0.5, 0.25), (2.0, 4.0), (1.234, 0.678), (0.0, 8.0),
])
def test_window_indices_match_a_brute_force_walk(t_start, clip_s):
    s, n = bpw.window_indices(t_start, clip_s, FS)
    assert s == int(round(t_start * FS))
    assert n == int(round(clip_s * FS))
    assert np.arange(s, s + n).size == n


def test_window_indices_rejects_an_empty_or_negative_request():
    with pytest.raises(ValueError, match='clip_seconds'):
        bpw.window_indices(0.0, 0.0, FS)
    with pytest.raises(ValueError, match='t_start'):
        bpw.window_indices(-1.0, 1.0, FS)
    with pytest.raises(ValueError, match='fs_out'):
        bpw.window_indices(0.0, 1.0, 0)


def test_window_indices_does_not_assert_that_the_window_fits():
    # probing past the end must be answerable, not fatal
    s, n = bpw.window_indices(1000.0, 4.0, FS)
    assert (s, n) == (1_000_000, 4000)


# --------------------------------------------------------------------------- #
# usable: bounds first, slice second (Criterion 7, G7)
# --------------------------------------------------------------------------- #
def test_usable_rejects_every_out_of_bounds_case():
    mask = np.ones(100, dtype=bool)
    assert bpw.usable(mask, 0, 100) is True
    assert bpw.usable(mask, 0, 0) is False
    assert bpw.usable(mask, -1, 10) is False
    assert bpw.usable(mask, 95, 10) is False


def test_usable_rejects_a_window_over_a_hole():
    mask = np.ones(100, dtype=bool)
    mask[50] = False
    assert bpw.usable(mask, 0, 50) is True
    assert bpw.usable(mask, 0, 51) is False
    assert bpw.usable(mask, 50, 1) is False


def test_usable_does_not_mistake_a_truncated_all_true_slice_for_a_window():
    mask = np.ones(10, dtype=bool)
    # mask[8:100] is a non-empty all-True slice, but the window does not fit
    assert bool(mask[8:100].all())
    assert bpw.usable(mask, 8, 100) is False


# --------------------------------------------------------------------------- #
# read_window never returns a hole
# --------------------------------------------------------------------------- #
def test_read_window_returns_the_slice():
    shard = _shard()
    seg = bpw.read_window(shard, 0, 1000)
    assert seg.size == 1000
    assert not np.isnan(seg).any()
    assert np.array_equal(seg, shard['signal'][0:1000])


def test_read_window_raises_on_a_hole():
    shard = _shard()
    start, stop = int(shard['seg_start'][0]), int(shard['seg_stop'][0])
    with pytest.raises(ValueError, match='not usable'):
        bpw.read_window(shard, start, (stop - start) + 1)


def test_read_window_raises_out_of_bounds():
    with pytest.raises(ValueError, match='outside'):
        bpw.read_window(np.zeros(10, dtype=np.float32), 5, 10)


def test_read_window_rejects_a_negative_start_on_a_shard():
    shard = _shard()
    with pytest.raises(ValueError, match='not usable'):
        bpw.read_window(shard, -5, 10)


def test_read_window_raises_on_nan_even_without_a_mask():
    signal = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match='NaN'):
        bpw.read_window(signal, 0, 3)


# --------------------------------------------------------------------------- #
# segment_of: C3 as an executable assertion
# --------------------------------------------------------------------------- #
def test_segment_of_returns_the_one_containing_segment():
    shard = _shard(n_segments=2, seg_len=5000, gap=100)
    first = (int(shard['seg_start'][0]), int(shard['seg_stop'][0]))
    second_start = int(shard['seg_start'][1])

    assert bpw.segment_of(shard, first[0], 1000) == 0
    assert bpw.segment_of(shard, second_start, 1000) == 1
    assert bpw.segment_of(shard, first[1] - 1000, 1000) == 0


def test_segment_of_raises_value_error_for_an_unusable_window():
    shard = _shard(n_segments=2, seg_len=5000, gap=100)
    gap_start = int(shard['seg_stop'][0])
    with pytest.raises(ValueError, match='not usable'):
        bpw.segment_of(shard, gap_start, 100)


def test_segment_of_raises_assertion_error_when_no_segment_contains_the_window():
    shard = _shard()
    broken = dict(shard)
    broken['seg_start'] = np.array([], dtype=np.int64)
    broken['seg_stop'] = np.array([], dtype=np.int64)
    with pytest.raises(AssertionError, match='corrupt'):
        bpw.segment_of(broken, 0, 1000)


def test_segment_of_raises_assertion_error_on_a_straddle():
    shard = _shard()
    straddling = dict(shard)
    straddling['seg_start'] = np.array([0, 0], dtype=np.int64)
    straddling['seg_stop'] = np.array([5000, 6000], dtype=np.int64)
    with pytest.raises(AssertionError, match='corrupt'):
        bpw.segment_of(straddling, 0, 1000)


# --------------------------------------------------------------------------- #
# the clip is a consumer choice: no ceiling is enforced (A7)
# --------------------------------------------------------------------------- #
def test_a_clip_longer_than_the_minimum_segment_is_legal_but_unusable():
    shard = _shard(n_segments=1, seg_len=5000)
    s, n = bpw.window_indices(0.0, 6.0, FS)      # 6000 > 5000-sample segment
    assert n == 6000
    assert bpw.usable(shard['mask'], s, n) is False
    with pytest.raises(ValueError):
        bpw.read_window(shard, s, n)


def test_a_clip_exactly_filling_one_segment_is_usable():
    shard = _shard(n_segments=1, seg_len=4000)
    s, n = bpw.window_indices(0.0, 4.0, FS)
    assert bpw.usable(shard['mask'], s, n) is True
    assert bpw.segment_of(shard, s, n) == 0


# --------------------------------------------------------------------------- #
# verify_identity (C6)
# --------------------------------------------------------------------------- #
def test_verify_identity_accepts_the_matching_artefact(tmp_path):
    params = bpa.build_params([{'session': 'S1_T1', 'source_file': '/x'}])
    root = bpa.cache_root(tmp_path, bpa.cache_key(params))
    bpa.write_params(root, params)

    assert bpw.verify_identity(tmp_path, params) == root
    assert bpw.verify_identity(root, params) == root


def test_verify_identity_refuses_a_mismatch_with_the_field_diff(tmp_path):
    params = bpa.build_params([{'session': 'S1_T1', 'source_file': '/x'}])
    root = bpa.cache_root(tmp_path, bpa.cache_key(params))
    bpa.write_params(root, params)

    other = bpa.build_params([{'session': 'S1_T1', 'source_file': '/x'}],
                             min_segment_s=5.0)
    with pytest.raises(ValueError, match='min_segment_s'):
        bpw.verify_identity(tmp_path, other)
