"""Payload-level tests for the BP session artefact (ADR-004 Criteria 5–7).

These tests assert the two things a gate test cannot: that the 4 s rule is a
*segment* rule, and that the only arithmetic applied to a valid sample is
``x - segment_mean``.
"""
import numpy as np
import pytest

import bp_synth as bs
from data import bp_preprocess as bpp

FS = 1000


def _two_segment_trace(n=5000, gap=100):
    """Two valid segments whose arithmetic means are EXACTLY 70 and 110 mmHg.

    Criterion 6 names those two means, so the fixture is levelled rather than
    assumed: a sinusoid over a non-integer number of periods has a small residual
    mean that would otherwise blur the assertion.
    """
    def _levelled(block, target):
        block = np.asarray(block, dtype=np.float64)
        return block - block.mean() + float(target)

    return bs.concat(_levelled(bs.sine(n, level=0.0), 70.0),
                     bs.constant(gap, 20.0),
                     _levelled(bs.sine(n, level=0.0), 110.0))


# --------------------------------------------------------------------------- #
# segment validity (Criterion 5, G5)
# --------------------------------------------------------------------------- #
def test_segment_of_3999_samples_is_invalid_and_4000_is_valid():
    x = bs.concat(bs.sine(3999), [20.0], bs.sine(4000))
    out = bpp.preprocess_session(x)

    assert out['n_segments'] == 1
    assert out['seg_start'].tolist() == [4000]
    assert out['seg_stop'].tolist() == [8000]
    assert out['seg_seconds'].tolist() == [4.0]

    assert not out['mask'][:3999].any()
    assert out['cause'][:3999].tolist() == [bpp.CAUSE_SHORT] * 3999
    assert np.isnan(out['signal'][:3999]).all()

    assert out['mask'][4000:].all()
    assert out['cause'][4000:].tolist() == [bpp.CAUSE_VALID] * 4000
    assert not np.isnan(out['signal'][4000:]).any()


def test_short_segment_cause_only_applies_to_samples_that_passed_all_gates():
    x = bs.concat(bs.sine(3999), [20.0], bs.sine(50), [300.0], bs.sine(50))
    out = bpp.preprocess_session(x)
    assert out['cause'][3999] == bpp.CAUSE_RANGE
    assert out['cause'][4050] == bpp.CAUSE_RANGE
    assert set(np.unique(out['cause'][:3999]).tolist()) == {bpp.CAUSE_SHORT}


# --------------------------------------------------------------------------- #
# per-segment DC removal (Criterion 6, G6)
# --------------------------------------------------------------------------- #
def test_each_valid_segment_is_stored_with_zero_mean():
    x = _two_segment_trace()
    out = bpp.preprocess_session(x)

    assert out['n_segments'] == 2
    seg0 = slice(0, 5000)
    seg1 = slice(5100, 10100)
    assert abs(float(out['signal'][seg0].mean())) < 1e-3
    assert abs(float(out['signal'][seg1].mean())) < 1e-3

    assert out['seg_mean_mmhg'][0] == pytest.approx(70.0, abs=2e-3)
    assert out['seg_mean_mmhg'][1] == pytest.approx(110.0, abs=2e-3)


def test_no_sample_carries_the_other_segments_offset():
    x = _two_segment_trace()
    out = bpp.preprocess_session(x)
    # restoring the segment's own mean must reproduce the raw sample
    for start, stop, mean in zip(out['seg_start'], out['seg_stop'],
                                 out['seg_mean_mmhg']):
        restored = out['signal'][start:stop].astype(np.float64) + mean
        assert np.allclose(restored, x[start:stop], atol=1e-4)


def test_dc_removal_is_a_constant_not_a_filter():
    x = bs.sine(5000, level=90.0, amp=2.0)
    out = bpp.preprocess_session(x)
    stored = out['signal'][:5000].astype(np.float64)
    # shape preserved: the residual equals the raw block minus one constant
    offset = x[:5000] - stored
    assert float(offset.std()) < 1e-3, 'mean removal must be a constant offset'
    assert float(stored.std()) == pytest.approx(float(x[:5000].std()), rel=1e-3)


# --------------------------------------------------------------------------- #
# invariants (Criterion 7, G7)
# --------------------------------------------------------------------------- #
def test_signal_is_nan_exactly_where_the_mask_is_false():
    out = bpp.preprocess_session(_two_segment_trace())
    assert np.array_equal(np.isnan(out['signal']), ~out['mask'])


def test_mask_and_cause_say_the_same_thing():
    out = bpp.preprocess_session(_two_segment_trace())
    assert np.array_equal(out['mask'], out['cause'] == bpp.CAUSE_VALID)


def test_no_zero_is_ever_substituted_for_an_invalid_sample():
    out = bpp.preprocess_session(_two_segment_trace())
    invalid = out['signal'][~out['mask']]
    assert invalid.size > 0
    assert np.isnan(invalid).all()
    assert not (invalid == 0.0).any()


# --------------------------------------------------------------------------- #
# zero-segment sessions are shards, not failures (Criterion 8, G8)
# --------------------------------------------------------------------------- #
def test_all_invalid_trace_yields_a_typed_empty_segment_table():
    out = bpp.preprocess_session(bs.constant(5000, 20.0))
    assert out['valid_frac'] == 0.0
    assert out['n_segments'] == 0
    assert np.isnan(out['signal']).all()
    assert not out['mask'].any()
    assert out['seg_start'].dtype == np.int64
    assert out['seg_stop'].dtype == np.int64
    assert out['seg_mean_mmhg'].dtype == np.float64
    assert out['seg_seconds'].dtype == np.float64
    assert out['seg_start'].size == out['seg_stop'].size == 0


def test_flatline_only_trace_yields_no_segment():
    out = bpp.preprocess_session(bs.constant(5000, 90.0))
    assert out['n_segments'] == 0
    assert set(np.unique(out['cause']).tolist()) == {bpp.CAUSE_FLATLINE}


# --------------------------------------------------------------------------- #
# determinism and argument validation
# --------------------------------------------------------------------------- #
def test_two_calls_are_bit_identical():
    x = _two_segment_trace()
    a = bpp.preprocess_session(x)
    b = bpp.preprocess_session(x)
    for key in ('signal', 'mask', 'cause', 'seg_start', 'seg_stop',
                'seg_mean_mmhg', 'seg_seconds'):
        assert np.array_equal(a[key], b[key], equal_nan=True), key


def test_preprocess_does_not_mutate_the_input():
    x = _two_segment_trace()
    before = x.copy()
    bpp.preprocess_session(x)
    assert np.array_equal(x, before)


def test_resampling_is_refused():
    with pytest.raises(ValueError, match='no resampling'):
        bpp.preprocess_session(bs.sine(5000), fs_out=500)


def test_non_1d_trace_is_refused():
    with pytest.raises(ValueError, match='1-D'):
        bpp.preprocess_session(np.zeros((10, 10)))


def test_empty_trace_is_refused():
    with pytest.raises(ValueError, match='empty'):
        bpp.preprocess_session(np.array([]))


def test_foreign_store_dtype_is_refused():
    with pytest.raises(ValueError, match='frozen'):
        bpp.preprocess_session(bs.sine(5000), dtype='float64')


def test_provenance_scalars_are_reported():
    x = bs.concat(bs.sine(5000), [20.0], bs.sine(5000))
    out = bpp.preprocess_session(x)
    assert out['n_src'] == 10001
    assert out['fs_in'] == FS and out['fs_out'] == FS
    assert out['n_nonfinite'] == 0
    assert out['valid_frac'] == pytest.approx(10000 / 10001)
    assert out['signal'].dtype == np.float32
    assert out['mask'].dtype == np.bool_
    assert out['cause'].dtype == np.uint8
