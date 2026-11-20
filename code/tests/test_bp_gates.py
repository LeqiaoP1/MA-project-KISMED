"""Gate-level tests for the BP session artefact (ADR-004 Criteria 1–4).

Every assertion here is a boundary case: the ADR fixes both sides of each
threshold, so a test that only checks "a long run is rejected" would let an
off-by-one through.
"""
import warnings

import numpy as np
import pytest

import bp_synth as bs
from data import bp_preprocess as bpp


# --------------------------------------------------------------------------- #
# Gate 1 — dynamic range (Criterion 1, G1)
# --------------------------------------------------------------------------- #
def test_gate1_admits_the_exact_bounds():
    x = np.array([30.00, 250.00])
    alive, cause = bpp.gate1_range(x)
    assert alive.tolist() == [True, True]
    assert cause.tolist() == [bpp.CAUSE_VALID, bpp.CAUSE_VALID]


def test_gate1_rejects_just_outside_the_bounds():
    x = np.array([29.99, 250.01])
    alive, cause = bpp.gate1_range(x)
    assert alive.tolist() == [False, False]
    assert cause.tolist() == [bpp.CAUSE_RANGE, bpp.CAUSE_RANGE]


def test_gate1_is_per_sample_and_leaves_neighbours_alone():
    x = bs.concat(bs.sine(50), [29.99], bs.sine(50), [250.01], bs.sine(50))
    alive, cause = bpp.gate1_range(x)
    assert cause[50] == bpp.CAUSE_RANGE
    assert cause[101] == bpp.CAUSE_RANGE
    assert alive.sum() == x.size - 2
    # every neighbour is untouched
    for idx, sample in enumerate(x):
        if idx in (50, 101):
            continue
        assert cause[idx] == bpp.CAUSE_VALID


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_gate1_rejects_non_finite(bad):
    x = bs.concat(bs.sine(20), [bad], bs.sine(20))
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # a NaN comparison must not warn
        alive, cause = bpp.gate1_range(x)
    assert cause[20] == bpp.CAUSE_NONFINITE
    assert alive.sum() == x.size - 1


# --------------------------------------------------------------------------- #
# Gate 2 — flatline (Criterion 2, G2)
# --------------------------------------------------------------------------- #
def test_gate2_keeps_exactly_200_samples_and_rejects_201():
    kept = bs.concat(bs.sine(200), bs.constant(200, 85.0), bs.sine(200))
    rejected = bs.concat(bs.sine(200), bs.constant(201, 85.0), bs.sine(200))

    alive1, _ = bpp.gate1_range(kept)
    alive2, rej2 = bpp.gate2_flatline(kept, alive1)
    assert not rej2.any()
    assert bool(alive2[200:400].all()), 'a 200 ms run must be admitted'

    alive1, _ = bpp.gate1_range(rejected)
    alive2, rej2 = bpp.gate2_flatline(rejected, alive1)
    assert bool(rej2[200:401].all()), 'a 201 ms run must be rejected in full'
    assert not bool(alive2[200:401].any())
    # samples outside the run are unaffected
    assert not bool(rej2[:200].any()) and not bool(rej2[401:].any())


def test_gate2_gate1_rejection_breaks_the_run():
    x = bs.concat(bs.constant(100, 85.0), [20.0], bs.constant(100, 85.0))
    alive1, cause = bpp.gate1_range(x)
    assert cause[100] == bpp.CAUSE_RANGE

    alive2, rej2 = bpp.gate2_flatline(x, alive1)
    assert not rej2.any(), 'two 100-sample runs must not join across a dead sample'
    assert alive2.sum() == 200


def test_gate2_never_rejects_a_pulsatile_trace():
    x = bs.sine(8000)
    assert bs.max_equal_run(x) < 201, 'fixture is not pulsatile enough to test with'
    alive1, _ = bpp.gate1_range(x)
    alive2, rej2 = bpp.gate2_flatline(x, alive1)
    assert not rej2.any()
    assert alive2.all()


# --------------------------------------------------------------------------- #
# Gate 3 — rail dwell (Criterion 3, G3; Criterion 4, G4)
# --------------------------------------------------------------------------- #
def test_gate3_keeps_29_samples_and_rejects_30_at_y_max():
    kept = bs.concat(bs.sine(2000), bs.constant(29, 92.0))
    rejected = bs.concat(bs.sine(2000), bs.constant(30, 92.0))

    alive1, _ = bpp.gate1_range(kept)
    alive2, _ = bpp.gate2_flatline(kept, alive1)
    alive3, rej3 = bpp.gate3_rail(kept, alive2)
    assert not rej3.any(), 'a 29 ms dwell must be admitted'
    assert alive3.sum() == alive2.sum()

    alive1, _ = bpp.gate1_range(rejected)
    alive2, _ = bpp.gate2_flatline(rejected, alive1)
    alive3, rej3 = bpp.gate3_rail(rejected, alive2)
    assert bool(rej3[2000:2030].all()), 'a 30 ms dwell must be rejected'
    assert not bool(alive3[2000:2030].any())
    assert not bool(rej3[:2000].any())


def test_gate3_rejects_a_dwell_at_y_min():
    rejected = bs.concat(bs.sine(2000), bs.constant(30, 88.0))
    alive1, _ = bpp.gate1_range(rejected)
    alive2, _ = bpp.gate2_flatline(rejected, alive1)
    assert 88.0 == pytest.approx(float(rejected[alive2].min()))
    _, rej3 = bpp.gate3_rail(rejected, alive2)
    assert bool(rej3[2000:2030].all())


def test_gate3_ignores_an_interior_plateau():
    x = bs.concat(bs.sine(2000), bs.constant(50, 90.0), bs.sine(2000))
    alive1, _ = bpp.gate1_range(x)
    alive2, _ = bpp.gate2_flatline(x, alive1)
    y_max = float(x[alive2].max())
    y_min = float(x[alive2].min())
    assert y_min < 90.0 < y_max, 'the plateau must be strictly interior'
    _, rej3 = bpp.gate3_rail(x, alive2)
    assert not rej3.any(), 'an interior plateau is not a rail dwell'


def test_gate3_gate2_rejection_breaks_the_dwell():
    x = bs.concat(bs.constant(20, 95.0), bs.constant(250, 85.0),
                  bs.constant(20, 95.0))
    alive1, _ = bpp.gate1_range(x)
    alive2, rej2 = bpp.gate2_flatline(x, alive1)
    assert bool(rej2[20:270].all()), 'the 250-sample flatline must be removed first'

    _, rej3 = bpp.gate3_rail(x, alive2)
    assert not rej3.any(), 'two 20-sample dwells must not join across a dead sample'


def test_gate3_extrema_come_from_the_survivors_not_the_raw_trace():
    # The 300 mmHg spike is Gate-1 dead. If Gate 3 used the RAW maximum it would
    # be 300.0, the 92.0 dwell would not match it, and the dwell would survive.
    x = bs.concat(bs.sine(2000), [300.0], bs.constant(30, 92.0))
    alive1, cause = bpp.gate1_range(x)
    assert cause[2000] == bpp.CAUSE_RANGE
    alive2, _ = bpp.gate2_flatline(x, alive1)
    assert float(x[alive2].max()) == pytest.approx(92.0)
    _, rej3 = bpp.gate3_rail(x, alive2)
    assert bool(rej3[2001:2031].all())


def test_gate3_ymax_equal_ymin_is_one_dwell():
    x = bs.constant(150, 90.0)
    alive1, _ = bpp.gate1_range(x)
    alive2, rej2 = bpp.gate2_flatline(x, alive1)
    assert not rej2.any(), '150 samples is below the Gate-2 limit'
    alive3, rej3 = bpp.gate3_rail(x, alive2)
    assert bool(rej3.all())
    assert not bool(alive3.any())


# --------------------------------------------------------------------------- #
# order, survivors and one cause per sample (Criterion 4, G4)
# --------------------------------------------------------------------------- #
def test_every_rejected_sample_carries_exactly_one_cause():
    x = bs.concat(bs.sine(500), [300.0], bs.constant(250, 85.0),
                  bs.constant(40, 92.0), bs.sine(500))
    _, cause = bpp.admitted_mask(x)
    assert cause[500] == bpp.CAUSE_RANGE           # not also rail
    assert cause[501:751].tolist() == [bpp.CAUSE_FLATLINE] * 250
    assert cause[751:791].tolist() == [bpp.CAUSE_RAIL] * 40
    assert set(np.unique(cause).tolist()) <= set(bpp.CAUSE_NAMES)


def test_zero_survivors_never_reduce_an_empty_selection():
    x = bs.constant(5000, 20.0)                    # every sample fails Gate 1
    alive1, cause = bpp.gate1_range(x)
    assert not alive1.any()
    alive2, rej2 = bpp.gate2_flatline(x, alive1)
    assert not alive2.any() and not rej2.any()
    alive3, rej3 = bpp.gate3_rail(x, alive2)       # must not call .max() on []
    assert not alive3.any() and not rej3.any()
    assert cause.tolist() == [bpp.CAUSE_RANGE] * x.size


def test_alive2_empty_ends_the_cascade():
    x = bs.constant(2000, 90.0)                    # every sample fails Gate 2
    alive1, _ = bpp.gate1_range(x)
    alive2, rej2 = bpp.gate2_flatline(x, alive1)
    assert not alive2.any() and bool(rej2.all())
    alive3, rej3 = bpp.gate3_rail(x, alive2)
    assert not alive3.any() and not rej3.any()


# --------------------------------------------------------------------------- #
# the run-length helper itself
# --------------------------------------------------------------------------- #
def test_equal_runs_matches_a_brute_force_rle():
    x = bs.concat(bs.sine(300), bs.constant(40, 85.0), [np.nan],
                  bs.constant(40, 85.0), bs.sine(300), bs.constant(3, 99.0))
    alive = np.isfinite(x) & (x >= 30.0) & (x <= 250.0)
    assert bpp.equal_runs(x, alive) == bs.brute_equal_runs(x, alive)


def test_equal_runs_emits_no_nan_warning():
    x = bs.concat(bs.constant(5, 90.0), [np.nan], bs.constant(5, 90.0))
    alive = np.isfinite(x)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        runs = bpp.equal_runs(x, alive)
    assert runs == [(0, 5), (6, 11)]


def test_contiguous_runs_edges():
    assert bpp.contiguous_runs([False, True, True, False, True]) == [(1, 3), (4, 5)]
    assert bpp.contiguous_runs([]) == []
    assert bpp.contiguous_runs([False] * 4) == []
    assert bpp.contiguous_runs([True] * 4) == [(0, 4)]
