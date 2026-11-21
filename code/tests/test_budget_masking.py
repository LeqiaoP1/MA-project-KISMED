"""Contract tests for the constant-budget apportionment engine (ADR-002).

Design: ``code/notes/design/ADR-002-asymmetric-multimodal-masking.md``.
Plan + gate ids: ``code/notes/plan/features/ADR-002-constant-budget-masking-plan.md``
(Phase 1, §8).

The module is pure and CPU-only: no dataset, no model, no GPU. The ADR's own
numbers ($N_{\\text{budget}}=350$, $T=25$, $m=2$, $M=9$, $C_p=125$) are the
defaults every gate is read at.
"""
import math

import pytest
import torch

from core.budget_masking import (KINDS, MODE_INPAINTING, MODE_VIRTUAL_SENSING,
                                 BudgetPlan, ConstantBudgetApportioner,
                                 StreamSpec)

#: the ADR's geometry: 36 spatial patches x 25 tokens per tube = 900 tokens per
#: visual stream, 250 temporal tokens per physiological stream.
N_BUDGET = 350
T = 25
M_TUBES = 9
C_P = 125
SPATIAL = 36
PHYSIO = 250


def visual(name='tir'):
    return StreamSpec(name, 'visual', SPATIAL, tokens_per_unit=T)


def signal(name='resp'):
    return StreamSpec(name, 'signal', PHYSIO, tokens_per_unit=1)


def make(streams=None, **kw):
    streams = streams if streams is not None else [visual('tir'), visual('rgb2d'),
                                                   signal('resp')]
    kw.setdefault('n_budget', N_BUDGET)
    return ConstantBudgetApportioner(streams, **kw)


def two_signal(**kw):
    return make([visual('tir'), visual('rgb2d'), signal('bp'), signal('resp')],
                **kw)


def draw(app, steps, seed=0, **kw):
    gen = torch.Generator().manual_seed(seed)
    return [app.sample_step(generator=gen, **kw) for _ in range(steps)]


# --------------------------------------------------------------------------- #
# G1 — the defaults reproduce the ADR's numbers
# --------------------------------------------------------------------------- #
def test_defaults_reproduce_the_adr_numbers_in_mode1():
    plans = draw(make(p_virtual_sensing=1.0), steps=200, seed=0)
    for plan in plans:
        assert plan.mode == MODE_VIRTUAL_SENSING
        assert plan.kept_tokens['resp'] == 0
        s_tir = plan.kept_units['tir']
        s_rgb = plan.kept_units['rgb2d']
        assert 5 <= s_tir <= 9 and 5 <= s_rgb <= 9
        assert s_tir + s_rgb == 14
        assert plan.kept_tokens['tir'] == s_tir * T


def test_defaults_reproduce_the_adr_numbers_in_mode2():
    plans = draw(two_signal(p_virtual_sensing=0.0), steps=400, seed=1)
    for plan in plans:
        assert plan.mode == MODE_INPAINTING
        for name in ('bp', 'resp'):
            assert 0 <= plan.kept_tokens[name] <= C_P
        s_total = plan.kept_units['tir'] + plan.kept_units['rgb2d']
        assert 4 <= s_total <= 14
        assert plan.total_tokens == N_BUDGET


# --------------------------------------------------------------------------- #
# G2/G3/G4 — invariants under sampling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('p', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('app_factory', ['one_signal', 'two_signal'])
def test_exact_sum_caps_floors_and_tube_multiples(app_factory, p):
    app = make(p_virtual_sensing=p) if app_factory == 'one_signal' else \
        two_signal(p_virtual_sensing=p)
    for plan in draw(app, steps=2_500, seed=7):
        assert plan.total_tokens == N_BUDGET
        for spec in app.visual:
            kept_tokens = plan.kept_tokens[spec.name]
            kept_units = plan.kept_units[spec.name]
            assert kept_tokens % T == 0
            assert 2 <= kept_units <= M_TUBES
            assert kept_tokens == kept_units * T
        for spec in app.signal:
            assert 0 <= plan.kept_tokens[spec.name] <= C_P
        if plan.mode == MODE_VIRTUAL_SENSING:
            assert all(plan.kept_tokens[s.name] == 0 for s in app.signal)


def test_ten_thousand_draws_hold_the_exact_sum():
    """Criterion 2: the invariant survives 1e4 draws per mode."""
    for p in (0.0, 1.0):
        app = two_signal(p_virtual_sensing=p)
        plans = draw(app, steps=10_000, seed=11)
        assert {pl.mode for pl in plans} == {MODE_INPAINTING if p == 0
                                             else MODE_VIRTUAL_SENSING}
        assert all(pl.total_tokens == N_BUDGET for pl in plans)


# --------------------------------------------------------------------------- #
# G7 — the mode frequency matches p_virtual_sensing
# --------------------------------------------------------------------------- #
def test_mode1_frequency_matches_the_configured_probability():
    p, n = 0.30, 10_000
    plans = draw(make(p_virtual_sensing=p), steps=n, seed=3)
    hits = sum(pl.mode == MODE_VIRTUAL_SENSING for pl in plans)
    sd = math.sqrt(n * p * (1 - p))
    assert abs(hits - n * p) <= 4 * sd, (hits, n * p, sd)


def test_probability_extremes_are_absorbing():
    assert all(pl.mode == MODE_VIRTUAL_SENSING
               for pl in draw(make(p_virtual_sensing=1.0), steps=50, seed=0))
    assert all(pl.mode == MODE_INPAINTING
               for pl in draw(make(p_virtual_sensing=0.0), steps=50, seed=0))


# --------------------------------------------------------------------------- #
# G6 — the visual split degenerates correctly at both ends
# --------------------------------------------------------------------------- #
def test_visual_split_at_the_top_of_the_range():
    """n_budget 450 => 18 tubes => the only legal split is 9 + 9."""
    app = make(n_budget=450, p_virtual_sensing=1.0)
    for plan in draw(app, steps=100, seed=5):
        assert plan.kept_units['tir'] == M_TUBES
        assert plan.kept_units['rgb2d'] == M_TUBES


def test_visual_split_at_the_bottom_of_the_range():
    """Physiological saturation leaves 100 tokens => 4 tubes => 2 + 2."""
    app = _Scripted([125, 125, 2],
                    streams=[visual('tir'), visual('rgb2d'), signal('bp'),
                             signal('resp')],
                    n_budget=N_BUDGET, p_virtual_sensing=0.0)
    plan = app.sample_step(generator=torch.Generator().manual_seed(0))
    assert plan.kept_units['tir'] == 2
    assert plan.kept_units['rgb2d'] == 2
    assert plan.total_tokens == N_BUDGET


# --------------------------------------------------------------------------- #
# G14 — the alignment remainder is charged to the FIRST physiological stream
# --------------------------------------------------------------------------- #
class _Scripted(ConstantBudgetApportioner):
    """Engine whose integer draws are scripted (alignment arithmetic only)."""

    def __init__(self, draws, **kw):
        super().__init__(**kw)
        self._script = list(draws)

    def _randint(self, lo, hi, generator):
        value = self._script.pop(0)
        assert lo <= value <= hi, (value, lo, hi)
        return value


def test_alignment_remainder_goes_to_the_first_physiological_stream():
    """K_bp=100, K_resp=101 => R=24 => BP 124 / Resp 101 / 5 visual tubes."""
    app = _Scripted(
        [100, 101, 2],                 # raw bp, raw resp, visual split draw
        streams=[visual('tir'), visual('rgb2d'), signal('bp'), signal('resp')],
        n_budget=N_BUDGET, p_virtual_sensing=0.0)
    plan = app.sample_step(generator=torch.Generator().manual_seed(0))
    assert plan.kept_tokens['bp'] == 124
    assert plan.kept_tokens['resp'] == 101
    assert plan.kept_units['tir'] + plan.kept_units['rgb2d'] == 5
    assert 2 <= plan.kept_units['tir'] <= 3
    assert plan.total_tokens == N_BUDGET


def test_alignment_falls_through_to_the_next_stream_when_bp_is_full():
    """BP at its cap cannot absorb R, so Resp receives it instead."""
    # bp=125 (full), resp=101 -> R = (350-125-101) % 25 = 24; bp+24 > 125,
    # resp+24 == 125 -> the remainder moves to the SECOND stream.
    app = _Scripted([125, 101, 2],
                    streams=[visual('tir'), visual('rgb2d'), signal('bp'),
                             signal('resp')],
                    n_budget=N_BUDGET, p_virtual_sensing=0.0)
    plan = app.sample_step(generator=torch.Generator().manual_seed(0))
    assert plan.kept_tokens['bp'] == 125
    assert plan.kept_tokens['resp'] == 125
    assert plan.total_tokens == N_BUDGET


def test_alignment_subtracts_when_no_stream_can_absorb():
    """Unaligned cap: 360 budget, T=20, C_p=125 => subtract T-R from the first."""
    app = _Scripted([125, 3],
                    streams=[StreamSpec('tir', 'visual', SPATIAL, 20),
                             StreamSpec('rgb2d', 'visual', SPATIAL, 20),
                             signal('resp')],
                    n_budget=360, p_virtual_sensing=0.0)
    plan = app.sample_step(generator=torch.Generator().manual_seed(0))
    # R = (360-125) % 20 = 15 -> 125+15 > 125, so subtract 20-15 = 5
    assert plan.kept_tokens['resp'] == 120
    assert plan.total_tokens == 360
    assert (plan.kept_tokens['tir'] + plan.kept_tokens['rgb2d']) == 240


# --------------------------------------------------------------------------- #
# G15 — a single physiological stream exercises the P = 1 residual branch
# --------------------------------------------------------------------------- #
def test_single_physiological_stream_stays_aligned_and_inside_the_cap():
    app = make(p_virtual_sensing=0.0)          # tir, rgb2d, resp only
    for plan in draw(app, steps=2_000, seed=13):
        kept = plan.kept_tokens['resp']
        assert kept % T == 0 and 0 <= kept <= C_P
        assert plan.kept_tokens['tir'] + plan.kept_tokens['rgb2d'] == \
            N_BUDGET - kept


# --------------------------------------------------------------------------- #
# G8 — replayability and one count per step
# --------------------------------------------------------------------------- #
def test_same_seed_replays_and_a_different_seed_does_not():
    app = make(p_virtual_sensing=0.5)
    a = draw(app, steps=50, seed=4)
    b = draw(app, steps=50, seed=4)
    c = draw(app, steps=50, seed=99)
    assert a == b
    assert a != c


# --------------------------------------------------------------------------- #
# G9 — construction rejects an infeasible tuple
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('streams,kw,needle', [
    ([visual('tir'), signal('resp')], {},
     'exactly TWO visual streams'),
    ([visual('tir'), visual('rgb'), visual('rgb2d'), signal('resp')], {},
     'exactly TWO visual streams'),
    ([visual('tir'), visual('rgb2d')], {},
     'one or two physiological streams'),
    ([visual('tir'), visual('rgb2d'), signal('bp'), signal('resp'),
      signal('ecg')], {}, 'one or two physiological streams'),
    ([visual('tir'), visual('rgb2d'), signal('resp')], {'n_budget': 340},
     'condition 1'),
    ([visual('tir'), visual('rgb2d'), signal('resp')], {'n_budget': 500},
     'condition 2'),
    ([visual('tir'), visual('rgb2d'), signal('bp'), signal('resp')],
     {'n_budget': 300}, 'condition 3'),
    ([visual('tir'), visual('rgb2d'), signal('resp')],
     {'video_cap_ratio': 0.05, 'video_min_tubes': 2}, 'below video_min_tubes'),
])
def test_infeasible_configurations_raise(streams, kw, needle):
    with pytest.raises(ValueError) as exc:
        make(streams, **kw)
    assert needle in str(exc.value)


def test_mismatched_tube_layouts_are_rejected():
    with pytest.raises(ValueError, match='tube layout'):
        make([StreamSpec('tir', 'visual', SPATIAL, 25),
              StreamSpec('rgb2d', 'visual', SPATIAL, 12), signal('resp')])


def test_bad_ratios_and_probabilities_are_rejected():
    for kw, needle in [({'video_cap_ratio': 0.0}, 'video_cap_ratio'),
                       ({'physio_cap_ratio': 1.5}, 'physio_cap_ratio'),
                       ({'p_virtual_sensing': -0.1}, 'p_virtual_sensing'),
                       ({'p_virtual_sensing': 1.1}, 'p_virtual_sensing')]:
        with pytest.raises(ValueError, match=needle):
            make(**kw)


# --------------------------------------------------------------------------- #
# explicit allocations (evaluation pins / G10a)
# --------------------------------------------------------------------------- #
def test_plan_from_counts_validates_the_budget_and_the_units():
    app = make()
    plan = app.plan_from_counts({'tir': 225, 'rgb2d': 125, 'resp': 0})
    assert isinstance(plan, BudgetPlan)
    assert plan.mode == MODE_VIRTUAL_SENSING
    assert plan.kept_units == {'tir': 9, 'rgb2d': 5, 'resp': 0}

    with pytest.raises(ValueError, match='n_budget|visible tokens'):
        app.plan_from_counts({'tir': 225, 'rgb2d': 225, 'resp': 0})
    with pytest.raises(ValueError, match='multiple of T'):
        app.plan_from_counts({'tir': 224, 'rgb2d': 126, 'resp': 0})
    with pytest.raises(ValueError, match='outside'):
        app.plan_from_counts({'tir': 0, 'rgb2d': 100, 'resp': 250})
    with pytest.raises(ValueError, match='missing'):
        app.plan_from_counts({'tir': 350})
    with pytest.raises(ValueError, match='undeclared'):
        app.plan_from_counts({'tir': 225, 'rgb2d': 125, 'resp': 0, 'eda': 0})


def test_plan_from_counts_accepts_a_mode2_allocation():
    app = two_signal()
    plan = app.plan_from_counts({'tir': 50, 'rgb2d': 50, 'bp': 125,
                                 'resp': 125})
    assert plan.mode == MODE_INPAINTING
    assert plan.total_tokens == N_BUDGET
