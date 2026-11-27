"""The EVALUATION-ONLY budget-exempt plan: a FULLY VISIBLE visual stream.

Design of record: the constant-budget contract is ADR-002 / ADR-007 §2.6 — the
encoder sees exactly ``n_budget`` visible tokens on every TRAINING step. Nothing
here changes that: the sampler cannot set ``BudgetPlan.budget_exempt``, and the
flag is reachable only through ``plan_from_counts(..., exempt_budget=True)``,
which exists so an evaluation can state "encode the whole clip" (the
remote-sensing / deployment condition: 36 tubes x 25 tokens = 900 TIR tokens and
0 respiration tokens on the two-stream arm).

What is asserted here:
* the exempt plan states its own length and keeps the stratified split legal;
* the SAME counts without the flag are still refused (contract intact);
* sampled (training) plans are never exempt;
* the exempt plan still validates whole tubes and per-stream unit counts;
* a fully covering stratified draw returns every tube, i.e. nothing is masked.
"""
import torch

import pytest

from core.budget_masking import (REGIME_SINGLE_VISUAL, SPATIAL_ANATOMY_AWARE,
                                 StreamSpec, ConstantBudgetApportioner)
from core.mask_placement import place_video_stratified

N_BUDGET = 200
T = 25
SPATIAL = 36
PHYSIO = 250
CORE = (14, 15, 20, 21)


def specs():
    return [StreamSpec('tir', 'visual', SPATIAL, T, spatial_grid=(6, 6)),
            StreamSpec('resp', 'signal', PHYSIO, 1)]


def make(**kw):
    d = dict(n_budget=N_BUDGET, video_cap_ratio=0.25, physio_cap_ratio=0.40,
             video_min_tubes=4, spatial_sampling=SPATIAL_ANATOMY_AWARE,
             physio_draw_min=1)
    d.update(kw)
    return ConstantBudgetApportioner(specs(), **d)


def test_exempt_plan_states_its_own_length_and_keeps_the_split_legal():
    plan = make().plan_from_counts({'tir': 900, 'resp': 0}, kept_core=len(CORE),
                                   exempt_budget=True)
    assert plan.budget_exempt is True
    assert plan.total_tokens == 900                    # NOT n_budget
    assert plan.kept_units == {'tir': SPATIAL, 'resp': 0}
    assert plan.regime == REGIME_SINGLE_VISUAL
    assert plan.core_units == {'tir': len(CORE)}
    assert plan.periph_units == {'tir': SPATIAL - len(CORE)}


def test_the_same_counts_without_the_flag_are_refused():
    """The constant-budget contract itself is unchanged."""
    with pytest.raises(ValueError, match='n_budget'):
        make().plan_from_counts({'tir': 900, 'resp': 0}, kept_core=len(CORE))


def test_sampled_plans_are_never_exempt():
    app = make(virtual_sensing_warmup_p=0.10, virtual_sensing_warmup_epochs=20,
               p_virtual_sensing=0.50)
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        plan = app.sample_step(g, epoch=0)
        assert plan.budget_exempt is False
        assert plan.total_tokens == N_BUDGET


def test_exempt_plan_still_validates_tubes_and_unit_counts():
    app = make()
    with pytest.raises(ValueError, match='multiple of T'):
        app.plan_from_counts({'tir': 901, 'resp': 0}, kept_core=len(CORE),
                             exempt_budget=True)
    with pytest.raises(ValueError, match='outside its'):
        app.plan_from_counts({'tir': 1000, 'resp': 0}, kept_core=len(CORE),
                             exempt_budget=True)          # 40 tubes > 36
    with pytest.raises(ValueError, match='outside its'):
        app.plan_from_counts({'tir': 900, 'resp': 300}, kept_core=len(CORE),
                             exempt_budget=True)          # 300 > 250 tokens


def test_a_full_stratified_draw_masks_nothing():
    idx = place_video_stratified(len(CORE), SPATIAL - len(CORE), SPATIAL, T,
                                 CORE, generator=torch.Generator().manual_seed(0),
                                 batch_size=2)
    assert tuple(idx.shape) == (2, SPATIAL * T)
    kept = torch.zeros(2, SPATIAL * T, dtype=torch.long)
    kept.scatter_(1, idx, 1)
    assert int(kept.sum()) == 2 * SPATIAL * T               # every prefix kept
    tubes = kept.view(2, SPATIAL, T)
    assert bool((tubes.sum(dim=2) == T).all())              # every tube complete
