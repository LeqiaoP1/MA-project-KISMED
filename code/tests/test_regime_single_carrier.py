"""ADR-006 three-regime masking: the single-carrier class (gates G1-G10).

Design of record:
``code/notes/design/ADR-006-three-regime-virtual-sensing-masking.md``.
Plan and gate ids:
``code/notes/plan/features/ADR-006-three-regime-masking-plan.md`` (§6).

Two kinds of case live here:

* pure CPU apportioner arithmetic at the ADR's own numbers ($N=350$, $T=25$,
  $m=2$, $M=9$, $C_p=125$, $S_v=14$);
* one model-level case on the small ``MultiModalMAE`` geometry used by the
  other multimae suites, for the contract-7 gradient path of a ZERO-TOKEN
  visual stream.

The compatibility proof (DoD 1/9) is deliberately *not* re-implemented from
scratch: G1 compares the new sampler against the pre-ADR-006 selection path
(one uniform draw per step, then the same private helpers), and
``tests/test_curriculum_schedule.py`` / ``tests/test_budget_masking.py`` stay
unmodified and green on their own.
"""
import math

import pytest
import torch
import yaml

from core import multimae as mm
from core.budget_masking import (MODE_INPAINTING, MODE_VIRTUAL_SENSING,
                                 REGIME_BI_VISUAL, REGIME_INPAINTING,
                                 REGIME_SINGLE, ConstantBudgetApportioner,
                                 StreamSpec)

#: the ADR-006 geometry and the shipped arm's schedule.
N_BUDGET = 350
T = 25
SPATIAL = 36
PHYSIO = 250
C_P = 125
S_V = 14
P_WARM = 0.10
P_TARGET = 0.50
P_S = 0.20
E_WARM = 20


def visual(name='tir'):
    return StreamSpec(name, 'visual', SPATIAL, tokens_per_unit=T)


def signal(name='resp'):
    return StreamSpec(name, 'signal', PHYSIO, tokens_per_unit=1)


def make(**kw):
    streams = kw.pop('streams', None) or [visual('tir'), visual('rgb2d'),
                                          signal('resp')]
    kw.setdefault('n_budget', N_BUDGET)
    return ConstantBudgetApportioner(streams, **kw)


def draw(app, steps, seed=0, **kw):
    gen = torch.Generator().manual_seed(seed)
    return [app.sample_step(generator=gen, **kw) for _ in range(steps)]


def _summary(plan):
    return (plan.mode, plan.mode_by_counts, plan.regime, plan.carrier,
            tuple(sorted(plan.kept_tokens.items())))


# --------------------------------------------------------------------------- #
# G1 -- neutrality: the disabled class consumes no RNG
# --------------------------------------------------------------------------- #
def _pre_change_path(app, steps, seed, epoch=None):
    """The ADR-005 selection path, re-implemented: ONE draw per step.

    A difference here means the new branch moved the RNG stream, which is what
    DoD 1 forbids. The trailing draw is what catches a sample that is CONSUMED
    and then ignored -- a counts-only comparison can pass by luck. The
    threshold is ``p_at(epoch)``: that is what the pre-ADR-006 sampler used.
    """
    gen = torch.Generator().manual_seed(seed)
    p = app.p_at(epoch)
    plans = []
    for _ in range(steps):
        u = float(torch.rand(1, dtype=torch.float64, generator=gen).item())
        plans.append(app._mode1(gen) if u < p else app._mode2(gen))
    nxt = float(torch.rand(1, dtype=torch.float64, generator=gen).item())
    return plans, nxt


@pytest.mark.parametrize('kw', [
    {},                                              # keys entirely absent
    {'single_carrier_tubes': S_V},                    # S_v set, p_s still 0
    {'single_carrier_tubes': S_V, 'p_virtual_sensing': P_TARGET,
     'virtual_sensing_warmup_p': P_WARM,
     'virtual_sensing_warmup_epochs': E_WARM},
])
def test_disabled_class_is_bit_identical_and_does_not_move_the_rng(kw):
    app = make(**{'p_virtual_sensing': 0.50, **kw})
    assert app.p_single_modality == 0.0
    for epoch in (None, 0, E_WARM, 149):
        reference, ref_next = _pre_change_path(app, steps=500, seed=5,
                                               epoch=epoch)
        gen = torch.Generator().manual_seed(5)
        new = [app.sample_step(generator=gen, epoch=epoch) for _ in range(500)]
        assert [_summary(p) for p in new] == [_summary(p) for p in reference], epoch
        assert float(torch.rand(1, dtype=torch.float64,
                                generator=gen).item()) == ref_next, epoch
        assert all(p.regime != REGIME_SINGLE for p in new)


# --------------------------------------------------------------------------- #
# G2 -- the p_s schedule
# --------------------------------------------------------------------------- #
def test_p_s_schedule_is_the_phase_step_function():
    app = make(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
               virtual_sensing_warmup_epochs=E_WARM,
               p_single_modality=P_S, single_modality_warmup_p=0.0)
    assert app.p_s_at(None) == P_S
    assert all(app.p_s_at(e) == 0.0 for e in range(0, E_WARM))
    assert all(app.p_s_at(e) == P_S for e in range(E_WARM, 201))
    values = [app.p_s_at(e) for e in range(0, 201)]
    assert all(b >= a for a, b in zip(values, values[1:]))


# --------------------------------------------------------------------------- #
# G3 -- construction-time validation (ADR-006 §2.6)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('kw,needle', [
    ({'p_single_modality': 0.30, 'single_modality_warmup_p': 0.50,
      'virtual_sensing_warmup_epochs': 10}, 'single_modality_warmup_p'),
    ({'p_single_modality': 1.50}, 'p_single_modality'),
    ({'single_modality_warmup_p': -0.1}, 'single_modality_warmup_p'),
    ({'single_carrier_tubes': 10}, 'single_carrier_tubes'),
    ({'p_single_modality': 0.20,
      'streams': [visual('tir'), signal('resp')]}, 'two visual'),
])
def test_infeasible_single_carrier_tuples_are_rejected(kw, needle):
    with pytest.raises(ValueError) as exc:
        make(**kw)
    assert needle in str(exc.value).lower()


def test_ordering_is_not_enforced_when_the_curriculum_is_off():
    """E_warm = 0 never reads p_s_warm, so it must stay unconstrained."""
    app = make(p_virtual_sensing=0.0, p_single_modality=0.0,
               single_modality_warmup_p=0.9)
    assert app.virtual_sensing_warmup_epochs == 0
    assert app.p_s_at(0) == 0.0


def test_zero_derives_the_sole_carrier_allocation():
    assert make().single_carrier_tubes == S_V
    assert make(single_carrier_tubes=S_V).single_carrier_tubes == S_V


# --------------------------------------------------------------------------- #
# G4 -- the realized regime mix
# --------------------------------------------------------------------------- #
def test_regime_mix_follows_the_two_schedules():
    app = make(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
               virtual_sensing_warmup_epochs=E_WARM,
               p_single_modality=P_S, single_modality_warmup_p=0.0)
    n = 10_000
    phase1 = draw(app, steps=n, seed=7, epoch=E_WARM // 4)
    assert sum(p.regime == REGIME_SINGLE for p in phase1) == 0

    phase2 = draw(app, steps=n, seed=7, epoch=E_WARM + 10)
    singles = [p for p in phase2 if p.regime == REGIME_SINGLE]
    f_single = len(singles) / n
    p_joint = P_TARGET * P_S
    assert abs(f_single - p_joint) <= 4 * math.sqrt(
        p_joint * (1 - p_joint) / n), f_single
    # 3A and 3B are equiprobable INSIDE the class.
    a = sum(p.carrier == 'rgb2d' for p in singles)
    b = sum(p.carrier == 'tir' for p in singles)
    assert abs(a - b) <= 4 * math.sqrt(len(singles) * 0.25), (a, b)
    # The three classes partition the run.
    regimes = {p.regime for p in phase2}
    assert regimes == {REGIME_INPAINTING, REGIME_BI_VISUAL, REGIME_SINGLE}


# --------------------------------------------------------------------------- #
# G5 -- allocation exactness on a single-carrier step
# --------------------------------------------------------------------------- #
def test_single_carrier_holds_the_whole_budget_and_zeroes_the_rest():
    app = make(p_virtual_sensing=1.0, p_single_modality=1.0,
               single_carrier_tubes=S_V)
    plans = draw(app, steps=200, seed=11)
    seen = set()
    for plan in plans:
        assert plan.regime == REGIME_SINGLE
        assert plan.total_tokens == N_BUDGET
        carrier = plan.carrier
        seen.add(carrier)
        assert plan.kept_units[carrier] == S_V
        assert plan.kept_tokens[carrier] == S_V * T
        for name in ('tir', 'rgb2d', 'resp'):
            if name != carrier:
                assert plan.kept_tokens[name] == 0
                assert plan.kept_units[name] == 0
    assert seen == {'tir', 'rgb2d'}


def test_the_class_is_subordinate_to_the_sensor_absent_draw():
    """ADR-006 §2.7: p_s is a share OF the sensor-absent steps, not of all."""
    app = make(p_virtual_sensing=0.5, p_single_modality=1.0)
    n = 10_000
    plans = draw(app, steps=n, seed=13)
    f = sum(p.regime == REGIME_SINGLE for p in plans) / n
    assert abs(f - 0.5) <= 4 * math.sqrt(0.25 / n), f


# --------------------------------------------------------------------------- #
# G6 -- the labels are additive, never redefinitions
# --------------------------------------------------------------------------- #
def test_single_carrier_keeps_the_adr005_labels_intact():
    app = make(p_virtual_sensing=1.0, p_single_modality=1.0)
    plan = draw(app, steps=1, seed=3)[0]
    assert plan.regime == REGIME_SINGLE              # new label
    assert plan.carrier in ('tir', 'rgb2d')
    assert plan.mode_by_counts == MODE_VIRTUAL_SENSING   # ADR-005 meaning
    assert plan.mode == MODE_VIRTUAL_SENSING

    bi = draw(make(p_virtual_sensing=1.0), steps=1, seed=3)[0]
    assert bi.regime == REGIME_BI_VISUAL
    assert bi.carrier is None
    assert bi.mode_by_counts == MODE_VIRTUAL_SENSING


# --------------------------------------------------------------------------- #
# G8 -- explicit evaluation pins resolve the regime
# --------------------------------------------------------------------------- #
def test_plan_from_counts_labels_single_carrier_allocations():
    app = make()
    plan = app.plan_from_counts({'tir': N_BUDGET, 'rgb2d': 0, 'resp': 0})
    assert plan.regime == REGIME_SINGLE and plan.carrier == 'tir'
    assert plan.total_tokens == N_BUDGET

    plan = app.plan_from_counts({'tir': 0, 'rgb2d': N_BUDGET, 'resp': 0})
    assert plan.regime == REGIME_SINGLE and plan.carrier == 'rgb2d'


def test_plan_from_counts_keeps_the_multi_carrier_regimes():
    app = make()
    bi = app.plan_from_counts({'tir': 175, 'rgb2d': 175, 'resp': 0})
    assert bi.regime == REGIME_BI_VISUAL and bi.carrier is None
    inp = app.plan_from_counts({'tir': 75, 'rgb2d': 150, 'resp': 125})
    assert inp.regime == REGIME_INPAINTING and inp.carrier is None


def test_plan_from_counts_still_enforces_the_multi_carrier_floor():
    """A 9-tube / 1-tube split is a multi-carrier plan and keeps its floor."""
    with pytest.raises(ValueError):
        make().plan_from_counts({'tir': 225, 'rgb2d': 25, 'resp': 100})


# --------------------------------------------------------------------------- #
# G7 -- a zeroed visual stream keeps a gradient path (contract 7)
# --------------------------------------------------------------------------- #
def _model(**over):
    kw = dict(streams=('tir', 'rgb2d', 'resp'), embed_dim=32, enc_depth=1,
              enc_num_heads=2, dec_depth=1, tubelet=(2, 16, 16),
              input_size=64, num_frames=8, sig_kernel=8, seq_len=32,
              fps=25.0, fs=100.0, temporal_stride=1, n_budget=20)
    kw.update(over)
    return mm.MultiModalMAE(**kw)


def _batch(model, B=2):
    v = (B, 3, model.num_frames, model.input_size, model.input_size)
    return {'tir': torch.rand(*v), 'rgb2d': torch.rand(*v),
            'resp': torch.randn(B, 1, model.seq_len)}


def test_zeroed_visual_stream_still_reaches_the_encoder():
    torch.manual_seed(0)
    m = _model(p_virtual_sensing=1.0, p_single_modality=1.0)
    assert m.single_carrier_tubes == m.n_budget // m.specs[0].tokens_per_unit
    plan = m.plan_from_counts({'tir': m.n_budget, 'rgb2d': 0, 'resp': 0})
    assert plan.regime == REGIME_SINGLE and plan.carrier == 'tir'

    batch = _batch(m)
    captured = {}

    def _keep(_mod, _inp, out):
        out.retain_grad()
        captured['z'] = out

    hook = m.enc_norm.register_forward_hook(_keep)
    try:
        out = m(batch, plan=plan)
        assert out['regime'] == REGIME_SINGLE
        assert out['carrier'] == 'tir'
        assert out['kept']['rgb2d'] == 0 and out['kept']['resp'] == 0
        assert tuple(captured['z'].shape) == (2, m.n_budget, m.embed_dim)
        # The ZEROED visual stream is still a reconstruction target, and its
        # loss must reach the shared encoder through the decoder conditioning.
        out['losses']['rgb2d'].backward(retain_graph=True)
        assert float(captured['z'].grad.abs().sum()) > 0.0
        captured['z'].grad = None
        out['losses']['resp'].backward()
        assert float(captured['z'].grad.abs().sum()) > 0.0
    finally:
        hook.remove()


# --------------------------------------------------------------------------- #
# G9/G10 -- the arm configuration resolves and differs by ONE key
# --------------------------------------------------------------------------- #
CONFIGS = ('stage2_budget_3regime_single_modality.yaml',)


def _load(name):
    with open(f'configs/pretrain/{name}') as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize('name', CONFIGS)
def test_arm_config_carries_the_adr006_deltas(name):
    cfg = _load(name)
    assert cfg['streams'] == 'tir,rgb2d,resp'
    assert cfg['n_budget'] == N_BUDGET
    assert cfg['p_single_modality'] == P_S
    assert cfg['single_modality_warmup_p'] == 0.0
    assert cfg['single_carrier_tubes'] == S_V
    assert cfg['single_carrier_tubes'] == cfg['n_budget'] // T
    # E_warm is NOT part of the ADR-006 delta: it matches the ADR-005 arm so the
    # step class is the only difference between the two arms.
    assert cfg['virtual_sensing_warmup_epochs'] == 50

