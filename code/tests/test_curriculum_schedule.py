"""ADR-005 virtual-sensing curriculum and contract-4 mode label (gates G1-G7).

Design of record:
``code/notes/design/archive/ADR-005-asymmetric-multimodal-masking-3streams.md``.
Plan and gate ids:
``code/notes/plan/features/archive/ADR-005-asymmetric-multimodal-masking-3streams-plan.md``
(§6, G1-G7).

The apportioner cases are pure CPU arithmetic; the last three cases build the
small ``MultiModalMAE`` geometry used by ``test_multimae_rgb2d.py`` (32-d,
one block, 20-token budget) so the epoch plumbing is exercised end to end
without a dataset or a GPU.
"""
import math

import pytest
import torch

from core import multimae as mm
from core.budget_masking import (MODE_INPAINTING, MODE_VIRTUAL_SENSING,
                                 ConstantBudgetApportioner, StreamSpec)
from engines.pretrain import evaluate_pretrain

#: the ADR's geometry and shipped schedule.
N_BUDGET = 350
T = 25
C_P = 125
SPATIAL = 36
PHYSIO = 250
P_WARM = 0.10
P_TARGET = 0.50
E_WARM = 50
#: Mode 2 draws K_tilde uniformly on [0, 125] inclusive -- the ONLY draw that
#: yields K_resp = 0, which is why contract 4 predicts a 1/126 relabel rate.
MODE2_DRAWS = C_P + 1


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


class _Scripted(ConstantBudgetApportioner):
    """Engine whose integer draws are scripted (deterministic mode checks)."""

    def __init__(self, draws, **kw):
        super().__init__(**kw)
        self._script = list(draws)

    def _randint(self, lo, hi, generator):
        value = self._script.pop(0)
        assert lo <= value <= hi, (value, lo, hi)
        return value


# --------------------------------------------------------------------------- #
# G1 -- E_warm = 0 is bit-identical to the pre-ADR-005 draw
# --------------------------------------------------------------------------- #
def _pre_change_draw(app, steps, seed):
    """The ADR-002 selection path, re-implemented exactly.

    One float64 draw per step, compared against ``p_virtual_sensing`` (no
    epoch, no schedule), then the same private helpers the sampler uses. A
    difference here means the curriculum changed the RNG stream, which is what
    DoD 1 forbids.
    """
    gen = torch.Generator().manual_seed(seed)
    plans = []
    for _ in range(steps):
        u = float(torch.rand(1, dtype=torch.float64, generator=gen).item())
        plans.append(app._mode1(gen) if u < app.p_virtual_sensing
                     else app._mode2(gen))
    return plans


def test_e_warm_zero_replays_the_pre_change_sequence():
    app = make(p_virtual_sensing=0.50)          # E_warm = 0 by signature
    assert app.virtual_sensing_warmup_epochs == 0
    reference = _pre_change_draw(app, steps=500, seed=5)
    # Explicit epochs of every phase must not move the draw, and neither may
    # the no-epoch call.
    for epoch in (0, 7, 149):
        assert draw(app, steps=500, seed=5, epoch=epoch) == reference
    assert draw(app, steps=500, seed=5) == reference
    # The resolved label is additive: the sampled label is untouched.
    assert reference[0].mode in (MODE_VIRTUAL_SENSING, MODE_INPAINTING)


# --------------------------------------------------------------------------- #
# G2 -- schedule resolution and monotonicity
# --------------------------------------------------------------------------- #
def test_schedule_resolution_and_monotonicity():
    app = make(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
               virtual_sensing_warmup_epochs=E_WARM)
    assert app.p_at(None) == P_TARGET
    assert all(app.p_at(e) == P_WARM for e in range(0, E_WARM))
    assert all(app.p_at(e) == P_TARGET for e in range(E_WARM, 201))
    values = [app.p_at(e) for e in range(0, 201)]
    assert all(b >= a for a, b in zip(values, values[1:]))


# --------------------------------------------------------------------------- #
# G3 -- construction-time validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('kw,needle', [
    ({'virtual_sensing_warmup_p': 0.60,
      'virtual_sensing_warmup_epochs': 10}, 'virtual_sensing_warmup_p'),
    ({'virtual_sensing_warmup_epochs': -1}, 'virtual_sensing_warmup_epochs'),
])
def test_invalid_curriculum_is_rejected_at_construction(kw, needle):
    with pytest.raises(ValueError) as exc:
        make(p_virtual_sensing=P_TARGET, **kw)
    assert needle in str(exc.value)


def test_ordering_is_not_enforced_when_the_curriculum_is_off():
    """E_warm = 0 never reads p_warm, so p_target = 0.0 must stay legal.

    This is the neutrality guarantee of DoD 1: an ADR-002 configuration that
    sets ``p_virtual_sensing: 0.0`` and never mentions the new keys must keep
    building.
    """
    app = make(p_virtual_sensing=0.0)          # p_warm keeps its 0.10 default
    assert app.virtual_sensing_warmup_epochs == 0
    assert app.p_at(0) == 0.0


def test_e_warm_zero_is_accepted_and_pins_the_target():
    app = make(p_virtual_sensing=P_TARGET)
    assert all(app.p_at(e) == P_TARGET for e in (0, 1, 149, 10_000))


# --------------------------------------------------------------------------- #
# G4 -- the Mode-1 fraction reproduces p(e)
# --------------------------------------------------------------------------- #
def test_mode1_fraction_follows_the_schedule():
    app = make(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
               virtual_sensing_warmup_epochs=E_WARM)
    n = 10_000
    for epoch, p in ((E_WARM // 5, P_WARM), (E_WARM + 10, P_TARGET)):
        plans = draw(app, steps=n, seed=epoch + 1, epoch=epoch)
        f_mode = sum(pl.mode == MODE_VIRTUAL_SENSING for pl in plans) / n
        f_counts = sum(pl.mode_by_counts == MODE_VIRTUAL_SENSING
                       for pl in plans) / n
        assert abs(f_mode - p) <= 4 * math.sqrt(p * (1 - p) / n), (
            epoch, f_mode, p)
        # Contract 4 adds the K_tilde = 0 draws of Mode 2 to the same label.
        expected = p + (1.0 - p) / MODE2_DRAWS
        sd = math.sqrt(expected * (1.0 - expected) / n)
        assert abs(f_counts - expected) <= 4 * sd, (epoch, f_counts, expected)


# --------------------------------------------------------------------------- #
# G5 -- contract 4: the resolved label is a function of the counts
# --------------------------------------------------------------------------- #
def test_mode_by_counts_is_the_contract4_label_over_random_draws():
    app = make(p_virtual_sensing=0.5)
    for plan in draw(app, steps=2_000, seed=21):
        expected = (MODE_VIRTUAL_SENSING
                    if all(plan.kept_tokens[s.name] == 0 for s in app.signal)
                    else MODE_INPAINTING)
        assert plan.mode_by_counts == expected


def test_zero_draw_is_relabelled_but_keeps_the_sampled_mode():
    """>The K_tilde = 0 draw: sampled Mode 2, contract-4 Mode 1."""
    app = _Scripted([0, 7],
                    streams=[visual('tir'), visual('rgb2d'), signal('resp')],
                    n_budget=N_BUDGET, p_virtual_sensing=0.0)
    plan = app.sample_step(generator=torch.Generator().manual_seed(0))
    assert plan.kept_tokens['resp'] == 0
    assert plan.mode == MODE_INPAINTING
    assert plan.mode_by_counts == MODE_VIRTUAL_SENSING


def test_a_kept_physiological_token_is_mode2_by_counts():
    app = _Scripted([25, 7],
                    streams=[visual('tir'), visual('rgb2d'), signal('resp')],
                    n_budget=N_BUDGET, p_virtual_sensing=0.0)
    plan = app.sample_step(generator=torch.Generator().manual_seed(0))
    assert plan.kept_tokens['resp'] == 25
    assert plan.mode == MODE_INPAINTING
    assert plan.mode_by_counts == MODE_INPAINTING


def test_explicit_plans_get_the_same_resolved_label():
    app = make()
    mode1 = app.plan_from_counts({'tir': 175, 'rgb2d': 175, 'resp': 0})
    mode2 = app.plan_from_counts({'tir': 175, 'rgb2d': 150, 'resp': 25})
    assert (mode1.mode, mode1.mode_by_counts) == (
        MODE_VIRTUAL_SENSING, MODE_VIRTUAL_SENSING)
    assert (mode2.mode, mode2.mode_by_counts) == (
        MODE_INPAINTING, MODE_INPAINTING)


# --------------------------------------------------------------------------- #
# G6/G7 -- the model forwards the epoch and keeps the gradient path
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


def test_model_forwards_the_epoch_to_the_policy():
    m = _model(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
               virtual_sensing_warmup_epochs=E_WARM)
    n = 300
    for epoch, p in ((0, P_WARM), (E_WARM, P_TARGET)):
        hits = 0
        for i in range(n):
            _, plan, _ = m.make_masks(
                2, torch.device('cpu'),
                counts_generator=torch.Generator().manual_seed(1000 + i),
                epoch=epoch)
            hits += int(plan.mode == MODE_VIRTUAL_SENSING)
        fraction = hits / n
        assert abs(fraction - p) <= 4 * math.sqrt(p * (1 - p) / n), (
            epoch, fraction, p)


def test_forward_payload_exposes_the_schedule_and_keeps_the_budget():
    torch.manual_seed(0)
    m = _model(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
               virtual_sensing_warmup_epochs=E_WARM)
    batch = _batch(m)
    for epoch, expected_p in ((0, P_WARM), (E_WARM, P_TARGET)):
        captured = {}
        hook = m.enc_norm.register_forward_hook(
            lambda _mod, _inp, out: captured.setdefault('z', out))
        try:
            out = m(batch,
                    counts_generator=torch.Generator().manual_seed(0),
                    epoch=epoch)
        finally:
            hook.remove()
        assert tuple(captured['z'].shape) == (2, m.n_budget, m.embed_dim)
        assert sum(out['kept'].values()) == m.n_budget
        assert out['p_virtual'] == pytest.approx(expected_p)
        assert out['mode_resolved'] in (MODE_VIRTUAL_SENSING,
                                        MODE_INPAINTING)
        assert set(out['kept_units']) == set(m.streams)


def test_resp_loss_reaches_the_encoder_when_fully_masked():
    """Contract 7 / R3: K_resp = 0 still produces a non-zero dL_resp/dz."""
    torch.manual_seed(0)
    m = _model(p_virtual_sensing=1.0)          # Mode 1 on every step
    batch = _batch(m)
    captured = {}

    def _keep(_mod, _inp, out):
        out.retain_grad()
        captured['z'] = out

    hook = m.enc_norm.register_forward_hook(_keep)
    try:
        out = m(batch, counts_generator=torch.Generator().manual_seed(0))
        assert out['kept']['resp'] == 0
        assert out['mode_resolved'] == MODE_VIRTUAL_SENSING
        out['losses']['resp'].backward()
    finally:
        hook.remove()
    grad = captured['z'].grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


def test_validation_replays_within_a_phase_and_only_moves_the_threshold():
    """§2.7: the validation DRAW is pinned; only p(e) follows the phase."""
    torch.manual_seed(0)
    m = _model(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
               virtual_sensing_warmup_epochs=E_WARM)
    loader = [_batch(m) for _ in range(4)]
    cpu = torch.device('cpu')
    a = evaluate_pretrain(m, loader, cpu, mask_base=7, epoch=1)
    b = evaluate_pretrain(m, loader, cpu, mask_base=7, epoch=2)
    # Both epochs are in Phase I, so nothing in the record may move.
    assert a['kept'] == b['kept']
    assert a['mode'] == b['mode']
    assert a['resolved'] == b['resolved']
    # Phase II re-uses the same draw sequence, so Mode 1 can only gain steps.
    c = evaluate_pretrain(m, loader, cpu, mask_base=7, epoch=E_WARM)
    assert (c['resolved'].get('mode1', 0.0)
            >= a['resolved'].get('mode1', 0.0))
    assert (c['mode'].get('mode1', 0.0) >= a['mode'].get('mode1', 0.0))


# --------------------------------------------------------------------------- #
# Manifest -- the policy is not checkpointed (plan §4.4 / Trap 4)
# --------------------------------------------------------------------------- #
def test_masking_manifest_is_written_and_guards_a_resume(tmp_path):
    import argparse

    from runners.run_pretrain import write_masking_manifest

    def args_for(out):
        return argparse.Namespace(output_dir=str(out), epochs=150,
                                  clip_duration=8.0)

    plan = argparse.Namespace(fold=0, digest='deadbeef')
    stage1 = {'path': '', 'loaded': 0, 'random': True, 'patch_stream': ''}
    m = _model(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
               virtual_sensing_warmup_epochs=E_WARM)

    first = write_masking_manifest(args_for(tmp_path), m, plan=plan,
                                   mask_base=42, stage1=stage1)
    assert (tmp_path / 'masking.json').exists()
    assert first['policy']['virtual_sensing_warmup_epochs'] == E_WARM
    assert first['kernel']['random_init_visuals'] == ['tir', 'rgb2d']

    # An identical record is accepted and left untouched.
    again = write_masking_manifest(args_for(tmp_path), m, plan=plan,
                                   mask_base=42, stage1=stage1)
    assert again['policy'] == first['policy']

    # A different schedule is REFUSED rather than silently overwritten: the
    # resumed run would replay a different mask sequence.
    other = _model(p_virtual_sensing=P_TARGET, virtual_sensing_warmup_p=P_WARM,
                   virtual_sensing_warmup_epochs=E_WARM + 10)
    with pytest.raises(SystemExit, match='masking policy'):
        write_masking_manifest(args_for(tmp_path), other, plan=plan,
                               mask_base=42, stage1=stage1)
