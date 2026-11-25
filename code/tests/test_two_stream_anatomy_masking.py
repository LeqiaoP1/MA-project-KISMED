"""ADR-007 gates G1-G13: the two-stream anatomy-aware constant-budget policy.

Design of record:
``code/notes/design/ADR-007-two-stream-anatomy-aware-constant-budget-masking.md``.
Plan + gate ids:
``code/notes/plan/features/ADR-007-two-stream-anatomy-aware-masking-plan.md``.

The gates are read at the ADR's own numbers: ``N = 200``, ``T = 25``, 36 spatial
tubes, 250 physiological tokens (``R = 10``), ``C_p = 100``, ``S_tir in [4, 8]``,
``A_core = {14, 15, 20, 21}`` and ``S_core in {1, 2}``.

Nothing here is a GPU test: the apportioner and the placement are pure CPU code,
and the model gates run a deliberately tiny encoder (``enc_depth=1``) whose only
job is to produce the ``[B, 200, D]`` contract and a gradient.
"""
import argparse
import collections
import os
import subprocess
import sys

import pytest
import torch

from core.budget_masking import (MODE_INPAINTING, MODE_VIRTUAL_SENSING,
                                 REGIME_INPAINTING, REGIME_SINGLE,
                                 REGIME_SINGLE_VISUAL, SPATIAL_ANATOMY_AWARE,
                                 SPATIAL_POLICY_SINGLE_VISUAL, StreamSpec,
                                 ConstantBudgetApportioner)
from core.mask_placement import (counts_generator, place_video_stratified,
                                 placement_generator)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = 'configs/pretrain/stage2_budget_tir_resp_2stream_anatomy.yaml'

N_BUDGET = 200
T = 25
SPATIAL = 36
PHYSIO = 250
C_P = 100
CORE = (14, 15, 20, 21)
S_TIR_MIN = 4
S_TIR_MAX = 8


def specs():
    return [StreamSpec('tir', 'visual', SPATIAL, T, spatial_grid=(6, 6)),
            StreamSpec('resp', 'signal', PHYSIO, 1)]


def make(**kw):
    """The ADR-007 arm's tuple; ``kw`` overrides one key at a time."""
    d = dict(n_budget=N_BUDGET, video_cap_ratio=0.25, physio_cap_ratio=0.40,
             video_min_tubes=S_TIR_MIN,
             spatial_sampling=SPATIAL_ANATOMY_AWARE, physio_draw_min=1)
    d.update(kw)
    return ConstantBudgetApportioner(specs(), **d)


def draw(app, steps, seed=0, epoch=None):
    g = torch.Generator().manual_seed(int(seed))
    return [app.sample_step(g, epoch=epoch) for _ in range(steps)]


def signature(plan):
    return (int(plan.mode), int(plan.mode_by_counts), plan.regime,
            tuple(sorted(plan.kept_tokens.items())),
            tuple(sorted(plan.kept_units.items())))


# --------------------------------------------------------------------------- #
# G1 - constant budget and the [B, 200, D] encoder input
# --------------------------------------------------------------------------- #
def test_g1_budget_is_constant_on_every_step_of_both_phases():
    app = make(virtual_sensing_warmup_p=0.10, virtual_sensing_warmup_epochs=20,
               p_virtual_sensing=0.50)
    for epoch in (0, 20, 149):                    # Phase I, boundary, Phase II
        for plan in draw(app, steps=20_000, seed=epoch, epoch=epoch):
            assert plan.total_tokens == N_BUDGET
            assert plan.kept_tokens['tir'] + plan.kept_tokens['resp'] == N_BUDGET
            assert plan.kept_tokens['tir'] % T == 0


def test_g1_encoder_input_is_b200d():
    """The dense `[B, 200, 768]` contract, on the real default embed width."""
    from core.multimae import MultiModalMAE
    m = MultiModalMAE(streams=('tir', 'resp'), enc_depth=1, dec_depth=1,
                      enc_num_heads=4, tubelet=(2, 16, 16), input_size=96,
                      num_frames=50, sig_kernel=32, seq_len=8000, fps=25.0,
                      fs=1000.0, temporal_stride=4, n_budget=N_BUDGET,
                      video_cap_ratio=0.25, physio_cap_ratio=0.40,
                      video_min_tubes=S_TIR_MIN, p_virtual_sensing=1.0,
                      spatial_sampling=SPATIAL_ANATOMY_AWARE,
                      physio_draw_min=1)
    B = 2
    x = {'tir': torch.randn(B, 3, 50, 96, 96), 'resp': torch.randn(B, 1, 8000)}
    for epoch in (0, 25):
        out = m(x, counts_generator=torch.Generator().manual_seed(epoch),
                placement_generator=torch.Generator().manual_seed(100 + epoch),
                epoch=epoch)
        assert out['n_batch'] == N_BUDGET          # z.shape[1], asserted in forward
        assert m.embed_dim == 768
        for s in m.streams:
            assert s in out['losses_mse']          # loss-key stability


# --------------------------------------------------------------------------- #
# G2 - mode/bound compliance and the exact Mode-2 distribution
# --------------------------------------------------------------------------- #
def test_g2_mode1_is_the_whole_budget_on_the_sole_visual():
    plans = draw(make(p_virtual_sensing=1.0), steps=2_000, seed=1)
    for plan in plans:
        assert plan.mode == MODE_VIRTUAL_SENSING
        assert plan.mode_by_counts == MODE_VIRTUAL_SENSING
        assert plan.regime == REGIME_SINGLE_VISUAL
        assert plan.kept_tokens == {'tir': 200, 'resp': 0}
        assert plan.kept_units == {'tir': 8, 'resp': 0}


def test_g2_mode2_is_uniform_over_25_50_75_100_with_zero_leakage():
    steps = 100_000
    plans = draw(make(p_virtual_sensing=0.0), steps=steps, seed=2)
    counter = collections.Counter(plan.kept_tokens['resp'] for plan in plans)
    assert set(counter) == {25, 50, 75, 100}
    for k, count in counter.items():
        assert abs(count / steps - 0.25) <= 0.01, (k, count / steps)
    for plan in plans:
        assert plan.mode == MODE_INPAINTING
        assert plan.mode_by_counts == MODE_INPAINTING
        assert plan.regime == REGIME_INPAINTING
        assert plan.kept_tokens['resp'] > 0            # no Mode-2 -> Mode-1 leak
        assert plan.kept_units['tir'] == (N_BUDGET - plan.kept_tokens['resp']) // T
        assert plan.kept_units['tir'] in (4, 5, 6, 7)


# --------------------------------------------------------------------------- #
# G3 - stratification: certain core hit, disjoint strata, per-sample diversity
# --------------------------------------------------------------------------- #
def test_g3_every_sample_keeps_one_or_two_core_tubes():
    app = make(p_virtual_sensing=0.5)
    B = 80
    steps = 1_250                                    # 10^5 sample masks
    seen_core = collections.Counter()
    tube_totals = collections.Counter()
    for step in range(steps):
        plan = app.sample_step(counts_generator(1234, 0, step), epoch=0)
        s_core = int(plan.core_units['tir'])
        s_periph = int(plan.periph_units['tir'])
        assert s_core in (1, 2)
        assert s_core + s_periph == int(plan.kept_units['tir'])
        idx = place_video_stratified(
            s_core, s_periph, SPATIAL, T, CORE,
            generator=placement_generator(1234, 0, step, rank=0),
            batch_size=B)
        assert idx.shape == (B, (s_core + s_periph) * T)
        spatial = idx % SPATIAL
        core_hits = torch.isin(spatial, torch.tensor(CORE)).sum(dim=1)
        assert bool(torch.all(core_hits == s_core * T)), s_core
        # no duplicates inside a row (sorted, strictly increasing)
        assert bool(torch.all(idx[:, 1:] > idx[:, :-1]))
        seen_core[s_core] += 1
        for row in idx.tolist():
            # encoder index = t*n_units + s, so ``i % SPATIAL`` is the tube.
            tubes = sorted({i % SPATIAL for i in row})
            assert len(tubes) == s_core + s_periph
            tube_totals[(len([p for p in tubes if p in CORE]), len(tubes))] += 1
    assert seen_core[1] > 0 and seen_core[2] > 0
    assert all(core_n in (1, 2) for core_n, _ in tube_totals)


def test_g3_placement_differs_across_samples_of_one_step():
    idx = place_video_stratified(1, 7, SPATIAL, T, CORE,
                                 generator=torch.Generator().manual_seed(7),
                                 batch_size=8)
    assert len({tuple(row) for row in idx.tolist()}) > 1


# --------------------------------------------------------------------------- #
# G4 - the two-phase curriculum
# --------------------------------------------------------------------------- #
def test_g4_schedule_and_realized_shares():
    app = make(virtual_sensing_warmup_p=0.10, virtual_sensing_warmup_epochs=20,
               p_virtual_sensing=0.50)
    assert app.p_at(19) == pytest.approx(0.10)
    assert app.p_at(20) == pytest.approx(0.50)
    assert app.p_at(None) == pytest.approx(0.50)

    for epoch, p in ((19, 0.10), (20, 0.50)):
        steps = 10_000
        plans = draw(app, steps=steps, seed=epoch, epoch=epoch)
        share = sum(plan.mode_by_counts == MODE_VIRTUAL_SENSING
                    for plan in plans) / steps
        sigma = (p * (1 - p) / steps) ** 0.5
        assert abs(share - p) <= 4 * sigma, (epoch, share, p)


# --------------------------------------------------------------------------- #
# G5 - determinism from (seed, epoch, step), diversity within a batch
# --------------------------------------------------------------------------- #
def test_g5_replaying_a_key_reproduces_the_plan():
    app = make()
    a = [signature(p) for p in
         [app.sample_step(counts_generator(99, 3, s), epoch=3) for s in range(200)]]
    b = [signature(p) for p in
         [app.sample_step(counts_generator(99, 3, s), epoch=3) for s in range(200)]]
    c = [signature(p) for p in
         [app.sample_step(counts_generator(99, 4, s), epoch=4) for s in range(200)]]
    assert a == b
    assert a != c


def test_g5_an_independent_process_derives_the_same_counts():
    code = (
        'import torch\n'
        'from core.budget_masking import StreamSpec, ConstantBudgetApportioner\n'
        'from core.mask_placement import counts_generator\n'
        'specs=[StreamSpec("tir","visual",36,25,(6,6)),StreamSpec("resp","signal",250,1)]\n'
        'app=ConstantBudgetApportioner(specs,n_budget=200,video_cap_ratio=0.25,'
        'physio_cap_ratio=0.40,video_min_tubes=4,spatial_sampling="anatomy_aware",'
        'physio_draw_min=1)\n'
        'print([app.sample_step(counts_generator(7,5,s),epoch=5).kept_tokens["resp"]'
        ' for s in range(20)])\n')
    out = subprocess.run([sys.executable, '-c', code], cwd=CODE_DIR,
                         capture_output=True, text=True, check=True).stdout
    remote = [int(x) for x in out.strip().strip('[]').split(',')]
    app = make()
    local = [app.sample_step(counts_generator(7, 5, s), epoch=5)
             .kept_tokens['resp'] for s in range(20)]
    assert remote == local


# --------------------------------------------------------------------------- #
# G6 - the zero-sensor gradient path
# --------------------------------------------------------------------------- #
def _tiny_model(**kw):
    from core.multimae import MultiModalMAE
    d = dict(streams=('tir', 'resp'), embed_dim=48, enc_depth=2,
             enc_num_heads=4, dec_depth=1, tubelet=(2, 16, 16), input_size=96,
             num_frames=50, sig_kernel=32, seq_len=8000, fps=25.0, fs=1000.0,
             temporal_stride=4, n_budget=N_BUDGET, video_cap_ratio=0.25,
             physio_cap_ratio=0.40, video_min_tubes=S_TIR_MIN,
             p_virtual_sensing=1.0)
    d.update(kw)
    return MultiModalMAE(**d)


def test_g6_mode1_resp_loss_reaches_the_encoder():
    torch.manual_seed(0)
    m = _tiny_model(spatial_sampling=SPATIAL_ANATOMY_AWARE, physio_draw_min=1)
    x = {'tir': torch.randn(2, 3, 50, 96, 96), 'resp': torch.randn(2, 1, 8000)}
    out = m(x, counts_generator=torch.Generator().manual_seed(0),
            placement_generator=torch.Generator().manual_seed(1), epoch=0)
    assert out['kept']['resp'] == 0 and out['kept']['tir'] == N_BUDGET
    assert float(out['losses_mse']['resp'].detach()) > 0.0
    out['losses_mse']['resp'].backward()
    grads = [p.grad for p in m.enc_blocks.parameters() if p.grad is not None]
    assert grads, 'the shared encoder received no gradient at all'
    assert max(float(g.abs().max()) for g in grads) > 0.0


# --------------------------------------------------------------------------- #
# G7 - feasibility, in normative rule order
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('kw,needle', [
    ({'n_budget': 210}, 'condition 1'),
    ({'physio_cap_ratio': 0.05}, 'condition 2'),
    ({'n_budget': 350, 'video_min_tubes': 2}, 'condition 4'),
    ({'video_min_tubes': 2}, 'condition 5'),
    ({'core_tubes_max': 5}, 'condition 5b'),
    ({'periph_min_tubes': 0}, 'condition 5b'),
    ({'core_row_bounds': (2, 9)}, 'core_row_bounds'),
    ({'core_col_bounds': (3, 1)}, 'core_col_bounds'),
    ({'p_single_modality': 0.20}, 'p_single_modality'),
    ({'spatial_sampling': 'stratified'}, 'spatial_sampling'),
])
def test_g7_infeasible_tuples_raise_at_construction(kw, needle):
    with pytest.raises(ValueError) as exc:
        make(**kw)
    assert needle in str(exc.value)


def test_g7_anatomy_aware_needs_a_two_dimensional_grid():
    bad = [StreamSpec('tir', 'visual', SPATIAL, T),
           StreamSpec('resp', 'signal', PHYSIO, 1)]
    with pytest.raises(ValueError, match='spatial_grid'):
        ConstantBudgetApportioner(bad, n_budget=N_BUDGET, video_cap_ratio=0.25,
                                  physio_cap_ratio=0.40,
                                  video_min_tubes=S_TIR_MIN,
                                  spatial_sampling=SPATIAL_ANATOMY_AWARE)


def test_g7_a_non_multiple_physiological_cap_is_legal():
    """§2.6(5) uses FLOOR division on purpose (plan §1.2)."""
    app = make(physio_cap_ratio=0.45, video_min_tubes=3, p_virtual_sensing=0.0)
    assert app.physio_cap_tokens['resp'] == 112
    assert app.s_tir_min == 3
    app = make(physio_cap_ratio=0.45, video_min_tubes=3, p_virtual_sensing=0.0,
               spatial_sampling='uniform')
    plans = draw(app, steps=2_000, seed=3)
    assert min(plan.kept_units['tir'] for plan in plans) >= 4
    assert max(plan.kept_tokens['resp'] for plan in plans) <= 100


def test_g7_one_visual_stream_declares_one_physiological_stream():
    with pytest.raises(ValueError, match='exactly ONE physiological'):
        ConstantBudgetApportioner(
            [StreamSpec('tir', 'visual', SPATIAL, T, (6, 6)),
             StreamSpec('bp', 'signal', PHYSIO, 1),
             StreamSpec('resp', 'signal', PHYSIO, 1)],
            n_budget=N_BUDGET, video_cap_ratio=0.25, physio_cap_ratio=0.20,
            video_min_tubes=S_TIR_MIN)


# --------------------------------------------------------------------------- #
# G8 - the 2-visual mask sequence is still bit-identical
# --------------------------------------------------------------------------- #
def test_g8_two_visual_neutrality_fixture():
    from scripts.project.capture_adr007_neutrality import digest_of
    import json
    from scripts.project import capture_adr007_neutrality as cap
    with open(cap.FIXTURE) as fh:
        ref = json.load(fh)
    for ctx, want in ref['epoch_contexts'].items():
        epoch = None if ctx == 'None' else int(ctx)
        got = digest_of(epoch, steps=int(want['n_steps']))
        assert got['digest'] == want['digest'], ctx
        assert got['next_draw'] == want['next_draw'], ctx
        assert got['trailing_plan'] == want['trailing_plan'], ctx


# --------------------------------------------------------------------------- #
# G9 - the arm configuration resolves and carries the adr tuple
# --------------------------------------------------------------------------- #
def test_g9_arm_config_resolves_the_adr_tuple():
    import importlib
    old = sys.argv
    sys.argv = ['prog', '-c', os.path.join(CODE_DIR, CONFIG)]
    try:
        a = importlib.import_module('runners.run_pretrain').get_args()
    finally:
        sys.argv = old
    assert a.data_set == 'tir_roi'
    assert a.streams == 'tir,resp'
    assert a.n_budget == 200 and a.video_min_tubes == 4
    assert a.physio_cap_ratio == pytest.approx(0.40)
    assert a.virtual_sensing_warmup_epochs == 20
    assert a.spatial_sampling == SPATIAL_ANATOMY_AWARE
    assert a.core_row_bounds == '2,3' and a.core_col_bounds == '2,3'
    assert (a.core_tubes_min, a.core_tubes_max) == (1, 2)
    assert a.periph_min_tubes == 1 and a.physio_draw_min == 1
    assert (a.spatial_patches, a.physio_tokens) == (36, 250)
    assert a.temporal_stride == 4 and a.sig_kernel == 32 and a.fs == 1000.0
    # the cache lineage refuses the legacy raw-path filters
    assert a.min_signal_spread == 0.0 and a.rail_touch_v == 0.0


def test_g9_model_publishes_the_resolved_policy():
    from core.multimae import MultiModalMAE
    m = MultiModalMAE(streams=('tir', 'resp'), enc_depth=1, enc_num_heads=4,
                      dec_depth=1, tubelet=(2, 16, 16), input_size=96,
                      num_frames=50, sig_kernel=32, seq_len=8000, fps=25.0,
                      fs=1000.0, temporal_stride=4, n_budget=N_BUDGET,
                      video_cap_ratio=0.25, physio_cap_ratio=0.40,
                      video_min_tubes=S_TIR_MIN,
                      spatial_sampling=SPATIAL_ANATOMY_AWARE,
                      physio_draw_min=1)
    assert m.apportioner.policy == SPATIAL_POLICY_SINGLE_VISUAL
    assert m.apportioner.video_cap_units == {'tir': S_TIR_MAX}
    assert m.core_coords == ((2, 2), (2, 3), (3, 2), (3, 3))
    assert m.core_indices == CORE
    assert (m.s_tir_min, m.s_tir_max) == (S_TIR_MIN, S_TIR_MAX)


# --------------------------------------------------------------------------- #
# G10 - the evaluation pin
# --------------------------------------------------------------------------- #
def _pin_args(**kw):
    base = dict(kept_tir=0.0, kept_resp=0.0, kept_rgb2d=0.5, kept_bp=1.0,
                resp_probe='hidden', span_seconds=1.0,
                pin_single_carrier='none', pin_core_tubes=-1)
    base.update(kw)
    return argparse.Namespace(**base)


def _pin_model(sampling=SPATIAL_ANATOMY_AWARE):
    from core.multimae import MultiModalMAE
    return MultiModalMAE(streams=('tir', 'resp'), enc_depth=1,
                         enc_num_heads=4, dec_depth=1, tubelet=(2, 16, 16),
                         input_size=96, num_frames=50, sig_kernel=32,
                         seq_len=8000, fps=25.0, fs=1000.0,
                         temporal_stride=4, n_budget=N_BUDGET,
                         video_cap_ratio=0.25, physio_cap_ratio=0.40,
                         video_min_tubes=S_TIR_MIN,
                         spatial_sampling=sampling, physio_draw_min=1)


def test_g10_pin_reaches_exactly_200_over_0_and_100_over_100():
    from runners.run_eval_recon import pin_masking
    plan, caption = pin_masking(_pin_model(), _pin_args())
    assert plan.kept_tokens == {'tir': 200, 'resp': 0}
    assert plan.mode_by_counts == MODE_VIRTUAL_SENSING
    assert plan.core_units == {'tir': 1} and plan.periph_units == {'tir': 7}
    assert 'spatial_sampling=anatomy_aware' in caption
    assert 'S_core=1' in caption and 'S_periph=7' in caption

    plan, _ = pin_masking(_pin_model(),
                          _pin_args(kept_resp=0.4, resp_probe='random'))
    assert plan.kept_tokens == {'tir': 100, 'resp': 100}
    assert plan.kept_units['tir'] == S_TIR_MIN


def test_g10_pin_core_is_refused_on_a_uniform_checkpoint():
    from runners.run_eval_recon import pin_masking
    with pytest.raises(ValueError, match='kept_core'):
        pin_masking(_pin_model(sampling='uniform'), _pin_args(pin_core_tubes=2))
    plan, caption = pin_masking(_pin_model(sampling='uniform'), _pin_args())
    assert plan.core_units is None
    assert 'spatial_sampling=uniform' in caption


# --------------------------------------------------------------------------- #
# G12 - the uniform ablation draws no core and does not move the RNG
# --------------------------------------------------------------------------- #
def test_g12_uniform_ablation_is_rng_neutral_and_coreless():
    a = make(spatial_sampling='uniform', physio_draw_min=1, core_tubes_min=1,
             core_tubes_max=1, periph_min_tubes=3, core_row_bounds=(0, 0),
             core_col_bounds=(0, 1))
    b = make(spatial_sampling='uniform', physio_draw_min=1, core_tubes_min=3,
             core_tubes_max=3, periph_min_tubes=1)
    seq_a = draw(a, steps=500, seed=11)
    seq_b = draw(b, steps=500, seed=11)
    assert [signature(p) for p in seq_a] == [signature(p) for p in seq_b]
    assert all(plan.core_units is None for plan in seq_a)


# --------------------------------------------------------------------------- #
# G13 - the replay report's expectations follow the draw floor
# --------------------------------------------------------------------------- #
def test_g13_replay_expectation_has_no_zero_draw_under_the_adr_floor():
    from scripts.project.mask_policy_report import expected_zero_draw_share
    app = make(physio_draw_min=1)
    assert expected_zero_draw_share(app) == 0.0
    legacy = make(physio_draw_min=0)
    assert expected_zero_draw_share(legacy) == pytest.approx(1.0 / (C_P + 1))


def test_g13_report_expectation_matches_the_realized_leak():
    """Observed 0.0 under the ADR floor, and no plan labelled Mode 1 on a Mode-2 draw."""
    app = make(p_virtual_sensing=0.0)
    plans = draw(app, steps=5_000, seed=5)
    leaks = [p for p in plans if p.mode == MODE_INPAINTING
             and p.mode_by_counts == MODE_VIRTUAL_SENSING]
    assert leaks == []


def test_g13_report_main_writes_the_stratified_payload(tmp_path):
    """``mask_policy_report.main`` end-to-end: replay + JSON, no GPU forward."""
    import json
    import importlib
    from scripts.project import mask_policy_report as report

    old = sys.argv
    sys.argv = ['prog', '-c', os.path.join(CODE_DIR, CONFIG)]
    try:
        cargs = importlib.import_module('runners.run_pretrain').get_args()
    finally:
        sys.argv = old
    ckpt = tmp_path / 'ckpt.pth'
    torch.save({'args': cargs}, ckpt)
    report.main(['--checkpoint', str(ckpt), '--epochs', '4',
                 '--steps_per_epoch', '50', '--out-dir', str(tmp_path),
                 '--tag', 'adr007'])
    payload = json.load(open(tmp_path / 'adr007_policy_replay.json'))
    strat = payload['stratification']
    assert strat['spatial_sampling'] == 'anatomy_aware'
    assert strat['core_indices'] == list(CORE)
    assert strat['S_core']['min'] >= 1 and strat['S_core']['max'] <= 2
    assert strat['core_hit_share'] == 1.0
    assert payload['criterion_1']['mode2_k_tilde_zero_share']['expected'] == 0.0
    assert payload['criterion_1']['exact_sum']['ok'] is True
    # a representative replay populates BOTH classes (an empty one would make
    # the report's mode-conditional means vacuous)
    assert payload['mode_split']['resolved']['mode1'] > 0
    assert payload['mode_split']['resolved']['mode2'] > 0


# --------------------------------------------------------------------------- #
# G9/Phase 4.2 - the masking manifest (integration boundary, not just flags)
# --------------------------------------------------------------------------- #
class _Plan:
    fold = 0
    digest = 'test-digest'


def _manifest_args(tmp_path):
    return argparse.Namespace(output_dir=str(tmp_path), epochs=150,
                              clip_duration=8.0)


def test_g9_manifest_records_the_adr007_policy(tmp_path):
    import json
    from runners.run_pretrain import write_masking_manifest
    model = _pin_model()
    rec = write_masking_manifest(_manifest_args(tmp_path), model, plan=_Plan(),
                                 mask_base=1234, stage1={'random': True})
    pol = rec['policy']
    assert pol['adr'] == 'ADR-007'
    assert pol['spatial_sampling'] == 'anatomy_aware'
    assert pol['physio_draw_min'] == 1
    assert pol['core_coords'] == [[2, 2], [2, 3], [3, 2], [3, 3]]
    assert pol['core_indices'] == list(CORE)
    assert pol['periph_tubes'] == SPATIAL - len(CORE)
    assert (pol['s_tir_min'], pol['s_tir_max']) == (S_TIR_MIN, S_TIR_MAX)
    assert rec['geometry']['signal_tokens_per_visual_token'] == 10
    on_disk = json.load(open(tmp_path / 'masking.json'))
    assert on_disk['policy']['adr'] == 'ADR-007'


def test_g12_uniform_manifest_nulls_the_core_fields(tmp_path):
    from runners.run_pretrain import write_masking_manifest
    model = _pin_model(sampling='uniform')
    rec = write_masking_manifest(_manifest_args(tmp_path), model, plan=_Plan(),
                                 mask_base=1234, stage1={'random': True})
    pol = rec['policy']
    assert pol['adr'] == 'off-adr-007'
    assert pol['core_coords'] is None and pol['core_indices'] == []
    assert pol['core_row_bounds'] is None and pol['core_col_bounds'] is None
    assert pol['periph_tubes'] is None


def test_g8_two_visual_manifest_keeps_its_key_set(tmp_path):
    """The resume guard compares key SETS: ADR-007 must add none for 2 visuals."""
    from core.multimae import MultiModalMAE
    from runners.run_pretrain import write_masking_manifest
    model = MultiModalMAE(streams=('tir', 'rgb2d', 'resp'), enc_depth=1,
                          enc_num_heads=4, dec_depth=1, tubelet=(2, 16, 16),
                          input_size=96, num_frames=50, sig_kernel=32,
                          seq_len=8000, fps=25.0, fs=1000.0,
                          temporal_stride=4, n_budget=350,
                          video_cap_ratio=0.25, physio_cap_ratio=0.50,
                          video_min_tubes=2)
    rec = write_masking_manifest(_manifest_args(tmp_path), model, plan=_Plan(),
                                 mask_base=1234, stage1={'random': True})
    for key in ('adr', 'spatial_sampling', 'physio_draw_min', 'core_coords',
                'core_indices', 's_tir_min', 's_tir_max', 'periph_min_tubes'):
        assert key not in rec['policy'], key
    assert rec['policy']['video_cap_units'] == {'tir': 9, 'rgb2d': 9}


# --------------------------------------------------------------------------- #
# Phase 4.3 - validation payload: stratification is a SIBLING of components
# --------------------------------------------------------------------------- #
def _eval_loader(batches=2, B=2):
    return [{'tir': torch.randn(B, 3, 50, 96, 96),
             'resp': torch.randn(B, 1, 8000)} for _ in range(batches)]


def test_phase43_validation_reports_stratification_not_components():
    from engines.pretrain import evaluate_pretrain
    model = _tiny_model(spatial_sampling=SPATIAL_ANATOMY_AWARE,
                        physio_draw_min=1)
    val = evaluate_pretrain(model, _eval_loader(), torch.device('cpu'),
                            mask_base=7, epoch=0)
    # components stays the MSE component set (test_stage2_protocol pins this)
    assert set(val['components']) == {'mse_tir', 'mse_resp'}
    # ... while the stratified tube counts are their own dict, visual only
    assert set(val['stratification']) == {'core_tubes_tir', 'periph_tubes_tir'}
    assert 1.0 <= val['stratification']['core_tubes_tir'] <= 2.0
    assert val['stratification']['periph_tubes_tir'] >= 3.0
    assert val['kept']['kept_tir'] == N_BUDGET


def test_phase43_uniform_run_reports_no_stratification():
    from engines.pretrain import evaluate_pretrain
    model = _tiny_model(spatial_sampling='uniform', physio_draw_min=1)
    val = evaluate_pretrain(model, _eval_loader(batches=1),
                            torch.device('cpu'), mask_base=7, epoch=0)
    assert val['stratification'] == {}
    assert set(val['components']) == {'mse_tir', 'mse_resp'}


# --------------------------------------------------------------------------- #
# Review regressions: the draw floor on a 2-signal tuple, and the parsers
# --------------------------------------------------------------------------- #
def _two_visual_two_signal(**kw):
    streams = [StreamSpec('tir', 'visual', SPATIAL, T, (6, 6)),
               StreamSpec('rgb2d', 'visual', SPATIAL, T, (6, 6)),
               StreamSpec('bp', 'signal', 60, 1),
               StreamSpec('resp', 'signal', 60, 1)]
    d = dict(n_budget=350, video_cap_ratio=0.25, physio_cap_ratio=0.50,
             video_min_tubes=2, p_virtual_sensing=0.0, physio_draw_min=1)
    d.update(kw)
    return ConstantBudgetApportioner(streams, **d)


def test_draw_floor_on_two_signals_never_breaks_the_alignment():
    """Review #4: nothing may be clamped OUT of the modulo alignment.

    ``C_p = 30 >= T = 25`` for both signals, which is what the shared guard
    requires once ``physio_draw_min > 0``. The old clamp could push a stream to
    25 after the downward branch and leave the sum un-aligned, raising inside
    the sampler; the draw must instead stay legal for every step.
    """
    app = _two_visual_two_signal()
    for plan in draw(app, steps=3_000, seed=4):
        assert plan.total_tokens == 350
        assert plan.mode_by_counts == MODE_INPAINTING
        kept = plan.kept_tokens
        assert 0 <= kept['bp'] <= 30 and 0 <= kept['resp'] <= 30
        assert kept['bp'] + kept['resp'] > 0          # no Mode-2 -> Mode-1 leak
        assert plan.kept_units['tir'] + plan.kept_units['rgb2d'] == \
            (350 - kept['bp'] - kept['resp']) // T


def test_draw_floor_rejects_a_cap_below_T_on_two_visuals():
    """Review #4.2: ``C_p < T`` with the floor is a startup error, not a crash."""
    with pytest.raises(ValueError, match='needs every physiological cap'):
        _two_visual_two_signal(physio_cap_ratio=0.05)     # C_p = 12 < T = 25


def test_parse_int_csv_accepts_a_sequence_and_names_the_flag():
    """Review #1: YAML may write ``[2, 3]``; the error must name the flag."""
    from core.multimae import _parse_int_csv
    assert _parse_int_csv('2,3') == (2, 3)
    assert _parse_int_csv([2, 3]) == (2, 3)
    assert _parse_int_csv((4, 5)) == (4, 5)
    with pytest.raises(ValueError, match='core_row_bounds'):
        _parse_int_csv('2,x', flag='core_row_bounds')


def test_budget_plan_requires_symmetric_stratified_dicts():
    """Review #3: a key present in only one of the two dicts must raise."""
    from core.budget_masking import BudgetPlan
    plan = BudgetPlan(mode=1, kept_tokens={'tir': 200, 'resp': 0},
                      kept_units={'tir': 8, 'resp': 0}, mode_by_counts=1,
                      regime=REGIME_SINGLE_VISUAL,
                      core_units={'tir': 1}, periph_units={'tir': 7})
    assert plan.core_units == {'tir': 1}
    with pytest.raises(ValueError, match='same streams'):
        BudgetPlan(mode=1, kept_tokens={'tir': 200, 'resp': 0},
                   kept_units={'tir': 8, 'resp': 0}, mode_by_counts=1,
                   regime=REGIME_SINGLE_VISUAL,
                   core_units={'tir': 1}, periph_units={})
    with pytest.raises(ValueError, match='undeclared stream'):
        BudgetPlan(mode=1, kept_tokens={'tir': 200}, kept_units={'tir': 8},
                   mode_by_counts=1, regime=REGIME_SINGLE_VISUAL,
                   core_units={'rgb2d': 1}, periph_units={'rgb2d': 7})
