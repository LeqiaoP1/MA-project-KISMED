"""ADR-017 conformance: gap-controlled masked-span placement (``masked_span``).

Design: ``code/notes/design/ADR-017-masked-span-physio-placement.md`` (the
contract — the invariant ids I1-I6 and the requirements R1-R7 are cited per
test). Plan: ``code/notes/plan/features/ADR-017-masked-span-physio-placement-plan.md``
(Phase 7 lists exactly these gates).

Geometry: the arm's — ``N_s = 250`` respiration tokens of 32 ms (ADR-007),
``n_budget = 200``, ``C_p = 100``, so the apportioner can draw
``K_resp in {25, 50, 75, 100}``. That set is DERIVED here from
``ConstantBudgetApportioner.mode2_counts`` and never transcribed (plan D4):
if ADR-007's counts move, these tests follow them.
"""
import argparse
import json
import types

import pytest
import torch

from core.budget_masking import (SIGNAL, VISUAL, ConstantBudgetApportioner,
                                 StreamSpec)
from core.mask_placement import (MASKED_SPAN, SCATTERED, SPAN,
                                 admissible_gap_counts, place_signal,
                                 validate_physio_placement)
from core.multimae import _tiny_model

N_S = 250                     # respiration tokens at ADR-007 geometry
G_MIN, G_MAX = 63, 125        # the arm's gap bounds (2.016 s / 4.0 s)
CPU = torch.device('cpu')


def _arm_apportioner(**over):
    """The ADR-008/ADR-017 two-stream policy, as an apportioner."""
    kw = dict(n_budget=200, video_cap_ratio=0.25, physio_cap_ratio=0.40,
              video_min_tubes=4, p_virtual_sensing=0.5,
              virtual_sensing_warmup_p=0.10,
              virtual_sensing_warmup_epochs=20,
              spatial_sampling='anatomy_aware', core_row_bounds=(2, 3),
              core_col_bounds=(2, 3), core_tubes_min=1, core_tubes_max=2,
              periph_min_tubes=1, physio_draw_min=1)
    kw.update(over)
    return ConstantBudgetApportioner(
        [StreamSpec('tir', VISUAL, 36, tokens_per_unit=25,
                    spatial_grid=(6, 6)),
         StreamSpec('resp', SIGNAL, N_S, tokens_per_unit=1)], **kw)


ARM = _arm_apportioner()
ARM_COUNTS = ARM.mode2_counts('resp')


def _masked_runs(row, n_units=N_S):
    """Lengths of the maximal MASKED runs of one row, in tokens."""
    keep = torch.zeros(int(n_units), dtype=torch.bool)
    keep[row] = True
    runs, run = [], 0
    for v in (~keep).tolist() + [False]:
        if v:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    return runs


def _state(gen):
    return gen.get_state()


# --------------------------------------------------------------------------- #
# D4 — the count set comes from the apportioner, not from a literal
# --------------------------------------------------------------------------- #
def test_mode2_counts_is_derived_from_the_arm_tuple():
    assert ARM_COUNTS == (25, 50, 75, 100)
    # A different cap moves the set, which a transcribed literal would miss.
    # ``video_min_tubes`` is an ASSERTION of (n_budget - C_p) / T, so it has to
    # follow the cap: (200 - 50) / 25 = 6.
    assert _arm_apportioner(physio_cap_ratio=0.20, video_min_tubes=6) \
        .mode2_counts('resp') == (25, 50)


def test_mode2_counts_matches_a_sampled_histogram():
    """The O(C_p) enumeration must equal what ``sample_step`` really draws."""
    gen = torch.Generator().manual_seed(11)
    seen = set()
    for _ in range(20000):
        plan = ARM.sample_step(generator=gen, epoch=30)
        kept = int(plan.kept_tokens['resp'])
        if kept:
            seen.add(kept)
    assert tuple(sorted(seen)) == ARM_COUNTS


def test_mode2_counts_needs_one_physiological_stream():
    """Two signal streams are legal for the POLICY (the 4-stream arm), so the
    refusal must come from ``mode2_counts`` itself — hence two visual streams."""
    two_signals = ConstantBudgetApportioner(
        [StreamSpec('tir', VISUAL, 36, tokens_per_unit=25,
                    spatial_grid=(6, 6)),
         StreamSpec('rgb2d', VISUAL, 36, tokens_per_unit=25,
                    spatial_grid=(6, 6)),
         StreamSpec('bp', SIGNAL, N_S, tokens_per_unit=1),
         StreamSpec('resp', SIGNAL, N_S, tokens_per_unit=1)],
        n_budget=350, video_cap_ratio=0.25, physio_cap_ratio=0.5,
        video_min_tubes=2, p_virtual_sensing=0.5)
    with pytest.raises(ValueError, match='ONE physiological stream'):
        two_signals.mode2_counts('resp')


def test_admissible_gap_counts_matches_the_adr_table():
    # ADR-017 §2.1: N = {2} for M in {150, 175}, {2, 3} for M in {200, 225}
    assert admissible_gap_counts(150, 100, G_MIN, G_MAX) == (2,)
    assert admissible_gap_counts(175, 75, G_MIN, G_MAX) == (2,)
    assert admissible_gap_counts(200, 50, G_MIN, G_MAX) == (2, 3)
    assert admissible_gap_counts(225, 25, G_MIN, G_MAX) == (2, 3)
    # no gap at all, and bounds that cannot tile
    assert admissible_gap_counts(0, N_S, G_MIN, G_MAX) == ()
    assert admissible_gap_counts(150, 100, 100, 100) == ()


# --------------------------------------------------------------------------- #
# I1-I3 — exact count, bounded runs, separation (10^4 draws per K_resp)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('k_resp', ARM_COUNTS)
def test_i1_i2_i3_exact_count_bounded_runs_and_separation(k_resp):
    """I3 is checked through the RUN COUNT: if two gaps ever touched, the
    number of maximal masked runs would fall below the drawn ``n``."""
    gen = torch.Generator().manual_seed(20261127)
    batch, repeats = 500, 20                      # 10^4 draws per count
    masked = N_S - k_resp
    for _ in range(repeats):
        idx = place_signal(k_resp, N_S, generator=gen, mode=MASKED_SPAN,
                           gap_min_tokens=G_MIN, gap_max_tokens=G_MAX,
                           batch_size=batch)
        assert idx.shape == (batch, k_resp)
        assert int(idx.min()) >= 0 and int(idx.max()) < N_S
        if k_resp > 1:                            # I1: strictly ascending
            assert bool(torch.all(idx[:, 1:] > idx[:, :-1]))
        for row in idx:
            runs = _masked_runs(row)
            assert all(G_MIN <= g <= G_MAX for g in runs), runs     # I2
            assert len(runs) in admissible_gap_counts(                # I2/I3
                masked, k_resp, G_MIN, G_MAX)
            assert sum(runs) == masked                                # I1


def test_every_sample_carries_exactly_the_step_count():
    idx = place_signal(50, N_S, generator=torch.Generator().manual_seed(3),
                       mode=MASKED_SPAN, gap_min_tokens=G_MIN,
                       gap_max_tokens=G_MAX, batch_size=64)
    assert idx.shape == (64, 50)
    assert len({row.numel() for row in idx}) == 1


# --------------------------------------------------------------------------- #
# I4 — the two degenerate counts draw NOTHING
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('kept', [0, N_S])
def test_i4_degenerate_counts_consume_no_random_number(kept):
    gen = torch.Generator().manual_seed(5)
    before = _state(gen).clone()
    idx = place_signal(kept, N_S, generator=gen, mode=MASKED_SPAN,
                       gap_min_tokens=G_MIN, gap_max_tokens=G_MAX,
                       batch_size=4)
    assert torch.equal(_state(gen), before)
    if kept == 0:
        assert idx.shape == (4, 0)
    else:
        assert idx.shape == (4, N_S)
        assert torch.equal(idx[0], torch.arange(N_S))


# --------------------------------------------------------------------------- #
# R3 — deterministic replay, per-sample diversity
# --------------------------------------------------------------------------- #
def test_replay_is_identical_and_samples_differ():
    def draw(seed):
        return place_signal(50, N_S, generator=torch.Generator().manual_seed(
            seed), mode=MASKED_SPAN, gap_min_tokens=G_MIN,
            gap_max_tokens=G_MAX, batch_size=8)

    first, again, other = draw(7), draw(7), draw(8)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert len({tuple(r.tolist()) for r in first}) == 8


def test_the_gap_count_varies_across_a_batch():
    """D3: ``n`` is drawn per SAMPLE. ``K = 25`` gives ``M = 225``, whose
    admissible set is {2, 3}, so a batch of 64 must show both."""
    idx = place_signal(25, N_S, generator=torch.Generator().manual_seed(9),
                       mode=MASKED_SPAN, gap_min_tokens=G_MIN,
                       gap_max_tokens=G_MAX, batch_size=64)
    widths = {len(_masked_runs(row)) for row in idx}
    assert widths == {2, 3}
    # K = 100 -> M = 150 admits one gap count only, so nothing may vary
    idx = place_signal(100, N_S, generator=torch.Generator().manual_seed(9),
                       mode=MASKED_SPAN, gap_min_tokens=G_MIN,
                       gap_max_tokens=G_MAX, batch_size=64)
    assert {len(_masked_runs(row)) for row in idx} == {2}


# --------------------------------------------------------------------------- #
# I6 / R6 — refusals, at construction and at probe time
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('mode,kw,needle', [
    (MASKED_SPAN, dict(gap_min_tokens=125, gap_max_tokens=63),
     'gap_min_tokens'),
    (MASKED_SPAN, dict(gap_min_tokens=100, gap_max_tokens=100), 'tile'),
    (MASKED_SPAN, dict(gap_min_tokens=G_MIN, gap_max_tokens=N_S + 1),
     'gap_min_tokens'),
    (MASKED_SPAN, dict(gap_min_tokens=G_MIN, gap_max_tokens=G_MAX,
                       span_tokens=10), 'exclusive'),
    (SCATTERED, dict(gap_min_tokens=G_MIN, gap_max_tokens=G_MAX),
     'only meaningful'),
    (SPAN, dict(span_tokens=10, gap_min_tokens=G_MIN), 'only meaningful'),
])
def test_place_signal_refuses_illegal_knob_pairings(mode, kw, needle):
    with pytest.raises(ValueError, match=needle):
        place_signal(100, N_S, mode=mode, **kw)


@pytest.mark.parametrize('kw,needle', [
    (dict(gap_min_tokens=125, gap_max_tokens=63), 'gap_min'),
    (dict(gap_min_tokens=100, gap_max_tokens=100), 'cannot tile'),
    (dict(gap_min_tokens=G_MIN, gap_max_tokens=N_S + 1), 'gap_min'),
    (dict(gap_min_tokens=G_MIN, gap_max_tokens=G_MAX, span_tokens=10),
     'exclusive'),
    (dict(gap_min_tokens=G_MIN, gap_max_tokens=G_MAX, n_streams=2),
     'one token axis'),
])
def test_validate_physio_placement_refuses_at_construction(kw, needle):
    """The keeping count is K_resp = 75 (M = 175), which the arm's bounds
    tile with exactly one gap count."""
    with pytest.raises(ValueError, match=needle):
        validate_physio_placement(mode=MASKED_SPAN, n_tokens=N_S,
                                  kept_counts=(75,), **kw)


def test_probe_pinned_counts_outside_kappa_are_refused():
    """The Stage-2 probe pins ``--kept_resp``, which is not restricted to the
    apportioner's set: an infeasible pin must be refused, naming the feasible
    ones (ADR-017 R6)."""
    validate_physio_placement(mode=MASKED_SPAN, n_tokens=N_S,
                              gap_min_tokens=G_MIN, gap_max_tokens=G_MAX,
                              kept_counts=(100,))
    with pytest.raises(ValueError, match=r'CAN tile: \[100\]'):
        validate_physio_placement(mode=MASKED_SPAN, n_tokens=N_S,
                                  gap_min_tokens=G_MIN, gap_max_tokens=G_MAX,
                                  kept_counts=(100, 200))


def test_a_neutral_placement_still_validates():
    validate_physio_placement(mode=SCATTERED, n_tokens=N_S)
    validate_physio_placement(mode=SPAN, n_tokens=N_S, span_tokens=25)


def test_scattered_rejects_a_stale_span_length():
    """ADR-017 §2.2 exclusivity is SYMMETRIC: a span length must not survive a
    switch to ``scattered`` (it would look authoritative and be ignored)."""
    with pytest.raises(ValueError, match='exclusive'):
        place_signal(50, N_S, mode=SCATTERED, span_tokens=25)
    with pytest.raises(ValueError, match='exclusive'):
        validate_physio_placement(mode=SCATTERED, n_tokens=N_S,
                                  span_tokens=25)
    with pytest.raises(ValueError, match='exclusive'):
        _tiny_model(physio_placement=SCATTERED, physio_span_tokens=7)


# --------------------------------------------------------------------------- #
# R5 — scattered and legacy span are untouched by the new branch
# --------------------------------------------------------------------------- #
def test_scattered_is_the_documented_pre_adr_draw():
    k, b, seed = 50, 6, 13
    gen = torch.Generator().manual_seed(seed)
    before = _state(gen).clone()
    idx = place_signal(k, N_S, generator=gen, mode=SCATTERED, batch_size=b)
    # the pre-ADR body: one rand(b, n) draw, argsort, first k, sorted
    ref_gen = torch.Generator().manual_seed(seed)
    ref = torch.rand(b, N_S, generator=ref_gen).argsort(dim=1)[:, :k]
    assert torch.equal(idx, ref.sort(dim=1).values)
    consumed = _state(gen)
    expected = ref_gen.get_state()
    assert torch.equal(consumed, expected), 'RNG stream changed'
    assert not torch.equal(before, consumed)


@pytest.mark.parametrize('mode,extra', [(SCATTERED, {}),
                                        (SPAN, {'span_tokens': 25})])
def test_new_keywords_default_to_the_old_behaviour(mode, extra):
    """The gap bounds default to ``None``/0, so a pre-ADR call site draws the
    SAME indices and the SAME generator state."""
    def draw(explicit):
        gen = torch.Generator().manual_seed(17)
        kw = dict(extra)
        if explicit:
            kw.update(gap_min_tokens=None, gap_max_tokens=None)
        idx = place_signal(50, N_S, generator=gen, mode=mode, batch_size=8,
                           **kw)
        return idx, _state(gen).clone()

    old, old_state = draw(False)
    new, new_state = draw(True)
    assert torch.equal(old, new)
    assert torch.equal(old_state, new_state)


# --------------------------------------------------------------------------- #
# I5 — the forward pass sees no masked token as visible
# --------------------------------------------------------------------------- #
def _spanned_tiny_model(**over):
    kw = dict(physio_placement=MASKED_SPAN, physio_gap_min_tokens=3,
              physio_gap_max_tokens=5)
    kw.update(over)
    return _tiny_model(**kw)


def test_i5_every_gathered_respiration_token_is_visible():
    model = _spanned_tiny_model()
    counts = model.apportioner.mode2_counts('resp')
    assert counts == (10,)          # tiny geometry: K_resp = 10, M = 10
    plan = model.plan_from_counts({'tir': 30, 'rgb2d': 40, 'resp': 10})
    masks, _plan, idx = model.make_masks(
        4, CPU, plan=plan,
        placement_generator=torch.Generator().manual_seed(0))
    kept = masks['resp'].gather(1, idx['resp'])
    assert int(kept.sum()) == 0
    for row in (1 - masks['resp']).bool():
        runs = _masked_runs(torch.where(row)[0], n_units=model.physio_tokens)
        assert all(3 <= g <= 5 for g in runs), runs


def test_i5_mode1_still_masks_the_whole_stream():
    model = _spanned_tiny_model()
    # both visual streams sit at their cap (4 tubes = 40 tokens each)
    plan = model.plan_from_counts({'tir': 40, 'rgb2d': 40, 'resp': 0})
    masks, _plan, idx = model.make_masks(
        2, CPU, plan=plan,
        placement_generator=torch.Generator().manual_seed(1))
    assert int(masks['resp'].sum()) == 2 * model.physio_tokens
    assert idx['resp'].shape == (2, 0)


def test_i5_forward_gathers_only_visible_tokens():
    """The real forward pass: the masks it builds are the ones ``make_masks``
    produced for the same seed, and every gathered respiration token is
    visible (ADR-017 I5)."""
    model = _spanned_tiny_model()
    B = 3
    plan = model.plan_from_counts({'tir': 30, 'rgb2d': 40, 'resp': 10})
    ref, _plan, idx = model.make_masks(
        B, CPU, plan=plan,
        placement_generator=torch.Generator().manual_seed(4))
    batch = {
        'tir': torch.randn(B, 3, model.num_frames, model.input_size,
                           model.input_size),
        'rgb2d': torch.randn(B, 3, model.num_frames, model.input_size,
                             model.input_size),
        'resp': torch.randn(B, 1, model.seq_len),
    }
    out = model(batch, plan=plan,
                placement_generator=torch.Generator().manual_seed(4))
    assert torch.equal(out['masks']['resp'], ref['resp'])
    assert int(out['masks']['resp'].gather(1, idx['resp']).sum()) == 0
    assert torch.isfinite(out['loss'])
    assert out['preds']['resp'].shape == (B, model.physio_tokens,
                                          model.sig_kernel)
    # the encoder still sees the constant budget
    assert int((out['masks']['tir'] == 0).sum(dim=1).unique().item()) == 30


def test_model_construction_refuses_masked_span_with_two_signal_streams():
    with pytest.raises(ValueError, match='one token axis'):
        _tiny_model(streams=('tir', 'rgb2d', 'bp', 'resp'),
                    physio_placement=MASKED_SPAN,
                    physio_gap_min_tokens=3, physio_gap_max_tokens=5)


def test_model_construction_refuses_infeasible_bounds():
    with pytest.raises(ValueError, match='cannot tile'):
        _spanned_tiny_model(physio_gap_min_tokens=20,
                            physio_gap_max_tokens=20)


def test_model_construction_refuses_the_span_token_pairing():
    with pytest.raises(ValueError, match='exclusive'):
        _spanned_tiny_model(physio_span_tokens=4)


# --------------------------------------------------------------------------- #
# runner: the manifest records the bounds, and the resume guard is strict
# --------------------------------------------------------------------------- #
def _manifest_inputs(tmp_path, model):
    args = argparse.Namespace(output_dir=str(tmp_path), epochs=1,
                              clip_duration=8.0)
    plan = types.SimpleNamespace(fold=0, digest='deadbeef')
    return args, model, plan


def test_manifest_records_the_bounds_only_when_non_neutral(tmp_path):
    from runners.run_pretrain import write_masking_manifest

    args, model, plan = _manifest_inputs(tmp_path, _spanned_tiny_model())
    rec = write_masking_manifest(args, model, plan=plan, mask_base=0,
                                 stage1={'path': '', 'loaded': 0,
                                         'random': True})
    pol = rec['policy']
    assert pol['physio_placement'] == MASKED_SPAN
    assert pol['physio_gap_min_tokens'] == 3
    assert pol['physio_gap_max_tokens'] == 5
    # resolved seconds: tokens * sig_kernel / fs
    assert pol['physio_gap_seconds'] == [round(3 * 8 / 100.0, 6),
                                         round(5 * 8 / 100.0, 6)]
    # JSON round-trip: the reloaded record must compare equal to the fresh one
    # (``_stable`` compares whole dicts, so a tuple would break every resume)
    with open(tmp_path / 'masking.json') as fh:
        reloaded = json.load(fh)
    assert reloaded['policy'] == pol
    assert isinstance(reloaded['policy']['physio_gap_seconds'][0], float)


def test_manifest_omits_neutral_bounds(tmp_path):
    from runners.run_pretrain import write_masking_manifest

    args, model, plan = _manifest_inputs(
        tmp_path, _tiny_model(physio_placement=SCATTERED,
                              physio_gap_min_tokens=0,
                              physio_gap_max_tokens=0))
    rec = write_masking_manifest(args, model, plan=plan, mask_base=0,
                                 stage1={'path': '', 'loaded': 0,
                                         'random': True})
    assert 'physio_gap_min_tokens' not in rec['policy']
    assert 'physio_gap_seconds' not in rec['policy']


def test_manifest_accepts_a_same_config_resume(tmp_path):
    """A second write with the SAME policy must pass ``_stable()`` and leave
    the run's first record untouched (the trap the guarded keys avoid)."""
    from runners.run_pretrain import write_masking_manifest

    args, model, plan = _manifest_inputs(tmp_path, _spanned_tiny_model())
    first = write_masking_manifest(args, model, plan=plan, mask_base=0,
                                   stage1={'path': '', 'loaded': 0,
                                           'random': True})
    second = write_masking_manifest(args, model, plan=plan, mask_base=0,
                                    stage1={'path': '', 'loaded': 0,
                                            'random': True})
    assert second == first


def test_resume_with_different_bounds_is_refused(tmp_path):
    from runners.run_pretrain import write_masking_manifest

    args, model, plan = _manifest_inputs(tmp_path, _spanned_tiny_model())
    write_masking_manifest(args, model, plan=plan, mask_base=0,
                           stage1={'path': '', 'loaded': 0, 'random': True})
    other = _spanned_tiny_model(physio_gap_min_tokens=4,
                                physio_gap_max_tokens=5)
    with pytest.raises(SystemExit, match='different masking policy'):
        write_masking_manifest(args, other, plan=plan, mask_base=0,
                              stage1={'path': '', 'loaded': 0,
                                      'random': True})


# --------------------------------------------------------------------------- #
# evaluator: pin_masking clears stale placement attributes (ADR-017 R7)
# --------------------------------------------------------------------------- #
def _pin_args(**over):
    kw = dict(kept_tir=0.5, kept_rgb2d=0.5, kept_resp=0.5, kept_bp=1.0,
              resp_probe='random', span_seconds=4.0,
              pin_single_carrier='none', exempt_budget=False,
              pin_core_tubes=-1, gap_min_tokens=-1, gap_max_tokens=-1)
    kw.update(over)
    return argparse.Namespace(**kw)


def test_pin_masking_clears_stale_placement_attributes():
    """Probing a span-trained checkpoint under another mode must ZERO the span
    length, or the symmetric exclusivity check would refuse the probe."""
    from runners.run_eval_recon import pin_masking

    model = _tiny_model(physio_placement=SPAN, physio_span_tokens=3)
    assert model.physio_span_tokens == 3
    _plan, caption = pin_masking(model, _pin_args(resp_probe='random'))
    assert model.physio_placement == SCATTERED
    assert model.physio_span_tokens == 0
    assert model.physio_gap_min_tokens == 0
    assert model.physio_gap_max_tokens == 0
    assert 'scattered' in caption


def test_pin_masking_reproduces_the_checkpoint_gap_rule():
    from runners.run_eval_recon import pin_masking

    model = _spanned_tiny_model()                    # recorded bounds 3..5
    _plan, caption = pin_masking(model, _pin_args(resp_probe='masked_span'))
    assert model.physio_placement == MASKED_SPAN
    assert model.physio_span_tokens == 0
    assert (model.physio_gap_min_tokens, model.physio_gap_max_tokens) == (3, 5)
    assert 'gaps 3..5 tokens' in caption


def test_pin_masking_refuses_an_infeasible_pinned_count():
    """R6 end to end: bounds that tile the arm's own K (10) but not the pinned
    K (5) must refuse the PROBE, and name the counts that would work."""
    from runners.run_eval_recon import pin_masking

    model = _tiny_model(physio_placement=MASKED_SPAN,
                        physio_gap_min_tokens=10,
                        physio_gap_max_tokens=10)
    # M = 20 - 5 = 15 cannot be tiled by gaps of exactly 10
    with pytest.raises(ValueError, match='cannot tile'):
        pin_masking(model, _pin_args(resp_probe='masked_span',
                                     exempt_budget=True, kept_resp=0.25))


# --------------------------------------------------------------------------- #
# floor probe: the policy is rebuilt from the checkpoint ARGS, exactly
# --------------------------------------------------------------------------- #
def _ckpt_args(**over):
    """The arm's recorded args, with the CSV strings a checkpoint really has."""
    kw = dict(streams='tir,resp', tubelet='2,16,16', input_size=96,
              clip_duration=8.0, fps=25.0, fs=1000.0, temporal_stride=4,
              sig_kernel=32, seq_len=0, n_budget=200, video_cap_ratio=0.25,
              physio_cap_ratio=0.40, video_min_tubes=4,
              p_virtual_sensing=0.5, virtual_sensing_warmup_p=0.10,
              virtual_sensing_warmup_epochs=20,
              spatial_sampling='anatomy_aware', core_row_bounds='2,3',
              core_col_bounds='2,3', core_tubes_min=1, core_tubes_max=2,
              periph_min_tubes=1, physio_draw_min=1)
    kw.update(over)
    return argparse.Namespace(**kw)


def test_floor_probe_rebuilds_the_arm_policy_from_checkpoint_args():
    """Regression for the geometry derivation: the temporal decimation must be
    applied (50 frames, not 200) and the CSV ``A_core`` bounds must be parsed,
    or ``grid_t`` — and therefore the Mode-2 count set — is wrong."""
    from runners.run_probe_masking_baselines import _apportioner_from_args

    app, n_signal = _apportioner_from_args(_ckpt_args())
    assert n_signal == N_S
    assert app.tokens_per_unit == 25             # 50 frames / tubelet_t 2
    assert app.mode2_counts('resp') == ARM_COUNTS
