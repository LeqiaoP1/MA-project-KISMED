"""Contract tests for the AMMM masking sampler (``core/ammm_masking.py``).

Design: ``code/notes/design/Asymmetric_MultiModal_Masking.md`` §3.
Plan + gate ids: ``code/notes/plan/Asymmetric_MultiModal_Masking_plan.md``
(Phase 1, G1-G20).

The module is pure and CPU-only: no dataset, no model, no GPU. Every test is
parametrised over MODALITY COMBINATIONS, because "same code path for any subset"
is the acceptance criterion (gate G10) — the 4-stream case is only the superset.
"""
import math

import pytest
import torch

import core.multimae as mm
from core import ammm_masking as ammm
from core.ammm_masking import MaskPlan, StreamSpec, plan_step

#: masking-unit counts of the design's geometry: rgb 8x8 spatial patches,
#: tir 6x6, and 25 temporal tokens for a 1-D stream (``seq_len//sig_kernel``).
UNITS = {'rgb': 64, 'tir': 36, 'bp': 25, 'resp': 25}
#: per-stream caps of the design: video kept <= 25 %, physio uncapped.
CAPS = {'rgb': 0.25, 'tir': 0.25, 'bp': 1.0, 'resp': 1.0}

#: (name, kind) pairs. The last entry uses names that do not exist in the
#: taxonomy on purpose: the module must be name-agnostic (G10).
COMBOS = {
    'tir_resp': (('tir', 'visual'), ('resp', 'signal')),
    'rgb_bp': (('rgb', 'visual'), ('bp', 'signal')),
    'tir_resp_bp': (('tir', 'visual'), ('resp', 'signal'), ('bp', 'signal')),
    'rgb_tir_bp_resp': (('rgb', 'visual'), ('tir', 'visual'),
                        ('bp', 'signal'), ('resp', 'signal')),
    'unknown_names': (('camA', 'visual'), ('ecgB', 'signal')),
}


def _specs(combo) -> list:
    return [StreamSpec(n, k, UNITS.get(n, 25)) for n, k in COMBOS[combo]] \
        if isinstance(combo, str) else [StreamSpec(n, k, UNITS.get(n, 25))
                                        for n, k in combo]


def _caps(specs) -> dict:
    return {s.name: CAPS.get(s.name, 0.25 if s.kind == ammm.VISUAL else 1.0)
            for s in specs}


def _run(combo, steps=200, seed=0, **kw):
    specs = _specs(combo)
    caps = kw.pop('caps', None) or _caps(specs)
    gen = torch.Generator().manual_seed(seed)
    return specs, [plan_step(specs, caps=caps, generator=gen, **kw)
                   for _ in range(steps)]


# --------------------------------------------------------------------------- #
# G1/G2/G7 — caps, ranges and the non-empty encoder input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('combo', sorted(COMBOS))
def test_caps_ranges_and_never_empty(combo):
    """kept <= limit per stream, kept >= 0, and sum(kept) >= 1 on every step."""
    specs, plans = _run(combo, steps=300, seed=7)
    limits = {s.name: (s.n_units if s.kind == ammm.SIGNAL
                       else math.floor(CAPS.get(s.name, 0.25) * s.n_units))
              for s in specs}
    for p in plans:
        assert p.kept_total >= 1, 'encoder input must never be empty (G7)'
        assert set(p.kept_units) == set(p.active)
        for name, kept in p.kept_units.items():
            assert 0 <= kept <= limits[name], (name, kept, limits[name])


def test_visual_kept_never_exceeds_the_quarter_cap():
    """tir 36 -> <= 9 and rgb 64 -> <= 16 spatial patches (G1)."""
    _, plans = _run('rgb_tir_bp_resp', steps=400, seed=3)
    assert max(p.kept_units['tir'] for p in plans) <= 9
    assert max(p.kept_units['rgb'] for p in plans) <= 16
    assert min(p.kept_units['tir'] for p in plans) >= 0


def test_counts_vary_across_steps_but_one_count_per_stream_per_step():
    """G4/G7: a step yields ONE scalar per stream; steps may differ."""
    _, plans = _run('tir_resp', steps=200, seed=11)
    tir = [p.kept_units['tir'] for p in plans]
    resp = [p.kept_units['resp'] for p in plans]
    assert len(set(tir)) > 1 and len(set(resp)) > 1
    assert len(plans[0].kept_units) == len(plans[0].active)


def test_zero_kept_is_legal_and_still_never_empty():
    """G2/G15: 0 kept (100 % masked) happens, and the step still encodes."""
    _, plans = _run('rgb_tir_bp_resp', steps=500, seed=5,
                    alpha=0.05)          # sparse draws exercise the 0 case
    zeroed = [p for p in plans if 0 in p.kept_units.values()]
    assert zeroed, 'alpha=0.05 should produce at least one fully masked stream'
    assert all(p.kept_total >= 1 for p in plans)


# --------------------------------------------------------------------------- #
# G19 — kept_ratio_override precedence (the ablation must survive)
# --------------------------------------------------------------------------- #
def test_override_zero_survives_min_kept_one():
    specs, plans = _run('tir_resp', steps=50, seed=1, min_kept=1,
                        overrides={'tir': 0.0})
    for p in plans:
        assert p.kept_units['tir'] == 0, 'min_kept must not override a pin'
        assert p.kept_units['resp'] >= 1, 'min_kept still applies to drawn'
    assert specs  # specs kept for readability of the parametrisation


def test_override_is_clamped_by_the_cap_only_not_by_min_kept():
    specs = _specs('tir_resp')
    caps = _caps(specs)
    plan = plan_step(specs, caps=caps, min_kept=5, overrides={'tir': 0.0},
                     generator=torch.Generator().manual_seed(0))
    assert plan.kept_units['tir'] == 0
    assert 1 <= plan.kept_units['resp'] <= 25


def test_override_above_cap_raises():
    specs = _specs('tir_resp')
    with pytest.raises(ValueError, match='outside'):
        plan_step(specs, caps=_caps(specs), overrides={'tir': 0.5})


def test_override_of_unknown_stream_raises():
    specs = _specs('tir_resp')
    with pytest.raises(ValueError, match='undeclared'):
        plan_step(specs, caps=_caps(specs), overrides={'rgb': 0.0})


def test_overrides_leave_the_simplex_and_sum_r_can_exceed_one():
    """The Dirichlet simplex applies to the DRAWN subset only (design §3.2.1)."""
    specs = _specs('tir_resp')
    plan = plan_step(specs, caps=_caps(specs),
                     overrides={'tir': 0.25, 'resp': 1.0},
                     generator=torch.Generator().manual_seed(0))
    assert plan.kept_units == {'tir': 9, 'resp': 25}
    assert plan.kept_total == 34
    assert sum(plan.r.values()) == pytest.approx(1.25)


# --------------------------------------------------------------------------- #
# G20 — the degenerate single-stream draw must keep varying
# --------------------------------------------------------------------------- #
def test_single_free_stream_is_not_pinned_to_one():
    """Pinning tir must NOT leave resp permanently at 100 % kept (G20)."""
    _, plans = _run('tir_resp', steps=200, seed=2, overrides={'tir': 0.0})
    r = [p.r['resp'] for p in plans]
    kept = {p.kept_units['resp'] for p in plans}
    assert len(set(r)) > 100, 'a 1-D Dirichlet would give a constant 1.0'
    assert len(kept) > 3
    assert max(r) <= 1.0 and min(r) >= 0.0


# --------------------------------------------------------------------------- #
# G15 — global-visibility guard: total, deterministic, one unit
# --------------------------------------------------------------------------- #
def test_all_pinned_zero_grants_exactly_one_unit_to_the_first_declared():
    specs = _specs('tir_resp')
    plan = plan_step(specs, caps=_caps(specs),
                     overrides={'tir': 0.0, 'resp': 0.0},
                     generator=torch.Generator().manual_seed(0))
    assert plan.kept_units == {'tir': 1, 'resp': 0}
    assert plan.kept_total == 1


def test_guard_prefers_the_larger_pinned_ratio():
    """tir 0.005 -> 0 units, resp 0.015 -> 0 units; resp has the larger ratio."""
    specs = _specs('tir_resp')
    plan = plan_step(specs, caps=_caps(specs),
                     overrides={'tir': 0.005, 'resp': 0.015},
                     generator=torch.Generator().manual_seed(0))
    assert plan.kept_units['resp'] == 1
    assert plan.kept_units['tir'] == 0


# --------------------------------------------------------------------------- #
# G3/G14 — co-existence fallback picks the first DECLARED stream of the kind
# --------------------------------------------------------------------------- #
def test_full_dropout_restores_first_declared_of_each_kind():
    specs = _specs('rgb_tir_bp_resp')
    plan = plan_step(specs, caps=_caps(specs), dropout_p=1.0,
                     generator=torch.Generator().manual_seed(0))
    assert plan.active == ('rgb', 'bp')          # first visual + first signal
    assert set(plan.dropped) == {'rgb', 'tir', 'bp', 'resp'}
    assert plan.kept_total >= 1


def test_tir_only_visual_run_restores_tir_not_rgb():
    specs = _specs('tir_resp')
    plan = plan_step(specs, caps=_caps(specs), dropout_p=1.0,
                     generator=torch.Generator().manual_seed(0))
    assert plan.active == ('tir', 'resp')


def test_dropout_actually_drops_and_keeps_coexistence():
    _, plans = _run('rgb_tir_bp_resp', steps=300, seed=9, dropout_p=0.5)
    assert any(p.dropped for p in plans)
    for p in plans:
        kinds = {s.kind for s in _specs('rgb_tir_bp_resp') if s.name in p.active}
        assert kinds == set(ammm.KINDS)


# --------------------------------------------------------------------------- #
# G12 — declared-set / configuration validation (ValueError, never IndexError)
# --------------------------------------------------------------------------- #
def test_visual_only_declared_set_is_rejected():
    specs = [StreamSpec('rgb', ammm.VISUAL, 64), StreamSpec('tir', ammm.VISUAL, 36)]
    with pytest.raises(ValueError, match='>= 1 signal'):
        plan_step(specs, caps={'rgb': 0.25, 'tir': 0.25})


def test_signal_only_declared_set_is_rejected():
    specs = [StreamSpec('bp', ammm.SIGNAL, 25), StreamSpec('resp', ammm.SIGNAL, 25)]
    with pytest.raises(ValueError, match='>= 1 visual'):
        plan_step(specs, caps={'bp': 1.0, 'resp': 1.0})


@pytest.mark.parametrize('bad', [
    [StreamSpec('tir', ammm.VISUAL, 36), StreamSpec('tir', ammm.SIGNAL, 25)],
    [StreamSpec('tir', 'audio', 36), StreamSpec('resp', ammm.SIGNAL, 25)],
    [StreamSpec('tir', ammm.VISUAL, 0), StreamSpec('resp', ammm.SIGNAL, 25)],
])
def test_malformed_specs_raise(bad):
    with pytest.raises(ValueError):
        plan_step(bad, caps={s.name: 1.0 for s in bad})


def test_cap_that_forbids_any_kept_unit_raises():
    specs = _specs('tir_resp')
    with pytest.raises(ValueError, match='0 kept units'):
        plan_step(specs, caps={'tir': 0.01, 'resp': 1.0})


@pytest.mark.parametrize('kw', [{'dropout_p': -0.1}, {'dropout_p': 1.1},
                                {'min_kept': -1}])
def test_invalid_scalars_raise(kw):
    specs = _specs('tir_resp')
    with pytest.raises(ValueError):
        plan_step(specs, caps=_caps(specs), **kw)


# --------------------------------------------------------------------------- #
# determinism + helper surface
# --------------------------------------------------------------------------- #
def test_same_seed_replays_the_plan_and_a_different_seed_does_not():
    specs = _specs('rgb_tir_bp_resp')
    kw = dict(caps=_caps(specs), dropout_p=0.3)
    a = [plan_step(specs, generator=torch.Generator().manual_seed(4), **kw)
         for _ in range(3)]
    b = [plan_step(specs, generator=torch.Generator().manual_seed(4), **kw)
         for _ in range(3)]
    c = [plan_step(specs, generator=torch.Generator().manual_seed(99), **kw)
         for _ in range(3)]
    assert [(p.active, p.kept_units, p.dropped) for p in a] == \
           [(p.active, p.kept_units, p.dropped) for p in b]
    assert a != c


def test_masked_units_complement_and_plan_is_count_only():
    specs = _specs('rgb_tir_bp_resp')
    plan = plan_step(specs, caps=_caps(specs),
                     generator=torch.Generator().manual_seed(0))
    masked = plan.masked_units(specs)
    for s in specs:
        if s.name in plan.active:
            assert plan.kept_units[s.name] + masked[s.name] == s.n_units
    assert isinstance(plan, MaskPlan)
    assert plan.kept_total == sum(plan.kept_units.values())
    # counts only: nothing tensor-shaped is returned
    assert all(isinstance(v, int) for v in plan.kept_units.values())


# --------------------------------------------------------------------------- #
# MultiModalMAE wiring: `masking: ammm` vs the untouched legacy path
# --------------------------------------------------------------------------- #
def _tiny_ammm(**over):
    """Smallest model on the AMMM path (tir is a 1x1 patch grid, so the kept
    cap must be 1.0: a 25 % cap cannot keep a single patch)."""
    kw = dict(streams=('tir', 'resp'), embed_dim=32, enc_depth=1,
              enc_num_heads=2, dec_depth=1, num_frames=20, input_size=16,
              sig_kernel=8, seq_len=80, fps=25.0, fs=100.0, temporal_stride=1,
              masking='ammm', video_kept_cap=1.0)
    kw.update(over)
    return mm.MultiModalMAE(**kw)


def _tiny_batch(model, B=2):
    return {'tir': torch.rand(B, 3, model.num_frames, model.input_size,
                              model.input_size),
            'resp': torch.randn(B, 1, model.seq_len)}


def test_model_masks_are_the_plan_counts():
    """``make_masks`` honours the plan: masked == n_units - kept per stream.

    A video kept count is in SPATIAL patches and is replicated over the
    ``grid_t`` tubelet steps, so its masked TOKEN count is
    ``n_visual - kept*grid_t``; a 1-D kept count is already in tokens.
    """
    model = _tiny_ammm()
    for _ in range(5):
        masks = model.make_masks(2, torch.device('cpu'))
        plan = model._last_mask_plan
        assert plan.kept_total >= 1
        for spec in model.ammm_specs:
            per_unit = model.grid_t if spec.kind == ammm.VISUAL else 1
            n = (model.n_visual if spec.kind == ammm.VISUAL
                 else model.n_signal)
            masked = int(masks[spec.name].sum(dim=1)[0])
            assert masked == n - plan.kept_units[spec.name] * per_unit


def test_model_skips_the_adapter_of_a_fully_masked_stream():
    """G16: 0 kept => no adapter call, no gradient, but the loss still exists."""
    model = _tiny_ammm(kept_ratio_override={'tir': 0.0})
    out = model(_tiny_batch(model))
    assert bool(out['masks']['tir'].all()), 'the stream must be 100 % masked'
    assert float(out['losses_mse']['tir'].detach()) > 0.0   # all tokens scored
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert model.adapters['tir'].patch_embed.weight.grad is None
    assert model.enc_norm.weight.grad is not None    # the other stream carries it


def test_model_fully_kept_stream_reports_an_explicit_zero():
    """G6: a 0 %-masked stream has no masked position -> key present, value 0."""
    model = _tiny_ammm(kept_ratio_override={'tir': 1.0})
    out = model(_tiny_batch(model))
    assert int(out['masks']['tir'].sum()) == 0
    assert 'tir' in out['losses_mse']
    assert float(out['losses_mse']['tir'].detach()) == 0.0


def test_model_reports_the_plan_for_logging():
    """AMMM adds plan diagnostics; the legacy output dict is unchanged."""
    model = _tiny_ammm()
    out = model(_tiny_batch(model))
    for key in ('active', 'kept', 'r', 'n_batch'):
        assert key in out, key
    # one entry per DECLARED stream (config-stable key set for the metric logger)
    assert set(out['kept']) == set(model.streams)
    assert out['n_batch'] >= 1

    legacy = mm.MultiModalMAE(streams=('tir', 'resp'), embed_dim=32,
                              enc_depth=1, enc_num_heads=2, dec_depth=1,
                              num_frames=20, input_size=16, sig_kernel=8,
                              seq_len=80, fps=25.0, fs=100.0,
                              temporal_stride=1)
    out_legacy = legacy(_tiny_batch(legacy))
    assert 'kept' not in out_legacy and 'n_batch' not in out_legacy


def test_model_legacy_path_is_the_default_and_has_no_ammm_state():
    model = mm.MultiModalMAE(streams=('tir', 'resp'), embed_dim=32,
                             enc_depth=1, enc_num_heads=2, dec_depth=1,
                             num_frames=20, input_size=16, sig_kernel=8,
                             seq_len=80, fps=25.0, fs=100.0, temporal_stride=1)
    assert model.masking == 'ratio'
    assert model.ammm_specs == [] and model.ammm_caps == {}
    masks = model.make_masks(4, torch.device('cpu'))
    assert set(masks) == {'tir', 'resp'}             # legacy = mask_ratio_*


def test_model_rejects_the_unimplemented_ammm_combinations():
    with pytest.raises(NotImplementedError, match='modality dropout'):
        _tiny_ammm(modality_dropout=0.3)
    with pytest.raises(NotImplementedError, match='span'):
        _tiny_ammm(physio_mask='span', mask_span_s={'resp': 0.32})


def test_model_rejects_an_unknown_masking_name():
    with pytest.raises(ValueError, match="'ratio' or 'ammm'"):
        _tiny_ammm(masking='asymmetric')


def test_model_rejects_a_cap_that_cannot_keep_one_unit():
    with pytest.raises(ValueError, match='0 kept units'):
        _tiny_ammm(video_kept_cap=0.1)     # 1x1 patch grid, cap 0.1 -> limit 0


def test_model_rejects_an_override_above_the_cap():
    # 2x2 patch grid so cap 0.5 is itself valid (limit 2); the override is not
    with pytest.raises(ValueError, match='outside'):
        _tiny_ammm(input_size=32, video_kept_cap=0.5,
                   kept_ratio_override={'tir': 0.9})


@pytest.mark.parametrize('spec,expected', [
    (0.25, {'*': 0.25}),
    ('0.25', {'*': 0.25}),
    ('tir=0.25,resp=1.0', {'tir': 0.25, 'resp': 1.0}),
    ({'tir': 0.25}, {'tir': 0.25}),
    ('', {}),
    (None, {}),
])
def test_parse_kept_cap_forms(spec, expected):
    assert mm.parse_kept_cap(spec) == expected
