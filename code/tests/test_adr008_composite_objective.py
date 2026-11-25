"""ADR-008 §3.1 conformance: the composite respiration objective.

Gates (see ``code/notes/plan/features/ADR-008-...-plan.md`` §4 Phase 6):
    G1  neutral bit-identity / no module on the neutral path (C1.1, D1, M2)
    G2  anti-collapse on a synthetic sinusoid (C1.2)
    G3  Mode-2 masking integrity + degenerate targets (C1.3)
    G6  FP16 / BF16 autocast safety (C1.5)
    G7  startup rejection of every §2.6 precondition (C1.7, D3)
    G8  observability key sets, per step and per validation pass (§2.7)
    G9  non-destructive gradient-balance probe (§2.7, D6)

Everything here runs on CPU in ~10 s: the geometry-sensitive numbers are tested
on PURE tensors at the shipped shape (``L = 8000``, ``k = 32``), and the model
integration runs on ``_tiny_model`` with the ADR-007 single-visual policy.
"""
import json
import math

import pytest
import torch

from core.criterion import MaskedMSELoss
from core.multimae import _tiny_model
from core.resp_composite import (BandLimitedSpectralLoss, CompositeRespLoss,
                                 MaskedPearsonLoss)
from engines.pretrain import evaluate_pretrain
from scripts.project.capture_adr008_neutrality import FIXTURE, capture_one

CPU = torch.device('cpu')

#: the shipped ADR-007 geometry the ADR-008 numbers are stated at
FS, K, NS = 1000.0, 32, 250
SEQ_LEN = K * NS                       # 8000
M_PTS = {'mode1': NS * K,              # 8000
         'mode2': {'k25': (NS - 25) * K, 'k100': (NS - 100) * K}}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sine(freq: float, n: int = SEQ_LEN, fs: float = FS):
    """Unit-variance sinusoid at ``freq`` Hz (Var = 1 by construction)."""
    t = torch.arange(n, dtype=torch.float64) / fs
    return (math.sqrt(2.0) * torch.sin(2.0 * math.pi * freq * t)).float()


def _composite(**over):
    kw = dict(fs=FS, sig_kernel=K, n_signal=NS, mse_weight=1.0,
              pearson_weight=2.0, spectral_weight=1.0)
    kw.update(over)
    return CompositeRespLoss(**kw)


def adr008_model(**over):
    """The smallest model that carries the ADR-008 objective.

    One visual + one physiological stream (the §2.4(8) cardinality), the
    ADR-007 single-visual masking policy (``n_budget 80`` = 8 tubes in Mode 1),
    and ``target_norm='none'`` because §2.6 rule 6 forbids per-token
    normalisation for a composite term.
    """
    kw = dict(streams=('tir', 'resp'), n_budget=80, video_cap_ratio=0.5,
              video_min_tubes=7, target_norm='none')
    kw.update(over)
    return _tiny_model(**kw)


def _batch(model, batch: int = 3, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    return {
        'tir': torch.randn(batch, 3, model.num_frames, model.input_size,
                           model.input_size, generator=g),
        'resp': torch.randn(batch, 1, model.seq_len, generator=g),
    }


def _plan(model, mode: int):
    counts = ({'tir': 80, 'resp': 0} if mode == 1
              else {'tir': 70, 'resp': 10})
    return model.apportioner.plan_from_counts(counts)


# --------------------------------------------------------------------------- #
# G1 — neutral defaults are the pre-change run (C1.1, D1, M2)
# --------------------------------------------------------------------------- #
def test_g1_neutral_objective_is_bit_identical_to_the_phase0_fixture():
    """The captured baseline must reproduce exactly: values, grads, params, keys."""
    torch.set_num_threads(1)
    with open(FIXTURE) as fh:
        ref = json.load(fh)['tiny']
    got = capture_one('tiny', backward=True)
    assert got['n_parameters'] == ref['n_parameters']
    assert got['state_keys'] == ref['state_keys']
    assert got['modes'] == ref['modes']


def test_g1_neutral_model_allocates_no_objective_module():
    m = adr008_model()
    assert m.composite_active is False
    assert m.objective_active is False
    assert m.resp_loss is None and m.align_proj is None
    assert not [k for k in m.state_dict() if k.startswith('align_proj.')]

    out = m(_batch(m), plan=_plan(m, 2))
    assert 'objective' not in out and 'diagnostics' not in out


def test_g1_non_unit_alpha_activates_the_composite_without_touching_mse():
    """M2: alpha != 1.0 is NOT neutral, and `resp_mse` stays the raw MSE."""
    lam = 0.5                       # _tiny_model's signal_weight
    m = adr008_model(resp_loss_mse_weight=0.5)
    assert m.composite_active and m.objective_active
    assert m.resp_loss is not None

    out = m(_batch(m), plan=_plan(m, 1))
    raw = float(out['losses_mse']['resp'].detach())
    assert abs(out['objective']['resp_mse'] - raw) < 1e-6
    assert abs(float(out['losses']['resp'].detach()) - 0.5 * lam * raw) < 1e-6
    assert out['objective']['resp_pearson'] == 0.0
    assert out['objective']['resp_spectral'] == 0.0


# --------------------------------------------------------------------------- #
# G2 — anti-collapse (C1.2)
# --------------------------------------------------------------------------- #
def test_g2_flat_prediction_costs_pearson_and_spectral_one():
    c = _composite()
    y = _sine(0.25)
    tgt = y.reshape(1, NS, K)
    flat = torch.zeros(1, NS, K)
    mse = MaskedMSELoss()(flat, tgt, torch.ones(1, NS, dtype=torch.long))

    out = c(mse, flat, tgt, torch.zeros(1, SEQ_LEN), y.view(1, SEQ_LEN), NS)
    assert abs(float(out['pearson']) - 1.0) < 1e-5
    assert abs(float(out['spectral']) - 1.0) < 1e-4

    # the repelling gradient: a flat prediction is not a stationary point of
    # L_Pearson (its gradient scales as ||p_tilde||^-1, §2.2.2)
    p = torch.zeros(1, NS, K, requires_grad=True)
    p_full = torch.zeros(1, SEQ_LEN, requires_grad=True)
    out = c(mse, p, tgt, p_full, y.view(1, SEQ_LEN), NS)
    (out['pearson'] + out['spectral']).backward()
    assert float(p.grad.abs().sum()) > 0.0


def test_g2_exact_prediction_costs_nothing():
    c = _composite()
    y = _sine(0.25)
    tgt = y.reshape(1, NS, K)
    mse = MaskedMSELoss()(tgt, tgt, torch.ones(1, NS, dtype=torch.long))
    out = c(mse, tgt, tgt, y.view(1, SEQ_LEN), y.view(1, SEQ_LEN), NS)
    for key in ('mse', 'pearson', 'spectral', 'total'):
        assert float(out[key]) < 1e-6


@pytest.mark.parametrize('freq', [0.10, 0.60])
def test_g2_band_edges_are_penalised(freq):
    """A prediction on the WRONG in-band rate still costs (no free lunch at the edge)."""
    spec = BandLimitedSpectralLoss(fs=FS, seq_len=SEQ_LEN)
    y = _sine(0.25)
    assert float(spec(_sine(freq).view(1, SEQ_LEN), y.view(1, SEQ_LEN))) > 0.5


@pytest.mark.parametrize('freq', [0.05, 0.90])
def test_g2_out_of_band_predictions_do_not_reduce_the_loss(freq):
    """The passband must be selective: outside-band energy cannot match the target."""
    spec = BandLimitedSpectralLoss(fs=FS, seq_len=SEQ_LEN)
    y = _sine(0.25)
    assert float(spec(_sine(freq).view(1, SEQ_LEN), y.view(1, SEQ_LEN))) > 0.5


# --------------------------------------------------------------------------- #
# G3 — Mode-2 masking integrity + degenerate targets (C1.3)
# --------------------------------------------------------------------------- #
def test_g3_mode2_terms_are_computed_from_masked_tokens_only():
    """Contracts 2/3 through the REAL forward: visible slots cannot contribute."""
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0)
    x = _batch(m)
    out = m(x, plan=_plan(m, 2))

    pred = out['preds']['resp']                     # zeros at visible slots
    mask = out['masks']['resp']                     # [B, N] 1 = masked
    tgt = x['resp'].reshape(x['resp'].shape[0], m.n_signal, m.sig_kernel)
    b, n_masked = pred.shape[0], int(mask[0].sum())
    ids = torch.argsort(mask, dim=1, descending=True, stable=True)[:, :n_masked]
    expand = ids.unsqueeze(-1).expand(-1, -1, m.sig_kernel)
    pm = pred.gather(1, expand).reshape(b, -1)
    tm = tgt.gather(1, expand).reshape(b, -1)

    ref_pearson = MaskedPearsonLoss()(pm, tm)
    mask_col = mask.unsqueeze(-1).to(pred.dtype)
    p_full = (mask_col * pred + (1.0 - mask_col) * tgt).reshape(b, -1)
    ref_spectral = BandLimitedSpectralLoss(fs=m.fs, seq_len=m.seq_len)(
        p_full, tgt.reshape(b, -1))

    assert abs(out['objective']['resp_pearson'] - float(
        ref_pearson.detach())) < 1e-6
    assert abs(out['objective']['resp_spectral'] - float(
        ref_spectral.detach())) < 1e-5

    # predictions at UNMASKED positions are placeholders and must be inert
    assert torch.equal(pred[mask == 0], torch.zeros_like(pred[mask == 0]))
    junk = pred.clone()
    junk[mask == 0] = 1e3
    p_junk = (mask_col * junk + (1.0 - mask_col) * tgt).reshape(b, -1)
    assert torch.equal(p_junk, p_full)
    assert torch.equal(MaskedPearsonLoss()(pm, tm), ref_pearson)


def test_g3_degenerate_zero_target_is_a_finite_zero():
    c = _composite()
    zeros = torch.zeros(1, NS, K)
    p = zeros.clone().requires_grad_(True)
    p_full = torch.zeros(1, SEQ_LEN, requires_grad=True)
    out = c(torch.zeros(()), p, zeros, p_full, torch.zeros(1, SEQ_LEN), NS)
    assert float(out['pearson']) == 0.0
    assert float(out['spectral']) == 0.0
    # EXCLUDED, not "zero with garbage gradients": the terms are constants, so
    # there is no graph to differentiate (and no NaN/Inf anywhere)
    total = out['pearson'] + out['spectral']
    assert torch.isfinite(total)
    assert not total.requires_grad


def test_g3_zero_masked_tokens_emits_explicit_zeros():
    c = _composite()
    out = c(torch.tensor(2.0), None, None, None, None, 0)
    assert float(out['total']) == 0.0
    for key in ('mse', 'pearson', 'spectral'):
        assert float(out[key]) == 0.0


# --------------------------------------------------------------------------- #
# G6 — mixed precision (C1.5)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_g6_autocast_forward_and_backward_stay_finite(dtype):
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0,
                     align_weight=0.5, align_proj_dim=16)
    x = _batch(m)
    plan = _plan(m, 2)
    m.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=dtype):
        out = m(x, plan=plan)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


# --------------------------------------------------------------------------- #
# G7 — startup rejection (C1.7, §2.6, D3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('over, needle', [
    # rule 1
    (dict(resp_loss_mse_weight=0.0), 'rule 1'),
    (dict(resp_loss_mse_weight=-0.1), 'must be >= 0'),
    # rule 2
    (dict(resp_loss_spectral_weight=1.0, spectral_f_min=0.0), 'rule 2'),
    (dict(resp_loss_spectral_weight=1.0, spectral_f_max=60.0), 'rule 2'),
    # rule 3
    (dict(resp_loss_spectral_weight=1.0, spectral_n_fft=64), 'rule 3'),
    (dict(resp_loss_spectral_weight=1.0, spectral_f_min=0.40,
          spectral_f_max=0.41), '|K_resp| >= 1'),
    # rule 4
    (dict(align_weight=0.5, align_temperature=0.0), 'rule 4'),
    (dict(align_weight=0.5, align_proj_dim=8), 'rule 4'),
    (dict(align_weight=0.5, align_core_weight=0.0), 'rule 4'),
    (dict(streams=('tir', 'rgb2d', 'resp'), align_weight=0.5), 'rule 4'),
    # rule 5
    (dict(grad_balance_freq=-1), 'rule 5'),
    # rule 6 (D3) -- target coherence, and the beta half of the guard
    (dict(resp_loss_spectral_weight=1.0, target_norm='token'), 'rule 6'),
    (dict(resp_loss_pearson_weight=1.0, target_norm='token'), 'rule 6'),
    (dict(streams=('tir', 'rgb2d', 'bp', 'resp'),
          resp_loss_pearson_weight=1.0), 'exactly ONE physiological stream'),
])
def test_g7_infeasible_objective_rejected_at_construction(over, needle):
    with pytest.raises(ValueError, match=needle):
        adr008_model(**over)


def test_g7_defaults_construct():
    """The rejection suite is only meaningful if the legal tuple builds."""
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0, align_weight=0.5,
                     align_proj_dim=16, grad_balance_freq=50)
    assert m.composite_active and m.align_proj is not None
    assert m.resp_loss.spectral.n_fft_resolved == 1024      # 4*2**ceil(log2 160)
    assert m.resp_loss.spectral.band_bin_list == [2, 3, 4, 5, 6]


# --------------------------------------------------------------------------- #
# G8 — observability (§2.7)
# --------------------------------------------------------------------------- #
OBJECTIVE_KEYS = {'resp_mse', 'resp_pearson', 'resp_spectral', 'loss_align',
                  'align_active'}


@pytest.mark.parametrize('mode', [1, 2])
def test_g8_objective_key_set_is_config_determined(mode):
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0, align_weight=0.5,
                     align_proj_dim=16)
    out = m(_batch(m), plan=_plan(m, mode))
    assert set(out['objective']) == OBJECTIVE_KEYS
    assert out['objective']['align_active'] == (1 if mode == 2 else 0)
    # the raw per-stream MSE dict keeps its STREAM names (the `mse_` prefix is
    # added by the engine's logger, not by the model)
    assert set(out['losses_mse']) == {'tir', 'resp'}


def test_g8_validation_block_is_a_sibling_of_components():
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0, align_weight=0.5,
                     align_proj_dim=16)
    val = evaluate_pretrain(m, [_batch(m, 2), _batch(m, 2, seed=4)], CPU,
                            mask_base=7, epoch=0)
    assert set(val['components']) == {'mse_tir', 'mse_resp'}
    assert set(val['objective']) == OBJECTIVE_KEYS
    assert 'stratification' in val


def test_g8_neutral_validation_block_is_empty():
    m = adr008_model()
    val = evaluate_pretrain(m, [_batch(m, 2)], CPU, mask_base=7, epoch=0)
    assert val['objective'] == {}
    assert set(val['components']) == {'mse_tir', 'mse_resp'}


# --------------------------------------------------------------------------- #
# G9 — non-destructive gradient-balance probe (§2.7, D6)
# --------------------------------------------------------------------------- #
def test_g9_probe_is_emitted_only_on_probe_steps():
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0, grad_balance_freq=2)
    x, plan = _batch(m), _plan(m, 2)
    for step, present in ((0, True), (1, False), (2, True), (3, False)):
        out = m(x, plan=plan, step=step)
        assert ('diagnostics' in out) is present, step
        if present:
            d = out['diagnostics']
            assert d['grad_norm_tir'] is not None and d['grad_norm_tir'] > 0
            assert d['grad_norm_resp'] is not None and d['grad_norm_resp'] > 0
            assert d['grad_ratio_resp_to_tir'] > 0
            assert math.isfinite(d['grad_ratio_resp_to_tir'])


def test_g9_probe_never_touches_parameter_grads():
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0, grad_balance_freq=1)
    x, plan = _batch(m), _plan(m, 2)
    out = m(x, plan=plan, step=0)
    # nothing has been backwarded yet: the probe must not create `.grad`
    assert all(p.grad is None for p in m.parameters())

    out['loss'].backward()
    before = {n: (None if p.grad is None else p.grad.detach().clone())
              for n, p in m.named_parameters()}
    m(x, plan=plan, step=2)                       # probe runs inside forward
    after = {n: (None if p.grad is None else p.grad) for n, p in
             m.named_parameters()}
    for name, ref in before.items():
        if ref is None:
            assert after[name] is None
        else:
            assert torch.equal(ref, after[name]), name


def test_g9_probe_value_matches_a_reference_dz_slice():
    """The probe = || dL/dz || over the visual rows, taken from the FULL z."""
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0, grad_balance_freq=1)
    x, plan = _batch(m), _plan(m, 2)
    captured = {}

    hook = m.enc_norm.register_forward_hook(
        lambda mod, inp, out: captured.setdefault('z', out))
    try:
        out = m(x, plan=plan, step=0)
    finally:
        hook.remove()
    z = captured['z']
    # Mode 2 keeps 70 visual tokens, and `tir` is the first stream in `z`
    ref = float(torch.autograd.grad(
        out['losses']['tir'], z, retain_graph=True)[0][:, :70].norm(2))
    assert abs(out['diagnostics']['grad_norm_tir'] - ref) < 1e-5 * max(1.0, ref)


def test_g9_freq_zero_costs_nothing():
    m = adr008_model(resp_loss_pearson_weight=2.0,
                     resp_loss_spectral_weight=1.0)
    calls = []
    real = torch.autograd.grad

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    torch.autograd.grad = counting
    try:
        out = m(_batch(m), plan=_plan(m, 2), step=0)
    finally:
        torch.autograd.grad = real
    assert 'diagnostics' not in out
    assert calls == []
