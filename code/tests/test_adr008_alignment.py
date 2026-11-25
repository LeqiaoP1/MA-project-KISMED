"""ADR-008 §3.1 conformance: cross-modal alignment, manifest and transfer.

Gates (see ``code/notes/plan/features/ADR-008-...-plan.md`` §4 Phase 6):
    G4  InfoNCE directivity, anatomy weighting, single-row safety (C1.4)
    G5  Mode-1 gate: zero loss, zero gradient, still in the graph (C1.4/§2.4(6))
    G10 manifest record + resume guard (D4, §2.7)
    G12 the arm YAML resolves the §2.8 tuple (config gate; the resource half is
        a smoke-run deliverable)
    G11 Stage-3 transfer isolation (C1.6, R6)

``G8``'s DDP aggregation half lives in ``test_val_gather.py`` (the objective
block is merged by the same ``_merge_sums`` path, which that suite already
covers); here only the single-process key sets are asserted.
"""
import argparse
import json
import os

import pytest
import torch

from core.align_infonce import (AlignProjection, anatomy_weighted_pool,
                                symmetric_infonce)
from core.multimae import _tiny_model
from runners.run_pretrain import write_masking_manifest

CPU = torch.device('cpu')
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = os.path.join('configs', 'pretrain',
                   'stage2_budget_tir_resp_2stream_anatomy_adr008.yaml')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def adr008_model(**over):
    """One visual + one physiological stream (the §2.4(8) cardinality)."""
    kw = dict(streams=('tir', 'resp'), n_budget=80, video_cap_ratio=0.5,
              video_min_tubes=7, target_norm='none')
    kw.update(over)
    return _tiny_model(**kw)


def _align_model(**over):
    kw = dict(resp_loss_pearson_weight=2.0, resp_loss_spectral_weight=1.0,
              align_weight=0.5, align_proj_dim=16)
    kw.update(over)
    return adr008_model(**kw)


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


class _Plan:
    """Minimal stand-in for a protocol plan (the manifest reads 2 fields)."""

    def __init__(self, fold: int = 0, digest: str = 'deadbeef'):
        self.fold = fold
        self.digest = digest


# --------------------------------------------------------------------------- #
# G4 — directivity, anatomy weighting, single-row safety (C1.4)
# --------------------------------------------------------------------------- #
def test_g4_alignment_is_gated_on_the_physiological_latent():
    m = _align_model()
    x = _batch(m)

    mode1 = m(x, plan=_plan(m, 1))
    assert mode1['objective']['align_active'] == 0
    assert mode1['objective']['loss_align'] == 0.0

    mode2 = m(x, plan=_plan(m, 2))
    assert mode2['objective']['align_active'] == 1
    assert mode2['objective']['loss_align'] > 0.0


def test_g4_alignment_reaches_the_visual_encoder():
    """The regularizer must inject gradient into the VISUAL adapter path.

    Isolation by construction: two models identical except ``align_weight``
    (the same init draws, because the extra heads are registered last and
    initialised after every pre-existing module). A non-zero difference in the
    adapters' gradients can therefore come only from the InfoNCE term.
    """
    grads = {}
    for weight in (0.0, 50.0):
        torch.manual_seed(0)
        m = _align_model(align_weight=weight)
        m.zero_grad(set_to_none=True)
        out = m(_batch(m), plan=_plan(m, 2))
        out['loss'].backward()
        grads[weight] = (m.adapters['tir'].patch_embed.weight.grad.clone(),
                         m.adapters['resp'].embed.weight.grad.clone())
    assert not torch.equal(grads[0.0][0], grads[50.0][0]), 'visual adapter'
    assert not torch.equal(grads[0.0][1], grads[50.0][1]), 'signal adapter'


def test_g4_positive_pairs_beat_permuted_pairs():
    torch.manual_seed(0)
    proj = AlignProjection(32, 16)
    raw = torch.randn(4, 32)
    va, vb = proj(raw), proj(raw.clone())
    assert float(symmetric_infonce(va, vb).detach()) < float(
        symmetric_infonce(va, vb[torch.tensor([1, 0, 3, 2])]).detach())


def test_g4_single_row_batch_returns_zero():
    proj = AlignProjection(32, 16)
    out = symmetric_infonce(proj(torch.randn(1, 32)),
                            proj(torch.randn(1, 32)))
    assert float(out) == 0.0 and not out.requires_grad


def test_g4_core_weight_emphasises_core_tokens():
    """d u / d z is core_weight / sum(w) on core rows, 1 / sum(w) elsewhere."""
    z = torch.randn(2, 6, 4, requires_grad=True)
    ids = torch.tensor([[0, 1, 4, 7, 13, 20]] * 2)     # tubes 0,1,4,7,13,20
    core = (4, 20)
    u = anatomy_weighted_pool(z, ids, core, n_spatial=36, core_weight=2.0)
    u.sum().backward()
    spatial = ids % 36
    is_core = torch.isin(spatial, torch.tensor(core))
    g = z.grad.abs().sum(dim=-1).sum(dim=0)          # per token
    assert float(g[is_core[0]].min()) > float(g[~is_core[0]].max())


def test_g4_core_weight_one_is_the_plain_mean():
    z = torch.randn(2, 6, 4)
    ids = torch.tensor([[0, 1, 4, 7, 13, 20]] * 2)
    assert torch.allclose(anatomy_weighted_pool(z, ids, (4, 20), 36, 1.0),
                          z.mean(dim=1), atol=1e-6)
    # ... and so is an EMPTY A_core (the `uniform` ablation), same code path
    assert torch.allclose(anatomy_weighted_pool(z, ids, (), 36, 2.0),
                          z.mean(dim=1), atol=1e-6)


# --------------------------------------------------------------------------- #
# G5 — Mode-1 distributed safety (C1.4, §2.4(6))
# --------------------------------------------------------------------------- #
def test_g5_mode1_gives_the_heads_a_zero_not_missing_gradient():
    m = _align_model()
    m.zero_grad(set_to_none=True)
    out = m(_batch(m), plan=_plan(m, 1))
    out['loss'].backward()
    params = dict(m.align_proj.named_parameters())
    assert params, 'the projection heads must exist when align_weight > 0'
    for name, p in params.items():
        assert p.grad is not None, f'{name} fell out of the autograd graph'
        assert float(p.grad.abs().max()) == 0.0, name


def test_g5_align_gather_is_disabled_outside_training():
    """The validation pass is unpadded, so it must never call all_gather."""
    m = _align_model()
    m.eval()
    try:
        out = m(_batch(m), plan=_plan(m, 2))
        assert out['objective']['align_active'] == 1
        assert out['objective']['loss_align'] > 0.0      # local negatives only
        assert 'diagnostics' not in out                  # never probes in eval
    finally:
        m.train(True)


# --------------------------------------------------------------------------- #
# G10 — manifest record + resume guard (D4, §2.7)
# --------------------------------------------------------------------------- #
def _manifest_args(tmp_path):
    return argparse.Namespace(output_dir=str(tmp_path), epochs=1,
                              clip_duration=1.6)


def test_g10_manifest_records_the_objective_tuple(tmp_path):
    m = _align_model(grad_balance_freq=2)
    rec = write_masking_manifest(_manifest_args(tmp_path), m,
                                 plan=_Plan(), mask_base=1234, stage1={})
    obj = rec['objective']
    assert obj['adr'] == 'ADR-008'
    assert obj['spectral_n_fft'] == 1024
    assert obj['spectral_band_bins'] == [2, 3, 4, 5, 6]
    assert obj['align_weight'] == 0.5
    assert obj['signal_weight'] == 0.5
    # a LOGGING cadence is recorded under `run`, not inside the objective
    assert rec['run']['grad_balance_freq'] == 2
    assert 'grad_balance_freq' not in obj
    with open(os.path.join(str(tmp_path), 'masking.json')) as fh:
        assert json.load(fh)['objective'] == obj


def test_g10_neutral_manifest_has_no_objective_block(tmp_path):
    m = adr008_model()
    rec = write_masking_manifest(_manifest_args(tmp_path), m,
                                 plan=_Plan(), mask_base=1234, stage1={})
    assert 'objective' not in rec
    assert 'grad_balance_freq' not in rec['run']


def test_g10_changing_align_weight_blocks_resume(tmp_path):
    write_masking_manifest(_manifest_args(tmp_path), _align_model(),
                           plan=_Plan(), mask_base=1234, stage1={})
    with pytest.raises(SystemExit, match='different masking policy'):
        write_masking_manifest(_manifest_args(tmp_path),
                               _align_model(align_weight=1.0),
                               plan=_Plan(), mask_base=1234, stage1={})


def test_g10_changing_the_diagnostic_cadence_still_resumes(tmp_path):
    write_masking_manifest(_manifest_args(tmp_path),
                           _align_model(grad_balance_freq=2),
                           plan=_Plan(), mask_base=1234, stage1={})
    rec = write_masking_manifest(_manifest_args(tmp_path),
                                 _align_model(grad_balance_freq=50),
                                 plan=_Plan(), mask_base=1234, stage1={})
    # the FIRST record is kept untouched
    assert rec['run']['grad_balance_freq'] == 2


def test_g10_legacy_record_without_objective_resumes_on_neutral(tmp_path):
    """A pre-ADR-008 masking.json has no `objective` key; neutral must resume."""
    args = _manifest_args(tmp_path)
    write_masking_manifest(args, adr008_model(), plan=_Plan(), mask_base=1234,
                           stage1={})
    path = os.path.join(str(tmp_path), 'masking.json')
    with open(path) as fh:
        legacy = json.load(fh)
    legacy.pop('objective', None)
    with open(path, 'w') as fh:
        json.dump(legacy, fh)
    write_masking_manifest(args, adr008_model(), plan=_Plan(), mask_base=1234,
                           stage1={})            # must not raise


# --------------------------------------------------------------------------- #
# G12 — the arm YAML resolves the §2.8 tuple
# --------------------------------------------------------------------------- #
def test_g12_arm_config_resolves_the_objective_tuple(monkeypatch):
    from runners.run_pretrain import get_args

    monkeypatch.setattr('sys.argv',
                        ['run_pretrain.py', '-c', ARM])
    a = get_args()
    assert a.streams == 'tir,resp'
    assert a.signal_weight == 5.0
    assert a.resp_loss_mse_weight == 1.0
    assert a.resp_loss_pearson_weight == 2.0
    assert a.resp_loss_spectral_weight == 1.0
    assert (a.spectral_f_min, a.spectral_f_max) == (0.10, 0.60)
    assert a.spectral_n_fft == 0
    assert a.align_weight == 0.5
    assert a.align_proj_dim == 128
    assert a.align_temperature == 0.07
    assert a.align_core_weight == 1.0
    assert a.align_gather_ddp is True
    assert a.grad_balance_freq == 50
    # the masking policy is the ADR-007 arm's, unchanged
    assert a.n_budget == 200 and a.physio_draw_min == 1
    assert a.spatial_sampling == 'anatomy_aware'


# --------------------------------------------------------------------------- #
# G11 — Stage-3 transfer isolation (C1.6, R6)
# --------------------------------------------------------------------------- #
def test_g11_transfer_copies_the_encoder_and_skips_the_alignment_heads(tmp_path):
    from core.waveform_model import build_waveform_model, load_stage2_encoder

    stage2 = _align_model()                 # embed 64, depth 1, 4 heads
    ckpt = os.path.join(str(tmp_path), 'stage2.pth')
    torch.save({'model': stage2.state_dict(), 'args': None, 'epoch': 0}, ckpt)

    stage3 = build_waveform_model(argparse.Namespace(
        streams='tir', model='', enc_embed_dim=64, enc_depth=1,
        enc_num_heads=4, mlp_ratio=4.0, tubelet='4,8,8', input_size=32,
        clip_duration=1.6, fps=25.0, temporal_stride=1, num_frames=0,
        seq_len=160, sig_kernel=16, fs=100.0, pos_init='sincos3d',
        head_hidden=0, head_style='linear', head_init='transfer',
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        tir_channels=3))

    info = load_stage2_encoder(stage3, ckpt, target='resp')

    sd = stage2.state_dict()
    assert any(k.startswith('align_proj.') for k in sd), \
        'the checkpoint must carry the heads for this test to mean anything'
    cur = stage3.state_dict()
    expected = [k for k in sd
                if k.startswith(('enc_blocks.', 'enc_norm.', 'adapters.',
                                 'positions.'))
                and k in cur and tuple(cur[k].shape) == tuple(sd[k].shape)]
    assert info['loaded'] == len(expected) > 0
    assert info['shape_mismatch'] == 0
    assert not any(k.startswith('align_proj.') for k in expected)


# --------------------------------------------------------------------------- #
# Phase 1.5 / 4.6 — package re-exports, and the recon path's own loader
# --------------------------------------------------------------------------- #
def test_phase15_core_package_reexports_the_adr008_names():
    """`core` re-exports the objective + alignment helpers (plan Phase 1.5)."""
    import core

    for name in ('MaskedPearsonLoss', 'BandLimitedSpectralLoss',
                 'CompositeRespLoss', 'AlignProjection', 'anatomy_weighted_pool',
                 'symmetric_infonce'):
        assert hasattr(core, name), name


def _tiny_pretrain_args():
    """A ``run_pretrain`` args namespace matching ``_tiny_model``'s geometry."""
    return argparse.Namespace(
        streams='tir,resp', tubelet='4,8,8', clip_duration=1.6, fps=25.0,
        temporal_stride=1, seq_len=160, input_size=32, sig_kernel=8,
        tir_channels=3, model='', enc_embed_dim=64, enc_depth=1,
        enc_num_heads=4, mlp_ratio=4.0, dec_depth=1,
        n_budget=80, video_cap_ratio=0.5, physio_cap_ratio=0.5,
        video_min_tubes=7, p_virtual_sensing=0.5,
        virtual_sensing_warmup_p=0.1, virtual_sensing_warmup_epochs=0,
        single_carrier_tubes=0, p_single_modality=0.0,
        single_modality_warmup_p=0.0, spatial_sampling='uniform',
        core_row_bounds='2,3', core_col_bounds='2,3', core_tubes_min=1,
        core_tubes_max=2, periph_min_tubes=1, physio_draw_min=0,
        spatial_patches=0, physio_tokens=0, signal_weight=0.5,
        loss_weights='', pos_init='sincos3d', target_norm='none',
        physio_placement='scattered', physio_span_tokens=0,
        # ADR-008 tuple
        resp_loss_mse_weight=1.0, resp_loss_pearson_weight=2.0,
        resp_loss_spectral_weight=1.0, spectral_f_min=0.10,
        spectral_f_max=0.60, spectral_n_fft=0, align_weight=0.5,
        align_proj_dim=16, align_temperature=0.07, align_core_weight=1.0,
        align_gather_ddp=True, grad_balance_freq=0)


def test_phase46_recon_builder_loads_an_adr008_checkpoint_cleanly(monkeypatch):
    """`run_eval_recon.build_model` rebuilds the heads from `ckpt['args']`.

    The requirement is stronger than G11's transfer check: the RECONSTRUCTION
    path must reconstruct the model so faithfully that loading the checkpoint
    reports ZERO missing and ZERO unexpected tensors -- otherwise a probe would
    silently score a partly random model (the failure mode that runner exists to
    prevent).
    """
    from core import multimae as mm
    from runners.run_eval_recon import build_model

    args = _tiny_pretrain_args()
    src = mm.build_pretraining_model(args)
    assert src.align_proj is not None and src.resp_loss is not None
    ckpt = {'model': src.state_dict(), 'args': args}

    seen = {}
    real = mm.MultiModalMAE.load_state_dict

    def spy(self, state_dict, *a, **kw):
        out = real(self, state_dict, *a, **kw)
        seen['missing'], seen['unexpected'] = out
        return out

    monkeypatch.setattr(mm.MultiModalMAE, 'load_state_dict', spy)
    built = build_model(ckpt, CPU)
    assert seen, 'build_model never called load_state_dict'
    assert seen['missing'] == []
    assert seen['unexpected'] == []
    assert built.align_proj is not None
