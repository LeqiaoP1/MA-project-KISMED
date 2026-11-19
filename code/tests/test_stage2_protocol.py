"""Stage-2 protocol contract: pinned masks, deterministic validation, refusals.

These cover the parts of ADR-001 that only exist once Stage 2 is wired up:
I5 (the test role is never loaded), I6/Criterion 2 (a validation pass is a
deterministic function of the weights) and I7's "only the val role selects".
"""
import argparse

import pytest
import torch

from core.multimae import _tiny_model
from data.datasets import build_pretraining_dataset
from engines.pretrain import evaluate_pretrain
from protocol import ProtocolError, build_plan
from runners._common import resolve_val_kept

CPU = torch.device('cpu')


def _masks(model, seed=None, **kwargs):
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    with torch.no_grad():
        return model.make_masks(2, CPU, generator=generator, **kwargs)


def _plan(fold=0):
    return build_plan([f'S{i:03d}' for i in range(1, 7)], fold,
                      split_seed=42, split_stratify='none', dev_smoke=True)


# --------------------------------------------------------------------------- #
# I6 — the mask path can be pinned, and the training path is untouched
# --------------------------------------------------------------------------- #
def test_mask_placement_is_reproducible_with_a_seeded_generator():
    model = _tiny_model()
    first, second, other = _masks(model, 7), _masks(model, 7), _masks(model, 8)
    for stream in first:
        assert torch.equal(first[stream], second[stream])
    assert any(not torch.equal(first[s], other[s]) for s in first)


def test_generator_none_is_the_global_rng():
    """The training path must stay bit-identical to the pre-pinning code.

    ``torch.rand(..., generator=None)`` IS ``torch.rand(...)``, so seeding the
    global RNG and calling ``make_masks`` twice must reproduce the same masks.
    """
    model = _tiny_model()
    torch.manual_seed(0)
    first = model.make_masks(2, CPU)
    torch.manual_seed(0)
    second = model.make_masks(2, CPU)
    for stream in first:
        assert torch.equal(first[stream], second[stream])


def test_kept_override_pins_the_masked_fraction():
    model = _tiny_model()                      # mask_ratio_resp = 0.50
    masks = _masks(model, 1, kept_override={'resp': 0.50})
    assert masks['resp'].float().mean().item() == pytest.approx(0.5)
    # kept=0 is CLAMPED to N-1 masked tokens: the ratio policy always leaves one
    # visible token, because that token is the 1-D stream's only gradient path
    # into the encoder (`_random_mask`'s `min(N - 1, ...)`). AMMM can reach a
    # genuine 0-kept stream; the ratio policy cannot, by design.
    nothing_kept = _masks(model, 1, kept_override={'resp': 0.0})
    assert nothing_kept['resp'].float().mean().item() == pytest.approx(0.9)


def test_kept_override_rejects_an_undeclared_stream():
    with pytest.raises(ValueError, match='undeclared'):
        _masks(_tiny_model(), 1, kept_override={'bp': 0.5})


def test_span_placement_cannot_be_pinned_by_a_fraction():
    """A span budget is a token count, so a kept FRACTION must be refused."""
    model = _tiny_model(physio_mask='span')
    with pytest.raises(ValueError, match='SPAN'):
        _masks(model, 1, kept_override={'resp': 0.5})


# --------------------------------------------------------------------------- #
# I6 / Criterion 2 — the validation pass
# --------------------------------------------------------------------------- #
class _MaskSensitiveModel(torch.nn.Module):
    """Stub whose loss depends on the mask draw it is given."""

    def forward(self, x, mask_generator=None, kept_override=None):
        generator = mask_generator or torch.Generator().manual_seed(0)
        draw = torch.rand(4, generator=generator)
        return {'loss': draw.sum(),
                'losses_mse': {'resp': draw.mean()},
                'masks': {'resp': torch.zeros(2, 4)}}


def _loader(batches=3):
    return [{'resp': torch.zeros(2, 1, 8)} for _ in range(batches)]


def test_evaluate_pretrain_is_deterministic():
    model = _MaskSensitiveModel()
    first = evaluate_pretrain(model, _loader(), CPU, mask_base=123)
    second = evaluate_pretrain(model, _loader(), CPU, mask_base=123)
    assert first['val_loss'] == second['val_loss']
    assert first['n_samples'] == 6 and first['n_batches'] == 3
    other = evaluate_pretrain(model, _loader(), CPU, mask_base=124)
    assert other['val_loss'] != first['val_loss']


def test_evaluate_pretrain_records_its_components():
    result = evaluate_pretrain(_MaskSensitiveModel(), _loader(1), CPU,
                               mask_base=5, kept_override={'resp': 0.5})
    assert set(result['components']) == {'mse_resp'}
    assert result['mask_base'] == 5
    assert result['pinned'] == {'resp': 0.5}


def test_evaluate_pretrain_refuses_an_empty_loader():
    with pytest.raises(SystemExit, match='no sample'):
        evaluate_pretrain(_MaskSensitiveModel(), [], CPU, mask_base=0)


# --------------------------------------------------------------------------- #
# I5 — Stage 2 may not build the test role
# --------------------------------------------------------------------------- #
def test_stage2_refuses_the_test_role():
    args = argparse.Namespace(data_set='tir_roi', subjects='')
    with pytest.raises(ProtocolError, match='never loaded'):
        build_pretraining_dataset(args, _plan(), role='test')


def test_stage2_refuses_an_empty_role_after_subject_filtering():
    args = argparse.Namespace(data_set='tir_roi', subjects='NOT_A_SUBJECT')
    with pytest.raises(RuntimeError, match='is empty'):
        build_pretraining_dataset(args, _plan(), role='train')


# --------------------------------------------------------------------------- #
# D6 — an unpinned stochastic validation pass is refused
# --------------------------------------------------------------------------- #
def test_val_kept_is_required_when_evaluating():
    args = argparse.Namespace(eval_freq=5, val_kept_tir=0.25,
                              val_kept_resp=None)
    with pytest.raises(SystemExit, match='not comparable'):
        resolve_val_kept(args, ['tir', 'resp'])
    assert resolve_val_kept(args, ['tir', 'resp'], eval_freq=0) == {'tir': 0.25}


def test_val_kept_range_is_validated():
    args = argparse.Namespace(eval_freq=5, val_kept_tir=1.5)
    with pytest.raises(SystemExit, match=r'\[0, 1\]'):
        resolve_val_kept(args, ['tir'])
