"""Stage-2 protocol contract: pinned masks, deterministic validation, refusals.

These cover the parts of ADR-001 that only exist once Stage 2 is wired up:
I5 (the test role is never loaded), I6/Criterion 2 (a validation pass is a
deterministic function of the weights) and I7's "only the val role selects".

ADR-002 replaced the per-stream ``val_kept_<stream>`` pins with a FIXED-SEED mask
draw sequence: the counts generator is seeded from ``(mask_base, epoch, step)``
without the rank, so the validation pass replays the same modes and counts.
"""
import argparse

import pytest
import torch

from core.mask_placement import counts_generator, placement_generator
from core.multimae import _tiny_model
from data.datasets import build_pretraining_dataset
from engines.pretrain import evaluate_pretrain
from protocol import ProtocolError, build_plan

CPU = torch.device('cpu')


def _masks(model, seed=None, **kwargs):
    cg = pg = None
    if seed is not None:
        cg = counts_generator(seed, 0, 0)
        pg = placement_generator(seed, 0, 0, rank=0)
    with torch.no_grad():
        masks, plan, _idx = model.make_masks(
            2, CPU, counts_generator=cg, placement_generator=pg, **kwargs)
    return masks


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
    """With no generator the draw uses the global RNG, so seeding it pins the
    masks -- the property the training path relies on for replayability."""
    model = _tiny_model()
    torch.manual_seed(0)
    first = model.make_masks(2, CPU)[0]
    torch.manual_seed(0)
    second = model.make_masks(2, CPU)[0]
    for stream in first:
        assert torch.equal(first[stream], second[stream])


def test_an_explicit_plan_pins_the_counts_exactly():
    """An evaluation states the allocation; the masks must match it."""
    model = _tiny_model()
    plan = model.plan_from_counts({'tir': 40, 'rgb2d': 40, 'resp': 0})
    masks, got, idx = model.make_masks(4, CPU, plan=plan,
                                       placement_generator=torch.Generator(
                                           ).manual_seed(1))
    assert got is plan
    assert int((masks['tir'] == 0).sum(dim=1).max()) == 40
    assert int((masks['rgb2d'] == 0).sum(dim=1).max()) == 40
    assert int((masks['resp'] == 0).sum()) == 0        # fully masked stream
    assert idx['tir'].shape == (4, 40)


def test_an_impossible_plan_is_refused():
    model = _tiny_model()
    with pytest.raises(ValueError, match='n_budget|visible tokens'):
        model.plan_from_counts({'tir': 80, 'rgb2d': 80, 'resp': 0})


def test_span_placement_keeps_the_exact_count():
    model = _tiny_model(physio_placement='span', physio_span_tokens=3)
    plan = model.plan_from_counts({'tir': 40, 'rgb2d': 40, 'resp': 0})
    masks, _plan_out, _idx = model.make_masks(2, CPU, plan=plan)
    assert int((masks['resp'] == 0).sum()) == 0


# --------------------------------------------------------------------------- #
# I6 / Criterion 2 — the validation pass
# --------------------------------------------------------------------------- #
class _MaskSensitiveModel(torch.nn.Module):
    """Stub whose loss depends on the mask draw it is given."""

    def forward(self, x, counts_generator=None, placement_generator=None,
                plan=None):
        generator = counts_generator or torch.Generator().manual_seed(0)
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
                               mask_base=5)
    assert set(result['components']) == {'mse_resp'}
    assert result['mask_base'] == 5
    assert result['pinned'] == {}      # ADR-002: no fraction is pinned


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
# D6 replacement — validation replays a deterministic mask DRAW sequence
# --------------------------------------------------------------------------- #
def test_validation_replays_the_same_plan_sequence():
    """Two validation passes derive identical counts (no hand-pinned fractions).

    ADR-002 §6: the counts generator is seeded from ``(mask_base, batch)`` and
    the modes/counts therefore replay exactly, while the loss stays a function
    of the weights only.
    """
    model = _tiny_model()
    a = [model.make_masks(2, CPU,
                          counts_generator=counts_generator(5, 0, i),
                          placement_generator=placement_generator(5, 0, i,
                                                                  rank=0))[1]
         for i in range(3)]
    b = [model.make_masks(2, CPU,
                          counts_generator=counts_generator(5, 0, i),
                          placement_generator=placement_generator(5, 0, i,
                                                                  rank=0))[1]
         for i in range(3)]
    assert [p.kept_tokens for p in a] == [p.kept_tokens for p in b]
    assert [p.mode for p in a] == [p.mode for p in b]
