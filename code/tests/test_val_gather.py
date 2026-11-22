"""Validation must be world-size independent.

The val loader is DDP-sharded, so each rank scores only its own slice. Before
the all-gather in `evaluate_pretrain`, the reported `val_loss` was computed from
that slice alone -- a 4-rank run reported a quarter of the val role, and the
number was not comparable between a 2-GPU and a 4-GPU run of the same recipe.
"""
import os
import sys

import pytest
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from engines.pretrain import _merge_sums, evaluate_pretrain


class _StubModel(torch.nn.Module):
    """Minimal multimodal-MAE-shaped forward for the validation pass."""

    def __init__(self, loss_value=1.0):
        super().__init__()
        self.loss_value = float(loss_value)
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch, counts_generator=None, placement_generator=None):
        n = int(batch['tir'].shape[0])
        return {
            'loss': torch.tensor(self.loss_value) + self.w,
            'masks': {'tir': torch.zeros(n, 1)},
            'losses_mse': {'tir': 1.0},
            'kept': {'tir': 50.0},
            'mode': 1,
        }


def _loader(n_batches=2, batch_n=2):
    return [{'tir': torch.zeros(batch_n, 3, 2, 4, 4)} for _ in range(n_batches)]


def test_merge_sums_adds_per_rank_weights_and_tolerates_missing_keys():
    merged = _merge_sums([
        {'mse_tir': 4.0, 'mode1': 4.0},
        {'mse_tir': 6.0, 'mode2': 2.0},       # this rank never drew mode 1
    ])
    assert merged == {'mse_tir': 10.0, 'mode1': 4.0, 'mode2': 2.0}


def test_single_process_validation_is_unchanged():
    out = evaluate_pretrain(_StubModel(1.0), _loader(), torch.device('cpu'),
                            mask_base=42)
    assert out['n_samples'] == 4
    assert out['n_batches'] == 2
    assert out['val_loss'] == pytest.approx(1.0)
    assert out['components']['mse_tir'] == pytest.approx(1.0)
    assert out['kept']['kept_tir'] == pytest.approx(50.0)


def test_ddp_validation_sums_every_ranks_slice(monkeypatch):
    remote = {
        'loss_sum': 4.0, 'n_samples': 2, 'n_batches': 1,
        'components': {'mse_tir': 6.0}, 'kept': {'kept_tir': 20.0},
        'mode': {'mode2': 2.0},          # disjoint key: contributes on its own
    }

    def fake_gather(out_list, payload):
        out_list[0] = payload
        out_list[1] = remote

    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'all_gather_object', fake_gather)

    out = evaluate_pretrain(_StubModel(1.0), _loader(), torch.device('cpu'),
                            mask_base=42)

    # local slice: 4 samples at loss 1.0; remote: 2 samples summing to 4.0
    assert out['n_samples'] == 6
    assert out['n_batches'] == 3
    assert out['val_loss'] == pytest.approx(8.0 / 6.0)
    assert out['components']['mse_tir'] == pytest.approx(10.0 / 6.0)
    assert out['kept']['kept_tir'] == pytest.approx(220.0 / 6.0)
    assert out['mode']['mode1'] == pytest.approx(4.0 / 6.0)
    assert out['mode']['mode2'] == pytest.approx(2.0 / 6.0)


def test_gather_is_skipped_when_not_distributed(monkeypatch):
    called = []
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.distributed, 'all_gather_object',
                        lambda *a, **k: called.append(1))

    out = evaluate_pretrain(_StubModel(2.0), _loader(), torch.device('cpu'),
                            mask_base=42)
    assert called == []
    assert out['n_samples'] == 4
    assert out['val_loss'] == pytest.approx(2.0)
