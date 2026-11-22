"""W&B loss logging: per-step train loss + per-epoch validation loss.

The four-stream arm runs 150 epochs, so a per-EPOCH aggregate (150 points) is
too coarse to diagnose a divergence -- `train/loss` is written every optimizer
step by the engine through the generic ``log_writer`` sink, and `val/loss` is
written on every eval epoch. Both must share ONE step axis (the global optimizer
step), otherwise the backend rejects the later points as non-monotonic.
"""
import os
import sys

import pytest
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from engines.pretrain import train_one_epoch
from utils.wandb_logger import WandbLogger


class _RecordingWriter:
    """Stands in for the W&B logger; records (tag, value, step)."""

    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))


class _StubModel(torch.nn.Module):
    """Returns the exact dict shape the multimodal engine consumes."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch, counts_generator=None, placement_generator=None):
        assert isinstance(batch, dict) and 'tir' in batch
        assert counts_generator is not None and placement_generator is not None
        return {'loss': self.w + 1.0, 'losses_mse': {'tir': 1.0},
                'kept': {'tir': 50.0}, 'mode': 1.0, 'n_batch': 350.0}


def _batches(n):
    return [{'tir': torch.zeros(1, 3, 4, 8, 8)} for _ in range(n)]


def test_engine_writes_one_train_loss_per_optimizer_step():
    writer = _RecordingWriter()
    model = _StubModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    train_one_epoch(
        model=model, data_loader=_batches(3), optimizer=optimizer,
        device=torch.device('cpu'), epoch=1, loss_scaler=None,
        log_writer=writer, start_steps=5, steps_per_epoch=3, rank=0)

    assert [c[0] for c in writer.calls] == ['train/loss'] * 3
    # The step axis is the GLOBAL optimizer step, so epoch 1 resumes at 5.
    assert [c[2] for c in writer.calls] == [5, 6, 7]
    assert all(c[1] == pytest.approx(1.0) for c in writer.calls)


def test_engine_without_a_writer_still_runs():
    model = _StubModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    stats = train_one_epoch(
        model=model, data_loader=_batches(2), optimizer=optimizer,
        device=torch.device('cpu'), epoch=0, loss_scaler=None,
        log_writer=None, start_steps=0, steps_per_epoch=2, rank=0)
    assert stats['loss'] == pytest.approx(1.0)


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, metrics, step=None):
        self.calls.append((dict(metrics), step))


def test_add_scalar_delegates_to_the_run():
    run = _FakeRun()
    WandbLogger(run).add_scalar('train/loss', 1.5, 3)
    assert run.calls == [({'train/loss': 1.5}, 3)]


def test_disabled_logger_is_a_noop():
    logger = WandbLogger(None)
    assert not logger.enabled
    logger.add_scalar('train/loss', 1.5, 3)      # must not raise
    logger.log({'val/loss': 1.0}, step=1)
    logger.finish()


def test_fourstream_config_enables_wandb_and_the_runner_wires_it():
    import yaml
    cfg_path = os.path.join(
        CODE_DIR, 'configs', 'pretrain',
        'stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml')
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg['log_wandb'] is True
    assert cfg['wandb_project']

    src = open(os.path.join(CODE_DIR, 'runners', 'run_pretrain.py')).read()
    assert 'log_writer=wandb_logger' in src
    assert "wandb_logger.log({'val/loss'" in src
    engine = open(os.path.join(CODE_DIR, 'engines', 'pretrain.py')).read()
    assert "add_scalar('train/loss'" in engine
