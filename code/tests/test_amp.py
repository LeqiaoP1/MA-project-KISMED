"""BDD acceptance criteria for the AMP switch (analysis/hpc_AMP.md).

CPU-only: no CUDA is required. Cases that need a CUDA *decision* monkeypatch
``torch.cuda.is_available`` and pass a ``cuda`` device, so only the POLICY is
asserted (autocast/scaler on or off), never a real kernel.

Scenario map (see the "BDD Acceptance Criteria" section of hpc_AMP.md):
  1. Default (--amp_dtype none) preserves FP32 execution.
  2. fp16 enables autocast AND loss scaling.
  3. bf16 enables autocast WITHOUT loss scaling.
  4. Checkpoint resumption warns on a dtype mismatch.
  5. DDP gradient accumulation + plumbing (no_sync / runner wiring).
"""
import argparse
import os
import sys

import pytest
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils import (AMP_DTYPES, NativeScalerWithGradNormCount, add_amp_args,
                   autocast_context, resolve_amp, scaler_enabled)
from utils.checkpoint import load_model, save_model

CUDA = torch.device('cuda')


@pytest.fixture
def fake_cuda(monkeypatch):
    """Make ``resolve_amp`` believe a CUDA device is active (no real GPU)."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    return CUDA


# --------------------------------------------------------------------------- #
# Scenario 1: default behaviour preserves FP32 execution
# --------------------------------------------------------------------------- #
def test_default_is_fp32_autocast_off_and_scaler_bypassed():
    cfg = resolve_amp('none', torch.device('cpu'))
    assert cfg.mode == 'none'
    assert not cfg.enabled and not cfg.scaler
    assert scaler_enabled(cfg) is False
    assert cfg.describe().startswith('amp_dtype=none')


def test_explicit_none_disables_autocast_even_on_cuda(fake_cuda):
    cfg = resolve_amp('none', fake_cuda)
    assert not cfg.enabled and not cfg.scaler
    with autocast_context(cfg):
        assert not torch.is_autocast_enabled()


def test_legacy_none_amp_config_keeps_the_scaler():
    """A caller that passes ONLY a loss scaler (no amp_config) keeps scaling."""
    assert scaler_enabled(None) is True


# --------------------------------------------------------------------------- #
# Scenario 2: fp16 enables autocast and loss scaling
# --------------------------------------------------------------------------- #
def test_fp16_enables_autocast_and_loss_scaling(fake_cuda):
    cfg = resolve_amp('fp16', fake_cuda)
    assert cfg.enabled and cfg.scaler
    assert cfg.dtype is torch.float16
    assert scaler_enabled(cfg) is True


# --------------------------------------------------------------------------- #
# Scenario 3: bf16 enables autocast without loss scaling
# --------------------------------------------------------------------------- #
def test_bf16_enables_autocast_without_loss_scaling(fake_cuda):
    cfg = resolve_amp('bf16', fake_cuda)
    assert cfg.enabled and not cfg.scaler
    assert cfg.dtype is torch.bfloat16
    assert scaler_enabled(cfg) is False


def test_amp_degrades_to_none_on_cpu(capsys):
    cfg = resolve_amp('fp16', torch.device('cpu'))
    assert not cfg.enabled and not cfg.scaler
    assert 'disabling AMP' in capsys.readouterr().out


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_amp('int8', CUDA)


# --------------------------------------------------------------------------- #
# Scaler bypass: still measures the grad norm, clips and steps
# --------------------------------------------------------------------------- #
def test_scaler_bypass_still_steps_and_returns_grad_norm():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = NativeScalerWithGradNormCount()
    before = model.weight.detach().clone()
    loss = model(torch.randn(3, 4)).sum()
    norm = scaler(loss, opt, clip_grad=None, parameters=model.parameters(),
                  update_grad=True, scaler_enabled=False)
    assert norm is not None and float(norm) >= 0.0
    assert not torch.equal(before, model.weight.detach())      # stepped


def test_scaler_bypass_accumulates_gradients_between_steps():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)          # frozen weights
    scaler = NativeScalerWithGradNormCount()
    for _ in range(2):
        scaler(model(torch.randn(3, 4)).sum(), opt,
               parameters=model.parameters(), update_grad=False,
               scaler_enabled=False)
    assert model.weight.grad is not None


# --------------------------------------------------------------------------- #
# Scenario 4: checkpoint metadata + dtype-mismatch warning
# --------------------------------------------------------------------------- #
class _Args:
    def __init__(self, out, amp='none', resume=''):
        self.output_dir = out
        self.amp_dtype = amp
        self.resume = resume


def test_checkpoint_records_amp_dtype_and_warns_on_mismatch(tmp_path, capsys):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = NativeScalerWithGradNormCount()
    save_model(_Args(str(tmp_path), amp='none'), 0, model, model, opt, scaler)

    ckpt = os.path.join(str(tmp_path), 'checkpoints', 'checkpoint-0000.pth')
    assert torch.load(ckpt, map_location='cpu',
                      weights_only=False)['amp_dtype'] == 'none'

    # resume in a DIFFERENT mode -> a visible warning, training proceeds
    load_model(_Args(str(tmp_path), amp='fp16', resume=ckpt), model, opt, scaler)
    out = capsys.readouterr().out
    assert 'amp_dtype' in out and 'WARNING' in out


def test_old_checkpoint_without_amp_metadata_still_resumes(tmp_path, capsys):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    (tmp_path / 'checkpoints').mkdir()
    ckpt = str(tmp_path / 'checkpoints' / 'checkpoint-0000.pth')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'epoch': 0}, ckpt)                              # no 'amp_dtype'
    load_model(_Args(str(tmp_path), amp='bf16', resume=ckpt), model, opt,
               NativeScalerWithGradNormCount())
    assert 'WARNING' not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Scenario 5: plumbing -- every training runner exposes and forwards the switch
# --------------------------------------------------------------------------- #
TRAIN_RUNNERS = ('run_pretrain.py', 'run_finetune.py', 'run_waveform.py',
                 'run_au_probe.py')


def test_common_args_expose_the_switch():
    with open(os.path.join(CODE_DIR, 'runners', '_common.py')) as fh:
        assert 'add_amp_args' in fh.read()


@pytest.mark.parametrize('runner', TRAIN_RUNNERS)
def test_every_training_runner_resolves_and_passes_amp(runner):
    with open(os.path.join(CODE_DIR, 'runners', runner)) as fh:
        src = fh.read()
    assert 'add_common_args' in src, f'{runner} does not use the common args'
    assert 'resolve_amp' in src and 'amp_config' in src, \
        f'{runner} does not resolve/pass the AMP policy'


def test_waveform_runner_uses_the_fp32_loss():
    """The numerically sensitive waveform criteria must stay in FP32."""
    with open(os.path.join(CODE_DIR, 'runners', 'run_waveform.py')) as fh:
        assert 'loss_in_fp32=True' in fh.read()


def test_add_amp_args_defaults_to_none_and_lists_three_modes():
    p = argparse.ArgumentParser()
    add_amp_args(p)
    assert p.parse_args([]).amp_dtype == 'none'
    for mode in AMP_DTYPES:
        assert p.parse_args(['--amp_dtype', mode]).amp_dtype == mode
    with pytest.raises(SystemExit):                 # argparse rejects the rest
        p.parse_args(['--amp_dtype', 'int8'])
