"""``utils.wandb_logger``: the consumer of ``--log_wandb``.

The flag was parsed but never used until 2026-11-08, so these tests pin the
contract that makes it safe in a multi-hour job:

* disabled => a complete no-op that does not even IMPORT wandb,
* enabled  => one rank-0 run whose config carries the args,
* any failure (missing package, failing ``init``) => logging disabled, never an
  exception, and
* inside a Slurm job the default mode is ``offline`` (compute nodes have no
  internet) while a local run defaults to ``online``.
"""
import argparse
import os
import sys
import types

import pytest

from utils.wandb_logger import WandbLogger


class _FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, data, step=None):
        self.logged.append((dict(data), step))

    def finish(self):
        self.finished = True


def _args(tmp_path, **over):
    kw = dict(log_wandb=True, wandb_project='thesis-project',
              output_dir=str(tmp_path / 'stage2_ammm_fold3'),
              data_set='tir_roi', masking='ammm', n_folds=5, fold=3,
              epochs=150, input_size=96, lr=3.75e-5)
    kw.update(over)
    return argparse.Namespace(**kw)


@pytest.fixture
def fake_wandb(monkeypatch):
    """Install a stub ``wandb`` module and record what init received."""
    calls = {}
    module = types.ModuleType('wandb')

    def init(**kwargs):
        calls['init'] = kwargs
        calls['run'] = _FakeRun()
        return calls['run']

    module.init = init
    monkeypatch.setitem(sys.modules, 'wandb', module)
    # keep the environment deterministic: no ambient offline/online/dir/id
    for var in ('WANDB_MODE', 'WANDB_DIR', 'WANDB_RUN_ID', 'SLURM_JOB_ID'):
        monkeypatch.delenv(var, raising=False)
    return calls


# --------------------------------------------------------------------------- #
# disabled paths
# --------------------------------------------------------------------------- #
def test_disabled_is_a_noop_and_does_not_import_wandb(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, 'wandb', raising=False)
    log = WandbLogger.from_args(_args(tmp_path, log_wandb=False))
    assert log.enabled is False
    log.log({'loss': 1.0}, step=0)          # must not raise
    log.finish()
    assert 'wandb' not in sys.modules       # lazy import really is lazy


def test_missing_package_disables_instead_of_crashing(tmp_path, monkeypatch,
                                                      capsys):
    # ``sys.modules['wandb'] = None`` makes ``import wandb`` raise ImportError
    monkeypatch.setitem(sys.modules, 'wandb', None)
    log = WandbLogger.from_args(_args(tmp_path))
    assert log.enabled is False
    assert 'not' in capsys.readouterr().out.lower()


def test_a_failing_init_never_kills_the_run(tmp_path, monkeypatch, capsys):
    module = types.ModuleType('wandb')

    def boom(**kwargs):
        raise RuntimeError('no internet')

    module.init = boom
    monkeypatch.setitem(sys.modules, 'wandb', module)
    log = WandbLogger.from_args(_args(tmp_path))
    assert log.enabled is False
    assert 'init failed' in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# enabled path
# --------------------------------------------------------------------------- #
def test_enabled_builds_one_run_with_the_args_as_config(tmp_path, fake_wandb):
    log = WandbLogger.from_args(_args(tmp_path), job_type='pretrain',
                                tags=('tir_roi', 'ammm'),
                                extra={'data/steps_per_epoch': 255})
    assert log.enabled is True
    init = fake_wandb['init']
    assert init['project'] == 'thesis-project'
    assert init['name'] == 'stage2_ammm_fold3'        # per-fold run name
    assert init['job_type'] == 'pretrain'
    assert init['tags'] == ['tir_roi', 'ammm']
    assert init['config']['masking'] == 'ammm'
    assert init['config']['input_size'] == 96
    assert init['config']['data/steps_per_epoch'] == 255
    # wandb adds its own ``wandb/`` level, so dir = the RUN dir (not run/wandb)
    assert init['dir'] == str(tmp_path / 'stage2_ammm_fold3')


def test_offline_is_the_default_inside_a_slurm_job(tmp_path, fake_wandb,
                                                   monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    log = WandbLogger.from_args(_args(tmp_path))
    assert log.mode == 'offline'                       # no egress on compute nodes
    assert os.environ['WANDB_MODE'] == 'offline'


def test_online_is_the_default_outside_slurm(tmp_path, fake_wandb):
    log = WandbLogger.from_args(_args(tmp_path))
    assert log.mode == 'online'


def test_mode_env_var_is_respected(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv('WANDB_MODE', 'online')
    monkeypatch.setenv('SLURM_JOB_ID', '12345')       # would otherwise force offline
    log = WandbLogger.from_args(_args(tmp_path))
    assert log.mode == 'online'


def test_log_forwards_scalars_and_drops_non_finite(tmp_path, fake_wandb):
    log = WandbLogger.from_args(_args(tmp_path))
    log.log({'train/loss': 0.0912, 'train/mse_resp': float('nan'),
             'train/kept_tir': 4, 'junk': object()}, step=7)
    run = fake_wandb['run']
    data, step = run.logged[-1]
    assert data == {'train/loss': 0.0912, 'train/kept_tir': 4.0}
    assert step == 7


def test_finish_closes_once(tmp_path, fake_wandb):
    log = WandbLogger.from_args(_args(tmp_path))
    run = fake_wandb['run']
    log.finish()
    assert run.finished is True
    log.finish()                               # idempotent
    log.log({'x': 1.0})                        # no-op after finish


def test_resume_uses_wandb_run_id(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv('WANDB_RUN_ID', 'abc123')
    WandbLogger.from_args(_args(tmp_path))
    init = fake_wandb['init']
    assert init['id'] == 'abc123'
    assert init['resume'] == 'allow'
