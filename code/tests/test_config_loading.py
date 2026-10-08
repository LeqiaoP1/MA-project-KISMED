"""Integration tests: the REAL runners against the SHIPPED configs.

Two things are guarded here:

* the unknown-key guard must ACCEPT every config in ``configs/pretrain/`` and
  ``configs/finetune/`` -- a failure means the guard would break a real
  experiment, not just a synthetic case, and
* the resolved values of the HPC TIR-ROI pair must still satisfy the geometry
  contract that ``run_waveform.py`` enforces between Stage 2 and Stage 3. A
  silent drift there is expensive: Stage 3 aborts only after the multi-hour
  Stage-2 run has finished.

Configs are runner-SPECIFIC. ``configs/finetune/example.yaml`` is the
classification template (it uses ``nb_classes`` / ``smoothing``), so it is
loaded with ``run_finetune`` -- the same file is deliberately rejected by
``run_waveform``, which is asserted below.
"""
import glob
import importlib
import os
import sys

import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGE2_HPC = 'configs/pretrain/stage2_hpc_tir_roi_resp.yaml'
STAGE3_HPC = 'configs/finetune/resp_tir_roi_hpc.yaml'

SHIPPED_CONFIGS = sorted(
    path
    for folder in ('configs/pretrain', 'configs/finetune')
    for path in glob.glob(os.path.join(CODE_DIR, folder, '*.yaml'))
)


def _runner_for(cfg_path: str) -> str:
    """Map a config file to the runner it belongs to."""
    rel = os.path.relpath(cfg_path, CODE_DIR).replace(os.sep, '/')
    base = os.path.basename(cfg_path)
    if rel.startswith('configs/pretrain/'):
        return 'runners.run_pretrain'
    if base.startswith('au_'):
        return 'runners.run_au_probe'
    if base == 'example.yaml':              # the CLASSIFICATION template
        return 'runners.run_finetune'
    return 'runners.run_waveform'


def _args_for(cfg_path: str, monkeypatch):
    """Parse ``cfg_path`` with its runner's real argument parser."""
    monkeypatch.setattr(sys, 'argv', ['prog', '-c', cfg_path])
    return importlib.import_module(_runner_for(cfg_path)).get_args()


def _stage2(monkeypatch):
    return _args_for(os.path.join(CODE_DIR, STAGE2_HPC), monkeypatch)


def _stage3(monkeypatch):
    return _args_for(os.path.join(CODE_DIR, STAGE3_HPC), monkeypatch)


# --------------------------------------------------------------------------- #
# regression: the guard must not break any shipped config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('cfg', SHIPPED_CONFIGS,
                         ids=lambda p: os.path.relpath(p, CODE_DIR))
def test_shipped_config_is_accepted(cfg, monkeypatch):
    """Every config in the repo still loads through its own runner."""
    assert _args_for(cfg, monkeypatch) is not None


def test_config_inventory_is_not_empty():
    """Guard against the parametrisation silently collapsing to nothing."""
    assert len(SHIPPED_CONFIGS) >= 20
    assert any(p.endswith('stage2_hpc_tir_roi_resp.yaml') for p in SHIPPED_CONFIGS)
    assert any(p.endswith('resp_tir_roi_hpc.yaml') for p in SHIPPED_CONFIGS)


PRETRAIN_CONFIGS = sorted(
    path for path in SHIPPED_CONFIGS
    if os.path.relpath(path, CODE_DIR).startswith('configs/pretrain/')
)


def test_every_pretrain_config_declares_a_videomae_init(monkeypatch):
    """Stage 1 has no blank path: every shipped Stage-2 config must name a
    VideoMAE corpus (never '' / random / scratch / mae / timm / large)."""
    assert PRETRAIN_CONFIGS
    for cfg in PRETRAIN_CONFIGS:
        a = _args_for(cfg, monkeypatch)
        spec = a.pretrained_encoder
        rel = os.path.relpath(cfg, CODE_DIR)
        assert spec and spec not in ('none', 'random', 'scratch', 'c0'), (
            f'{rel}: blank Stage-1 init {spec!r}')
        assert not spec.startswith(('mae:', 'timm:')) and spec != 'large', (
            f'{rel}: removed Stage-1 source {spec!r}')
        assert spec.startswith(('videomae', 'base')), (
            f'{rel}: not a VideoMAE corpus spec {spec!r}')


def test_ssv2_twin_pins_the_ssv2_corpus(monkeypatch):
    cfg = os.path.join(CODE_DIR,
                       'configs/pretrain/stage2_local_pretrained_ssv2.yaml')
    assert _args_for(cfg, monkeypatch).pretrained_encoder == 'videomae:ssv2'


# --------------------------------------------------------------------------- #
# the HPC TIR-ROI pair: resolved values + the cross-stage contract
# --------------------------------------------------------------------------- #
def test_stage2_hpc_resolves_the_intended_values(monkeypatch):
    a = _stage2(monkeypatch)
    assert a.data_set == 'tir_roi'
    assert a.streams == 'tir,resp'
    assert a.input_size == 112
    assert a.roi_padding == pytest.approx(0.2)
    assert a.min_signal_spread == pytest.approx(0.1)
    assert a.rail_touch_v == pytest.approx(9.90)
    assert a.clip_duration == pytest.approx(8.0)
    assert a.clip_stride == pytest.approx(2.0)
    assert a.sig_kernel == 16
    assert a.tubelet == '2,16,16'
    assert a.physio_norm == 'session'       # the ONLY target-norm knob
    assert a.pretrained_encoder == 'videomae:k400'
    # large-batch MAE convention: blr is set and lr stays None so it is DERIVED
    # (setting a YAML `lr` would silently switch the runner to an absolute LR).
    assert a.lr is None
    assert a.blr == pytest.approx(1.5e-4)


def test_mask_ratios_differ_between_visual_and_physio(monkeypatch):
    """The shipped pair relies on asymmetric masking (visual 0.90 / resp 0.50).

    Dropping ``mask_ratio_resp: 0.50`` from the YAML would silently fall back to
    the argparse default of 0.90 -- the documented resp-collapse mode.
    """
    a = _stage2(monkeypatch)
    assert a.mask_ratio_tir == pytest.approx(0.90)
    assert a.mask_ratio_resp == pytest.approx(0.50)


def test_stage2_and_stage3_agree_on_the_roi_contract(monkeypatch):
    """Stage 3 ABORTS on any mismatch of these keys vs the Stage-2 checkpoint."""
    from runners.run_waveform import ROI_CONTRACT_KEYS

    s2, s3 = _stage2(monkeypatch), _stage3(monkeypatch)
    for key in ROI_CONTRACT_KEYS + ('clip_duration', 'fps', 'tubelet',
                                    'temporal_stride', 'sig_kernel', 'fs',
                                    'task_set'):
        assert getattr(s2, key) == getattr(s3, key), (
            f'{key}: stage2={getattr(s2, key)!r} != stage3={getattr(s3, key)!r}')
    # seq_len == 0 means "derive the window length": both must derive the same.
    assert (s2.seq_len or int(round(s2.clip_duration * s2.fs))) == \
           (s3.seq_len or int(round(s3.clip_duration * s3.fs)))


def test_stage3_hpc_points_at_the_stage2_checkpoint_dir(monkeypatch):
    """The default `finetune` must live under the Stage-2 run's output dir."""
    a = _stage3(monkeypatch)
    assert a.data_set == 'tir_roi_resp'
    assert a.target == 'resp'
    assert 'stage2_hpc_tir_roi_resp' in a.finetune
    assert a.finetune.endswith('.pth')
    # tir_roi reads the RAW tree; data_path is supplied by the job script.
    assert a.max_clips == 0


# --------------------------------------------------------------------------- #
# end-to-end behaviour through a real runner
# --------------------------------------------------------------------------- #
def test_typo_is_rejected_by_a_real_runner(tmp_path, monkeypatch):
    cfg = tmp_path / 'typo.yaml'
    cfg.write_text('input_size: 112\nroi_paddingg: 0.2\n')
    monkeypatch.setattr(sys, 'argv', ['prog', '-c', str(cfg)])
    with pytest.raises(SystemExit) as exc:
        importlib.import_module('runners.run_pretrain').get_args()
    msg = str(exc.value)
    assert 'roi_paddingg' in msg
    assert "did you mean 'roi_padding'" in msg


def test_cli_still_overrides_a_shipped_config(monkeypatch):
    monkeypatch.setattr(
        sys, 'argv',
        ['prog', '-c', os.path.join(CODE_DIR, STAGE2_HPC),
         '--input_size', '224'])
    assert importlib.import_module('runners.run_pretrain').get_args().input_size == 224


def test_waveform_rejects_the_classification_template(monkeypatch):
    """Configs are runner-specific -- and that is now enforced, not ignored."""
    cfg = os.path.join(CODE_DIR, 'configs/finetune/example.yaml')
    monkeypatch.setattr(sys, 'argv', ['prog', '-c', cfg])
    with pytest.raises(SystemExit) as exc:
        importlib.import_module('runners.run_waveform').get_args()
    msg = str(exc.value)
    assert 'nb_classes' in msg
    assert 'smoothing' in msg
