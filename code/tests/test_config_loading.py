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


def test_stage3_hpc_ships_a_usable_huber_beta(monkeypatch):
    """``L_time`` is a point-wise Huber with ONE knob: ``huber_beta``.

    There is no term selector any more, so what is worth guarding is that the
    shipped config still supplies a valid transition point and that the removed
    ``--time_loss`` flag did not creep back into the runner.
    """
    a = _stage3(monkeypatch)
    assert a.huber_beta > 0
    assert not hasattr(a, 'time_loss')


PRETRAIN_CONFIGS = sorted(
    path for path in SHIPPED_CONFIGS
    if os.path.relpath(path, CODE_DIR).startswith('configs/pretrain/')
)


#: Stage-2 options that were DELETED with the MR-STFT term (2026-11-09).
#: Neither the runner parser nor any shipped pretrain YAML may mention them: an
#: unknown YAML key is a hard error, so a stale key breaks the run at startup.
REMOVED_SPECTRAL_KEYS = ('spectral_weight', 'spectral_weights',
                         'spectral_fft_sizes', 'spectral_hop_ratio')


def test_stage2_carries_no_spectral_keys(monkeypatch):
    """The Stage-2 MR-STFT term is gone from the runner AND from the configs."""
    a = _stage2(monkeypatch)
    for name in REMOVED_SPECTRAL_KEYS:
        assert not hasattr(a, name), f'run_pretrain still defines --{name}'
    assert PRETRAIN_CONFIGS
    for cfg in PRETRAIN_CONFIGS:
        rel = os.path.relpath(cfg, CODE_DIR)
        args = _args_for(cfg, monkeypatch)
        for name in REMOVED_SPECTRAL_KEYS:
            assert not hasattr(args, name), f'{rel}: stale {name}'


#: Stage-2 options DELETED with the EDA modality (2026-11-18). A stale YAML key
#: is a hard error in ``parse_args_with_config``, so the runner must not define
#: it AND no shipped pretrain config may mention it.
REMOVED_EDA_KEYS = ('mask_ratio_eda',)


def test_stage2_carries_no_eda_keys(monkeypatch):
    """The EDA modality is gone from the Stage-2 runner AND from the configs."""
    a = _stage2(monkeypatch)
    for name in REMOVED_EDA_KEYS:
        assert not hasattr(a, name), f'run_pretrain still defines --{name}'
    assert PRETRAIN_CONFIGS
    for cfg in PRETRAIN_CONFIGS:
        rel = os.path.relpath(cfg, CODE_DIR)
        args = _args_for(cfg, monkeypatch)
        for name in REMOVED_EDA_KEYS:
            assert not hasattr(args, name), f'{rel}: stale {name}'
        assert 'eda' not in [s.strip() for s in args.streams.split(',')], (
            f'{rel}: EDA is no longer a supported stream ({args.streams!r})')


#: The explicit random-init aliases accepted by ``run_pretrain``.
BLANK_INIT_SPECS = ('', 'none', 'random', 'scratch', 'c0')


def test_every_pretrain_config_declares_a_valid_init(monkeypatch):
    """Every shipped Stage-2 config declares either a VideoMAE corpus or the
    explicit random-init path. Random init is only allowed for
    ``project_multimae_small``, which has no published Stage-1 checkpoint."""
    assert PRETRAIN_CONFIGS
    for cfg in PRETRAIN_CONFIGS:
        a = _args_for(cfg, monkeypatch)
        spec = str(a.pretrained_encoder or '')
        rel = os.path.relpath(cfg, CODE_DIR)
        if spec in BLANK_INIT_SPECS:
            assert a.model == 'project_multimae_small', (
                f'{rel}: blank Stage-1 init {spec!r} is only allowed for '
                f'project_multimae_small; got model={a.model!r}')
            continue
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
    assert a.input_size == 96
    assert a.roi_padding == pytest.approx(0.1)
    assert a.min_signal_spread == pytest.approx(0.1)
    assert a.rail_touch_v == pytest.approx(9.90)
    assert a.clip_duration == pytest.approx(8.0)
    assert a.clip_stride == pytest.approx(2.0)
    assert a.sig_kernel == 16
    assert a.tubelet == '2,16,16'
    assert a.physio_norm == 'zscore'        # the ONLY label normalisation
    assert a.pretrained_encoder == 'videomae:k400'
    # large-batch MAE convention: blr is set and lr stays None so it is DERIVED
    # (setting a YAML `lr` would silently switch the runner to an absolute LR).
    assert a.lr is None
    assert a.blr == pytest.approx(1.5e-4)


def test_stage2_exposes_the_adr002_policy_surface(monkeypatch):
    """The seven ADR-002 keys are the whole masking surface.

    The superseded per-stream ratios / Dirichlet knobs must be GONE: a stale
    YAML key is a hard error, and a stale flag would be a silent no-op.
    """
    a = _stage2(monkeypatch)
    for key in ('n_budget', 'video_cap_ratio', 'physio_cap_ratio',
                'video_min_tubes', 'p_virtual_sensing', 'spatial_patches',
                'physio_tokens'):
        assert hasattr(a, key), f'run_pretrain does not define --{key}'
    for gone in ('masking', 'mask_ratio_tir', 'mask_ratio_resp',
                 'dirichlet_alpha', 'video_kept_cap', 'physio_kept_cap',
                 'modality_dropout', 'min_kept', 'kept_ratio_override',
                 'physio_mask', 'mask_span_s'):
        assert not hasattr(a, gone), f'run_pretrain still defines --{gone}'


def test_the_budget_arm_carries_the_adr_geometry(monkeypatch):
    """The no-curriculum comparator declares the ADR-005 §2.2 units explicitly.

    Its tokenizer geometry is the I1 one (``2,16,16`` at decimation 4) so the
    VideoMAE kernel loads; it omits the curriculum keys, so ``E_warm = 0`` and
    ``p(e) = p_target`` from the first step -- which is what isolates R6 when
    it is diffed against the curriculum arm.
    """
    a = _args_for(os.path.join(
        CODE_DIR, 'configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml'),
        monkeypatch)
    assert a.streams == 'tir,rgb2d,resp'
    assert a.tubelet == '2,16,16' and a.input_size == 96
    assert a.temporal_stride == 4
    assert a.virtual_sensing_warmup_epochs == 0
    assert a.sig_kernel == 32 and a.fs == pytest.approx(1000.0)
    assert a.n_budget == 350
    assert a.video_cap_ratio == pytest.approx(0.25)
    assert a.physio_cap_ratio == pytest.approx(0.50)
    assert a.video_min_tubes == 2
    assert a.p_virtual_sensing == pytest.approx(0.50)
    assert a.spatial_patches == 36 and a.physio_tokens == 250


def test_the_fourstream_arm_carries_the_requested_geometry(monkeypatch):
    """The four-stream arm is the ADR-002 geometry plus the requested 4 s hop."""
    a = _args_for(os.path.join(
        CODE_DIR,
        'configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml'),
        monkeypatch)
    assert a.data_set == 'tir_roi_resp_rgb2d_bp'
    assert a.model == 'project_multimae_small'
    assert a.pretrained_encoder == 'none'
    assert a.streams == 'tir,rgb2d,bp,resp'
    assert a.clip_duration == pytest.approx(8.0)
    assert a.clip_stride == pytest.approx(4.0)
    assert a.temporal_stride == 2
    assert a.tubelet == '4,16,16'
    assert a.input_size == 96
    assert a.fs == pytest.approx(1000.0)
    assert a.sig_kernel == 32
    assert a.bp_fs == pytest.approx(1000.0)
    assert a.min_signal_spread == pytest.approx(0.0)
    assert a.rail_touch_v == pytest.approx(0.0)
    assert a.n_budget == 350
    assert a.video_cap_ratio == pytest.approx(0.25)
    assert a.physio_cap_ratio == pytest.approx(0.50)
    assert a.video_min_tubes == 2
    assert a.spatial_patches == 36
    assert a.physio_tokens == 250


def test_stage2_and_stage3_agree_on_the_roi_contract(monkeypatch):
    """Stage 3 ABORTS on any mismatch of these keys vs the Stage-2 checkpoint."""
    from runners.run_waveform import ROI_CONTRACT_KEYS

    s2, s3 = _stage2(monkeypatch), _stage3(monkeypatch)
    for key in ROI_CONTRACT_KEYS + ('clip_duration', 'fps', 'tubelet',
                                    'temporal_stride', 'sig_kernel', 'fs',
                                    'task_set'):
        assert getattr(s2, key) == getattr(s3, key), (
            f'{key}: stage2={getattr(s2, key)!r} != stage3={getattr(s3, key)!r}')
    # ``task_groups`` reaches the namespace as a MAPPING (not a string), so it
    # is compared PARSED -- and it must be compared at all: with a non-empty
    # task_set a drifted tier definition silently evaluates Stage 3 on a
    # different corpus than the encoder was pre-trained on. This is the
    # "MUST mirror" comment finally enforced (see
    # tests/test_task_group_configs.py for every shipped config).
    from data.task_groups import parse_task_groups

    assert parse_task_groups(s2.task_groups) == parse_task_groups(s3.task_groups)
    # seq_len == 0 means "derive the window length": both must derive the same.
    assert (s2.seq_len or int(round(s2.clip_duration * s2.fs))) == \
           (s3.seq_len or int(round(s3.clip_duration * s3.fs)))


def test_stage2_and_stage3_share_the_fold_partition_rule(monkeypatch):
    """Different partition knobs = fold k trained on fold k's test subjects.

    ``check_stage2_fold_contract`` compares ``fold``/``n_folds`` only, so the
    two configs could drift to different ROLE LISTS while every guard still
    passed. These two keys decide the role lists; they must match, and the
    shipped pair must not silently fall back to the legacy contiguous blocks
    (which make fold == sex block on BP4D+).
    """
    s2, s3 = _stage2(monkeypatch), _stage3(monkeypatch)
    for key in ('split_seed', 'split_stratify'):
        assert getattr(s2, key) == getattr(s3, key), (
            f'{key}: stage2={getattr(s2, key)!r} != stage3={getattr(s3, key)!r}')
    assert s2.split_seed is not None, 'the shipped pair must pin a partition seed'
    assert s2.split_stratify == 'sex', \
        'the shipped pair must stratify: sorted blocks are a sex confound'


def test_stage3_hpc_leaves_the_encoder_to_the_sweep(monkeypatch):
    """The k-fold sweep must NEVER inherit a hardcoded encoder.

    Stage-3 fold k has to start from the Stage-2 run that never saw fold k, so
    the encoder cannot be a config constant: ``submit_sweep.sh`` picks the
    newest Stage-2 checkpoint of the SAME fold directory and passes it as
    ``--finetune``, and ``check_stage2_fold_contract`` refuses a checkpoint
    whose ``partition_digest`` differs (ADR-001 D4). A hardcoded whole-corpus
    path would not pair, i.e. it cannot report a leaked CV number that looks
    healthy.
    """
    a = _stage3(monkeypatch)
    assert a.data_set == 'tir_roi_resp'
    assert a.target == 'resp'
    assert a.finetune == ''          # deliberately empty -> the sweep decides
    # tir_roi reads the RAW tree; data_path is supplied by the sweep.
    assert a.max_clips == 0

    # ... and the sweep must actually pair fold-for-fold through the record,
    # otherwise an empty `finetune` would degrade into "train from random
    # weights".
    sweep = os.path.join(CODE_DIR, 'scripts', 'hpc', 'submit_sweep.sh')
    with open(sweep) as fh:
        script = fh.read()
    assert '--finetune "$s2"' in script
    assert '--exp_root "$EXP_ROOT" --fold "$k"' in script
    assert '--eval_role test' in script, (
        'the sweep must score the test role in its own process (I5)')


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
