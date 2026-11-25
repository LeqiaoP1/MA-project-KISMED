# 04 — Configs and Execution

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/configs/`, `code/scripts/`, `code/runners/` on 2026-10-04.

---

## 1. Config System — `runners/_common.py::parse_args_with_config`

YAML files supply **argparse defaults only**. Explicit CLI flags always override YAML values.

```python
config_parser.add_argument('-c', '--config', ...)
cfg = yaml.safe_load(open(cfg_args.config))
parser.set_defaults(**cfg)   # YAML -> argparse defaults
return parser.parse_args(remaining)  # CLI -> final override
```

**Priority** (highest to lowest): explicit CLI flag > YAML value > argparse hardcoded default.

Invocation pattern:

```bash
# From code/:
python runners/run_pretrain.py -c configs/pretrain/stage2_local.yaml [--override_flag value ...]
```

---

## 2. Config File Inventory

### Stage-2 Pretrain (`configs/pretrain/`)

| File | Purpose |
|------|---------|
| `stage2_local.yaml` | Local smoke, from scratch, `streams: rgb,bp`, 2 sessions, 8 clips |
| `stage2_local_scratch.yaml` | Local wiring proof with extended rationale comments |
| `stage2_local_pretrained.yaml` | Local with VideoMAE init, 11 sessions, 32 clips |
| `stage2_local_rgb_roi_bp.yaml` | RGB face-ROI + BP, raw tree |
| `stage2_local_tir_roi_resp.yaml` | TIR-ROI + RESP, raw tree |
| `stage2_budget_tir_roi_resp_rgb2d.yaml` | TIR-ROI + `rgb2d` + RESP (`streams: tir,rgb2d,resp`) — the no-curriculum comparator at the ADR-005 geometry, HPC |
| `stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml` | TIR-ROI + `rgb2d` + BP + RESP, ViT-Small random init — the four-stream ADR-002 arm |
| `stage2_multimodal.yaml` | HPC full-data template, 800 epochs, input_size 224 |
| `example.yaml` | Annotated reference template |

### Stage-3 Finetune (`configs/finetune/`)

| File | Target | Data |
|------|--------|------|
| `bp_local.yaml` | BP | canonical, RGB-only, 64 px, subject-split |
| `bp.yaml` | BP | canonical, HPC |
| `bp_rgb_roi_local.yaml` | BP | RGB ROI, raw tree |
| `bp_rgb_roi_phaseprobe_local.yaml` | BP | RGB ROI, phase-probe ablation |
| `resp_local.yaml` | RESP | canonical, RGB-only |
| `resp.yaml` | RESP | canonical, HPC |
| `resp_tir_roi_local.yaml` | RESP | TIR ROI, 8 s clip, stride 2, fft 128,256,512 |
| `resp_tir_roi_local_matched.yaml` | RESP | TIR ROI, Stage-3 recipe from the local shared-encoder model (`transposed_conv`, `head_init: none`, `layer_decay 0.75`) |
| `resp.yaml` | RESP | canonical, HPC |
| `au_local.yaml` | AU probe | canonical, multi-label BCE |
| `au_local_pretrained.yaml` | AU probe | canonical, with Stage-2 encoder init |
| `example.yaml` | Annotated reference template |

---

## 3. Key YAML Parameters

### 3.1 Common to all stages

| Key | Type | Description |
|-----|------|-------------|
| `model` | str | Registered model name, e.g. `project_multimae_base` |
| `device` | str | `'cuda'` (default; falls back to CPU silently if unavailable) |
| `seed` | int | Base random seed; each DDP rank gets `seed + rank` |
| `output_dir` | str | Root for checkpoints + logs; default `$OUTPUT_DIR` or `<repo>/output` |
| `resume` | str | Checkpoint path to resume from (`$RESUME` or `''`) |
| `num_workers` | int | DataLoader workers; default `$NUM_WORKERS` or 8 |
| `pin_mem` | bool | DataLoader pin_memory; default True |
| `tir_channels` | int | TIR channels: 3 (false-colour, default) or 1 (legacy luma) |
| `log_wandb` | bool | Enable Weights & Biases logging |
| `wandb_project` | str | W&B project name; default `'thesis-project'` |
| `max_sessions` / `max_clips` / `max_entries` | int | Dev/smoke caps |

### 3.2 Stage-2 pre-training

| Key | Type | Description |
|-----|------|-------------|
| `streams` | str | Comma-separated: e.g. `rgb,tir,bp,resp` |
| `tubelet` | str | `t,ph,pw`: e.g. `2,16,16` |
| `input_size` | int | Square frame px: 64 local / 96 HPC ROI / 224 full-frame |
| `fps` / `fs` | float | Video fps / signal sample rate Hz |
| `clip_duration` / `clip_stride` | float | Clip length s / window hop s (`clip_stride` is NOT part of the Stage-3 ROI contract) |
| `temporal_stride` | int | Frame decimation inside window (1 = every frame) |
| `seq_len` | int | Signal samples; 0 = `clip_duration * fs` |
| `pos_init` | str | `sincos3d` or `random` |
| `n_budget` / `video_cap_ratio` / `physio_cap_ratio` | int / float / float | ADR-002 constant-budget masking: visible tokens per step and per-stream caps |
| `video_min_tubes` / `p_virtual_sensing` / `physio_placement` / `physio_span_tokens` | int / float / str / int | ADR-002 floor, Mode-1 probability, 1-D placement and span length |
| `spatial_patches` / `physio_tokens` | int | Tokenizer-derived unit counts; passed only to ASSERT identity |
| `signal_weight` | float | Loss weight for every physio stream (visual streams stay 1.0). This knob IS ADR-008's `lambda_resp` (arm value `5.0`) |
| `resp_loss_mse_weight` / `resp_loss_pearson_weight` / `resp_loss_spectral_weight` | float | ADR-008 `alpha` / `beta` / `gamma` of the composite respiration objective; neutral defaults `1.0` / `0.0` / `0.0` build NO extra module |
| `spectral_f_min` / `spectral_f_max` / `spectral_n_fft` | float / int | ADR-008 respiratory passband (Hz) and rFFT length (`0` = auto, `4*2**ceil(log2 L)`); required `0 < f_min < f_max <= fs/2` and a non-empty bin set |
| `align_weight` / `align_proj_dim` / `align_temperature` / `align_core_weight` / `align_gather_ddp` | float / int / float / float / bool | ADR-008 symmetric InfoNCE: weight (`0.0` = no heads), projection width (>= 16), cosine temperature, `A_core` pooling weight, cross-rank negative gather (training only) |
| `grad_balance_freq` | int | ADR-008 §2.7 diagnostic cadence for `grad_ratio_resp_to_tir` (`0` = OFF, zero extra autograd). A LOGGING cadence, recorded under `run` in `masking.json`, not part of the resume-guarded objective block |
| `target_norm` | str | `token` (per-token z-score), `clip` (per-clip z-score) or `none` (IDENTITY). **NON-ROI paths only**: the ROI lineages (`tir_roi`, `rgb_roi`, `tir_roi_resp_rgb2d`, `tir_roi_resp_rgb2d_bp`) are pinned to `none` because the dataset owns the z-score |
| `physio_norm` (Stage 2) | str | `clip` (= `zscore`) per-clip z-score for every ROI lineage, including the four-stream arm. `none` / `ac` / `session` are refused |
| `pretrained_encoder` | str | `videomae:k400` / `videomae:ssv2` / `videomae` / `base` / a local `.pth`; the blank aliases `none` / `random` / `scratch` / `c0` are the explicit random-init path (required for `project_multimae_small`) |
| `roi_cache` / `resp_cache` / `rgb2d_cache` / `bp_artefact` | str | Offline artefact OUT_ROOTs for the ROI lineages; each is matched by its full frozen key |
| `rgb2d_store` / `rgb2d_landmarks` | str | `native` or a pixel store (`112`); `face` or a 49-point preset |
| `bp_fs` / `bp_cache_lru` | float / int | BP artefact rate (must equal `fs`) and shard LRU |
| `epochs` / `batch_size` / `update_freq` / `save_ckpt_freq` | int | Training schedule |
| `opt` / `lr` / `blr` / `min_lr` / `warmup_epochs` | str / float | Optimizer + LR |
| `weight_decay` / `clip_grad` | float | Regularisation + grad clip (0.0 = OFF) |
| `data_path` / `data_set` | str | Dataset root / type selector |

### 3.3 Stage-3 fine-tuning (additional keys)

| Key | Type | Description |
|-----|------|-------------|
| `target` | str | `bp` / `resp` |
| `finetune` | str | Path to Stage-2 checkpoint |
| `sig_kernel` | int | Signal window size per token; must satisfy `sig_kernel/fs == tubelet_t*stride/fps` |
| `physio_norm` | str | `clip` (= `zscore`) per-clip z-score — the SAME setting, name and values as the ROI Stage-2 paths, so one clip yields one identical target in both stages. `none` / `ac` / `session` were removed and are REFUSED |
| `head_hidden` / `head_style` / `head_init` | int / str | Head architecture + initialisation |
| `train_ratio` / `split_by` / `val_subject` | float / str | Split policy |
| `alpha` / `beta` / `gamma` | float | `WaveformJointLoss` term weights |
| `fft_sizes` | str | Stage-3 STFT window sizes |
| `eval_band` | str | `f_low,f_high` Hz for Tier-2 spectral metrics |
| `eval_freq` / `save_preds` | int / bool | Validation frequency / save raw predictions |

---

## 4. Execution Environments

### 4.1 Local (WSL / single GPU) — `code/scripts/env_local.sh`

```bash
cd code/
source scripts/env_local.sh
python runners/run_pretrain.py -c configs/pretrain/stage2_local.yaml
# or via wrapper:
bash scripts/local/pretrain_rgb_roi_bp.sh
```

| Variable | Default |
|----------|---------|
| `RAW_DATA_PATH` | `<repo>/data/raw/BP4D` |
| `DATA_PATH` | `<repo>/data/processed/bp4d_canonical` |
| `OUTPUT_DIR` | `<repo>/output` |
| `NUM_WORKERS` | `4` |
| `INITIAL_MODELS_DIR` | `<repo>/models/initial` |

**Constraints**: single-process (no DDP), `input_size=64`, small `max_sessions` / `max_entries` for smoke tests.

### 4.2 HPC Cluster (Lichtenberg / Slurm, multi-GPU) — `code/scripts/env_hpc.sh`

Must edit: `VENV`, `RAW_DATA_PATH`, `DATA_PATH`, `PROJ_DIR`, `OUTPUT_DIR`, `PARTITION`, `GPU_TYPE`, `GPUS_PER_NODE`.

Available Slurm batch files (`code/scripts/hpc/`): `submit_sweep.sbatch` (the ADR-001 five-fold sweep and the only Stage-2/Stage-3 path), plus the platform jobs `submit_prepare.sbatch`, `submit_inspect.sbatch`, `submit_inspect_physio.sbatch`.

**Launch method**: SLURM `srun` (not torchrun). `init_distributed_mode` auto-detects `SLURM_PROCID` and derives `MASTER_ADDR` from `SLURM_NODELIST`.

**HPC config differences vs local**:
- `input_size: 224`, `epochs: 800`, `batch_size: 16`, `blr: 1.5e-4` (effective LR = `blr * total_batch / 256`)
- `clip_stride: 1.0` (overlapping windows, more clips)
- No `max_sessions` / `max_entries` limits

---

## 5. Runner Entry Points

| Runner | Stage | Key flags |
|--------|-------|-----------|
| `run_pretrain.py` | Stage 2 | `-c`, `--streams`, `--pretrained_encoder`, `--model` |
| `run_waveform.py` | Stage 3 | `-c`, `--target`, `--finetune`, `--sig_kernel`, `--physio_norm` |
| `run_finetune.py` | Classification | `-c`, `--model`, `--finetune`, `--nb_classes` |
| `run_au_probe.py` | AU probe | `-c`, `--probe` (linear/ft), `--finetune` |
| `run_evaluate.py` | Offline eval (arrays) | `--pred_path`, `--target_path`, `--fs`, `--tier`, `--waveform` |
| `run_evaluate_session.py` | Session-level eval | `--pred_dir`, session assembly, `--tier` |
| `run_evaluate_subject.py` | Subject-level rollup | `--pred_dir`, bootstrap CI, `--rate_window_s` |
| `run_evaluate_kfold.py` | Pooled k-fold eval | `--root`, `--run_prefix`, `--folds`, `--out` |
| `run_download_weights.py` | Weight download | `--list`, `--model` |
| `run_inspect_data.py` | Dataset inspection | `--plot`, `--max_sessions`, `--split` |
| `run_inspect_physio.py` | Physio signal QC | signal stats |
| `run_inspect_rgb_bp.py` | RGB-BP correlation study | — |
| `run_inspect_thermal.py` | TIR channel verification | — |
| `run_inspect_tir_resp.py` | TIR-RESP dataset check | ROI crop verification |
| `run_inspect_au.py` | AU label distribution | — |
| `run_visualize.py` | Visualisation | predicted vs. target waveforms |

**Common flags** (all runners via `add_common_args`): `--device`, `--seed`, `--dist_url`, `--local_rank`, `--output_dir`, `--resume`, `--log_wandb`, `--wandb_project`, `--num_workers`, `--tir_channels`, `--pin_mem`, `--max_sessions`, `--max_clips`, `--max_entries`.

### 5.1 Session-level assembly — `evaluation/assemble.py`

Stage 3 predicts ONE fixed-length window per clip, so a session-level score
requires rebuilding the whole recording first (training-free post-processing; the
protocol of record is `design/Evaluation_RespWaveform_protocol.md` §2/§4):

- `clip_offsets` / `group_entries` — offset `round(t_start * fs)` per clip, NOT
  `t_start * fs / temporal_stride` (only the VIDEO is decimated; the 1-D target
  keeps the full-rate window). Entries are grouped per session and time-sorted.
- `assemble_session(preds, offsets, fs, window='hann')` — Hann overlap-add with
  weight-sum normalisation, returning **`(t, y, covered)`**. Uncovered samples
  are `nan`, never 0, and callers MUST score `y[covered]` (a zero-fill moved one
  session's Pearson by 0.18 on the 2026-10-15 run). A covered sample whose window
  weights sum to 0 (the taper edge of a LONE window) falls back to the
  unweighted mean, so a single-clip session needs no special case.
- `affine_calibrate(pred, ref)` — least-squares `(a, b)` for the AMPLITUDE
  metrics only, because a per-clip z-scored prediction is defined up to a
  per-clip affine map. `pearson` / `psd_mae` / RR intervals need none;
  `mae_affine` / `rmse_affine` are ORACLE-calibrated and not deployable.

Consumers: `runners/run_evaluate_session.py` (per session + figure),
`run_evaluate_subject.py` (median over the subject's sessions + bootstrap CI),
`run_evaluate_kfold.py` (pooled folds). Tests: `tests/test_assemble_session.py`.

---

## 6. Pretrained Weight Management — `models/pretrained.py`

Downloads Stage-1 ViT weights (VideoMAE-base, MAE-base, etc.) from Hugging Face to `$INITIAL_MODELS_DIR`.
`resolve_encoder_weights('base', initial_dir)` maps short names to checkpoint paths.

> On HPC: pre-download on a **login node** (compute nodes have no internet). Use `run_download_weights.py --list` to see available specs.

---

## 7. Environment Variable Override Matrix

| Env Var | Argparse flag | Default |
|---------|--------------|---------|
| `OUTPUT_DIR` | `--output_dir` | `<project_root>/output` |
| `RESUME` | `--resume` | `''` |
| `NUM_WORKERS` | `--num_workers` | `8` |
| `TIR_CHANNELS` | `--tir_channels` | `3` |

`DATA_PATH`, `RAW_DATA_PATH`, and `INITIAL_MODELS_DIR` are consumed directly inside dataset builders and `models/pretrained.py::initial_dir`, not through argparse.
