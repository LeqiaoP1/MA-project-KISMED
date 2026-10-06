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
| `stage2_local.yaml` | Local smoke (VideoMAE K400 init), `streams: rgb,bp`, 2 sessions, 8 clips |
| `stage2_local_pretrained.yaml` | Local with VideoMAE **K400** init, 11 sessions, full local clip budget |
| `stage2_local_pretrained_ssv2.yaml` | Twin of the above with VideoMAE **SSV2** init (the corpus A/B) |
| `stage2_local_rgb_roi_bp.yaml` | RGB face-ROI + BP, raw tree |
| `stage2_local_tir_roi_resp.yaml` | TIR-ROI + RESP, raw tree |
| `stage2_local_tir_roi_crossmae.yaml` | TIR-ROI, Solution A cross-MAE |
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
| `resp_tir_roi_crossmae.yaml` | RESP | TIR ROI, cross-MAE head |
| `resp_tir_roi_local_matched.yaml` | RESP | TIR ROI, matched geometry variant |
| `eda.yaml` | EDA | canonical, HPC |
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
| `streams` | str | Comma-separated: e.g. `rgb,tir,bp,resp,eda` |
| `tubelet` | str | `t,ph,pw`: e.g. `2,16,16` |
| `input_size` | int | Square frame px: 64 (local) / 224 (HPC) |
| `fps` / `fs` | float | Video fps / signal sample rate Hz |
| `clip_duration` / `clip_stride` | float | Clip length s / window hop s. The TIR-ROI/RESP study uses **8.0 / 1.0** (an 8 s window with a 1 s hop -> 7/8 overlap, to keep the transient breathing dynamics); the hop decides WHICH windows exist, so it belongs to the corpus contract — keep it equal in Stage 2 and 3 |
| `temporal_stride` | int | Frame decimation inside window (1 = every frame) |
| `seq_len` | int | Signal samples; 0 = `clip_duration * fs` |
| `pos_init` | str | `sincos3d` or `random` |
| `mask_ratio_{rgb\|tir\|bp\|resp\|eda}` | float | Per-stream masking ratio |
| `signal_weight` | float | Loss weight for physio streams (default 0.5) |
| `target_norm` | str | `token` (per-token z-score) or `clip` (per-clip z-score) |
| `spectral_weight` | float | MR-STFT loss weight; **0.0 in all current configs** |
| `spectral_fft_sizes` / `spectral_hop_ratio` | str / float | STFT config |
| `pretrained_encoder` | str | Stage-1 VideoMAE corpus: `videomae:k400` / `videomae:ssv2` / `videomae` / `base` (= k400) / a local path. Blank/random is **rejected** |
| `videomae_dataset` | str | `k400` (default) / `ssv2`: corpus for a bare `videomae` spec |
| `rail_touch_v` | float | TIR-ROI: drop a clip whose respiration window contains ANY sample that touched the recorder rail (`abs(x) >=` volts; 0.0 = off). **The cleaning knob**; the shipped `9.90` sits 0.1 V inside the `+/-10 V` clamp, so a dead/pinned channel AND a clipped flat-topped trough are both removed. Keep equal in Stage 2/3 |
| `min_signal_spread` | float | TIR-ROI: drop a window whose spread (`max−min`) is below this many volts (0.0 = off). The shipped `0.1` is the **dead-signal guard**: every corpus window below it sits at a median level of `+9.1 V` (pinned just inside the clamp, 0.05–0.09 V ripple); checked AFTER `rail_touch_v`, keep equal in Stage 2/3 |
| `epochs` / `batch_size` / `update_freq` / `save_ckpt_freq` | int | Training schedule |
| `opt` / `lr` / `blr` / `min_lr` / `warmup_epochs` | str / float | Optimizer + LR |
| `weight_decay` / `clip_grad` | float | Regularisation + grad clip (0.0 = OFF) |
| `data_path` / `data_set` | str | Dataset root / type selector |

### 3.3 Stage-3 fine-tuning (additional keys)

| Key | Type | Description |
|-----|------|-------------|
| `target` | str | `bp` / `resp` / `eda` |
| `finetune` | str | Path to Stage-2 checkpoint |
| `sig_kernel` | int | Signal window size per token; must satisfy `sig_kernel/fs == tubelet_t*stride/fps` |
| `signal_norm` | str | `none` or `zscore` |
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

Available Slurm batch files (`code/scripts/hpc/`): `submit_inspect.sbatch`, `submit_inspect_physio.sbatch`, `submit_prepare.sbatch`, `submit_waveform.sbatch`.

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
| `run_waveform.py` | Stage 3 | `-c`, `--target`, `--finetune`, `--sig_kernel`, `--signal_norm` |
| `run_finetune.py` | Classification | `-c`, `--model`, `--finetune`, `--nb_classes` |
| `run_au_probe.py` | AU probe | `-c`, `--probe` (linear/ft), `--finetune` |
| `run_evaluate.py` | Offline eval | `--preds_dir`, `--target`, `--fs` |
| `run_evaluate_session.py` | Session-level eval | `--preds_dir`, session assembly |
| `run_download_weights.py` | Weight download | `--list`, `--model` |
| `run_inspect_data.py` | Dataset inspection | `--plot`, `--max_sessions`, `--split` |
| `run_inspect_physio.py` | Physio signal QC | signal stats |
| `run_inspect_rgb_bp.py` | RGB-BP correlation study | — |
| `run_inspect_thermal.py` | TIR channel verification | — |
| `run_inspect_tir_resp.py` | TIR-RESP dataset check | ROI crop verification |
| `run_inspect_au.py` | AU label distribution | — |
| `run_visualize.py` | Visualisation | predicted vs. target waveforms |

**Common flags** (all runners via `add_common_args`): `--device`, `--seed`, `--dist_url`, `--local_rank`, `--output_dir`, `--resume`, `--log_wandb`, `--wandb_project`, `--num_workers`, `--tir_channels`, `--pin_mem`, `--max_sessions`, `--max_clips`, `--max_entries`.

---

## 6. Pretrained Weight Management — `models/pretrained.py`

Downloads the Stage-1 **VideoMAE ViT-Base** checkpoints from Hugging Face to `$INITIAL_MODELS_DIR`: `videomae:k400` (Kinetics-400) and `videomae:ssv2` (Something-Something-v2). `resolve_encoder_weights('videomae:k400', initial_dir)` maps a corpus spec to a checkpoint path; Stage-1 callers pass `require=True` so a blank spec is a hard error (no random-init backbone). ImageNet-MAE / timm / ViT-Large sources were removed.

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
