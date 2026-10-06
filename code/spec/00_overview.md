# 00 — System Overview

> **READ-ONLY SPECIFICATION** — reverse-engineered from the live codebase on 2026-10-04.
> All claims are grounded in actual source files. Implicit assumptions and risks are flagged explicitly.

---

## 1. Research Domain & Purpose

This project implements a **multi-stage multi-modal masked autoencoder** for physiological waveform regression from facial video, operating on the **BP4D+** dataset (Binghamton University spontaneous facial behaviour database, biological signals edition).

The scientific goal is to reconstruct continuous 1-D physiological waveforms — **blood pressure (BP)**, **respiration (RESP)**, and **electrodermal activity (EDA)** — from:
- **RGB** facial video (JPEG frame sequences, ~25 fps)
- **Thermal-IR (TIR)** video (`.wmv` false-colour rainbow rendering, ~25 fps)

under a simulated **contact-sensor failure** scenario (Stage 3 only receives visual input).

The architecture is derived from two reference codebases:

| Reference | Role |
|-----------|------|
| **VideoMAE** (`tmp/videomae/`) | 3-D tubelet patch embedding, tube masking, VideoMAE-style norm-pix reconstruction |
| **MultiMAE** (`tmp/MultiMAE/`) | Multi-modal token fusion, per-stream asymmetric masking, shared cross-modal encoder |

---

## 2. Three-Stage Training Pipeline

```
Stage 1  ──►  Stage 2  ──►  Stage 3
(pretrained    (multimodal    (waveform
 ViT encoder)   MAE pretrain)  regression)
```

### Stage 1 — Pretrained Visual Encoder
- **Source**: Published VideoMAE **ViT-Base** weights — Kinetics-400 (`videomae:k400`) or Something-Something-v2 (`videomae:ssv2`) — downloaded by `runners/run_download_weights.py` into `models/initial/`.
- **Loading**: `core/multimae.load_pretrained_encoder` copies the transformer blocks and the 3-D tubelet `Conv3d` patch embed into the Stage-2 `MultiModalMAE` encoder. A blank/random or non-VideoMAE source is rejected.
- **Purpose**: Provides the initial encoder weights (tubelet Conv3d) for Stage 2. The Stage-1 checkpoint is **not retrained** in this project; the project compares the K400 vs SSV2 corpora as these initial weights.

### Stage 2 — Multimodal Masked Autoencoder Pre-training
- **Model**: `MultiModalMAE` (`core/multimae.py`)
- **Runner**: `runners/run_pretrain.py`
- **Engine**: `engines/pretrain.py::train_one_epoch`
- **Contract**: ≥1 visual stream (rgb / tir) AND ≥1 physiological 1-D signal (bp / resp / eda). All-visual or all-signal stream lists are rejected at construction time.
- **Objective**: Asymmetric masked reconstruction (VideoMAE-style `norm_pix`) — visual streams masked 50–75 %, physiological signals masked 90 %+. Per-modality `MaskedMSELoss`, combined as a weighted sum. An optional per-stream MR-STFT magnitude loss is available but **disabled in all current configs** (`spectral_weight: 0.0`).

### Stage 3 — Waveform Regression Fine-tuning
- **Model**: `MultiModalWaveformRegressor` (`core/waveform_model.py`)
- **Runner**: `runners/run_waveform.py`
- **Engine**: `engines/waveform.py` (evaluation) + inline training loop in `run_waveform.py`
- **Contract**: Visual streams ONLY (simulated sensor failure). The Stage-2 encoder weights are transferred verbatim (identical module names: `adapters.*`, `positions.*`, `enc_blocks.*`, `enc_norm`).
- **Objective**: `WaveformJointLoss` = α·`StdLoss` + β·`PearsonLoss` + γ·`MultiResolutionSTFTLoss` (`core/waveform_losses.py`).

### Add-on — AU Occurrence Probe
- **Model**: `MultiModalMAEProbe` (`core/au_probe.py`)
- **Runner**: `runners/run_au_probe.py` / `runners/run_inspect_au.py`
- **Purpose**: Diagnostic — tests whether facial action units (AUs) are linearly decodable from frozen Stage-2 encoder features. Does **not** modify Stage-2 weights.

---

## 3. High-Level Data Flow

```
data/raw/BP4D/
  <subject>/<task>/
    2D+3D/         (RGB JPEG frames)
    Thermal/       (TIR .wmv, false-colour)
    Physiology/    (CSV: time, bp, resp, eda)
        |
        v  data/prepare_bp4d.py
data/processed/bp4d_canonical/
  <session>/
    rgb/           (ordered JPEG frames)
    tir.wmv        (single ~60 s video)
    signals.csv    (header: time, bp, resp, eda)
        |
        v  code/data/  (paired_dataset.py / rgb_roi_dataset.py / tir_resp_dataset.py)
  Dataset.__getitem__()
    -> {'rgb': [C, T, H, W], 'tir': [C, T, H, W], 'bp': [1, S], ...}
        |
        v  runners/_common.py::make_data_loader
  DistributedSampler / RandomSampler -> DataLoader
        |
        v  Stage-2: engines/pretrain.py
  MultiModalMAE.forward(x)
    -> {'loss': scalar, 'losses_mse': {...}, 'losses_spectral': {...}}
        |
        v  Stage-3: run_waveform.py training loop
  MultiModalWaveformRegressor.forward(x)
    -> [B, output_len]   (predicted waveform)
        |
        v  engines/waveform.py / evaluation/
  Tier-1: MAE, RMSE, Pearson
  Tier-2: Welch PSD, dominant frequency error
  Tier-3: HRV metrics (RMSSD, pNN50, ShanEn) via NeuroKit2
```

---

## 4. Modular Decoupling Rules

| Rule | Enforcement |
|------|-------------|
| All source code lives under `code/`. `data/`, `models/`, `output/` are read-only artifacts. | Directory convention. |
| `code/data/` never imports from `code/engines/` or `code/runners/`. | Import graph. |
| `code/models/build.py::create_model` is the single model instantiation entry point for YAML-driven runners. | `@register_model` decorator in `core/registry.py`. |
| DDP bootstrap is centralised in `code/utils/dist.py::init_distributed_mode`. Runners must call `init_env(args)` from `runners/_common.py`. | Convention; not enforced by type system. |
| Seeding is performed in `runners/_common.py::init_env` (`torch.manual_seed`, `np.random.seed`, `random.seed` with `args.seed + get_rank()`). No standalone `set_seed()` helper. | Convention. |
| YAML configs supply argparse **defaults** only; explicit CLI flags always override. | `parse_args_with_config` in `runners/_common.py`. |
| Stage-2 and Stage-3 encoder geometry (tubelet / input_size / num_frames) must match exactly. Mismatches raise `ValueError` at load time. | `load_stage2_encoder` in `core/waveform_model.py`. |
| `torch.amp.autocast` is **not** used. AMP is handled by `NativeScalerWithGradNormCount` in `utils/native_scaler.py`. | Code review. |
| Metric logging uses `utils/logger.py::MetricLogger` (MultiMAE style). `stdlib.logging` and ad-hoc `print()` are not used for trainer metrics. | Convention; some `print()` calls remain in non-metric paths. |

---

## 5. Repository Layout (`code/` only)

```
code/
├── core/           # Model primitives, losses, registry
│   ├── multimae.py          # Stage-2 MultiModalMAE
│   ├── waveform_model.py    # Stage-3 MultiModalWaveformRegressor
│   ├── model.py             # ProjectViT (Stage-1 baseline)
│   ├── blocks.py            # Block, CrossAttention, DecoderBlock, Mlp, Attention
│   ├── input_adapters.py    # PatchedInputAdapter, SignalInputAdapter, SemSegInputAdapter
│   ├── output_adapters.py   # SpatialOutputAdapter
│   ├── criterion.py         # MaskedMSELoss, MaskedL1Loss, MaskedCrossEntropyLoss
│   ├── waveform_losses.py   # PearsonLoss, StdLoss, MultiResolutionSTFTLoss, WaveformJointLoss
│   ├── au_probe.py          # MultiModalMAEProbe
│   └── registry.py          # @register_model decorator + model_entrypoint
├── data/           # Datasets, masking generators, video I/O
├── models/         # Model factory (build.py) + pretrained weight download (pretrained.py)
├── engines/        # Training / evaluation loops
├── runners/        # CLI entry points (_common.py + run_*.py)
├── configs/        # YAML experiment configs
│   ├── pretrain/   # Stage-2 configs
│   └── finetune/   # Stage-3 configs
├── utils/          # DDP, logger, LR schedule, optimizer, checkpoint, metrics, pos_embed
├── evaluation/     # Post-hoc metric computation and clinical analysis
├── scripts/        # Shell wrappers (env_local.sh, env_hpc.sh, local/, hpc/)
└── tests/          # Reserved for pytest suite (currently empty)
```
