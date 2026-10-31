#!/usr/bin/env python3
"""Regenerate the six read-only spec files under ``code/notes/spec/``.

Run from anywhere:  python code/scripts/project/write_specs.py

WARNING -- THE FILES ON DISK ARE NEWER THAN THIS SCRIPT. The spec was first
reverse-engineered on 2026-10-04 and has been hand-updated since (for example
``00_overview.md`` on disk names the VideoMAE K400/SSV2 sources and adds a
"Loading" bullet that this script does not emit). Running it OVERWRITES those
hand-applied updates without warning. Before using it, re-derive the body
against the current code; never run it blind over a live spec.
"""
import pathlib

# code/scripts/project/write_specs.py -> repo root -> code/notes/spec
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SPEC = REPO_ROOT / "code" / "notes" / "spec"
SPEC.mkdir(parents=True, exist_ok=True)

# ─── FILE 0 ───────────────────────────────────────────────────────────────────
(SPEC / "00_overview.md").write_text(r"""# 00 — System Overview

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
- **Source**: Published VideoMAE-base / MAE-base weights, downloaded by `runners/run_download_weights.py` into `models/initial/`.
- **Model**: `ProjectViT` (`core/model.py`) — a standard 2-D ViT scaffold.
- **Purpose**: Provides the initial encoder weights (tubelet Conv3d) for Stage 2. The Stage-1 checkpoint is **not retrained** in this project.

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
""", encoding="utf-8")
print("wrote 00_overview.md")

# ─── FILE 1 ───────────────────────────────────────────────────────────────────
(SPEC / "01_data_pipeline.md").write_text(r"""# 01 — Data Pipeline

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/data/` on 2026-10-04.

---

## 1. Raw Dataset Layout (BP4D+)

```
data/raw/BP4D/
  <Subject>/          e.g. F001, M001
    <Task>/           e.g. T1 .. T10
      2D+3D/          JPEG frames (1392x1040, ~25 fps)
      Thermal/        TIR .wmv (false-colour WMV3/yuv420p, ~25 fps)
      Physiology/     signals CSV: [time, bp, resp, eda]
      2DFeatures/     landmark tracks (IRFeatures for TIR ROI)
```

> **Important**: The TIR stream is a **false-colour (rainbow) rendering**, not a grayscale image. Verified with two independent decoders (OpenCV wmv3/yuv420p + PyAV): 3 colour planes, mean |U-128| ~23–27 on every session. Default channel count is **3**. Use `--tir_channels 1` only to match legacy checkpoints trained before 2026-09.

---

## 2. Canonical Processed Layout

Output of `data/prepare_bp4d.py` -> `data/processed/bp4d_canonical/`:

```
<session>/           e.g. F001_T1
  rgb/               ordered JPEG frames
  tir.wmv            single ~60 s video
  signals.csv        header: [time, bp, resp, eda] at fs Hz
```

The session name encodes subject (`'F001_T1'.rsplit('_', 1)[0]` -> `'F001'`), enabling subject-disjoint splits.

---

## 3. Dataset Classes

### 3.1 Canonical Paired Layout — `data/paired_dataset.py`

| Class | Stage | Returns |
|-------|-------|---------|
| `PairedSessionDataset` | Stage 3 fine-tuning | `(samples [C,T,H,W], target [seq_len])` |
| `PairedPretrainDataset` | Stage 2 pre-training | `{stream: tensor}` dict |

**Session discovery** (`scan_sessions`): scans `data_path` for dirs containing `rgb/`, `tir.wmv`, `signals.csv`. Falls back to first `*.wmv`. Raises `FileNotFoundError` if no valid sessions found.

**Split policy**:
- `split_by='session'` (default): splits sorted session list at `train_ratio`. **Risk: subject leakage.**
- `split_by='subject'`: groups by subject prefix; subject-disjoint. **Recommended for evaluation.**
- Stage-2 (`PairedPretrainDataset`): always `train_ratio=1.0` (pretrain on all).

**Dev caps** (applied in order): `max_sessions` -> `max_clips` -> `max_entries`.

**Signal normalisation** (`signal_norm`): `'none'` (raw) or `'zscore'` (population std, ddof=0, per clip).

**`PairedPretrainDataset.__getitem__` output shapes**:

```python
{
  'rgb':  FloatTensor [3, T, H, W],
  'tir':  FloatTensor [C_tir, T, H, W],  # C_tir = 3 (default) or 1
  'bp':   FloatTensor [1, seq_len],
  'resp': FloatTensor [1, seq_len],
  'eda':  FloatTensor [1, seq_len],
}
```

### 3.2 RGB Face-ROI — `data/rgb_roi_dataset.py`

| Class | Stage | Notes |
|-------|-------|-------|
| `RGBRoiDataset` | Base | Landmark-cropped face from RGB frames |
| `RGBRoiPretrainDataset` | Stage 2 | `{'rgb': [3,T,H,W], 'bp': [1,S], ...}` |
| `RGBRoiFinetuneDataset` | Stage 3 | `(rgb_crop [3,T,H,W], target [output_len])` |

**ROI crop pipeline**: 2DFeature landmarks -> bounding box + `roi_padding` -> crop + resize to `input_size x input_size`. Sessions where 2DFeatures frame count differs from JPEG count are rejected (`unusable`).

**Builders**: `build_rgb_roi_pretrain_dataset(args)`, `build_rgb_roi_finetune_dataset(is_train, test_mode, args)`.

### 3.3 TIR-ROI + Respiration — `data/tir_resp_dataset.py`

| Class | Stage | Notes |
|-------|-------|-------|
| `BP4DPlusTIRRespDataset` | Base | Nose/mouth ROI from raw TIR frames |
| `TirRoiRespPretrainDataset` | Stage 2 | `{'tir': [C,T,H,W], 'resp': [1,S]}` |
| `TirRoiRespFinetuneDataset` | Stage 3 | `(tir_crop [C,T,H,W], target [output_len])` |

**ROI crop pipeline**: `IRFeatures` landmarks -> `roi_box_from_landmarks(..., crop_rule)` + `roi_padding`. With `crop_rule: square` (the default) the padded box is squared about its centre FIRST, so the rescale applies ONE scale factor (`sx == sy`) and the anatomy is never deformed; `crop_rule: box` keeps the legacy per-axis geometry for A/B. A box that cannot be placed without cutting the landmark envelope raises `RoiBoxOutOfFrame`. Clips whose target-landmark centroid travels further than `max_sway_px` (40 source px; 0 = off) inside the window are DROPPED (counted as `clips_dropped_motion`), and clips touching a (0,0) sentinel row are dropped whole.

**Label validity + the respiration cache (2026-10-31).** The TIR-ROI label path decides validity ONCE on the RAW signal and can read the label from an offline artefact (`data/resp_preprocess.py`, `data/resp_cache.py`, built by `runners/run_build_resp_cache.py`). A sample is invalid when it sits in a saturation EVENT (`abs(V) > 9.95` lasting `>= 1 s`), in ANY `5 s` window with `ptp < 0.3 V`, or in a valid run shorter than `8 s`. The artefact stores the RUN-MEAN-REMOVED signal on a fixed `fs_out == --fs` grid with `NaN` exactly where the mask is False; with `--resp_cache` a window is usable iff `mask[s:s+n].all()` and nothing re-tests its content. `rail_touch_v` / `min_signal_spread` are the **LEGACY raw path** knobs and are refused (set to 0) when a cache is attached, so content is never filtered twice. Contract: `design/RESP_session_preprocess.md`; measured impact: `eval/tir_resp/resp_preprocess_impact.md`.

**Task groups** (`data/task_groups.py`): the shipped tiers (v2, 2026-10-24) are `low=[T2,T3,T5]`, `moderate=[T1,T4]`, `high=[T6,T7,T8,T9,T10]` -- a PARTITION of `T1`..`T10`, canonical form `data.task_groups.SHIPPED_TASK_GROUPS`, pinned by `tests/test_task_group_configs.py`. Stage-2 and Stage-3 must use **identical** `task_groups` / `task_set`.

**Builders**: `build_tir_roi_pretrain_dataset(args)`, `build_tir_roi_finetune_dataset(is_train, test_mode, args)`.

### 3.4 AU Dataset — `data/au_dataset.py`

Per-frame AU occurrence labels; subject-disjoint split. Used exclusively by the AU probe.

---

## 4. Dataset Dispatcher — `data/datasets.py`

`build_dataset(is_train, test_mode, args)` dispatches on `args.data_set`:

| `data_set` value | Dataset class |
|-----------------|---------------|
| `'bp4d+'`, `'paired'` | `PairedSessionDataset` |
| `'tir_roi'`, `'tir_roi_resp'`, `'tir-roi'` | `TirRoiRespFinetuneDataset` |
| `'rgb_roi'`, `'rgb_roi_resp'`, `'rgb-roi'` | `RGBRoiFinetuneDataset` |
| custom registered | `@register_dataset` decorator |

`build_pretraining_dataset(args)` dispatches similarly for Stage-2.

---

## 5. Video I/O — `data/video_io.py`

### RGB (JPEG sequences)
- `read_image(path, target_size, gray)`: uses coarsest JPEG DCT scale that stays >= `target_size` (libjpeg 1/2, 1/4, 1/8) — ~1.6x faster than full decode + crop.
- `read_image_range(image_dir, start, n, target_size, gray)` -> `uint8 [T, H, W, C]`

### TIR (.wmv)
- `open_video(path)`: prefers `CV2ClipReader` (OpenCV), falls back to `DecordClipReader` (decord/ffmpeg).
- Both expose `.read_all(gray, target_size)` -> `uint8 [T, H, W, C]` and `.read_range(start, n, gray)`.
- **Implicit Assumption**: `CV2ClipReader.read_range(start=0)` does not seek. Open a fresh reader per clip range when absolute frame index matters.

### Temporal Alignment
`data/alignment.py` maps each RGB frame index to the nearest TIR frame to handle fps mismatch.

---

## 6. Masking Generators — `data/masking_generator.py`

All return `torch.LongTensor [num_patches]`, `1` = masked (to reconstruct), `0` = visible.

| Class | Strategy | Default use |
|-------|-----------|-------------|
| `RandomMaskingGenerator(input_size, mask_ratio)` | Uniform random | 1-D signals (mask_ratio >= 0.90) |
| `TubeMaskingGenerator(input_size, mask_ratio)` | Same spatial patches across all frames | Video (mask_ratio 0.50–0.90) |
| `MultiModalMaskingGenerator(stream_generators)` | Compose one per stream | Stage-2 |

Default Stage-2 mask ratios (from configs): RGB 0.90, TIR 0.90, BP/RESP/EDA 0.90.

> At `input_size=64`, `patch_size=16`: 4x4=16 spatial patches. Tube mask 0.90 keeps 1–2 visible per frame — <=12.5% of spatial context.

---

## 7. Signal Processing

`_load_signals_csv` infers `fs` from median `diff(t)` of the time column. `nan_to_num` applied to all columns.

Column aliases: `bp <- (bp, ppg, pulse)`, `resp <- (resp, respiration)`, `eda <- (eda, gsr, scr, electrodermal)`.

**Space-time alignment** (enforced by `MultiModalMAE._check_space_time_alignment`):

```
sig_kernel / fs  ==  tubelet_t * temporal_stride / fps
n_signal tokens  ==  Gt (temporal tubelet steps)
```

Default geometry verification:

```
sig_kernel=8, fs=100  ->  8/100 = 0.080 s per token
tubelet_t=2, stride=1, fps=25  ->  2*1/25 = 0.080 s per token  OK
seq_len=400, sig_kernel=8  ->  50 tokens == Gt=50  OK
```

Any violation raises `ValueError` at model construction (never silently misaligned).

---

## 8. DataLoader — `runners/_common.py::make_data_loader`

```python
DataLoader(dataset,
           sampler=DistributedSampler(...)  # or RandomSampler / SequentialSampler,
           batch_size=args.batch_size,
           num_workers=args.num_workers,
           pin_memory=args.pin_mem,   # default True
           drop_last=True)            # training loaders
```

`check_loader_not_empty(loader, name, args)` raises `SystemExit` with actionable message when 0 batches (e.g. dataset smaller than `batch_size` with `drop_last=True`).

---

## 9. Batch Shapes by Stage

| Stage | Type | Shape |
|-------|------|-------|
| Stage 2 | `{stream: Tensor}` | `{'rgb': [B,3,T,H,W], 'bp': [B,1,S], ...}` |
| Stage 3 | `(Tensor, Tensor)` | `([B,C,T,H,W], [B,output_len])` |
| AU probe | `(Tensor, Tensor)` | `([B,C,T,H,W], [B,n_aus])` |

Masking is applied **inside** `MultiModalMAE.forward`, not in the dataset or collate function.
""", encoding="utf-8")
print("wrote 01_data_pipeline.md")

# ─── FILE 2 ───────────────────────────────────────────────────────────────────
(SPEC / "02_model_architecture.md").write_text(r"""# 02 — Model Architecture

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/core/` on 2026-10-04.
> All tensor shapes are documented from actual `forward()` implementations.

---

## 1. Model Registry — `core/registry.py`

timm-style global registry. `@register_model` stores a factory by name. `models/build.py::create_model(name, **kwargs)` is the single public entry point for YAML-driven runners.

| Registered name | Class | Stage |
|----------------|-------|-------|
| `project_vit_small_patch16_224` | `ProjectViT` | 1 (baseline) |
| `project_vit_base_patch16_224` | `ProjectViT` | 1 (baseline) |
| `project_vit_large_patch16_224` | `ProjectViT` | 1 (baseline) |
| `project_vit_huge_patch16_224` | `ProjectViT` | 1 (baseline) |
| `project_multimae_small` | `MultiModalMAE` | 2 |
| `project_multimae_base` | `MultiModalMAE` | 2 |
| `project_multimae_large` | `MultiModalMAE` | 2 |
| `project_multimae_huge` | `MultiModalMAE` | 2 |

Stage-3 and AU-probe models use dedicated builders (`build_waveform_model`, `build_au_probe_model`), not `create_model`.

---

## 2. Transformer Blocks — `core/blocks.py`

### `Attention` (self-attention)

**Forward**: `x: [B, N, C]` -> `[B, N, C]`

```
qkv = Linear(C, 3C) -> [B, N, 3, heads, head_dim]
q, k, v = qkv.unbind(0)     # each [B, heads, N, head_dim]

Fast path (torch >= 2.0):
  x = F.scaled_dot_product_attention(q, k, v)   # never materialises [B,heads,N,N]
Fallback:
  attn = (q @ k.T * scale).softmax(-1) @ v      # [B,heads,N,N] materialised

x = x.transpose(1,2).reshape(B, N, C)
x = proj(x)
```

> **Critical**: At 224 px (9800 tokens), the naive path requires ~9 GB per block per sample. SDPA (torch >= 2.0) is mandatory for the full-resolution geometry.

### `CrossAttention`

**Forward**: `x: [B, N, C]`, `context: [B, M, C]` -> `[B, N, C]`

Queries from `x`; keys/values from `context`. Used in `DecoderBlock` for Solution-A cross-stream decoding (RESP stream attending to the full visual encoder output).

### `Block` (pre-norm encoder block)

```
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(Mlp(LayerNorm(x)))
```

**Forward**: `[B, N, D]` -> `[B, N, D]`

### `DecoderBlock` (self-attn + cross-attn + MLP)

```
x = x + DropPath(self_attn(LayerNorm(x)))
x = x + DropPath(cross_attn(query_norm(x), context_norm(context)))
x = x + DropPath(Mlp(LayerNorm(x)))
```

**Forward**: `x: [B, N, D]`, `context: [B, M, D]` -> `[B, N, D]`

### `Mlp`

`fc1(in -> hidden) -> GELU -> Dropout -> fc2(hidden -> out)`. Default `hidden = in * mlp_ratio` (typically 4).

---

## 3. Input Adapters

### `TubeletEmbed` (`core/multimae.py`) — Stage-2 / Stage-3 primary adapter

**Forward**: `x: [B, C, T, H, W]` -> `[B, Gt*Gh*Gw, D]`

```python
patch_embed = nn.Conv3d(C, D, kernel_size=(t,ph,pw), stride=(t,ph,pw))
x = patch_embed(x)            # [B, D, Gt, Gh, Gw]
x = x.flatten(2).transpose(1, 2)  # [B, Gt*Gh*Gw, D]
```

Grid sizes at standard geometries:

| `input_size` | `num_frames` | `tubelet` | `stride` | `Gt` | `Gh=Gw` | `n_visual` |
|-------------|-------------|----------|----------|------|---------|-----------|
| 64 | 100 | 2,16,16 | 1 | 50 | 4 | 800 |
| 64 | 200 | 2,16,16 | 2 | 50 | 4 | 800 |
| 224 | 100 | 2,16,16 | 1 | 50 | 14 | 9800 |

### `SignalEmbed` (`core/multimae.py`) — Stage-2 1-D signal adapter

**Forward**: `x: [B, 1, S]` -> `[B, Ns, D]`

```python
embed = nn.Conv1d(1, D, kernel_size=sig_kernel, stride=sig_kernel)
x = embed(x)          # [B, D, Ns]
x = x.transpose(1, 2)  # [B, Ns, D]
```

Default: `sig_kernel=8`, `S=400` -> `Ns=50`.

### Scaffold adapters (`core/input_adapters.py`) — used by `ProjectViT` only

| Class | Forward signature |
|-------|------------------|
| `PatchedInputAdapter` | `[B, C, H, W]` -> `[B, N, D]` via Conv2d |
| `SignalInputAdapter` | `[B, C, T]` -> `[B, N, D]` via Conv1d |
| `SemSegInputAdapter` | `[B, num_classes, H, W]` -> `[B, N, D]` via 1x1 conv + patch embed |

---

## 4. Position Embeddings — `utils/pos_embed.py`

`_PosMask(num_tokens, dim_tokens)` stores per-stream learnable `pos_embed [1, N, D]` and `mask_token [1, 1, D]`.

| Modality | Type | Shape | Function |
|----------|------|-------|----------|
| RGB / TIR (video) | 3-D sincos (t,h,w) | `[1, Gt*Gh*Gw, D]` | `get_3d_sincos_pos_embed` |
| BP / RESP / EDA | 1-D sincos (temporal) | `[1, Ns, D]` | Same time block as 3-D |
| `ProjectViT` | 2-D sincos | `[1, N+1, D]` | `get_2d_sincos_pos_embed` |

3-D embedding dim split (VideoMAE convention): `d_t = d_h = 2*(D//6)`, `d_w = D - 2*d_t`. For D=768: d_t=d_h=256, d_w=256.

`pos_init='sincos3d'` (default) initialises with sincos; `'random'` keeps trunc_normal (std=0.02).

---

## 5. Output Adapters — `core/output_adapters.py`

### `SpatialOutputAdapter`

**Forward**: `x: [B, N, D]` -> `[B, N, num_channels]`

```python
head = nn.Linear(D, num_channels)
```

`num_channels` = `tubelet_t * ph * pw * C` for video; `sig_kernel` for signals.

> `init(dim_tokens)` must be called before `forward()`. If skipped, `head is None` -> NoneType error with no context.

---

## 6. Stage-1 Baseline: `ProjectViT` — `core/model.py`

Standard 2-D ViT. Registered as `project_vit_{small|base|large|huge}_patch16_224`.

| Variant | `embed_dim` | `depth` | `num_heads` |
|---------|------------|---------|------------|
| small | 384 | 12 | 6 |
| base | 768 | 12 | 12 |
| large | 1024 | 24 | 16 |
| huge | 1280 | 32 | 16 |

### `forward(x)` tensor flow

```
x: [B, C, H, W]
  -> Conv2d(C, D, k=p, s=p) -> [B, D, H/p, W/p]
  -> flatten(2).transpose(1,2) -> [B, N, D]   N = (H/p)*(W/p)
  -> cat([cls_token, x], dim=1) -> [B, N+1, D]
  -> + pos_embed -> Dropout
  -> Block x depth -> LayerNorm -> [B, N+1, D]
  -> x[:, 0] -> [B, D] -> head -> [B, num_classes]
```

`forward_waveform(x)`: `[B, D] -> Linear(D, output_len) -> [B, output_len]`
(CLS-based baseline; the implementation plan recommends a decoder over all patch tokens for fine-grained waveforms.)

---

## 7. Stage-2: `MultiModalMAE` — `core/multimae.py`

### Construction-time validation

1. Rejects all-visual or all-signal `streams`.
2. Validates `tubelet` divides `input_size` and `num_frames`.
3. `_check_space_time_alignment()`: enforces `sig_kernel/fs == tubelet_t*temporal_stride/fps` AND `n_signal == Gt`. Raises `ValueError` otherwise.
4. `pos_init` in `{'sincos3d', 'random'}`.
5. `target_norm` in `{'token', 'clip'}`.
6. `resp_in_encoder` / `signal_decoder` consistency validated.

### Sub-modules

| Key | Module | Description |
|-----|--------|-------------|
| `adapters` | `nn.ModuleDict` | Per-stream `TubeletEmbed` or `SignalEmbed` |
| `positions` | `nn.ModuleDict` | Per-stream `_PosMask` (pos_embed + mask_token) |
| `enc_blocks` | `nn.ModuleList` | `enc_depth` x `Block` |
| `enc_norm` | `nn.LayerNorm` | Post-encoder norm |
| `dec_blocks` | `nn.ModuleList` | `dec_depth=2` x `Block` |
| `dec_norm` | `nn.LayerNorm` | Post-decoder norm |
| `heads` | `nn.ModuleDict` | Per-stream Linear head: `D -> patch_dim` |
| `cross_decoder.*` | `nn.ModuleList` | `DecoderBlock x cross_attn_depth` (Solution A only) |

### `forward(x: Dict[str, Tensor])` tensor flow

```
Input: {'rgb': [B,3,T,H,W], 'tir': [B,3,T,H,W], 'bp': [B,1,S]}

Step 1 — Tokenise + pos embed:
  tokens_s = adapters[s](x[s]) + positions[s].pos_embed   # [B, N_s, D]

Step 2 — Asymmetric masking:
  mask_s from MultiModalMaskingGenerator()                  # [N_s] in {0,1}
  visible_s = tokens_s[:, mask_s==0, :]                    # [B, N_vis_s, D]

Step 3 — Concatenate visible encoder-stream tokens:
  z_vis = cat([visible_s for s in encoder_streams], dim=1) # [B, sum(N_vis_s), D]

Step 4 — Shared ViT encoder:
  for blk in enc_blocks: z_vis = blk(z_vis)
  z_vis = enc_norm(z_vis)                                  # [B, sum(N_vis_s), D]

Step 5 — Re-insert mask tokens:
  full_s[:, mask_s==0] = z_vis[slice_s]
  full_s[:, mask_s==1] = positions[s].mask_token

Step 6 — Shared lightweight decoder (dec_depth=2):
  z_dec = cat([full_s for s in streams], dim=1)
  for blk in dec_blocks: z_dec = blk(z_dec)
  z_dec = dec_norm(z_dec)                                  # [B, sum(N_s), D]

Step 7 — Per-stream linear heads:
  pred_s = heads[s](z_dec[slice_s])                        # [B, N_s, patch_dim_s]

Step 8 — Norm-pix targets (per-token or per-clip z-score based on target_norm)

Step 9 — Weighted masked MSE + optional MR-STFT loss

Returns: {'loss': scalar, 'losses_mse': {...}, 'losses_spectral': {...}}
```

### Geometry variants

| Variant | `embed_dim` | `enc_depth` | `enc_num_heads` |
|---------|------------|------------|----------------|
| small | 384 | 12 | 6 |
| base | 768 | 12 | 12 |
| large | 1024 | 24 | 16 |
| huge | 1280 | 32 | 16 |

`dec_depth=2` (hardcoded default). Decoder uses the same `embed_dim` as encoder.

---

## 8. Stage-3: `MultiModalWaveformRegressor` — `core/waveform_model.py`

### Sub-modules (identical key names to Stage-2 encoder)

| Key | Description |
|-----|-------------|
| `adapters` (`nn.ModuleDict`) | Per-stream `TubeletEmbed` |
| `positions` (`nn.ModuleDict`) | Per-stream `_PosMask` (pos_embed only; mask_token unused) |
| `enc_blocks` (`nn.ModuleList`) | `enc_depth` x `Block` |
| `enc_norm` (`nn.LayerNorm`) | Post-encoder norm |
| `waveform_head` | `Linear(D,k)` / `Sequential(D->h,GELU,h->k)` / `WaveformUpsampleHead` |

> Key name identity with Stage-2 is critical: `load_stage2_encoder` transfers weights by exact key match.

### `forward(x)` tensor flow

```
x: {'rgb': [B,3,T,H,W]}  (or plain Tensor for single-stream)

For each stream s:
  xt = _fit_time(x[s], num_frames, s)    # trim if T > n; raise ValueError if T < n
  tok = adapters[s](xt)                  # [B, n_visual, D]
  assert tok.shape[1] == n_visual        # hard geometry check
  tok = tok + positions[s].pos_embed

z = cat(parts, dim=1)                    # [B, S * n_visual, D]
for blk in enc_blocks: z = blk(z)
z = enc_norm(z)

For each stream i:
  zi = z[:, i*n_visual : (i+1)*n_visual]         # [B, n_visual, D]
  zi = zi.reshape(B, grid_t, grid_h*grid_w, D)
  zi = zi.mean(dim=2)                            # [B, grid_t, D]
h = stack(feats, dim=0).mean(dim=0)              # [B, grid_t, D]

pred = waveform_head(h)                          # [B, grid_t, k]
return pred.reshape(B, -1)                      # [B, output_len]
```

`output_len = grid_t * k`; must be a multiple of `grid_t` (enforced at construction).

### Head variants

| `head_style` | `head_hidden` | Module |
|-------------|--------------|--------|
| `'linear'` | 0 | `Linear(D, k)` |
| `'linear'` | h > 0 | `Linear(D,h) -> GELU -> Linear(h,k)` |
| `'transposed_conv'` | N/A | `WaveformUpsampleHead` |

### `WaveformUpsampleHead` tensor flow

```
h: [B, grid_t, D]
  -> transpose(1,2)                      -> [B, D, grid_t]
  -> Conv1d(D, hidden, k=3, pad=1)       -> [B, hidden, grid_t]
  -> GELU
  -> ConvTranspose1d(hidden, 1, k=s, s=s) -> [B, 1, grid_t*s]
  -> transpose(1,2)                      -> [B, grid_t, s]
```

`s = samples_per_token`. Segment k covers exactly tubelet window k (time-alignment preserved).

---

## 9. AU Probe: `MultiModalMAEProbe` — `core/au_probe.py`

Identical module keys to Stage-2 (`adapters`, `positions`, `enc_blocks`, `enc_norm`) + linear `head = Linear(D, num_classes)`.

### `forward(x)` tensor flow

```
x: Dict[str, [B, C, T, H, W]]  (visual streams only)

For each stream s:
  xt = _resize_t(x[s], num_frames)    # trailing trim if T > n; raise if T < n
  tok = adapters[s](xt) + positions[s].pos_embed   # [B, N, D]

z = cat(parts, dim=1)                # [B, S*N, D]
for blk in enc_blocks: z = blk(z)
z = enc_norm(z)
z = z.mean(dim=1)                   # [B, D]  (mean-pool; no CLS token)
return head(z)                      # [B, num_classes]
```

Probe mode: `freeze_features()` sets `requires_grad=False` on all non-head params. `set_train(mode)` keeps frozen layers in `eval()`.

---

## 10. Loss Functions

### Masked Reconstruction — `core/criterion.py`

All: `pred: [B,N,D]`, `target: [B,N,D]`, `mask: [B,N]` (1 = masked = compute loss).

| Loss | Formula |
|------|---------|
| `MaskedMSELoss` | `mean_masked(mean_D((pred-target)^2))` |
| `MaskedL1Loss` | `mean_masked(mean_D(\|pred-target\|))` |
| `MaskedCrossEntropyLoss` | `mean_masked(CE(pred_BNC, target_BN))` |

Denominator: `mask.sum() + 1e-6`.

### Stage-3 Waveform Losses — `core/waveform_losses.py`

All accept `[B,T]` or `[B,1,T]` (channel squeezed via `_to_1d`).

**`PearsonLoss`**: `mean(1 - r)` where `r = (p_norm . t_norm) / (||p_norm|| * ||t_norm||)`.

**`StdLoss`** (L_time): `mean(|std(pred) - std(target)|)`. Population std with `eps=1e-8` under sqrt to avoid NaN on constant input.

**`MultiResolutionSTFTLoss`**: for each `n_fft` in `fft_sizes`:

```
P_mag = |STFT(pred, n_fft, hann)|    [B, n_fft//2+1, frames]
T_mag = |STFT(target, n_fft, hann)|
sc = ||P_mag - T_mag|| / (||T_mag|| + eps)             (spectral convergence)
lm = mean(|log(P_mag + 1e-6) - log(T_mag + 1e-6)|)    (log-magnitude L1)
total += sc + lm
```

Degenerate-target guard: skip samples with `||T_mag|| <= 1e-6 * max_batch(||T_mag||)`.

**`WaveformJointLoss`**: `L = alpha*StdLoss + beta*PearsonLoss + gamma*MultiResolutionSTFTLoss`.  
Default alpha=beta=gamma=1.0. Weights stored as `register_buffer`.

---

## 11. Stage-2 -> Stage-3 Weight Transfer — `core/waveform_model.py::load_stage2_encoder`

| Source key prefix | Destination |
|-------------------|-------------|
| `enc_blocks.*` | `enc_blocks.*` |
| `enc_norm.*` | `enc_norm.*` |
| `adapters.<s>.*` | `adapters.<s>.*` |
| `positions.<s>.pos_embed` | `positions.<s>.pos_embed` |
| `heads.<target>.*` | `waveform_head.*` (if `head_init='transfer'`) |

Geometry guard: `pos_embed` shape mismatch raises `ValueError` with actionable message.

Stage-1 ViT -> Stage-2: `canonicalise_vit_state_dict` handles MAE, VideoMAE, and HF-transformers checkpoint layouts. 2-D `patch_embed.proj.weight [D, C, ph, pw]` is inflated along the tubelet time axis to `[D, C, t, ph, pw]` (repeating, divided by t).
""", encoding="utf-8")
print("wrote 02_model_architecture.md")

# ─── FILE 3 ───────────────────────────────────────────────────────────────────
(SPEC / "03_engine_and_loss.md").write_text(r"""# 03 — Engine and Loss

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/engines/`, `code/utils/`, `code/runners/` on 2026-10-04.

---

## 1. Engine Overview

| File | Function(s) | Stage | Notes |
|------|------------|-------|-------|
| `engines/pretrain.py` | `train_one_epoch` | Stage 2 | Batch is `dict {stream: Tensor}` |
| `engines/finetune.py` | `train_one_epoch`, `evaluate` | Stage 3 / classification | Supervised `(samples, targets)` tuples |
| `engines/waveform.py` | `evaluate_waveforms`, `predict_waveforms` | Stage 3 eval | `@torch.no_grad()` decorated |
| `engines/au_probe.py` | `train_one_epoch_au`, `evaluate_au` | AU probe | BCE + per-AU F1 |
| `engines/visualize.py` | `visualize` | Debug | Prediction visualisation |

**No `trainer.py` or `evaluator.py`** exist. The Stage-3 training loop lives directly in `runners/run_waveform.py`.

---

## 2. Stage-2 Pre-training Loop — `engines/pretrain.py::train_one_epoch`

### Batch contract

```python
# dict batch (multimodal MAE):
x = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
out = model(x);  loss = out['loss'] / update_freq

# plain tensor (single-stream fallback):
samples = batch[0].to(device, non_blocking=True)
out = model(samples)
loss = (out['loss'] if isinstance(out, dict) else out) / update_freq
```

### Per-step LR / WD update

```python
sched_idx = step if start_steps is None else start_steps + step
for pg in optimizer.param_groups:
    pg['lr'] = lr_schedule_values[sched_idx] * pg.get('lr_scale', 1.0)
    if pg['weight_decay'] > 0:
        pg['weight_decay'] = wd_schedule_values[sched_idx]
```

`lr_scale` enables layer-wise LR decay without overwriting the per-group scale. Groups without `lr_scale` default to 1.0.

### Gradient step — AMP path

```python
grad_norm = loss_scaler(
    loss, optimizer, clip_grad=max_norm,
    parameters=model.parameters(),
    update_grad=(data_iter_step + 1) % update_freq == 0)
if (data_iter_step + 1) % update_freq == 0:
    optimizer.zero_grad()
```

### Gradient step — non-AMP path

```python
loss.backward()
if max_norm is not None and max_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
if (data_iter_step + 1) % update_freq == 0:
    optimizer.step(); optimizer.zero_grad()
```

### Metric logging (all `.item()` before storing)

```python
metric_logger.update(loss=loss.item() * update_freq)
metric_logger.update(lr=optimizer.param_groups[0]['lr'])
if grad_norm: metric_logger.update(grad_norm=grad_norm.item())
# Per-stream diagnostics (value already a Python float from model forward):
for name, val in out.get('losses_mse', {}).items():
    metric_logger.update(**{f'mse_{name}': val})
for name, val in out.get('losses_spectral', {}).items():
    metric_logger.update(**{f'spec_{name}': val})
```

---

## 3. Fine-tuning Loop — `engines/finetune.py::train_one_epoch`

```python
samples, targets = batch[:2]
samples = samples.to(device, non_blocking=True)
targets = targets.to(device, non_blocking=True)
if mixup_fn: samples, targets = mixup_fn(samples, targets)
outputs = model(samples)
loss = criterion(outputs, targets) / update_freq
```

### `evaluate` (classification only)

```python
@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()   # hardcoded — classification only
    model.eval()
    ...
    acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    metric_logger.update(loss=loss.item())
```

> **Potential Risk**: `CrossEntropyLoss` is hardcoded. This function is **not** suitable for waveform regression. Stage-3 uses `engines/waveform.py::evaluate_waveforms` instead.

---

## 4. Stage-3 Waveform Evaluation — `engines/waveform.py`

### `predict_waveforms`

```python
@torch.no_grad()
def predict_waveforms(data_loader, model, device):
    model.eval()
    for batch in data_loader:
        samples, target = batch[:2]
        samples = samples.to(device, non_blocking=True)
        out = model(samples)      # [B, T]
        preds.append(to_numpy(out)); targets.append(to_numpy(target))
    return np.concatenate(preds, 0), np.concatenate(targets, 0)  # [N, T]
```

### `evaluate_waveforms`

```python
pred, target = predict_waveforms(data_loader, model, device)
results = time_domain_metrics(pred, target)   # Tier 1: MAE, RMSE, Pearson
if with_spectral:
    results.update(spectral_metrics(pred, target, fs=fs, band=band))  # Tier 2
```

### Metric tiers

| Tier | Metrics | Library |
|------|---------|---------|
| 1 | MAE, RMSE, Pearson | numpy |
| 2 | Welch PSD, dominant freq error, HF power | scipy (optional) |
| 3 | RMSSD, pNN50, MedianNN, ShanEn | neurokit2 (optional, offline only) |

Tier 3 requires >= 30 s of signal; run offline via `runners/run_evaluate.py`.

---

## 5. AU Probe Engine — `engines/au_probe.py`

- Default `criterion = nn.BCEWithLogitsLoss()` (multi-label binary).
- `model.set_train(True)` instead of `model.train(True)` — respects linear-probe frozen-layer semantics.
- Threshold 0.5; macro-averaged F1 over per-AU scores.
- `loss.item()` called before `metric_logger.update` — safe.

---

## 6. Multi-task Loss Composition

### Stage-2 weighted sum

```python
# In MultiModalMAE.forward:
L_total = sum(loss_weights.get(s, signal_weight if s in signals else 1.0)
              * MaskedMSELoss(pred_s, target_s, mask_s)
              for s in streams)
       + sum(spectral_weights.get(s, 0.0) * STFT_s for s in signal_streams)
```

Default: visual streams = 1.0, physiological streams = 0.5 (`signal_weight`).
`spectral_weight=0.0` in all current configs (decision 2026-09-26).

### Stage-3 joint loss

```python
# WaveformJointLoss.forward:
L = alpha * StdLoss(pred, target)
  + beta  * PearsonLoss(pred, target)
  + gamma * MultiResolutionSTFTLoss(pred, target)
```

Gradient norm balance (measured 2026-09-28, N=496 @ 100 Hz):
- `StdLoss`: `1/sqrt(N) ≈ 0.0449` (~1.81 % of composite)
- `MultiResolutionSTFTLoss`: ~63–84 % of composite norm at alpha=beta=gamma=1

---

## 7. AMP — `utils/native_scaler.py::NativeScalerWithGradNormCount`

```python
self._scaler = torch.cuda.amp.GradScaler()  # deprecated PyTorch >= 2.4

def __call__(self, loss, optimizer, clip_grad=None, ...):
    self._scaler.scale(loss).backward()
    if update_grad:
        if clip_grad is not None and clip_grad > 0:  # GUARD: 0.0 = OFF
            self._scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            self._scaler.unscale_(optimizer)
            norm = get_grad_norm_(parameters)
        self._scaler.step(optimizer); self._scaler.update()
    return norm
```

> **Critical fix (2026-09-23)**: Before this fix, `clip_grad=0.0` (the default in many configs) entered the `clip_grad_norm_` branch with `max_norm=0.0`, which scales every gradient to zero (`clip_coef = 0 / (norm + eps)`). Learning was silently disabled. The guard `clip_grad > 0` now routes 0.0 to the no-clip path.

> **No `torch.amp.autocast`** anywhere in the codebase. Forward passes run in full precision; only the backward is scaled.

---

## 8. LR Schedule — `utils/lr_sched.py::cosine_scheduler`

Returns a numpy array of length `epochs * niter_per_ep`:
- Linear warmup from `start_warmup_value` to `base_value` over `warmup_epochs`.
- Cosine decay from `base_value` to `final_value`.

Applied **per step** in engine loops:

```python
pg['lr'] = lr_schedule_values[sched_idx] * pg.get('lr_scale', 1.0)
```

---

## 9. Optimizer — `utils/optim_factory.py::create_optimizer`

Supports: `adamw` (default), `adam`, `sgd`.

Parameter grouping:
- `no_decay`: 1-D params, biases, `no_weight_decay()` set -> `weight_decay=0`.
- `decay`: all others -> `weight_decay=args.weight_decay`.

Layer-wise LR decay (`build_layer_decay_assigner`):

```python
def get_num_layer_for_multimae(var_name, num_max_layer):
    if var_name.startswith(('adapters.', 'positions.')): return 0   # most decayed
    if var_name.startswith('enc_blocks.'): return int(block_idx) + 1
    return num_max_layer - 1                                        # top (LR scale 1.0)
```

`lr_scale[i] = layer_decay ^ (num_layers + 1 - i)`.

---

## 10. DDP — `utils/dist.py::init_distributed_mode`

| Launch mode | Detection | Backend |
|-------------|-----------|---------|
| `torchrun` / `torch.distributed.launch` | `RANK`, `WORLD_SIZE`, `LOCAL_RANK` | NCCL |
| SLURM `srun` | `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID` | NCCL |

SLURM `MASTER_ADDR` auto-derived from `SLURM_NODELIST`. `MASTER_PORT` defaults to `29500`.

```python
torch.distributed.init_process_group(backend='nccl', init_method='env://', ...)
torch.distributed.barrier()
setup_for_distributed(args.rank == 0)  # suppress non-master prints
```

`args.distributed=False` when no env vars found (local single-GPU mode).

DDP helpers: `get_rank`, `get_world_size`, `is_main_process`, `save_on_master`, `get_model`, `all_gather`.

`MetricLogger.synchronize_between_processes()` calls `dist.all_reduce` on each meter's `(count, total)`.

---

## 11. Checkpointing — `utils/checkpoint.py`

Checkpoint dict: `{'model', 'optimizer', 'epoch', 'scaler', 'args', ['model_ema']}`.

```python
torch.load(path, map_location='cpu', weights_only=False)
# weights_only=False required: 'args' is argparse.Namespace with pickled numpy scalars.
# torch >= 2.6 defaults to weights_only=True and refuses unpickling.
```

`auto_resume_model` reads `{output_dir}/checkpoints/latest_checkpoint.txt` if no explicit `--resume`.

EMA: `utils/model_ema.py`. Updated after each optimizer step: `model_ema.update(model)`.

---

## 12. Metric Logging — `utils/logger.py`

`SmoothedValue`: running window deque + global (count, total). Properties: `median`, `avg`, `global_avg`, `max`, `value`.

`MetricLogger.update(**kwargs)`: calls `.item()` on any Tensor before storing -> no computational graph retained (defensive).

`log_every(iterable, print_freq, header)`: wraps the data iterator, prints ETA and per-step timing.
""", encoding="utf-8")
print("wrote 03_engine_and_loss.md")

# ─── FILE 4 ───────────────────────────────────────────────────────────────────
(SPEC / "04_configs_and_execution.md").write_text(r"""# 04 — Configs and Execution

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
| `clip_duration` / `clip_stride` | float | Clip length s / window hop s |
| `temporal_stride` | int | Frame decimation inside window (1 = every frame) |
| `seq_len` | int | Signal samples; 0 = `clip_duration * fs` |
| `pos_init` | str | `sincos3d` or `random` |
| `mask_ratio_{rgb\|tir\|bp\|resp\|eda}` | float | Per-stream masking ratio |
| `signal_weight` | float | Loss weight for physio streams (default 0.5) |
| `target_norm` | str | `token` (per-token z-score) or `clip` (per-clip z-score) |
| `spectral_weight` | float | MR-STFT loss weight; **0.0 in all current configs** |
| `spectral_fft_sizes` / `spectral_hop_ratio` | str / float | STFT config |
| `pretrained_encoder` | str | `base` / `large` / `''` (from scratch) |
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
""", encoding="utf-8")
print("wrote 04_configs_and_execution.md")

# ─── FILE 5 ───────────────────────────────────────────────────────────────────
(SPEC / "05_risks_and_compliance.md").write_text(r"""# 05 — Risks and Compliance

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/` on 2026-10-04.
> This file documents hardcoded parameters, missing dimension assertions, potential OOM hazards,
> and other compliance risks found in the live codebase.
> **No source files were modified during this audit.**

---

## 1. Hardcoded Parameters

### `core/criterion.py`

| Location | Value | Risk |
|----------|-------|------|
| Module-level `_EPS = 1e-6` | Denominator in all masked losses | Safe in practice; could surface if `mask.sum() == 0` (all tokens visible). |
| `MaskedMSELoss.__init__`: `patch_size=16`, `stride=1` | Default constructor args | Unused in current `forward()` — misleading interface; callers may assume they affect computation. |

### `core/waveform_losses.py`

| Location | Value | Risk |
|----------|-------|------|
| Module-level `_EPS = 1e-8` | Denominator in waveform losses | Slightly tighter than criterion.py; consistent within file. |
| `MultiResolutionSTFTLoss.__init__`: `fft_sizes=(64,128,256)` | Default FFT windows | **Wrong for RESP** (0.1–0.6 Hz): 64/128-sample windows (0.64/1.28 s) are shorter than one breath period. `resp_tir_roi_local.yaml` correctly overrides to `128,256,512`. |
| `_warned_degenerate = False` | Instance-level warning flag | Warning fires once per model instance. All subsequent degenerate batches are silently skipped — no per-epoch count in logs. |

### `core/multimae.py`

| Location | Value | Risk |
|----------|-------|------|
| `STREAM_CHANNELS = {'rgb':3,'tir':3,'bp':1,'resp':1,'eda':1}` | Default channel counts | A new modality requires updating this dict. |
| `dec_depth=2` (constructor default) | Decoder depth | Not exposed as a YAML key; changing requires a code edit or explicit CLI override. |
| `_VISUAL_STREAMS = ('rgb','tir')` | Valid visual stream names | Hardcoded tuple; adding a new visual modality requires editing two constants. |
| `_SIGNAL_STREAMS = ('bp','resp','eda')` | Valid signal stream names | Same risk. |
| `spectral_weight=0.0` constructor default | MR-STFT disabled | Intentional (decision 2026-09-26). Documented in all YAML comments. |

### `utils/dist.py`

| Location | Value | Risk |
|----------|-------|------|
| `MASTER_PORT = '29500'` | Default SLURM port | Port collision if multiple jobs on the same node. Override via `MASTER_PORT` env var per job. |
| `args.dist_backend = 'nccl'` | Fixed backend | NCCL requires CUDA. CPU-only or Gloo-based runs need a code change. |

### `runners/_common.py`

| Location | Value | Risk |
|----------|-------|------|
| `--device` default `'cuda'` | Silent CPU fallback | No warning printed when CUDA unavailable; a slow CPU run may appear correct. |
| `--wandb_project` default `'thesis-project'` | Hardcoded W&B project | All runs log to the same project by default; must override for separate experiments. |
| `torch.backends.cudnn.benchmark = True` | Set in `init_env` | Disables deterministic CUDA algorithms. Disable for strict reproducibility. |

### `utils/native_scaler.py`

| Location | Value | Risk |
|----------|-------|------|
| `torch.cuda.amp.GradScaler()` | Deprecated PyTorch >= 2.4 | Currently functional; migrate to `torch.amp.GradScaler('cuda')` before removal. |

---

## 2. Missing Tensor Dimension Assertions

### Missing (risks)

| File | Class / Method | Missing check |
|------|---------------|---------------|
| `core/input_adapters.py` | `PatchedInputAdapter.forward` | No `assert x.ndim == 4`; a 3-D input raises inside Conv2d with an unhelpful error. |
| `core/input_adapters.py` | `SemSegInputAdapter.forward` | No channel count assertion. |
| `core/input_adapters.py` | `SignalInputAdapter.forward` | No `ndim` check. |
| `core/output_adapters.py` | `SpatialOutputAdapter.forward` | No `assert self.head is not None`; if `init()` was skipped, raises NoneType with no context. |
| `core/blocks.py` | `CrossAttention.forward` | No check that `x` and `context` share the same `C`; mismatch surfaces inside Linear. |

### Present — positive examples

| Location | Assertion |
|----------|-----------|
| `MultiModalMAE.__init__` | `assert input_size % ph == 0`, `assert num_frames % t == 0` |
| `MultiModalMAE._check_space_time_alignment` | `ValueError` on time-axis misalignment |
| `MultiModalWaveformRegressor.__init__` | `raise ValueError` if `output_len % grid_t != 0` |
| `MultiModalWaveformRegressor.forward` | `raise ValueError` if `tok.shape[1] != n_visual` |
| `MultiModalMAEProbe.__init__` | `assert input_size % ph == 0`, `assert num_frames % t == 0` |

---

## 3. Missing `.item()` Calls — GPU OOM Risk

A tensor loss stored without `.item()` retains the computational graph in GPU memory.

### Safe paths (`.item()` IS called)

| File | Location | Status |
|------|----------|--------|
| `engines/pretrain.py` | `metric_logger.update(loss=loss.item() * update_freq)` | OK |
| `engines/pretrain.py` | `metric_logger.update(grad_norm=grad_norm.item())` | OK |
| `engines/finetune.py` | `metric_logger.update(loss=loss.item() * update_freq)` | OK |
| `engines/finetune.py::evaluate` | `metric_logger.update(loss=loss.item())` | OK |
| `engines/au_probe.py` | `metric_logger.update(loss=loss.item() * ...)` | OK |
| `engines/au_probe.py::evaluate_au` | `total_loss += loss.item() * b` | OK |
| `utils/logger.py::MetricLogger.update` | Calls `.item()` on any Tensor before storing | OK (defensive) |

### Potential Risk — `MultiResolutionSTFTLoss`

```python
total = torch.tensor(0.0, device=p.device, dtype=p.dtype)
total = total + sc + lm   # accumulates as GPU tensor
```

`total` is returned as a tensor, passed to `.backward()`, then extracted with `.item()` by the engine's `metric_logger.update`. The graph is held only for one batch. **No persistent OOM risk; low severity.**

### Action Required — `runners/run_waveform.py`

The Stage-3 training loop is inline in this large file. Verify all metric scalars use `.item()` before accumulation:

```bash
grep -n "\.item()" code/runners/run_waveform.py
```

---

## 4. Unhandled Edge Cases

### 4.1 `engines/finetune.py::evaluate` — hardcoded `CrossEntropyLoss`

```python
criterion = torch.nn.CrossEntropyLoss()   # classification only
```

If called for waveform regression, silently produces incorrect metrics. Stage-3 uses `engines/waveform.py::evaluate_waveforms`. No runtime guard. **Risk: Medium.**

### 4.2 `data/paired_dataset.py::_pad_time` — edge padding without warning

```python
np.pad(arr, pad, mode='edge')
```

Short clips (session end) are padded by repeating the last frame/sample. No warning emitted. **Risk: Low (training), Medium (evaluation).**

### 4.3 `data/masking_generator.py::TubeMaskingGenerator` — dimension naming confusion

```python
self.height, self.width, self.depth = input_size[0], input_size[1], input_size[2]
# input_size = (T, H, W) but:  height=T  width=H  depth=W
```

The maths is correct (`total = T*H*W`) but `__repr__` outputs `[depth, height, width]` = `[W, T, H]` — incorrect display order. **Risk: Low functional; High readability. Implicit Assumption.**

### 4.4 `core/au_probe.py::load_au_probe_weights` — `print()` not logger

Load summary uses `print()` rather than `MetricLogger`. Suppressed on non-master DDP ranks by `setup_for_distributed`. Not a bug; deviates from logging convention.

### 4.5 `utils/dist.py::all_gather` — deprecated `torch.ByteStorage`

```python
storage = torch.ByteStorage.from_buffer(buffer)
tensor = torch.ByteTensor(storage).to('cuda')
```

Deprecated in PyTorch >= 2.0. **Risk: Low now; Medium in future PyTorch versions. Implicit Assumption (written for PyTorch 1.x patterns).**

### 4.6 `data/paired_dataset.py` — `split_by='session'` default allows subject leakage

Alphabetical session split at `train_ratio` may put `F001_T1` in train and `F001_T2` in val. Always use `split_by='subject'` for evaluation configs. **Risk: Medium (evaluation validity).**

### 4.7 `core/waveform_losses.py::MultiResolutionSTFTLoss` — `_warned_degenerate` fires once per instance

After the first degenerate batch, all subsequent ones are silently skipped in every subsequent epoch. No per-epoch degenerate count appears in logs. **Risk: Low (training integrity); High (data quality monitoring).**

### 4.8 `core/waveform_model.py` — no `@torch.inference_mode()` on `forward`

Callers must explicitly use `torch.no_grad()` during evaluation. `engines/waveform.py::predict_waveforms` does this correctly. Ad-hoc usage outside the engine may retain unnecessary gradients.

---

## 5. Summary Risk Table

| Risk | Severity | File(s) | Recommended action |
|------|----------|---------|-------------------|
| `split_by='session'` default allows subject leakage | Medium | `paired_dataset.py`, YAML configs | Always set `split_by: subject` in eval configs |
| `finetune.py::evaluate` hardcodes `CrossEntropyLoss` | Medium | `engines/finetune.py` | Verify not called for waveform regression |
| MR-STFT default `fft_sizes=(64,128,256)` unsuitable for RESP | Medium | `waveform_losses.py` | Override to `128,256,512` in all RESP configs |
| `torch.cuda.amp.GradScaler` deprecated | Low-Medium | `utils/native_scaler.py` | Migrate to `torch.amp.GradScaler('cuda')` |
| `torch.ByteStorage` deprecated | Low | `utils/dist.py` | Migrate `all_gather` to `torch.frombuffer` |
| `TubeMaskingGenerator` dimension naming confusion | Low | `data/masking_generator.py` | Rename `height/width/depth` -> `temporal/spatial_h/spatial_w` |
| `_warned_degenerate` fires once only | Low | `core/waveform_losses.py` | Add per-epoch degenerate batch counter |
| `SpatialOutputAdapter.forward` without `init()` guard | Low | `core/output_adapters.py` | Add `assert self.head is not None` |
| Unused `patch_size`/`stride` in `MaskedMSELoss` | Low | `core/criterion.py` | Remove misleading constructor args or implement norm_pix |
| `_pad_time` edge-padding without warning | Low | `data/paired_dataset.py` | Add warning log when padding occurs |
| `MASTER_PORT=29500` fixed | Low | `utils/dist.py` | Set `MASTER_PORT` env var per job in `.sbatch` |
| Silent CPU fallback on `--device cuda` | Low | `runners/_common.py` | Add explicit CUDA availability warning |
""", encoding="utf-8")
print("wrote 05_risks_and_compliance.md")

print(f"\nAll 6 spec files written to {SPEC}/")
for f in sorted(SPEC.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
