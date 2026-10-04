# 01 — Data Pipeline

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

**ROI crop pipeline**: `IRFeatures` landmarks -> `roi_box_from_landmarks` + `roi_padding`. Clips with all-(0,0) sentinel rows dropped. Optional `min_signal_spread` rejects near-constant (railed) respiration windows.

**Task groups** (`data/task_groups.py`): e.g. `low=[T2,T3]`, `moderate=[T4,T7,T8,T10]`, `high=[T1,T5,T6,T9]`. Stage-2 and Stage-3 must use **identical** `task_groups` / `task_set`.

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
