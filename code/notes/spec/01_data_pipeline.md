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

**Split policy (Cross-Validation)** — implemented 2026-10-18:
- `split_by='session'` (historical default): splits sorted session list at `train_ratio`. **Risk: subject leakage.**
- `split_by='subject'`: groups by subject prefix; subject-disjoint. Legacy, used only when `n_folds <= 1`.
- **GroupK-Fold Cyclic Rotation** (`utils/splits.py::get_stage_splits`, design `code/notes/design/kfold_val_strategy_design_v2.md`, rollout `code/notes/plan/subject_kfold_plan.md`): one `subject_folds` partition of sorted unique subject ids via `np.array_split`; fold `k` = test, all other folds = train, and Option B carves ~10% of the non-test pool as val with `np.random.default_rng(42 + fold)`. Entered through `--n_folds` / `--fold` / `--use_small_val`; `n_folds <= 1` reproduces the legacy ratio keys exactly.
- **Roles**: `build_dataset(..., split_role='train'|'val'|'test')`. Stage 3 Option A builds no val view — an EMPTY role is refused, because `discover_sessions(subjects=[])` means "every subject".
- `split_by='session'` is rejected when `n_folds > 1` (one subject's sessions could span two roles).
- Stage-2: `pretrain_split='all'` (default) or `'fold'`, which excludes that fold's test subjects from the pretrain corpus and writes `fold_manifest.json` into the run dir.
- Stage-3: Option A (0% Val, fixed-epoch checkpointing, test fold scored at the end) or Option B (~10% Val subset).

**Dev caps** (applied in order): `max_sessions` -> `max_clips` -> `max_entries`.

**Signal normalisation** (`physio_norm`): `'none'` (raw), `'ac'` (remove the clip mean, Stage 3), `'clip'`/`'zscore'` (population std, ddof=0, per clip; the two names are aliases) or `'session'` (population stats over the WHOLE session's `Resp_Volts.txt`). It is THE target-normalisation knob of the pipeline and the DATASET owns it in every ROI lineage (`tir_roi`, `rgb_roi`) in both stages.

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

**Frame source — extracted OR archived** (added 2026-10-28): the JPEGs may be an extracted `2D+3D/<S>/<T>/` directory **or** a per-subject archive `2D+3D/<S>.zip` (BP4D+ ships 140 archives and unpacks nothing). `discover_rgb_sessions` handles both and tags the spec with `source_kind` (`'dir'` | `'zip'`); a subject present both ways uses the directory. An archived task is addressed by a source URI `zip+file://<abs zip>!/<member prefix>` (`zip_source` / `split_zip_source` / `is_zip_source`), which is what `spec['rgb_dir']` / `entry['rgb_dir']` hold, so `os.path.join(entry['rgb_dir'], name)` still names a frame and `_read_rgb` / `_source_size` decode it with `cv2.imdecode` (same pixels as `cv2.imread` on the extracted file). Consequence for runs: `$RAW_DATA_PATH` may point at a zip-only mount, `stats['sessions_from_zip']` records how many sessions were read that way, and the per-process `ZipFile` handle cache is keyed on `os.getpid()` so a forked DataLoader worker reopens instead of sharing a seek offset.

**Builders**: `build_rgb_roi_pretrain_dataset(args)`, `build_rgb_roi_finetune_dataset(is_train, test_mode, args)`.

### 3.3 TIR-ROI + Respiration — `data/tir_resp_dataset.py`

| Class | Stage | Notes |
|-------|-------|-------|
| `BP4DPlusTIRRespDataset` | Base | Nose/mouth ROI from raw TIR frames |
| `TirRoiRespPretrainDataset` | Stage 2 | `{'tir': [C,T,H,W], 'resp': [1,S]}` |
| `TirRoiRespFinetuneDataset` | Stage 3 | `(tir_crop [C,T,H,W], target [output_len])` |

**ROI crop pipeline**: `IRFeatures` landmarks -> `roi_box_from_landmarks` + `roi_padding`. Clips with all-(0,0) sentinel rows dropped. Optional `rail_touch_v` rejects a window that TOUCHED the recorder rail -- i.e. it contains any sample with `abs(x) >= rail_touch_v` volts (the recorder clamps at `+/-10 V`, so the shipped `9.90` drops every window holding a rail-valued target sample: a dead/pinned channel AND a genuinely clipped trough). Optional `min_signal_spread` (shipped `0.1`) then rejects a window whose spread (`max−min`) is below it -- the dead-signal guard for a channel PINNED just inside a clamp without touching it; it is checked second and is complementary to the rail test.

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
