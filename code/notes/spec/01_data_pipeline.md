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
      Physiology/     signals CSV: [time, bp, resp]
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
  signals.csv        header: [time, bp, resp] at fs Hz
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
- **GroupK-Fold Cyclic Rotation** (`utils/splits.py::get_stage_splits`, design `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md`, rollout `code/notes/plan/ADR-001-subject-kfold-plan.md`): one `subject_folds` partition of sorted unique subject ids via `np.array_split`; fold `k` = test, all other folds = train, and Option B carves ~10% of the non-test pool as val with `np.random.default_rng(42 + fold)`. Entered through `--n_folds` / `--fold` / `--use_small_val`; `n_folds <= 1` reproduces the legacy ratio keys exactly.
- **Partition rule** (`--split_seed` / `--split_stratify`, added 2026-11-01): `split_seed=None` + `split_stratify='none'` is the historical sorted-contiguous cut, which on BP4D+ makes fold == sex block (`F001..` sorts before `M001..`). A seed with `split_stratify='sex'` shuffles within each sex stratum and interleaves the strata proportionally, so every fold is a near-representative sample while the `array_split` fold SIZES are unchanged. Requires the seed; both stages of one sweep must use the same pair, and `fold_manifest.json` records it.
- **Roles**: `build_dataset(..., split_role='train'|'val'|'test')`. Stage 3 Option A builds no val view — an EMPTY role is refused, because `discover_sessions(subjects=[])` means "every subject".
- `split_by='session'` is rejected when `n_folds > 1` (one subject's sessions could span two roles).
- Stage-2: `pretrain_split='all'` (default) or `'fold'`, which excludes that fold's test subjects from the pretrain corpus and writes `fold_manifest.json` into the run dir.
- Stage-3: Option A (0% Val, fixed-epoch checkpointing, test fold scored at the end) or Option B (~10% Val subset).

**Dev caps** (applied in order): `max_sessions` -> `max_clips` -> `max_entries`.

**Signal normalisation** (`physio_norm`): a **per-clip z-score** — one mean/std from the clip's own samples (population std, `ddof=0`, eps `ZSCORE_EPS`). That is the ONLY label normalisation of the pipeline, applied by the DATASET in every ROI lineage (`tir_roi`, `rgb_roi`) in both stages (`data.tir_resp_dataset.TARGET_NORM`); the model is pinned to `target_norm='none'`. `'clip'` and `'zscore'` are synonyms, and the retired `'none'` / `'ac'` / `'session'` values are refused by `check_physio_norm` rather than silently mapped.

**`PairedPretrainDataset.__getitem__` output shapes**:

```python
{
  'rgb':  FloatTensor [3, T, H, W],
  'tir':  FloatTensor [C_tir, T, H, W],  # C_tir = 3 (default) or 1
  'bp':   FloatTensor [1, seq_len],
  'resp': FloatTensor [1, seq_len],
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

**Role (S1 terminology, 2026-11-17).** This lineage emits the `rgb` stream and is **not deprecated**: it is the only visible stream reachable from `PairedSessionDataset` (`data_set: bp4d+`), the AU probe and Stage 3. It is, however, the BP-arm Stage-2 source only, and it can never be fused with `tir` (the model has ONE global `input_size`), so any visible+thermal run uses `rgb2d` instead (§3.5).

### 3.3 TIR-ROI + Respiration — `data/tir_resp_dataset.py`

| Class | Stage | Notes |
|-------|-------|-------|
| `BP4DPlusTIRRespDataset` | Base | Nose/mouth ROI from raw TIR frames |
| `TirRoiRespPretrainDataset` | Stage 2 | `{'tir': [C,T,H,W], 'resp': [1,S]}` |
| `TirRoiRespFinetuneDataset` | Stage 3 | `(tir_crop [C,T,H,W], target [output_len])` |

**ROI crop pipeline**: `IRFeatures` landmarks -> `roi_box_from_landmarks(..., crop_rule)` + `roi_padding`. With `crop_rule: square` (the default) the padded box is squared about its centre FIRST, so the rescale applies ONE scale factor (`sx == sy`) and the anatomy is never deformed; `crop_rule: box` keeps the legacy per-axis geometry for A/B. A box that cannot be placed without cutting the landmark envelope raises `RoiBoxOutOfFrame`. Clips whose target-landmark centroid travels further than `max_sway_px` (40 source px; 0 = off) inside the window are DROPPED (counted as `clips_dropped_motion`), and clips touching a (0,0) sentinel row are dropped whole.

**Label validity + the respiration cache (2026-10-31).** The TIR-ROI label path decides validity ONCE on the RAW signal and can read the label from an offline artefact (`data/resp_preprocess.py`, `data/resp_cache.py`, built by `runners/run_build_resp_cache.py`). A sample is invalid when it sits in a saturation EVENT (`\|V\| > 9.95` lasting `>= 1 s`), in ANY `5 s` window with `ptp < 0.3 V`, or in a valid run shorter than `8 s` (the last three are the cache key's `sat_v`/`sat_min_s`, `flat_window_s`/`flat_ptp_v`, `min_run_s`). The artefact stores the RUN-MEAN-REMOVED signal on a fixed `fs_out == --fs` grid with `NaN` exactly where the mask is False; with `--resp_cache` a window is usable iff `mask[s:s+n].all()` and nothing re-tests its content. The two window rules below are the **LEGACY raw path** and are refused (set to 0) when a cache is attached, so content is never filtered twice. Contract: `design/RESP_session_preprocess.md`; measured impact: `eval/tir_resp/resp_preprocess_impact.md`.

**Legacy window rules (raw `Resp_Volts.txt` path only).** `rail_touch_v` rejects a window that TOUCHED the recorder rail -- i.e. it contains any sample with `abs(x) >= rail_touch_v` volts (the recorder clamps at `+/-10 V`, so the shipped `9.90` drops every window holding a rail-valued target sample: a dead/pinned channel AND a genuinely clipped trough). `min_signal_spread` (shipped `0.1`) then rejects a window whose spread (`max−min`) is below it -- the dead-signal guard for a channel PINNED just inside a clamp without touching it; it is checked second and is complementary to the rail test.

**Task groups** (`data/task_groups.py`): the shipped tiers (v2, 2026-10-24) are `low=[T2,T3,T5]`, `moderate=[T1,T4]`, `high=[T6,T7,T8,T9,T10]` -- a PARTITION of `T1`..`T10`, canonical form `data.task_groups.SHIPPED_TASK_GROUPS`, pinned by `tests/test_task_group_configs.py`. Stage-2 and Stage-3 must use **identical** `task_groups` / `task_set`.

**Builders**: `build_tir_roi_pretrain_dataset(args)`, `build_tir_roi_finetune_dataset(is_train, test_mode, args)`.

### 3.4 AU Dataset — `data/au_dataset.py`

Per-frame AU occurrence labels; subject-disjoint split. Used exclusively by the AU probe.

### 3.5 2D-Texture ROI stream (`rgb2d`) — cached (2026-11-10, SHIPPED)

A THIRD cached stream so Stage 2 can run `tir` + `rgb2d` + `resp` on ONE clip window. Contract: `design/2DTexture_ROI_cache_preprocessing.md` (v2). Build order: `plan/2DTexture_ROI_cache_implementation_plan.md`. Nothing in the existing TIR/RESP artefact changes value.

| Piece | Name |
|---|---|
| Cache module | `data/rgb2d_cache.py` (an additive `keys=` parameter on `roi_cache.find_cache` is its only cross-module dependency) |
| Builder | `runners/run_build_2d_roi_cache.py` (`--init/--build/--status/--index/--check N`, array via `--array_index`/`--array_count`) |
| Dataset | `data/trimodal_dataset.py::TirRoiResp2DPretrainDataset` (subclasses `TirRoiRespPretrainDataset`) |
| `data_set` | `tir_roi_resp_rgb2d`, `trimodal` — **Stage-2 only** |
| Streams | `streams: tir,rgb2d,resp`; `_VISUAL_STREAMS = ('rgb','tir','rgb2d')` in `core/multimae.py` (+ `STREAM_CHANNELS['rgb2d'] = 3`) |

**Artefact** (`<out_root>/<CACHE_KEY>/tasks/<S>_<T>.{dat,lm.dat,json}`): hull pixels `uint8 [T, H, W, 3]` RGB at **native** resolution, the 49-point track `float32 [T, 49, 2]`, and a `.json` of scalars (`box`, `pad`, `src_hw`, `hull_hw`, shapes, byte total) written LAST as the commit marker. Raw `np.memmap`, offset 0, **no `.npy` header** — `shape`/`dtype` come from the `.json`; a `.dat` without its `.json` is `incomplete`, a size mismatch is `truncated`. ~0.19 MB/frame at a ~250 px hull (~0.4 TB per corpus, ~23x smaller than the raw 1392x1040 JPEGs).

**Cache key**: `rgb2droi_<landmark_set>_pad<p>_<crop_rule>_window-hull_clip<c>s_src<W>x<H>_native_v1_<10hex>`. IN: `store_resolution='native'`, `roi_padding`, `crop_rule`, `clip_seconds`, `fps`, `src_w/h`, landmark set, `valid_frame_rule='finite-excluded'`, `source_policy`. OUT: `input_size`, `clip_stride`, `temporal_stride`, `max_sway_px`, `min_valid_ratio`, the resize kernel — the TIR doctrine (*store native, rescale at load*) makes them all load-time knobs, so one cache serves every hop, fold and `input_size` (64/96/112).

**Gates are DATASET-level**, counted per rule: `clips_dropped_2d_short` (window past `n_frames`), `_invalid` (`min_valid_ratio`, default 1.0, over finite target landmarks), `_envelope` (the clip's own box in-frame share `< area_min_frac`, default 0.5), `_sway` (`max_sway_px`, measured in Phase 0 — v1's 15 px is NOT adopted unmeasured). The builder skips only geometry-fatal sessions: `missing_2d_features`, `invalid_2d_features`, `image_count_mismatch` (`n_jpg != n_mat`), `missing_image_seq`, `undecodable_images`, `box_out_of_frame` (hull in-frame share `< 0.5`), `task_too_large`.

**Load path**: `open_shard` returns a `ShardHandle` over two `np.memmap(mode='r')` mappings (context manager; `close()` drops the references, since `mmap.close()` behind a live numpy view raises `BufferError`). `__getitem__` computes the clip's own square box, maps it hull-local, asserts containment (I2 — `IndexError`, never clamp/wrap) and resizes ONCE to the shared `input_size` (`INTER_AREA` downscaling, `INTER_LINEAR` otherwise). `input_size` must equal the `tir` value because the model has ONE patch grid; 96 in the shipped config yields 6x6 = 36 patches per frame for both visual streams.

**Frame source**: extracted `2D+3D/<S>/<T>/*.jpg` when present, else the members of `2D+3D/<S>.zip` decoded in memory via `cv2.imdecode` — `2D+3D` is **never extracted to disk**, so the cache is the only on-disk form of this stream.

---

## 4. Dataset Dispatcher — `data/datasets.py`

`build_dataset(is_train, test_mode, args)` dispatches on `args.data_set`:

| `data_set` value | Dataset class |
|-----------------|---------------|
| `'bp4d+'`, `'paired'` | `PairedSessionDataset` |
| `'tir_roi'`, `'tir_roi_resp'`, `'tir-roi'` | `TirRoiRespFinetuneDataset` |
| `'rgb_roi'`, `'rgb_roi_resp'`, `'rgb-roi'` | `RGBRoiFinetuneDataset` |
| `'tir_roi_resp_rgb2d'`, `'trimodal'` | Stage-2 only: `TirRoiResp2DPretrainDataset` (§3.5); `build_dataset` has no fine-tune view |
| custom registered | `@register_dataset` decorator |

`build_pretraining_dataset(args)` dispatches similarly for Stage-2: `TIR_ROI_DATA_SETS` -> `build_tir_roi_pretrain_dataset`, `RGB_ROI_DATA_SETS` -> `build_rgb_roi_pretrain_dataset`, and `TIR_ROI_RGB2D_DATA_SETS` -> `build_tir_roi_rgb2d_pretrain_dataset` (in `data/trimodal_dataset.py`; it cannot reuse the TIR builder, which forwards `args.streams` into a class that refuses anything outside `('tir','resp')`).

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

**The `rgb2d` stream does NOT go through `alignment.py`** (§3.5): it assumes jpg index == thermal frame index == `IRFeatures` line index, exactly as the TIR and `rgb_roi` arms already do. The F037 lead-frame case (design §7) is an open item, not a solved problem — do not assume this stream is trigger-aligned.

---

## 6. Masking Generators — `data/masking_generator.py`

All return `torch.LongTensor [num_patches]`, `1` = masked (to reconstruct), `0` = visible.

| Class | Strategy | Default use |
|-------|-----------|-------------|
| `RandomMaskingGenerator(input_size, mask_ratio)` | Uniform random | 1-D signals (mask_ratio >= 0.90) |
| `TubeMaskingGenerator(input_size, mask_ratio)` | Same spatial patches across all frames | Video (mask_ratio 0.50–0.90) |
| `MultiModalMaskingGenerator(stream_generators)` | Compose one per stream | Stage-2 |

Default Stage-2 mask ratios (from configs): RGB 0.90, TIR 0.90, BP/RESP 0.90.

> At `input_size=64`, `patch_size=16`: 4x4=16 spatial patches. Tube mask 0.90 keeps 1–2 visible per frame — <=12.5% of spatial context.

---

## 7. Signal Processing

`_load_signals_csv` infers `fs` from median `diff(t)` of the time column. `nan_to_num` applied to all columns.

Column aliases: `bp <- (bp, ppg, pulse)`, `resp <- (resp, respiration)`.

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
| Stage 2 (`tir_roi_resp_rgb2d`, §3.5) | `{stream: Tensor}` | `{'tir': [B,3,T,96,96], 'rgb2d': [B,3,T,96,96], 'resp': [B,1,S]}` — **one shared window** for all three; both visual streams report `T` frames and `6x6 = 36` patches/frame |
| Stage 3 | `(Tensor, Tensor)` | `([B,C,T,H,W], [B,output_len])` |
| AU probe | `(Tensor, Tensor)` | `([B,C,T,H,W], [B,n_aus])` |

Masking is applied **inside** `MultiModalMAE.forward`, not in the dataset or collate function.
