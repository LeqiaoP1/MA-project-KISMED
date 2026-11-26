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

**Split policy**:
- `split_by='session'` (default): splits sorted session list at `train_ratio`. **Risk: subject leakage.**
- `split_by='subject'`: groups by subject prefix; subject-disjoint. **Recommended for evaluation.**
- Stage-2 (`PairedPretrainDataset`): always `train_ratio=1.0` (pretrain on all).

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

**Builders**: `build_rgb_roi_pretrain_dataset(args)`, `build_rgb_roi_finetune_dataset(is_train, test_mode, args)`.

**Role (S1 terminology, 2026-11-17).** This lineage emits the `rgb` stream and is **not deprecated**: it is the only visible stream reachable from `PairedSessionDataset` (`data_set: bp4d+`), the AU probe and Stage 3. It is, however, the BP-arm Stage-2 source only, and it can never be fused with `tir` (the model has ONE global `input_size`), so any visible+thermal run uses `rgb2d` instead (§3.5).

### 3.3 TIR-ROI + Respiration — `data/tir_resp_dataset.py`

| Class | Stage | Notes |
|-------|-------|-------|
| `BP4DPlusTIRRespDataset` | Base | Nose/mouth ROI from raw TIR frames |
| `TirRoiRespPretrainDataset` | Stage 2 | `{'tir': [C,T,H,W], 'resp': [1,S]}` |
| `TirRoiRespFinetuneDataset` | Stage 3 | `(tir_crop [C,T,H,W], target [output_len])` |

**ROI crop pipeline**: `IRFeatures` landmarks -> `roi_box_from_landmarks(..., crop_rule)` + `roi_padding`. With `crop_rule: square` (the default) the padded box is squared about its centre FIRST, so the rescale applies ONE scale factor (`sx == sy`) and the anatomy is never deformed; `crop_rule: box` keeps the legacy per-axis geometry for A/B. A box that cannot be placed without cutting the landmark envelope raises `RoiBoxOutOfFrame`. Clips whose target-landmark centroid travels further than `max_sway_px` (40 source px; 0 = off) inside the window are DROPPED (counted as `clips_dropped_motion`), and clips touching a (0,0) sentinel row are dropped whole.

**Label validity + the respiration cache (2026-10-31).** The TIR-ROI label path decides validity ONCE on the RAW signal and can read the label from an offline artefact (`data/resp_preprocess.py`, `data/resp_cache.py`, built by `runners/run_build_resp_cache.py`). A sample is invalid when it sits in a saturation EVENT (`abs(V) > 9.95` lasting `>= 1 s`), in ANY `5 s` window with `ptp < 0.3 V`, or in a valid run shorter than `8 s`. The artefact stores the RUN-MEAN-REMOVED signal on a fixed `fs_out == --fs` grid with `NaN` exactly where the mask is False; with `--resp_cache` a window is usable iff `mask[s:s+n].all()` and nothing re-tests its content. `rail_touch_v` / `min_signal_spread` are the **LEGACY raw path** knobs and are refused (set to 0) when a cache is attached, so content is never filtered twice. Contract: `design/ADR-003-resp-session-cache-artefact.md`; the artefact's own build accounting and source-rate evidence: `eval/resp/resp_artefact_build.md` (the artefact is modality-independent; `eval/tir_resp/resp_preprocess_impact.md` keeps only the TIR dataset's clip accounting). The window rule itself lives in `data/resp_window.py` and NOWHERE else (`window_indices`, `usable`, `read_window`, `window_cause`, `assert_clip_fits`, `verify_identity`), so the dataset and the evaluator cannot drift apart. Two build-time constraints are refused at `--init`: `fs_out` must DIVIDE `fs_in` (an integral decimation factor is what lands every run on exact master-grid instants, so no run is placed with a sub-sample phase error), and `--resample_taps_per_phase` must be `>= 10` (the run-edge pad and the discarded output count are derived from the filter's real support — 100 source samples at 1000 -> 100 Hz). A hole NARROWER than one output period is widened to cover at least one grid instant; the mask is never dilated, so a run bounded by wider holes keeps every sample and an exactly 8 s run still yields 800 usable samples.

**Task groups** (`data/task_groups.py`): the shipped tiers (v2, 2026-10-24) are `low=[T2,T3,T5]`, `moderate=[T1,T4]`, `high=[T6,T7,T8,T9,T10]` -- a PARTITION of `T1`..`T10`, canonical form `data.task_groups.SHIPPED_TASK_GROUPS`, pinned by `tests/test_task_group_configs.py`. Stage-2 and Stage-3 must use **identical** `task_groups` / `task_set`.

**Builders**: `build_tir_roi_pretrain_dataset(args)`, `build_tir_roi_finetune_dataset(is_train, test_mode, args)`.

### 3.4 AU Dataset — `data/au_dataset.py`

Per-frame AU occurrence labels; subject-disjoint split. Used exclusively by the AU probe.

### 3.5 2D-Texture ROI stream (`rgb2d`) — cached (2026-11-10, SHIPPED)

A THIRD cached stream so Stage 2 can run `tir` + `rgb2d` + `resp` on ONE clip window. Contract: `design/ADR-012-rgb2d-native-memmap-roi-cache.md` (v2). Build order: `plan/features/ADR-012-rgb2d-native-memmap-roi-cache-plan.md`. Nothing in the existing TIR/RESP artefact changes value.

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

### 3.6 Four-stream Stage-2 (`tir` + `rgb2d` + `bp` + `resp`) — 2026-11-21

Adds the 1000 Hz BP waveform to the tri-modal window. Contract: `data/fourstream_dataset.py`; cache contract `data/bp_artefact.py` + `data/bp_windows.py`; preflight `notes/eval/baselines/tir_rgb2d_bp_resp/cache_preflight.md`.

| Piece | Name |
|---|---|
| Dataset | `data/fourstream_dataset.py::TirRoiResp2DBPPretrainDataset` (subclasses `TirRoiResp2DPretrainDataset`) |
| `data_set` | `tir_roi_resp_rgb2d_bp` — **Stage-2 only** (`build_dataset` refuses it) |
| Streams | `streams: tir,rgb2d,bp,resp`; two visuals + two physiologies |
| Builder | `build_tir_roi_rgb2d_bp_pretrain_dataset(args)` |
| Config | `configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml` |

**Window**: the parent's `t_start`/`frame_start` is used unchanged. The BP window is `s = round(t_start*1000)`, `n = round(kept_seconds*1000) = seq_len`; a clip survives only when `bp_windows.usable(mask, s, n)` is true (the whole 8 s lies in one valid segment). The surviving `bp` window is z-scored per clip with the same `_normalize` used by `resp` (`ddof=0`). Output: `tir [B,3,T,96,96]`, `rgb2d [B,3,T,96,96]`, `resp [B,1,8000]`, `bp [B,1,8000]`.

**BP artefact**: `runners/run_build_bp_artefact.py`; identity `bpsession_src1000_range30-250_flat200ms_rail30ms_seg4s_v1_<hash>`; `fs_in == fs_out == 1000`, no resampling. `bp_artefact` is the OUT_ROOT (parent of the keyed directory). A `skips/` marker is a legitimate exclusion and drops the clip; a missing shard without a marker is an incomplete build and raises. Runner flags: `--bp_artefact`, `--bp_fs` (must equal `--fs`), `--bp_cache_lru`.

**Geometry**: 8 s, 100 frames after `temporal_stride=2`, `tubelet=4,16,16`, `input_size=96`, `fs=1000`, `sig_kernel=32`; 36 spatial tubes x 25 tokens per visual stream, 250 tokens per physio; `n_budget=350`, `video_cap_ratio=0.25`, `physio_cap_ratio=0.50`, `video_min_tubes=2`.

**Initialisation**: the shipped arm uses `project_multimae_small` + `pretrained_encoder: none` (explicit random init; no small Stage-1 checkpoint is published).

---

## 4. Dataset Dispatcher — `data/datasets.py`

`build_dataset(is_train, test_mode, args)` dispatches on `args.data_set`:

| `data_set` value | Dataset class |
|-----------------|---------------|
| `'bp4d+'`, `'paired'` | `PairedSessionDataset` |
| `'tir_roi'`, `'tir_roi_resp'`, `'tir-roi'` | `TirRoiRespFinetuneDataset` |
| `'rgb_roi'`, `'rgb_roi_resp'`, `'rgb-roi'` | `RGBRoiFinetuneDataset` |
| `'tir_roi_resp_rgb2d'`, `'trimodal'` | Stage-2 only: `TirRoiResp2DPretrainDataset` (§3.5); `build_dataset` has no fine-tune view |
| `'tir_roi_resp_rgb2d_bp'` | Stage-2 only: `TirRoiResp2DBPPretrainDataset` (§3.6); `build_dataset` has no fine-tune view |
| custom registered | `@register_dataset` decorator |

`build_pretraining_dataset(args)` dispatches similarly for Stage-2: `TIR_ROI_DATA_SETS` -> `build_tir_roi_pretrain_dataset`, `RGB_ROI_DATA_SETS` -> `build_rgb_roi_pretrain_dataset`, `TIR_ROI_RGB2D_DATA_SETS` -> `build_tir_roi_rgb2d_pretrain_dataset` (in `data/trimodal_dataset.py`; it cannot reuse the TIR builder, which forwards `args.streams` into a class that refuses anything outside `('tir','resp')`), and `FOURSTREAM_DATA_SETS` -> `build_tir_roi_rgb2d_bp_pretrain_dataset` (in `data/fourstream_dataset.py`).

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
| Stage 2 (`tir_roi_resp_rgb2d_bp`, §3.6) | `{stream: Tensor}` | `{'tir': [B,3,T,96,96], 'rgb2d': [B,3,T,96,96], 'bp': [B,1,8000], 'resp': [B,1,8000]}` — **one shared window** for all four; visuals report `T=100` frames and `6x6=36` patches/frame |
| Stage 3 | `(Tensor, Tensor)` | `([B,C,T,H,W], [B,output_len])` |
| AU probe | `(Tensor, Tensor)` | `([B,C,T,H,W], [B,n_aus])` |

Masking is applied **inside** `MultiModalMAE.forward`, not in the dataset or collate function.
