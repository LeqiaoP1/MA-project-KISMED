# ADR-012: 2D-Texture (`rgb2d`) Native Memmap ROI Cache and Multimodal Dataset Alignment

- **Status**: Accepted
- **Date**: 2026-11-10 (v2 contract landed 2026-11-10; codified as ADR-012 on 2026-11-26)
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/plan/features/ADR-012-rgb2d-native-memmap-roi-cache-plan.md`, `code/notes/design/ADR-011-landmark-roi-geometry-and-tir-cache.md` (shared square-ROI & store-native doctrine), `code/notes/design/ADR-013-visual-clip-and-masked-cube-normalization.md` (model-side input norm), `code/notes/spec/01_data_pipeline.md` (§3.5)
- **Supersedes**: `2DTexture_ROI_cache_preprocessing.md` (v2 promoted into ADR structure; v1 prerendered clip-stack design permanently retired)

> **Scope.** This ADR governs one research aspect: **the offline ingestion, native-resolution
> memory-mapped ROI caching, quality gating, and synchronized multimodal dataset join of the BP4D+
> 2D facial texture visible-light stream (`rgb2d`)**. Thermal ROI geometry (`ADR-011`), model-side
> input z-scoring (`ADR-013`), and Stage-2 masking policies (`ADR-006`, `ADR-007`) are governed
> separately.

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope
Visible-light 2D texture frames (`1392 × 1040` RGB, 25 fps) in BP4D+ carry high-definition surface reflectance, micro-expression, and rPPG information complementary to thermal-infrared (`tir`) video. Because `MultiModalMAE` enforces a single global `input_size` across all visual streams (`ADR-014`), the legacy full-frame/64 px `rgb` arm cannot coexist with the `96 × 96` TIR ROI. A dedicated visible stream — **`rgb2d`** — must deliver synchronized facial ROI clips on the exact same spatial-temporal grid (`[B, 3, T, 96, 96]`) and frame window (`frame_start:frame_end`) as `tir`, `resp`, and `bp`.

### 1.2 Design Pressures
1. **On-the-fly decode bottleneck.** An uncompressed 8 s clip (200 frames at `1392 × 1040`) is ~290 MB; decoding raw JPEG sequences or zipped archives (`2D+3D/<S>.zip`) on the fly during distributed training severely bottlenecks GPUs.
2. **Why v1 prerendered `.npz` clip stacks failed.** Baking `input_size=96`, `clip_stride`, and selection thresholds into `.npz` clip stacks duplicates overlapping frames ~5.5×, forces full-corpus rebuilds on every threshold/hop change, and prevents memory-mapping (a materializing reader would re-allocate ~1.37 GB per session under DataLoader shuffling).
3. **Modality-specific edge cases.** Unlike `IRFeatures` (`(0, 0)` sentinels), lost `2DFeatures` frames appear as empty `[]` / `NaN` records; `2D+3D` archives must never be extracted to disk; and subjects leaning near frame edges require explicit black padding with a two-level out-of-frame guard.
4. **Modality-relative sway scale.** Measured across 1,375 paired sessions (`code/notes/eval/rgb2d/tir_rgb2d_sway_ratio.py`), 2D facial landmarks report **3.2×** the pixel excursion of TIR for identical head motion; copying TIR's 40 px gate would drop 99% of sessions.

### 1.3 Requirements
- **R1 — Distinct canonical stream name (`rgb2d`).** Register `rgb2d` separately from `rgb` so checkpoints can never pair the wrong front-end.
- **R2 — Store native, rescale once at load (`ADR-011` parity).** Store native-resolution RGB8 square-hull crops per session (`store_resolution='native'`), performing a single isotropic `cv2.resize` (`INTER_AREA` downscale / `INTER_LINEAR` upscale) to `input_size` at load time.
- **R3 — Zero-copy memory-mapped binary storage.** Store frames and landmarks in headerless raw binary files (`.dat`, `.lm.dat`) paired with a `.json` commit marker, handing out read-only `np.memmap` views with deterministic file-descriptor cleanup.
- **R4 — Two-level out-of-frame & zero-tolerance tracking gates.** Enforce black-padded square crops with a 50% in-frame area threshold at both task build time and clip load time, plus 100% finite 49-point tracking (`min_valid_ratio = 1.0`) and a calibrated load-time sway gate (`rgb2d_max_sway_px = 110`).
- **R5 — Strict window alignment.** Multimodal datasets (`trimodal`, `fourstream`) post-filter the TIR+RESP entry list so every modality is sliced from the exact same `(frame_start, frame_end)` window.

---

## 2. Decision & Parameter Exposure

**Build a task-level, native-resolution, raw memory-mapped `rgb2d` ROI cache (`CACHE_FORMAT_VERSION = 1`, prefix `rgb2droi`) from 49-point `2DFeatures` tracks with black-padded square window hulls and JSON commit markers; apply all clip selection gates at dataset load time and perform the single isotropic rescale to `input_size` per clip.**

### 2.1 Geometry, Coordinate Frame & Padding Rules
1. **Landmarks & box:** All 49 1-indexed facial landmarks (`face = 1..49`), `crop_rule = 'square'`, `roi_padding = 0.10` (per-side expansion $\Rightarrow 1.2\times$ longer axis). Unclamped square spans (`square_span_2d`) may extend beyond `[0, 1392) × [0, 1040)`.
2. **Task-level square hull (`hull_box_2d`):** Union over every finite-tracked stride-1 `clip_seconds` window's unclamped square span, expanded to a square. Missing border pixels are filled with black (`cv2.BORDER_CONSTANT, value=0`) and converted `BGR → RGB` before writing.
3. **Binding coordinate-frame invariant:** The stored array's origin **is** the unclamped hull origin `(hx0, hy0)`, black bars included:
   $$\text{stored\_index} = \text{source\_coordinate} - \text{hull\_origin} \qquad (\text{NEVER add } \texttt{pad}).$$
   `pad = [top, bottom, left, right]` is strictly descriptive. Any clip box falling outside `[0, hull_w) × [0, hull_h)` raises `IndexError` (invariant **I2**).
4. **Two-level out-of-frame gate (`in_frame_share >= 0.50`):**
   - *Build time*: Tasks with hull `in_frame_share < 0.50` are skipped (`box_out_of_frame`).
   - *Load time*: Clips whose own `square_span_2d` has `in_frame_share < rgb2d_area_min_frac` (default `0.50`) are dropped (`clips_dropped_2d_envelope`), computed solely from `.lm.dat` without touching pixel pages.

### 2.2 Binary Memmap Storage Layout & Cache Identity
Under `<out_root>/<CACHE_KEY>/tasks/`:
- `<S>_<T>.dat` — raw headerless `uint8` array `[T, H, W, 3]` (RGB8, `H == W`, native hull pixels, page-aligned offset 0).
- `<S>_<T>.lm.dat` — raw headerless `float32` array `[T, 49, 2]` (per-frame landmark track; ~0.6 MB/session, enabling pixel-free clip gating).
- `<S>_<T>.json` — atomic commit marker written **last** after flushing/closing both memmaps; carries `session`, `source`, `n_frames`, `box`, `pad`, `src_hw`, `hull_hw`, `store_resolution='native'`, `area_in_frac`, and `{shape, dtype, nbytes}` for integrity verification (`--status` detects `incomplete` and `truncated` shards).
- **Frame ingestion (`source_policy='dir-then-zip'`):** Reads extracted directories when present, otherwise decodes members of `2D+3D/<S>.zip` in memory (never extracts to disk). Unreadable in-flight archives during corpus copy are skipped transiently without writing permanent skip markers.
- **FD lifecycle:** `ShardHandle` (context manager, LRU-bounded via `rgb2d_cache_lru=8`) drops memmap references on `close()` without calling `._mmap.close()` directly; per-frame `np.ascontiguousarray` slices ensure no view outlives `__getitem__`.
- **Cache key (`COMPARE_KEYS`):** Includes `format_version=1`, `corpus`, `roi_landmarks`, `roi_padding`, `crop_rule`, `box_rule='window-hull'`, `clip_seconds`, `fps`, `valid_frame_rule='finite-excluded'`, `pixel_format='rgb8'`, `store_resolution='native'`, `source_policy`. Excludes `input_size`, `clip_stride`, `rgb2d_max_sway_px`, `rgb2d_min_valid_ratio`, `rgb2d_area_min_frac`, `task_set`, and `temporal_stride`.

### 2.3 Dataset Intersection & Load-Time Filter Order
Multimodal pre-training datasets (`TirRoiResp2DPretrainDataset`, `TirRoiResp2DBPPretrainDataset`) intersect session coverage with the TIR/RESP parent (raising `FileNotFoundError` on missing unskipped shards) and filter parent entries in fixed attribution order:
1. `frame_end > n_2d` → `clips_dropped_2d_short`
2. Invalid-row fraction `> 1 - rgb2d_min_valid_ratio` → `clips_dropped_2d_invalid`
3. `in_frame_share(clip_box) < rgb2d_area_min_frac` → `clips_dropped_2d_envelope`
4. `landmark_sway > rgb2d_max_sway_px` → `clips_dropped_2d_sway`

Per-clip `__getitem__` slices each frame's `square_span_2d` box out of the native hull, resizes once to `input_size` (`INTER_AREA` if $s \le \ell$ else `INTER_LINEAR`), outputs `[3, T, s, s]` float32 in `[0, 1]`, and applies the exact same `temporal_stride` decimation as `tir`.

### 2.4 Logical Parameter Surface

| Parameter | Type | Default | Constraints | Role |
|---|---|---|---|---|
| `rgb2d_cache` | path | env `RGB2D_CACHE` | valid v1 cache root | Offline native `rgb2d` cache location |
| `rgb2d_cache_lru` | int | `8` | `>= 1` | Max open `ShardHandle`s per worker (~2 fds each) |
| `rgb2d_landmarks` | str | `face` (`1..49`) | 1-indexed 49-pt subset | Landmark set for box & sway; in cache key |
| `rgb2d_min_valid_ratio` | float | `1.0` | `(0.0, 1.0]` | Required finite-landmark frame share per clip (load-time) |
| `rgb2d_max_sway_px` | float | `110.0` | `>= 0` (`0` = off) | Load-time centroid sway gate calibrated to TIR 40 px |
| `rgb2d_area_min_frac` | float | `0.5` | `[0.0, 1.0]` | Minimum in-frame share of clip box (load-time) |
| `source_policy` | str | `'dir-then-zip'` | `'dir-then-zip'` | Builder frame ingestion rule; in cache key |

### 2.5 Rejected Alternatives
- **v1 prerendered `(N_clips, 200, 96, 96, 3)` clip stacks**: Rejected because it bakes `input_size`, `clip_stride`, and selection thresholds into disk storage and duplicates overlapping frames ~5.5×.
- **`.npz` / `.npy` shard format**: `.npz` cannot be memory-mapped without materializing multi-hundred-MB arrays; `.npy` duplicates shape/dtype headers outside the JSON commit marker.
- **Unmeasured 15 px sway threshold (v1 draft)**: Rejected after Phase-0 measurement (`code/notes/eval/rgb2d/sway_retention.md`) showed 15 px keeps only 0.9% of windows; calibrated to 110 px via paired TIR/RGB2D ratio.
- **Reusing TIR's `(0, 0)` sentinel predicate**: Rejected because `2DFeatures` encodes lost frames as empty/NaN fields (`valid_frame_rule='finite-excluded'`).

---

## 3. Falsifiable Criteria

- **Criterion 1 (Cache key & view semantics)**: Unit tests confirm cache key sensitivity (changes on `store_resolution`/`roi_padding`/`crop_rule`/`clip_seconds`; invariant to `input_size`/`clip_stride`/`max_sway_px`), zero-copy memmap views (`s.base is not None`, `s.flags.owndata is False`, `np.shares_memory`), exact coordinate-frame mapping without `+pad`, and `IndexError` on hull escape.
- **Criterion 2 (One-resample parity across `input_size`)**: Cropping and resizing from a native hull reproduces `cv2.resize(crop(box), (s, s), INTER_AREA)` identically for $s \in \{64, 96, 112\}$.
- **Criterion 3 (Corpus build & `--check` parity)**: `run_build_2d_roi_cache.py --status` reports zero `incomplete`/`truncated` shards and `--check N` passes chunk-wise `np.array_equal` verification.
- **Criterion 4 (Multimodal alignment)**: Trimodal/fourstream datasets yield synchronized `{'tir': [3,T,96,96], 'rgb2d': [3,T,96,96], 'resp': [1,L]}` tensors passing `MultiModalMAE._check_space_time_alignment` and loudly rejecting mismatched spatial grids.

- **Acceptance Artifacts**: `tests/test_rgb2d_cache.py`, `tests/test_trimodal_dataset.py`, `tests/test_multimae_rgb2d.py`, `code/notes/eval/rgb2d/build_preflight.md`, `code/notes/eval/rgb2d/sway_retention.md`.

---

## 4. Consequences & Trade-offs

- **Positive**: Enables visible+thermal fusion (`tir` + `rgb2d`) on one exact patch grid (`96 × 96`) with zero zip extraction, near-zero DataLoader RAM overhead via OS page-cache memmaps, and free load-time tuning of `input_size`, `clip_stride`, and quality gates.
- **Negative / Technical Debt**:
  - Task-wide native square hulls average `621–1036 px` (~1.37 GB/session, **~2.25 TB on disk** across 1,390 sessions, ~6% of the 40 TB HPC scratch quota) — accepted as the disk price for native load-time tight cropping.
  - Subject `F037` initial lead-frame discrepancy between 2D and thermal indices remains a documented open item (index alignment assumed as in existing arms).
  - Scope is Stage-2 pre-training (`trimodal` / `fourstream`) only; no Stage-3 `rgb2d` regressor runner is currently wired.

---

## 5. Reproducibility Record

- **Run / Job IDs**: Implemented & verified 2026-11-10 (Phases 0–6, 592 suite tests passing); corpus built on Lichtenberg HPC via `scripts/hpc/submit_2d_roi_cache.sbatch` (`--array=0-31`).
- **Evidence Artifacts**: `code/notes/eval/rgb2d/sway_retention.md`, `code/notes/eval/rgb2d/tir_rgb2d_sway_ratio.json`, `code/notes/eval/rgb2d/build_preflight.md`, `code/notes/eval/baselines/tir_rgb2d_resp/`.
- **Tests / Smoke Checks**: `code/tests/test_rgb2d_cache.py`, `code/tests/test_trimodal_dataset.py`, `code/tests/test_multimae_rgb2d.py`.
- **Rollback / Rejection Condition**: Omit `rgb2d` from `--streams` and select `data_set: tir_roi` (`ADR-007` two-stream arm); existing TIR/RESP shards and caches are completely untouched.
