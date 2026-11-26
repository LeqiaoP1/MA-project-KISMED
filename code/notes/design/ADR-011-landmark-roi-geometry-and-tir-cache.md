# ADR-011: Landmark-Guided Square ROI Geometry, Motion Gating, and Native TIR Hull Cache

- **Status**: Accepted
- **Date**: 2026-10-29 (contract updated to `roi_padding=0.10` / `input_size=96` on 2026-10-30; codified as ADR-011 on 2026-11-26)
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/plan/features/ADR-011-landmark-roi-geometry-and-tir-cache-plan.md`, `code/notes/spec/01_data_pipeline.md` (§3.2–§3.3), `code/notes/design/ADR-012-rgb2d-native-memmap-roi-cache.md` (inherits this geometry doctrine for `rgb2d`), `code/notes/design/ADR-013-visual-clip-and-masked-cube-normalization.md` (model-side input norm)
- **Supersedes**: `ROI_crop_and_rescaling.md` (promoted into ADR structure; retired legacy `crop_rule='box'` and `roi_padding=0.20 / input_size=112` lineage)

> **Scope.** This ADR governs one research aspect: **how facial Regions of Interest (ROIs) are
> derived from tracked landmarks, gated for excessive head motion, rescaled isotropically to the
> encoder's square grid, and cached at native resolution for the thermal-infrared (`tir`) lineage**
> (fixing a single modality-independent ROI geometry contract that `rgb2d` in `ADR-012` and any
> future RGB-ROI port inherit). Model-side clip z-scoring (`ADR-013`) and Stage-2 masking
> (`ADR-007`) are governed separately.

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope
BP4D+ raw frames are full-room recordings (`726 × 480` TIR, `1392 × 1040` visible), whereas perinasal convective airflow occupies only a small fraction of the frame. With a $16 \times 16$ patch tokenizer and a compact spatial grid (`6 × 6 = 36` patches at `input_size = 96`), feeding full frames spends most tokens on irrelevant background.

### 1.2 Design Pressures (Measured over 21,231 Emitted TIR Clips, 1,260 Sessions / 138 Subjects)
1. **Anisotropic deformation under rectangular cropping.** Legacy `crop_rule='box'` stretched rectangular envelopes (median `95 × 110`, aspect `0.86`; widest `413 × 252`; collinear `nostrils` `108 × 13`, aspect `8.3`) independently along $x$ and $y$, distorting facial anatomy as a function of head pose and landmark subset.
2. **Head-sway box inflation.** A clip-static min/max box absorbs large head excursions into oversized crops (up to `413 px` pre-gate), reducing face occupancy and causing 2–4× downscaling. Box width cannot be used as a proxy for motion because Spearman $\rho(\text{sway}, \text{face width}) = +0.15$.
3. **Coupling of padding and input resolution ($\sigma = s / \ell$).** Inspection showed `roi_padding = 0.20` with `input_size = 112` left avoidable background margin (face occupancy 30.3%). Tightening padding to `0.10` shrinks median square side $\ell$ to `95 px`: keeping `input_size = 112` would upsample 84.9% of clips ($\sigma = 1.18$), whereas pairing `roi_padding = 0.10` with `input_size = 96` holds median density at $\sigma = 1.01$, lifts face occupancy to **41.1%**, and reduces tokens/frame from 49 to 36.
4. **Storage vs. load-time separation.** Resizing clips inside an offline cache couples the cache to `input_size`, `clip_stride`, and selection thresholds, forcing multi-hundred-GB rebuilds per experiment.

### 1.3 Requirements
- **R1–R4 (Face-localised, landmark-derived, clip-static, deterministic):** One deterministic static box per clip computed from tracked landmarks over all $T$ source frames *before* `temporal_stride` decimation, so each spatial tubelet tracks fixed anatomy without frame-to-frame interpolation beat.
- **R5 & R9 (Strict square crop & isotropic rescale):** Expand the padded envelope to a square before resizing so $s_x = s_y = \sigma = s / \ell$ exactly.
- **R6 (One modality-independent geometry contract):** TIR and visible lineages share identical box algebra, rescale rules, and motion-gate statistics; only landmark files, frame sizes, lost-frame predicates, and modality-scaled pixel thresholds differ.
- **R7 (Native offline hull cache):** Store native, unscaled session-level crops so one artefact serves every `input_size`, `clip_stride`, and selection threshold with byte-exact parity against live decoding.
- **R8 (Explicit motion exclusion):** Drop clips exceeding `max_sway_px` rather than absorbing large head excursions into oversized boxes.

---

## 2. Decision & Parameter Exposure

**Enforce a three-layer separated ROI pipeline — (1) selection & motion gating (`max_sway_px`), (2) clip-static landmark square box (`crop_rule='square'`, `roi_padding=0.10`), and (3) single load-time isotropic rescale (`input_size=96`, `INTER_AREA` downscale / `INTER_LINEAR` upscale) — backed by a native-resolution session square-hull cache (`roi_cache` v3).**

### 2.1 Landmark Label Space & Presets (Invariant I1)
- **I1 (1-indexed boundary type):** All public knobs (`roi_landmarks`, cache metadata) use **1-indexed** user-guide landmark labels (`IRFeatures`: 28 pts, Fig. 3; `2DFeatures`: 49 pts, Fig. 1). Conversion to 0-based array indices occurs at a single entry point.
- Canonical presets: `nose_mouth` (TIR default: `9–13, 20–26`; 2D: `13–14, 15–19, 32–49`), `nostrils`, `nostril_mouth`, `nose_tip`, and `face` (`1–49`, RGB only).

### 2.2 Square Box Algebra (Invariants I6, I7, I9)
For a clip spanning source frames $[f_0, f_0 + T)$ (pre-decimation, **I9**) and finite target landmarks $L$:
1. Envelope: $x_{\min} = \min_{t,i} x_{t,i}$, $x_{\max} = \max_{t,i} x_{t,i}$ (and likewise $y$).
2. Per-side padding $p = \texttt{roi\_padding}$ (fraction of **each axis' own extent**, so $p = 0.10 \Rightarrow 1.2\times$ extent):
   $$\tilde{x}_0 = x_{\min} - p(x_{\max} - x_{\min}), \qquad \tilde{x}_1 = x_{\max} + p(x_{\max} - x_{\min}).$$
3. Square expansion about centre $c = (c_x, c_y)$ with single integer side $n = \max(\texttt{min\_size}, \lceil \max(\tilde{x}_1 - \tilde{x}_0, \tilde{y}_1 - \tilde{y}_0) \rceil)$:
   $$ix_0 = \lfloor c_x - n/2 \rfloor, \quad ix_1 = ix_0 + n, \qquad iy_0 = \lfloor c_y - n/2 \rfloor, \quad iy_1 = iy_0 + n.$$
   Guarantees $ix_1 - ix_0 \equiv iy_1 - iy_0 = n$ (**I6**), resolving collinear sets (`nostrils`, `nose_tip`) structurally without a separate `min_extent` guard.
4. **Translate-to-fit, never shrink (TIR):** Shift $[ix_0, ix_1)$ into $[0, W)$ (and $y$ into $[0, H)$) preserving side $n$. If $n > W$ or $n > H$, or if translation clips the unpadded landmark envelope ($ix_0 > \lfloor x_{\min} \rfloor$ or $ix_1 < \lceil x_{\max} \rceil + 1$), raise `RoiBoxOutOfFrame` and drop the clip explicitly (`clips_dropped_envelope`, **I7**).

### 2.3 Validity & Head Motion Gates (Invariants I8, I10)
- **TIR sentinel gate (I8):** Any frame containing `(0, 0)` tracker-loss sentinels or missing target landmarks drops every overlapping clip.
- **Motion gate (I10):** Computed on landmark centroids $c_t = \frac{1}{|L|}\sum_{i \in L}(x_{ti}, y_{ti})$ before box sizing:
  $$\text{sway} = \max_{t \in [f_0, f_0+T)} \left\lVert c_t - \operatorname{median}_\tau(c_\tau) \right\rVert_2.$$
  Clips with $\text{sway} > \texttt{max\_sway\_px}$ are dropped and counted (`clips_dropped_motion`). At TIR `max_sway_px = 40.0`: keeps **96.83%** of clips, 1,238/1,260 sessions, 138/138 subjects (worst task `T4`: 80.2%), cutting max square side from `353 px` to `174 px` (median `95 px`, p95 `125 px`).

### 2.4 Single Load-Time Isotropic Rescale & Dataset Output Contract (Invariants I4, I5)
- Scale factor $\sigma = s / n$ ($s_x = s_y = \sigma$), constant across all frames of the clip (**I4**).
- Kernel chosen strictly from geometry: `cv2.INTER_AREA` when $s \le n$, `cv2.INTER_LINEAR` when $s > n$.
- **One resample only (I5):** No intermediate resize or pre-scaling in storage.
- **Dataset output range:** Float32 tensor `[3, T, s, s]` in `[0, 1]` (RGB channel order). Model-side clip z-scoring occurs downstream per `ADR-013`.
- **Stage-2/Stage-3 contract:** `ROI_CONTRACT_KEYS = ('roi_landmarks', 'roi_padding', 'input_size', 'crop_rule', 'max_sway_px')`; any mismatch between Stage-2 checkpoint and Stage-3 config aborts startup.

### 2.5 Native Session Square-Hull Cache (`roi_cache` v3; Invariants I2, I3)
- Stores per session `<S>_<T>` one native-resolution integer crop equal to the bounding box (**window hull**) over every sentinel-valid stride-1 `clip_seconds` window's square box, plus `box`, `landmarks`, `n_frames`, `src_hw`.
- **I2 (Containment):** Every legal clip square box is provably contained inside the stored hull; any escape raises `IndexError` (never silently clamps or falls back to decode).
- **I3 (Key independence):** Cache key includes `format_version=3`, `roi_landmarks`, `roi_padding`, `crop_rule`, `box_rule='window-hull'`, `clip_seconds`, `fps`, `store_resolution='native'`. Excludes `input_size`, `clip_stride`, `max_sway_px`, `task_set`, `temporal_stride`, and signal filters.

### 2.6 Logical Parameter Surface

| Parameter | Type | Default | Constraints | Role |
|---|---|---|---|---|
| `roi_landmarks` | str | `''` (`nose_mouth` TIR; `face` RGB) | 1-indexed preset or CSV | Defines anatomical box; in cache key & `ROI_CONTRACT_KEYS` |
| `crop_rule` | str | `'square'` | `'square' \| 'box'` | Square isotropic rule (`'box'` legacy A/B only); in cache key & `ROI_CONTRACT_KEYS` |
| `roi_padding` | float | `0.10` | `[0.0, 0.5]` | Per-side margin on longer axis; paired with `input_size`; in cache key & `ROI_CONTRACT_KEYS` |
| `input_size` | int | `96` (HPC ROI) / `64` (local) | multiple of patch size `16` | Load-time isotropic rescale target; NOT in cache key; in `ROI_CONTRACT_KEYS` |
| `max_sway_px` | float | `40.0` (TIR) | `>= 0` (`0` = off) | Load-time motion gate; NOT in cache key; in `ROI_CONTRACT_KEYS` |
| `clip_seconds` / `fps` | float | `8.0` / `25.0` | `> 0` | Window span for hull construction; in cache key |
| `roi_cache` | path | unset | matching v3 cache root | Attaches offline native TIR hull cache |

### 2.7 Rejected Alternatives
- **Per-frame tracking box**: Breaks temporal tubelet semantics and introduces per-frame resampling beat.
- **Fixed 1:1 native window without rescaling**: Loses anatomy-scale normalization (face fills only 44% of patch) and drops 4.8–11.9% of clips on geometry alone.
- **Box-width motion gate**: Conflates face size with motion ($\rho = +0.15$), discarding large-faced subjects.
- **`roi_padding=0.10` with `input_size=112`**: Upsamples 84.9% of clips ($\sigma = 1.18$) for 36% extra tokens with no added information.
- **Resizing inside the cache**: Duplicates ~78 GB caches per `input_size` lineage.

---

## 3. Falsifiable Criteria

- **Criterion 1 (Geometry & motion-gate invariants I1–I11)**: Unit tests verify exact square output (`x1 - x0 == y1 - y0`), collinear expansion (`100 × 5 → 100 × 100`), safe boundary translation vs. `RoiBoxOutOfFrame` drop, sway threshold filtering with counter accounting, and Stage-2/Stage-3 `ROI_CONTRACT_KEYS` abort on mismatch.
- **Criterion 2 (Cache containment & byte parity)**: `run_build_roi_cache.py --check --hops 1.0 2.0 --input_size 96 --crop_rule square` and `cache_parity_probe.py` confirm 100% hull containment (0 escapes) and byte-identical tensors between cached and live-decoded frames.
- **Criterion 3 (Corpus geometry ledger)**: `code/notes/eval/archive/tir_resp/roi_geometry_ledger.py` exits `0` with 0 mismatches against published figures (`p=0.10, s=96, max_sway=40`: median side 95 px, p95 125 px, max 174 px, median $\sigma = 1.01$, 96.83% clips kept, 138/138 subjects retained, 41.1% median face occupancy).

- **Acceptance Artifacts**: `tests/test_roi_square_rule.py`, `tests/test_roi_motion_gate.py`, `tests/test_roi_cache.py`, `tests/test_tir_resp_sentinel.py`, `code/notes/eval/archive/tir_resp/roi_geometry_ledger.md`.

---

## 4. Consequences & Trade-offs

- **Positive**: Eliminates non-affine anatomical stretching, caps extreme head-sway crops, raises face patch occupancy from 30.3% to 41.1%, cuts token count by 27% (49 → 36 per frame), and accelerates loading ~200× (1,425 ms → 7 ms/sample via v3 cache).
- **Negative / Technical Debt**:
  - Moving from `0.20 / 112` to `0.10 / 96` invalidates old checkpoints (lineage-locked `pos_embed`) and required rebuilding the TIR cache (73.0 GiB / 1,385 shards).
  - The 40 px motion gate has task-biased attrition (T4 retains 80.2% vs. T3 100%); per-task reports must disclose gate status.
  - `data/rgb_roi_dataset.py` (legacy RGB-BP path) remains on `crop_rule='box'` until ported, whereas `rgb2d` (`ADR-012`) already implements this square contract.

---

## 5. Reproducibility Record

- **Run / Job IDs**: HPC cache build Slurm array `55785296` (32/32 completed, 2026-10-30), key `tirroi_nose_mouth_pad0.10_square_window-hull_clip8s_src726x480_native_v3_fa9f91bdd5` (1,385 cached / 15 skipped / 0 missing, 78.4 GB).
- **Evidence Artifacts**: `code/notes/eval/archive/tir_resp/roi_geometry_ledger.md`, `code/notes/eval/archive/tir_resp/hpc_precomputation.md`.
- **Tests / Smoke Checks**: `code/tests/test_roi_square_rule.py`, `code/tests/test_roi_motion_gate.py`, `code/tests/test_roi_cache.py`, `code/tests/test_tir_resp_sentinel.py`, `code/tests/test_config_loading.py`.
- **Rollback / Rejection Condition**: Set `crop_rule: box` and `max_sway_px: 0` in YAML/CLI to reproduce the legacy unsquared, ungated pixel path without code rollback.
