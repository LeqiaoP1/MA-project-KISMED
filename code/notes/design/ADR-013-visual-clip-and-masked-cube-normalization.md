# ADR-013: Model-Side Visual Clip Normalization and Masked-Cube Target Standardization

- **Status**: Accepted
- **Date**: 2026-11-01 (codified as ADR-013 on 2026-11-26)
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/plan/features/ADR-013-visual-clip-and-masked-cube-normalization-plan.md`, `code/notes/spec/02_model_architecture.md` (§7–§8), `code/notes/spec/03_engine_and_loss.md` (§6), `code/notes/design/ADR-011-landmark-roi-geometry-and-tir-cache.md` (dataset `[0, 1]` boundary)
- **Supersedes**: `Video_input_Preprocessing.md` (promoted into ADR structure)

> **Scope.** This ADR governs one research aspect: **how visual video tensors (`[B, C, T, H, W]`)
> are normalized inside Stage-2 and Stage-3 models prior to patch tokenization and how visual
> reconstruction targets are standardized for masked cubes**. Dataset-side ROI extraction and
> `[0, 1]` pixel scaling (`ADR-011`, `ADR-012`) and physiological 1-D label normalization
> (`ADR-003`, `ADR-004`) are out of scope here.

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope
Visual datasets (`tir`, `rgb2d`, `rgb`) emit `float32` clips scaled to `[0, 1]` (`uint8 / 255`). Prior to this decision, both `MultiModalMAE` (Stage 2) and `MultiModalWaveformRegressor` (Stage 3) passed raw `[0, 1]` clips directly into the bare `Conv3d` `TubeletEmbed` adapter (mean ≈ `0.499`, std ≈ `0.289`), while Stage-2 target construction applied local per-cube z-scoring indiscriminately to both masked and unmasked visual tokens.

### 1.2 Design Pressures
1. **Unnormalized encoder inputs.** Raw `[0, 1]` clips vary across subjects, ambient temperature scales (false-colour TIR), and camera illumination (RGB/RGB2D); feeding unstandardized intensities leaves `TubeletEmbed` sensitive to global affine brightness/offset shifts.
2. **Stage-2 / Stage-3 parity risk.** Placing normalization inside individual dataset classes risks silent drift between `tir_roi`, `rgb2d`, `rgb_roi`, and `bp4d+` or between Stage-2 pre-training and Stage-3 fine-tuning, while breaking `[0, 1]` dataset inspectors (`_check_clip`).
3. **Target-tensor leakage on unmasked cubes.** Applying local per-cube normalization to unmasked tokens in `MultiModalMAE._targets` violated the VideoMAE/MultiMAE contract that only **masked** cubes undergo local per-cube standardization while unmasked cubes retain clip-normalized values.

### 1.3 Requirements
- **R1 — Clip-level per-channel z-scoring before patch division.** Every visual stream is standardized per sample and per channel over `(T, H, W)` after temporal trimming (`_fit_time`) and before `adapters[s]`.
- **R2 — Masked-only per-cube target standardization.** Visual reconstruction targets standardize a tubelet locally (over its `C · t · p_h · p_w` elements) if and only if `mask == 1`; unmasked tokens retain their clip-normalized pixel values.
- **R3 — Single model-side implementation across both stages.** One shared helper governs Stage 2 and Stage 3 across all visual streams (`tir`, `rgb2d`, `rgb`), leaving dataset `[0, 1]` contracts and offline caches untouched.
- **R4 — DDP invariance and zero configurability.** Per-sample statistics require no cross-rank collectives and expose no toggle knob, ensuring every checkpoint built under this contract has one unambiguous input space.

---

## 2. Decision & Parameter Exposure

**Apply deterministic, non-configurable per-sample, per-channel clip z-scoring (`zscore_clip`, $\epsilon = 10^{-6}$) inside both Stage-2 and Stage-3 models immediately before visual patch embedding, and restrict local per-cube target standardization in Stage 2 strictly to masked visual cubes (`mask == 1`).**

### 2.1 Normative Mathematical Specification
For any visual clip $x \in \mathbb{R}^{B \times C \times T \times H \times W}$ (post-`_fit_time`, `float32`), compute per sample $b$ and per channel $c$:
$$\mu_{b,c} = \frac{1}{THW}\sum_{t,h,w} x_{b,c,t,h,w}, \qquad
\sigma^2_{b,c} = \frac{1}{THW}\sum_{t,h,w} (x_{b,c,t,h,w} - \mu_{b,c})^2 \quad (\texttt{unbiased=False}),$$
$$\hat{x}_{b,c,t,h,w} = \frac{x_{b,c,t,h,w} - \mu_{b,c}}{\sqrt{\sigma^2_{b,c} + \epsilon}}, \qquad \epsilon = 10^{-6}.$$

**Canonical forward order (identical in Stage 2 and Stage 3):**
```text
_fit_time -> zscore_clip (visual streams only) -> adapters[s] -> positions[s] -> [masking] -> encoder
```

**Stage-2 visual target tensor (`_targets(x, stream, mask)`):**
Let `flat` $\in \mathbb{R}^{B \times N \times (C \cdot t \cdot p_h \cdot p_w)}$ be the patchified clip-normalized tensor $\hat{x}$ in tubelet order `(t, h, w)`, and let `zcube` standardize each token locally over its last axis with the same $\epsilon = 10^{-6}$ (`unbiased=False`). Then:
$$\text{target}_{b,n} = \begin{cases} \texttt{zcube}_{b,n} & \text{if } M_{b,n} = 1 \text{ (masked cube)} \\ \texttt{flat}_{b,n} & \text{if } M_{b,n} = 0 \text{ (unmasked cube).} \end{cases}$$
`MaskedMSELoss` evaluates reconstruction error strictly over `M == 1` positions. 1-D physiological streams (`resp`, `bp`) are untouched here (`TARGET_NORM = 'clip'` at dataset load, `target_norm = 'none'` in ROI models).

### 2.2 Invariants
- **I1 (Target-space affine invariance):** Because per-cube z-scoring is affine-invariant, $\texttt{zcube}(\texttt{zscore\_clip}(x)) = \texttt{zcube}(x)$ up to $\epsilon$; frozen-model masked MSE is numerically unchanged when comparing against an already-z-scored reference.
- **I2 (Input affine invariance):** Adapter tokens satisfy $\text{adapters}[s](\text{zscore\_clip}(ax + b)) = \text{adapters}[s](\text{zscore\_clip}(x))$ for any $a > 0, b \in \mathbb{R}$ in both Stage 2 and Stage 3.
- **I3 (Constant-channel safety & batch independence):** A constant channel yields exact `0.0` (no `NaN`/`Inf`), and sample $b$ never influences sample $b'$.
- **I4 (Dataset contract unchanged):** Datasets remain `[0, 1]` (`uint8 / 255`); no offline ROI cache key or stored byte changes.

### 2.3 Logical Parameter Surface
**None (by decision).** Just like 1-D `TARGET_NORM = 'clip'`, visual input normalization has **no CLI flag, no YAML key, and no constructor toggle**. One fixed normalization contract eliminates silent checkpoint-input mismatches. Constant `EPS = 1e-6` is shared with `MaskedMSELoss._EPS`.

### 2.4 Rejected Alternatives
- **Dataset-side normalization**: Rejected because every visual dataset would have to duplicate it, Stage 2 and Stage 3 could drift, and `[0, 1]` dataset inspectors (`_check_clip`, visualizers) would break.
- **Visible-only clip statistics**: Rejected because making input normalization depend on the random mask destroys input-space reproducibility without knowing the mask.
- **Single scalar across all 3 RGB/TIR channels**: Rejected in favor of per-channel equalization across the three planes of false-colour TIR (noted as a reversible 1-line ablation if future colour-hue studies demand it).
- **Configurable `--video_norm` knob**: Rejected to prevent undocumented mixed-space checkpoints.

---

## 3. Falsifiable Criteria

- **Criterion 1 (Helper & target-space unit gates G1–G5)**:
  - Per-channel mean $\approx 0$, std $\approx 1$ over `(T, H, W)`; constant channel produces finite `0.0`; batch slots strictly independent.
  - Unmasked cube targets equal clip-normalized `flat` (per-token mean $\neq 0$), while masked cube targets have per-token mean $\approx 0$, std $\approx 1$.
  - Adapter tokens (and Stage-3 waveform outputs) are invariant to `2x + 5` scaling to `1e-6`, while differing by $O(1)$ from unnormalized `[0, 1]` inputs.
  - Frozen-model masked MSE matches emulated pre-change target space to `1e-6`.
- **Criterion 2 (Smoke & self-test)**: `python -m core.multimae` self-tests pass and synthetic Stage-2/Stage-3 steps execute with finite loss and `mean=0.0 / std=1.0` encoder inputs.

- **Acceptance Artifacts**: `code/tests/test_video_input_norm.py` (8 tests), `code/tests/test_physio_norm.py`.

---

## 4. Consequences & Trade-offs

- **Positive**: Equalizes visual input distributions across subjects, lighting, and thermal ranges; enforces exact Stage-2/Stage-3 input-space parity; preserves all offline ROI caches untouched.
- **Negative / Technical Debt**:
  - **Every visual Stage-2 and Stage-3 checkpoint predating 2026-11-01 must be retrained** (encoder saw `[0, 1]` before, z-scored clips after).
  - **Accepted trade-off (global stats leakage):** Clip statistics include masked cubes, leaking two scalars per channel into the visible input — the standard global-normalization trade-off, documented explicitly rather than accidental.
  - Per-channel equalization slightly alters false-colour TIR hue relations.
  - Under CUDA autocast (`ADR-016`), statistics must remain numerically safe (FP32 stats recommended if FP16 overflow is ever observed).

---

## 5. Reproducibility Record

- **Run / Job IDs**: Landed 2026-11-01; all subsequent runs (`…_zclip` and all `ADR-002`–`ADR-008` runs) execute under this contract.
- **Evidence Artifacts**: As-built verification record in `code/notes/plan/features/ADR-013-visual-clip-and-masked-cube-normalization-plan.md` §12 (497 passed at landing).
- **Tests / Smoke Checks**: `code/tests/test_video_input_norm.py`, `code/tests/test_physio_norm.py`.
- **Rollback / Rejection Condition**: Because this is a non-configurable architectural invariant, reverting requires a superseding ADR and retraining all downstream checkpoints.
