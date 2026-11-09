# Video Input Preprocessing — Implementation Plan

Status: **LANDED (2026-11-01)** — all phases implemented; see §12 for the
as-built record. Contract of record: `code/notes/design/Video_input_Preprocessing.md`.
**On any conflict the design wins**; §2 lists the conflicts found in the shipped
code.

Scope: the **visual input** of the two models that consume video clips —
`core/multimae.py::MultiModalMAE` (Stage 2) and
`core/waveform_model.py::MultiModalWaveformRegressor` (Stage 3).
**No dataset change**: the datasets keep returning `uint8/255 -> [0, 1]`, so
`_check_clip`, both inspect runners and every `[0,1]` test stay valid.

Consequence up front: the ENCODER INPUT distribution changes, so **every visual
checkpoint must be retrained** (§7). The target space does NOT change (§3.3).

---

## 1. What this lands

1. **Clip-level input normalisation** — before patch division, the whole visual
   clip is z-scored per channel (§3.1), per sample, in both models.
2. **Per-cube target normalisation restricted to MASKED cubes** — the GT pixels
   of a masked cube are standardised with that cube's own mean/variance, and only
   those positions enter the masked MSE (§3.2).
3. **Unmasked cubes carry the clip-normalised values** — no local per-cube map
   (§3.2).
4. **One implementation** — a single helper in `core/input_adapters.py` called by
   both models, so Stage 2, Stage 3, `tir_roi`, `rgb_roi` and the non-ROI
   `bp4d+` visual streams all share one input space by construction.

---

## 2. Conflicts with the shipped code (design wins)

| # | clause | shipped code | design | resolution |
|---|--------|--------------|--------|------------|
| C1 | clip-level input z-score before patch division | the dataset delivers `[0,1]` (`/255` in `data/tir_resp_dataset.py::_load`, `data/rgb_roi_dataset.py::_load`, `data/paired_dataset.py::_load_visual`) and both models patch-embed it **verbatim** (`TubeletEmbed` is a bare `Conv3d`; measured: doubling the input doubles the tokens) | z-score the whole clip first | add `zscore_clip` and call it in both models (§5 P1/P3/P4) |
| C2 | per-cube normalisation only for masked cubes | `MultiModalMAE._targets` z-scores **every** token of the visual stream | masked cubes only | pass the mask into `_targets` (§5 P2) |
| C3 | decoder prediction vs locally-normalised GT, MSE | `MaskedMSELoss(pred, tgt, mask)` — masked positions only | same | already compliant, no change |
| C4 | unmasked cubes: no local normalisation | unmasked tokens ALSO receive per-cube targets (the loss ignores them, so training is unaffected; the returned target tensor is wrong) | clip-normalised values | follows from C2 |

---

## 3. The contract this plan freezes

### 3.1 Clip normalisation (the input)

For each **visual** stream `s`, per SAMPLE and per CHANNEL, over the clip's
`(T, H, W)`:

```
mu   = mean(x, dims=(2,3,4))
var  = var(x, dims=(2,3,4), unbiased=False)
xhat = (x - mu) / sqrt(var + eps)
```

* `eps = 1e-6` — the SAME constant as the per-cube target and
  `core/criterion.py::MaskedMSELoss._EPS`. No second epsilon.
* Position in the forward pass: **after** `_fit_time` (so the statistics describe
  the exact clip the adapters see, including the tubelet trim) and **before**
  `adapters[s]` (i.e. before patch division, as the design requires).
* `float32` in, `float32` out; shape and dtype preserved.
* Per-sample statistics ⇒ **no cross-rank reduction**, so the transform is
  bit-identical under any DDP world size.
* 1-D streams (`resp`/`bp`/`eda`) are NOT touched: their label normalisation is
  the dataset's per-clip z-score (`data.tir_resp_dataset.TARGET_NORM`) and the
  model stays identity (`target_norm='none'` for the ROI lineages).

### 3.2 Per-cube target (the loss)

* For a **masked** cube: `(pixels - mean_cube) / sqrt(var_cube + eps)` over that
  cube's `C*t*ph*pw` values — i.e. the existing computation, unchanged.
* For an **unmasked** cube: the clip-normalised patch values, no local map.
* The decoder prediction is compared against those targets by `MaskedMSELoss`,
  which keeps only `mask == 1` positions.
* Token order stays the tubelet order `(t, h, w)` from
  `Tensor.view(B, C, Gt, t, Gh, ph, Gw, pw).permute(0,2,4,6,1,3,5,7)`; one cube ==
  one token.

### 3.3 What must NOT change (invariants to assert, not assume)

* **Target space.** A per-cube z-score is affine-invariant, so
  `zcube(clip_z(x)) == zcube(x)` up to eps. The masked MSE of a frozen model on
  frozen inputs must be **numerically identical** before and after this change —
  that equality is the proof that only the input moved (gate G3).
* Geometry: `_check_space_time_alignment`, `_fit_time` (trim only, never pad),
  token counts, positional embeds, `mask_token`, masking generator, loss weights.
* Mask semantics: `mask == 1` means "masked / to reconstruct".
* The datasets' `[0,1]` output contract and everything built on it.

---

## 4. Decisions

| # | decision | value |
|---|----------|-------|
| D1 | statistics shape | **per channel** over `(T,H,W)`, per sample (not one global scalar). Equalises the three planes of the false-colour TIR rendering; the single-scalar alternative preserves hue exactly but leaves channel scales apart |
| D2 | implementation site | **model-side**, one helper in `core/input_adapters.py`; datasets unchanged. Rationale: the design's two video rules then live together, Stage 2 and Stage 3 cannot drift, and every visual dataset complies without edits |
| D3 | scope | **all visual streams** (`tir`, `rgb`; ROI and non-ROI lineages) — one input contract for one set of weights |
| D4 | epsilon | `sqrt(var + 1e-6)`, shared with the per-cube target |
| D5 | knob | **none** — no CLI flag, no YAML key, no constructor argument. Same rule as the 1-D label normalisation: one behaviour |
| D6 | `/255` in the datasets | **stays**. It is now a dtype/scale step only; the z-score makes it affine-irrelevant, and removing it would churn inspectors for no gain |
| D7 | `_targets(..., mask=None)` | means "all tokens masked" — only for unit tests / diagnostics; `forward` always passes the real mask |
| D8 | denormalisation | none. Targets stay in per-cube z units; nothing in the pipeline reconstructs volts from a Stage-2 loss |

---

## 5. Phases

### Phase 1 — the helper (`core/input_adapters.py`)

**Step 1** add `EPS = 1e-6` and

```python
def zscore_clip(x, eps: float = EPS):
    """Per-channel z-score over the WHOLE clip [B,C,T,H,W] (per sample)."""
```

implemented with `dims=(2,3,4)`, `unbiased=False`, `torch.sqrt(var + eps)`.
Docstring must state: it is THE visual input preprocessing, it runs before patch
division, it is not configurable.

**Step 2** export it in `__all__`.

**Gate G1** — unit tests on the helper: per-channel mean ≈ 0 / std ≈ 1; a
constant channel yields exactly 0 (eps guard, no NaN); dtype and shape preserved;
batch entries are independent (a constant clip in slot 1 does not shift slot 0).

### Phase 2 — masked-only per-cube target (`core/multimae.py`)

**Step 3** `_targets(self, x, stream, mask=None)`:

* visual branch: build `flat [B, N, C*t*ph*pw]` exactly as today, compute
  `mean/var` over the last dim (today's code), then
  `tgt = torch.where(mask.unsqueeze(-1).bool(), zcube, flat)` when `mask` is
  given; `mask=None` ⇒ all `zcube` (D7).
* 1-D branch untouched (`target_norm` semantics unchanged).

**Step 4** `forward` passes `loss_mask` (the same tensor the loss uses) into
`_targets` for every stream; for 1-D streams the argument is ignored.

**Gate G2** — a non-uniform clip with a hand-built mask: unmasked tokens' target
equals the clip-normalised flat values (per-token mean ≠ 0), masked tokens' target
has per-token mean ≈ 0 / std ≈ 1.

### Phase 3 — Stage 2 wiring (`core/multimae.py::forward`)

**Step 5** in the `xin` loop:

```python
xin[s] = self._fit_time(x[s], n, s)
if s in self.visual:
    xin[s] = zscore_clip(xin[s])
```

so the adapters AND `_targets` both see the normalised clip (the design's
"unmasked cubes receive clip-level normalisation" is then true of the target
tensor as well).

**Gate G3** — **loss invariance**: freeze a tiny model (fixed seed), a fixed
clip and a fixed mask; the total loss must equal the pre-change value to 1e-6.

**Gate G4** — **input affine-invariance**: `tokens(2*x + 5) == tokens(x)` element
wise (to 1e-6) for the normalised path, and the tokens must differ from the
pre-change ones (the change is real, not a no-op).

### Phase 4 — Stage 3 wiring (`core/waveform_model.py::forward`)

**Step 6** same two lines in `MultiModalWaveformRegressor.forward` right after
`xt = self._fit_time(x[s], self.num_frames, s)`. Every stream of this model is
validated visual, so no branch is needed.

**Gate G5** — Stage-2/Stage-3 parity: for the same weights and clip, the Stage-3
waveform output is identical for `x` and `2x+5`, and the Stage-2 encoder sees the
same normalised tokens it saw in training.

### Phase 5 — verification

**Step 7** `../.env/bin/python -m pytest` (489 + new module) and
`../.env/bin/python -m core.multimae` (the module's built-in self-test; the `-m`
form is required because the package uses relative imports. It feeds
`torch.randn`, which is already z-like, so it must pass unchanged).

**Step 8** local smoke on synthetic data (no corpus needed): one Stage-2 step and
one Stage-3 step, asserting finite loss and logging the per-channel statistics the
model now receives.

### Phase 6 — docs

**Step 9** `code/notes/design/Video_input_Preprocessing.md` — append an
"Implementation" section mapping each clause to symbol (this document stays the
authority; the design gains the decisions D1/D4/D5 and the leakage note).

**Step 10** `code/notes/spec/02_model_architecture.md` — forward order becomes
`_fit_time -> zscore_clip (visual) -> adapters -> positions -> masks -> encoder`;
`_targets` documented as masked-only. Mirror the same text in
`code/scripts/project/write_specs.py` (never run it blind — see its header).

**Step 11** `code/README.md` and `code/notes/design/ROI_crop_and_rescaling.md`
§"tensor contents": state that `[0, 1]` is the **dataset** contract and the MODEL
z-scores before patching — otherwise `[0,1]` reads as "the network input".

---

## 6. Test plan

| file | test | asserts |
|---|---|---|
| `tests/test_video_input_norm.py` | `test_zscore_clip_per_channel` | mean≈0 / std≈1 per channel over (T,H,W), all 3 channels |
| " | `test_zscore_clip_constant_channel_is_finite_zero` | eps guard; no NaN/Inf |
| " | `test_zscore_clip_dtype_and_batch_independence` | float32 preserved; slots independent |
| " | `test_encoder_input_is_affine_invariant` | G4: tokens(2x+5) == tokens(x); and != pre-change tokens |
| " | `test_masked_mse_is_unchanged_by_input_normalisation` | G3: frozen model+clip+mask, loss equal to a stored reference |
| " | `test_unmasked_cube_target_is_clip_space` | G2: unmasked per-token mean ≠ 0, masked ≈ 0 / std ≈ 1 |
| " | `test_stage3_is_affine_invariant` | G5 |
| `tests/test_physio_norm.py` | `test_target_norm_none_leaves_visual_streams_normalised` | update to pass an explicit mask |

---

## 7. What this invalidates

| artefact | status |
|---|---|
| any Stage-2 checkpoint with visual streams (`stage2_hpc_tir_roi_resp`, `stage2_local_tir_roi_*`, `stage2_local_rgb_roi_bp`, `stage2_local*` with `rgb`/`tir`) | **retrain** — the encoder saw `[0,1]`, it will now see z-scored clips |
| Stage-3 checkpoints built on those encoders (`resp_tir_roi_hpc*`, `bp_rgb_roi_*`) | **retrain**, for the same reason |
| every reported visual number in `notes/eval/**` | stays as a record of the previous input contract; new runs need new prefixes |
| label space / target space | **unchanged** — no evaluator, cache key or metric is affected |
| ROI cache, resp cache | **unchanged** — neither stores model-space pixels |

Run prefixes: keep the existing ones for old results; new runs should carry a
marker (e.g. `…_zclip`) so a checkpoint's input contract is readable from its path.

---

## 8. Risks and open items

1. **Masked statistics leak into the visible input.** The clip statistics are
   computed over the whole clip, including the masked cubes (the design says
   "entire clip"). Two scalars per channel therefore describe hidden content.
   This is the standard trade-off for global normalisation; it is recorded here so
   it is a decision, not an accident. Alternative (visible-only statistics) makes
   the input depend on the mask — not recommended.
2. **Per-channel vs hue.** Per-channel equalisation slightly distorts the hue
   relations of the false-colour TIR rendering. If an ablation shows the colour
   mapping matters, the single-scalar variant is a one-line change in
   `zscore_clip` (and a new design decision).
3. **AMP.** The repo has no `torch.autocast`; AMP is the custom scaler in
   `utils/native_scaler.py`, so the transform runs in float32. If autocast is ever
   introduced, compute the statistics in float32 explicitly.
4. **Determinism / DDP.** Per-sample statistics need no collective; the transform
   is identical at any world size. No seeding change.
5. **Not in scope.** `core/model.py::ProjectViT` (2-D ViT baseline) and
   `core/au_probe.py` (consumes encoder features, not pixels) keep their own input
   contract.
6. **Discoverability.** With the datasets unchanged, nobody can see the model's
   input space from a tensor dump. Optional follow-up: a one-shot log line in each
   model's first forward (per-channel mean/std seen by the encoder).

---

## 9. Run recipes

```bash
# from code/
../.env/bin/python -m pytest -q                     # full suite
../.env/bin/python -m pytest -q tests/test_video_input_norm.py
../.env/bin/python -m core.multimae                  # in-module self-test
```

Stage-2/Stage-3 smoke on synthetic data requires no corpus (the tests build
their own trees), so gates G1–G5 are CPU-only and fast.

---

## 10. Doc impact

| document | change |
|---|---|
| `design/Video_input_Preprocessing.md` | add "Implementation" + decisions D1/D4/D5 + the leakage note (contract stays authoritative) |
| `spec/02_model_architecture.md` + `scripts/project/write_specs.py` | forward order with `zscore_clip`; `_targets` masked-only |
| `spec/03_engine_and_loss.md` | loss section: targets are per-cube z for MASKED cubes only |
| `README.md`, `design/ROI_crop_and_rescaling.md` | `[0,1]` is the dataset contract, not the network input |
| `plan/TirROI_Resp_plan.md`, `plan/RGB_BP_plan.md` | one line pointing at this plan for the input contract |

---

## 11. BDD Scenarios

**Scenario 1: Clip-level input normalization**
* **Given** a raw visual video clip tensor with shape `[B, C, T, H, W]`
* **When** the clip is passed through the `zscore_clip` function
* **Then** the output tensor should retain the exact same shape and dtype
* **And** each channel per sample should have a mean of approximately 0 and a standard deviation of approximately 1

**Scenario 2: Masked cube target normalization**
* **Given** a visual video clip and a binary mask indicating masked patches
* **When** the target tensor is generated by the `_targets` method
* **Then** the target values for the masked patches should be normalized using their local per-cube mean and variance

**Scenario 3: Unmasked cube target normalization**
* **Given** a visual video clip and a binary mask indicating unmasked patches
* **When** the target tensor is generated by the `_targets` method
* **Then** the target values for the unmasked patches should match the clip-level normalized values without any local per-cube normalization applied

**Scenario 4: Affine invariance of the target space**
* **Given** a frozen MultiMAE model, a fixed input clip, and a fixed mask
* **When** the input clip is scaled and shifted (e.g., `2*x + 5`)
* **Then** the computed masked MSE loss should remain numerically identical (up to `1e-6`) to the loss of the original clip

**Scenario 5: Stage-3 waveform model affine invariance**
* **Given** a frozen MultiModalWaveformRegressor model and a fixed input clip
* **When** the input clip is scaled and shifted
* **Then** the Stage-3 waveform output should be identical to the output of the original clip

---

## 12. As built (2026-11-01)

| phase | landed as |
|---|---|
| P1 helper | `core/input_adapters.py`: `EPS = 1e-6`, `zscore_clip(x, eps=EPS)`, both in `__all__` |
| P2 masked-only target | `core/multimae.py::MultiModalMAE._targets(x, stream, mask=None)` — `torch.where(mask.unsqueeze(-1).bool(), zcube, flat)`; `forward` passes `loss_mask` |
| P3 Stage 2 | `_fit_time -> zscore_clip (visual) -> adapters` in `MultiModalMAE.forward` |
| P4 Stage 3 | same two lines in `MultiModalWaveformRegressor.forward` (every stream validated visual) |
| P5 tests | `code/tests/test_video_input_norm.py` (8 tests, gates G1-G5); `tests/test_physio_norm.py::test_target_norm_none_leaves_visual_streams_normalised` now passes an explicit all-ones mask |
| P6 docs | this plan, `design/Video_input_Preprocessing.md` (Implementation + D1/D4/D5 + leakage note), `spec/02_model_architecture.md` (§7.4 step 1, new `_targets` note, §8.3, §10), `spec/03_engine_and_loss.md` §6, `scripts/project/write_specs.py` (same three edits), `README.md`, `design/ROI_crop_and_rescaling.md` §4.4, `plan/TirROI_Resp_plan.md` §4, `plan/RGB_BP_plan.md` §4 |

Verification actually performed:

* `../.env/bin/python -m pytest -q` -> **497 passed** (489 before + 8 new).
* `../.env/bin/python -m core.multimae` -> `mask_self_test: ALL PASS` and
  `stft_self_test: ALL PASS`. (At the time this also ran `spectral_self_test`;
  that self-test was removed with the Stage-2 MR-STFT term on 2026-11-09, and
  the asymmetric `cross_mae_self_test` went with the cross-MAE variant on
  2026-11-07.)
* Synthetic smoke (CPU, no corpus): one Stage-2 step and one Stage-3 step,
  finite loss; the encoder receives per-channel `mean=0.0 / std=1.0` for a
  `[0,1]` clip (before: `mean=0.4990 / std=0.2885`).

Deviations from this plan (both recorded, neither changes a contract):

1. **G3 compares against an emulated pre-change path, not a stored float.** The
   test patches `core.multimae.zscore_clip` to the identity and feeds an
   already-z-scored clip. A hardcoded reference loss would be brittle across
   torch versions for no extra information.
2. **G4/G5 measure the tokens at `adapters.<stream>`, not at `enc_norm`.** The
   encoder's first `LayerNorm` is nearly blind to a global affine rescaling of
   its input, so an `enc_norm` probe reports a difference of ~1e-5 for a clip
   whose adapter tokens differ by O(1) — it silently certifies nothing. The
   adapter output is the actual input space.
3. `python core/multimae.py` cannot run the in-module self-test (relative
   imports); use `python -m core.multimae` from `code/`. §9 is corrected by this
   note.

