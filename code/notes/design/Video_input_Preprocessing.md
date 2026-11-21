
# purpose

This is desgin contract for the video signals like TIR-ROI.

# Video input preprocessing

Applied to the Entire Video Clip: Before BEFORE patch division, the raw video clip as a whole undergoes standard input preprocessing (z-score) to prepare the input patches for the network.


# Per-patch target normalization (only for masked cubes for loss computation)

Local Patch Normalization: For the masked cubes that the decoder needs to reconstruct, it should calculate a per-cube mean and variance locally within each individual patch/cube.


Ground-truth pixel values inside each masked cube are standardized using that cube's local mean and variance.


The decoder's predicted pixels are compared against these locally normalized target pixels using Mean Squared Error (MSE) loss.

# Unmasked cubes 
do not undergo local per-patch normalization, but they do receive clip-level input normalization.

---

# Implementation (landed 2026-11-01)

Plan of record: `code/notes/plan/features/video_input_preprocessing_implementation_plan.md`.
This section maps every clause above to the symbol that implements it. **The
clauses above stay authoritative**; on any conflict the contract wins and the
code is the bug.

| clause | symbol |
|---|---|
| clip-level z-score, before patch division | `core/input_adapters.py::zscore_clip` (`EPS = 1e-6`) |
| Stage 2 calls it | `core/multimae.py::MultiModalMAE.forward` — after `_fit_time`, before `adapters[s]` |
| Stage 3 calls it | `core/waveform_model.py::MultiModalWaveformRegressor.forward` — after `_fit_time`, before `adapters[s]` |
| masked cubes standardised locally | `core/multimae.py::MultiModalMAE._targets(..., mask)` (`torch.where(mask, zcube, flat)`) |
| unmasked cubes keep clip-level normalisation | same line — the `flat` branch |
| MSE against the local targets | `core/criterion.py::MaskedMSELoss` (unchanged; keeps `mask == 1`) |

Forward order, identical in both models:

```
_fit_time -> zscore_clip (visual streams only) -> adapters -> positions -> masks -> encoder -> decoder
```

Tests: `code/tests/test_video_input_norm.py` (gates G1-G5).

## Decisions

| # | decision | value |
|---|---|---|
| D1 | statistics shape | **per channel** over `(T, H, W)`, per sample — equalises the three planes of the false-colour TIR rendering |
| D4 | epsilon | `sqrt(var + 1e-6)`, the same constant as the per-cube target; no second epsilon |
| D5 | knob | **none** — no CLI flag, no YAML key, no constructor argument. One behaviour, so a checkpoint's input space is readable from the code alone |

## What did NOT change

The datasets keep returning `[0, 1]` (`/255`). That is the **dataset**
contract, not the network input: the model z-scores before patching.

* **Target space** untouched — a per-cube z-score is affine-invariant, so
  `zcube(zscore_clip(x)) == zcube(x)` up to eps. The frozen-model masked MSE is
  numerically unchanged
  (`tests/test_video_input_norm.py::test_masked_mse_is_unchanged_by_input_normalisation`).
* Geometry, token counts, positional embeds, `mask_token`, the masking
  generator, the loss weights, the label space, the ROI/resp caches: unchanged.
* **Every visual checkpoint must be retrained** — the encoder saw `[0, 1]` and
  now sees z-scored clips. Old `notes/eval/**` numbers record the previous input
  contract; new runs carry a `…_zclip` marker in their run prefix.

## Known trade-off (a decision, not an accident)

The clip statistics are computed over the **whole** clip, masked cubes included,
so two scalars per channel describe hidden content. This is the standard
trade-off for global normalisation. The alternative (visible-only statistics)
would make the input depend on the mask — the same pixels would land in a
different input space depending on the sampling, and the transform would no
longer be reproducible without the mask.

Limitation: per-channel equalisation slightly distorts the hue relations of the
false-colour TIR rendering. If an ablation shows the colour mapping matters, the
single-scalar variant is a one-line change inside `zscore_clip` plus a new
decision row above.


