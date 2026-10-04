# Solution A vs Solution C — Asymmetric Cross-MAE vs the shared-encoder MAE

Design note for the **asymmetric cross-MAE** variant ("Solution A") of the TIR-ROI →
respiration pre-training, and how it coexists with the shipped shared-encoder design
("Solution C") in a single codebase.

* Architecture spec: `code/design/SolutionA_asymCrossMAE.md`
* Diagrams: `code/design/SolutionA_asymCrossMAE_diagram.md`
* Task / data plan of record: `code/plan/TirROI_Resp_plan.md`
* Span-masking rationale: `code/design/SpanMask_PhysioSignals.md`

---

## 1. Why a second pre-training objective

The shipped Stage-2 SSL is a **shared-encoder** masked autoencoder
(`core/multimae.py::MultiModalMAE`): the visible tokens of *every* stream are
concatenated into one ViT sequence, and each stream is then reconstructed by a
shared self-attention decoder built from *its own* visible tokens plus `mask_token`s.

Measured consequence (see `TirROI_Resp_plan.md` §9 and the repo memory notes):
respiration at 100 Hz with content in 0.16–0.4 Hz is oversampled ~30–60×
(`sig_kernel 16` = 16 samples per token). Under a **scattered** mask every masked
resp token has visible resp neighbours 0.16–0.32 s away, so the resp loss is
satisfied by *within-modality interpolation*. A five-line interpolator of the visible
resp scores `mse_resp` 0.129 / `spec_resp` 0.98 with **zero video**, beating the
trained 102 M-parameter model (1.05 / ~5.0). The encoder therefore has no incentive to
learn a TIR → respiration mapping, and Stage-3 inherits an encoder that carries
appearance but no usable respiratory phase.

**Solution A removes the shortcut structurally, not by tuning a loss weight:** the
physio stream is never fed to the encoder at all. It survives only as a
*reconstruction target*, decoded by a target-driven cross-attention head that can only
succeed by reading the video.

---

## 2. What differs — exactly one thing: the pre-training stage

| | Solution C (shipped) | Solution A (new) |
|---|---|---|
| encoder input | visible **tir + resp** tokens | visible **tir** tokens only |
| resp decoding | shared self-attention decoder over the stream's **own** visible tokens + mask tokens | **cross-attention** decoder: learned per-time-slot queries attend to the whole visual encoder output |
| resp loss support | masked resp tokens only | **all** time slots (all-ones mask) |
| tir decoding | self-attention pixel decoder | self-attention pixel decoder (unchanged) |
| data pipeline | — | **identical** (see §3) |
| encoder module names / shapes | `adapters.*`, `positions.*`, `enc_blocks.*`, `enc_norm.*` | **identical** |
| Stage-3 | `MultiModalWaveformRegressor` | the **same** model, either checkpoint |

Because only encoder *membership* and the *physio decoder* change, the encoder stays
byte-for-byte compatible: either pre-trained checkpoint loads into the same Stage-3
model (`core/waveform_model.py`), which whitelists only
`enc_blocks.` / `enc_norm.` / `adapters.` / `positions.`.

---

## 3. Shared invariants (must hold; each is checked)

**I1 — identical data path.** The change touches `core/` only; nothing under `data/`
is edited. Both Stage-2 dataset builders (`build_tir_roi_pretrain_dataset`,
`build_tir_roi_finetune_dataset`) are pure `getattr(args, …)` reads of the same config
keys, and `TirRoiRespFinetuneDataset` **subclasses** `TirRoiRespPretrainDataset`, so
the ROI (`resolve_roi_landmarks` + `roi_padding` / `roi_quantile`), the task selection
(`task_groups` / `task_set`) and `min_signal_spread` are shared by construction. The
two pre-train configs differ only in four model keys plus `output_dir`; the two
Stage-3 configs differ only in `finetune:` plus `output_dir:`.

**I2 — identical encoder interface.** `adapters.tir.*`, `positions.tir.*`,
`enc_blocks.*`, `enc_norm.*` have identical keys and shapes in both solutions. The
only extra Solution A tensors are `signal_queries.*` and `xdec_blocks.*`, which the
Stage-3 loader ignores.

**I3 — Stage-3 is arm-agnostic.** The matched config pair
(`resp_tir_roi_local_matched.yaml` vs `resp_tir_roi_crossmae.yaml`) shares
`head_style`, `head_hidden`, `head_init` and `layer_decay`, so a comparison differs
only in which pre-training produced the encoder.

**I4 — no init/RNG side effects.** The cross-attention modules are constructed LAST and
only when a stream actually uses `cross_attn`, so the default path registers no extra
module (identical `state_dict` keys) and consumes no extra RNG draws. Verified: the
Solution C smoke reproduces the recorded `mse_tir 2.0890 / mse_resp 3.2286` exactly.

---

## 4. Code map

| File | Change |
|---|---|
| `core/blocks.py` | `CrossAttention` + `DecoderBlock` ported from `tmp/MultiMAE/multimae/multimae_utils.py`, modernised to `scaled_dot_product_attention` like the existing `Attention`. |
| `core/multimae.py` | `MultiModalMAE.__init__(resp_in_encoder, signal_decoder, cross_attn_depth, query_init)`; `parse_signal_decoder()`; `_as_bool()`; `self.encoder_streams`; lazily built `signal_queries` / `xdec_blocks`; `_init_signal_queries()`; encoder-only token gather + cross-attention decode branch in `forward()`; new `[encoder]` build log; `cross_mae_self_test()`. |
| `core/waveform_model.py` | `WaveformUpsampleHead` (spec §4.1), `head_style` / `head_init`, and a gated Stage-2 head transfer. |
| `utils/optim_factory.py` | `get_num_layer_for_multimae`, `LayerDecayValueAssigner`, `build_layer_decay_assigner`. |
| `engines/finetune.py`, `engines/pretrain.py` | step LR schedule now multiplies each group's `lr_scale` — without this the single cosine curve would overwrite the layer-wise decay. |
| `runners/run_pretrain.py` | `--resp_in_encoder`, `--signal_decoder`, `--cross_attn_depth`, `--query_init`. |
| `runners/run_waveform.py` | `--head_style`, `--head_init`, `--layer_decay` + assigner wiring. |

### Knobs

* `resp_in_encoder` (bool, default `true`). `false` removes the `resp` stream from the
  encoder. Accepts YAML booleans and `--resp_in_encoder false` strings safely.
* `signal_decoder` (`''` | `'self_attn'` | `'cross_attn'`, or `resp=cross_attn,bp=self_attn`,
  or a YAML mapping). Note that a `cross_attn` stream **must** be outside the encoder and a
  `self_attn` stream **must** be inside it — both mismatches raise a named error.
* `cross_attn_depth` (default 2): `DecoderBlock` stack depth.
* `query_init` (`sincos3d` | `random`): time-anchored (1-D temporal sincos) vs
  trunc-normal query initialisation.
* Stage-3 `head_style` (`linear` | `transposed_conv`), `head_init` (`transfer` | `none`),
  `layer_decay` (float, `1.0` = off).

### Configs

* `configs/pretrain/stage2_local_tir_roi_resp.yaml` — Solution C (unchanged).
* `configs/pretrain/stage2_local_tir_roi_crossmae.yaml` — Solution A.
* `configs/finetune/resp_tir_roi_local.yaml` — historical Stage-3 (unchanged).
* `configs/finetune/resp_tir_roi_local_matched.yaml` — matched arm 1 (Solution C encoder).
* `configs/finetune/resp_tir_roi_crossmae.yaml` — matched arm 2 (Solution A encoder).

All five Stage-3/Stage-2 configs ship the spec §4 recipe: `head_style: transposed_conv`,
`head_init: none`, `layer_decay: 0.75`.

---

## 5. Running

```bash
cd code
# Solution A pre-training (the encoder Solution A Stage-3 needs)
python -u runners/run_pretrain.py -c configs/pretrain/stage2_local_tir_roi_crossmae.yaml \
    2>&1 | tee ../output/pretrain/stage2_local_tir_roi_crossmae/train.log

# matched Stage-3 arms (identical except the checkpoint)
python -u runners/run_waveform.py -c configs/finetune/resp_tir_roi_local_matched.yaml
python -u runners/run_waveform.py -c configs/finetune/resp_tir_roi_crossmae.yaml
```

Smoke (fast, `--warmup_epochs 0` is required for a 1-epoch run):

```bash
python -u runners/run_pretrain.py -c configs/pretrain/stage2_local_tir_roi_crossmae.yaml \
    --epochs 1 --warmup_epochs 0 --max_entries 2 --batch_size 2 --num_workers 0 \
    --output_dir /tmp/tir_roi_crossmae_probe
```

---

## 6. Verification

* `python -m core.multimae` — three self-tests, all pass:
  `mask_self_test`, `spectral_self_test`, `cross_mae_self_test`. The last asserts the two
  properties that would otherwise fail silently: **no resp token reaches the encoder**
  (measured as the encoder input length, not by reading a flag) and the **RESP-only loss
  produces a non-zero encoder gradient** (a fully masked stream would send exactly zero
  through a self-attention decoder built from its own visible tokens).
* Regression: the Solution C smoke reproduces the recorded `mse_tir 2.0890 /
  mse_resp 3.2286` bit-for-bit.
* Solution A smoke: `[encoder] streams fed to the encoder: ['tir']`, params 121.2 M vs
  102.3 M for Solution C (the ~19 M cross-attention decoder), trains stably.
* Config parity: pre-train pair differs only in `resp_in_encoder`, `signal_decoder`,
  `cross_attn_depth`, `query_init`, `output_dir`; Stage-3 pair only in
  `finetune`, `output_dir`.
* Encoder parity: identical key sets and shapes; the only Solution A-only tensors are
  `signal_queries.*` / `xdec_blocks.*`.
* Stage-3 smoke: prints `head_init='none': the Stage-2 decoder head is NOT transferred`,
  `[layer_decay] decay 0.75 over 12 encoder blocks -> 14 groups; lr scale 0.02376
  (tokenizer) .. 1 (top)`, and trains.

---

## 7. Notes / limits

* Solution A's `mask_ratio_resp` is **unused** (the loss covers every time slot); the
  build log says so.
* The cross-attention head cannot inherit the Stage-2 decoder head, so the matched pair
  ships `head_init: none` for **both** arms — otherwise the comparison would be
  confounded by a one-sided head transfer.
* `query_init: sincos3d` anchors query *i* to the same instant as physical token *i*;
  `random` is available as an ablation.
* Layer-wise LR decay places the tokenizer (`adapters.*`, `positions.*`) at layer 0
  (most decayed) and `enc_norm` / `waveform_head` at the top layer (scale 1.0), matching
  the VideoMAE recipe.
* Solution A pre-training costs more per step than Solution C (an extra ~19 M-parameter
  decoder and its attention), and its initial `mse_resp` is **higher** (~5.7 vs ~3.2) —
  that difference *is* the removed shortcut.
