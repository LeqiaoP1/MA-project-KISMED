# Solution A: Asymmetric Cross-MAE

This document is the design contract for the thermal-ROI respiration
pre-training variant implemented by `core.multimae.MultiModalMAE`. It describes
the asymmetric pre-training path and its compatibility with the existing
waveform fine-tuning model. The shared-encoder alternative remains available as
Solution C.

## 1. Objective and anti-shortcut constraint

The model reconstructs a continuous respiration waveform from a thermal
infrared (TIR) facial video clip. In the shared-encoder design, visible
respiration tokens can make the masked respiration objective a local
interpolation problem. Solution A removes that shortcut structurally:

* `resp_in_encoder: false` means that only visible TIR tokens enter the ViT
  encoder.
* Respiration is a reconstruction target, not an encoder input.
* A learned query for each output time slot cross-attends to the complete
  encoded visual sequence.

This is an inductive bias, not proof that the encoder learns respiratory
features. The training plan must include the ablations in §8.

## 2. Data and geometry contracts

### 2.1 TIR input

The dataset supplies false-colour, three-channel TIR frames; the channels must
not be silently converted to grayscale. The model input is:

```text
x_tir: [B, 3, T, H, W]
```

The default local configuration uses an 8-second clip, 25 fps, `input_size=64`,
`tubelet=(2, 16, 16)`, and `temporal_stride=2`:

```text
T          = 100 frames
H, W       = 64, 64
sig_kernel = 16 samples
L          = 800 respiration samples at fs=100 Hz
grid_t     = T / tubelet_t = 50
n_signal   = L / sig_kernel = 50
n_visual   = grid_t * (H/16) * (W/16) = 800
```

The alignment identity is mandatory:

```text
tubelet_t * temporal_stride / fps == sig_kernel / fs
T / tubelet_t == L / sig_kernel
```

`MultiModalMAE._check_space_time_alignment()` must reject a mismatch at model
construction time. The alternative stride-1 geometry is `T=200`,
`sig_kernel=8`, `grid_t=100`, `L=800`, and `n_visual=1600`; checkpoints from
different temporal strides are not interchangeable.

The tubelet projection maps the input to:

```text
tubelet features: [B, D, T/2, H/16, W/16]
visual tokens:    [B, n_visual, D]
```

The implementation must preserve the `(time, height, width)` flattening order
and use the matching 3-D positional embedding. Shape assertions are required
at the input, adapter output, token gather, decoder output, and waveform head.

### 2.2 ROI and synchronization

The default `nose_mouth` ROI uses the 12 one-indexed landmarks
`(9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26)`, equivalent to zero-based
indices `(8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25)`. This is not the
contiguous range `18..28`. Other named presets, including `nostrils`, are
supported by `resolve_roi_landmarks()`.

The ROI box is static for a clip and is computed from all source frames before
temporal decimation. The configured `roi_padding` must be applied consistently in
pre-training and fine-tuning. Missing, invalid, or out-of-frame landmarks must
fail the sample with an explicit dataset error; they must not silently produce an
arbitrary crop.

Respiration timestamps must be aligned to the kept video window before
resampling. Resampling uses the configured source sampling rate and
`fs=100 Hz`; the implementation must define its interpolation/anti-aliasing
policy and reject clips with insufficient or non-monotonic timestamps.

### 2.3 Target normalization and quality filtering

For a valid clip, retain the raw target statistics and expose both:

```text
y_norm = (y - mean_y) / (std_y + eps)
target_mean = mean_y
target_std  = std_y
```

The Stage-1 and Stage-3 regression objectives operate on `y_norm`. The
statistics are required for optional reconstruction in volts and must not be
discarded. Use one explicit quality criterion, not an ambiguous “std or
spread” choice: the dataset rejects a clip when
`max(y) - min(y) < min_signal_spread` (the shipped TIR-ROI/RESP configs use
`0.1 V`).
The guard is applied before normalization.

Train, validation, and test splits must be subject-disjoint. Unit tests must
assert that the subject-ID intersections of every pair of splits are empty.

## 3. Stage 1: asymmetric pre-training

### 3.1 Masking and encoder

The default TIR mask ratio is `0.90`. Masking is performed over visual
tubelets, and the mask tensor must identify the exact flattened token
positions. The encoder receives only the visible TIR tokens:

```text
z_v = Encoder(x_tir_visible)       # [B, n_visible, D]
```

The response mask ratio is unused by the cross-attention response path because
the response loss covers all `L` target samples. The implementation must keep
the TIR token order and provide an `ids_restore` mapping for the pixel decoder.

### 3.2 TIR pixel decoder

The pixel decoder reconstructs masked TIR tubelets. It must:

1. project encoder features to decoder width;
2. insert learned mask tokens at the positions given by `ids_restore`;
3. add decoder positional embeddings in the restored `(time, height, width)`
   order;
4. predict one pixel vector per restored tubelet; and
5. compute MSE only at masked positions.

The exact contract is:

```text
pred_tir_masked: [B, n_masked, tubelet_t * 16 * 16 * 3]
gt_tir_masked:   [B, n_masked, tubelet_t * 16 * 16 * 3]
L_tir = MSE(pred_tir_masked, gt_tir_masked)
```

### 3.3 Respiration cross-attention decoder

Create `L=800` learned query positions, initialized by the configured
`query_init` (`sincos3d` is the default and uses the temporal component; random
initialization is supported for ablation). Add the query positional signal at
every forward pass. The decoder computes:

```text
q_resp: [B, L, D]
k_resp = v_resp = z_v: [B, n_visible, D]
h_resp = CrossAttention(q_resp, k_resp, v_resp)
y_pred = Linear(h_resp): [B, L]
L_resp = MSE(y_pred, y_norm)
```

The decoder is non-causal and may use all encoded video tokens for every output
time slot. Its query length must be derived from the dataset/model geometry,
not hard-coded independently of `L`.

## 4. Stage-1 objective

```text
L_ssl = lambda_tir * L_tir + lambda_resp * L_resp
```

The shipped recipe uses `lambda_tir=1.0` and a configured
`lambda_resp` in the range `0.5..1.0`. Losses must be finite and their scales
must be logged separately. Stage 1 intentionally omits MR-STFT; spectral
losses are introduced only in supervised fine-tuning after numerical behavior
has been checked.

## 5. Stage 2: waveform fine-tuning

Discard both Stage-1 decoder heads and retain only the compatible encoder
parameters (`adapters.*`, `positions.*`, `enc_blocks.*`, and `enc_norm.*`).
The Stage-3 `MultiModalWaveformRegressor` consumes the same TIR stream with
zero masking.

The regression head must first aggregate the spatial visual grid while
preserving the temporal grid:

```text
encoder tokens: [B, grid_t * grid_h * grid_w, D]
temporal tokens: [B, grid_t, D]       # explicit spatial reduction
waveform:       [B, L]                # temporal upsampling to target length
```

The configured `head_style=transposed_conv` performs the temporal expansion;
the linear head remains available for compatibility. The model must assert the
expected `grid_t` and output `L` rather than silently interpolating a
desynchronized sequence.

Fine-tune all encoder parameters end-to-end with the configured layer-wise
learning-rate decay (`layer_decay=0.75` in the matched recipe). The optimizer
must preserve each parameter group's decay multiplier when applying the global
schedule.

## 6. Fine-tuning loss

Predictions and targets are compared in normalized units:

```text
L_ft = lambda_pearson * (1 - r_eps)
     + lambda_spec * L_MR-STFT(y_pred, y_norm)
```

Do not include a standard-deviation penalty on the normalized target: its
standard deviation is approximately one by construction, making the original
`|std(y_pred)-std(y_true)|` term redundant. If amplitude calibration in volts
is required, add a separate denormalized metric or an explicitly weighted
voltage-domain loss using `target_std`; do not mix the two semantics silently.

`r_eps` must use a variance floor/epsilon and define a finite fallback for
near-constant predictions or targets. MR-STFT must define window sizes,
hop sizes, FFT sizes, log-magnitude epsilon, and behavior for clips shorter
than a window.

## 7. Configuration and compatibility

Solution A is selected by:

```yaml
resp_in_encoder: false
signal_decoder: resp=cross_attn
cross_attn_depth: 2
query_init: sincos3d
```

`cross_attn` is valid only for a stream outside the encoder; a self-attention
decoder is valid only for a stream inside it. Invalid combinations must raise
a named `ValueError`. The default Solution C path must retain its module names,
state-dict keys, initialization order, and RNG behavior.

The Solution A and Solution C encoders must remain load-compatible with the
same Stage-3 model. Cross-attention-only tensors (`signal_queries.*` and
`xdec_blocks.*`) are pre-training extras and must be ignored by the Stage-3
encoder loader.

## 8. Required verification

The implementation is not considered verified until the following tests pass:

* input, token, mask, decoder, and waveform output shape assertions;
* exact mask-ratio and `ids_restore` ordering checks;
* no respiration token reaches the Solution A encoder;
* all response queries attend to visual encoder output and produce `[B, L]`;
* mismatched `T/L/sig_kernel` geometry raises before training;
* finite Pearson and MR-STFT losses for normal and near-constant inputs;
* zero-variance/spread filtering and raw-statistics retention;
* subject-disjoint split assertions;
* Solution A/C encoder checkpoint compatibility;
* regression tests for the default Solution C path and `cross_mae_self_test()`;
* ablations for visual-only reconstruction, response loss, mask ratio, shuffled
  targets, and direct versus scheduled transition from 90% masking to 0%.

Report normalized waveform metrics (MAE/RMSE and Pearson correlation), spectral
error, respiration-rate error, and denormalized voltage metrics when amplitude
recovery is claimed. A successful pixel reconstruction alone is not evidence
of physiological recovery.
