# 02 — Model Architecture

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/core/` on 2026-10-04;
> updated 2026-10-05 to incorporate design documents in `code/notes/design/`.
> All tensor shapes are documented from actual `forward()` implementations.
> Design-level contracts from `code/notes/design/` are marked **[design]**.

---

## 1. Model Registry — `core/registry.py`

timm-style global registry. `@register_model` stores a factory by name. `models/build.py::create_model(name, **kwargs)` is the single public entry point for YAML-driven runners.

| Registered name | Class | Stage |
|----------------|-------|-------|
| `project_vit_small_patch16_224` | `ProjectViT` | 1 (baseline) |
| `project_vit_base_patch16_224` | `ProjectViT` | 1 (baseline) |
| `project_vit_large_patch16_224` | `ProjectViT` | 1 (baseline) |
| `project_vit_huge_patch16_224` | `ProjectViT` | 1 (baseline) |
| `project_multimae_small` | `MultiModalMAE` | 2 |
| `project_multimae_base` | `MultiModalMAE` | 2 |
| `project_multimae_large` | `MultiModalMAE` | 2 |
| `project_multimae_huge` | `MultiModalMAE` | 2 |

Stage-3 and AU-probe models use dedicated builders (`build_waveform_model`, `build_au_probe_model`), not `create_model`.

---

## 2. Transformer Blocks — `core/blocks.py`

### `Attention` (self-attention)

**Forward**: `x: [B, N, C]` -> `[B, N, C]`

```
qkv = Linear(C, 3C) -> [B, N, 3, heads, head_dim]
q, k, v = qkv.unbind(0)     # each [B, heads, N, head_dim]

Fast path (torch >= 2.0):
  x = F.scaled_dot_product_attention(q, k, v)   # never materialises [B,heads,N,N]
Fallback:
  attn = (q @ k.T * scale).softmax(-1) @ v      # [B,heads,N,N] materialised

x = x.transpose(1,2).reshape(B, N, C)
x = proj(x)
```

> **Critical**: At 224 px (9800 tokens), the naive path requires ~9 GB per block per sample. SDPA (torch >= 2.0) is mandatory for the full-resolution geometry.

### `Block` (pre-norm encoder block)

```
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(Mlp(LayerNorm(x)))
```

**Forward**: `[B, N, D]` -> `[B, N, D]`

### `Mlp`

`fc1(in -> hidden) -> GELU -> Dropout -> fc2(hidden -> out)`. Default `hidden = in * mlp_ratio` (typically 4).

---

## 3. Input Adapters

### `TubeletEmbed` (`core/multimae.py`) — Stage-2 / Stage-3 primary adapter

**Forward**: `x: [B, C, T, H, W]` -> `[B, Gt*Gh*Gw, D]`

```python
patch_embed = nn.Conv3d(C, D, kernel_size=(t,ph,pw), stride=(t,ph,pw))
x = patch_embed(x)            # [B, D, Gt, Gh, Gw]
x = x.flatten(2).transpose(1, 2)  # [B, Gt*Gh*Gw, D]
```

The implementation must preserve the `(time, height, width)` flattening order and use the matching 3-D positional embedding. Shape assertions are required at the adapter output.

Grid sizes at standard geometries:

| `input_size` | `num_frames` | `tubelet` | `stride` | `Gt` | `Gh=Gw` | `n_visual` |
|-------------|-------------|----------|----------|------|---------|-----------|
| 64 | 100 | 2,16,16 | 1 | 50 | 4 | 800 |
| 64 | 200 | 2,16,16 | 2 | 50 | 4 | 800 |
| 224 | 100 | 2,16,16 | 1 | 50 | 14 | 9800 |

**TIR-ROI geometry**:

```
T = 100 frames  (8 s x 25 fps / temporal_stride 2)
H, W = 64, 64
sig_kernel = 16 samples
L = 800 respiration samples at fs=100 Hz
grid_t = T / tubelet_t = 50
n_signal = L / sig_kernel = 50
n_visual = grid_t * (H/16) * (W/16) = 800
```

Alignment identity (enforced at construction):

```
tubelet_t * temporal_stride / fps == sig_kernel / fs
T / tubelet_t == L / sig_kernel
```

### `SignalEmbed` (`core/multimae.py`) — Stage-2 1-D signal adapter

**Forward**: `x: [B, 1, S]` -> `[B, Ns, D]`

```python
embed = nn.Conv1d(1, D, kernel_size=sig_kernel, stride=sig_kernel)
x = embed(x)          # [B, D, Ns]
x = x.transpose(1, 2)  # [B, Ns, D]
```

Default: `sig_kernel=8`, `S=400` -> `Ns=50`.

### Scaffold adapters (`core/input_adapters.py`) — used by `ProjectViT` only

| Class | Forward signature |
|-------|------------------|
| `PatchedInputAdapter` | `[B, C, H, W]` -> `[B, N, D]` via Conv2d |
| `SignalInputAdapter` | `[B, C, T]` -> `[B, N, D]` via Conv1d |
| `SemSegInputAdapter` | `[B, num_classes, H, W]` -> `[B, N, D]` via 1x1 conv + patch embed |

---

## 4. Position Embeddings — `utils/pos_embed.py`

`_PosMask(num_tokens, dim_tokens)` stores per-stream learnable `pos_embed [1, N, D]` and `mask_token [1, 1, D]`.

| Modality | Type | Shape | Function |
|----------|------|-------|----------|
| `rgb` / `tir` / `rgb2d` (video) | 3-D sincos (t,h,w) | `[1, Gt*Gh*Gw, D]` | `get_3d_sincos_pos_embed` |
| BP / RESP | 1-D sincos (temporal) | `[1, Ns, D]` | Same time block as 3-D |
| `ProjectViT` | 2-D sincos | `[1, N+1, D]` | `get_2d_sincos_pos_embed` |

3-D embedding dim split (VideoMAE convention): `d_t = d_h = 2*(D//6)`, `d_w = D - 2*d_t`. For D=768: d_t=d_h=256, d_w=256.

`pos_init='sincos3d'` (default) initialises with sincos; `'random'` keeps trunc_normal (std=0.02).

---

## 5. Output Adapters — `core/output_adapters.py`

### `SpatialOutputAdapter`

**Forward**: `x: [B, N, D]` -> `[B, N, num_channels]`

```python
head = nn.Linear(D, num_channels)
```

`num_channels` = `tubelet_t * ph * pw * C` for video; `sig_kernel` for signals.

> `init(dim_tokens)` must be called before `forward()`. If skipped, `head is None` -> NoneType error with no context.

---

## 6. Stage-1 Baseline: `ProjectViT` — `core/model.py`

Standard 2-D ViT. Registered as `project_vit_{small|base|large|huge}_patch16_224`.

| Variant | `embed_dim` | `depth` | `num_heads` |
|---------|------------|---------|------------|
| small | 384 | 12 | 6 |
| base | 768 | 12 | 12 |
| large | 1024 | 24 | 16 |
| huge | 1280 | 32 | 16 |

### `forward(x)` tensor flow

```
x: [B, C, H, W]
  -> Conv2d(C, D, k=p, s=p) -> [B, D, H/p, W/p]
  -> flatten(2).transpose(1,2) -> [B, N, D]   N = (H/p)*(W/p)
  -> cat([cls_token, x], dim=1) -> [B, N+1, D]
  -> + pos_embed -> Dropout
  -> Block x depth -> LayerNorm -> [B, N+1, D]
  -> x[:, 0] -> [B, D] -> head -> [B, num_classes]
```

`forward_waveform(x)`: `[B, D] -> Linear(D, output_len) -> [B, output_len]`
(CLS-based baseline; the implementation plan recommends a decoder over all patch tokens for fine-grained waveforms.)

---

## 7. Stage-2: `MultiModalMAE` — `core/multimae.py`

### 7.1 Stage-2 pre-training objective

**[design]** — `SolutionC_EarlyTokenMixing_stage2.md`

The Stage-2 SSL path is a shared-encoder masked autoencoder: the visible tokens of every stream enter one ViT, and each stream is reconstructed by the shared self-attention decoder over its own visible tokens plus `mask_token`s. TIR video and the 1-D physiological signal are treated symmetrically.

### 7.2 Construction-time validation

1. Rejects all-visual or all-signal `streams`.
2. Validates `tubelet` divides `input_size` and `num_frames`.
3. `_check_space_time_alignment()`: enforces `sig_kernel/fs == tubelet_t*temporal_stride/fps` AND `n_signal == Gt`. Raises `ValueError` on mismatch.
4. `pos_init` in `{'sincos3d', 'random'}`.
5. `target_norm` in `{'token', 'clip', 'none'}` (`'none'` = IDENTITY: the dataset already normalised the 1-D stream). NON-ROI paths only — `run_pretrain.pin_roi_target_norm` forces `'none'` for `tir_roi`/`rgb_roi`.

### 7.3 Sub-modules

| Key | Module |
|-----|--------|
| `adapters` | `nn.ModuleDict` — per-stream `TubeletEmbed` or `SignalEmbed` |
| `positions` | `nn.ModuleDict` — per-stream `_PosMask` (pos_embed + mask_token) |
| `enc_blocks` | `nn.ModuleList` — `enc_depth` x `Block` |
| `enc_norm` | `nn.LayerNorm` — post-encoder norm |
| `dec_blocks` | `nn.ModuleList` — `dec_depth=2` x `Block` (TIR pixel decoder) |
| `dec_norm` | `nn.LayerNorm` — post-decoder norm |
| `heads` | `nn.ModuleDict` — per-stream Linear head `D -> patch_dim` |

### 7.4 `forward(x)` — shared-encoder

```
Input: {'tir': [B,3,T,H,W], 'resp': [B,1,S]}  (or rgb,bp, etc.)

Step 1 — Visual input preprocessing + tokenise (all streams):
  for visual s: x[s] = zscore_clip(x[s])                 # per-channel z over (T,H,W), per sample
  tokens_s = adapters[s](x[s]) + positions[s].pos_embed   # [B, N_s, D]

Step 2 — Constant-budget masking (ADR-002):
  plan = ConstantBudgetApportioner.sample_step(counts_generator)  # mode + counts
  for s: ids_keep[s] = place_video/place_signal(plan.kept_tokens[s],
                            placement_generator)                   # [B, k_s]
  mask_s = 1 - scatter(ids_keep[s])                                 # [B, N_s], 1 = masked

Step 3 — Concatenate visible tokens (ALL streams in encoder):
  z_vis = cat([tokens_s[:, ids_keep[s], :] for s in streams], dim=1)
  assert z_vis.shape == [B, n_budget, D]              # contract 1

Step 4 — Shared ViT encoder:
  for blk in enc_blocks: z_vis = blk(z_vis)
  z_vis = enc_norm(z_vis)                       # [B, n_budget, D]

Step 5 — Per-stream decoder input: encoder latents + this stream's mask tokens:
  dec_s = cat([z_vis, mask_token x n_masked_s], dim=1)
  dec_s[z_vis part]  += 0                        # z already carries positions
  dec_s[mask part]   += positions[s].pos_embed[ids_masked_s]

Step 6 — Shared lightweight self-attention decoder (dec_depth=2), ONE stream at a
          time; the z PREFIX is what gives a 0-kept stream a gradient path to the
          encoder (contract 7):
  for blk in dec_blocks: dec_s = blk(dec_s)
  pred_masked_s = heads[s](dec_s[:, n_budget:])
  pred_s = scatter(pred_masked_s)                # [B, N_s, patch_dim_s]

Step 7 — Norm-pix targets + weighted masked MSE loss
Returns: {'loss', 'losses', 'losses_mse', 'masks', 'preds', 'mode', 'kept',
          'kept_units', 'n_batch'}
```

`zscore_clip` (`core/input_adapters.py`) runs AFTER `_fit_time` and BEFORE patch
division, so the statistics describe the clip the adapters actually see. The
datasets keep returning `[0, 1]` — that is the dataset contract, not the network
input. Contract: `design/Video_input_Preprocessing.md`.

`_targets(x, stream, mask)` is **masked-only**: a masked cube is standardised
with its OWN mean/var over its `C*t*ph*pw` values; an unmasked cube carries the
clip-normalised values (no local map), and `MaskedMSELoss` keeps only
`mask == 1`. `mask=None` means "every cube masked" (unit tests only). The 1-D
streams ignore the mask — their label space is the dataset's per-clip z-score
decided by `target_norm`.

### 7.5 Masking — the ADR-002 constant-budget policy

**[design]** — `design/ADR-002-asymmetric-multimodal-masking.md`, implemented in
`core/budget_masking.py` (policy: counts only) + `core/mask_placement.py`
(placement: indices only). There is **one** masking policy; `mask_ratio_*`, the
Dirichlet sampler and the `masking:` selector were removed with it.

**Masking units.** A visual unit is one spatial tube (one patch + all of its
temporal positions), so a visual count is always a multiple of
`tokens_per_unit = Gt`. A physiological unit is one temporal token:

| stream kind | unit | units per stream | tokens per unit | tokens |
| :--- | :--- | ---: | ---: | ---: |
| visual (`tir`, `rgb2d`, `rgb`) | spatial tube | `Gh*Gw` = 36 | `Gt` = 25 | 900 |
| physiological (`bp`, `resp`) | temporal token | `seq_len//sig_kernel` = 250 | 1 | 250 |

**Two modes, one constant budget.** Every step the apportioner draws a mode and
one kept-token count per stream such that
`sum(kept_tokens) == n_budget` (default 350):

| Mode | draw |
| :--- | :--- |
| 1 — pure virtual sensing (p = `p_virtual_sensing`) | `K_p = 0` for every physiological stream; the two visual streams split `n_budget/T` tubes inside `[video_min_tubes, M]` |
| 2 — waveform inpainting | each physiological stream draws `K_p` uniform on `[0, C_p]`; the modulo remainder is charged to the FIRST declared physiological stream (a deliberate asymmetry); the visual residual is split the same clamped way |

Caps and floors are derived from the run's parameters (`M =
floor(video_cap_ratio * units)`, `C_p = floor(physio_cap_ratio * unit count)`),
and three feasibility conditions are validated at CONSTRUCTION (budget a multiple
of `T`; `2m <= n_budget/T <= 2M`; `n_budget - P*C_p >= 2mT`). An infeasible tuple
raises `ValueError` naming the condition and the numbers.

**Placement** (`core/mask_placement.py`) is drawn per SAMPLE: `place_video`
keeps the full temporal column of `k` spatial tubes; `place_signal` keeps `k`
tokens either scattered (uniform without replacement, default) or as contiguous
spans of `physio_span_tokens`. Counts are batch-uniform, placements are not.

| knob | default | meaning |
| :--- | ---: | :--- |
| `n_budget` | 350 | visible encoder tokens on EVERY step |
| `video_cap_ratio` | 0.25 | kept-ratio ceiling per visual stream |
| `physio_cap_ratio` | 0.50 | kept-ratio ceiling per physiological stream |
| `video_min_tubes` | 2 | kept-tube floor per visual stream |
| `p_virtual_sensing` | 0.50 | probability of Mode 1 |
| `spatial_patches` | 36 | tokenizer-derived unit count per visual stream |
| `physio_tokens` | 250 | tokenizer-derived unit count per physiological stream |

`spatial_patches` / `physio_tokens` are derived from the tokenizer; passing them
explicitly is allowed only when it agrees, and the model refuses otherwise.

**RNG contract.** Counts come from a generator seeded by `(base_seed, epoch,
step)` WITHOUT the rank, so every DDP rank derives the same plan without a
collective. Placements add the rank, so samples differ across ranks. Validation
replays a fixed-seed sequence instead of pinning per-stream fractions.

**Loss-key stability.** Every declared stream emits its `losses_mse` entry on
every step and every rank (an explicit zero when it has no masked position). A
0-kept stream's adapter receives no gradient — the adapter is skipped — while
its loss still reaches the encoder through the decoder's `z` prefix.

### 7.6 Geometry variants

| Variant | `embed_dim` | `enc_depth` | `enc_num_heads` |
|---------|------------|------------|----------------|
| small | 384 | 12 | 6 |
| base | 768 | 12 | 12 |
| large | 1024 | 24 | 16 |
| huge | 1280 | 32 | 16 |

`dec_depth=2` (hardcoded default). Decoder uses the same `embed_dim` as encoder.

---

## 8. Stage-3: `MultiModalWaveformRegressor` — `core/waveform_model.py`

### 8.1 Overview

**[design]** — `SolutionC_EarlyTokenMixing_stage3.md`

Stage 3 is the **sensor-failure** condition: only visual input, **no masking**, no 1-D physio stream. The Stage-2 encoder is fine-tuned end-to-end; `freeze_encoder()` exists but is not called.

### 8.2 Sub-modules (identical key names to Stage-2 encoder)

| Key | Description |
|-----|-------------|
| `adapters` (`nn.ModuleDict`) | Per-stream `TubeletEmbed` |
| `positions` (`nn.ModuleDict`) | Per-stream `_PosMask` (pos_embed only; mask_token unused) |
| `enc_blocks` (`nn.ModuleList`) | `enc_depth` x `Block` |
| `enc_norm` (`nn.LayerNorm`) | Post-encoder norm |
| `waveform_head` | `Linear(D,k)` / `Sequential(D->h,GELU,h->k)` / `WaveformUpsampleHead` |

> Key name identity with Stage-2 is critical: `load_stage2_encoder` transfers weights by exact key match.

### 8.3 `forward(x)` tensor flow

**[design]** — `SolutionC_EarlyTokenMixing_stage3.md §Input+Encoder`

```
x: {'rgb': [B,3,T,H,W]}  (or plain Tensor; NO masking anywhere)

For each stream s:
  xt = _fit_time(x[s], num_frames, s)    # trim if T > n; raise ValueError if T < n
  xt = zscore_clip(xt)                   # per-channel z over (T,H,W), per sample
  tok = adapters[s](xt)                  # [B, n_visual, D]
  assert tok.shape[1] == n_visual        # hard geometry check
  tok = tok + positions[s].pos_embed

z = cat(parts, dim=1)                    # [B, S * n_visual, D]
for blk in enc_blocks: z = blk(z)
z = enc_norm(z)

For each stream i:
  zi = z[:, i*n_visual : (i+1)*n_visual]         # [B, n_visual, D]
  zi = zi.reshape(B, grid_t, grid_h*grid_w, D)
  zi = zi.mean(dim=2)                            # [B, grid_t, D]
h = stack(feats, dim=0).mean(dim=0)              # [B, grid_t, D]

pred = waveform_head(h)                          # [B, grid_t, k]
return pred.reshape(B, -1)                      # [B, output_len]
```

`output_len = grid_t * k`; must be a multiple of `grid_t` (enforced at construction).

The SAME `zscore_clip` as Stage 2 (`core/input_adapters.py`) runs before the
adapter, so Stage 3 consumes the exact input space the Stage-2 encoder was
pre-trained on. Every Stage-3 stream is validated visual, so no branch is needed.
There is no masking and no target normalisation in this path.

The model must assert the expected `grid_t` and output `L` rather than silently interpolating a desynchronised sequence.

### 8.4 Head variants

| `head_style` | `head_hidden` | Module |
|-------------|--------------|--------|
| `'linear'` | 0 | `Linear(D, k)` |
| `'linear'` | h > 0 | `Linear(D,h) -> GELU -> Linear(h,k)` |
| `'transposed_conv'` | N/A | `WaveformUpsampleHead` (recommended) |

### 8.5 `WaveformUpsampleHead` tensor flow

Spatial aggregation must preserve the temporal grid before upsampling to the target length.

```
h: [B, grid_t, D]
  -> transpose(1,2)                       -> [B, D, grid_t]
  -> Conv1d(D, hidden, k=3, pad=1)        -> [B, hidden, grid_t]
  -> GELU
  -> ConvTranspose1d(hidden, 1, k=s, s=s) -> [B, 1, grid_t*s]
  -> transpose(1,2)                       -> [B, grid_t, s]
```

`s = samples_per_token`. Segment k covers exactly tubelet window k (time-alignment preserved).

---

## 9. AU Probe: `MultiModalMAEProbe` — `core/au_probe.py`

Identical module keys to Stage-2 (`adapters`, `positions`, `enc_blocks`, `enc_norm`) + linear `head = Linear(D, num_classes)`.

### `forward(x)` tensor flow

```
x: Dict[str, [B, C, T, H, W]]  (visual streams only)

For each stream s:
  xt = _resize_t(x[s], num_frames)    # trailing trim if T > n; raise if T < n
  tok = adapters[s](xt) + positions[s].pos_embed   # [B, N, D]

z = cat(parts, dim=1)                # [B, S*N, D]
for blk in enc_blocks: z = blk(z)
z = enc_norm(z)
z = z.mean(dim=1)                   # [B, D]  (mean-pool; no CLS token)
return head(z)                      # [B, num_classes]
```

Probe mode: `freeze_features()` sets `requires_grad=False` on all non-head params. `set_train(mode)` keeps frozen layers in `eval()`.

---

## 10. Loss Functions

### Masked Reconstruction — `core/criterion.py`

All: `pred: [B,N,D]`, `target: [B,N,D]`, `mask: [B,N]` (1 = masked = compute loss).

| Loss | Formula |
|------|---------|
| `MaskedMSELoss` | `mean_masked(mean_D((pred-target)^2))` |
| `MaskedL1Loss` | `mean_masked(mean_D(\|pred-target\|))` |
| `MaskedCrossEntropyLoss` | `mean_masked(CE(pred_BNC, target_BN))` |

Denominator: `mask.sum() + 1e-6`.

Targets come from `MultiModalMAE._targets`: for VISUAL streams only the MASKED
cubes are per-cube z-scored; unmasked cubes carry the clip-normalised values
(`design/Video_input_Preprocessing.md`). The 1-D targets are decided by
`target_norm` (`token` / `clip` / `none`) and are unaffected by `mask`.

### Stage-3 Waveform Losses — `core/waveform_losses.py`

All accept `[B,T]` or `[B,1,T]` (channel squeezed via `_to_1d`).

**`PearsonLoss`**: `mean(1 - r)` where `r = (p_norm . t_norm) / (||p_norm|| * ||t_norm||)`. Must use a variance floor/epsilon and define a finite fallback for near-constant predictions or targets.

**`L_time` (point-wise Huber)**: `torch.nn.SmoothL1Loss(beta=huber_beta)`, `mean` reduction, on the raw point-wise error (no per-sample centring). `huber_beta` is in TARGET-STD units; the label is z-scored per clip, so its std is ~1 and the default `1.0` sits at ~1 std — quadratic inside that band, linear beyond.
Amplitude-only (std-matching) is NOT used: it is blind to POLARITY and SHAPE (an inverted waveform has the same std, mean and magnitude spectrum). A point-wise term does charge a constant phase offset, which is accepted here — the waveform, not just the rate, is the deliverable.

**`MultiResolutionSTFTLoss`**: for each `n_fft` in `fft_sizes`:

```
P_mag = |STFT(pred, n_fft, hann)|    [B, n_fft//2+1, frames]
T_mag = |STFT(target, n_fft, hann)|
sc = ||P_mag - T_mag|| / (||T_mag|| + eps)             (spectral convergence)
lm = mean(|log(P_mag + 1e-6) - log(T_mag + 1e-6)|)    (log-magnitude L1)
total += sc + lm
```

Degenerate-target guard: skip samples with `||T_mag|| <= 1e-6 * max_batch(||T_mag||)`.

**[design, historical]** — `SpanMask_PhysioSignals.md §5`: while this term was also wired into Stage 2 (removed 2026-11-09), it was informative **only after span masking removed the interpolation shortcut**. Under scattered masking its optimum is already achieved by the linear interpolator (`spec_interp 0.98`); at the weights tried it produced `spec_resp 3.7e8` and destroyed the model. Under a 4 s span the interpolator degrades to `spec_interp 2.37` — a magnitude-spectral loss can then detect spectral damage that amplitude loss forgives.

**[design]** — `SpanMask_PhysioSignals.md §3`: MSE and MR-STFT are complementary and neither is sufficient alone:

| Error type | MSE | MR-STFT |
|-----------|----|--------|
| 1 s boxcar smoothing | 0.090 (best in table!) | 2.330 (bad) |
| Sign flip (-z) | 4.000 (worst) | 0.000 (free) |
| 1 s time shift | 2.557 | 2.016 |

**`WaveformJointLoss`**: `L = alpha*L_time + beta*PearsonLoss + gamma*MultiResolutionSTFTLoss`, where `L_time` is the point-wise Huber described above (`huber_beta` in target-std units).  
Default alpha=beta=gamma=1.0. Weights stored as `register_buffer`.

---

## 11. Stage-2 -> Stage-3 Weight Transfer — `core/waveform_model.py::load_stage2_encoder`

| Source key prefix | Destination |
|-------------------|-------------|
| `enc_blocks.*` | `enc_blocks.*` |
| `enc_norm.*` | `enc_norm.*` |
| `adapters.<s>.*` | `adapters.<s>.*` |
| `positions.<s>.pos_embed` | `positions.<s>.pos_embed` |
| `heads.<target>.*` | `waveform_head.*` (if `head_init='transfer'`) |

Geometry guard: `pos_embed` shape mismatch raises `ValueError` with actionable message.

Stage-1 VideoMAE -> Stage-2: `canonicalise_vit_state_dict` handles the official MAE/VideoMAE and the HF-transformers VideoMAE checkpoint layouts. The 3-D `patch_embed.proj.weight [D, C, t, ph, pw]` (VideoMAE tubelet) is copied **verbatim**; a tubelet mismatch raises. The legacy 2-D boxcar inflation was removed with the non-VideoMAE sources.

---

## 12. Design Config Map

| Config file | Purpose |
|------------|---------|
| `configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml` | Stage 2, shared-encoder MAE at the ADR-002 geometry (the live arm) |
| `configs/pretrain/stage2_hpc_tir_roi_resp.yaml` | Stage-2 record (single visual stream: refused under contract 5) |
| `configs/finetune/resp_tir_roi_local.yaml` | Historical Stage-3 |
| `configs/finetune/resp_tir_roi_local_matched.yaml` | Stage 3, `head_style: transposed_conv`, `head_init: none`, `layer_decay: 0.75` |

### Masking YAML keys (ADR-002 — the whole surface)

```yaml
n_budget: 350              # constant visible encoder tokens per step
video_cap_ratio: 0.25      # kept-ratio ceiling per visual stream
physio_cap_ratio: 0.50     # kept-ratio ceiling per physiological stream
video_min_tubes: 2         # kept-tube floor per visual stream
p_virtual_sensing: 0.50    # probability of Mode 1
spatial_patches: 36        # tokenizer-derived, asserted at build
physio_tokens: 250         # tokenizer-derived, asserted at build
physio_placement: scattered  # 'scattered' | 'span' (placement, not policy)
physio_span_tokens: 0      # required only when physio_placement: span
```
