# 02 — Model Architecture

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/core/` on 2026-10-04;
> updated 2026-10-05 to incorporate design documents in `code/design/`.
> All tensor shapes are documented from actual `forward()` implementations.
> Design-level contracts from `code/design/` are marked **[design]**.

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

### `CrossAttention`

**Forward**: `x: [B, N, C]`, `context: [B, M, C]` -> `[B, N, C]`

Queries from `x`; keys/values from `context`. Used in `DecoderBlock` for **Solution A** cross-stream decoding (RESP stream attending to the full visual encoder output). Ported from `tmp/MultiMAE/multimae/multimae_utils.py`, modernised to use `scaled_dot_product_attention`.

**[design]** — `SolutionA_vs_SolutionC.md §4`: the cross-attention modules are constructed **last** and only when a stream actually uses `cross_attn`, so the default Solution C path registers no extra module (identical `state_dict` keys) and consumes no extra RNG draws.

### `Block` (pre-norm encoder block)

```
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(Mlp(LayerNorm(x)))
```

**Forward**: `[B, N, D]` -> `[B, N, D]`

### `DecoderBlock` (self-attn + cross-attn + MLP)

```
x = x + DropPath(self_attn(LayerNorm(x)))
x = x + DropPath(cross_attn(query_norm(x), context_norm(context)))
x = x + DropPath(Mlp(LayerNorm(x)))
```

**Forward**: `x: [B, N, D]`, `context: [B, M, D]` -> `[B, N, D]`

Used exclusively in Solution A (`signal_decoder: cross_attn`). Not constructed on the Solution C path.

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

**[design]** — `SolutionA_asymCrossMAE.md §2.1`: the implementation must preserve the `(time, height, width)` flattening order and use the matching 3-D positional embedding. Shape assertions are required at the adapter output.

Grid sizes at standard geometries:

| `input_size` | `num_frames` | `tubelet` | `stride` | `Gt` | `Gh=Gw` | `n_visual` |
|-------------|-------------|----------|----------|------|---------|-----------|
| 64 | 100 | 2,16,16 | 1 | 50 | 4 | 800 |
| 64 | 200 | 2,16,16 | 2 | 50 | 4 | 800 |
| 224 | 100 | 2,16,16 | 1 | 50 | 14 | 9800 |

**TIR-ROI geometry** (Solution A / `SolutionA_asymCrossMAE.md §2.1`):

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
| RGB / TIR (video) | 3-D sincos (t,h,w) | `[1, Gt*Gh*Gw, D]` | `get_3d_sincos_pos_embed` |
| BP / RESP / EDA | 1-D sincos (temporal) | `[1, Ns, D]` | Same time block as 3-D |
| `ProjectViT` | 2-D sincos | `[1, N+1, D]` | `get_2d_sincos_pos_embed` |

3-D embedding dim split (VideoMAE convention): `d_t = d_h = 2*(D//6)`, `d_w = D - 2*d_t`. For D=768: d_t=d_h=256, d_w=256.

`pos_init='sincos3d'` (default) initialises with sincos; `'random'` keeps trunc_normal (std=0.02).

**[design]** — `SolutionA_asymCrossMAE.md §3.3`: the Solution A cross-attention RESP decoder creates `L` learned query positions initialised by `query_init` (`sincos3d` uses the temporal component; `random` is an ablation knob). The positional signal is re-added at every forward pass.

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

### 7.1 Two Pre-training Variants: Solution C (default) and Solution A

**[design]** — `SolutionA_vs_SolutionC.md`, `SolutionA_asymCrossMAE.md`

The codebase supports two structurally different Stage-2 pre-training objectives that share **one encoder interface** and differ only in how the physiological stream is handled:

| | Solution C (default, shipped) | Solution A (asymmetric cross-MAE) |
|---|---|---|
| Config switch | `resp_in_encoder: true` (default) | `resp_in_encoder: false` |
| Encoder input | Visible TIR + visible RESP tokens | Visible TIR tokens **only** |
| RESP decoding | Shared self-attention decoder (own visible tokens + mask tokens) | **Cross-attention decoder**: learned per-time-slot queries attend to full visual encoder output |
| RESP loss support | Masked resp tokens only | **All** time slots (all-ones mask) |
| TIR decoding | Self-attention pixel decoder | Self-attention pixel decoder (unchanged) |
| Extra Stage-2 modules | None | `signal_queries.*` + `xdec_blocks.*` |
| Encoder keys/shapes | `adapters.*`, `positions.*`, `enc_blocks.*`, `enc_norm.*` | **Identical** — either checkpoint loads into the same Stage-3 model |

**Why Solution A** (`SolutionA_vs_SolutionC.md §1`): under Solution C with scattered random masking, a 5-line linear interpolator using **zero video** reaches `mse_resp 0.129` / `spec_resp 0.98`, beating the trained 102 M-parameter model (`1.052` / `~5.0`). Solution A removes the shortcut structurally: the physio stream never enters the encoder. The initial `mse_resp` is higher (~5.7 vs ~3.2) — that difference *is* the removed shortcut.

### 7.2 Construction-time validation

1. Rejects all-visual or all-signal `streams`.
2. Validates `tubelet` divides `input_size` and `num_frames`.
3. `_check_space_time_alignment()`: enforces `sig_kernel/fs == tubelet_t*temporal_stride/fps` AND `n_signal == Gt`. Raises `ValueError` on mismatch.
4. `pos_init` in `{'sincos3d', 'random'}`.
5. `target_norm` in `{'token', 'clip', 'none'}` (`'none'` = IDENTITY: the dataset already normalised the 1-D stream). NON-ROI paths only — `run_pretrain.pin_roi_target_norm` forces `'none'` for `tir_roi`/`rgb_roi`.
6. `resp_in_encoder` / `signal_decoder` consistency: a `cross_attn` stream must be **outside** the encoder; a `self_attn` stream must be **inside** it. Invalid combinations raise a named `ValueError`.

### 7.3 Sub-modules

| Key | Module | Present in |
|-----|--------|-----------|
| `adapters` | `nn.ModuleDict` — per-stream `TubeletEmbed` or `SignalEmbed` | Both |
| `positions` | `nn.ModuleDict` — per-stream `_PosMask` (pos_embed + mask_token) | Both |
| `enc_blocks` | `nn.ModuleList` — `enc_depth` x `Block` | Both |
| `enc_norm` | `nn.LayerNorm` — post-encoder norm | Both |
| `dec_blocks` | `nn.ModuleList` — `dec_depth=2` x `Block` (TIR pixel decoder) | Both |
| `dec_norm` | `nn.LayerNorm` — post-decoder norm | Both |
| `heads` | `nn.ModuleDict` — per-stream Linear head `D -> patch_dim` | Both |
| `signal_queries.*` | `nn.Parameter` — learned RESP query positions | **Solution A only** |
| `xdec_blocks.*` | `nn.ModuleList` — `DecoderBlock x cross_attn_depth` | **Solution A only** |

### 7.4 Solution C `forward(x)` — shared-encoder (default)

```
Input: {'tir': [B,3,T,H,W], 'resp': [B,1,S]}  (or rgb,bp, etc.)

Step 1 — Tokenise + pos embed (all streams):
  tokens_s = adapters[s](x[s]) + positions[s].pos_embed   # [B, N_s, D]

Step 2 — Per-stream asymmetric masking:
  mask_s = masking_generator[s]()               # [N_s] in {0,1}
  visible_s = tokens_s[:, mask_s==0, :]         # [B, N_vis_s, D]

Step 3 — Concatenate visible tokens (ALL streams in encoder):
  z_vis = cat([visible_s for s in encoder_streams], dim=1)  # [B, sum(N_vis_s), D]

Step 4 — Shared ViT encoder:
  for blk in enc_blocks: z_vis = blk(z_vis)
  z_vis = enc_norm(z_vis)                       # [B, sum(N_vis_s), D]

Step 5 — Re-insert mask tokens:
  full_s[:, mask_s==0] = z_vis[slice_s]
  full_s[:, mask_s==1] = positions[s].mask_token

Step 6 — Shared lightweight self-attention decoder (dec_depth=2):
  z_dec = cat([full_s for s in streams], dim=1)
  for blk in dec_blocks: z_dec = blk(z_dec)
  z_dec = dec_norm(z_dec)                       # [B, sum(N_s), D]

Step 7 — Per-stream linear heads:
  pred_s = heads[s](z_dec[slice_s])             # [B, N_s, patch_dim_s]

Step 8 — Norm-pix targets + weighted masked MSE loss
Returns: {'loss': scalar, 'losses_mse': {...}, 'losses_spectral': {...}}
```

### 7.5 Solution A `forward(x)` — asymmetric cross-MAE

**[design]** — `SolutionA_asymCrossMAE.md §3`, `SolutionA_asymCrossMAE_diagram.md §1`

```
Input: {'tir': [B,3,T,H,W], 'resp': [B,1,S]}

Step 1 — Tokenise TIR only (resp excluded from encoder):
  tir_tokens = adapters['tir'](x['tir']) + positions['tir'].pos_embed   # [B, n_visual, D]

Step 2 — Tube mask TIR (ratio=0.90):
  mask_tir in {0,1} [n_visual]
  z_v = tir_tokens[:, mask_tir==0, :]           # [B, n_visible ~80, D]

Step 3 — Shared ViT encoder (TIR visible tokens ONLY — no resp):
  for blk in enc_blocks: z_v = blk(z_v)
  z_v = enc_norm(z_v)                           # [B, n_visible, D]

Step 4 — TIR pixel self-attention decoder (same as Solution C):
  re-insert mask tokens -> dec_blocks -> heads['tir']
  pred_tir_masked: [B, n_masked, tubelet_t*16*16*3]
  L_tir = MSE(pred_tir_masked, gt_tir_masked)

Step 5 — RESP cross-attention decoder (Solution A only):
  q_resp = signal_queries + query_pos_embed     # [B, n_signal=50, D]
  for blk in xdec_blocks:
      q_resp = blk(q_resp, context=z_v)         # DecoderBlock: self-attn + cross-attn
  y_pred = Linear(q_resp).reshape(B, -1)        # [B, L=800]
  L_resp = MSE(y_pred, y_norm)                  # over ALL time slots

Step 6 — Combined SSL loss:
  L_ssl = 1.0 * L_tir + lambda_resp * L_resp
  (lambda_resp in [0.5, 1.0]; NO spectral term in Stage-2)

Returns: {'loss': scalar, 'losses_mse': {'tir': ..., 'resp': ...}}
```

**Verified properties** (`SolutionA_vs_SolutionC.md §6`):
- `[encoder] streams fed to the encoder: ['tir']` — resp tokens never reach the encoder (asserted by `cross_mae_self_test()`).
- The RESP-only loss produces a **non-zero encoder gradient**.
- Solution C smoke reproduces `mse_tir 2.0890 / mse_resp 3.2286` bit-for-bit.
- Solution A extra parameters: ~19 M (cross-attention decoder). Solution C: 102.3 M.

### 7.6 Masking strategies — `_random_mask` vs `_span_mask`

**[design]** — `SpanMask_PhysioSignals.md` (implemented 2026-09-26)

Two strategies for 1-D physiological stream masking, selected by `physio_mask`:

| `physio_mask` | Default | Behaviour |
|--------------|---------|----------|
| `'random'` | **Yes** | Picks `k` tokens independently at uniform random. Historical behaviour reproduced bit-for-bit. |
| `'span'` | No | Masks `n_spans` contiguous blocks of `span` tokens with random starts, shared count across batch. |

**The interpolation shortcut problem** (`SpanMask_PhysioSignals.md §1–§2`): under `'random'` masking at ratio 0.5 the resp stream (period 2.5–6.25 s, token grid 160 ms) is oversampled 16–39×. A linear interpolator using **zero video** achieves `mse_resp 0.129` vs the trained model's `1.052` — the encoder has no incentive to learn cross-modal features.

**Measured difficulty by mask pattern** (`SpanMask_PhysioSignals.md §4.2`, 24 clips x 30 realisations, 25/50 tokens masked):

| Mask pattern | `mse_interp` | vs `mse_const (~1.0)` | Solvable without video? |
|-------------|-------------|----------------------|------------------------|
| Scattered (shipped) | 0.129 | 8x better | **Yes** |
| 4 spans (~1 s each) | 0.841 | still better | Yes |
| 2 spans (~2 s each) | 1.370 | worse | **No** |
| **1 span (4 s)** | **1.459** | **1.5x worse** | **No** |

**`_span_mask` contract** (`SpanMask_PhysioSignals.md §4.3–§4.4`):

```python
def _span_mask(self, B, device, N, mask_ratio, span, n_spans=1):
    k = max(1, min(N - 1, span * n_spans))          # keep >= 1 visible
    start = torch.randint(0, N - k + 1, (B, 1), device=device)
    idx = (start + torch.arange(k, device=device)).clamp(max=N - 1)
    m = torch.zeros(B, N, device=device, dtype=torch.long)
    m.scatter_(1, idx, 1)
    return m
```

Critical invariant: `span` and `n_spans` are drawn **once per step** and shared across the batch. If visible counts differ between samples, the batch-max gather in `forward` will silently leak masked tokens into the encoder, producing better loss with no real gradient signal.

**Per-modality span defaults** (`SpanMask_PhysioSignals.md §4.1`):

| Stream | Band | `mask_span_s` | Tokens @160 ms |
|--------|------|--------------|---------------|
| `bp` | 1–2.5 Hz | 1.0 s | 6 |
| `resp` | 0.16–0.4 Hz | 4.0 s | 25 |
| `eda` | aperiodic | 8.0 s | 50 |

Parameterisation (scheme A — recommended): `mask_ratio` and `mask_span_s` are the primaries; `n_spans = round(mask_ratio * clip_seconds / mask_span_s)` is derived and logged.

**YAML knobs** (additive; `'random'` is the unchanged default):

```yaml
physio_mask: span                    # 'random' (default) or 'span'
mask_span_s: 4.0                     # global scalar, or {resp: 4.0, bp: 1.0}
mask_n_spans: 0                      # 0 = derive from mask_ratio and mask_span_s
```

### 7.7 Geometry variants

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

**[design]** — `SolutionC_EarlyTokenMixing_stage3.md`, `SolutionA_asymCrossMAE.md §5`

Stage 3 is the **sensor-failure** condition: only visual input, **no masking**, no 1-D physio stream. The Stage-2 encoder is fine-tuned end-to-end; `freeze_encoder()` exists but is not called. Both Solution A and Solution C encoders load into the **same** `MultiModalWaveformRegressor` — Solution A's extra tensors (`signal_queries.*`, `xdec_blocks.*`) are silently ignored by `load_stage2_encoder`.

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

**[design]** — `SolutionA_asymCrossMAE.md §5`: the model must assert the expected `grid_t` and output `L` rather than silently interpolating a desynchronised sequence.

### 8.4 Head variants

**[design]** — `SolutionA_vs_SolutionC.md §4`: the matched ablation pair ships `head_style: transposed_conv`, `head_init: none`, `layer_decay: 0.75` for **both** arms to avoid confounding by a one-sided head transfer.

| `head_style` | `head_hidden` | Module |
|-------------|--------------|--------|
| `'linear'` | 0 | `Linear(D, k)` |
| `'linear'` | h > 0 | `Linear(D,h) -> GELU -> Linear(h,k)` |
| `'transposed_conv'` | N/A | `WaveformUpsampleHead` (recommended) |

### 8.5 `WaveformUpsampleHead` tensor flow

**[design]** — `SolutionA_asymCrossMAE.md §5`: spatial aggregation must preserve the temporal grid before upsampling to the target length.

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

### Stage-3 Waveform Losses — `core/waveform_losses.py`

All accept `[B,T]` or `[B,1,T]` (channel squeezed via `_to_1d`).

**`PearsonLoss`**: `mean(1 - r)` where `r = (p_norm . t_norm) / (||p_norm|| * ||t_norm||)`.
**[design]** — `SolutionA_asymCrossMAE.md §6`: must use a variance floor/epsilon and define a finite fallback for near-constant predictions or targets.

**`StdLoss`** (L_time): `mean(|std(pred) - std(target)|)`. Population std with `eps=1e-8` under sqrt to avoid NaN.
**[design]** — `SolutionA_asymCrossMAE.md §6`: the design note advises **not** including StdLoss on the normalised target (its std is ~1.0 by construction, making it largely redundant with Pearson). The shipped `WaveformJointLoss` includes it at `alpha=1.0`; the matched pair configs do not override this.

**`MultiResolutionSTFTLoss`**: for each `n_fft` in `fft_sizes`:

```
P_mag = |STFT(pred, n_fft, hann)|    [B, n_fft//2+1, frames]
T_mag = |STFT(target, n_fft, hann)|
sc = ||P_mag - T_mag|| / (||T_mag|| + eps)             (spectral convergence)
lm = mean(|log(P_mag + 1e-6) - log(T_mag + 1e-6)|)    (log-magnitude L1)
total += sc + lm
```

Degenerate-target guard: skip samples with `||T_mag|| <= 1e-6 * max_batch(||T_mag||)`.

**[design]** — `SpanMask_PhysioSignals.md §5`: the STFT loss is informative **only after span masking removes the interpolation shortcut**. Under scattered masking its optimum is already achieved by the linear interpolator (`spec_interp 0.98`); at the weights tried it produced `spec_resp 3.7e8` and destroyed the model. Under a 4 s span the interpolator degrades to `spec_interp 2.37` — a magnitude-spectral loss can then detect spectral damage that amplitude loss forgives.

**[design]** — `SpanMask_PhysioSignals.md §3`: MSE and MR-STFT are complementary and neither is sufficient alone:

| Error type | MSE | MR-STFT |
|-----------|----|--------|
| 1 s boxcar smoothing | 0.090 (best in table!) | 2.330 (bad) |
| Sign flip (-z) | 4.000 (worst) | 0.000 (free) |
| 1 s time shift | 2.557 | 2.016 |

**`WaveformJointLoss`**: `L = alpha*StdLoss + beta*PearsonLoss + gamma*MultiResolutionSTFTLoss`.  
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

**[design]** — `SolutionA_vs_SolutionC.md §2` (Invariant I2): the only Solution A-only tensors are `signal_queries.*` and `xdec_blocks.*`; the loader whitelists only `enc_blocks.` / `enc_norm.` / `adapters.` / `positions.` and silently ignores the rest. Either pre-trained checkpoint loads into the same fine-tuning model without any key renaming.

Stage-1 VideoMAE -> Stage-2: `canonicalise_vit_state_dict` handles the official MAE/VideoMAE and the HF-transformers VideoMAE checkpoint layouts. The 3-D `patch_embed.proj.weight [D, C, t, ph, pw]` (VideoMAE tubelet) is copied **verbatim**; a tubelet mismatch raises. The legacy 2-D boxcar inflation was removed with the non-VideoMAE sources.

---

## 12. Design Config Map

**[design]** — `SolutionA_vs_SolutionC.md §4`

| Config file | Arm | Key differences |
|------------|-----|-----------------|
| `configs/pretrain/stage2_local_tir_roi_resp.yaml` | Solution C | `resp_in_encoder: true` (default) |
| `configs/pretrain/stage2_local_tir_roi_crossmae.yaml` | Solution A | `resp_in_encoder: false`, `signal_decoder: resp=cross_attn` |
| `configs/finetune/resp_tir_roi_local.yaml` | Historical Stage-3 | Unchanged |
| `configs/finetune/resp_tir_roi_local_matched.yaml` | Solution C encoder | `head_style: transposed_conv`, `head_init: none`, `layer_decay: 0.75` |
| `configs/finetune/resp_tir_roi_crossmae.yaml` | Solution A encoder | Identical recipe to matched |

Pre-train pair differs only in: `resp_in_encoder`, `signal_decoder`, `cross_attn_depth`, `query_init`, `output_dir`.
Stage-3 pair differs only in: `finetune`, `output_dir`.

### Solution A YAML knobs

```yaml
resp_in_encoder: false
signal_decoder: resp=cross_attn   # or 'cross_attn' for all signal streams
cross_attn_depth: 2               # DecoderBlock stack depth
query_init: sincos3d              # 'sincos3d' or 'random' (ablation)
```

### Span masking YAML knobs (additive; `'random'` is the unchanged default)

```yaml
physio_mask: span          # 'random' (default) or 'span'
mask_span_s: 4.0           # global scalar or {resp: 4.0, bp: 1.0}
mask_n_spans: 0            # 0 = derive from mask_ratio and mask_span_s
```
