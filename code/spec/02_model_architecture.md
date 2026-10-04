# 02 — Model Architecture

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/core/` on 2026-10-04.
> All tensor shapes are documented from actual `forward()` implementations.

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

Queries from `x`; keys/values from `context`. Used in `DecoderBlock` for Solution-A cross-stream decoding (RESP stream attending to the full visual encoder output).

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

Grid sizes at standard geometries:

| `input_size` | `num_frames` | `tubelet` | `stride` | `Gt` | `Gh=Gw` | `n_visual` |
|-------------|-------------|----------|----------|------|---------|-----------|
| 64 | 100 | 2,16,16 | 1 | 50 | 4 | 800 |
| 64 | 200 | 2,16,16 | 2 | 50 | 4 | 800 |
| 224 | 100 | 2,16,16 | 1 | 50 | 14 | 9800 |

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

### Construction-time validation

1. Rejects all-visual or all-signal `streams`.
2. Validates `tubelet` divides `input_size` and `num_frames`.
3. `_check_space_time_alignment()`: enforces `sig_kernel/fs == tubelet_t*temporal_stride/fps` AND `n_signal == Gt`. Raises `ValueError` otherwise.
4. `pos_init` in `{'sincos3d', 'random'}`.
5. `target_norm` in `{'token', 'clip'}`.
6. `resp_in_encoder` / `signal_decoder` consistency validated.

### Sub-modules

| Key | Module | Description |
|-----|--------|-------------|
| `adapters` | `nn.ModuleDict` | Per-stream `TubeletEmbed` or `SignalEmbed` |
| `positions` | `nn.ModuleDict` | Per-stream `_PosMask` (pos_embed + mask_token) |
| `enc_blocks` | `nn.ModuleList` | `enc_depth` x `Block` |
| `enc_norm` | `nn.LayerNorm` | Post-encoder norm |
| `dec_blocks` | `nn.ModuleList` | `dec_depth=2` x `Block` |
| `dec_norm` | `nn.LayerNorm` | Post-decoder norm |
| `heads` | `nn.ModuleDict` | Per-stream Linear head: `D -> patch_dim` |
| `cross_decoder.*` | `nn.ModuleList` | `DecoderBlock x cross_attn_depth` (Solution A only) |

### `forward(x: Dict[str, Tensor])` tensor flow

```
Input: {'rgb': [B,3,T,H,W], 'tir': [B,3,T,H,W], 'bp': [B,1,S]}

Step 1 — Tokenise + pos embed:
  tokens_s = adapters[s](x[s]) + positions[s].pos_embed   # [B, N_s, D]

Step 2 — Asymmetric masking:
  mask_s from MultiModalMaskingGenerator()                  # [N_s] in {0,1}
  visible_s = tokens_s[:, mask_s==0, :]                    # [B, N_vis_s, D]

Step 3 — Concatenate visible encoder-stream tokens:
  z_vis = cat([visible_s for s in encoder_streams], dim=1) # [B, sum(N_vis_s), D]

Step 4 — Shared ViT encoder:
  for blk in enc_blocks: z_vis = blk(z_vis)
  z_vis = enc_norm(z_vis)                                  # [B, sum(N_vis_s), D]

Step 5 — Re-insert mask tokens:
  full_s[:, mask_s==0] = z_vis[slice_s]
  full_s[:, mask_s==1] = positions[s].mask_token

Step 6 — Shared lightweight decoder (dec_depth=2):
  z_dec = cat([full_s for s in streams], dim=1)
  for blk in dec_blocks: z_dec = blk(z_dec)
  z_dec = dec_norm(z_dec)                                  # [B, sum(N_s), D]

Step 7 — Per-stream linear heads:
  pred_s = heads[s](z_dec[slice_s])                        # [B, N_s, patch_dim_s]

Step 8 — Norm-pix targets (per-token or per-clip z-score based on target_norm)

Step 9 — Weighted masked MSE + optional MR-STFT loss

Returns: {'loss': scalar, 'losses_mse': {...}, 'losses_spectral': {...}}
```

### Geometry variants

| Variant | `embed_dim` | `enc_depth` | `enc_num_heads` |
|---------|------------|------------|----------------|
| small | 384 | 12 | 6 |
| base | 768 | 12 | 12 |
| large | 1024 | 24 | 16 |
| huge | 1280 | 32 | 16 |

`dec_depth=2` (hardcoded default). Decoder uses the same `embed_dim` as encoder.

---

## 8. Stage-3: `MultiModalWaveformRegressor` — `core/waveform_model.py`

### Sub-modules (identical key names to Stage-2 encoder)

| Key | Description |
|-----|-------------|
| `adapters` (`nn.ModuleDict`) | Per-stream `TubeletEmbed` |
| `positions` (`nn.ModuleDict`) | Per-stream `_PosMask` (pos_embed only; mask_token unused) |
| `enc_blocks` (`nn.ModuleList`) | `enc_depth` x `Block` |
| `enc_norm` (`nn.LayerNorm`) | Post-encoder norm |
| `waveform_head` | `Linear(D,k)` / `Sequential(D->h,GELU,h->k)` / `WaveformUpsampleHead` |

> Key name identity with Stage-2 is critical: `load_stage2_encoder` transfers weights by exact key match.

### `forward(x)` tensor flow

```
x: {'rgb': [B,3,T,H,W]}  (or plain Tensor for single-stream)

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

### Head variants

| `head_style` | `head_hidden` | Module |
|-------------|--------------|--------|
| `'linear'` | 0 | `Linear(D, k)` |
| `'linear'` | h > 0 | `Linear(D,h) -> GELU -> Linear(h,k)` |
| `'transposed_conv'` | N/A | `WaveformUpsampleHead` |

### `WaveformUpsampleHead` tensor flow

```
h: [B, grid_t, D]
  -> transpose(1,2)                      -> [B, D, grid_t]
  -> Conv1d(D, hidden, k=3, pad=1)       -> [B, hidden, grid_t]
  -> GELU
  -> ConvTranspose1d(hidden, 1, k=s, s=s) -> [B, 1, grid_t*s]
  -> transpose(1,2)                      -> [B, grid_t, s]
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

**`StdLoss`** (L_time): `mean(|std(pred) - std(target)|)`. Population std with `eps=1e-8` under sqrt to avoid NaN on constant input.

**`MultiResolutionSTFTLoss`**: for each `n_fft` in `fft_sizes`:

```
P_mag = |STFT(pred, n_fft, hann)|    [B, n_fft//2+1, frames]
T_mag = |STFT(target, n_fft, hann)|
sc = ||P_mag - T_mag|| / (||T_mag|| + eps)             (spectral convergence)
lm = mean(|log(P_mag + 1e-6) - log(T_mag + 1e-6)|)    (log-magnitude L1)
total += sc + lm
```

Degenerate-target guard: skip samples with `||T_mag|| <= 1e-6 * max_batch(||T_mag||)`.

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

Stage-1 ViT -> Stage-2: `canonicalise_vit_state_dict` handles MAE, VideoMAE, and HF-transformers checkpoint layouts. 2-D `patch_embed.proj.weight [D, C, ph, pw]` is inflated along the tubelet time axis to `[D, C, t, ph, pw]` (repeating, divided by t).
