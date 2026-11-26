# ADR-015: Stage-3 Waveform Regression Architecture, Encoder Transfer Contract, and Joint Loss

- **Status**: Accepted
- **Date**: 2026-09-25 (revised 2026-11-01; codified as ADR-015 on 2026-11-26)
- **Author**: KISMED MA research team
- **Related Spec/Plan**: Pre-existing baseline (`code/notes/spec/02_model_architecture.md` §8 & §10–§11, `code/notes/spec/03_engine_and_loss.md` §4 & §6, `code/core/waveform_model.py`, `code/core/waveform_losses.py`, `code/runners/run_waveform.py`); Stage-2 encoder architecture governed by `ADR-014`; visual clip norm governed by `ADR-013`; split/selection governed by `ADR-001`; subject-level evaluation governed by `ADR-009`
- **Supersedes**: `SolutionC_EarlyTokenMixing_stage3.md` (promoted from PlantUML diagram + local-RGB notes into full lineage-general ADR)

> **Scope.** This ADR governs one research aspect: **the Stage-3 downstream supervised waveform
> regression model (`MultiModalWaveformRegressor`), its Stage-2 encoder transfer & geometry-contract
> verification (`load_stage2_encoder`, `ROI_CONTRACT_KEYS`), and its composite fine-tuning objective
> (`WaveformJointLoss`)**. Stage-2 pre-training (`ADR-014`, `ADR-007`, `ADR-008`), offline session
> assembly & evaluation metrics (`ADR-009`), and split/selection protocol (`ADR-001`) are governed
> separately.

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope
Stage 3 represents the **deployed contact-sensor failure scenario**: no physiological waveform (`resp` or `bp`) enters the network at all. Visual clip(s) alone (`tir` for respiration, `rgb` for BP/respiration) are fed unmasked through the Stage-2 pretrained ViT encoder, and a lightweight temporal head predicts the continuous 1-D waveform window `[B, output_len]`.

### 1.2 Design Pressures
1. **Strict sensor-absence guarantee.** Contact-sensor failure must be structural (1-D streams omitted entirely), not merely simulated by masking, so no physiological information can leak into the encoder during fine-tuning or inference.
2. **Silent geometry or ROI-contract drift.** Transferring a Stage-2 checkpoint trained on one `(input_size, num_frames, roi_padding, crop_rule, max_sway_px)` onto a mismatched Stage-3 config produces silently degraded or crashing runs unless guarded explicitly at load time.
3. **Time-aligned waveform decoding.** A single global CLS/mean pool collapses temporal phase across an 8 s clip. Each temporal tubelet index $g_t \in \{0, \dots, G_t - 1\}$ must decode directly into its aligned contiguous waveform segment of length `samples_per_token = output_len / G_t`.
4. **Composite waveform supervision & degenerate-target explosion.** Pure time-domain error allows phase drift or flattened peaks, whereas scale-invariant spectral convergence ($\|P - T\| / (\|T\| + \epsilon)$) explodes to $\sim 10^{10}$ on flat/railed targets (`||T|| ≈ 0`) without a degenerate-target guard.

### 1.3 Requirements
- **R1 — Visual-only, unmasked, end-to-end fine-tuning.** Reject any non-visual stream at model construction; feed all $N_{\text{visual}} = G_t G_h G_w$ tokens with zero masking; optimize encoder and head end-to-end.
- **R2 — Verbatim Stage-2 encoder key parity & loud compatibility guards.** Transfer `adapters.<s>.*`, `positions.<s>.pos_embed`, `enc_blocks.*`, and `enc_norm.*` by exact key match, ignoring Stage-2-only decoder/`align_proj.*` keys and aborting on any `pos_embed` shape mismatch or `ROI_CONTRACT_KEYS` disagreement.
- **R3 — Temporally aligned head variants.** Spatial mean-pooling per tubelet time step followed by a modular head (`linear`, 2-layer MLP, or `transposed_conv` `WaveformUpsampleHead`) where segment $k$ covers tubelet window $k$.
- **R4 — FP32 composite waveform supervision (`WaveformJointLoss`).** Combine point-wise Huber (`SmoothL1Loss`), negative Pearson correlation, and degenerate-target-guarded Multi-Resolution STFT, evaluated in FP32 even under mixed precision (`ADR-016`).

---

## 2. Decision & Parameter Exposure

**Deploy `MultiModalWaveformRegressor` as a visual-only, unmasked end-to-end regressor sharing exact encoder key names with Stage 2 (`ADR-014`), guarded by `load_stage2_encoder` and `ROI_CONTRACT_KEYS`, decoding per-timestep spatially pooled latents through a time-aligned head supervised by FP32 `WaveformJointLoss = α·L_Huber + β·L_Pearson + γ·L_MR-STFT`.**

### 2.1 Forward Tensor Flow & Head Variants
```text
Input x: {s: [B, 3, T, H, W]}   (visual streams only; 1-D streams rejected at __init__)
  1. Per stream s:
       xt  = _fit_time(x[s], num_frames, s)            # trim trailing if T > n; raise if T < n
       xt  = zscore_clip(xt)                           # ADR-013 (exact Stage-2 parity)
       tok = adapters[s](xt) + positions[s].pos_embed  # [B, G_t*G_h*G_w, D]; hard shape assert
  2. Shared ViT encoder (no masking):
       z   = enc_norm(enc_blocks(cat(toks, dim=1)))    # [B, S * n_visual, D]
  3. Per-timestep spatial mean pooling (averaged across visual streams if S > 1):
       h   = mean_{s, h, w}(z)                         # [B, G_t, D]
  4. Temporal regression head (output_len % G_t == 0; k = output_len // G_t):
       pred = waveform_head(h).reshape(B, output_len)  # [B, output_len]
```

**Supported head variants (`head_style`, `head_hidden`):**
- `head_style='linear'`, `head_hidden=0` *(default)*: `nn.Linear(D, k)`.
- `head_style='linear'`, `head_hidden=h > 0`: `Linear(D, h) → GELU → Linear(h, k)`.
- `head_style='transposed_conv'`: `WaveformUpsampleHead` (`Conv1d(D, hidden, k=3, pad=1) → GELU → ConvTranspose1d(hidden, 1, kernel=k, stride=k)`), preserving exact segment-to-tubelet alignment.

### 2.2 Stage-2 → Stage-3 Transfer & Compatibility Guards
1. **Weight transfer (`load_stage2_encoder`):** Copies `enc_blocks.*`, `enc_norm.*`, `adapters.<s>.*`, and `positions.<s>.pos_embed`. Stage-2-only weights (`dec_blocks.*`, `dec_norm.*`, `align_proj.*`, unused stream adapters) are excluded by construction.
2. **Head initialization (`head_init`):** `'transfer'` copies `heads.<target>.*` into `waveform_head` only when `head_style == 'linear'`, `head_hidden == 0`, and `samples_per_token == sig_kernel` (accepting legacy `heads.bvp` for `bp`); `'none'` initializes the head randomly (used in controlled comparisons).
3. **Hard geometry & ROI contract guards:**
   - Any `pos_embed` shape mismatch raises `ValueError`.
   - `check_stage2_roi_contract` (`runners/run_waveform.py`) compares `ROI_CONTRACT_KEYS = ('roi_landmarks', 'roi_padding', 'input_size', 'crop_rule', 'max_sway_px')` against recorded Stage-2 checkpoint args and aborts (`SystemExit`) on any mismatch, warning loudly if a legacy checkpoint predates `crop_rule`/`max_sway_px`.
   - Partition digest (`ADR-001`) must match the fold's `partition.json` so Stage 3 never fine-tunes on held-out test subjects.

### 2.3 Stage-3 Joint Supervision Objective (`WaveformJointLoss`, FP32)
$$\mathcal{L}_{\text{joint}} = \alpha\,\mathcal{L}_{\text{time}} + \beta\,\mathcal{L}_{\text{Pearson}} + \gamma\,\mathcal{L}_{\text{MR-STFT}}$$
*(Namespace note per `GLOSSARY.md`: Stage-3 $(\alpha, \beta, \gamma)$ denote Huber / Pearson / MR-STFT and are distinct from Stage-2 `ADR-008` composite weights.)*

1. **$\mathcal{L}_{\text{time}}$ (point-wise Huber):** `nn.SmoothL1Loss(beta=huber_beta, reduction='mean')` on raw point-wise errors (`huber_beta` in target-std units, default `1.0`). This is the sole polarity- and scale-sensitive term.
2. **$\mathcal{L}_{\text{Pearson}}$ (phase locking):** `mean(1 - r)` over centered `[B, L]` waveforms ($\epsilon = 10^{-8}$). Scale- and shift-invariant.
3. **$\mathcal{L}_{\text{MR-STFT}}$ (multi-resolution spectral fidelity):** For each window $n_{\text{fft}} \in \texttt{fft\_sizes}$ (Hann window, `hop_ratio = 0.25`; windows exceeding `output_len` dropped before training), computes spectral convergence $\|P_{\text{mag}} - T_{\text{mag}}\| / (\|T_{\text{mag}}\| + \epsilon)$ plus log-magnitude L1 $\operatorname{mean}(|\log(P_{\text{mag}} + 10^{-6}) - \log(T_{\text{mag}} + 10^{-6})|)$.
   - **Degenerate-target guard:** Samples with $\|T_{\text{mag}}\| \le 10^{-6} \cdot \max_{\text{batch}}(\|T_{\text{mag}}\|)$ are excluded from the STFT term (emitting `0.0` if none qualify), preventing `~1e10` gradient explosions on flat/railed clips while remaining bit-identical on healthy batches.
4. **FP32 enforcement (`ADR-016`):** `run_waveform.py` passes `loss_in_fp32=True`, casting `outputs.float()` and evaluating `WaveformJointLoss` outside `autocast`.

### 2.4 Logical Parameter Surface

| Parameter | Type | Default | Constraints | Role |
|---|---|---|---|---|
| `streams` | CSV | `tir` (TIR-ROI) / `rgb` (RGB) | visual streams only (`rgb`, `tir`, `rgb2d`) | Visual input streams |
| `target` | str | `resp` / `bp` | `'resp' \| 'bp'` | 1-D physiological regression target |
| `head_style` / `head_hidden` | str / int | `'linear'` / `0` | `'linear' \| 'transposed_conv'`; `>= 0` | Temporal head architecture |
| `head_init` | str | `'transfer'` (code) / `'none'` (shipped configs) | `'transfer' \| 'none'` | Whether to copy Stage-2 `heads.<target>` |
| `alpha`, `beta`, `gamma` | float | `1.0`, `1.0`, `1.0` | `>= 0` | Weights for Huber, Pearson, MR-STFT |
| `huber_beta` | float | `1.0` | `> 0` | Huber transition in target-std units |
| `fft_sizes` | int CSV | `'128,256,512'` (TIR-ROI 8 s); `'64,128,256'` (local 4 s) | `n_fft <= output_len` | STFT window lengths |
| `save_preds` / `eval_only` | bool | `False` | single-process required | Dumps `preds.npy`, `targets.npy`, `entries.json` for `ADR-009` offline evaluation |

### 2.5 Rejected Alternatives
- **Global CLS/mean pooling across time (`ProjectViT.forward_waveform`)**: Rejected because it discards per-tubelet temporal alignment and shares no keys with the 3-D Stage-2 encoder.
- **Frozen encoder linear probing for waveform regression**: Rejected as the primary Stage-3 recipe (end-to-end fine-tuning is used; `freeze_encoder()` exists only as an unused diagnostic helper).
- **Unguarded MR-STFT**: Rejected after a single flat railed clip caused `4.3e10` loss and `2.5e9` gradient norm.

---

## 3. Falsifiable Criteria

- **Criterion 1 (Transfer & contract guards)**: Loading a Stage-2 checkpoint transfers all encoder weights (`enc_blocks`, `enc_norm`, `adapters`, `positions`), ignores `align_proj.*` / decoder weights, raises `ValueError` on `pos_embed` geometry mismatch, and aborts via `check_stage2_roi_contract` on any `ROI_CONTRACT_KEYS` mismatch.
- **Criterion 2 (Loss & numerical safety)**: `WaveformJointLoss` handles flat/zero-energy targets without `NaN`/`Inf` explosion, executes in FP32 even when `--amp_dtype bf16|fp16`, and maintains affine invariance on visual inputs (`ADR-013`).
- **Criterion 3 (Downstream scientific gate)**: Evaluated via `ADR-009` over all 5 subject-disjoint folds (`ADR-001`) using `--eval_role test --save_preds` and `run_evaluate_kfold.py` against the `ADR-003` respiration artefact.

- **Acceptance Artifacts**: `metrics.jsonl`, `metrics_final.json`, `selection.json`, `preds.npy`, `targets.npy`, `entries.json`, and downstream `ADR-009` reports (`kfold_*.json`).

---

## 4. Consequences & Trade-offs

- **Positive**: Strict separation between pre-training and sensor-absent deployment; exact temporal alignment between tubelets and predicted samples; leak-free Stage-2 → Stage-3 weight & contract transfer; numerically stable composite supervision.
- **Negative / Technical Debt**:
  - `--save_preds` and `--eval_only` must run single-process (`WORLD_SIZE=1`) so one rank writes the complete prediction array.
  - `MultiModalWaveformRegressor` requires `output_len % G_t == 0`; when Stage-2 uses `temporal_stride=4` ($G_t = 25$, `ADR-007`) and Stage-3 targets 100 Hz (`output_len=800`, $k=32$) or 1000 Hz (`output_len=8000`, $k=320$), `fft_sizes` and `sig_kernel` must match the chosen grid.

---

## 5. Reproducibility Record

- **Run / Job IDs**: Shipped across all 5-fold Stage-3 sweeps (`resp_tir_roi_hpc.yaml`, `bp_rgb_roi_local.yaml`).
- **Evidence Artifacts**: `code/notes/eval/archive/tir_resp/resp_5fold_cv_96respgrid.md`, `code/notes/eval/archive/tir_resp/resp_5fold_cv.md`.
- **Tests / Smoke Checks**: `code/tests/test_video_input_norm.py`, `code/tests/test_amp.py`, `code/tests/test_config_loading.py`, `code/tests/test_adr008_alignment.py` (transfer isolation gate).
- **Rollback / Rejection Condition**: Core Stage-3 architecture of record; individual loss terms can be ablated cleanly by setting `alpha`, `beta`, or `gamma` to `0.0`.
