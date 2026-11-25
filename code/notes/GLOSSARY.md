# Project Terminology & Symbol Glossary (KISMED MA)

This document is the authoritative dictionary of canonical names, mathematical
symbols, abbreviations, and deprecated aliases for this repository and the
associated thesis (`report/TUDaThesis/`). **Ground truth** is the executable code
under `code/` plus the active configs under `code/configs/`; where this file and
the code disagree, the code wins and this file is the bug.

---

## 0. Acronym Disambiguation (read first)

| Acronym | Meaning in this project | NOT to be confused with |
| :--- | :--- | :--- |
| **BP** | Blood-pressure waveform modality (`bp`, `[B, 1, L]`) | `bpm` (breaths/min, a *unit*) |
| **bpm** | Breaths per minute — the derived respiration-rate readout (`rate_metrics['mae_bpm']`) | blood pressure |
| **TIR** | Thermal-infrared video stream (false-colour, 3-channel) | `Thermal/` (raw on-disk directory literal) |
| **RESP** | Respiration waveform modality / Stage-3 target | — |
| **AMMM** | Asymmetric Multi-Modal Masking — the REMOVED per-step Dirichlet policy (ADR-002 superseded it) | the live **constant-budget** policy (`core/budget_masking.py`) |

---

## 1. Terminology Lifecycle & Deprecation Table

| Canonical Term | Legacy / Forbidden Alias | Since | Scope | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **`rgb`** | — (**not** deprecated; prose "RGB" is a *modality name*, not a stream identifier) | — | `code/core/`, `code/data/`, `code/configs/` | **(S1, 2026-11-17) Legacy-but-supported — NOT deprecated.** The canonical 224 px / `rgb_roi` 64 px visible-light stream, and the ONLY visible input for `PairedSessionDataset` (`data_set: bp4d+`), the AU probe and every Stage-3 config. **Not preferred for fusion:** it cannot share a patch grid with `tir`, so any visible+thermal run must use `rgb2d` (see §2). |
| **`rgb2d`** | `RGB2d`, "RGB2D", "RGB texture" (prose only) | 2026-11-10 | `code/core/multimae.py`, `code/data/rgb2d_cache.py`, `code/data/trimodal_dataset.py`, `code/configs/` | BP4D+ 2D-texture face ROI served from the offline cache. Lower-case identifier; a distinct name from `rgb` so a checkpoint can never be paired with the wrong front-end. **(S1) PREFERRED visible stream whenever `tir` is in the run** — the only visible stream defined on the TIR patch grid. |
| **`tir`** | `thermal`, `IR`, `FLIR` (in prose/identifiers) | 2026-09 | Everywhere **except** the carve-out below | Standardises on thermal-infrared. |
| **`Thermal/`, `find_thermal_video`, `run_inspect_thermal`** | — (**frozen**, do not rename) | — | `data/raw/BP4D/Thermal/`, `code/data/tir_resp_dataset.py:225`, `code/runners/run_inspect_thermal.py` | Raw-tree path literal and published API symbols; renaming breaks data discovery. |
| **`roi_landmarks`** (with `ROI_LANDMARK_PRESETS`) | `resp_roi`, `crop_face`, `roi_crop` (never existed in code) | 2026-10 | `code/data/tir_resp_dataset.py:185` | Canonical ROI selector: presets `nose_mouth` \| `nostrils` \| `nostril_mouth` \| `nose_tip`, or a CSV of 1-indexed landmark labels. |
| **`crop_rule`** | "aspect-preserving crop" | 2026-10 | `code/data/tir_resp_dataset.py:159-173` | `square` (dataset default; one scale factor, no deformation) \| `box` (legacy per-axis, A/B only). |

---

## 2. Modalities & Streams

Closed stream set: visual `('rgb', 'tir', 'rgb2d')`, physiological
`('bp', 'resp')`; per-stream channels in `STREAM_CHANNELS`
(`code/core/multimae.py:106`).

| Term | Shape | Channels | Notes |
| :--- | :--- | :--- | :--- |
| **`rgb`** | `[B, 3, T, H, W]` | 3 | Canonical 224 px / `rgb_roi` 64 px visible-light arm (`stage2_local_pretrained.yaml`: `streams: rgb,bp,resp`). Legacy-but-supported; **cannot** coexist with `tir` in one model (ONE global `input_size`) — use `rgb2d` for fusion. |
| **`tir`** | `[B, 3, T, H, W]` | 3 default / 1 legacy (`--tir_channels 1`) | **False-colour (rainbow) rendering** with a burned-in degC legend — not luma, not a "3-channel duplicate". |
| **`rgb2d`** | `[B, 3, T, H, W]` | 3 | 2D-texture face ROI from the offline cache (`store_resolution='native'`). Must share `input_size` / `tubelet` / `num_frames` with `tir`, and is cropped from the **same frames**. **Preferred visible stream in any `tir`-containing run** (`streams: tir,rgb2d,resp`). |
| **`bp` / `resp`** | `[B, 1, L]` | 1 | 1-D physiological waveform, length `L = round(kept_seconds * fs)`. |

- **TIR**: thermal-infrared; captures surface temperature dynamics and
  micro-vascular emissions.
- **BP**: blood-pressure waveform (1-D). **RESP**: respiration (1-D, chest /
  airflow derived).
- **Stage-2 contract**: >=1 visual stream **and** >=1 physiological stream;
  all-visual / all-signal lists are rejected at model build.
- **Visible-stream preference (S1, 2026-11-17)**: `rgb` is **not deprecated** —
  it remains the only visible stream for `PairedSessionDataset` (`data_set:
  bp4d+`), the AU probe (`data_set: bp4d+`) and every Stage-3 config. Prefer
  `rgb2d` whenever `tir` is in the same run: `MultiModalMAE` holds ONE global
  `input_size` (`code/core/multimae.py:202`, enforced by
  `_check_space_time_alignment`), so a 224/64 px `rgb` can never be paired with
  the 96 px TIR ROI, whereas `rgb2d` shares `input_size` / `tubelet` /
  `num_frames` with `tir` by construction. `streams: tir,rgb2d,resp` is the only
  visible+thermal fusion configuration.
- **Stage-3**: `streams: tir` (visual input only) plus a single 1-D **target**
  (`target: resp`) — `resp` is not a stream there.
- **`tir_channels`**: 3 is the shipped default; `1` only reproduces
  pre-2026-09 checkpoints (it changes adapter geometry, so checkpoints are not
  interchangeable between settings).

---

## 3. Geometry, Tokenisation & Masking

| Name | Meaning | Shipped value |
| :--- | :--- | :--- |
| `tubelet` | Conv3d tubelet `(t, h, w)` | `2,16,16` |
| `sig_kernel` | samples per 1-D token | `16` (every ROI config); `8` (canonical `bp4d+` configs) |
| `temporal_stride` | decimation INSIDE the window (not the hop) | `2` (ROI lineage), `1` (canonical lineage) |
| `clip_duration` | window length | `8.0` s (ROI lineage), `4.0` s (canonical) |
| `clip_stride` | window hop | per config (`1.0` local TIR-ROI, `2.0` HPC, `3.0`/`4.0` elsewhere); NOT part of the ROI Stage-2/3 contract |
| `input_size` | ROI side after resize (one patch grid for all visuals) | `64` local, `96` HPC ROI, `224` full-frame HPC |
| `seq_len` | 1-D length; `0` => `round(kept_seconds * fs)` | `0` |

**Hard space-time invariant** (`MultiModalMAE._check_space_time_alignment`,
construction-time):
`R = n_signal / Gt` must be an INTEGER and `R * sig_kernel/fs ==
tubelet_t * temporal_stride / fps` (one signal token tiles a visual token
an integer number of times; `R = 1` is the classic aligned grid, `R = 10` is
the ADR-002 arm).

**Masking terms**
- **Constant-budget masking** (ADR-002, `core/budget_masking.py`): the ONE
  masking policy. Every step samples a MODE and one kept-TOKEN count per stream
  such that all counts sum to `n_budget` (default 350), so the encoder input has
  a constant length. **Mode 1** (pure virtual sensing, `p_virtual_sensing`) keeps
  no physiological token; **Mode 2** (waveform inpainting) keeps a drawn
  allocation per physiological stream and gives the visual streams the residual
  in whole tubes.
- **Masking unit**: a visual unit is one **spatial tube** (one patch + all of
  its temporal positions, so a visual count is a multiple of `Gt`); a
  physiological unit is one **temporal token**.
- **Caps / floors** (`video_cap_ratio` 0.25, `physio_cap_ratio` 0.50,
  `video_min_tubes` 2): under the **two-visual** policy (ADR-002..006) every
  visual stream keeps 2..9 tubes (50..225 tokens), every physiological stream at
  most 125 of 250 tokens. Under the **one-visual** policy (ADR-007) the same
  `video_cap_ratio` is a *ceiling* on the kept fraction (8/36 = 22.2 %) and the
  cap in tubes IS the Mode-1 allocation `n_budget / T`; see below.
- **Placement** (`core/mask_placement.py`): which units survive, drawn per
  SAMPLE — `scattered` (uniform without replacement, default), `span`
  (contiguous blocks of `physio_span_tokens`), or — under ADR-007 —
  `anatomy_aware` (`place_video_stratified`, stratified core + periphery).
  Counts are batch-uniform, placements are not.
- **ADR-007 one-visual policy** (`streams: tir,resp`, `n_budget = 200`,
  `spatial_sampling: anatomy_aware`): the Phase-1 single-camera thermal →
  respiration arm. Terms:
  - **$\mathcal{A}_{core}$ / `A_core`** — the anatomical core *subgrid* of the
    `6 × 6` patch grid that carries the convective airflow plume
    (`core_row_bounds × core_col_bounds`, default rows 2-3 × cols 2-3 → tubes
    `{14, 15, 20, 21}`; index = `row * G_w + col`).
  - **$\mathcal{A}_{periph}$ / `A_periph`** — the remaining 32 tubes (nasal
    bridge, cheeks, upper lip, chin).
  - **$S_{core}$ / `core_tubes_min..max`** — kept core tubes per step (`{1, 2}`
    for the arm). `S_core ≥ 1` is what makes the core hit **certain** instead of
    a probability. **$S_{periph}$** — the remaining kept tubes, floored by
    `periph_min_tubes` ($m_{periph}$).
  - **$S_{tir}^{min}$ / `video_min_tubes`** — the visual floor at physiological
    saturation, `(n_budget − C_p) // T` (4 for the arm). It is an *assertion* of
    the derivation, not a free knob.
  - **`physio_draw_min`** — lower bound of the Mode-2 physiological draw. `0`
    keeps the ADR-002..006 draw (with its `K_tilde = 0` fall-through); `1` is
    ADR-007 §2.5 (`Uniform{1..C_p}`), so **no Mode-2 step can leak into Mode 1**.
  - **`REGIME_SINGLE_VISUAL`** (`'single_visual'`) — the ADR-007 step class:
    `regime == 'inpainting'` iff some signal stream keeps > 0 tokens, else
    `single_visual` on a one-visual run (`single`/`bi_visual` stay the
    two-visual labels).
- **`n_budget` / `spatial_patches` / `physio_tokens`**: the constant visible
  budget and the two tokenizer-derived unit counts (the latter two are asserted
  against the tokenizer at build, never tuned).
- **MultiMAE**: multi-modal masked autoencoder — visible tokens of **all**
  streams are concatenated into one shared ViT encoder; per-stream linear heads
  reconstruct masked tubelets/windows; masked MSE is computed per modality on
  masked positions only, then weighted-summed.
- **VideoMAE**: the Stage-1 encoder pre-training paradigm (`videomae:k400`,
  `videomae:ssv2`) whose 3-D Conv3d tubelet patch embed is inherited verbatim.
- **Zero-shot evaluation**: `run_eval_recon.py` — RESP reconstruction from a
  **frozen** Stage-2 checkpoint, encoder weights never updated. No
  unseen-modality transfer is implemented.

---

## 4. Label Normalisation

- **`physio_norm`** in `{clip, zscore}` (synonyms). Validated by
  `check_physio_norm`; `none` / `ac` / `session` are **refused**, not mapped.
- **`TARGET_NORM = 'clip'`** (`code/data/tir_resp_dataset.py:153`): one
  per-clip z-score (population std, `ddof=0`, eps `ZSCORE_EPS = 1e-6`), applied
  by the **DATASET** in every ROI lineage, in both stages.
- **`target_norm`** (model side) in `{token, clip, none}`;
  `pin_roi_target_norm` forces `'none'` (identity) for `tir_roi` / `rgb_roi`,
  so the model never re-normalises a 1-D stream.

---

## 5. Loss Terms & Weights

**Stage 2** — per-modality masked MSE on masked positions (`mask == 1` marks
the predicted token): `MaskedMSELoss`, plus `MaskedL1Loss` /
`MaskedCrossEntropyLoss` (`code/core/criterion.py`).
- `signal_weight` (`--signal_weight`, default `0.5`): the weight for **every**
  physiological stream; visual streams stay `1.0`. In ADR-008 terms this knob
  **is** $\lambda_\text{resp}$ (the arm sets it to `5.0`).
- `loss_weights` (`--loss_weights`, CSV, one value per stream in `--streams`
  order): per-stream override of the whole policy.

**Stage 2 — ADR-008 composite respiration objective**
(`code/core/resp_composite.py`, `code/core/align_infonce.py`):

$$L_\text{resp} = \alpha L_\text{MSE} + \beta L_\text{Pearson} + \gamma L_\text{spectral},\qquad
L_\text{pretrain} = L_\text{tir} + \lambda_\text{resp} L_\text{resp}
+ \lambda_\text{align}\mathbb{I}(K_\text{resp} > 0 \land K_\text{tir} > 0)L_\text{align}$$

- **Namespace warning.** These $\alpha, \beta, \gamma$ are **Stage-2** and mean
  MSE / masked Pearson / band-limited spectral. The **Stage-3**
  $\alpha, \beta, \gamma$ of §5 mean time / Pearson / MR-STFT. Never mix them
  in one sentence or one table row.
- `L_MSE`: the unchanged `MaskedMSELoss` on the full `[B, N_s, k]` tensors
  (bit-identical to the pure-MSE arm, so `mse_resp` stays comparable).
- `L_Pearson`: `1 - r` over the **masked tokens only**, `eps` inside each
  square root, degenerate-variance targets excluded (`var_floor`).
- `L_spectral`: band-limited ($[f_{\min}, f_{\max}]$ Hz) PSD L1 between
  unit-sum power distributions of the full clip, with **detached** ground
  truth spliced into the visible tokens.
- Flags: `--resp_loss_mse_weight` ($\alpha$, `1.0`), `--resp_loss_pearson_weight`
  ($\beta$, `0.0`), `--resp_loss_spectral_weight` ($\gamma$, `0.0`),
  `--spectral_f_min` / `--spectral_f_max` (`0.10` / `0.60` Hz),
  `--spectral_n_fft` (`0` = auto, $4\cdot 2^{\lceil\log_2 L\rceil}$).
- `L_align`: symmetric InfoNCE (`--align_weight` $\lambda_\text{align}$, `0.0`;
  `--align_proj_dim` $d_\text{proj}$, `--align_temperature` $\tau$,
  `--align_core_weight` $w_\text{core}$, `--align_gather_ddp`) on
  anatomy-weighted pooled encoder latents, **only** on steps with
  $K_\text{resp} > 0$. The two `align_proj.*` heads are Stage-2 only and are
  excluded from the Stage-3 transfer by construction.
- `--grad_balance_freq` (`0`): per-`freq`-step diagnostic
  $\lVert\nabla_z(\lambda_\text{resp}L_\text{resp})\rVert_2 /
  \lVert\nabla_z L_\text{tir}\rVert_2$; a **logging cadence**, not an objective
  parameter.
- There is **no spectral term on the VISUAL streams in Stage 2**: the
  2026-11-09 removal still holds for `tir` / `rgb2d`; ADR-008 reintroduced a
  band-limited spectral term for the **physiological** stream only.

---

## 6. Evaluation Tiers (`code/evaluation/`, offline, non-differentiable)

- **Tier 1 — temporal**: `mae`, `rmse`, `pearson`; plus
  `band_limited_pearson` (drift-robust headline metric), `amplitude_metrics`
  (`amp_ratio`, `amp_rel_err`), `estimate_rate_bpm` / `rate_metrics`
  (`mae_bpm` = breaths per minute).
- **Tier 2 — spectral**: Welch PSD consistency (`psd_mae`),
  `dominant_freq_error_hz`, band-limited out-of-band energy (`hf_power_rel`,
  `target_hf_power_rel`, `hf_lo_hz`), diagnostics `psd_band_bins` /
  `psd_skipped`; plus MR-STFT tracking. Degenerate (energy-free) targets are
  skipped, not scored.
- **Tier 3 — clinical**: `extract_hrv_metrics` on the **predicted BP waveform**
  via NeuroKit2 => `rmssd_ms`, `pnn50`, `median_nn_ms`, `shannon_entropy`, plus
  `n_rr` and `peak_success`. Requires >=30 s of signal; 8 s clips return NaN
  fields by design. **No systolic/diastolic BP estimation is implemented.**

---

## 7. Mathematical Notation Standards (Thesis Alignment)

| Concept | Symbol | Tensor Shape in Code | Description |
| :--- | :--- | :--- | :--- |
| Batch size | $B$ | `x.shape[0]` | Independent samples per step. |
| Channel count | $C$ | `x.shape[1]` | 3 for `rgb` / `tir` / `rgb2d`, 1 for `bp` / `resp`. |
| Video frames | $T$ | `x.shape[2]` | Post-`temporal_stride` frames in a clip. |
| Signal length | $L$ | `x.shape[-1]` (1-D streams) | `L = round(kept_seconds * fs)`; distinct from $T$. |
| Spatial extent | $H \times W$ | `x.shape[3:]` | Video height / width. |
| Token count | $N$ | `x.shape[1]` after adapters | $N = G_t G_h G_w$ (video), $N = G_t$ (1-D). |
| Tubelet grid | $G_t, G_h, G_w$ | `T/tubelet_t`, `H/tubelet_h`, `W/tubelet_w` | Flattening order `g_t G_h G_w + g_h G_w + g_w`. |
| Embedding dimension | $D$ | `x.shape[-1]` | Shared transformer token width. |
| Space-time alignment | — | `n_signal == Gt` | One 1-D token is one video tubelet in time. |
| Stage-2 loss weight | $\lambda_m$ | `--signal_weight`, `--loss_weights` | $\lambda_m = 1$ for visual streams, $\lambda_m =$ `signal_weight` for physio streams. |
| Stage-2 composite weights | $\alpha, \beta, \gamma$ (Stage 2) | `--resp_loss_mse_weight`, `--resp_loss_pearson_weight`, `--resp_loss_spectral_weight` | MSE / masked Pearson / band-limited spectral (ADR-008 §2.2). Neutral $(1, 0, 0)$. |
| Stage-2 physiological weight | $\lambda_\text{resp}$ | `--signal_weight` | Weight of $L_\text{resp}$ against the visual weight `1.0`; ADR-008 arm value `5.0`. |
| Stage-2 alignment weight | $\lambda_\text{align}$ | `--align_weight` | Weight of the InfoNCE regularizer; `0.0` allocates no heads. |
| Masked sample count | $M_\text{pts}$ | `n_masked * k` | Correlation length per sample: `8000` in Mode 1, `{4800, 5600, 6400, 7200}` in Mode 2. |
| Respiratory bin set | $\mathcal{K}_\text{resp}$ | `BandLimitedSpectralLoss.band_bins` | FFT bins inside $[f_{\min}, f_{\max}]$; `16` bins at `L = 8000`, `fs = 1000 Hz`. |
| FFT length | $N_\text{fft}$ | `--spectral_n_fft` | `0` = auto, $4\cdot 2^{\lceil\log_2 L\rceil}$ (`32768` at `L = 8000`). |
| Alignment projection | $d_\text{proj}$ | `--align_proj_dim` | InfoNCE projection width (`128`); adds `1,378,048` parameters at $D = 768$. |
| InfoNCE temperature | $\tau$ | `--align_temperature` | Cosine logit scale (`0.07`). |
| Core pooling weight | $w_\text{core}$ | `--align_core_weight` | Weight of $A_\text{core}$ tokens in the thermal pool (`1.0` = plain mean). |
| Stage-3 loss weights | $\alpha, \beta, \gamma$ (Stage 3) | `--alpha`, `--beta`, `--gamma` | Time / Pearson / MR-STFT. |

---

## 8. Dataset, Split & Config Terms

- `data_set`: `bp4d+` / `paired`, `tir_roi` / `tir_roi_resp` / `tir-roi`,
  `rgb_roi` / `rgb_roi_resp` / `rgb-roi`, `tir_roi_resp_rgb2d` / `trimodal`
  (Stage-2 only), `tir_roi_resp_rgb2d_bp` (Stage-2 only, four-stream).
- Dataset classes: `PairedPretrainDataset`, `PairedSessionDataset`,
  `RGBRoiPretrainDataset`, `RGBRoiFinetuneDataset`, `TirRoiRespPretrainDataset`,
  `TirRoiRespFinetuneDataset`, `TirRoiResp2DPretrainDataset`,
  `TirRoiResp2DBPPretrainDataset`.
- Splits: `FoldPlan` (immutable role assignment), `partition.json`
  (`<exp_root>/fold<k>/`, write-once), `partition digest` (sha256 over the roles
  and their seeds), `split_seed` / `split_stratify` (`none` / `sex`),
  `val_select` (`fixed_epoch` / `best_val`), `val_criterion`, `val_kept_<stream>`,
  `eval_freq`, `dev_smoke`, `selection.json`, `--eval_role test`.
  Retired: `n_folds`, `use_small_val`, `split_role`, `train_ratio`,
  `val_subject`, `split_by`, `pretrain_split`, `fold_manifest.json`.
- Task labels: `T1`-`T10`; task groups (`SHIPPED_TASK_GROUPS`, v2):
  `low=[T2,T3,T5]`, `moderate=[T1,T4]`, `high=[T6,T7,T8,T9,T10]`;
  `task_set`, `task_level`.
- ROI: `roi_padding` (frozen cache key), `crop_rule`, `roi_landmarks`,
  `max_sway_px`.
- Respiration artefact (`resp_cache`): the FROZEN identity fields `fs_in`,
  `fs_out`, `sat_v`, `sat_min_s`, `flat_window_s`, `flat_ptp_v`, `min_run_s`,
  `resample{kind, taps_per_phase, up, down, kernel_radius_src}`, `store`; the
  shard members `signal` (NaN where invalid), `mask` (the SOLE validity
  statement), `cause` (`0` valid / `1` rail / `2` flatline / `3` short run /
  `4` non-finite); the window arithmetic `s = round(t_start·fs_out)`,
  `n = round(clip_seconds·fs_out)`.
- `resp_preprocess` (decision + grid: `validity_mask`, `contiguous_runs`,
  `preprocess_session`), `resp_cache` (identity + shard IO + `find_cache`),
  `resp_window` (the ONLY window rule: `window_indices`, `usable`,
  `read_window`, `window_cause`, `assert_clip_fits`, `verify_identity`).
- Blood-pressure artefact (`bp_artefact`): the FROZEN identity fields
  `format_version`, `corpus`, `fs_in`, `fs_out`, `gate1_min_mmhg`,
  `gate1_max_mmhg`, `flatline_max_ms`, `rail_dwell_min_ms`, `min_segment_s`,
  `store`; the shard members `signal` (NaN where invalid), `mask` (the SOLE
  validity statement), `cause` (`0` valid / `1` non-finite / `2` range /
  `3` flatline / `4` rail / `5` short segment), `seg_start`, `seg_stop`,
  `seg_mean_mmhg`, `seg_seconds`. **`segment`** here means a maximal run of
  admitted samples on the 1000 Hz grid (>= `min_segment_s`), never a clip.
  `bp_preprocess` (decision: `gate1_range`, `gate2_flatline`, `gate3_rail`,
  `admitted_mask`, `preprocess_session`), `bp_artefact` (identity + shard IO +
  `find_cache` + `discover_sessions`), `bp_windows` (the ONLY window rule:
  `window_indices`, `usable`, `read_window`, `segment_of`, `verify_identity`).
- `rgb2d_cache` (`STORE_RESOLUTION='native'`).
- Legacy raw-path gates (refused when the artefact is attached): `rail_touch_v`,
  `min_signal_spread`.
- Models: `project_multimae_base` (768-d / 12 layers), `pretrained_encoder` in
  `{videomae:k400, videomae:ssv2}`.
