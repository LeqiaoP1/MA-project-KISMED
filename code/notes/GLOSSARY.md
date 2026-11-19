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
| **AMMM** | Asymmetric Multi-Modal Masking (`core/ammm_masking.py`, `masking: ammm`) | the legacy `masking: ratio` path |

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
`sig_kernel/fs == tubelet_t * temporal_stride / fps` **and** `n_signal == Gt`.

**Masking terms**
- **Asymmetric masking**: one ratio per stream, applied independently.
- **Shipped ratios** (configs are authoritative): visual `0.90`; `resp`
  `0.50`; `bp` `0.50` (TIR arm) or `0.70` (`stage2_local_rgb_roi_bp.yaml`). A
  physio ratio near 1.0 is forbidden in practice — a 1-D stream's only gradient
  path to the encoder is its VISIBLE tokens.
- **AMMM** (Asymmetric Multi-Modal Masking): `masking: ammm` draws per-step
  kept-unit counts via a Dirichlet over the drawn streams. Knobs:
  `dirichlet_alpha` (1.0), `video_kept_cap` (0.25), `physio_kept_cap` (1.0),
  `min_kept` (0), `modality_dropout` (0.0), `kept_ratio_override` (`{}`).
- **`physio_mask`**: `random` (scattered dropout, historical default) or `span`
  (contiguous blocks of `mask_span_s` seconds). `n_spans` is DERIVED, not a key.
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
  physiological stream; visual streams stay `1.0`.
- `loss_weights` (`--loss_weights`, CSV, one value per stream in `--streams`
  order): per-stream override of the whole policy.

**Stage 3** — `WaveformJointLoss` (`code/core/waveform_losses.py`):

$$L_\text{joint} = \alpha L_\text{time} + \beta L_\text{Pearson} + \gamma L_\text{MR-STFT}$$

- `L_time`: `SmoothL1Loss` (Huber) on `pred - target`; `huber_beta` is the
  transition point in target-std units (default `1.0`). The only
  polarity- and scale-sensitive term.
- `L_Pearson`: `1 - r`; scale- and shift-invariant.
- `L_MR-STFT`: multi-resolution spectral error, `fft_sizes = (64, 128, 256)`,
  spectral convergence + log-magnitude L1. Samples whose target has
  (near-)zero energy are excluded via `target_norm_floor` (default `1e-6`).
- Flags: `--alpha`, `--beta`, `--gamma`, `--huber_beta`, `--fft_sizes`.
- There is **no spectral term in Stage 2** (removed 2026-11-09).

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
| Stage-3 loss weights | $\alpha, \beta, \gamma$ | `--alpha`, `--beta`, `--gamma` | Time / Pearson / MR-STFT. |

---

## 8. Dataset, Split & Config Terms

- `data_set`: `bp4d+` / `paired`, `tir_roi` / `tir_roi_resp` / `tir-roi`,
  `rgb_roi` / `rgb_roi_resp` / `rgb-roi`, `tir_roi_resp_rgb2d` / `trimodal`
  (Stage-2 only).
- Dataset classes: `PairedPretrainDataset`, `PairedSessionDataset`,
  `RGBRoiPretrainDataset`, `RGBRoiFinetuneDataset`, `TirRoiRespPretrainDataset`,
  `TirRoiRespFinetuneDataset`, `TirRoiResp2DPretrainDataset`.
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
- Caches: `resp_cache` / `resp_preprocess` (`sat_v`, `sat_min_s`,
  `flat_window_s`, `flat_ptp_v`, `min_run_s`, `fs_out`); `rgb2d_cache`
  (`STORE_RESOLUTION='native'`).
- Legacy raw-path gates (refused when a cache is attached): `rail_touch_v`,
  `min_signal_spread`.
- Models: `project_multimae_base` (768-d / 12 layers), `pretrained_encoder` in
  `{videomae:k400, videomae:ssv2}`.
