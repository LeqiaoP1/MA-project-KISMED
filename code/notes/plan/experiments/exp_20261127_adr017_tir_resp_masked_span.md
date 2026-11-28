# EXP-20261127-01: Two-Stream TIR-ROI + RESP Stage-2 Pre-Training with ADR-017 Masked-Span Placement

- **Experiment ID**: `EXP-20261127-01`
- **Author**: LeqiaoP1
- **Status**: Phase A complete — fold-0 Stage 2 finished at `checkpoint-0149.pth` and Gates **G0, G2 and G3 PASS** ([G0 record](../../eval/baselines/tir_resp/adr017_floor_probe_fold0_val.md), [G2/G3 record](../../eval/baselines/tir_resp/adr017_pretext_fold0.md)). Phase B (five-fold Stage 2 + Stage-3 adoption follow-up) is **not pursued**; G4 is not evaluated and ADR-017 stays `Proposed`.
- **Associated ADR**: [`code/notes/design/ADR-017-masked-span-physio-placement.md`](../../design/ADR-017-masked-span-physio-placement.md) (with [`ADR-007`](../../design/ADR-007-two-stream-anatomy-aware-constant-budget-masking.md) masking counts/curriculum and [`ADR-008`](../../design/ADR-008-composite-objective-and-cross-modal-alignment.md) composite objective unchanged)
- **Target Thesis Section**: Chapter 3 (Methodology — 1-D physiological masked-span placement on top of constant-budget masking); Chapter 4 (Experiments — two-stream `tir,resp` Stage-2 pretext diagnostics and paired Stage-3 waveform ablation vs. `scattered` placement).

> **Protocol Audit Summary**
> - **Files and configs inspected**: `code/notes/GLOSSARY.md`, `code/notes/design/ADR-017-masked-span-physio-placement.md`, `code/notes/plan/features/ADR-017-masked-span-physio-placement-plan.md`, `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`, `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml`, `code/configs/finetune/resp_tir_roi_hpc.yaml`, `code/notes/eval/baselines/tir_resp/{README.md,mode1_pretext_fold0.md,multimae_style_panel.md}`, `code/runners/{run_pretrain.py,run_eval_recon.py,run_probe_masking_baselines.py,run_partition.py,run_build_resp_cache.py}`, `code/scripts/hpc/{submit.sh,submit_pretrain_sweep.sbatch,submit_pretrain_sweep.sh,submit_pretrain_chain.sh,submit_sweep.sbatch,submit_sweep.sh,submit_roi_cache.sbatch}`.
> - **Active stage and runner**: Stage 2 pre-training via `code/runners/run_pretrain.py` (preceded by zero-GPU Criterion-3a floor probe via `code/runners/run_probe_masking_baselines.py` and followed by Stage-2 zero-shot probes via `code/runners/run_eval_recon.py`; adoption is gated on a downstream five-fold Stage-3 follow-up via `code/runners/run_waveform.py` + `code/runners/run_evaluate_kfold.py`).
> - **Falsifiable hypothesis**: Replacing `scattered` 1-D respiration placement with bounded contiguous masked spans (`physio_placement: masked_span`, hypothesis `63..125` tokens = `2.016..4.000 s`) eliminates intra-modal local interpolation in Mode 2 (`mse_interp > mse_const` for every $K_\text{resp} \in \{25, 50, 75, 100\}$), forcing Mode-2 steps (90 % of Phase I, 50 % of Phase II) to extract respiratory phase from `tir` and improving matched inpainting and downstream Stage-3 `band_limited_pearson` without degrading Mode-1 sensing.
> - **Exact parameter delta** (vs. `stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`): only `physio_placement: scattered -> masked_span`, `physio_gap_min_tokens: 0 -> 63`, `physio_gap_max_tokens: 0 -> 125` (hypothesis pending Gate G0 sweep); every other key is byte-identical.
> - **Baseline and metric contract**: Fold-0 ADR-008 scattered checkpoint in `code/notes/eval/baselines/tir_resp/` (Mode-1 `TIR-200` subject-median `r_masked = +0.3256 [+0.2444, +0.4061]`, `Var(ŷ) = 0.4716`, spectral peak error `4.92 bpm`; remote sensing `TIR-900` `+0.3819 [+0.2731, +0.5349]`; scattered Mode-2 `resp-max` `+0.9989`, `mid` `+0.9986`). Adoption metric: five-fold Stage-3 subject-level `band_limited_pearson` (0.10–0.60 Hz), threshold `>= +0.02` with 95 % CI excluding 0 against a matched ADR-008 Stage-3 fine-tune (baseline Stage-3 five-fold record does not exist yet and must be run with the identical recipe).
> - **Decision and rollback rule**: Adopt `masked_span` as arm default only if G0–G4 pass (five-fold Stage-3 Δ `band_limited_pearson >= +0.02`, 95 % CI > 0); reject if G0 is infeasible or Stage-3 Δ `<= -0.02`; otherwise keep `scattered`. Rollback: restore `physio_placement: scattered`, `physio_gap_min_tokens: 0`, `physio_gap_max_tokens: 0`.
> - **Local smoke and HPC commands**: See §7 (`python -u runners/run_pretrain.py -c configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml ...`) and §8 (`submit_pretrain_chain.sh` / `submit.sh scripts/hpc/submit_pretrain_sweep.sbatch`, plus `submit_sweep.sbatch` caveats for the five-fold Stage-3 adoption follow-up).

---

## 1. Scientific Hypothesis and Baseline

- **Core Hypothesis**: Under the ADR-008 two-stream (`tir,resp`) arm, Mode-2 waveform-inpainting steps constitute 90 % of Phase-I epochs (epochs 0–19) and 50 % of Phase-II epochs (epochs 20–149), yet `scattered` placement at 32 ms token resolution leaves 96.8–100 % of masked respiration gaps shorter than 1 s (`0.06–0.22 s` median gap). Those short gaps are solvable by intra-modal local interpolation (`r_masked = +0.9989` at 100/100 and `+0.9986` at 125/75 on the fold-0 ADR-008 checkpoint), so Mode-2 respiration gradients do not compel the shared encoder to read respiratory cues from `tir`. Constraining every maximal masked respiration run to a contiguous gap in $[g_\text{min}, g_\text{max}]$ (`physio_placement: masked_span`, hypothesis `63..125` tokens = `2.016..4.000 s`, spanning 0.2–2.4 cycles of the 0.10–0.60 Hz respiratory passband) breaks local interpolation (`mse_interp > mse_const` across all admissible Mode-2 counts $\mathcal{K}_2 = \{25, 50, 75, 100\}$) while keeping token counts, encoder shape `[B, 200, 768]`, ADR-007 curriculum, ADR-008 objective, and Mode 1 untouched. Consequently, (1) the trained encoder will outperform the no-video linear interpolator on matched `masked_span` inpainting without degrading Mode-1 cross-modal sensing, and (2) the resulting representation will improve downstream Stage-3 contactless respiration recovery.
- **Comparison Baseline**: [`code/notes/eval/baselines/tir_resp/`](../../eval/baselines/tir_resp/README.md) — matched fold-0 ADR-008 two-stream `scattered` arm (`checkpoint-0149.pth`, epoch 149 = all 150, `stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`, 27 held-out test subjects / 213 sessions / 1,742 clips, 5 repeats, seed 0):
  - Mode-2 `scattered` `resp-max` (`100/100`): subject-median `r_masked` = **+0.9989** `[+0.9986, +0.9990]`, MSE mean = `0.0037 z²` ([`multimae_style_panel.md`](../../eval/baselines/tir_resp/multimae_style_panel.md) §4).
  - Mode-2 `scattered` `mid` (`125/75`): subject-median `r_masked` = **+0.9986** `[+0.9983, +0.9987]`, MSE mean = `0.0043 z²`.
  - Mode-1 `TIR-200` (`200/0`): subject-median `r_masked` = **+0.3256** `[+0.2444, +0.4061]`, `Var(ŷ)` = `0.4716` `[0.457, 0.530]`, spectral peak error = `4.92 bpm` `[4.01, 5.61]`, MSE mean = `0.9794 z²` (median `0.963 z²`), Phase-II median `grad_ratio` = `66.5` ([`mode1_pretext_fold0.md`](../../eval/baselines/tir_resp/mode1_pretext_fold0.md)).
  - Budget-exempt remote sensing `TIR-900` (`900/0`): subject-median `r_masked` = **+0.3819** `[+0.2731, +0.5349]`, `Var(ŷ)` = `0.6518`, spectral peak error = `3.66 bpm`; rolled-TIR control `TIR-900-rolled`: subject-median `r_masked` = **+0.0823** `[+0.0379, +0.1527]`.
  - **Stage-3 five-fold baseline**: Does **not** exist yet for the matched ADR-008 1000 Hz / `resp_cache` arm. Before evaluating Adoption Gate G4, the matched scattered ADR-008 arm must be fine-tuned under the identical five-fold Stage-3 recipe.
- **Data Lineage**: `data_set: tir_roi` (`TirRoiRespPretrainDataset`), reading the offline native-resolution TIR-ROI pixel cache (`--roi_cache`, ADR-011) and the offline 1000 Hz respiration session cache (`--resp_cache`, ADR-003). ADR-017 modifies no dataset or cache key; both existing caches remain valid unchanged.
- **Expected Delta**:
  - **Pre-training (G0, fold-0 `val`, zero GPU)**: Under the adopted `(g_min, g_max)`, `mse_interp > mse_const` for all $K_\text{resp} \in \{25, 50, 75, 100\}$ (while `scattered` sanity check satisfies `mse_interp << mse_const`).
  - **Stage-2 fold-0 `test` (G2, non-gating)**: On matched `masked_span` probes (`resp-max` 100/100 and `mid` 125/75), subject-median paired `r_masked(model) - r_masked(np.interp)` has 95 % bootstrap CI strictly `> 0`.
  - **Stage-2 fold-0 `test` (G3, non-gating)**: Mode-1 (`TIR-200`) subject-median `r_masked` does not drop by more than `0.02` below `+0.3256` (i.e. stays `>= +0.3056`); `TIR-900` remote-sensing probe and `TIR-900-rolled` control reported alongside.
  - **Stage-3 five-fold `test` (G4, adoption gate)**: Pooled subject-median `band_limited_pearson` (0.10–0.60 Hz) improvement `>= +0.02` over the matched scattered ADR-008 five-fold Stage-3 arm with 95 % bootstrap CI excluding `0`.
  - **Within-arm caveat (ADR-017 §4)**: Stage-2 `loss`, `mse_resp`, and `grad_ratio` are expected to **rise** relative to the `scattered` arm because Mode-2 gaps are no longer locally interpolatable; they are monitored within-arm only and are not cross-arm comparison metrics.

---

## 2. Configuration and Parameter Delta

*Hierarchy: explicit CLI flags > YAML config > built-in defaults. Only values changed from the verified comparator (`stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`) are listed as deltas.*

- **Stage**: Stage 2 pre-training (fold 0 first for Criterion 3b diagnostics; five-fold expansion + Stage 3 only after G0–G3 are audited)
- **Runner**: `code/runners/run_pretrain.py`
- **Base Comparator Config**: `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`
- **Arm Config (shipped)**: `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml`
- **Output Artifacts Directory**:
  - Local smoke: `/tmp/exp_20261127_adr017_tir_resp/`
  - HPC fold-0 / five-fold sweep: `$WORK_SCRATCH/exp_20261127_adr017_tir_resp/fold<k>/stage2/` (mirrored summary records in `code/notes/eval/` after completion)
- **Initial Checkpoint**:
  - Stage 2: `pretrained_encoder: videomae:k400` (`patch_stream: tir`; pre-fetched once on a login node via `bash scripts/hpc/download_weights.sh` so `tir` receives the pretrained Conv3d `(2,16,16)` kernel verbatim with zero auxiliary visual adapter debt).
  - Stage 3 (follow-up Criterion 3c only): `<exp_root>/fold<k>/stage2/checkpoints/checkpoint-0149.pth`, matched by `partition_digest` against `<exp_root>/fold<k>/partition.json`.
- **Explicit Parameter Overrides (exact delta vs. `stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`)**:

```yaml
# ONLY differences vs. stage2_budget_tir_resp_2stream_anatomy_adr008.yaml:
physio_placement: masked_span    # was: scattered
physio_gap_min_tokens: 63        # was: 0 (neutral); 63 x 32 ms = 2.016 s (Criterion-3a hypothesis)
physio_gap_max_tokens: 125       # was: 0 (neutral); 125 x 32 ms = 4.000 s (Criterion-3a hypothesis)
physio_span_tokens: 0            # unchanged; required 0 when physio_placement != span
```

*Note on G0 update rule*: `(63, 125)` is ADR-017's Criterion-3a hypothesis, not a pre-frozen value. If the Phase-5 / Gate-G0 floor sweep selects a wider feasible pair (e.g. `63:150`, `75:150`, or `87:150` — keeping $g_\text{max} \le 150$ so $M \in \{175, 200, 225\}$ still retain $n \ge 2$ gaps), update `physio_gap_min_tokens` and `physio_gap_max_tokens` in `stage2_budget_tir_resp_2stream_anatomy_adr017.yaml` (or pass matching CLI overrides) and record the adopted point before launching Stage 2.

- **Key Inherited Base Parameters (byte-identical to the ADR-008 comparator, held fixed)**:

```yaml
# Data, split & geometry
data_set: tir_roi
streams: tir,resp
tir_channels: 3
model: project_multimae_base
pretrained_encoder: videomae:k400
patch_stream: tir
clip_duration: 8.0
clip_stride: 4.0
fps: 25.0
temporal_stride: 4
tubelet: 2,16,16
input_size: 96
fs: 1000.0
sig_kernel: 32
seq_len: 0
roi_padding: 0.10
crop_rule: square
max_sway_px: 40
roi_landmarks: ''
min_signal_spread: 0.0
rail_touch_v: 0.0
physio_norm: zscore
split_seed: 42
split_stratify: sex
eval_freq: 5
val_select: fixed_epoch
val_criterion: joint_loss

# ADR-007 constant-budget masking policy (counts, modes, curriculum unchanged)
n_budget: 200
video_cap_ratio: 0.25
physio_cap_ratio: 0.40
video_min_tubes: 4
p_virtual_sensing: 0.50
virtual_sensing_warmup_p: 0.10
virtual_sensing_warmup_epochs: 20
spatial_sampling: anatomy_aware
core_row_bounds: 2,3
core_col_bounds: 2,3
core_tubes_min: 1
core_tubes_max: 2
periph_min_tubes: 1
physio_draw_min: 1
spatial_patches: 36
physio_tokens: 250

# ADR-008 composite objective & InfoNCE alignment (unchanged)
signal_weight: 5.0
resp_loss_mse_weight: 1.0
resp_loss_pearson_weight: 2.0
resp_loss_spectral_weight: 1.0
spectral_f_min: 0.10
spectral_f_max: 0.60
spectral_n_fft: 0
align_weight: 0.5
align_proj_dim: 128
align_temperature: 0.07
align_core_weight: 1.0
align_gather_ddp: true
grad_balance_freq: 50

# Optimisation
epochs: 150
batch_size: 16
save_ckpt_freq: 10
opt: adamw
blr: 1.5e-4
min_lr: 0.0
warmup_epochs: 10
```

Do not use `temporal_window`, `mask_ratio_visual`, `mask_ratio_signal`, or `loss_weight_waveform`; these are not active repo keys.

---

## 3. Data and Split Provenance

- **Dataset**: `data_set: tir_roi` (`TirRoiRespPretrainDataset`, raw BP4D+ tree paired with offline caches via `--roi_cache` and `--resp_cache`).
- **Subjects/Sessions/Windows**:
  - Fold-0 partition verified on the identical ADR-008 `tir_roi` lineage (`code/notes/eval/baselines/tir_resp/README.md` & `mode1_pretext_fold0.md`): 97 train / 28 test subjects assigned in `partition.json` (plus val subjects; exact train/val clip counts are read from the run's `partition.json` and log — not estimated here); **27 held-out test subjects / 213 sessions / 1,742 clips** pass the corpus validity gates on fold 0.
  - Folds 1–4 counts: **Not yet measured**; read from each fold's `partition.json` and training log.
- **Partition**: `<exp_root>/fold<k>/partition.json` (write-once via `runners/run_partition.py`; verified identical digest across Stage 2, Stage-2 probes, and Stage 3).
- **Split Policy**: `split_seed=42`, `split_stratify=sex`, subject-disjoint `train` / `val` / `test` roles (ADR-001).
- **Fold Scope**:
  - **Phase A (Diagnostic)**: Fold `0` only (`test` never loaded during pre-training; `val` used for Gate G0 floor sweep and fixed-seed validation curve; `test` scored once after epoch 149 for Gates G2/G3).
  - **Phase B (Adoption, conditional on G0–G3)**: All five folds `0 1 2 3 4`, sequential, followed by matched five-fold Stage 3 for both ADR-017 and ADR-008 arms.
- **Cache Dependencies**:
  - **TIR-ROI pixel cache (`--roi_cache`)**: Required for HPC training throughput. Cache keys: `roi_padding=0.10`, `crop_rule=square`, `roi_landmarks=''` (12-point `nose_mouth`), `clip_seconds=8`, `fps=25`. Build command (CPU-only, from `code/`):
    ```bash
    sbatch scripts/hpc/submit_roi_cache.sbatch
    # then on login node:
    python -u runners/run_build_roi_cache.py --status --out_root "$WORK_SCRATCH/tir_roi_cache"
    python -u runners/run_build_roi_cache.py --index  --out_root "$WORK_SCRATCH/tir_roi_cache"
    python -u runners/run_build_roi_cache.py --check  --out_root "$WORK_SCRATCH/tir_roi_cache"
    ```
  - **1000 Hz respiration session cache (`--resp_cache`)**: Required (`min_signal_spread: 0.0`, `rail_touch_v: 0.0` are pinned to `0.0` because validity comes solely from the cache; non-zero legacy raw filters are refused when `--resp_cache` is attached). Build command (CPU-only, from `code/`):
    ```bash
    python -u runners/run_build_resp_cache.py --init --build \
      --out_root "$WORK_SCRATCH/resp_cache" --raw_root "$RAW_DATA_PATH"
    python -u runners/run_build_resp_cache.py --status --out_root "$WORK_SCRATCH/resp_cache"
    python -u runners/run_build_resp_cache.py --check 20 --out_root "$WORK_SCRATCH/resp_cache"
    ```
- **Selection Rule**: `val_select: fixed_epoch` (`val_criterion: joint_loss`, `eval_freq: 5`). The evaluated checkpoint is the final epoch (`checkpoint-0149.pth` at 150 epochs). Validation replays a fixed-seed mask sequence (ADR-007 §2.7), so validation loss is within-phase only and must not be pooled across `E_warm = 20` or compared across placements.

---

## 4. Geometry and Modality Contract

- **Visual Stream**: `tir` (3-channel false-colour thermal ROI), shape `[B, 3, 50, 96, 96]` (200 source frames at `fps = 25.0` decimated by `temporal_stride = 4` → `T = 50` frames). With `tubelet = (2, 16, 16)`, grid is `(G_t, G_h, G_w) = (25, 6, 6)` → `36` spatial tubes × `25` tokens/tube = `900` total visual tokens.
- **Physiological Stream**: `resp` (1-D respiration waveform), shape `[B, 1, 8000]` (`clip_duration = 8.0 s × fs = 1000.0 Hz`, `seq_len = 0`). With `sig_kernel = 32`, `N_s = 250` signal tokens (`32 ms` each).
- **ROI Contract**: `roi_padding: 0.10`, `crop_rule: square` (isotropic rescale, no aspect distortion), `roi_landmarks: ''` (12-point TIR `nose_mouth` set), `max_sway_px: 40`.
- **Time Alignment**:
  $$\frac{\text{sig\_kernel}}{fs} = \frac{32}{1000} = 0.032\ \text{s},\qquad \frac{\text{tubelet}_t \cdot \text{temporal\_stride}}{fps} = \frac{2 \cdot 4}{25} = 0.320\ \text{s},$$
  giving exact integer ratio $R = N_s / G_t = 250 / 25 = 10$ signal tokens per visual tubelet (`MultiModalMAE._check_space_time_alignment`).
- **Masking Geometry & Feasibility (ADR-007 + ADR-017)**:
  - Constant encoder input length: `[B, 200, 768]` on every training step (`n_budget = 200`).
  - Mode 1 (virtual sensing): `(K_tir, K_resp) = (200, 0)` (8 of 36 tubes, `S_core ∈ {1, 2}`, `S_periph = 8 - S_core`). Consumes **zero** placement RNG draws (ADR-017 R4/I4), keeping Mode-1 and remote-sensing masks paired with the ADR-008 arm.
  - Mode 2 (waveform inpainting): $\mathcal{K}_2 = \{25, 50, 75, 100\}$, masked counts $\mathcal{M}_2 = \{225, 200, 175, 150\}$. Under hypothesis `(63, 125)`: $\mathcal{N}(150) = \mathcal{N}(175) = \{2\}$ and $\mathcal{N}(200) = \mathcal{N}(225) = \{2, 3\}$ (every gap `2.016..4.000 s`, separated by ≥ 1 kept token).
- **Normalisation**: Dataset-side per-clip z-score `physio_norm: zscore` (`ddof=0`, `ZSCORE_EPS = 1e-6`); model-side `target_norm` pinned to `'none'` by `pin_roi_target_norm` for every ROI lineage.
- **Checkpoint Compatibility**:
  - `96 px` / `tubelet (2,16,16)` / `tir_channels = 3` matches `videomae:k400` and the ADR-008 fold-0 checkpoint verbatim (`103,906,080` parameters, including `1,378,048` Stage-2-only alignment-head parameters dropped on Stage-3 transfer). Incompatible with `64 px` local checkpoints or `tir_channels = 1`.
  - **Stage-3 geometry alignment note**: `code/configs/finetune/resp_tir_roi_hpc.yaml` currently ships the legacy `100 Hz` / `temporal_stride: 2` / `sig_kernel: 16` / raw-filter (`min_signal_spread: 0.1`, `rail_touch_v: 9.900`) parameters. Before executing Criterion-3c Stage-3 fine-tuning, both the ADR-008 and ADR-017 Stage-3 runs must use matching `fs: 1000`, `temporal_stride: 4`, `sig_kernel: 32`, `min_signal_spread: 0.0`, `rail_touch_v: 0.0`, and `--roi_cache` / `--resp_cache` (identical across both arms; `run_waveform.py` aborts on ROI/partition mismatch).

---

## 5. Evaluation Contract

Three evaluation layers enforce strict separation between design validation (`val`), Stage-2 pretext diagnostics (`test`, non-gating), and downstream adoption (`test`, five-fold Stage 3, gating).

1. **Layer 0 — Criterion-3a No-Video Floor Probe (zero GPU, fold-0 `val`)**:
   `runners/run_probe_masking_baselines.py` scoring `np.interp` (linear between visible samples, constant-clamped at edges) vs. constant-zero predictor (`mse_interp`, `mse_const`, `r_masked_interp`) across all $K \in \{25, 50, 75, 100\}$, `--repeats 5`, `--seed 0`, over `scattered`, legacy `span`, and `--gap_grid 63:125,63:150,75:150,87:150`.
2. **Layer A — Stage-2 Zero-Shot Probes (fold-0 `test`, non-gating, `runners/run_eval_recon.py`)**:
   - **Matched `masked_span` inpainting** (`--resp_probe masked_span --interp_baseline --pretext_metrics --repeats 5 --seed 0 --batch_size 16 --preload`): scored at `resp-max` (`--kept_tir 0.111 --kept_resp 0.40` → 100/100) and `mid` (`--kept_tir 0.139 --kept_resp 0.30` → 125/75), plus paired comparison against `r_masked_interp` and `mse_const` on identical masks.
   - **Mode-1 budgeted sensing** (`--resp_probe hidden --kept_tir 1.0 --kept_resp 0.0 --pretext_metrics`): `TIR-200` (`200/0`), reporting subject-median `r_masked`, `Var(ŷ)`, and spectral peak error (bpm) paired against the ADR-008 baseline.
   - **Remote-sensing deployment & content control** (`--exempt_budget --kept_tir 1.0 --resp_probe hidden --pretext_metrics`): `TIR-900` (`900/0`) and `TIR-900-rolled` (`--ablate_stream tir --ablate_mode roll`).
   - **Within-arm training diagnostics**: `mse_tir`, `mse_resp`, `resp_pearson`, `resp_spectral`, `grad_ratio` (read within-arm only; never compared numerically across placements).
3. **Layer B — Stage-3 Downstream Headline (five-fold `test`, gating, `runners/run_waveform.py --eval_role test` + `runners/run_evaluate_kfold.py`, ADR-009)**:
   - **Primary Tier 1 Metrics**: `band_limited_pearson` (0.10–0.60 Hz, headline, higher is better), plus `pearson`, `mae`, `rmse`, `amp_ratio`, `amp_rel_err`, and `mae_bpm`.
   - **Tier 2 Spectral Metrics**: `psd_mae`, `dominant_freq_error_hz`, `hf_power_rel` (and MR-STFT tracking).
   - **Tier 3 Physiological Metrics**: **Not applicable** — `extract_hrv_metrics` applies to `bp` waveforms ≥ 30 s; this experiment targets 8 s `resp` clips (and the repository does not estimate systolic/diastolic BP).
   - **Aggregation**: Layer A aggregates clip → median per subject → median with 95 % bootstrap CI over subjects (2,000 resamples), matching `multimae_style_panel.md`. Layer B aggregates clip → session → median per subject → pooled five-fold subject median with 95 % bootstrap CI over all held-out subjects (ADR-009).
   - **Baseline**: Verified fold-0 Stage-2 baseline in `code/notes/eval/baselines/tir_resp/` (`TIR-200` `r_masked = +0.3256 [+0.2444, +0.4061]`, `TIR-900` `+0.3819 [+0.2731, +0.5349]`). Five-fold Stage-3 baseline for ADR-008 does not yet exist and must be produced with the identical Stage-3 recipe before reading G4.

- **Numerical Gate**:

| Gate | Metric | Required Threshold | Measurement Scope |
|---|---|---:|---|
| G0 (3a, pre-train) | `mse_interp` vs `mse_const` under `masked_span` (and `scattered` sanity) | `mse_interp > mse_const` for **every** $K \in \{25, 50, 75, 100\}$; `scattered` sanity `mse_interp < 0.25 * mse_const` | Fold-0 `val` role, `>= 5` draws/clip, zero GPU |
| G1 (C2, smoke/res) | Smoke invariants + CPU placement latency | `0` NaN/Inf, `[B, 200, 768]`, both modes observed, manifest/banner record `masked_span` + bounds + seconds + $\mathcal{N}(M)$; placement `< 5 ms/step` at `B = 16` | Local smoke + unit timing |
| G2 (3b, Stage-2) | Paired subject-median `r_masked(model) - r_masked(np.interp)` on matched `masked_span` (`100/100` & `125/75`) | 95 % bootstrap CI lower bound `> 0` | Fold-0 `test` role (27 subjects, 5 draws/clip), non-gating |
| G3 (3b, Stage-2) | Mode-1 (`TIR-200`) subject-median `r_masked` vs. ADR-008 (`+0.3256`) | `>= +0.3056` (no drop `> 0.02`; flag & inspect if lower) | Fold-0 `test` role (27 subjects, 5 draws/clip), non-gating |
| G4 (3c, Adoption) | Pooled 5-fold Stage-3 subject-median Δ `band_limited_pearson` (`masked_span` − `scattered` ADR-008) | **Adopt**: `>= +0.02` & 95 % CI excludes `0`; **Reject**: `<= -0.02`; else keep `scattered` | All 5 folds held-out `test` subjects (ADR-009) |

---

## 6. Resource Envelope

- **Hardware**:
  - Gate G0 (floor probe): CPU-only (login or `deflt` CPU node; zero GPU).
  - Stage-2 pre-training: Lichtenberg HPC `acc_short` (chained 30-min links via `submit_pretrain_chain.sh`) or `acc_long` (`submit_pretrain_sweep.sbatch`), 1 node, `gpu:a100:4`, `ntasks=4` (DDP world size 4), `cpus-per-task=16`, `--mem-per-cpu=1G` (chain) or `4G` (`submit_pretrain_sweep.sbatch`).
  - Stage-2 zero-shot probes (`run_eval_recon.py`): 1 GPU (`a100` or `rtx6000`), submitted **through Slurm** (`sbatch` / `submit.sh`, `--gres=gpu:a100:1`), ~10–15 min per probe condition at 5 repeats with `--preload`. **Never on a login node** — the `log*` hosts share one communal T4 with other tenants and kill terminal-attached processes on session teardown (`notes/spec/04_configs_and_execution.md` §4.2.1). Measured 2026-11-28: a login-node probe chain was `SIGKILL`ed after `preloaded 1742 clip(s) into host RAM`, leaving no traceback and no output.
- **Peak VRAM Cap**: **Provisional estimate** — identical to the matched ADR-008 two-stream arm (same 103,906,080 parameters, `[B, 200, 768]` encoder input, `[B, 3, 50, 96, 96]` TIR input, `[B, 1, 8000]` RESP input; placement runs on CPU and adds `< 5 ms/step` and `0` GPU parameters). Record actual peak allocated/reserved VRAM from link 1 rather than assuming.
- **Batch Size**: `16` per GPU (`64` global across 4 GPUs; `lr = blr * 64 / 256 = 3.75e-5` peak after 10-epoch linear warmup).
- **Workers**: `num_workers` default from runner (`--num_workers 0` in local smoke).
- **Expected Runtime**:
  - Fold-0 Stage 2 (150 epochs, 4× A100): **Measured reference from identical ADR-008 arm** (`code/notes/eval/baselines/tir_resp/README.md`) — **13 `acc_short` links × 25 min (~5.4 wall-clock hours, ~21.7 GPU-hours)** per fold.
  - Five-fold Stage 2 (if promoted): ~27 wall-clock hours (~108 GPU-hours) across 5 folds (estimate scaled linearly from measured fold-0 ADR-008 runtime).
  - Five-fold Stage 3 (Criterion 3c): **Not yet estimated** for the 1000 Hz `resp_cache` lineage; measure s/epoch on fold 0 before launching all 5 folds.
- **Abort Rule**:
  - Gate G0 fails at all feasible `(g_min, g_max)` pairs → **do not submit Stage 2**.
  - Any `NaN`/`Inf` loss, CUDA OOM, or `mask_placement` / `budget_masking` / `multimae` invariant failure.
  - Encoder input shape deviates from `[B, 200, 768]`.
  - Any `masking.json` policy mismatch on resume (`_stable()` check) or `partition_digest` mismatch across stages.
  - `mse_tir` diverges or fails to decrease in Phase I (verify `grad_ratio` stability within-arm).

---

## 7. Local Smoke Verification

Run from `code/` with the active virtual environment (`source scripts/env_local.sh`):

**Step 7.1 — Unit & contract suite (zero data, CPU)**

```bash
pytest tests/test_adr017_masked_span_placement.py tests/test_mask_placement.py \
  tests/test_budget_masking.py tests/test_two_stream_anatomy_masking.py \
  tests/test_task_group_configs.py tests/test_config_loading.py -q
```

**Step 7.2 — Minimal one-step Stage-2 smoke (`--max_entries 2 --batch_size 1`)**

```bash
python -u runners/run_pretrain.py \
  -c configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml \
  --exp_root /tmp/exp_20261127_adr017_smoke --fold 0 --dev_smoke \
  --epochs 1 --warmup_epochs 0 --virtual_sensing_warmup_epochs 0 \
  --num_workers 0 --max_entries 2 --batch_size 1 \
  --output_dir /tmp/exp_20261127_adr017_smoke/out
```

*(Add `--roi_cache <LOCAL_ROI_CACHE> --resp_cache <LOCAL_RESP_CACHE>` when testing against local caches; if running on raw files without `--resp_cache`, override `--min_signal_spread 0.1 --rail_touch_v 9.900`.)*

**Step 7.3 — Multi-step Mode-1/Mode-2 & gradient-balance smoke (ADR-017 Criterion 2)**

```bash
python -u runners/run_pretrain.py \
  -c configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml \
  --exp_root /tmp/exp_20261127_adr017_c2 --fold 0 --dev_smoke \
  --epochs 1 --warmup_epochs 0 --virtual_sensing_warmup_epochs 0 \
  --max_entries 100 --batch_size 8 --num_workers 0 --grad_balance_freq 2 \
  --output_dir /tmp/exp_20261127_adr017_c2/out
```

- **Success Condition**: Exit code `0`, no `NaN`/`Inf`, encoder input `[B, 200, 768]` on every step, both Mode 1 (`single_visual`) and Mode 2 (`inpainting`) observed in Step 7.3, `[mask]` startup banner and `masking.json` record `physio_placement: masked_span`, both token bounds, resolved seconds (`2.016..4.000 s` at `63..125`), and $\mathcal{N}(M)$ for every $M \in \{150, 175, 200, 225\}$, and `partition.dev.json` written cleanly.
- **Recorded Artifacts**: Smoke stdout/stderr log, `<out>/masking.json`, `<out>/selection.json`, `<exp_root>/fold0/partition.dev.json`, and `<out>/checkpoints/`.

---

## 8. HPC and Full Run

All commands run from `code/` after `source scripts/env_hpc.sh && activate_project_env` (and ensuring `videomae:k400` weights were pre-fetched once via `bash scripts/hpc/download_weights.sh` and caches verified per §3).

### Step 8.1 — Gate G0: Criterion-3a No-Video Floor Sweep (CPU-only, MUST PASS FIRST)

Use the existing fold-0 ADR-008 checkpoint solely for its recorded geometry/policy args (no weights loaded, zero GPU):

```bash
ADR008_ROOT=/work/scratch/ne95ocyg/pretrain/adr008_2stream_fold0
python -u runners/run_probe_masking_baselines.py \
  --checkpoint "$ADR008_ROOT/fold0/stage2/checkpoints/checkpoint-0149.pth" \
  --partition "$ADR008_ROOT/fold0/partition.json" \
  --split val --repeats 5 --seed 0 \
  --gap_grid 63:125,63:150,75:150,87:150 \
  --out ../output/eval/adr017_floor/fold0_val_floor.json
```

Record the measured sweep under `code/notes/eval/baselines/tir_resp/` before proceeding. If `(63, 125)` fails `mse_interp > mse_const` at any $K \in \{25, 50, 75, 100\}$, update `physio_gap_min_tokens` and `physio_gap_max_tokens` to the tightest passing feasible point from the sweep (or abort if none pass).

### Step 8.2 — Phase A: Fold-0 Stage-2 Pre-Training (Chained on `acc_short`, recommended)

Matches the verified 13-link `acc_short` schedule used for the ADR-008 fold-0 baseline (`submit_pretrain_chain.sh` wraps `scripts/hpc/submit_pretrain_sweep.sbatch` -> `scripts/hpc/submit_pretrain_sweep.sh`, creating `<EXP_ROOT>/fold0/partition.json` once and auto-resuming across `--dependency=afterany` links):

```bash
EXP_ROOT="$WORK_SCRATCH/exp_20261127_adr017_tir_resp" \
FOLDS="0" \
CFG2="configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml" \
DATA_SET2="tir_roi" \
ROI_CACHE="$WORK_SCRATCH/tir_roi_cache" \
RESP_CACHE="$WORK_SCRATCH/resp_cache" \
LINKS=13 TIME="00:25:00" PARTITION="acc_short" GPU_TYPE="a100" GPUS=4 SAVE_FREQ=10 \
bash scripts/hpc/submit_pretrain_chain.sh
```

*(Dry-run first by prefixing `DRY_RUN=1` to inspect the exact `sbatch` chain. To run a single continuous allocation on `acc_long` instead of chaining on `acc_short`, submit via `submit.sh`:)*

```bash
EXP_ROOT="$WORK_SCRATCH/exp_20261127_adr017_tir_resp" \
FOLDS="0" \
CFG2="configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml" \
DATA_SET2="tir_roi" \
ROI_CACHE="$WORK_SCRATCH/tir_roi_cache" \
RESP_CACHE="$WORK_SCRATCH/resp_cache" \
GPU_TYPE="a100" GPUS=4 \
bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain_sweep.sbatch
```

### Step 8.3 — Phase A Evaluation: Fold-0 Stage-2 Zero-Shot Probes (Gates G2 & G3)

> **Compute placement**: submit these probes as a 1-GPU Slurm job (`acc_short`, `--gres=gpu:a100:1`).
> **Do not run them on a login node**: the shared communal T4 is contended by other tenants and
> terminal-attached processes are killed on session teardown with no traceback
> (`notes/spec/04_configs_and_execution.md` §4.2.1).

Once `<EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth` finishes (1 GPU):

```bash
CKPT="$WORK_SCRATCH/exp_20261127_adr017_tir_resp/fold0/stage2/checkpoints/checkpoint-0149.pth"
OUT_EVAL="../output/eval/exp_20261127_adr017_fold0"
mkdir -p "$OUT_EVAL"

# G2: matched masked_span inpainting + no-video np.interp / constant baselines on identical masks
for arm in "respmax 0.111 0.40" "mid 0.139 0.30"; do
  set -- $arm
  python -u runners/run_eval_recon.py --checkpoint "$CKPT" --split test \
    --kept_tir "$2" --kept_resp "$3" --resp_probe masked_span \
    --interp_baseline --pretext_metrics --repeats 5 --batch_size 16 \
    --preload --seed 0 --device cuda --label "adr017_fold0_${1}_masked_span" \
    --out "$OUT_EVAL/${1}_masked_span_fold0.json"
done

# G3: Mode-1 budgeted sensing (TIR-200), paired against ADR-008 mode1_fold0.json
python -u runners/run_eval_recon.py --checkpoint "$CKPT" --split test \
  --kept_tir 1.0 --kept_resp 0.0 --resp_probe hidden --pretext_metrics \
  --repeats 5 --batch_size 16 --preload --seed 0 --device cuda \
  --label adr017_fold0_mode1 --out "$OUT_EVAL/mode1_fold0.json"

# Remote-sensing deployment (TIR-900, budget-exempt) & rolled-TIR content control
python -u runners/run_eval_recon.py --checkpoint "$CKPT" --split test \
  --kept_tir 1.0 --resp_probe hidden --exempt_budget --pretext_metrics \
  --repeats 5 --batch_size 16 --preload --seed 0 --device cuda \
  --label adr017_fold0_remote_sensing --out "$OUT_EVAL/remote_sensing_fold0.json"

python -u runners/run_eval_recon.py --checkpoint "$CKPT" --split test \
  --kept_tir 1.0 --resp_probe hidden --exempt_budget --pretext_metrics \
  --ablate_stream tir --ablate_mode roll --repeats 5 --batch_size 16 \
  --preload --seed 0 --device cuda --label adr017_fold0_remote_sensing_tir_rolled \
  --out "$OUT_EVAL/remote_sensing_tir_rolled_fold0.json"

# Paired contrast vs. ADR-008 fold-0 Mode-1 baseline
python -m scripts.project.compare_recon_arms \
  --a ../output/eval/adr008_pretext/mode1_fold0.json \
  --b "$OUT_EVAL/mode1_fold0.json" \
  --metric r_masked --out "$OUT_EVAL/adr008_vs_adr017_mode1.json"
```

### Step 8.4 — Phase B: Five-Fold Stage-2 + Stage-3 Expansion (Gate G4, conditional on G2 & G3)

Only after fold 0 clears G2 and G3:

1. **Five-fold Stage 2** (sequential folds `0 1 2 3 4` via `submit_pretrain_sweep.sbatch`, or one `submit_pretrain_chain.sh` per fold for `FOLDS=1..4` on `acc_short`):
   ```bash
   EXP_ROOT="$WORK_SCRATCH/exp_20261127_adr017_tir_resp" \
   FOLDS="0 1 2 3 4" \
   CFG2="configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml" \
   DATA_SET2="tir_roi" \
   ROI_CACHE="$WORK_SCRATCH/tir_roi_cache" \
   RESP_CACHE="$WORK_SCRATCH/resp_cache" \
   GPU_TYPE="a100" GPUS=4 \
   bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain_sweep.sbatch
   ```
   *(Run the matched five-fold Stage 2 for `stage2_budget_tir_resp_2stream_anatomy_adr008.yaml` under its own `EXP_ROOT` with identical `split_seed=42`, `split_stratify=sex` so all five folds share identical subject partitions.)*
2. **Five-fold Stage 3 & pooling (Criterion 3c)**:
   - The repository's combined sweep wrapper `scripts/hpc/submit_sweep.sbatch` (`bash scripts/hpc/submit.sh scripts/hpc/submit_sweep.sbatch`) currently invokes `submit_sweep.sh` with `CFG3=configs/finetune/resp_tir_roi_hpc.yaml` (100 Hz / `temporal_stride: 2` / `sig_kernel: 16`) and without `--roi_cache` / `--resp_cache` flags. Before launching Stage 3 for Criterion 3c, pair a matched 1000 Hz / `resp_cache` Stage-3 config (identical for ADR-008 and ADR-017) with each fold's `<exp_root>/fold<k>/stage2/checkpoints/checkpoint-0149.pth` and `<exp_root>/fold<k>/partition.json`, score `--eval_role test` once per fold, and pool both arms via `python -u runners/run_evaluate_kfold.py --exp_root <EXP_ROOT>`.

---

## 9. Decision, Rollback, and Reporting

- **Scope decision (2026-11-28)**: the experiment stops at the fold-0 Stage-2 diagnostics. G0, G2 and G3 all pass (records above), but Phase B — five-fold Stage 2 and the matched five-fold Stage 3 required by G4 — is not pursued, so **no adoption decision is taken**: `masked_span` stays an experimental arm, `scattered` stays the arm default, and ADR-017 remains `Proposed`. Only the measured Stage-2 evidence is recorded (G0 and G2/G3 files under `code/notes/eval/baselines/tir_resp/`). Re-entering Phase B later requires no protocol change — the Stage-2 recipe, bounds and partition digest are unchanged and reproducible from this plan.
- **Accept**: Gates G0, G1, G2, G3, and G4 all pass (five-fold Stage-3 pooled subject-median Δ `band_limited_pearson >= +0.02` with 95 % bootstrap CI excluding `0` vs. the matched scattered ADR-008 arm), and all reproducibility artefacts are complete. Promote `masked_span` (with the G0-adopted bounds) to the arm default and update `ADR-017` status from `Proposed` to `Accepted`.
- **Retain for Analysis**: G0–G2 pass, G3 holds (`TIR-200` drop `<= 0.02`), and five-fold Stage-3 Δ `band_limited_pearson` falls in `(-0.02, +0.02)` (or CI includes `0`). Keep `scattered` as the default (simpler rule wins without evidence of benefit) and archive the fold-0 / five-fold analysis under `code/notes/eval/baselines/tir_resp/`.
- **Rollback**: Restore neutral placement defaults — `physio_placement: scattered`, `physio_gap_min_tokens: 0`, `physio_gap_max_tokens: 0` (i.e. revert to `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`). By ADR-017 R5, this is bit-identical to pre-ADR-017 behaviour.
- **Stop/Retire Rule**: Stop immediately and do not spend training GPU-hours if Gate G0 cannot satisfy `mse_interp > mse_const` at any feasible `(g_min, g_max)` pair; or retire ADR-017 (`Superseded`/`Deferred`, move ADR and feature plan to `archive/`) if five-fold Stage-3 Δ `band_limited_pearson <= -0.02`.
- **Reproducibility Artifacts**: Git commit SHA, exact config delta (`diff -u` vs. `adr008.yaml`), Gate-G0 floor JSON (`fold0_val_floor.json`), per-fold `partition.json` + digest, `masking.json`, `selection.json`, Stage-2 checkpoint SHA256, Slurm job IDs and runner logs, Stage-2 probe JSONs, paired contrast JSONs, and five-fold `kfold_*.json` for both arms.
- **Outcome Record**: Do **not** record measured results in this plan. Land verified outcomes under [`code/notes/eval/baselines/tir_resp/`](../../eval/baselines/tir_resp/README.md) (e.g. `adr017_floor_probe_fold0_val.md` for G0, `adr017_pretext_fold0.md` for G2/G3, and a five-fold Stage-3 comparison record for G4).
