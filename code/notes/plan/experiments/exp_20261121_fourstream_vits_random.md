# EXP-20261121-01: Four-stream Stage-2 pre-training (TIR + RGB2D + BP + RESP, ViT-Small, random init)

- **Experiment ID**: `EXP-20261121-01`
- **Author**: LeqiaoP1
- **Status**: **Ready (P0 cleared)**. The local fold-0 dev smoke passed on 2026-11-21; see `code/notes/eval/baselines/tir_rgb2d_bp_resp/stage2_smoke.md`. The first target-hardware run is the one-fold HPC smoke.
- **Associated ADR**: `code/notes/design/ADR-002-asymmetric-multimodal-masking.md` (masking); `code/notes/design/2DTexture_ROI_cache_preprocessing.md` (rgb2d contract)
- **Target Thesis Section**: Chapter 4, Stage-2 modality ablation / four-stream representation study

---

## 0. Grounding Summary

### 0.1 Files and configs inspected

| Area | Files |
|---|---|
| Terminology | `code/notes/GLOSSARY.md` |
| Template | `code/notes/plan/experiments/exp_template.md` |
| Base configs | `code/configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml`, `stage2_hpc_tir_roi_resp.yaml`, `stage2_local_pretrained.yaml` |
| Stage-2 runner | `code/runners/run_pretrain.py`, `code/runners/_common.py`, `code/engines/pretrain.py` |
| Model / masking | `code/core/multimae.py`, `code/core/budget_masking.py`, `code/core/mask_placement.py` |
| Datasets | `code/data/datasets.py`, `trimodal_dataset.py`, `tir_resp_dataset.py`, `paired_dataset.py` |
| Caches | `code/data/bp_artefact.py`, `bp_windows.py`, `bp_preprocess.py`, `resp_preprocess.py`; `runners/run_build_roi_cache.py`, `run_build_2d_roi_cache.py`, `run_build_resp_cache.py`, `run_build_bp_artefact.py` |
| HPC | `code/scripts/hpc/submit_sweep.sbatch`, `submit_sweep.sh`; pretrain-only `submit_pretrain_sweep.sbatch`, `submit_pretrain_sweep.sh` |
| Verified baseline | `code/notes/eval/rgb2d/resp_recon_zeroshot.md` |

### 0.2 Active implementation stage and runner

- **Stage**: Stage 2 pre-training only. No Stage 3 fine-tuning and no Stage-3 clinical metrics.
- **Runner**: `code/runners/run_pretrain.py`.
- **Model**: `project_multimae_small` (384-d / 12-layer / 6-head, `MULTIMAE_VARIANTS` in `core/multimae.py:123`). The code comment that “only base and large are supported” refers to the published VideoMAE Stage-1 weights; with the landed random-init path, the registered `small` entrypoint is valid.
- **Masking**: the live **ADR-002 constant-budget** policy (`core/budget_masking.py`), not the removed `masking: ammm` selector. See §0.4.

### 0.3 Falsifiable hypothesis, delta, baseline, decision

- **Falsifiable hypothesis**: At the ADR-002 geometry (8 s clip, `temporal_stride=2`, `tubelet=4,16,16`, `input_size=96`, `fs=1000`, `sig_kernel=32`), the constant-budget policy is feasible for **four** streams (two visuals + two 1 kHz physiologies), and a randomly initialised 384-d ViT-Small trains at least one epoch with `n_batch == 350` on every step, per-stream counts inside the caps/floors, both modes observed, and finite losses. The hypothesis is falsified by any budget, cap, mode-coverage, or finiteness violation.
- **Exact parameter delta**: add `bp` to the existing `tir,rgb2d,resp` arm; switch `model` from `project_multimae_base` to `project_multimae_small`; switch initialisation from `videomae:k400` to random; set `clip_stride: 4.0`; keep the ADR-002 geometry and seven masking keys.
- **Baseline**: `None` for the primary infrastructure gate. The verified `code/notes/eval/rgb2d/resp_recon_zeroshot.md` numbers are **not** a valid comparator for this run because they are ViT-Base + VideoMAE-K400, not ViT-Small + random. A matched 3-stream ViT-Small random-init control is defined in §5.4; it uses the existing `data_set: tir_roi_resp_rgb2d`.
- **Expected decision**: Accept the infrastructure gate if G1–G5 pass. The local fold-0 dev smoke passed all five on 2026-11-21 (`code/notes/eval/baselines/tir_rgb2d_bp_resp/stage2_smoke.md`). Do **not** claim a RESP-accuracy gain: the per-stream decoder has no cross-modal path when a stream is fully masked (`code/notes/eval/rgb2d/resp_recon_zeroshot.md`, “hidden” probe), so BP is not expected to improve RESP reconstruction.
- **Rollback**: restore `streams: tir,rgb2d,resp`, `data_set: tir_roi_resp_rgb2d`, `model: project_multimae_base`, `pretrained_encoder: videomae:k400`, `clip_stride: 2.0`.

### 0.4 P0 prerequisite gates (all cleared)

**State (2026-11-21).** P0-A, P0-B and P0-D are implemented; P0-C is the live ADR-002 policy; the four caches are complete and preflighted. P0-E remains optional and unimplemented. The table below is the original blocker audit; measured evidence is in `code/notes/eval/baselines/tir_rgb2d_bp_resp/cache_preflight.md`, `stage2_smoke.md` and `full_dataset_preflight.md`.

| Gate | Blocker | Evidence | What must land before any run |
|---|---|---|---|
| **P0-A** | No Stage-2 dataset emits `tir,rgb2d,bp,resp` on one window. | `data/trimodal_dataset.py:83-88` rejects any stream outside `(tir, rgb2d, resp)`; `TirRoiRespPretrainDataset` accepts only `(tir, resp)`; `run_pretrain.py` has no `--bp_artefact` / BP cache flag (grep finds only the raw-path help string). | Register the new dataset (proposed id `tir_roi_resp_rgb2d_bp`), attach the BP artefact via `bp_artefact.find_cache` + `bp_windows.usable/read_window`, z-score BP per clip, add it to `data/datasets.py`, `data/universe.py` and `pin_roi_target_norm`, and expose the BP-cache path in `run_pretrain.py`. |
| **P0-B** | No random-init Stage-1 path. | `run_pretrain.py:487-494` always calls `resolve_encoder_weights(..., require=True)` and `load_pretrained_encoder`; the `--pretrained_encoder` help explicitly says `random / scratch` is `REJECTED` (`run_pretrain.py:381-390`). | Add an explicit `pretrained_encoder: none` branch that skips `resolve_encoder_weights`/`load_pretrained_encoder` and logs `[stage1] random init`; add a test that `project_multimae_small` builds without a checkpoint. |
| **P0-C** | `masking: ammm` does not exist. | `GLOSSARY.md` §0 marks AMMM as the removed per-step Dirichlet policy; `tests/test_config_loading.py:188-197` asserts `masking`, `dirichlet_alpha`, `video_kept_cap`, `physio_kept_cap` are gone. | No code change: use the seven ADR-002 keys in §2. “Apply AMMM” is interpreted as the live ADR-002 constant-budget policy. |
| **P0-D** | No pretrain-only HPC sweep. | `scripts/hpc/submit_sweep.sh` always runs `run_waveform.py` (Stage 3) after Stage 2. No pretrain-only `.sbatch` exists in `scripts/hpc/`. | Add a pretrain-only wrapper (or a verified `SKIP_STAGE3` guard) before submitting five folds. Using `submit_sweep.sbatch` unmodified would violate “pre-training only”. |
| **P0-E** (optional, only for §5.4) | The zero-shot evaluator has no `--kept_bp` pin. | `runners/run_eval_recon.py:pin_masking` assigns `want[s] = 1.0` to any declared stream without a flag, then clamps BP to its 125-token cap, consuming visual budget. | Add an explicit BP pin (e.g. `--kept_bp 0.0`) so the 4-stream and 3-stream arms are evaluated with the same visual budget. |

**Cache verdict (2026-11-21).** The four artefacts are built, key-compatible and parity-checked: TIR-ROI 1385 cached / 0 missing, RGB2D px112 1360 / 0 missing, RESP 1000 Hz 1399 / 0 missing, BP 1400 / 0 missing. The BP 8 s / 4 s-hop usable yield is 98.14 % (13,278 / 13,530 windows). Measured evidence: `code/notes/eval/baselines/tir_rgb2d_bp_resp/cache_preflight.md`.

---

## 1. Scientific Hypothesis and Baseline

- **Core Hypothesis**: The ADR-002 constant-budget apportioner remains feasible with **two** physiological streams, and a randomly initialised ViT-Small can be pre-trained on the four-stream set without violating the constant 350-token budget, the per-stream caps, or numeric stability. No downstream accuracy gain is hypothesised: the per-stream decoder cannot route BP evidence into RESP when RESP is fully masked.
- **Comparison Baseline**: `None` for the infrastructure gate. The matched scientific control is the 3-stream ViT-Small random-init arm in §5.4 (not an existing record). The verified ViT-Base/VideoMAE tri-modal numbers in `code/notes/eval/rgb2d/resp_recon_zeroshot.md` are context only and are excluded from any gate.
- **Data Lineage**: Raw BP4D+ tree (`Thermal/`, `2D+3D/`, `Physiology/Resp_Volts.txt`, `Physiology/BP_mmHg.txt`). Visual streams come from the TIR-ROI and RGB2D caches; physiological streams come from the 1000 Hz RESP cache and the 1000 Hz BP artefact.
- **Expected Delta**: No numerical performance expectation. The only expected change is the presence of `kept_bp ∈ [0, 125]`, `mse_bp` finite, and `n_batch = 350` on every step.

---

## 2. Configuration and Parameter Delta

*Hierarchy: explicit CLI flags > YAML config > built-in defaults. Every key below exists in the current parser.*

- **Stage**: Stage 2 pre-training
- **Runner**: `code/runners/run_pretrain.py`
- **Base Config**: `code/configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml`
- **Config File**: `code/configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml`
- **Output Artifacts Directory**: `$EXP_ROOT/fold<k>/stage2/`
- **Initial Checkpoint**: none (explicit random init, `pretrained_encoder: none`).

The config below is the shipped file; every key is in the active parser.

```yaml
# --- data source -----------------------------------------------------------
data_set: tir_roi_resp_rgb2d_bp    # registered four-stream Stage-2 data_set
subjects: ''
split_seed: 42
split_stratify: sex
eval_freq: 5
val_select: fixed_epoch
val_criterion: joint_loss
tasks: ''
task_groups:
  low: [T2, T3, T5]
  moderate: [T1, T4]
  high: [T6, T7, T8, T9, T10]
task_set: ''

# --- model / initialisation ------------------------------------------------
model: project_multimae_small      # 384-d / 12 layers / 6 heads
pretrained_encoder: none           # explicit random init (landed)

# --- streams & tokenizer geometry ------------------------------------------
streams: tir,rgb2d,bp,resp         # two visuals + two 1 kHz physiologies
tir_channels: 3
tubelet: 4,16,16                   # 100 frames -> 25 temporal tokens per tube
fs: 1000.0
fps: 25.0
clip_duration: 8.0
clip_stride: 4.0                   # requested hop; NOT part of the Stage-3 ROI contract
temporal_stride: 2                 # 12.5 fps effective; tubelet = 320 ms
seq_len: 0                         # 0 => 8 s * 1000 Hz = 8000 samples
sig_kernel: 32                     # 32 ms per 1-D token; 250 tokens per stream
input_size: 96                     # 6x6 patch grid = 36 spatial tubes
pos_init: sincos3d
roi_padding: 0.10                  # frozen TIR + rgb2d cache key
crop_rule: square                  # frozen TIR + rgb2d cache key
max_sway_px: 40                    # TIR motion gate
roi_landmarks: ''                  # nose_mouth 12-point set
rgb2d_landmarks: face              # all 49 points; must match the cache build
rgb2d_store: 112                   # 112 px store; input_size 96 <= 112
rgb2d_cache_lru: 8
rgb2d_min_valid_ratio: 1.0
rgb2d_area_min_frac: 0.5
rgb2d_max_sway_px: 110
# The 1000 Hz RESP cache supersedes the legacy per-window gates; the dataset
# REFUSES non-zero values when the artefact is attached
# (tir_resp_dataset.py:_attach_resp_cache). The base tri-modal config still
# carries 0.1 / 9.900, which must be overridden to 0.0 for this run.
min_signal_spread: 0.0
rail_touch_v: 0.0

# --- ADR-002 constant-budget masking (the whole surface) --------------------
n_budget: 350
video_cap_ratio: 0.25              # <= 9 of 36 tubes = 225 tokens per visual
physio_cap_ratio: 0.50             # <= 125 of 250 tokens per physio
video_min_tubes: 2                 # >= 50 tokens per visual
p_virtual_sensing: 0.50
spatial_patches: 36                # tokenizer-derived; asserted at build
physio_tokens: 250                 # tokenizer-derived; asserted at build
physio_placement: scattered
physio_span_tokens: 0

# --- loss / normalisation ---------------------------------------------------
signal_weight: 0.5                 # applies to BOTH bp and resp; visuals stay 1.0
physio_norm: zscore                # dataset-side per-clip z-score, both physio streams

# --- training ---------------------------------------------------------------
epochs: 150
batch_size: 8                      # provisional per-GPU value; see §6
save_ckpt_freq: 10
opt: adamw
blr: 1.5e-4
min_lr: 0.0
warmup_epochs: 10                  # must be < epochs
weight_decay: 0.05
clip_grad: 5.0
num_workers: 8
log_wandb: false
```

**Rejected keys.** Do not use `temporal_window`, `mask_ratio_visual`, `mask_ratio_signal`, `loss_weight_waveform`, `masking`, `dirichlet_alpha`, `video_kept_cap`, `physio_kept_cap`, `min_kept`, `modality_dropout`, or `kept_ratio_override`. They are not in the active parser and are asserted gone by `tests/test_config_loading.py`.

---

## 3. Data and Split Provenance

- **Dataset**: `tir_roi_resp_rgb2d_bp` (proposed, P0-A). Dataset class: a four-stream extension of `TirRoiResp2DPretrainDataset` that preserves the parent’s `frame_start`/`frame_end` and adds the BP window on the same 1000 Hz grid.
- **Subjects/Sessions/Windows**: **Measured** on the full partition (2026-11-21, `code/notes/eval/baselines/tir_rgb2d_bp_resp/full_dataset_preflight.md`). `train` 6648 / 6386 / 6785 / 6307 / 6284 clips and `val` 1106 / 854 / 942 / 977 / 971 clips for folds 0–4; Stage 2 builds only `train`/`val`. The tri-modal fold-0 test set (3099 clips / 204 sessions / 27 subjects) is unchanged by this arm.
- **Partition**: `<exp_root>/fold<k>/partition.json`
- **Split Policy**: `split_seed=42`, `split_stratify=sex`, subject-disjoint train/val/test roles.
- **Fold Scope**: five folds, `fold 0..4`; Stage 2 builds only `train` and `val`, never `test`.
- **Cache Dependencies**:

| Stream | Cache | Exact key fields for this run | Build command (from `code/`) |
|---|---|---|---|
| `tir` | TIR-ROI pixel cache | `clip_seconds=8.0`, `fps=25.0`, `roi_padding=0.10`, `crop_rule=square`, landmarks nose_mouth | `python runners/run_build_roi_cache.py --init --out_root "$ROI_CACHE" --raw_root "$RAW_DATA_PATH" --clip_seconds 8.0 --fps 25.0 --roi_padding 0.10 --crop_rule square --roi_landmarks ""` then `--build --array_index "$IDX" --array_count "$CNT"` with the same flags |
| `rgb2d` | 2D-texture ROI cache | `clip_seconds=8.0`, `fps=25.0`, `roi_padding=0.10`, `crop_rule=square`, `rgb2d_landmarks=face`, `store_resolution=112` | `python runners/run_build_2d_roi_cache.py --init --out_root "$RGB2D_CACHE" --raw_root "$RAW_DATA_PATH" --clip_seconds 8.0 --fps 25.0 --roi_padding 0.10 --crop_rule square --roi_landmarks face --store_resolution 112` then `--build --array_index "$IDX" --array_count "$CNT"` |
| `resp` | RESP cache | `fs_in=1000`, `fs_out=1000`, default validity knobs (`min_run_s=8.0`) | `python runners/run_build_resp_cache.py --init --out_root "$RESP_CACHE" --raw_root "$RAW_DATA_PATH" --fs_in 1000 --fs_out 1000 --min_run_s 8.0` then `--build --array_index "$IDX" --array_count "$CNT"` |
| `bp` | BP artefact | `fs_in=1000`, `fs_out=1000`, `min_segment_s=4.0`, frozen `store` | `python runners/run_build_bp_artefact.py --init --out_root "$BP_CACHE" --raw_root "$RAW_DATA_PATH" --fs_in 1000 --fs_out 1000 --min_segment_s 4.0` then `--build --array_index "$IDX" --array_count "$CNT"` |

- **BP window rule**: `s = round(t_start * 1000)`, `n = round(8.0 * 1000) = 8000`; a clip survives only if `bp_windows.usable(mask, s, n)` is true (the whole 8 s lies in one valid segment). Yield: 98.14 % of the corpus’s 8 s windows, which costs **1.2–2.0 % of clips** per fold (0 sessions dropped, 0 missing shards) — `full_dataset_preflight.md` §1.
- **RESP-cache compatibility**: `TirRoiRespPretrainDataset._attach_resp_cache` raises when `min_signal_spread` or `rail_touch_v` is non-zero, because the cache stores the sole validity mask. The base tri-modal YAML still sets 0.1 / 9.900; this protocol overrides both to 0.0. This is a required override, not a tuning choice.
- **Selection Rule**: `fixed_epoch`, `val_criterion=joint_loss`; no early stopping.

---

## 4. Geometry and Modality Contract

- **Visual Streams**: `tir` `[B, 3, 100, 96, 96]` and `rgb2d` `[B, 3, 100, 96, 96]`.
- **Physiological Streams**: `bp` `[B, 1, 8000]` and `resp` `[B, 1, 8000]`.
- **ROI Contract**: `roi_padding=0.10`, `crop_rule=square`, `roi_landmarks=''` (nose_mouth), `rgb2d_landmarks=face`, `rgb2d_store=112`, `input_size=96`. Both visual streams are cropped from the same source frames and share one patch grid.
- **Time Alignment**:

  $$
  \text{frames} = \frac{8.0 \times 25.0}{2} = 100,\quad
  G_t = \frac{100}{4} = 25,\quad
  G_h G_w = 6 \times 6 = 36
  $$

  $$
  n_\text{signal} = \frac{8000}{32} = 250,\quad
  R = \frac{250}{25} = 10,\quad
  R \cdot \frac{32}{1000} = 0.32 = \frac{4 \times 2}{25}
  $$

  The hard invariant `R * sig_kernel/fs == tubelet_t * temporal_stride/fps` is satisfied at 0.32 s.
- **Budget Feasibility**:
  - Visual tokens per stream: `36 × 25 = 900`; physio tokens per stream: `250`.
  - Visual cap: `floor(0.25 × 36) = 9` tubes = 225 tokens; physio cap: `floor(0.50 × 250) = 125` tokens.
  - Mode 1: `350 // 25 = 14` tubes across two visuals.
  - Mode 2 with both physiologies at cap: `125 + 125 = 250` physio tokens; visual residual `350 − 250 = 100` = 4 tubes, exactly `2 × video_min_tubes`.
  - Feasibility condition `N_budget − P·C_p ≥ 2mT` is `350 − 2·125 = 100 ≥ 2·2·25 = 100`, i.e. equality. The configuration is feasible but **tight**; raising `physio_cap_ratio` above 0.50 makes it infeasible.
- **Normalisation**: the dataset z-scores each 8 s `resp` and `bp` clip (`physio_norm: zscore`); `run_pretrain.py` pins `target_norm='none'` for ROI lineages, so the model never re-normalises a 1-D stream. P0-A added the new `data_set` to `pin_roi_target_norm`, and the smoke confirmed the pin.
- **Checkpoint Compatibility**: the 96 px / 8 s / `tubelet=4,16,16` / `sig_kernel=32` geometry is a separate lineage from the 64 px configs. Stage 3 is out of scope; if later added, `run_waveform.py` will enforce the ROI contract and abort on a 64 px mismatch.

---

## 5. Evaluation Contract

### 5.1 Stage-2 monitor keys (exact, from `engines/pretrain.py`)

- `loss`, `mse_tir`, `mse_rgb2d`, `mse_bp`, `mse_resp`
- `kept_tir`, `kept_rgb2d`, `kept_bp`, `kept_resp`
- `mode` (`1` or `2`), `n_batch`
- Validation: `val_loss`, `components[mse_<stream>]`, `kept[kept_<stream>]`, `mode[mode1|mode2]`

### 5.2 Tier 1 / Tier 2 / Tier 3

- **Tier 1 temporal** (`pearson`, `band_limited_pearson`, `mae`, `rmse`, ...): **Not applicable**. This is pretraining only; no waveform is predicted.
- **Tier 2 spectral** (`psd_mae`, `dominant_freq_error_hz`, ...): **Not applicable**.
- **Tier 3 clinical** (`rmssd_ms`, ...): **Not applicable**; the repo does not estimate systolic/diastolic BP and 8 s clips are below the ≥30 s HRV requirement.
- **Stage-2 loss is a monitor, not a metric**: `mse_resp` plateaus at ~1.0 (the documented constant-predictor collapse). Do not gate on `mse_resp` or `mse_bp`.

### 5.3 Numerical gates (infrastructure)

| Gate | Metric | Required Threshold | Measurement Scope |
|---|---|---:|---|
| G1 | `n_batch` | exactly `350` on 100 % of training and validation steps | smoke + full run |
| G2 | `kept_tir`, `kept_rgb2d`, `kept_bp`, `kept_resp` | sum exactly `350` every step; `kept_tir`/`kept_rgb2d` multiples of 25 in `[50, 225]`; `kept_bp`/`kept_resp` in `[0, 125]` | per-step log |
| G3 | `mode` | both `mode1` and `mode2` observed; `mode1` share in `[0.35, 0.65]` over the first 100 steps (proposed threshold) | first 100 steps |
| G4 | `loss`, `mse_*`, `val_loss` | finite (no `NaN`/`Inf`) | smoke + first epoch |
| G5 | peak VRAM | `<= 90 %` of device memory (provisional; see §6) | smoke |

G1–G4 are direct consequences of ADR-002 contracts and are executable from the logger. G3’s `[0.35, 0.65]` band is an explicitly **proposed** threshold, not a repo constant.

- **Aggregation**: per-step values from `MetricLogger`; per-epoch validation means from `evaluate_pretrain`; folds are reported separately. No per-clip, session, subject, or pooled accuracy aggregation is produced by a pretraining-only run.
- **Baseline**: `None` for G1–G5. The optional matched control in §5.4 is the only admissible accuracy comparator, and it is not an existing record.

### 5.4 Optional matched scientific control (only after P0-E)

If P0-E lands, run the existing 3-stream `tir_roi_resp_rgb2d` arm with the **same** `project_multimae_small` random init and the same geometry, then score both frozen checkpoints with `runners/run_eval_recon.py` using `--resp_probe span --kept_resp 0.5 --repeats 5 --seed 0`. For the 4-stream arm, P0-E must add `--kept_bp 0.0` so BP does not consume visual budget; with `--kept_tir 0.10 --kept_rgb2d 0.50`, both arms then keep the same 225 visual tokens. The only admissible gate is:

| Gate | Metric | Required Threshold | Comparator |
|---|---|---:|---|
| G6 | `r_masked` (span50) | 4-stream arm `>=` 3-stream ViT-Small control `− 0.02` (proposed non-inferiority margin) | matched 3-stream ViT-Small random-init arm, same fold, same clips, same masks |

The ViT-Base numbers in `code/notes/eval/rgb2d/resp_recon_zeroshot.md` (`+0.5419` tri span50, `+0.9975` random50, `+0.0055` hidden) are **context only** and must not be used as G6’s comparator. They differ in encoder width (768 vs 384) and initialisation (VideoMAE K400 vs random).

---

## 6. Resource Envelope

- **Hardware**: HPC, 4× RTX 6000 or 4× A100 (Slurm). Local smoke on 1 GPU.
- **Peak VRAM Cap**: ≤ 24 GB per GPU at `batch_size=8`, 96 px, four streams, ViT-Small. **Measured**: 2.897 GB allocated / 3.271 GB reserved for one batch-8 step on a 15 GB T4 (the batch-1 smoke was 0.484 / 0.570 GB) — `code/notes/eval/baselines/tir_rgb2d_bp_resp/full_dataset_preflight.md` §2.
- **Batch Size**: 8 per GPU (confirmed safe at batch 8 single-process; ViT-Small is smaller than the ViT-Base config’s 16).
- **Workers**: 8 per rank.
- **Expected Runtime**: 831 steps/epoch on fold 0, 1.48 s/it steady state with 8 workers on a T4 ⇒ ≈ 20.5 min/epoch, ≈ 51 h for 150 epochs **on a T4**. The target-GPU figure is unmeasured; do not state a wall-clock claim for HPC until the one-fold run is timed.
- **Checkpoint Volume**: `save_ckpt_freq=10` ⇒ 15 checkpoints × 366 MB ≈ 5.5 GB/fold, ≈ 27 GB over five folds; verify the `EXP_ROOT` quota first.
- **Abort Rule**: OOM; any `NaN`/`Inf` in `loss`/`mse_*`; `n_batch != 350`; any cap/floor violation; partition digest mismatch; checkpoint geometry mismatch; `bp_windows.usable` false for every candidate clip.

---

## 7. Local Smoke Verification

Executed 2026-11-21: **PASS** on the real caches. The command below is the executed smoke.

```bash
EXP_ROOT=$WORK_SCRATCH/fourstream_smoke_exp_20261121_01
python -u runners/run_pretrain.py \
  -c configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml \
  --exp_root "$EXP_ROOT" --fold 0 --dev_smoke \
  --data_set tir_roi_resp_rgb2d_bp --raw_root "$RAW_DATA_PATH" \
  --epochs 1 --warmup_epochs 0 --num_workers 0 \
  --max_clips 1 --max_entries 2 --batch_size 1 \
  --roi_cache "$ROI_CACHE" \
  --rgb2d_cache "$RGB2D_CACHE" \
  --resp_cache "$RESP_CACHE" \
  --bp_artefact "$BP_ARTEFACT" \
  --output_dir "$EXP_ROOT/fold0/stage2"
```

Exact paths reported by the run:

- Config: `configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml`
- Output: `$WORK_SCRATCH/fourstream_smoke_exp_20261121_01/fold0/stage2/`
- Partition: `$WORK_SCRATCH/fourstream_smoke_exp_20261121_01/fold0/partition.dev.json`
- Result: PASS; metrics, checkpoint and peak VRAM are recorded in `code/notes/eval/baselines/tir_rgb2d_bp_resp/stage2_smoke.md`.

Matched 3-stream control (runnable after P0-B, using the existing dataset):

```bash
python -u runners/run_pretrain.py \
  -c configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml \
  --model project_multimae_small --pretrained_encoder none \
  --exp_root /tmp/exp_20261121_fourstream_vits_control \
  --fold 0 --dev_smoke \
  --epochs 1 --warmup_epochs 0 --num_workers 0 \
  --max_entries 2 --batch_size 1 \
  --roi_cache "$ROI_CACHE" \
  --resp_cache "$RESP_CACHE" \
  --rgb2d_cache "$RGB2D_CACHE" \
  --min_signal_spread 0 --rail_touch_v 0   # required with the RESP cache
```

- **Success Condition**: no crash; `n_batch == 350`; the four `kept_*` keys present; both modes observed if `p_virtual_sensing=0.5` over enough steps; no `NaN`/`Inf`; the dataset `stats` show a non-zero number of clips after the BP `usable` gate.
- **Recorded Artifacts**: runner log, `partition.dev.json`, checkpoint, `selection.json` (if written), dataset `stats` (including BP drop counters), and the exact cache keys printed by `find_cache`; measured evidence in `code/notes/eval/baselines/tir_rgb2d_bp_resp/stage2_smoke.md`.

---

## 8. HPC and Full Run

**P0-D cleared (2026-11-21)**: `scripts/hpc/submit_pretrain_sweep.sbatch` + `.sh` run partition + Stage 2 only. They never call `run_waveform.py`.

**Fold resolution**: there is no inherited Stage-2 checkpoint; every fold starts from the same random initialisation (`RNG seed 0`) and its own `<exp_root>/fold<k>/partition.json` (`split_seed=42`). The fold index only selects the partition; it must not seed the model differently.

The wrapper executes, per fold:

```bash
cd code
python -u runners/run_pretrain.py \
  -c configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml \
  --exp_root "$EXP_ROOT" --fold "$k" \
  --data_set tir_roi_resp_rgb2d_bp \
  --raw_root "$RAW_ROOT" \
  --roi_cache "$ROI_CACHE" \
  --rgb2d_cache "$RGB2D_CACHE" \
  --resp_cache "$RESP_CACHE" \
  --bp_artefact "$BP_ARTEFACT" \
  --output_dir "$EXP_ROOT/fold$k/stage2"
```

`--bp_artefact` is a live runner flag. Submit the first target-hardware test as ONE fold:

```bash
EXP_ROOT=<WORK_SCRATCH>/fourstream_fold0 \
FOLDS="0" GPU_TYPE=rtx6000 GPUS=1 \
ROI_CACHE=... RGB2D_CACHE=... RESP_CACHE=... BP_ARTEFACT=... \
bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain_sweep.sbatch
```

After that passes, run the five folds:

```bash
EXP_ROOT=<WORK_SCRATCH>/fourstream_v1 \
FOLDS="0 1 2 3 4" GPU_TYPE=rtx6000 GPUS=4 \
ROI_CACHE=... RGB2D_CACHE=... RESP_CACHE=... BP_ARTEFACT=... \
bash scripts/hpc/submit.sh scripts/hpc/submit_pretrain_sweep.sbatch
```

---

## 9. Decision, Rollback, and Reporting

- **Accept**: G1–G5 pass on the local smoke (already true: `code/notes/eval/baselines/tir_rgb2d_bp_resp/stage2_smoke.md`) and on the one-fold target-hardware smoke; the four-stream dataset reports non-zero clips after the BP gate; all checkpoints are written with the expected geometry.
- **Retain for Analysis**: G1–G5 pass but G6 (optional) is mixed; report the run as an infrastructure/representation result, not an accuracy result.
- **Rollback**: restore `streams: tir,rgb2d,resp`, `data_set: tir_roi_resp_rgb2d`, `model: project_multimae_base`, `pretrained_encoder: videomae:k400`, `clip_stride: 2.0`, and drop the BP cache flag. Do not delete the BP artefact until no other run references it.
- **Stop/Retire Rule**: do not submit the five-fold run until the one-fold target-hardware smoke passes. If the 8 s BP `usable` gate leaves zero clips on the corpus, retire the 8 s BP arm rather than silently shortening the clip or relaxing `bp_windows.usable`.
- **Reproducibility Artifacts**: git commit, config delta, all four cache keys, `partition.json` digest, checkpoint digest, runner log, metric JSON, and the exact `stats` counters.
- **Outcome Record**: measured outcomes are already landed under `code/notes/eval/baselines/tir_rgb2d_bp_resp/` (`cache_preflight.md`, `stage2_smoke.md`, `full_dataset_preflight.md`). Keep every future run there; do not edit this plan with measured results.

---

## 10. P0 Lift Plan (task DAG)

**Status (2026-11-21).** Tasks 1–16 are green. The local fold-0 dev smoke passed. The next steps are the one-fold target-hardware smoke, then the five-fold sweep. Measured evidence: `code/notes/eval/baselines/tir_rgb2d_bp_resp/cache_preflight.md` and `stage2_smoke.md`.

This is the implementation order that removes the blockers in §0.4. The todo list mirrors these IDs. Do not start a later phase before its dependencies are green.

| # | Task | Depends on | Primary files | Done when |
|---:|---|---|---|---|
| 1 | Add four-stream dataset class | — | new `code/data/fourstream_dataset.py` | A subclass/extension of `TirRoiResp2DPretrainDataset` returns `{tir, rgb2d, resp, bp}` on one `frame_start/frame_end`; BP uses `bp_windows.usable/read_window`; BP is per-clip z-scored; unusable BP windows are dropped with counters. |
| 2 | Register four-stream data_set | 1 | `code/data/datasets.py`, `code/data/universe.py` | The new `data_set` is in `available_data_sets()`; `build_pretraining_dataset` dispatches to the new builder; `corpus_subjects` enumerates the TIR family for it. |
| 3 | Add BP cache runner flag | 2 | `code/runners/run_pretrain.py` | `--bp_artefact` is accepted, passed to the builder, and required; a missing path or a cache-key mismatch aborts loudly. |
| 4 | Pin target_norm for new data_set | 2 | `code/runners/run_pretrain.py` | `pin_roi_target_norm` includes the new data_set and forces `target_norm='none'`. |
| 5 | Test four-stream dataset contract | 1–4 | new `code/tests/test_fourstream_dataset.py`, `code/tests/test_universe.py` | Tests cover keys/shapes, BP/TIR/RESP time alignment, z-score, unusable-window drop, and cache identity failure. |
| 6 | Regenerate data-pipeline spec | 5 | `code/notes/spec/01_data_pipeline.md`, `code/notes/GLOSSARY.md` | `python code/scripts/project/write_specs.py` succeeds; the new data_set and four-stream contract are documented. |
| 7 | Add random-init Stage-2 path | — | `code/runners/run_pretrain.py`, `code/models/pretrained.py` | `--pretrained_encoder none` calls `resolve_encoder_weights(..., require=False)`, skips the loader on an empty path, logs `[stage1] random init`, and leaves the model’s `trunc_normal_` init intact. |
| 8 | Test ViT-Small random init | 7 | new `code/tests/test_pretrain_random_init.py` or `test_pretrained_specs.py` | `project_multimae_small` builds with `none`; no checkpoint is resolved; an invalid non-empty spec still fails. |
| 9 | Create four-stream pretrain YAML | 5, 7 | `code/configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml` | YAML loads with no unknown keys; contains the exact §2 values, including `min_signal_spread: 0.0`, `rail_touch_v: 0.0`, `pretrained_encoder: none`. |
| 10 | Add four-stream config test | 9 | `code/tests/test_config_loading.py` | The shipped-config test covers the new YAML and rejects `masking` / Dirichlet / `mask_ratio_*` residue. |
| 11 | Add pretrain-only HPC wrapper | 3, 9 | new `code/scripts/hpc/submit_pretrain_sweep.sbatch` + `.sh` | Wrapper runs partition + `run_pretrain.py` only; no `run_waveform.py` call. |
| 12 | Verify no Stage 3 launch | 11 | HPC wrapper + logs | `grep -n run_waveform` is empty; a one-fold dry run logs only `stage2/`. |
| 13 | Build and preflight four caches | 3 | TIR-ROI, RGB2D, RESP-1000, BP-1000 artefacts | Each `find_cache` returns a root whose `params.json` matches the §3 key; each runner’s `--check` passes; BP `usable` yield is non-zero on the sampled sessions. |
| 14 | Run local one-step smoke | 9, 13 | `/tmp/exp_20261121_fourstream_vits/fold0/stage2/` | `--epochs 1 --warmup_epochs 0 --num_workers 0 --max_entries 2 --batch_size 1` finishes; G1–G5 pass. |
| 15 | Run fold-0 dev smoke | 14 | `$EXP_ROOT/fold0/stage2/` | One dev epoch logs the four `kept_*` keys, both modes, finite losses, and the BP drop counters. |
| 16 | Update protocol status and gates | 15 | this file | Status changes from P0-blocked to runnable; provisional VRAM/params/runtime and “Not yet measured” counts are replaced with smoke values. |
| 17 | Add optional BP eval pin | 7, 9 | `code/runners/run_eval_recon.py` | `--kept_bp` is exposed and `pin_masking` handles every declared physio stream explicitly. |
| 18 | Test matched recon budget | 17 | `code/tests/` | 4-stream with `--kept_bp 0.0` and 3-stream arms produce the same 225 visual tokens for `--kept_tir 0.10 --kept_rgb2d 0.50 --kept_resp 0.5`. |

**Critical path**: 1 → 2 → 3 → 13 → 14 → 15 → 16, with 7/9 feeding 14 and 11/12 required only for the full HPC run. Tasks 1–16 are green.

**Stop rule**: do not start the five-fold run until the one-fold target-hardware smoke passes; tasks 17–18 remain optional and are not required for the primary run.

---

## Appendix A — Why the requested “AMMM” is the constant-budget policy

ADR-002 is titled *Asymmetric Multi-Modal Masking*; its implemented policy is the constant-budget apportioner in `core/budget_masking.py` + `core/mask_placement.py`. The historical `masking: ammm` selector, `code/core/ammm_masking.py`, its Dirichlet knobs and its tests were deleted when ADR-002 landed (`GLOSSARY.md` §0; `tests/test_config_loading.py`). Therefore this protocol applies AMMM by setting the seven live keys in §2, not by adding a `masking:` key.

## Appendix B — Why the four-stream set is not the existing four-stream `bp4d+` arm

`PairedPretrainDataset` does support `streams: rgb,tir,bp,resp`, but its visual streams are `rgb` (`_PRETRAIN_VISUAL_STREAMS = ('rgb', 'tir')`), not `rgb2d`; it cannot serve the requested 96 px 2D-texture ROI. Conversely, `TirRoiResp2DPretrainDataset` serves `tir + rgb2d + resp` at 96 px but rejects `bp`. The requested intersection does not exist; P0-A is the only path.
