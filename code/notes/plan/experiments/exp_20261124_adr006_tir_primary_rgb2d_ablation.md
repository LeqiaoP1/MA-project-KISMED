# EXP-20261124-01: ADR-006 TIR-Primary Masking with a Matched RGB2D Contribution Ablation

- **Experiment ID**: `EXP-20261124-01`
- **Author**: LeqiaoP1
- **Status**: **Withdrawn** — this protocol belongs to the *alternative* design (`code/notes/design/alt-adr-006-visual-only-resp-conditioning.md`), not to the ADR of record. Superseded by a run protocol for `ADR-006-three-regime-virtual-sensing-masking.md`, which is not yet written.
- **Associated ADR**: `code/notes/design/ADR-006-visual-only-resp-conditioning-and-primary-stream-saturation.md`
- **Target Thesis Section**: Chapter 4 (experiments) — the pre-training ablation table; Chapter 3 (methodology) — the masking-policy and conditioning specification.

---

## 1. Scientific Hypothesis and Baseline

- **Core Hypothesis**: The ADR-005 arm's sensor-free respiration pathway is weak for two mechanical reasons, not for lack of capacity: (a) the signal decoder's prefix contains the stream's own visible tokens, so the respiration-visible step is solvable intra-modally and exerts almost no pressure on the visual latents (H1); and (b) the sensor-absent step spends 125–175 of its 350 tokens on a visual stream whose content the measurement shows is unused (H2). Removing (a) and (b) should raise the sensor-free pathway; and once (b) is in place, the RGB2D contribution question becomes answerable by a matched ablation (H3).
- **Comparison Baseline**: `code/notes/eval/baselines/tir_rgb2d_resp/` — the ADR-005 fold-0 checkpoint `checkpoint-0149.pth`, Mode-1 subject-median $r_{\text{masked}}$ = **+0.1129** [+0.0775, +0.1363] (Arm A, `tir 225 / rgb2d 125 / resp 0`), 1,594 clips / 201 sessions / 27 subjects on the fold-0 `test` role. The same record gives the inpainting reference +0.9959 and the RGB2D ablations (roll −0.0012, t = −0.47; TIR roll −0.0539, t = −9.96).
- **Data Lineage**: `data_set: tir_roi_resp_rgb2d` (TIR-ROI + 2D-texture ROI + `Resp_Volts.txt`); the TIR ROI cache, the 112-px 2D-texture cache and the 1000 Hz respiration cache are unchanged and stay valid — nothing in ADR-006 touches a cache key (clip length, fps and fs are untouched; `temporal_stride` is not a key).
- **Expected Delta**: Mode-1 subject-median $r_{\text{masked}}$ ≥ **+0.1329** (i.e. +0.02 over the baseline) on the fold-0 `test` role, per arm, reported with the paired subject-level CI. `3s-ctx` − `3s` and `2s` − `3s` are reported as the D4 answer. No headline (band-limited Pearson) expectation is stated: that gate is defined relative to a matched Stage-3 fine-tune of the ADR-005 checkpoint that does not exist yet.

---

## 2. Configuration and Parameter Delta

*Hierarchy: explicit CLI flags > YAML config > built-in defaults. Only deltas from the verified base are listed.*

- **Stage**: Stage 2 pre-training (three arms, one fold)
- **Runner**: `code/runners/run_pretrain.py`
- **Base Config**: `code/configs/pretrain/stage2_budget_3stream_curriculum.yaml` (the ADR-005 arm, p = 0.10 → 0.50 at $E_{\text{warm}}$ = 50, fold-0 checkpoint lineage)
- **Output Artifacts Directory**: `output/experiments/EXP-20261124-01/<arm>/`
- **Initial Checkpoint**: `videomae:k400` (pre-fetched on a login node; `patch_stream: tir` unchanged, so the RGB2D adapter stays random-init and this is recorded as ADR-005 I2 debt)
- **Base parameters held fixed across all arms** (ADR-005 §2.9, omitted from the deltas below): `n_budget 350`, `tokens_per_unit 25`, `physio_cap_ratio 0.50`, `video_min_tubes 2`, `p_virtual_sensing 0.50`, `virtual_sensing_warmup_p 0.10`, `virtual_sensing_warmup_epochs 50`, `physio_placement scattered`, `signal_weight 0.5`, `tubelet 2,16,16`, `input_size 96`, `temporal_stride 4`, `fs 1000`, `sig_kernel 32`, `clip_duration 8.0`, `clip_stride 4.0`, `epochs 150`, `blr 1.5e-4`, `batch_size 16`, `split_seed 42`, `split_stratify sex`, `val_select fixed_epoch`, `patch_stream tir`

**Common delta (all three arms)**

```yaml
mode1_primary_stream: tir   # D1: Mode-1 -> tir 350 / aux 0 / resp 0
signal_cond_scope: visual   # D2: signal decoders prefix visual latents only
```

**Arm deltas**

| Arm | Config file | Further delta | Purpose |
| :--- | :--- | :--- | :--- |
| `3s` | `configs/pretrain/stage2_budget_3stream_tir_primary.yaml` | none | arm of record (Target A, bimodal fusion) |
| `3s-ctx` | `configs/pretrain/stage2_budget_3stream_tir_primary_ctx.yaml` | `loss_weights: 1.0,0.0,0.5` | RGB2D as encoder context only; isolates content |
| `2s` | `configs/pretrain/stage2_budget_tir_only_2s.yaml` | `streams: tir,resp`, `video_cap_ratio: 0.389` | RGB2D removed, its tokens given to TIR |

Do not use `temporal_window`, `mask_ratio_visual`, `mask_ratio_signal` or `loss_weight_waveform`; these are not active repo keys.

---

## 3. Data and Split Provenance

- **Dataset**: `tir_roi_resp_rgb2d` — `data/fourstream_dataset.py` via `runners/_common.py`
- **Subjects/Sessions/Windows**: fold 0, `test` role = **1,594 clips / 201 sessions / 27 subjects** (measured; `F027` has no usable clip under the corpus gates). Train/val counts: **Not yet measured** in this record — read them from the run's partition record, do not estimate.
- **Partition**: `<exp_root>/fold0/partition.json`
- **Split Policy**: `split_seed=42`, `split_stratify=sex`, subject-disjoint train/val/test roles
- **Fold Scope**: fold 0 only. `test` is never loaded during pre-training; it is scored by the probe of §5.
- **Cache Dependencies**: TIR ROI cache, 112-px 2D-texture cache, Phase-3 respiration cache at `fs = 1000`. All are cache-key-compatible with ADR-006 (no geometry change).
- **Selection Rule**: `fixed_epoch` with `val_criterion: joint_loss`. The reported arm state is the final epoch; the ADR-006 arm is not selected on any metric — in particular not on the probe of §5.

---

## 4. Geometry and Modality Contract

- **Visual Stream**: `tir` `[B, 3, 50, 96, 96]` (primary, VideoMAE Conv3d kernel) and, in `3s`/`3s-ctx`, `rgb2d` `[B, 3, 50, 96, 96]` (auxiliary, random-init Conv3d)
- **Physiological Stream**: `resp` `[B, 8000]` → 250 tokens at `sig_kernel 32`
- **ROI Contract**: `roi_padding 0.10`, `crop_rule: square`, TIR 12-point nose+mouth landmarks, `max_sway_px 40`; `rgb2d_landmarks: face` (49 points), `rgb2d_store 112`, `rgb2d_min_valid_ratio 1.0`, `rgb2d_area_min_frac 0.5`
- **Time Alignment**: $\text{sig\_kernel}/fs = 32/1000 = 32\ \text{ms}$; $\text{tubelet}_t \cdot \text{temporal\_stride}/fps = 2 \cdot 4/25 = 320\ \text{ms}$; $R = 10$ signal tokens per visual token, exact
- **Normalisation**: dataset-side `physio_norm: zscore`; model-side `target_norm: none` (pinned for every ROI lineage)
- **Checkpoint Compatibility**: 96 px, `tubelet (2,16,16)` — identical to the ADR-005 arm, so `videomae:k400` loads bit-for-bit into the TIR adapter. `2s` drops one input stream but not the encoder geometry, so the same checkpoint applies to all three arms.

---

## 5. Evaluation Contract

Two evaluation layers. **Layer A** is the zero-shot probe and is the only layer any gate in this record reads. **Layer B** is the protocol headline and is registered here so its rule cannot be chosen later.

**Layer A — probe** (`runners/run_eval_recon.py`, pinned plan, `--resp_probe hidden`, `--repeats 5`, `--seed 0`, `--batch_size 16`, fold-0 `test` role)

- `r_masked` (Pearson on masked positions, dataset per-clip z-score), aggregated clip → median per subject → median with bootstrap 95 % CI over subjects (2,000 resamples). The ADR-005 record's own aggregation, unchanged, so the comparison is like-for-like.
- Every arm is probed with the **same** Mode-1 plan (`tir 350`, respiration absent, auxiliary 0 where declared), so the probe input is identical across arms.

**Layer B — headline** (`runners/run_evaluate_kfold.py` after Stage 3): band-limited Pearson (0.1–0.6 Hz), per session → median per subject → median with bootstrap 95 % CI over subjects, subject-disjoint test roles, against a matched Stage-3 fine-tune of the ADR-005 checkpoint under the identical recipe. Baseline value: **does not exist yet**; it must be produced before this gate can be read, and it may not be substituted by a Layer-A number.

**Numerical gates**

| Gate | Metric | Required Threshold | Measurement Scope |
|---|---|---:|---|
| G1 (P1) | Layer-A `r_masked` subject median, ADR-005 checkpoint, Mode-2 plan, `signal_cond_scope: visual` | must **drop by ≥ 0.50** vs +0.9959, else D2 is withdrawn | 3-subject positive-control set (F001, F003, F005; 210 clips) |
| G2 (P2) | Layer-A `r_masked` subject median, ADR-005 checkpoint, saturated Mode-1 plan | **≥ +0.1329** (Arm A +0.02), else D1 is withdrawn | all 27 held-out subjects |
| G3 | Layer-A `r_masked` subject median, each arm's own checkpoint | **≥ +0.1329** (baseline +0.02), reported with the paired CI | all 27 held-out subjects |
| G4 | `3s-ctx` − `3s` and `2s` − `3s` paired subject-level CI | decision rule of ADR-006 §3.3, not a pass/fail threshold | all 27 held-out subjects |
| G5 | Peak allocated / reserved VRAM at production batch; no `NaN`; conformance of ADR-006 §3.2 | record, **no NaN**, encoder input `[B, 350, D]` on 100 % of steps | smoke + first production step |

No gate may claim `better` or `improved` without an exact metric and comparator. Layer-A numbers are pretext diagnostics and must never be presented as a waveform-recovery result.

---

## 6. Resource Envelope

- **Hardware**: Slurm `acc` partition, `gpu:a100:4`, `ntasks=4` (DDP world size 4), 16 cpus/task
- **Peak VRAM Cap**: Provisional estimate — `3s`/`3s-ctx` should match the ADR-005 arm (same geometry, two video adapters at 96 px); `2s` should be lower (one adapter). Record it from G5 rather than assuming; the ADR-005 figure does not transfer without measurement.
- **Batch Size**: 16 per GPU
- **Workers**: `num_workers 8`
- **Expected Runtime**: Not estimated for the 150-epoch arms. Measure s/step and steps/epoch on `acc_short` first (the LR-schedule line prints the exact step count within minutes) and size from that.
- **Abort Rule**: OOM; `NaN`/`Inf` loss; any invariant assertion in `budget_masking`/`multimae`; an encoder input that is not `[B, 350, D]`; a Mode-1 step that is not `tir 350` (aux 0, resp 0); a masking record that disagrees with the resolved tuple on resume.

---

## 7. Local Smoke Verification

Run from `code/` with the active virtual environment, per arm:

```bash
source scripts/env_local.sh
python -u runners/run_pretrain.py \
    -c configs/pretrain/stage2_budget_3stream_tir_primary.yaml \
    --exp_root /tmp/exp20261124_3s --fold 0 --dev_smoke \
    --epochs 1 --warmup_epochs 0 --virtual_sensing_warmup_epochs 0 \
    --max_clips 100 --batch_size 8 --num_workers 0 \
    --output_dir /tmp/exp20261124_3s/out
```

Pass = no assertion failure, no `NaN`, `[B, 350, D]`, a saturated Mode-1 plan and a conforming Mode-2 plan in the masking record, and (for `2s`) acceptance of the single-visual set under ADR-006 feasibility rule 6.

## 8. Run Sequence and Decision Tree

1. Implement ADR-006 D1/D2/D5 + tests; smoke all three arms (§7).
2. **P1 and P2** on the ADR-005 checkpoint (§5, G1/G2). Both are ~2 GPU-hours total.
3. If **both** gates pass → submit `3s`, `3s-ctx`, `2s` (one fold each; `3s` is the arm of record and must finish first if GPU supply is limited).
4. If G1 fails → D2 withdrawn; re-issue the ADR with D1 only, and re-run step 2 for P2.
5. If G2 fails → D1 withdrawn; the ADR's Mode-1 allocation reverts to ADR-005's split and the arm is re-scoped.
6. If an arm passes G3 → it becomes a candidate for Stage 3; the headline gate (Layer B) then requires the matched ADR-005 Stage-3 fine-tune, which must be launched **before** the candidate arm is fine-tuned, with the identical recipe.
