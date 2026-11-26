
# EXP-[YYYYMMDD-SEQ]: [Descriptive Title]

- **Experiment ID**: `EXP-[YYYYMMDD-SEQ]`
- **Author**: LeqiaoP1
- **Status**: Draft
- **Associated ADR**: `code/notes/design/ADR-XXX-<slug>.md` or `None`
- **Target Thesis Section**: [Chapter/section and optional table reference]

---

## 1. Scientific Hypothesis and Baseline

- **Core Hypothesis**: [State the causal or mechanistic mechanism being tested.]
- **Comparison Baseline**: [Path to an existing record under `code/notes/eval/`, or `None` for a baseline-free smoke test.]
- **Data Lineage**: [Dataset class, cache/build lineage, and corpus artifact version.]
- **Expected Delta**: [Name exact repo metrics and proposed thresholds, or write `No numerical expectation` for an infrastructure-only run.]

---

## 2. Configuration and Parameter Delta

*Hierarchy: explicit CLI flags > YAML config > built-in defaults. List only values changed from the verified base config.*

- **Stage**: [Stage 2 pre-training | Stage 3 waveform fine-tuning]
- **Runner**: [e.g. `code/runners/run_pretrain.py`]
- **Base Config**: `code/configs/[pretrain|finetune]/<base_config>.yaml`
- **Output Artifacts Directory**: `output/experiments/EXP-[YYYYMMDD-SEQ]/`
- **Initial Checkpoint**:
  - Stage 2: [VideoMAE source, e.g. `videomae:k400`, or exact local `.pth`]
  - Stage 3: [Exact Stage-2 fold checkpoint and partition digest]
- **Explicit Parameter Overrides**:

```yaml
# Example: replace all placeholders. Delete fields not relevant to this stage.
data_set: tir_roi_resp
streams: tir                 # Closed set: rgb | tir | rgb2d | bp | resp
clip_duration: 8.0
fps: 25.0
temporal_stride: 2
tubelet: 2,16,16
input_size: 96
fs: 100
sig_kernel: 16
seq_len: 0
split_seed: 42
split_stratify: sex
val_select: fixed_epoch
lr: 1.0e-4
warmup_epochs: 5
epochs: 150
```

Stage-2 masking example:

```yaml
n_budget: 350
video_cap_ratio: 0.25
physio_cap_ratio: 0.50
video_min_tubes: 2
p_virtual_sensing: 0.50
physio_placement: scattered
physio_span_tokens: 0
signal_weight: 0.50
```

Stage-3 loss example:

```yaml
target: resp
streams: tir
alpha: 1.0
beta: 1.0
gamma: 1.0
huber_beta: 1.0
fft_sizes: '128,256,512'
```

Do not use `temporal_window`, `mask_ratio_visual`, `mask_ratio_signal`, or `loss_weight_waveform`; these are not active repo keys.

---

## 3. Data and Split Provenance

- **Dataset**: [Exact `data_set` value and dataset class.] 
- **Subjects/Sessions/Windows**: [Verified counts only; otherwise `Not yet measured`.]
- **Partition**: `<exp_root>/fold<k>/partition.json`
- **Split Policy**: `split_seed=42`, `split_stratify=sex`, subject-disjoint train/val/test roles.
- **Fold Scope**: [Fold IDs and whether test subjects are excluded from training.] 
- **Cache Dependencies**: [ROI cache, respiration cache, preprocessed artefact, and build command.] 
- **Selection Rule**: [fixed_epoch or best_val, including `val_criterion`.] 

---

## 4. Geometry and Modality Contract

- **Visual Stream**: [Canonical stream name and tensor shape.] 
- **Physiological Stream**: [Canonical stream name and 1-D shape.] 
- **ROI Contract**: [For ROI data: padding, crop rule, landmarks, sway gate.] 
- **Time Alignment**: [State $\text{sig\_kernel}/fs = \text{tubelet}_t \cdot \text{temporal\_stride}/fps$.]
- **Normalisation**: [Dataset-side `physio_norm` and model-side `target_norm`, if applicable.] 
- **Checkpoint Compatibility**: [Explain any local/HPC or 64 px/96 px incompatibility.] 

---

## 5. Evaluation Contract

- **Primary Tier 1 Metrics**: [Exact keys, direction, and threshold.] 
- **Tier 2 Spectral Metrics**: [Exact keys, direction, and threshold.] 
- **Tier 3 Physiological Metrics**: [Exact keys; state unavailable or not applicable.]
- **Aggregation**: [Per-clip, session, subject, and pooled k-fold aggregation.] 
- **Baseline**: [Verified baseline value and source record.] 
- **Numerical Gate**:

| Gate | Metric | Required Threshold | Measurement Scope |
|---|---|---:|---|
| G1 | [metric key] | [threshold + direction] | [held-out subjects/folds] |
| G2 | [metric key] | [threshold + direction] | [held-out subjects/folds] |
| G3 | [resource or physiological key] | [threshold + direction] | [run scope] |

No gate may claim `better` or `improved` without an exact metric and comparator.

---

## 6. Resource Envelope

- **Hardware**: [GPU type/count or `Not specified`.] 
- **Peak VRAM Cap**: [Number and unit, with measured source or `Provisional estimate`.]
- **Batch Size**: [Per-GPU value.] 
- **Workers**: [Per-rank value.] 
- **Expected Runtime**: [Measured or explicitly unestimated.] 
- **Abort Rule**: [OOM, NaN, invariant failure, or invalid geometry condition.] 

---

## 7. Local Smoke Verification

Run from `code/` with the active virtual environment:

```bash
python -u runners/run_<runner>.py \
  -c configs/<pretrain|finetune>/<config>.yaml \
  --exp_root /tmp/exp_<YYYYMMDD>_<slug> --fold 0 --dev_smoke \
  --epochs 1 --warmup_epochs 0 --num_workers 0 \
  --max_entries 2 --batch_size 1
```

Use `--max_clips 2` instead of `--max_entries 2` only when the selected runner supports it.

- **Success Condition**: [No crash, invariant failure, NaN, or split/checkpoint mismatch.] 
- **Recorded Artifacts**: [Log, selection record, smoke partition, and output directory.] 

---

## 8. HPC and Full Run

```bash
EXP_ROOT=<WORK_SCRATCH>/exp_<YYYYMMDD>_<slug> \
FOLDS="0 1 2 3 4" GPU_TYPE=<a100|h100|rtx6000> GPUS=4 \
bash scripts/hpc/submit.sh scripts/hpc/submit_sweep.sbatch
```

Use only an existing, relevant `.sbatch` wrapper. Explain any stage-specific pretrain checkpoint and fold resolution.

---

## 9. Decision, Rollback, and Reporting

- **Accept**: [All gates pass and artefacts are complete.] 
- **Retain for Analysis**: [Mixed thresholds or an infrastructure-only result.] 
- **Rollback**: [Exact config/CLI values to restore.] 
- **Stop/Retire Rule**: [Condition that prevents further compute.] 
- **Reproducibility Artifacts**: Git commit, config delta, partition digest, checkpoint digest, runner log, metric JSON, and warnings. 
- **Outcome Record**: [Do not populate measured outcomes here until verified; land them under `code/notes/eval/`.]
