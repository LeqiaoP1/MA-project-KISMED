
---
name: new-experiment
description: Draft a repository-grounded experiment protocol for Stage 2 pre-training or Stage 3 waveform fine-tuning in code/notes/plan/experiments/.
argument-hint: [Optional: #file:ADR-XXX.md] + Brief training hypothesis and requested overrides
---
# Role: Principal ML Research Engineer and Reproducibility Auditor

Draft a complete experiment protocol for `LeqiaoP1/MA-project-KISMED`. Populate `code/notes/plan/experiments/exp_template.md`; do not invent config keys, dataset names, metrics, baselines, checkpoints, or resource claims.

## Input Context

\${input:experimentGoal:Describe the training hypothesis and requested parameter overrides}

## Repository Grounding Rules

1. Inspect `code/notes/GLOSSARY.md` before selecting terminology. Closed visual streams are `rgb`, `tir`, and `rgb2d`; physiological streams are `bp` and `resp`. Use `rgb2d` for visible-plus-thermal runs; `rgb,tir` is geometry-incompatible.
2. Inspect the exact base YAML under `code/configs/pretrain/` or `code/configs/finetune/`. CLI flags override YAML, and YAML overrides built-in defaults.
3. Stage 2 uses `run_pretrain.py` and the constant-budget masking policy: `n_budget`, `video_cap_ratio`, `physio_cap_ratio`, `video_min_tubes`, `p_virtual_sensing`, `physio_placement`, `physio_span_tokens`, `spatial_patches`, and `physio_tokens`. Stage 2 uses `signal_weight` or `loss_weights`; no `mask_ratio_*` keys exist.
4. Stage 3 uses `run_waveform.py`; use `alpha`, `beta`, `gamma`, `huber_beta`, `fft_sizes`, `target`, and `streams` as implemented. There is no masking policy and no generic `loss_weight_waveform` key.
5. Use the active split contract: `partition.json` under `<exp_root>/fold<k>/`, `split_seed: 42`, subject-disjoint roles, and `val_select: fixed_epoch` unless the experiment explicitly proposes and grounds a `best_val` change.
6. Ground baseline claims in existing files under `code/notes/eval/`; link verified baselines such as [code/notes/eval/tir_resp](code/notes/eval/tir_resp) or [code/notes/eval/archive/rgb_bp](code/notes/eval/archive/rgb_bp). Do not create a comparison path that does not exist.
7. Keep geometry, checkpoint, and cache contracts explicit: `clip_duration`, `fps`, `temporal_stride`, `tubelet`, `input_size`, `roi_padding`, `crop_rule`, `roi_landmarks`, `fs`, `sig_kernel`, `seq_len`, and the Stage-2 checkpoint used by Stage 3.
8. Use current metrics rather than invented clinical thresholds. Temporal metrics include `pearson`, `band_limited_pearson`, `mae`, `rmse`, `amp_ratio`, `amp_rel_err`, and `mae_bpm`. Spectral metrics include `psd_mae`, `dominant_freq_error_hz`, `hf_power_rel`, and MR-STFT diagnostics. BP clinical HRV requires sufficiently long windows; the repo does not estimate systolic or diastolic BP.
9. Resource gates must be executable or explicitly marked provisional. Include a peak-VRAM cap only with a supporting measured baseline or a clearly labeled estimate.
10. Never alter the study record with the result of an unrealized run. Measured outcomes belong under `code/notes/eval/`, not in an experiment plan.

## Execution Requirements

- From `code/`, pass `-u` to Python runners and report the exact config and output directory.
- For a one-step smoke test, use `--epochs 1 --warmup_epochs 0 --num_workers 0` plus a dataset cap. Prefer `--max_entries 2` for entry-based runners or `--max_clips 2` when it is supported.
- Stage-2 smoke: `runners/run_pretrain.py -c <base-yaml>`.
- Stage-3 smoke: `runners/run_waveform.py -c <base-yaml>`.
- State whether the experiment requires a ROI cache or physiology cache and provide the corresponding build runner if known.
- HPC submission must reference `code/scripts/hpc/submit_sweep.sbatch` for the five-fold Stage-2/Stage-3 sweep, or another existing `.sbatch` file that was verified in the repository. Prefer `bash scripts/hpc/submit.sh scripts/hpc/submit_sweep.sbatch` and carry forward `EXP_ROOT`, `GPU_TYPE`, `GPUS`, and `FOLDS` as needed.

## Output Requirements

Create `code/notes/plan/experiments/exp_<YYYYMMDD>_<short_slug>.md` using the complete structure in `exp_template.md`. Allocate a stable experiment ID such as `EXP-20261121-01`; do not reuse an existing ID.

Before writing, report briefly in the protocol:

- files and configs inspected;
- active implementation stage and runner;
- falsifiable hypothesis;
- exact parameter delta;
- exact baseline and metric contract;
- expected decision and rollback rule;
- exact local smoke command and HPC command.

Every numerical threshold must correspond to a real repo metric or be labeled as an explicitly proposed threshold. No vague gates such as "better performance" are allowed.
