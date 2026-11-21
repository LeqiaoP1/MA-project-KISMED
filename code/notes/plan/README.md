# `notes/plan/` — implementation and run plans

Plans are split by **what they govern**, not by when they were written. Both
subfolders reference `../design/` for intent and `../spec/` for the generated
contract; neither is an authority over the code.

| folder | governs | authority for |
|---|---|---|
| [`experiments/`](experiments/) | **training-run protocols**: what is launched, with which config, how folds/roles/checkpoints are sequenced, and which gates the run must pass | the run procedure of record. Measured outcomes are **not** stored here — they land in [`../eval/`](../eval/). |
| [`features/`](features/) | **code implementation plans**: build order, phases, and acceptance gates for a feature whose intent is fixed by a `../design/ADR-*` record | the build sequence. The design record stays the contract; the plan says how it was (or will be) implemented. |

Rules of thumb:

- A document that names configs, `sbatch`/CLI invocations and fold orchestration
  belongs in `experiments/`.
- A document that maps an ADR onto modules, phases and tests belongs in
  `features/`.
- Results, sweeps and verdicts belong in `../eval/`; rationale/interpretation
  belongs in `../explanation/`.
- When a plan references a document in the other subfolder, use the full
  `code/notes/plan/<subfolder>/<file>.md` path so the link survives a move.

## Experiments (run protocols)

| document | scope |
|---|---|
| [`experiments/RGB_BP_plan.md`](experiments/RGB_BP_plan.md) | thesis plan of record for the RGB → BP/RESP waveform cascade (work items W1–W7) |
| [`experiments/TirROI_Resp_plan.md`](experiments/TirROI_Resp_plan.md) | procedure of record for the TIR-ROI → respiration protocol |
| [`experiments/kfold_run_checklist.md`](experiments/kfold_run_checklist.md) | executable five-fold runbook: freeze, partition, Stage 2 → Stage 3 → test → pool |
| [`experiments/local_downstream_plan.md`](experiments/local_downstream_plan.md) | local 64 px Stage-3 setup record (BP and RESP branches) |

## Features (implementation plans)

| document | ADR / contract |
|---|---|
| [`features/ADR-001-subject-kfold-plan.md`](features/ADR-001-subject-kfold-plan.md) | `design/ADR-001-subject-disjoint-split-and-selection.md` |
| [`features/ADR-002-constant-budget-masking-plan.md`](features/ADR-002-constant-budget-masking-plan.md) | `design/ADR-002-asymmetric-multimodal-masking.md` |
| [`features/ADR-003-resp-session-cache-plan.md`](features/ADR-003-resp-session-cache-plan.md) | `design/ADR-003-resp-session-cache-artefact.md` |
| [`features/ADR-004-bp-session-cache-plan.md`](features/ADR-004-bp-session-cache-plan.md) | `design/ADR-004-bp-session-cache-artefact.md` |
| [`features/2DTexture_ROI_cache_implementation_plan.md`](features/2DTexture_ROI_cache_implementation_plan.md) | `design/2DTexture_ROI_cache_preprocessing.md` |
| [`features/ROI_crop_implementation_plan.md`](features/ROI_crop_implementation_plan.md) | `design/ROI_crop_and_rescaling.md` |
| [`features/video_input_preprocessing_implementation_plan.md`](features/video_input_preprocessing_implementation_plan.md) | `design/Video_input_Preprocessing.md` |
| [`features/hpc_AMP_implementation.md`](features/hpc_AMP_implementation.md) | `design/hpc_AMP.md` |
| [`features/task_tiers_regroup_plan.md`](features/task_tiers_regroup_plan.md) | `design/TIR_ROI_RESP_TasksDistortion.md` |
