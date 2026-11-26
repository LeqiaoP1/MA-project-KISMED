# `notes/plan/` — implementation and run plans

Plans are split by **what they govern**, not by when they were written. Both
subfolders reference [`../design/`](../design/README.md) for intent and `../spec/` for the generated
contract; neither is an authority over the code.

| folder | governs | authority for |
|---|---|---|
| [`experiments/`](experiments/) | **training-run protocols**: what is launched, with which config, how folds/roles/checkpoints are sequenced, and which gates the run must pass | the run procedure of record. Measured outcomes are **not** stored here — they land in [`../eval/`](../eval/). |
| [`features/`](features/) | **code implementation plans**: build order, phases, and acceptance gates for a feature whose intent is fixed by a `../design/ADR-*` record | the build sequence. The design record stays the contract; the plan says how it was (or will be) implemented. |

Rules of thumb:

- A document that names configs, `sbatch`/CLI invocations and fold orchestration
  belongs in `experiments/`.
- Every file directly under `features/` MUST follow `ADR-NNN-<slug>-plan.md` and
  map 1-to-1 to an active `../design/ADR-NNN-<slug>.md` contract (enforced by
  `code/tests/test_notes_adr_governance.py`).
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
| [`experiments/exp_20261121_fourstream_vits_random.md`](experiments/exp_20261121_fourstream_vits_random.md) | four-stream ViT-Small random-init Stage-2 baseline run protocol |
| [`experiments/exp_20261124_adr006_tir_primary_rgb2d_ablation.md`](experiments/exp_20261124_adr006_tir_primary_rgb2d_ablation.md) | ADR-006 TIR-primary vs. RGB2D ablation run protocol |
| [`experiments/exp_template.md`](experiments/exp_template.md) | standard template for new experiment run protocols (`EXP-YYYYMMDD-SEQ`) |

## Features (implementation plans — 1-to-1 with `design/ADR-*`)

| document | ADR / contract |
|---|---|
| [`features/ADR-001-subject-kfold-plan.md`](features/ADR-001-subject-kfold-plan.md) | `design/ADR-001-subject-disjoint-split-and-selection.md` |
| [`features/ADR-003-resp-session-cache-plan.md`](features/ADR-003-resp-session-cache-plan.md) | `design/ADR-003-resp-session-cache-artefact.md` |
| [`features/ADR-004-bp-session-cache-plan.md`](features/ADR-004-bp-session-cache-plan.md) | `design/ADR-004-bp-session-cache-artefact.md` |
| [`features/ADR-006-three-regime-masking-plan.md`](features/ADR-006-three-regime-masking-plan.md) | `design/ADR-006-three-regime-virtual-sensing-masking.md` |
| [`features/ADR-007-two-stream-anatomy-aware-masking-plan.md`](features/ADR-007-two-stream-anatomy-aware-masking-plan.md) | `design/ADR-007-two-stream-anatomy-aware-constant-budget-masking.md` |
| [`features/ADR-008-composite-objective-and-cross-modal-alignment-plan.md`](features/ADR-008-composite-objective-and-cross-modal-alignment-plan.md) | `design/ADR-008-composite-objective-and-cross-modal-alignment.md` |
| [`features/ADR-010-bp4d-task-distortion-tiers-plan.md`](features/ADR-010-bp4d-task-distortion-tiers-plan.md) | `design/ADR-010-bp4d-task-distortion-tiers.md` |
| [`features/ADR-011-landmark-roi-geometry-and-tir-cache-plan.md`](features/ADR-011-landmark-roi-geometry-and-tir-cache-plan.md) | `design/ADR-011-landmark-roi-geometry-and-tir-cache.md` |
| [`features/ADR-012-rgb2d-native-memmap-roi-cache-plan.md`](features/ADR-012-rgb2d-native-memmap-roi-cache-plan.md) | `design/ADR-012-rgb2d-native-memmap-roi-cache.md` |
| [`features/ADR-013-visual-clip-and-masked-cube-normalization-plan.md`](features/ADR-013-visual-clip-and-masked-cube-normalization-plan.md) | `design/ADR-013-visual-clip-and-masked-cube-normalization.md` |
| [`features/ADR-016-mixed-precision-and-ddp-accumulation-plan.md`](features/ADR-016-mixed-precision-and-ddp-accumulation-plan.md) | `design/ADR-016-mixed-precision-and-ddp-accumulation.md` |

*(Foundational baselines `ADR-009`, `ADR-014`, and `ADR-015` were implemented prior to the
feature-plan workflow and map directly to [`../spec/`](../spec/). Superseded feature plans for
`ADR-002` and `ADR-005` are preserved in [`features/archive/`](features/archive/).)*
