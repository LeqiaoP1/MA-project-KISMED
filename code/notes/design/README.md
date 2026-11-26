# `notes/design/` — Architecture Decision Records (ADR) Registry

This directory is the **single authoritative design index** for the KISMED MA research study.
Per [`AGENTS.md`](../../../AGENTS.md) (§1 *Zero Redundancy*, §3 *Directory Layout*), **every active
research aspect is governed by exactly one ADR** in this folder, and every code implementation plan
under [`../plan/features/`](../plan/features/) pairs 1-to-1 with an ADR here.

- **Contract vs. Implementation**: `design/ADR-NNN-<slug>.md` defines the normative scientific &
  mathematical contract (what & why). `plan/features/ADR-NNN-<slug>-plan.md` defines the build
  order, module wiring, and test gates (how). Generated code-level specs live in
  [`../spec/`](../spec/); measured outcomes live in [`../eval/`](../eval/).
- **Immutability**: Published ADR IDs (`ADR-001`+) are never renumbered. When a decision is replaced,
  the superseded ADR and its feature plan move to [`archive/`](archive/) (and
  [`../plan/features/archive/`](../plan/features/archive/)) with a `Superseded by ADR-NNN` header.
- **Lifecycle Statuses**: `Proposed` → `Implemented (Pending Eval)` → `Accepted` | `Deferred` | `Superseded by ADR-NNN`.

---

## Active Research Aspects (One Aspect per ADR)

### A. Experimental Protocol & Evaluation
| ADR | Research Aspect | Status | Implementation Plan / Spec |
|---|---|---|---|
| [`ADR-001`](ADR-001-subject-disjoint-split-and-selection.md) | Subject-disjoint 5-fold split (70/10/20) & validation-only selection | `Accepted` | [`ADR-001-subject-kfold-plan.md`](../plan/features/ADR-001-subject-kfold-plan.md) |
| [`ADR-009`](ADR-009-resp-waveform-evaluation-protocol.md) | Subject-level respiration waveform evaluation (Hann assembly, passband, Tiers 1–3, bootstrap CI) | `Accepted` | Pre-existing baseline (`spec/03_engine_and_loss.md`, `code/evaluation/`) |
| [`ADR-010`](ADR-010-bp4d-task-distortion-tiers.md) | BP4D+ task distortion-tier taxonomy (`low`/`moderate`/`high`) & eval stratification | `Accepted` | [`ADR-010-bp4d-task-distortion-tiers-plan.md`](../plan/features/ADR-010-bp4d-task-distortion-tiers-plan.md) |

### B. Physiological Signal Artefacts
| ADR | Research Aspect | Status | Implementation Plan / Spec |
|---|---|---|---|
| [`ADR-003`](ADR-003-resp-session-cache-artefact.md) | Precomputed respiration session artefact (validity mask, per-run DC removal, grid) | `Accepted` | [`ADR-003-resp-session-cache-plan.md`](../plan/features/ADR-003-resp-session-cache-plan.md) |
| [`ADR-004`](ADR-004-bp-session-cache-artefact.md) | Precomputed blood-pressure session artefact (3-gate validity mask, segment DC removal) | `Accepted` | [`ADR-004-bp-session-cache-plan.md`](../plan/features/ADR-004-bp-session-cache-plan.md) |

### C. Visual ROI Geometry, Caching & Input Normalization
| ADR | Research Aspect | Status | Implementation Plan / Spec |
|---|---|---|---|
| [`ADR-011`](ADR-011-landmark-roi-geometry-and-tir-cache.md) | Landmark-guided square ROI geometry, isotropic rescale, motion gate & native TIR hull cache | `Accepted` | [`ADR-011-landmark-roi-geometry-and-tir-cache-plan.md`](../plan/features/ADR-011-landmark-roi-geometry-and-tir-cache-plan.md) |
| [`ADR-012`](ADR-012-rgb2d-native-memmap-roi-cache.md) | 2D-texture (`rgb2d`) native memmap ROI cache & multimodal stream join | `Accepted` | [`ADR-012-rgb2d-native-memmap-roi-cache-plan.md`](../plan/features/ADR-012-rgb2d-native-memmap-roi-cache-plan.md) |
| [`ADR-013`](ADR-013-visual-clip-and-masked-cube-normalization.md) | Model-side visual clip z-score input norm & masked-only per-cube target standardization | `Accepted` | [`ADR-013-visual-clip-and-masked-cube-normalization-plan.md`](../plan/features/ADR-013-visual-clip-and-masked-cube-normalization-plan.md) |

### D. Multimodal Architecture, Stage-2 Masking & Objectives
| ADR | Research Aspect | Status | Implementation Plan / Spec |
|---|---|---|---|
| [`ADR-014`](ADR-014-stage2-multimae-early-token-mixing-architecture.md) | Stage-2 `MultiModalMAE` early-token-mixing backbone, space-time alignment & VideoMAE init | `Accepted` | Pre-existing baseline (`spec/02_model_architecture.md`, `code/core/multimae.py`) |
| [`ADR-007`](ADR-007-two-stream-anatomy-aware-constant-budget-masking.md) | Stage-2 masking (Phase-1 active): 2-stream anatomy-aware constant-budget (`tir`, `resp`) | `Implemented (Pending Eval)` | [`ADR-007-two-stream-anatomy-aware-masking-plan.md`](../plan/features/ADR-007-two-stream-anatomy-aware-masking-plan.md) |
| [`ADR-006`](ADR-006-three-regime-virtual-sensing-masking.md) | Stage-2 masking (Phase-2 fusion): 3-regime multi-camera virtual sensing (`tir`, `rgb2d`, `resp`) | `Deferred` | [`ADR-006-three-regime-masking-plan.md`](../plan/features/ADR-006-three-regime-masking-plan.md) |
| [`ADR-008`](ADR-008-composite-objective-and-cross-modal-alignment.md) | Stage-2 composite respiration objective ($\mathcal{L}_\text{MSE}+\mathcal{L}_\text{Pearson}+\mathcal{L}_\text{spectral}$) & InfoNCE alignment | `Implemented (Pending Eval)` | [`ADR-008-composite-objective-and-cross-modal-alignment-plan.md`](../plan/features/ADR-008-composite-objective-and-cross-modal-alignment-plan.md) |
| [`ADR-015`](ADR-015-stage3-waveform-regression-and-joint-loss.md) | Stage-3 waveform regression head, Stage-2 encoder transfer contract & `WaveformJointLoss` | `Accepted` | Pre-existing baseline (`spec/02_model_architecture.md`, `code/core/waveform_model.py`) |

### E. Compute & Execution Infrastructure
| ADR | Research Aspect | Status | Implementation Plan / Spec |
|---|---|---|---|
| [`ADR-016`](ADR-016-mixed-precision-and-ddp-accumulation.md) | Automatic mixed precision (`--amp_dtype`) & DDP `no_sync` gradient accumulation | `Implemented (Pending Eval)` | [`ADR-016-mixed-precision-and-ddp-accumulation-plan.md`](../plan/features/ADR-016-mixed-precision-and-ddp-accumulation-plan.md) |

---

## Superseded / Archived Records (`archive/`)

Historical records are preserved read-only for provenance and must not be implemented from:

| Archived Document | Superseded By | Archived Feature Plan |
|---|---|---|
| [`archive/SpanMask_PhysioSignals.md`](archive/SpanMask_PhysioSignals.md) | `ADR-002`..`ADR-007` (`mask_placement.py`) & 2026-11-09 STFT removal | — |
| [`archive/ADR-002-asymmetric-multimodal-masking.md`](archive/ADR-002-asymmetric-multimodal-masking.md) | `ADR-005` → `ADR-006` / `ADR-007` | [`../plan/features/archive/ADR-002-constant-budget-masking-plan.md`](../plan/features/archive/ADR-002-constant-budget-masking-plan.md) |
| [`archive/ADR-005-asymmetric-multimodal-masking-3streams.md`](archive/ADR-005-asymmetric-multimodal-masking-3streams.md) | `ADR-006` → `ADR-007` | [`../plan/features/archive/ADR-005-asymmetric-multimodal-masking-3streams-plan.md`](../plan/features/archive/ADR-005-asymmetric-multimodal-masking-3streams-plan.md) |

---

## Authoring a New ADR

1. Copy [`template.md`](template.md) to `ADR-NNN-<kebab-slug>.md` using the next unused integer ID.
2. Keep each ADR scoped to **one research aspect**; cross-reference other aspects by ADR ID.
3. Add a row to this registry and — whenever code changes are planned — create a matching
   `code/notes/plan/features/ADR-NNN-<kebab-slug>-plan.md` and register it in
   [`../plan/README.md`](../plan/README.md).
4. Verify with `pytest code/tests/test_notes_adr_governance.py`.
