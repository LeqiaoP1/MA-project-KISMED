# ADR-[NUMBER]: [Short Title, e.g., Introduce Thermal-ROI Cropping Stream]

- **Status**: [Proposed | Implemented (Pending Eval) | Accepted | Deferred | Superseded by ADR-XXX]
- **Date**: YYYY-MM-DD
- **Author**: [Author Name]
- **Related Spec/Plan**: [`code/notes/plan/features/ADR-XXX-<slug>-plan.md` | `code/notes/spec/<file>.md` | `N/A`]
- **Supersedes**: [`ADR-XXX` | `N/A`]

> **Abstraction boundary.** An ADR specifies **what** is decided and **why** — research problem,
> mathematical definitions, invariants, logical parameters, falsifiable criteria, and trade-offs —
> independently of concrete module layouts or runner plumbing so the contract survives code
> refactoring. Implementation phases, target modules, YAML/CLI wiring, and test gates live in the
> paired `code/notes/plan/features/ADR-XXX-<slug>-plan.md` (or in `code/notes/spec/` for pre-existing
> foundational baselines).

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope
[State the single research aspect this ADR governs, what is explicitly **in scope**, and what is
**out of scope** (pointing to owning ADRs where relevant).]

### 1.2 Motivation / Design Pressures
[Distinguish measured repository evidence (`code/notes/eval/...`) from theoretical diagnosis or
hypotheses.]

### 1.3 Requirements
- **R1 — [Name].** [Verifiable requirement.]
- **R2 — [Name].** [Verifiable requirement.]

---

## 2. Decision & Parameter Exposure

[State the normative decision concisely in one bold paragraph.]

### 2.1 Normative Definitions & Mathematical Specification
[Formulas, shapes, invariants (`I1`, `I2`, ...), and algorithms — self-contained and unambiguous.]

### 2.2 Logical Parameter Surface
| Parameter | Type | Default | Allowed / Constraints | Role |
|---|---|---|---|---|
| `param_name` | `type` | `default` | `range / enum` | [purpose; neutral default if opt-in] |

[Only expose parameters that correspond to genuine configuration/CLI knobs (no hidden constants;
`AGENTS.md` §4). YAML keys supply argparse defaults; explicit CLI flags override them.]

### 2.3 Rejected Alternatives
- **[Alternative A]**: [Concrete technical or scientific reason for rejection.]

---

## 3. Falsifiable Criteria

[Define measurable acceptance and rejection criteria before executing experiments.]
- **Criterion 1 (Conformance / Invariants)**: [Exact mathematical, shape, determinism, or unit-test invariant.]
- **Criterion 2 (Smoke / Pipeline)**: [Bounded smoke command (`--max_sessions`, `--max_clips`, `--max_entries`) or build check that completes without error.]
- **Criterion 3 (Scientific / Resource Gate)**: [Exact metric, split/fold, subject-level aggregation per `ADR-009`, baseline, threshold, and VRAM/throughput bound — or `N/A` if purely structural.]

- **Acceptance Artifacts**: [e.g., unit-test suite, `partition.json`, `subject_metrics.json`, `kfold_*.json`, or `N/A`.]

---

## 4. Consequences & Trade-offs

- **Positive**: [Scientific, engineering, or reproducibility benefits.]
- **Negative / Technical Debt**: [Retraining requirements, compute/storage costs, or accepted limitations.]

---

## 5. Reproducibility Record

- **Run / Job IDs**: [Slurm job IDs, local run names, or `N/A`.]
- **Evidence Artifacts**: [Repository-relative paths under `code/notes/eval/`, or `N/A`.]
- **Tests / Smoke Checks**: [Test module paths under `code/tests/`, or `N/A`.]
- **Rollback / Rejection Condition**: [Exact condition and neutral setting to disable, reject, or revert.]
