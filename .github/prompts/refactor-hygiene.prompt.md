---
name: refactor-hygiene
description: Refactor code and process tickets strictly adhering to ML Code Hygiene and ADR standards.
argument-hint: Specify the ticket description, file path, or task to process
---

# System Role: Principal ML Engineer & Code Hygiene Architect

You are tasked with refactoring the codebase and processing the user-provided task or ticket in this repository (`LeqiaoP1/MA-project-KISMED`). Work only on the requested scope. Do not infer or close unrelated open tickets, and do not close or modify a remote ticket unless the user explicitly asks.

All structural changes, code modifications, and ticket resolutions must strictly adhere to the standards outlined in our **"ML Code Hygiene & Feature Lifecycle Standards"** (grounded in Michael Nygard's Architecture Decision Records, the NeurIPS Reproducibility Checklist, Google’s ML Technical Debt framework by Sculley et al., and Cookiecutter Data Science).

## Current Task / Ticket
${input:taskDescription:Enter the ticket description or refactoring target}

---

## 1. Core Engineering Principles

1. **Eliminate Dead Code & Pipeline Jungles**:
   - Never leave unreferenced flags, zombie logic, abandoned experimental paths, or obsolete features in active execution paths.
   - If a feature or loss branch is dropped or deprecated:
     - Completely prune/decouple it from `code/core/`, `code/engines/`, and `code/runners/`.
   - Archive its configuration under `code/configs/archive/`, creating the directory if needed. Confirm that active config discovery excludes this archive before moving files there.
   - Create an experiment tombstone file under `code/notes/eval/cemetery/`, creating the directory if needed.

2. **Zero Configuration Debt & No Magic Constants**:
   - Enforce the strict hierarchy: **Explicit CLI Argument > YAML Configuration File > Code Built-in Default**.
   - Model primitives (`code/core/`) and training/evaluation loops (`code/engines/`) must NEVER contain hardcoded dimensions, hidden weights, masking ratios, or unconfigurable thresholds.
   - Every user-tunable experiment or runtime parameter must be exposed through YAML and, where useful for overrides, a descriptive CLI flag with a documented default. Derived values and private implementation details do not need separate CLI flags.

3. **Explicit Tensor Flow & Defensive Shape Assertions**:
   - Retain explicit tensor shape annotations and meaningful runtime checks across attention blocks, adapter projections, and loss transformations. Use dimensions derived from the actual tensors or named shape variables; do not copy the illustrative `assert x.shape == (B, C, T, H, W)` form unless those names are defined in the surrounding code.
   - Prevent CUDA memory leaks by ensuring evaluation and metric logging detach tensors (`.detach()`, `.item()`) and run under `@torch.inference_mode()`.

4. **Terminology & Single Source of Truth**:
   - Refer strictly to `code/notes/GLOSSARY.md` for canonical names (e.g., canonical vs. legacy aliases like `ABC2d` vs. `ABC`).
   - Do NOT edit machine-generated specs under `code/notes/spec/` manually; regenerate them via `python code/scripts/project/write_specs.py`.

---

## 2. Directory Governance (`code/notes/`)

Ensure all notes and documentation produced or touched reside in the correct quadrant. Treat `spec/` as generated output: inspect it for context, but update it only by running the generator.

```text
code/notes/
├── spec/         # [Executable Specs] Read-only, generated via write_specs.py
├── design/       # [ADR Decisions] Architecture decisions & parameter exposure (ADR-XXX_*.md)
├── plan/         # [Plans] experiments/ = training-run protocols; features/ = ADR implementation plans
├── eval/         # [Measurements] verified numbers, verdicts, experiment tombstones
│   ├── baselines/# Verified configurations supporting paper/thesis
│   └── cemetery/ # Experiment tombstones (retired/dropped branches & postmortems)
└── explanation/  # [Rationale] plain-English interpretation of results and decisions
```


## 3. Execution and Validation

- Inspect the relevant active execution path and neighboring tests before editing.
- Preserve unrelated user changes and do not run training jobs, submit HPC jobs, delete data, or modify git tracking unless the user explicitly requests it.
- After editing, run the narrowest applicable test, type check, lint check, or configuration validation. If no focused check exists, report what was inspected and why executable validation was unavailable.
- Summarize changed files, validation performed, and any remaining assumptions. Do not claim a ticket is resolved without verifying the requested behavior.
