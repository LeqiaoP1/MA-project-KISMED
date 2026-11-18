# ADR-[NUMBER]: [Short Title, e.g., Introduce Thermal-ROI Cropping Stream]

- **Status**: [Proposed | Accepted | Deprecated | Superseded by ADR-XXX]
- **Date**: YYYY-MM-DD
- **Author**: [Author Name]
- **Related Spec/Plan**: [Repository-relative path or `N/A`]
- **Supersedes**: [ADR-XXX or `N/A`]

## 1. Context & Problem Statement
[Explain the research challenge or limitation motivating this change. Example: Full-frame TIR video contains excessive background noise, reducing the SNR for respiration waveform recovery.]

## 2. Decision & Parameter Exposure
[State the chosen design and why it is preferred over the rejected alternatives.]
- Target Modules: [Repository-relative source paths]
- Target Runner(s): [For example, `code/runners/run_waveform.py`]
- Config File(s): [Repository-relative YAML paths]
- Exposed Parameters:
  - `config_key` / `--cli-option` (type, default, allowed values): [purpose]

[List only parameters that are defined by the target runner's `argparse` parser.
YAML keys are argparse defaults; explicit CLI arguments override them.]

## 3. Falsifiable Criteria
[Define measurable acceptance and rejection criteria before running the experiment.]
- Criterion 1: [Exact metric, split/fold, aggregation unit, baseline, and threshold. For respiration, use the active subject-level evaluation protocol and its headline metric.]
- Criterion 2: [Smoke command, including caps such as `--max_sessions`, `--max_clips`, or `--max_entries`, completes successfully.]
- Criterion 3: [Resource threshold, including peak allocated/reserved VRAM, batch size, device, and precision mode, or `N/A`.]

- Evaluation Command(s): [Repository-relative runner command(s)]
- Acceptance Artifact(s): [For example, `subject_metrics.json`, `kfold_*.json`, `metrics_final.json`, or `N/A`]

## 4. Consequences & Trade-offs
- **Positive**: Higher feature saliency and improved physiological signal reconstruction.
- **Negative / Technical Debt**: Requires offline facial landmark caching prior to training.

## 5. Reproducibility Record
- **Run / Job IDs**: [Slurm job IDs, local run names, or `N/A`]
- **Evidence Artifacts**: [Repository-relative paths, output paths, or `N/A`]
- **Tests / Smoke Checks**: [Test paths and commands, or `N/A`]
- **Rollback / Rejection Condition**: [When to reject, disable, or revert this decision]
