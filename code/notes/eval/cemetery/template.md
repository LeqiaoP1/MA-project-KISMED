# Tombstone: [Retired Feature Name, e.g., Bidirectional Spectral Loss]

- **Retired Date**: YYYY-MM-DD
- **Last Active Config**: [Repository-relative config path or `N/A`]
- **Related ADR**: ADR-XXX
- **Replacement**: [Current feature, config, ADR, or `N/A`]
- **Code Status**: [Removed | Retained but disabled | Archived]

## 1. Initial Hypothesis
[State what the experiment was intended to achieve. Example: Enforcing frequency-domain PSD alignment would constrain periodicity and improve heart/breathing rate fidelity.]

## 2. Empirical Findings & Reason for Retirement
[Record only verified results and root-cause failure reasons. Use the active evaluation protocol and identify the aggregation unit.]
- **Empirical Results**: [Metric, split/fold, baseline, threshold, and measured result]
- **Bottleneck / Failure Mode**: [Observed technical cause, or `N/A`]
- **Decision Rule**: [Why these findings justify retirement]

## 3. Evidence Provenance
- **Run / Job IDs**: [Slurm job IDs, local run names, or `N/A`]
- **Evaluation Command(s)**: [Repository-relative runner command(s), or `N/A`]
- **Evidence Artifacts**: [For example, `metrics_final.json`, `subject_metrics.json`, `kfold_*.json`, logs, or `N/A`]
- **Protocol / Spec Version**: [Repository-relative path and revision date, or `N/A`]

## 4. Retirement and Code Status
- [ ] Removed or decoupled the feature from active execution paths.
- [ ] Removed or disabled corresponding parameters in active YAML configurations.
- [ ] Preserved the last active configuration, checkpoint, or equivalent reproduction reference.
- [ ] Updated affected specifications, glossary entries, and related ADRs.
- [ ] Verified that baseline pipelines run without requiring legacy dependencies.
- [ ] Confirmed that no active runner or config still enables the retired feature unintentionally.
