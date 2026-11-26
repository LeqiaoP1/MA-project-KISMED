# ADR-010: BP4D+ Task Distortion-Tier Taxonomy and Evaluation Stratification

- **Status**: Accepted
- **Date**: 2026-10-24 (codified as ADR-010 on 2026-11-26)
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/plan/features/ADR-010-bp4d-task-distortion-tiers-plan.md`, `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md`
- **Supersedes**: `TIR_ROI_RESP_TasksDistortion.md` (v1 tier mapping replaced by v2 partition on 2026-10-24)

> **Scope.** This ADR governs one research aspect: **the canonical partition of BP4D+ elicitation
> tasks (`T1`–`T10`) into three facial-motion / vocalization distortion tiers (`low`, `moderate`,
> `high`) for stratified evaluation reporting and optional corpus filtering**. Split assignment
> (`ADR-001`), motion-sway clip gating (`ADR-011`), and waveform scoring metrics (`ADR-009`) are
> governed separately.

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope
BP4D+ spans ten spontaneous affective elicitation tasks (`T1`–`T10`) ranging from passive video viewing to active singing, cold-pressor pain, and olfactory disgust. Head motion, non-rigid facial deformation (laughter, grimacing, AU 9 nose wrinkling), and forced vocalization alter both the perinasal thermal boundary layer and landmark tracking stability. Aggregating all tasks into one global score obscures whether a model succeeds on quasi-stationary breathing (`low` distortion) versus high-motion/speech regimes (`high` distortion).

### 1.2 Design Pressures
1. **Misclassified v1 tiers.** The initial v1 mapping (`low=[T2,T3]`, `moderate=[T4,T7,T8,T10]`, `high=[T1,T5,T6,T9]`) placed **T5** (short single-word "yes/no" interview with stable posture) in `high`, **T1** (listening to a joke with smiling/laughter) in `high` alongside active singing/pain, and **T7/T8/T10** (physical threat, cold-pressor pain/grimacing, olfactory disgust/nose-wrinkling) in `moderate` despite severe facial distortion.
2. **Hardcoded drift across configs and scripts.** Prior to v2, the tier string was duplicated across 8 YAML configs, survey scripts, gate probes, and Slurm wrappers with no single canonical constant or automated parity test.
3. **Post-hoc re-aggregation without retraining.** Changing tier labels must not invalidate trained checkpoints or force GPU retraining when `task_set` is unrestricted.

### 1.3 Requirements
- **R1 — Exact partition of `T1`–`T10`.** Every task in `T1`–`T10` belongs to **exactly one** tier (`low`, `moderate`, or `high`) so per-tier session counts sum to total scored sessions with zero unmapped or double-counted tasks.
- **R2 — Label-only default.** Unless `task_set` is explicitly restricted, tiers act solely as evaluation metadata and never alter the training corpus or invalidate checkpoints.
- **R3 — Single canonical source of truth.** One constant in code defines the shipped mapping, enforced across all configs, runners, and batch scripts by automated tests.
- **R4 — Auditable post-hoc relabelling.** Evaluation runners must accept an explicit `--task_groups` override and record provenance (`task_group_source: 'cli' | 'entries'`, `task_groups`, `task_levels`) in output JSONs, warning loudly if any task is unmapped.
- **R5 — Stage-2/Stage-3 corpus-restriction guard.** When a tier subset (`task_set`) restricts the training corpus, Stage 3 must verify parsed equivalence of `task_groups` against its Stage-2 checkpoint.

---

## 2. Decision & Parameter Exposure

**Adopt the v2 BP4D+ task distortion-tier partition (`low=[T2,T3,T5]`, `moderate=[T1,T4]`, `high=[T6,T7,T8,T9,T10]`) as the single canonical taxonomy (`SHIPPED_TASK_GROUPS`), keeping tiers label-only by default (`task_set=''`) while enforcing partition integrity, Stage-2/Stage-3 contract checks, and explicit provenance in all evaluation outputs.**

### 2.1 Normative v2 Distortion-Tier Partition

| Tier | Tasks (`n`) | Physical & Behavioral Criterion | Task Descriptions |
|---|---|---|---|
| **`low`** | `T2, T3, T5` (3) | Minimal head motion, passive viewing or brief single-word verbal responses | **T2** Watch 3D Avatar (passive); **T3** 911 Emergency Video (passive sadness, stable posture); **T5** True/False Interview (brief "yes"/"no", minimal vocal disruption) |
| **`moderate`** | `T1, T4` (2) | Non-rigid facial expression deformations or transient startle without sustained speech/pain | **T1** Listen to Joke (smiling, laughter); **T4** Burst of Sound (abrupt, transient startle reflex) |
| **`high`** | `T6, T7, T8, T9, T10` (5) | Active singing, sustained vocal confrontation, severe head recoil, pain hyperventilation, or perinasal grimacing | **T6** Silly Song (improvised singing); **T7** Physical Threat (withdrawal, pitch tilts); **T8** Cold Pressor (pain, hyperventilation, grimacing); **T9** Performance Complaint (anger, defensive speech); **T10** Smelly Odor (disgust, AU 9 nose wrinkling, gasping) |

**Delta vs. retired v1 mapping:** `T5: high → low`, `T1: high → moderate`, `T7, T8, T10: moderate → high` (counts: `3 / 2 / 5`).

**Canonical serialized form:**
```text
low=T2|T3|T5;moderate=T1|T4;high=T6|T7|T8|T9|T10
```

### 2.2 Invariants
- **I1 (Disjoint exhaustive partition):** $\bigcup_g \mathcal{T}_g = \{T1,\dots,T10\}$ and $\mathcal{T}_g \cap \mathcal{T}_{g'} = \emptyset$ for $g \neq g'$.
- **I2 (Corpus invariance when unrestricted):** With `task_set == ''` and `tasks == ''`, changing `task_groups` changes zero training clips and skips the Stage-2/Stage-3 `task_groups` mismatch warning.
- **I3 (Provenance & unmapped-task guard):** Every session, subject, and k-fold evaluation JSON records `task_group_source` (`'cli'` or `'entries'`), `task_groups`, and `task_levels`. Any scored task absent from the mapping triggers a loud warning so `breakdown_by_label` never silently drops sessions.
- **I4 (Historical preservation):** Prior v1 evaluation tables remain preserved verbatim in `code/notes/eval/`; v2 re-aggregations are appended as dated sections or written to distinct `*_regrouped.json` paths.

### 2.3 Logical Parameter Surface

| Parameter | Type | Default | Allowed / Constraints | Role |
|---|---|---|---|---|
| `task_groups` | str | `SHIPPED_TASK_GROUPS` (training/survey); `''` (eval runners) | semicolon-separated `tier=T_a\|T_b` partition of `T1..T10` | Defines tier membership; eval runners default to `''` to read stored `entries.json` labels unless explicitly overridden |
| `task_set` | str | `''` (all tasks) | CSV of tier names (e.g. `low`, `low,moderate`) | Optional corpus restriction filter; must match between Stage 2 and Stage 3 when non-empty |

### 2.4 Rejected Alternatives
- **Keeping v1 tiers**: Rejected because T5 (stable yes/no) was grouped with singing/anger while T7/T8/T10 (pain/grimacing/AU 9) were grouped with moderate tasks.
- **Allowing overlapping many-to-many tiers in shipped reports**: Rejected because survey aggregation and gate probes assume one unique group per task; non-partition mappings would double-count sessions.
- **Defaulting eval runners' `--task_groups` to `SHIPPED_TASK_GROUPS`**: Rejected because a silent default would retroactively relabel historical prediction dumps without explicit user intent.

---

## 3. Falsifiable Criteria

- **Criterion 1 (Partition & config synchronization)**: Automated unit tests confirm that `SHIPPED_TASK_GROUPS`, every shipped TIR-ROI YAML config, `run_survey_tir_resp.DEFAULT_TASK_GROUPS`, `gate_model_input.py`, and `scripts/hpc/gate_model_input.sbatch` all parse to the exact same partition of `T1`–`T10` (`T5 ∈ low`, `T1 ∈ moderate`, `T10 ∈ high`).
- **Criterion 2 (Relabel provenance & conservation)**: Executing session/subject/k-fold evaluators with `--task_groups` re-aggregates `T5` under `low`, records `task_group_source: 'cli'` and `task_levels`, and conserves total scored sessions (`sum(n_per_group) == n_total`).
- **Criterion 3 (Stage-2/3 contract guard)**: `check_stage2_roi_contract` prints a mismatch warning when `task_set` is active and differs between Stage 2 and Stage 3, and stays silent when both runs have `task_set == ''`.

- **Acceptance Artifacts**: `code/tests/test_task_group_configs.py`, `code/tests/test_config_loading.py`, and regrouped eval reports under `code/notes/eval/archive/tir_resp/`.

---

## 4. Consequences & Trade-offs

- **Positive**: Aligns distortion tiers with physical respiratory/facial-motion confounds; enables zero-GPU post-hoc re-evaluation of existing prediction dumps; eliminates multi-file config drift.
- **Negative / Technical Debt**: Historical v1 tables and notebooks (`notebooks/TirROI_Resp_Pipeline.ipynb`) reflect the older mapping and must be read alongside their dated v2 sections.

---

## 5. Reproducibility Record

- **Run / Job IDs**: CPU post-hoc re-aggregation completed 2026-10-24 (no GPU retraining required).
- **Evidence Artifacts**: Dated v2 sections in `code/notes/eval/archive/tir_resp/resp_5fold_cv.md` and `code/notes/eval/archive/tir_resp/finetuned1_subjectReport.md`; figures `resp_kfold_task_group_medians_v2.svg`, `resp_kfold_task_scatter_v2.svg`, `resp_T3_waveforms_v2.svg`.
- **Tests / Smoke Checks**: `code/tests/test_task_group_configs.py` (26 tests), `code/tests/test_config_loading.py`.
- **Rollback / Rejection Condition**: Pass the legacy v1 mapping string to `--task_groups` on any evaluator to reproduce v1 breakdowns exactly without altering code.
