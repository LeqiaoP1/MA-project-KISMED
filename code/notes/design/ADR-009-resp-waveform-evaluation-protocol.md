# ADR-009: Subject-Level Respiration Waveform Evaluation Protocol

- **Status**: Accepted
- **Date**: 2026-10-12 (revised 2026-11-18; codified as ADR-009 on 2026-11-26)
- **Author**: KISMED MA research team
- **Related Spec/Plan**: Pre-existing baseline (`code/notes/spec/03_engine_and_loss.md`, `code/evaluation/`, `code/runners/run_evaluate_{session,subject,kfold}.py`); split & selection governed by `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md`; reference artefact governed by `code/notes/design/ADR-003-resp-session-cache-artefact.md`; task distortion tiers governed by `code/notes/design/ADR-010-bp4d-task-distortion-tiers.md`
- **Supersedes**: `Evaluation_RespWaveform_protocol.md` (promoted verbatim into ADR structure)

> **Scope.** This ADR governs one research aspect: **how continuous respiration waveform predictions
> are assembled, scored against the ground-truth reference, aggregated across sessions and held-out
> subjects, and compared against trivial baselines**. Split partitioning (`ADR-001`), signal
> artefact cleaning (`ADR-003`), task distortion-tier definitions (`ADR-010`), and model training
> objectives (`ADR-008`, `ADR-015`) are out of scope here.

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope
The multi-stage pipeline (Stage-1 VideoMAE initialization → Stage-2 multimodal masked pre-training → Stage-3 waveform regression) exists to answer one thesis question:

> **Can the respiration waveform of an unseen subject be reconstructed from the thermal face ROI alone?**

Performance of the entire system is therefore **subject-level respiration reconstruction quality** on held-out test subjects. Stages 1 and 2 are judged solely by whether they improve this downstream subject-level metric, never by their own pretext reconstruction losses alone.

### 1.2 Design Pressures
1. **Clips are not independent observations.** Evaluating per clip inflates sample size, hides seam discontinuities across overlapping sliding windows, and conflates subject effects with clip counts.
2. **Reference/label preprocessing mismatch.** Scoring predictions against raw `Resp_Volts.txt` resampled with `np.interp` compares two different preprocessing spaces (no anti-alias filter, no per-run DC removal, no validity mask) and charges reference artefacts to the model.
3. **Baseline drift and scale ambiguity.** Predictions are trained on per-clip z-scored targets and carry no physical voltage scale; raw amplitude metrics require oracle per-session affine fitting against the reference and are neither deployable nor comparable across normalizations or clip lengths.
4. **Rate-only shortcuts.** A single sinusoid at the reference's dominant frequency can achieve high Pearson correlation without capturing inter-breath waveform morphology.

### 1.3 Requirements
- **R1 — Subject as the unit of evaluation.** Overlapping clips are assembled into continuous per-session waveforms and aggregated to one number per held-out subject, with subject-level uncertainty quantification.
- **R2 — Single reference space.** Training labels and evaluation references must come from the exact same precomputed respiration artefact (`ADR-003`), with invalid stretches masked as holes (never zero-filled).
- **R3 — Scale- and offset-invariant morphological/rhythm metrics.** Headline metrics must measure respiratory rhythm and shape without oracle amplitude calibration.
- **R4 — Mandatory baseline controls.** Every subject-level claim must be benchmarked against a constant floor, a single-tone frequency template, and off-face/motion confounds.
- **R5 — Auditability across 5 subject-disjoint folds.** Pooled cross-validation must enforce subject disjointness across folds (`ADR-001`) and record provenance (`selection`, `eval_role`, `reference_source`, `task_group_source`).

---

## 2. Decision & Parameter Exposure

**Evaluate respiration reconstruction at the held-out subject level by assembling per-clip predictions via Hann overlap-add, scoring them against the `ADR-003` respiration artefact over the `[0.10, 0.60] Hz` respiratory passband, reporting per-subject median band-limited Pearson with a subject-level 95% bootstrap confidence interval as the headline metric, and retiring oracle-calibrated amplitude from headline claims.**

### 2.1 Session Assembly & Single-Reference Contract
1. **Continuous session assembly:** All clips belonging to a held-out `(subject, task)` session are combined onto the session timeline using **Hann overlap-add** with weight-sum normalization (`evaluation/assemble.py`).
2. **Reference from `ADR-003` artefact only:** The ground-truth reference `(t, y, mask)` is read from the same `resp_cache` shard used for training (`ADR-003`), where invalid samples are `NaN`. Before scoring, the artefact's `mask` is **ANDed** with the clip-coverage mask so reference holes are excluded rather than scored as zero.
3. **Provenance:** Every session record writes `reference_source: 'resp_cache'` plus the resolved `resp_cache_key`. The legacy raw-file route (`reference_source: 'raw_resp_volts'`) is forbidden for reported thesis numbers.

### 2.2 Metric Hierarchy (Morphological Shape & Rhythm Only)

| Role | Metric | Definition / Rationale |
|---|---|---|
| **Headline** | **Band-limited Pearson** (`0.10–0.60 Hz`) | Per session on assembled valid samples, then median per subject; removes baseline drift and measures breathing rhythm/phase fidelity |
| Reference | Plain Pearson | Same aggregation without bandpass filtering; shown for drift contrast |
| Rate (waveform) | `dominant_freq_error_hz` (reported in bpm) | Peak frequency discrepancy inside `[0.10, 0.60] Hz` |
| Rate (derived vital sign) | `mae_bpm`, `% within 10%` (`rate_tol=0.10`), RMSE/bias (bpm) | Band-limited FFT peak (`estimate_rate_bpm`, parabolic sub-bin) over scored windows (default `rate_window_s = 30 s`) |
| Spectral (Tier 2) | `psd_mae` (in-band Welch PSD, unit-sum) | In-band spectral energy distribution consistency |
| Artifact (Tier 2) | `hf_power_rel` vs `target_hf_power_rel` | Out-of-band (token-seam / high-frequency) energy diagnostic |
| Morphology *(open requirement)* | Template-subtracted residual correlation & unit-depth per-breath shape | Required before making a strict inter-breath morphology claim beyond rate/phase tracking |

#### Retirement of Amplitude Metrics (Decision 2026-11-01)
Oracle-calibrated fields (`amp_ratio`, `amp_rel_err`, `mae_affine`, `rmse_affine`, produced via `affine_calibrate`) are **retired from all headline tables, console summaries, figure titles, and acceptance gates**. They remain in per-session/per-task/per-fold JSON blocks solely as non-gating diagnostics.

#### Scope of Tier-3 Clinical HRV
Tier-3 clinical HRV metrics (`rmssd_ms`, `pnn50`, `median_nn_ms`, `shannon_entropy` via NeuroKit2, requiring $\ge 30\text{ s}$) apply exclusively to **BP** waveform evaluation and are out of scope for respiration.

### 2.3 Mandatory Baselines & Subject-Level Aggregation
1. **Baselines every model must beat:**
   - *Constant / mean line*: Pearson $= 0$ (sanity floor).
   - *Single-tone template*: Least-squares $\cos/\sin$ fit at the reference's own peak frequency (the ceiling of any rate-only model). A waveform claim requires beating this template on a majority of sessions/subjects.
   - *Off-face / global-motion control*: Any claim must survive regressing out global head/chest motion channels.
2. **Aggregation hierarchy:**
   - **Per subject**: median across the subject's scored sessions (robust to short/noisy tasks).
   - **Across subjects**: median + IQR, plus a **95% bootstrap confidence interval over subjects** (each subject is one independent observation; clips within a subject are not).
   - Report the **fraction of subjects whose headline beats the single-tone template**, plus breakdowns by `per_task` (`T1`–`T10`) and `per_group` (`low`/`moderate`/`high` per `ADR-010`).
3. **Pooled 5-fold CV:** Pooling across `fold 0..4` (`ADR-001`) asserts exact subject disjointness across folds (aborting on any overlap or duplicate subject) and produces one pooled subject-level CI (`kfold_<run_prefix>.json`).

### 2.4 Logical Parameter Surface

| Parameter | Type | Default | Allowed / Constraints | Role |
|---|---|---|---|---|
| `resp_cache` | path | *(required for reported runs)* | valid `ADR-003` cache root | Supplies single-space ground-truth reference & validity mask |
| `resp_cache_key` | str | `''` (auto-resolve) | cache directory slug | Pins exact artefact identity |
| `fs` | float | `100.0` | must equal cache `fs_out` | Sampling rate for assembly, filtering, and PSD |
| `band` | `(f_lo, f_hi)` | `(0.10, 0.60)` Hz | `0 < f_lo < f_hi < fs/2` | Respiratory passband (`6–36 bpm`) |
| `rate_window_s` | float | `30.0` | `> 0` | Window length for derived BPM rate metrics (`rate_overall`) |
| `rate_tol` | float | `0.10` | `(0, 1)` | Relative tolerance for `% within 10%` BPM accuracy |
| `task_groups` | str | `''` (read from `entries.json`) | `ADR-010` partition syntax | Post-hoc relabelling override for `per_group` breakdown |

### 2.5 Rejected Alternatives
- **Per-clip evaluation as headline**: Rejected because overlapping clips violate independence and ignore boundary seams.
- **Raw `Resp_Volts.txt` + `np.interp` reference**: Rejected because it diverges from the `ADR-003` training label space.
- **Oracle-calibrated amplitude as headline metric**: Rejected because per-clip z-scoring removes absolute scale and per-session affine fitting leaks reference amplitude.
- **Evaluating test subjects during model selection**: Rejected by `ADR-001` (selection reads validation role only; test role is scored once at report time).

---

## 3. Falsifiable Criteria

- **Criterion 1 (Reference & hole integrity)**: Every reported evaluation artifact records `reference_source == 'resp_cache'`; invalid mask samples in the `ADR-003` shard are excluded from both waveform metrics and BPM rate windows (never zero-filled).
- **Criterion 2 (Pooled fold disjointness)**: Pooled k-fold aggregation raises an error on any subject overlap or duplication across folds, and per-group session counts sum exactly to total scored sessions.
- **Criterion 3 (Amplitude exclusion)**: `amp_ratio`, `amp_rel_err`, `mae_affine`, and `rmse_affine` never appear in `overall` summaries, console headlines, or acceptance gates.
- **Criterion 4 (Deterministic reproducibility)**: Re-running session, subject, and k-fold evaluators on identical `preds.npy` + `entries.json` + `resp_cache` yields bit-identical JSON metrics (with fixed bootstrap seed).

- **Acceptance Artifacts**: `session_metrics.json`, `subject_metrics.json`, `kfold_<run_prefix>.json`, and unit tests (`tests/test_evaluation_subject.py`, `tests/test_evaluation_breakdown.py`, `tests/test_resp_eval_reference.py`, `tests/test_protocol.py`).

---

## 4. Consequences & Trade-offs

- **Positive**: Enforces one leak-free, reference-consistent, subject-level yardstick across every Stage-2 and Stage-3 experiment (`ADR-006`, `ADR-007`, `ADR-008`, `ADR-015`).
- **Negative / Technical Debt**:
  - Pre-2026-10-31 numbers scored on raw `np.interp` references are on a different reference space and cannot be directly compared to `resp_cache` runs.
  - A dedicated inter-breath morphology metric (template-subtracted residual correlation / unit-depth breath shape) is not yet implemented in `code/evaluation/metrics.py`; until added, claims remain limited to rhythm/band-limited waveform correlation and breathing rate.
  - Paired cross-model significance testing (paired subject-level bootstrap difference) is a separate comparison step outside single-model reports.

---

## 5. Reproducibility Record

- **Run / Job IDs**: Applied across 5-fold TIR-ROI Stage-3 evaluations (`code/notes/eval/archive/tir_resp/resp_5fold_cv_96respgrid.md`).
- **Evidence Artifacts**: `code/notes/eval/archive/tir_resp/resp_5fold_cv_96respgrid.md`, `code/notes/eval/archive/tir_resp/resp_5fold_cv.md`.
- **Tests / Smoke Checks**: `code/tests/test_evaluation_subject.py`, `code/tests/test_evaluation_breakdown.py`, `code/tests/test_resp_eval_reference.py`, `code/tests/test_evaluation_amplitude.py`.
- **Rollback / Rejection Condition**: Not applicable (protocol contract of record). Any change to passband, assembly rule, or headline metric requires a superseding ADR.
