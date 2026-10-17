# Respiration Model — Subject-Level Performance Report

**Source:** `output/finetune/resp_tir_roi_hpc/subject_figures/` (28 figures) + the `subject_metrics.json` they render from.
**Run:** `resp_tir_roi_hpc` — thermal face-ROI → respiration waveform, contact-sensor failure simulated.
**Model:** `project_multimae_base` (Stage-3), encoder = Stage-2 `stage2_hpc_tir_roi_resp/checkpoint-0149.pth` · 8 s clips · fs 100 Hz · input 112 · `physio_norm: session`.
**Test set:** 28 held-out subjects (subject-disjoint), 273 sessions, 5143 clips.
**Report date:** 2026-10-13.

## Headline — band-limited Pearson (0.1–0.6 Hz), median per subject

| | value | 95 % CI |
|---|---|---|
| **Model** | **+0.569** | [+0.451, +0.624] |
| Single-tone template baseline | +0.535 | [+0.499, +0.545] |
| Subjects beating the template | **53.6 %** | — |

The model reconstructs the breathing *rhythm* above its own single-tone baseline, but only **marginally and not robustly** — it wins in barely more than half of subjects (50 % = chance), and the model CI overlaps the baseline CI.

## Consistency across subjects and tasks

The figures show a **highly uneven** per-subject picture (concordant with the wide IQRs in `subject_metrics.json`):

- **Strong:** M034 (+0.80), M032/M043 (+0.74), M031 (+0.73), M056 (+0.64), M045 (+0.62)
- **Weak / failed:** M055 (+0.03), M036 (+0.24), M042 (+0.33), M054 (+0.41), M040 (+0.43)

By distortion group the model tracks the expected difficulty ordering:

| group | tasks | median band-Pearson |
|---|---|---|
| low | T2, T3 | **+0.738** |
| moderate | T4, T7, T8, T10 | +0.524 |
| high | T1, T5, T6, T9 | +0.515 |

(T3 is the single best task at +0.83; T4 the worst at +0.40.)

## Derived vital sign — respiration rate (30 s windows)

- **MAE 4.66 BPM**, within-10 % acceptance **46 %** (n = 409 windows). Rate is consistent with the moderate Pearson: it can follow the rhythm in easy subjects but misses the acceptance band in most windows.

## Other channels

- **Shape:** `psd_mae` ≈ 0.05–0.13 (unit-sum in-band) — moderate spectral match.
- **Artifact:** `hf_power_rel` ≈ 1e-5 — negligible token-seam/high-frequency leakage.
- **Amplitude:** `mae_affine` ≈ 0.7–1.5 z-units — **oracle-calibrated**, not deployable (prediction is session-z-scored, reference is raw volts).

## Caveats (must qualify any claim)

1. **Selection leakage.** `best.pth` was chosen by val Pearson (+0.499) on the *same* subject-disjoint split that is reported here as "test" — per `Evaluation_RespWaveform_protocol.md` §3.1 this must not be the headline; only the LOSO (`--val_subject`) sweep would be clean.
2. **Barely above the rate-only baseline** — 0.569 vs 0.535, beating it in 54 % of subjects.
3. **Rate accuracy** (46 % within 10 %) is below what a reportable vital-sign claim needs.
4. Amplitudes need a true scale (oracle) to be meaningful.

**Verdict:** the model captures the respiration *rhythm* in a subset of subjects (band-Pearson up to ~0.8) but is **not yet a reliable whole-cohort respiration reconstructor** — the cohort median just edges the trivial template, and the subject-to-subject spread is large. The artifacts are `subject_metrics.json` (overall + per-subject/task/group) and 28 `subject_figures/subject_*.png`.
