# Evaluation Protocol — Respiration Waveform Reconstruction (tested subject)

> Status: **active** · authored 2026-10-12 · scope: Stage-3 respiration output
> (the final task of the whole model).

## 1. Objective

The whole pipeline (Stage 1 VideoMAE init -> Stage 2 masked multimodal pretrain
-> Stage 3 waveform finetune) exists to answer one question:

> **Can the respiration waveform of an unseen subject be reconstructed from the
> thermal face ROI alone?**

"Performance of the entire model" is therefore the **subject-level respiration
reconstruction quality**, and Stages 1/2 are judged only by whether they move
that number (ablation), never on their own pretext losses.

## 2. Unit of evaluation = the tested subject

Clips are an implementation detail. The evaluation object is **one held-out
subject**: all their clips are assembled into continuous per-session waveforms
(`evaluation/assemble.py`, Hann overlap-add) and compared to the raw
`Resp_Volts` reference. We then report **one number per tested subject**, and
aggregate across subjects.

## 3. Protocol

### 3.1 Now — a single fixed subject-disjoint test set
- Train / val / **test** subjects are disjoint. `split_by: subject`.
- **Model selection and reporting must be separated**: the epoch / checkpoint is
  chosen on the *val* subjects, and the reported number is on the *test*
  subjects. (Today `run_waveform.py` checkpoints on `best_pearson` over the same
  val split it also reports — that is selection leakage and must not be the
  headline.)
- `--test_subjects F001,F002,...` selects the reported set.

### 3.2 Later — k-fold subject-disjoint cross-validation
- The same command is run once per fold with that fold's held-out subject set;
  `subject_metrics.json` per fold, then pooled. The runner is written so a fold
  is *just a different `--test_subjects` list* — no code change.
- Leave-one-subject-out is the k = N limit (140 subjects); pragmatically k = 5–10
  folds of ~14–28 subjects each balances rigor and budget.

## 4. Metrics

The prediction is **session-z-scored** and the reference is **raw volts**
(`physio_norm: session`), so only scale/offset-invariant metrics are directly
valid; amplitude errors need the (oracle) affine calibration and must be flagged.

| Role | Metric | Rationale |
|---|---|---|
| **Headline** | **band-limited Pearson** (0.1–0.6 Hz), per session, median per subject | removes baseline drift; the scale-invariant "is the breathing rhythm reconstructed?" number |
| Reference | plain Pearson | same but drift-sensitive; shown for contrast |
| Rate | `dominant_freq_error_hz` (report in bpm) | breathing-rate accuracy |
| Shape | `psd_mae` (in-band, unit-sum) | spectral shape |
| Artifact | `hf_power_rel` vs `target_hf_power_rel` | out-of-band (token-seam) energy |
| Amplitude | `mae_affine` / `rmse_affine` | **oracle-calibrated** (needs the session's true scale); NOT deployable |

**Morphology is a distinct claim.** Rate/phase tracking (high Pearson) does not
imply waveform morphology. A morphology claim requires template-subtracted
residual correlation / per-breath shape, which are *not* assumed here.

## 4b. Future target — derived vital sign: respiration RATE (BPM)

When the training target changes from the waveform to a **summarized respiration
rate** (one averaged value per window), the evaluation switches to the two
literature-standard numbers, computed per window and pooled:

| Role | Metric | Definition |
|---|---|---|
| **Headline** | **MAE (BPM / breaths·min⁻¹)** | `mean(|rate_pred - rate_ref|)` in physical units — the universal reporting standard |
| Acceptance | **% within 10 %** | fraction of windows with `\|err\| ≤ 0.10 · rate_ref` (ICMI-2023 style; relative to the contact-sensor reading) |
| Supporting | RMSE (BPM), bias (BPM) | signed error / spread |

The rate is read from the band-limited FFT peak (`estimate_rate_bpm`,
0.1–0.6 Hz = 6–36 BPM, parabolic sub-bin). Because the metric is a rate, it is
**scale/offset-invariant by construction** — the voltage-space problem that
constrains the waveform metrics does not apply to it. Report the **window
length** with every number: a shorter window gives fewer cycles and a coarser
rate. Use the same window for prediction and reference.

Wired into `runners/run_evaluate_subject.py` as an optional block
(`--rate_window_s`, default 30 s; `--rate_tol`, default 0.10): per-session
`rate` blocks feed the per-subject rollup, and a pooled `rate_overall` block is
the headline.

## 5. Baselines a subject must beat

A reconstruction is only meaningful if it beats trivial competitors:

- **Constant / mean line** — Pearson 0 by construction (sanity floor).
- **Single-tone template** — a pure tone at the reference's own peak frequency
  (LS cos/sin fit; handed the true rate for free). This is the *best a
  rate-only model can do*; the model must score **above** it in a majority of
  sessions. (Evidence 2026-10-10: the Stage-3 model's 0.593 median tied this
  baseline's 0.600, winning only 49.1 % — i.e. currently NOT beaten.)
- **Off-face / global-motion control** — the frame-argmax finding (a chest/hair
  pixel drives the whole-frame gate). Any subject-level claim must survive
  regressing out the global/motion channel.

## 6. Aggregation & uncertainty

- **Per subject**: median of the subject's per-session metrics (robust to short,
  uninformative sessions; a 10 s session cannot show respiration).
- **Across subjects**: median + IQR, and a **bootstrap 95 % CI over subjects**
  (`bootstrap_ci`, `evaluation/metrics.py`) — each *subject* is one independent
  observation, clips within a subject are not.
- Report the **fraction of subjects whose headline beats the template baseline**.

## 7. Reporting artifacts

Per run (restricted to the test subjects):

- `subject_metrics.json` — `overall` (with CI), `per_subject`, `per_task`,
  `per_group`, `sessions`.
- `subject_figures/subject_<S>.png` — reconstructed vs reference for that
  subject's **median-quality** session (whole session + zoom).
- A one-line console summary: median band-limited Pearson [CI], n subjects,
  beat-template fraction.

## 8. Implementation map

| Piece | Location |
|---|---|
| band-pass + band-limited Pearson + template + bootstrap | `evaluation/metrics.py` |
| session assembly (Hann OLA), affine calibration | `evaluation/assemble.py` |
| subject-level runner | `runners/run_evaluate_subject.py` |
| reference reader / smoother (reused) | `runners/run_evaluate_session.py` |
| tests | `tests/test_evaluation_subject.py`, `tests/test_evaluation_breakdown.py` |

Command (from `code/`):

```bash
python runners/run_evaluate_subject.py --pred_dir OUT_DIR --waveform resp \
    --fs 100 --band 0.1,0.6 --test_subjects F001,F002,F003
```

## 9. Non-goals / open items

- Morphology metrics, per-breath period/amplitude — only if a morphology claim
  is made.
- Clinical HRV (Tier 3) is BP-only and out of scope for respiration.
- Statistical test *between* two models (e.g. std vs smoothl1) is a paired
  subject-level comparison — a follow-up, not part of the single-model report.
