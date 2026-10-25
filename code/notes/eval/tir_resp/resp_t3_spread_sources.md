# What explains the T3 per-session spread? A variance decomposition

> Measured 2026-10-25 · probes `t3_spread_explained.py` (per-session axes) and
> `roi_contrast_scan.py` (thermal appearance of the model's input crop)
> Rows: `output/eval/resp_kfold/{t3_spread_explained,roi_contrast_all}.json`
> Follows `resp_motion_confound.md`, which ruled head motion out.
> Plain-English version: `code/notes/explanation/resp_t3_why_some_predictions_are_better.md`

## Short answer

**One number -- how well the SAME subject is reconstructed on their OTHER nine
tasks -- explains 43 % of the T3 spread on its own, and nothing else adds to it.**
The T3 spread is overwhelmingly a *who-is-this-subject* effect, not a
T3-session effect. A second, partly independent factor is that the 5 CV folds are
**contiguous blocks of the sorted subject IDs**, so they are almost perfectly
confounded with gender (two folds are 100 % female, two 100 % male) -- the
apparent "fold/model" effect is mostly a cohort effect.

| axis | LOOCV R² of T3 quality | verdict |
|---|---|---|
| **subject trait** (`other_tasks_quality`) | **+0.430** | dominant |
| + gender | +0.425 | no gain (gender IS the trait) |
| + subject thermal appearance | +0.424 | no gain |
| + reference regularity/SNR | **+0.455** | ~+2 points, at the edge of noise |
| + head motion | +0.429 | no gain (already known) |
| fold dummies alone | +0.112 | real but confounded with gender |
| reference features alone | +0.095 | weak |
| **head motion alone** | **~0.00** | ruled out |

(LOOCV on rank-transformed variables, intercept included; a negative value would
mean "no better than the mean".)

## 1. The dominant axis: the subject, measured on other tasks

`other_tasks_quality` = the median band-limited Pearson of the same subject over
their other nine tasks (each predicted by the model that held that subject out).

* **rho = +0.667, p = 6e-18, n = 129.**
* Additive two-way model over ALL 1260 cells: subject dummies **R² 0.360** vs
  task dummies **0.086** (both 0.446) -- subject identity is 4.2x the task.
* Split-half on disjoint task sets (odd T1/T3/T5/T7/T9 vs even
  T2/T4/T6/T8/T10): Spearman **+0.693**, Spearman-Brown reliability **+0.819**
  (n 137). The person effect is reproducible, not a session artefact.
* Quartiles of `other_tasks_quality` -> median T3: **+0.219 / +0.500 / +0.582 /
  +0.790** (a 0.57 spread predicted without reading the T3 session).
* Evidence script + figure: `subject_effect_evidence.py`,
  `resp_t3_subject_effect.svg`.

### Model specification (the 36 % vs 8.6 % claim)

$$y_{ij} \;=\; \mu \;+\; \alpha_i \;+\; \beta_j \;+\; \varepsilon_{ij}$$

* $y_{ij}$ = band-limited Pearson of subject $i$ on task $j$; $i = 1..138$,
  $j = 1..10$.
* $\alpha_i$ = subject offset, $\beta_j$ = task offset, $\varepsilon_{ij}$ =
  residual. **No interaction term** -- one session per cell makes
  subject×task unidentifiable, so any genuine interaction sits in the residual.
* Fitted by ordinary least squares on dummy indicators: 1 intercept + 137 subject
  dummies + 9 task dummies = **147 parameters**, design matrix rank 147 (full).
  The first level of each block is absorbed into $\mu$, so $\alpha$ and $\beta$
  are offsets **relative to F001 and T1**, not to the grand mean.
* Data: **1260 of 1380 cells observed (91.3 % complete)**. The 120 missing cells
  are the sessions dropped by the corpus quality gates, so they are not missing
  at random -- but at 91 % completeness the imbalance is mild, and the two dummy
  blocks stay nearly orthogonal (**max |corr| between any subject dummy and any
  task dummy = 0.037**), which is why the unique shares equal the marginal ones.
* Fit: $R^2$ = 0.360 (subject), 0.086 (task), 0.446 (both); residual SD 0.247
  against a total SD of 0.312, residual df 1113. **Adjusted** $R^2$ = 0.282 /
  0.079 / 0.374, so the subject share is not an artefact of its 137 parameters.
* Partial F tests (each block dropped from the full model):
  subject | task $F(137, 1113) = 5.29$, $p = 1.7\times10^{-57}$;
  task | subject $F(9, 1113) = 19.24$, $p = 3.7\times10^{-30}$. Both effects are
  real; the question is only their relative size.
* Fitted values: intercept +0.329, subject offsets SD 0.190 spanning
  **-0.495 .. +0.354** (a 0.85 range on a metric whose total SD is 0.31), task
  offsets T2 +0.085, T3 +0.268, T4 -0.010, T5 +0.103, T6 +0.041, T7 -0.002,
  T8 +0.103, T9 +0.075, T10 -0.100.
* Residuals are mildly left-skewed (skew -0.49, excess kurtosis +0.63); $R^2$
  needs no normality, but a rerun with a rank transform gives the same ordering.
* **What the subject block really contains:** each subject was scored by exactly
  one fold's model, and folds are subject-disjoint, so $\alpha_i$ mixes "this
  person is readable" with "this fold's encoder is good". Adding fold dummies
  alone gives $R^2$ 0.108 and adds nothing to the subject block (nested, by
  construction). The design cannot separate the two; the split-half test below
  is what shows a genuine person component exists.
* Task-by-task, T3 correlates with the same subject's quality on:
  T2 **+0.606**, T9 **+0.574**, T8 **+0.528**, T7 +0.463, T5 +0.431, T6 +0.423,
  T1 +0.345, T4 +0.340, T10 +0.216. Every task is positive.
* It dominates every per-session variable, and no combination of them improves on
  it -- i.e. it is not a proxy for something else we can measure.

**The trait is NOT breathing.** Measured on the *other* tasks (so it is not the
same session again), a subject's median

| subject-median (other tasks) | rho vs T3 quality |
|---|---|
| `template_pearson_band` (how periodic their breathing is) | **-0.020** (p 0.82) |
| `amp_ref` (breath depth) | +0.098 (p 0.27) |
| `band_rms` (head motion) | -0.082 (p 0.36) |
| `jitter_rel` | -0.049 (p 0.58) |
| `face_scale_px` | +0.169 (p 0.056) |
| `roi_sat_frac` (palette saturation in the nostril box) | **+0.317** (p 2.5e-4) |

Note the contrast: *same-session* `template_pearson_band` is a real T3 correlate
(+0.349), but a subject's *typical* periodicity across tasks is not (-0.020). So
reference regularity is a **session** property, while T3 quality is a **person**
property. They are different axes that happen to overlap a little.

**The trait is not spatial contrast.** `roi_g_std` and `crop_g_std` are the most
subject-stable measurements in the whole set (within-subject CV of the same
subject across ten tasks: **9.7 %** and **7.8 %**, trait index 0.99) -- and they
predict T3 quality at rho +0.03 / -0.02. Stability is necessary, not sufficient.

**What does index it, weakly:** the fraction of the nostril box sitting at the
top of the palette (`roi_sat_frac`, +0.317 from other tasks, +0.377 same
session). More saturated -> better. It buys ~10 points of R² over the reference
baseline, and nothing over the trait. Interpretation is not settled: it may be a
hot face (higher skin temperature -> more thermal signal), a tighter crop, or a
proxy for the subject's whole recording session.

## 2. The fold "model effect" is really a cohort effect

`utils/splits.py::subject_folds` is `np.array_split(sorted(subjects), n_folds)`
-- **contiguous blocks of the sorted ID list**, not a random or stratified
partition. On this corpus, sorted IDs are all `F*` then all `M*`, so:

| fold | test subjects | F | M | T3 median `pearson_band` |
|---|---|---|---|---|
| 0 | F001-F028 | 28 | 0 | +0.652 |
| 1 | F029-F056 | 28 | 0 | **+0.305** |
| 2 | F057-M002 | 26 | 2 | +0.534 |
| 3 | M003-M030 | 0 | 28 | **+0.767** |
| 4 | M031-M058 | 0 | 27 | +0.612 |

Kruskal-Wallis across folds p < 1e-4; fold dummies alone give LOOCV R² +0.112.
The 0.46 gap between fold 1 and fold 3 is therefore **not** a clean reading of
"model quality": gender is 100 % confounded with fold, and so is any
acquisition-order/batch effect.

The gender gap itself:

* T3: **F +0.484 (n = 74) vs M +0.714 (n = 55)**, Mann-Whitney **p = 0.0028**
  (0.76 SD).
* Corpus-wide (all 1260 sessions): F +0.254 vs M +0.407, **p = 2.4e-12** -- so
  this is not a T3 curiosity.
* It is a special case of the subject trait, not an independent effect:
  partial rho(gender, T3 quality) goes **-0.264 -> -0.096** once
  `other_tasks_quality` is controlled.
* **It is not the reference and not the motion.** Across genders the reference
  is statistically identical: `template_pearson_band` 0.548 vs 0.534 (p 0.66),
  `band_snr` p 0.41, `amp_cv` p 0.41, `amp_ref` p 0.84, `rate_bpm` p 0.46,
  `duration_s` 74.24 vs 74.28 s (p 0.70), and the motion terms `band_rms` p 0.89
  and `jitter_rel` p 0.55. Where the genders DO differ: `roi_sat_frac`
  0.605 vs 0.702 (p 0.008), `roi_thermal_band` 1.89 vs 2.36 (p 0.012),
  `face_scale_px` 49.5 vs 51.6 (p 5e-5), and the model's amplitude error
  `amp_rel_err` 0.549 vs 0.320 (p 0.006).

**Actionable consequence:** per-fold numbers in `resp_5fold_cv.md` are 5
*different cohorts*, not 5 replicates, so their spread cannot be read as model
variance. If the fold split matters for a conclusion (it does for hyper-parameter
comparisons), switch `subject_folds` to a stratified/interleaved assignment
(e.g. round-robin over the sorted list, or stratify on gender + n_sessions).
That is a one-function change and it does not invalidate the pooled 1260-session
numbers, only the per-fold column.

## 3. Reference-side variables: real but small

Same-session Spearman vs T3 quality:

| feature | rho | p | meaning |
|---|---|---|---|
| `amp_cv` | -0.352 | 4e-5 | breath-to-breath depth variability hurts |
| `template_pearson_band` | +0.349 | 5e-5 | periodicity helps |
| `band_snr` | +0.324 | 2e-4 | in-band power / total detrended power |
| `drift_ratio` | -0.324 | 2e-4 | slow drift swamping the band hurts |
| `period_jitter` | -0.238 | 7e-3 | irregular rhythm hurts |
| `amp_ref` | +0.172 | 0.05 | deeper breathing helps |
| `duration_s` | +0.163 | 0.065 | longer session, more clips |
| `rate_bpm` | -0.021 | 0.81 | **rate does not matter** |
| `cycles` | -0.013 | 0.88 | **number of breaths does not matter** |

`amp_cv`, `band_snr` and `drift_ratio` are partly the same statement (a clean
periodic breath against a small drift), so the 8 reference variables together buy
only LOOCV R² **+0.095**, and **+0.025** on top of the subject trait.

## 4. Ruled out

* **Head motion** -- all terms ~0 (`band_rms` -0.046, `jitter_rel` -0.093,
  `infl` -0.124; see `resp_motion_confound.md`).
* **Crop geometry** -- `task_lin` +0.057, `clip_lin` +0.157 (weak, and gone once
  duration is controlled), `infl` -0.124, `face_px_in_input` +0.142.
* **Rate** -- `rate_bpm` and `cycles` are flat; consistent with the earlier
  finding that the model only carries one dominant frequency, so *which*
  frequency it is does not matter.
* **Clip count / coverage** -- `n_clips` +0.075, `valid_frac` +0.042.
* **Spatial image contrast** -- `crop_g_std` -0.020, `roi_g_std` +0.032.

## 5. What this means

1. Treat **T3 as a per-subject measurement, not a per-session one.** Two thirds of
   the T3 ranking is already decided by who the subject is; a subject's other
   tasks predict their T3 result better than anything measurable in the T3
   session itself. This is the quantitative justification for the
   subject-as-unit protocol already in `Evaluation_RespWaveform_protocol.md`.
2. The remaining ~55 % is mostly **model + noise**: 5 folds, 5 different Stage-2
   encoders, and no measured session property that explains it. Improving the T3
   *mean* therefore means improving the encoder, not filtering sessions.
3. **Fix the fold split before drawing any per-fold conclusion** (see section 2).
   The current contiguous split makes the per-fold column a gender/batch
   comparison.
4. If the residual subject trait is to be chased, the only lead here is the
   palette saturation / in-band thermal modulation of the nostril box -- and the
   earlier gate work says that signal is weak corpus-wide. The expected yield is
   low; the encoder is the better lever.

## Reproduce

```
cd code
../.env/bin/python notes/eval/tir_resp/t3_spread_explained.py \
    --out output/eval/resp_kfold/t3_spread_explained.json
../.env/bin/python notes/eval/tir_resp/roi_contrast_scan.py --workers 24 \
    --stride 10 --out output/eval/resp_kfold/roi_contrast_all.json
```

`t3_spread_explained.py` is ~30 s (it also reads the ROI cache for the T3
shards); `roi_contrast_scan.py` scans 1385 shards in **16 s** on 24 workers, so
the thermal-appearance axis can be re-measured for the whole corpus cheaply.
`subject_effect_evidence.py` re-derives the section-1 evidence and the figure in
under a second.
