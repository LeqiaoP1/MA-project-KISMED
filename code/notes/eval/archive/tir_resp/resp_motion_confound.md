# Does head motion explain the per-session spread of T3 respiration quality?

> Measured 2026-10-25 · probe `code/notes/eval/tir_resp/motion_vs_quality.py`
> Data: the existing 5-fold dumps (`output/finetune/resp_tir_roi_hpc_fold{0..4}`)
> + the `IRFeatures` landmark tracks. No GPU, no decode, no retraining.
> Figure: `resp_motion_vs_quality.svg` · rows: `output/eval/resp_kfold/
> {t3,all_tasks}_motion_vs_quality{,_per_clip}.json`
> Plain-English version: `code/notes/explanation/resp_t3_why_some_predictions_are_better.md`

## The question

T3 has the best mean quality of the ten tasks (+0.567 band-limited Pearson) and
yet a very wide per-session spread (worst T3 session -0.465, best +0.965). The
suspicion was that **head motion** drives it, because the shipped ROI crop is a
**whole-task union box** (`data/roi_cache.union_box`, min/max over every valid
frame + `roi_padding` 0.2): head motion is *not* compensated per frame, it
inflates a fixed crop whose extra area is static background. A recorded note on
the since-removed `DEFAULT_ROI_QUANTILE` asserted exactly that mechanism
("large/jerky ones sit at ~0.0", `hpc_precomputation.md` section 8).

## Verdict

**No. For T3 the motion hypothesis is not supported** -- the correlations are
indistinguishable from zero at both session and clip level, the quartile table is
flat, and the model's prediction does not track the head. Head motion *is* a real
degradation axis, but only in the tasks that have enough motion, and T3 is the
calmest task in the corpus, so T3's own spread must come from elsewhere.

## What was measured

Landmarks only (no pixels). Frame `f` of `IRFeatures/<S>_<T>.txt` is at
`t = f/25 s`; invalid frames follow production (`missing_frame_mask`, i.e. any
`(0,0)` sentinel or non-finite ROI landmark is dropped). Translations are in
**face radii** (divided by the session's median RMS landmark radius about the
centroid) so subjects at different distances are comparable.

| feature | definition |
|---|---|
| `band_rms` | RMS of the 0.1-0.6 Hz band-passed centroid path, face radii |
| `sway_rel` | RMS excursion of the detrended centroid, face radii |
| `speed_rel` | median per-frame centroid displacement, face radii / s |
| `jitter_rel` | `roi_box_impact.py`'s jitter (mean per-landmark temporal std of the 12 nose+mouth points) / face radius |
| `infl` | `task_lin / clip_lin` = whole-task union box side / median per-8 s-window box side; how much the FIXED crop is inflated |
| `band_frac` | share of detrended centroid variance in 0.1-0.6 Hz |
| `scale_std` | std of log(face radius) = lean toward / away from camera |

Model side, from the fold dumps: `r_pred_ref` (the reported metric, which this
probe reproduces **exactly** -- r = 1.0000, max |diff| = 0.0000 over all 1260
sessions), `r_pred_mot` (band-limited Pearson of the prediction against the head
centroid on the same timeline) and `r_ref_mot` (the physiological floor: chest
wall and head do move together a little, so `r_pred_mot` must be read against it).

**Hypothesis-B controls** (the competing explanation, "motion tasks just have
shallow, irregular breathing"): reference in-band amplitude `amp_ref`, session
duration, and `template_pearson_band` (how well the reference is fit by a pure
tone at its own peak frequency = how periodic the target is).

## Result 1 -- inside T3 nothing moves (n = 129 sessions)

Spearman against `pearson_band`:

| feature | rho | p | R² |
|---|---|---|---|
| `template_pearson_band` | **+0.349** | **0.0001** | 12.2 % |
| `amp_ref` | +0.172 | 0.05 | 3.0 % |
| `duration_s` | +0.163 | 0.065 | 2.7 % |
| `face_frac` | +0.142 | 0.11 | -- |
| `infl` | -0.124 | 0.16 | 1.5 % |
| `jitter_rel` | -0.093 | 0.29 | 0.9 % |
| `band_rms` | **-0.046** | **0.61** | **0.2 %** |
| `sway_rel` | -0.040 | 0.65 | 0.2 % |
| `band_frac` | +0.012 | 0.89 | -- |
| `speed_rel` | +0.071 | 0.42 | -- |
| `scale_std` | +0.107 | 0.23 | -- |

Partialling out duration and `amp_ref` leaves every motion term at zero
(`band_rms` -0.020, `sway_rel` -0.010, `jitter_rel` -0.070, `infl` -0.099).

Quintiles of `band_rms` -> median `pearson_band`: **+0.634 / +0.528 / +0.683 /
+0.550 / +0.607** -- no ordering at all. The only T3 variable that carries signal
is **how periodic the reference is**, not how much the head moves.

## Result 2 -- the model is not looking at the head

| | median r(pred, ref) | median r(pred, motion) | median r(ref, motion) |
|---|---|---|---|
| T3, low-motion half (n=65) | +0.544 | +0.023 | +0.033 |
| T3, high-motion half (n=64) | **+0.628** | -0.010 | +0.034 |

`|r(pred,motion)| - |r(ref,motion)|` has median -0.007 and is positive in
**46.5 %** of T3 sessions -- a coin flip (55.0 % over all tasks). And the
high-motion half of T3 is *better* predicted than the low-motion half, the
opposite of what a motion-corruption story predicts. So the prediction is not
riding a head-motion channel, and corruption is not visible where the motion is
largest.

## Result 3 -- per clip, T3 is a null result (4004 clips / 129 sessions)

The session number averages ~31 clips, so a local motion burst can be invisible.
Scoring each clip on its own 8 s window against its own motion:

| feature | rho | naive p | p with SESSION as the unit |
|---|---|---|---|
| clip `band_rms` | -0.064 | 5.5e-05 | **0.253** |
| clip `jitter_rel` | -0.025 | 0.11 | 0.53 |
| clip `speed_rel` | -0.010 | 0.53 | 0.83 |

The naive p is a fine example of pseudo-replication (clips of a session share
crop, subject and reference). Quintile medians of clip `band_rms`:
0.585 / 0.515 / 0.510 / 0.536 / **0.452** -- only the top quintile dips, and not
significantly. Directionally consistent with a threshold, not with a gradient.

### The null is not a metric artefact

T3, Spearman of each motion term against five different quality axes:

| metric | `band_rms` | `sway_rel` | `jitter_rel` |
|---|---|---|---|
| `pearson` (plain) | -0.065 (p .46) | -0.057 (p .52) | -0.111 (p .21) |
| `pearson_band` | -0.046 (p .61) | -0.040 (p .65) | -0.093 (p .29) |
| `mae_bpm` (error) | +0.024 (p .79) | +0.030 (p .74) | +0.063 (p .48) |
| `psd_mae` (error) | -0.009 (p .92) | -0.035 (p .69) | +0.019 (p .83) |
| `amp_rel_err` (error) | +0.060 (p .50) | +0.058 (p .51) | +0.122 (p .17) |

Every entry is within noise of zero; the error metrics even carry the expected
sign (motion -> worse) but at |rho| <= 0.12.

## Result 4 -- across tasks the effect is real, and T3 is the exception

Pooling all 1260 sessions: `band_rms` quartiles -> **+0.463 / +0.375 / +0.305 /
+0.212** (monotone), pooled rho -0.298. Because T3 is both the calmest and the
best task, the pooled number partly restates "which task is this". The
fixed-effects controls remove that:

| feature | pooled | within-task | within-task + subject |
|---|---|---|---|
| `band_rms` | -0.298 | **-0.219** | **-0.188** |
| `sway_rel` | -0.271 | -0.211 | -0.168 |
| `jitter_rel` | -0.247 | -0.192 | -0.135 |
| `speed_rel` | -0.202 | -0.135 | -0.092 |
| `infl` | +0.000 | -0.043 | +0.008 |
| `scale_std` | -0.149 | -0.080 | -0.090 |

So **given the same task and the same person, more in-band head motion still goes
with worse reconstruction** (rho ~ -0.19). Per-task, though, the effect is highly
uneven:

| task | median `band_rms` | median `pearson_band` | rho(`band_rms`, quality) |
|---|---|---|---|
| **T3** | **0.018** | **+0.567** | **-0.046** |
| T8 | 0.046 | +0.365 | -0.243 |
| T5 | 0.049 | +0.377 | -0.230 |
| T2 | 0.050 | +0.336 | -0.240 |
| T10 | 0.064 | +0.203 | -0.154 |
| T1 | 0.067 | +0.289 | -0.258 |
| T7 | 0.071 | +0.255 | -0.216 |
| T9 | 0.073 | +0.320 | -0.216 |
| T6 | 0.093 | +0.305 | -0.194 |
| T4 | 0.101 | +0.254 | -0.362 |

T3 is the **only task with no effect**, and it is the only task whose motion
sits almost entirely below the corpus's first quartile: **80/129 T3 sessions are
below the pooled Q1 (0.0235 face radii), only 20 exceed the pooled median and
only 7 exceed Q3 (0.0782)**. T3 simply does not enter the regime where motion
costs accuracy -- which is also the most economical reading of "T3 is the best
task".

## The recorded 2026-09-26 claim does not reproduce for T3

The `roi_quantile` note said small-box / low-jitter sessions reconstruct at
Pearson ~0.7 while large/jerky ones sit at ~0.0. At Stage-3, T3's worst `infl`
quartile is **+0.432** and worst `jitter_rel` quartile **+0.476** -- degraded,
not destroyed, and `infl` has no fixed-effects signal at all (+0.000 pooled,
-0.043 within task). The note described Stage-2 masked *reconstruction*; it does
not transfer to the Stage-3 downstream task as a correlation.

## Two counter-examples that make the null concrete

| session | `pearson_band` | `band_rms` | `sway_rel` | `jitter_rel` | `infl` | `amp_ratio` | `template_pearson_band` |
|---|---|---|---|---|---|---|---|
| M038_T3 (**calmest** in T3) | **-0.260** | 0.007 | 0.019 | 0.024 | 1.06 | 0.23 | 0.754 |
| M055_T3 (3rd most restless) | -0.009 | 0.058 | 0.119 | 0.136 | 1.47 | 0.001 | 0.368 |
| M042_T3 (**most restless** in T3) | **+0.280** | 0.112 | -- | -- | -- | -- | -- |

The **least-moving** T3 session is one of the worst, and its failure is an
amplitude collapse (`amp_ratio` 0.23: the prediction is ~4x too small) against a
*very* periodic reference (template 0.754) with a tiny face (47.99 px radius).
Meanwhile the **most restless** T3 session (4x the median in-band motion) is
merely unremarkable at +0.280. Motion cannot be the common cause: the ordering
is close to inverted.

## Caveats

- The motion features are 2-D landmark statistics at 25 fps: they capture rigid
  translation, scale and landmark jitter, **not** out-of-plane rotation, local
  facial deformation, or non-rigid skin motion. A session that rotates in depth
  is only partly captured (via `scale_std`, which is weak too).
- This measures **association with the reported metric** (band-limited Pearson,
  phase-dominated). A motion effect on morphology or amplitude could exist
  without moving `pearson_band`; `amp_ref`/`amp_ratio` are available in the same
  rows if that is the question.
- One session per subject per task, so n = 129 independent units for T3 and 1260
  overall; the per-clip p-values are reported with the session as the
  exchangeable unit for that reason.
- Unrelated but worth recording: `M038_T3` scores **+0.947** in the single-run
  dump (`output/finetune/resp_tir_roi_hpc/session_metrics.json`) and **-0.260**
  in the 5-fold dump. Those are different models (per-fold Stage-2 encoders), so
  the earlier "M038_T3 is genuinely impressive" reading refers to the single run
  only and is not a cross-validated number.

## What this implies for the next step

1. **Do not spend the next experiment on motion mitigation for T3.** Per-clip
   boxes / `roi_landmarks: nostrils` target exactly the mechanism measured here,
   and at Stage-3 it has no T3 signal.
2. **The T3 spread lives on the reference side.** `template_pearson_band`
   (+0.349) is the only significant correlate, together with amplitude and
   duration -- and all three are weak (R² <= 12 %). That is consistent with the
   earlier verdict that the model carries about one dominant frequency: where the
   true respiration is not periodic, there is nothing stable to lock onto.
3. If motion is still to be tested causally, do it on a **high-motion task**
   (T4: rho -0.362, or T1/T2/T8) where the effect is 5-8x larger; T3 is a
   low-power venue for that experiment.

## Reproduce

```
cd code
../.env/bin/python notes/eval/tir_resp/motion_vs_quality.py --task T3 --per_clip \
    --out output/eval/resp_kfold/t3_motion_vs_quality.json
../.env/bin/python notes/eval/tir_resp/motion_vs_quality.py --all_tasks \
    --out output/eval/resp_kfold/all_tasks_motion_vs_quality.json \
    --svg code/notes/eval/tir_resp/resp_motion_vs_quality.svg
```

CPU-only, ~13 s for T3, ~70 s for all 1260 sessions on the login node.
