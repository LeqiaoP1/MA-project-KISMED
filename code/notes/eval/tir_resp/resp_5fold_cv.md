# Respiration Stage-3: pooled 5-fold CV evaluation (all BP4D+ subjects)

> Measured 2026-10-23 · protocol `code/notes/design/Evaluation_RespWaveform_protocol.md` §3.2
> Run dirs `output/finetune/resp_tir_roi_hpc_fold{0..4}` · artifact
> `output/eval/resp_kfold/kfold_resp_tir_roi_hpc.json`

## What was measured

The 5-fold subject-disjoint sweep is complete, so every subject is predicted
exactly once -- by the model that never trained on it. Pooling the five folds
therefore gives the **cross-validated** number for the whole corpus:

| item | value |
|---|---|
| subjects / sessions / clips | **138 / 1260 / 21 231** |
| fold test blocks (`utils/splits.py::subject_folds`) | 27 / 28 / 28 / 28 / 27 subjects |
| subject partition check | disjoint, every subject in exactly one fold (runner asserts it) |

Reproduce with `code/runners/run_evaluate_kfold.py` (from `code/`):

```
../.env/bin/python runners/run_evaluate_kfold.py --root ../output/finetune \
    --run_prefix resp_tir_roi_hpc_fold --folds 0,1,2,3,4 \
    --out ../output/eval/resp_kfold/kfold_resp_tir_roi_hpc.json
```

Per session: clip predictions are Hann-overlap-added into one continuous
waveform (`evaluation/assemble.py`), sliced to the covered span, and scored
against the raw `Resp_Volts` reference.

## The three metric families

| family | definition | space |
|---|---|---|
| **pearson** | band-limited Pearson (0.1-0.6 Hz, brick-wall FFT) = headline; plain Pearson for contrast; single-tone **template** baseline | scale/offset invariant |
| **amplitude** | `amp_ratio` = in-band RMS(pred)/RMS(ref), `amp_rel_err` = \|ratio-1\| | **ORACLE-calibrated** (per-session `affine_calibrate` uses the true session scale -> not deployable) |
| **rate** | dominant in-band FFT peak per 30 s window -> MAE (BPM), % within 10 % | scale invariant |

Aggregation: one number per subject (median over that subject's sessions),
then median + bootstrap 95 % CI **over the 138 subjects**; the rate block is
also pooled over every window of every fold.

## Results

| metric (per-subject median) | value | 95 % CI |
|---|---|---|
| **pearson (band-limited 0.1-0.6 Hz)** | **+0.307** | [+0.252, +0.350] |
| single-tone template baseline | +0.535 | [+0.521, +0.547] |
| subjects beating the template | **18 %** | -- |
| **amplitude rel. error** (oracle) | **0.682** | [0.634, 0.720] |
| amplitude ratio (pred/ref, oracle) | 0.334 | -- |
| **rate MAE** | **4.64 BPM** | [4.11, 5.28] |
| rate, pooled 30 s windows | MAE 5.67 BPM, RMSE 8.02, bias +1.35, within 10 % **34.4 %** (n = 1726) | -- |
| supporting | mae_affine 1.03 V, psd_mae 0.091 | -- |

Session-level (n = 1260): band-limited Pearson median +0.326 (p10 -0.107,
p90 +0.714); **30.5 %** of sessions > 0.5 and **83.9 %** > 0.
Per-subject spread -0.154 (M054) .. +0.786 (M007).

### Per fold (session medians)

| fold | subjects | sessions | pearson (plain) | pearson_band | amp_rel_err | rate MAE (BPM) |
|---|---|---|---|---|---|---|
| 0 | 27 | 238 | +0.310 | +0.371 | 0.593 | 4.73 |
| 1 | 28 | 248 | +0.111 | +0.150 | 0.796 | 6.87 |
| 2 | 28 | 256 | +0.273 | +0.339 | 0.670 | 4.24 |
| 3 | 28 | 255 | +0.422 | +0.506 | 0.485 | 3.32 |
| 4 | 27 | 263 | +0.261 | +0.309 | 0.686 | 4.05 |

The plain-Pearson column reproduces each fold's own `session_metrics.json`
median exactly, i.e. this runner agrees with `run_evaluate_session.py`.

> **CAVEAT on the per-fold column (measured 2026-10-25).**
> `utils/splits.py::subject_folds` is `np.array_split(sorted(subjects), n_folds)`
> -- **contiguous blocks of the sorted subject ID list**. Sorted IDs are all `F*`
> then all `M*`, so the fold test sets are **F001-F028 / F029-F056 /
> F057-M002 / M003-M030 / M031-M058**: two folds 100 % female, two 100 % male.
> Gender (and any acquisition-order/batch effect) is therefore perfectly
> confounded with fold, and the per-fold spread above is a **cohort** spread, not
> a model-variance estimate. The pooled numbers (1260 sessions, subject-disjoint)
> are unaffected. Detail + the fix: `resp_t3_spread_sources.md` section 2.

### Per task / per group (medians)

| task | pearson_band | amp_rel_err | rate MAE | n |
|---|---|---|---|---|
| **T3** | **+0.567** | **0.469** | **1.08** | 129 |
| T5 | +0.377 | 0.622 | 4.04 | 129 |
| T8 | +0.365 | 0.604 | 5.01 | 131 |
| T2 | +0.336 | 0.667 | 4.31 | 129 |
| T9 | +0.320 | 0.700 | 5.76 | 133 |
| T6 | +0.305 | 0.680 | 5.90 | 129 |
| T1 | +0.289 | 0.644 | 4.95 | 124 |
| T7 | +0.255 | 0.755 | 4.92 | 125 |
| T4 | +0.254 | 0.745 | 5.01 | 115 |
| **T10** | **+0.203** | **0.699** | 4.39 | 116 |

Groups: **low** +0.447 / 0.577 / 2.36 BPM (n=258) > **high** +0.323 / 0.668 /
5.34 (n=515) > **moderate** +0.272 / 0.694 / 4.91 (n=487).

### Grouping v2 (2026-10-24, re-aggregated -- NOT a new run)

The shipped tiers changed: `low=[T2,T3,T5]`, `moderate=[T1,T4]`,
`high=[T6,T7,T8,T9,T10]` (T5 high->low, T1 high->moderate, T10
moderate->high; `data/task_groups.SHIPPED_TASK_GROUPS`, rationale in
`code/notes/design/TIR_ROI_RESP_TasksDistortion.md`). The table above is the
v1 split and is kept as measured. Re-aggregating the **same** five fold dumps
with `runners/run_evaluate_kfold.py --task_groups '...'` (no re-training; output
`output/eval/resp_kfold/kfold_resp_tir_roi_hpc_regrouped.json`, which records
`protocol.task_group_source: cli`) gives:

| group | tasks | pearson_band | amp_rel_err | rate MAE (BPM) | n |
|---|---|---|---|---|---|
| low | T2, T3, T5 | **+0.411** | 0.592 | **2.92** | 387 |
| moderate | T1, T4 | +0.278 | 0.690 | 5.01 | 239 |
| high | T6, T7, T8, T9, T10 | +0.291 | 0.685 | 5.18 | 634 |

Only the per-GROUP rows change under the regrouping; the corpus, the
checkpoints and the per-subject/per-task values are identical (grouping is
an aggregation). The ordering is unchanged (`low > high > moderate`) but the
low lead narrows (+0.447 -> +0.411) because T5 joins it while T1 leaves `high`
and T10 joins it -- a **composition** effect, not a model change.

Figures for this section (v2 = `_v2` in the filename):
`resp_kfold_task_group_medians_v2.svg` (per-task and per-group medians, rendered
by `plot_task_group_medians.py` from the regrouped JSON -- task rows in T1..T10
order, `low`-tier tasks in bold, header listing the tier definition),
`resp_kfold_task_scatter_v2.svg` (per-task Pearson vs rate MAE, colour = oracle
amplitude error, dot ring = distortion tier with an in-plot key; no caption
below the axes, `plot_task_scatter.py`) and
`resp_T3_waveforms_v2.svg` (representative T3 waveforms,
`plot_task_waveforms.py --no-footers`, i.e. no in-figure caption at all). The
unsuffixed SVGs in this folder are the v1 record and still show the old group
numbers.

## Verdict

1. **The rhythm is tracked weakly, and only weakly.** +0.307 per-subject
   band-limited Pearson over 138 held-out subjects is far from a
   reconstruction: 1 in 6 sessions still correlates negatively once drift is
   removed, and only 30 % clear 0.5.
2. **The model still loses to a single pure tone.** Handed the reference's own
   peak frequency, the template scores +0.535 and the model beats it in only
   **18 %** of subjects. This reproduces the 2026-10-10 finding (49 % on the
   old val split, 0.16-0.4 Hz band) at corpus scale: the model's information
   content is roughly "one dominant frequency", not morphology.
3. **Amplitude is ~3x too small.** After the oracle per-session calibration the
   in-band prediction carries only 33 % of the reference's breathing depth
   (relative error 0.68) and no subject reaches a 25 % error except 8 % of
   sessions. Under-scaled in-band output is exactly what the earlier `std` vs
   `smoothl1` A/B predicted (the model is smooth/over-damped).
4. **Rate error is large in physical units.** 4.6 breaths/min per subject
   (5.7 pooled) with a +1.35 BPM bias and only 34 % of 30 s windows within
   10 %. `dominant_freq_error_hz` looks tiny (0.08 Hz median) only because it
   is floored at one FFT bin -- read the BPM numbers instead.
5. T3 remains the one task where all three families agree (+0.567 / 0.469 /
   1.08 BPM), and `low` remains the best group. Both replicate the per-fold
   finding, so they are not a cohort artifact.

## Caveats

- **140 subjects in the tree, 138 in the evaluation.** `M049` has no
  `IRFeatures/*.txt` at all, so there are no landmarks and no ROI; `F004` has
  landmarks but its respiration channel is saturated at the -10 V recorder rail
  (15-80 % of samples per task), so **all 168** of its candidate 8 s windows trip
  `rail_touch_v: 9.9`. Neither subject yields a single clip. Only 86 of the 138
  subjects keep all ten tasks; the rest lose sessions to the same rail / spread /
  `(0,0)`-sentinel rules, which is why 138 subjects give 1260 sessions rather
  than 1380.
- **Fold = cohort.** The folds are contiguous blocks of sorted subject ids
  (F001-F028, F029-F056, F057-F082+M001-M002, M003-M030, M031-M058), so
  fold 3's lead (+0.506) is confounded with sex/recording cohort. Never read
  the fold spread as model quality.
- **Amplitude is oracle-calibrated**: the affine map is fitted with the
  reference in hand. It is the honest "how far off is the breathing depth"
  number, but it is not a deployable measurement.
- **Band differs from the earlier investigation**: 0.1-0.6 Hz here (protocol
  default) versus 0.16-0.4 Hz for the 2026-10-10 median 0.593. The wider band
  admits more drift, so do not compare those two numbers directly.
- `within_10pct`'s per-session median is 0 because most sessions yield only
  1-2 windows; quote the **pooled** window number (34 %) for rate acceptance.
- `fit`/`mae_affine` are in raw volts and are not comparable to the
  session-z-scored per-clip `mae` reported in `metrics_final.json`.
