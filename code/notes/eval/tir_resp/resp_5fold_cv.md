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
