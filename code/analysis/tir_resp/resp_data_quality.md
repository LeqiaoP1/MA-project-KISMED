# Respiration data quality — whole-corpus survey (TIR-ROI + RESP)

**Date:** 2026-10-06
**Scope:** every session in the raw BP4D+ tree (`$WORK_PROJ/test` =
`/work/projects/l0003511/test`), for the **thermal-ROI → respiration** pipeline
(`data_set: tir_roi`, streams `tir,resp`).
**Cluster:** Lichtenberg — CPU-only, no GPU, frames never decoded.

This is the reference report for "is the RESP target clean enough to
pre-train on?". It complements the per-clip figure inspector
(`runners/run_inspect_tir_resp.py`) with corpus-wide statistics.

---

## 1. How it was produced (reproducible)

```bash
# from code/  (login node; compute nodes have no internet)
mkdir -p logs
WORK_PROJ=/work/projects/l0003511 WORK_SCRATCH=/work/scratch/ne95ocyg \
  SURVEY_WORKERS=16 SURVEY_OUTPUT_DIR=/work/scratch/ne95ocyg/tir_resp_survey \
  sbatch -p deflt_short -t 00:25:00 -c 16 scripts/hpc/submit_survey_tir_resp.sbatch
```

| | |
|---|---|
| runner | `runners/run_survey_tir_resp.py` |
| job script | `scripts/hpc/submit_survey_tir_resp.sbatch` (`deflt_short`, 16 workers, ~2 min) |
| Slurm job | **55340713** (first attempt 55340576 was cancelled while queued on `deflt`) |
| outputs | `$WORK_SCRATCH/tir_resp_survey/tir_resp_survey.json`, `…_sessions.csv` |
| logs | `code/logs/survey_tir_resp_55340713.{out,err}` |

**Method.** The survey wraps the production dataset
`data/tir_resp_dataset.BP4DPlusTIRRespDataset` (so the *same* discovery and
usability gates apply) built per subject with `task_set=None` (all tasks),
`input_size=112`, `roi_padding=0.2`, `min_signal_spread=0.0` (keep every window;
the rail is classified afterwards). The "rail" is a sample `<= -9.999 V`.

**Task levels** use the `stage2_local_tir_roi_resp.yaml` / HPC config
convention: `low = {T2,T3}`, `moderate = {T4,T7,T8,T10}`,
`high = {T1,T5,T6,T9}`.

---

## 2. Coverage & usability

| | count |
|---|---|
| sessions discovered | **1400** (140 subjects × 10 tasks) |
| **sessions usable** | **1353 (96.6 %)** |
| sessions skipped | **47 (3.4 %)** |
| 8 s clips yielded | **6718** |

### Skip reasons (47)

| reason | n | detail |
|---|---|---|
| `missing_ir_features` | 15 | no `IRFeatures/<S>_<T>.txt` track |
| `all_clips_dropped` | 25 | every window contains an IR `(0,0)` sentinel line (1–4 windows) |
| `resp_too_short` | 6 | fewer than 8000 samples (< 8 s) for the 8 s window |
| `missing_resp_volts` | 1 | no `Resp_Volts.txt` |

**Whole-subject gaps (not random):**

* `missing_ir_features` (15): `F016_T2`, `F016_T3`, `F016_T4`, `F054_T10`,
  `M045_T2`, and **all 10 tasks of `M049`** (`T1…T10`).
* `resp_too_short` (6): `F010_T10` (6676 smp), `F029_T4` (5920), `F030_T4`
  (6440), `F042_T7` (2114), `F061_T4` (5280), `F081_T10` (4998).
  `missing_resp_volts`: `F082_T1`.
* `all_clips_dropped` spans many subjects (max 2 sessions for `F032`, `F043`,
  `F076`, `M029`); `F001_T8` is one example (see §6).

---

## 3. ⚠️ Respiration rail at −10 V — the main finding

The `Physiology/<S>/<T>/Resp_Volts.txt` channel **saturates at exactly
`-10.0000 V`** (the corpus' dead-channel floor), which **clips the respiration
troughs**. The plot below (F001_T10, `run_inspect_tir_resp.py`) shows ~37 % of
the 8 s window pinned to a flat −10 V line.

### 3.1 Whole-file rail fraction (1353 usable sessions)

| rail % | sessions | share |
|---|---|---|
| 0 % (clean) | 1089 | 80.5 % |
| 0–1 % | 36 | 2.7 % |
| 1–5 % | 84 | 6.2 % |
| 5–10 % | 36 | 2.7 % |
| 10–25 % | 50 | 3.7 % |
| 25–50 % | 25 | 1.8 % |
| 50–90 % | 28 | 2.1 % |
| > 90 % | 5 | 0.4 % |

Mean file rail **3.36 %**, median **0 %**.
**19.5 %** of sessions have any rail, **8.0 %** > 10 %, **2.4 %** > 50 %.

### 3.2 Impact on the clips that would be trained on (6718)

| clip severity | clips | share |
|---|---|---|
| clean (0 % railed) | 5994 | **89.2 %** |
| mild (0–10 %) | 233 | 3.5 % |
| moderate (10–50 %) | 318 | 4.7 % |
| severe (> 50 %) | 173 | **2.6 %** |
| 100 % railed | 34 | 0.5 % |
| flat (`max−min < 0.01 V`) | 36 | 0.5 % |

> **Key point.** The config knob `min_signal_spread: 0.01` drops **only** the
> fully-railed/constant windows (≈ 1 % of clips). The ~7 % of clips with 10 % →
> > 50 % clipping **pass silently** into pre-training, where their troughs are
> physically absent and no model can predict them (this inflates MAE/RMSE and
> weakens the periodic gradient).

### 3.3 Sessions where RESP is effectively unusable (16)

Every clip is > 50 % railed:

```
F024_T8  100.0% (7 clips)     M003_T1   92.0% (4)     F049_T6  70.3% (6)
F061_T5  100.0% (3)           M037_T2   88.1% (3)     F004_T4  69.2% (2)
F011_T1   98.6% (2)           F004_T3   78.5% (9)     F004_T10 61.2% (1)
F038_T3   96.4% (9)           F004_T7   75.7% (4)     F077_T10 60.4% (1)
                              M018_T2   74.0% (3)     F038_T5  56.4% (3)
                                                       M005_T4  56.3% (1)
                                                       F043_T4  52.3% (1)
```

Note `F004` is affected on **four** tasks (`T3`, `T4`, `T7`, `T10`) — a
subject-level sensor problem, not a task-level one.

---

## 4. Structural consistency

| check | result |
|---|---|
| measured fps deviates > 5 % from 25 fps | **0 sessions** — the frame↔time mapping is safe |
| `n_vid != n_ir` (video frames vs IRFeatures lines) | **22 sessions**, all `n_vid == n_ir + 12` (constant tracker offset; the dataset already uses `min()`) — e.g. `F011_T1` 519 vs 507, `F011_T2` 653 vs 641 |
| recording length (resp samples) | min 8000 · p10 17 249 · **median 34 818** · p90 75 208 · max 206 083 |
| sessions giving only 1 clip | **141** (95 of them < 2 clips) → short recordings thin the corpus |

**Clips per session:** 1 → 141 sessions · 2–4 → 658 · 5–9 → 438 · 10+ → 116.

---

## 5. By distortion level and by task

Levels (config convention):

```
low       sessions 275/280   clips 1619   clips_with_rail 138   mean file rail 3.31%
moderate  sessions 530/560   clips 1705   clips_with_rail 174   mean file rail 3.05%
high      sessions 548/560   clips 3394   clips_with_rail 412   mean file rail 3.68%
```

Rail prevalence is **roughly equal across levels** → the −10 V clipping is a
sensor/recording artifact, **not** a head-motion ("distortion level") effect.

Per task (used / discovered, clips, clips with any rail):

| task | used/discovered | clips | clips w/ rail |
|---|---|---|---|
| T1 | 133/140 | 443 | 78 |
| T2 | 137/140 | 406 | 52 |
| T3 | 138/140 | 1213 | 86 |
| T4 | 126/140 | 225 | 24 |
| T5 | 138/140 | 430 | 42 |
| T6 | 138/140 | 770 | 125 |
| T7 | 137/140 | 491 | 67 |
| T8 | 138/140 | 781 | 62 |
| T9 | 139/140 | 1751 | 167 |
| T10 | 129/140 | 208 | 21 |

Clip supply is skewed: **T3 (1213) and T9 (1751)** dominate because those
recordings are longest (median resp length ≈ 35 s → 4 clips; T3/T9 ≈ 75 s / 118 s).

---

## 6. Single-subject spot check (F001, low+moderate)

Produced with `runners/run_inspect_tir_resp.py` (job **55340348**), 5/5 clip
checks passed, `input_size 112`, `roi_padding 0.2`, `min_signal_spread 0.01`:

```
F001_T2  clip 0    roi 103x160px   resp raw[-10.000, -4.665]V  11.7% railed (whole file)
F001_T3  clip 2    roi  85x 94px   resp raw[ -7.509, -1.131]V   0.0%
F001_T4  clip 11   roi  82x 91px   resp raw[ -8.932, -3.223]V   0.0%
F001_T7  clip 13   roi 103x102px   resp raw[-10.000, -4.355]V  13.3%
F001_T10 clip 18   roi 172x102px   resp raw[-10.000, -1.712]V  36.9%
```

`F001_T8` yields **0 clips** — its only 8 s window hits a `(0,0)` sentinel line.

---

## 7. Recommendations before Stage-2 pre-training

The corpus is **89 % clean at clip level**, so a default run is viable, but:

1. **Add a rail-aware drop (preferred).** Extend
   `data/tir_resp_dataset.BP4DPlusTIRRespDataset` with an opt-in
   `max_rail_fraction` (e.g. `0.10`) that drops a clip whose respiration window
   is more than that fraction pinned at the rail — this removes the ~7 %
   moderate+severe clips instead of only the 0.5 % fully-railed ones.
   Surface it as a config key + CLI flag and document it next to
   `min_signal_spread`.
2. **Exclude the 16 RESP-unusable sessions** (§3.3) — or filter at session
   level (rail > 50 %) — for a cleaner pre-training set.
3. **Accept as-is** and rely on clip-level z-scoring, accepting that ~173 clips
   have over half their target physically clipped (a hard floor on MAE).

Whichever is chosen, keep the value **identical in the K400 and SSV2 runs** so
the corpus comparison is not confounded.

---

## 8. Artifacts

| path | content |
|---|---|
| `$WORK_SCRATCH/tir_resp_survey/tir_resp_survey.json` | full per-session records + aggregate + warnings |
| `$WORK_SCRATCH/tir_resp_survey/tir_resp_survey_sessions.csv` | one row per session (rail %, lengths, clip yield) |
| `$WORK_SCRATCH/inspect_tir_resp_lowmod/` | F001 low+moderate per-clip figures + JSONs |
| `code/logs/survey_tir_resp_55340713.out` | the human-readable report |

**Caveat.** Rail prevalence is measured against the `-9.999 V` threshold; the
underlying files use exactly `-10.0000` for the dead-channel marker, so the
threshold is not sensitive. Frame counts come from an OpenCV *container probe*
(no decode), matching what the dataset itself uses.
