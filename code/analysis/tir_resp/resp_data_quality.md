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
the rail is classified afterwards), `clip_seconds=8`, `clip_stride=1` (the
shipped hop; pass `--clip_stride 0` for the historical non-overlapping survey).
The "rail" is a sample `<= -10.0 V` (float32-exact; the corpus' dead channel
reads exactly `-10.0000`), and the ADOPTED clip rule additionally treats any
`abs(x) >= 9.90 V` sample as a rail touch (§3.4).

**Task levels** use the `stage2_local_tir_roi_resp.yaml` / HPC config
convention: `low = {T2,T3}`, `moderate = {T4,T7,T8,T10}`,
`high = {T1,T5,T6,T9}`.

---

## 2. Coverage & usability

| | 8 s hop | **1 s hop (shipped)** |
|---|---|---|
| sessions discovered | **1400** (140 subjects × 10 tasks) | **1400** |
| **sessions usable** | **1353 (96.6 %)** | **1361 (97.2 %)** |
| sessions skipped | **47 (3.4 %)** | **39 (2.8 %)** |
| 8 s clips yielded | **6718** | **48 999** |

(Geometry: the shipped TIR-ROI/RESP configs use an 8 s window with a **1 s
hop** — 7/8 overlap — since 2026-10-06; the earlier surveys used a
non-overlapping 8 s hop. A finer hop yields more windows, so fewer sessions end
up with every window dropped.)

### Skip reasons (47 at the 8 s hop / 39 at the 1 s hop)

| reason | n | detail |
|---|---|---|
| `missing_ir_features` | 15 | no `IRFeatures/<S>_<T>.txt` track |
| `all_clips_dropped` | 25 -> **17** | every window contains an IR `(0,0)` sentinel line (1–4 windows) |
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

### 3.1 Whole-file rail fraction (1361 usable sessions)

| rail % | sessions | share |
|---|---|---|
| 0 % (clean) | 1098 | 80.7 % |
| 0–1 % | 35 | 2.6 % |
| 1–5 % | 84 | 6.2 % |
| 5–10 % | 36 | 2.6 % |
| 10–25 % | 51 | 3.7 % |
| 25–50 % | 24 | 1.8 % |
| 50–90 % | 28 | 2.1 % |
| > 90 % | 5 | 0.4 % |

Mean file rail **3.33 %**, median **0 %**.
**19.3 %** of sessions have any rail, **7.9 %** > 10 %, **2.4 %** > 50 %.
These are FILE-level statistics, so they do not depend on the clip geometry (the
8 s-hop survey gave the same table within ±1 session).

### 3.2 Impact on the clips that would be trained on (48 999, 8 s window / 1 s hop)

| clip severity | clips | share |
|---|---|---|
| clean (0 % railed) | 43 800 | **89.4 %** |
| mild (0–10 %) | 1660 | 3.4 % |
| moderate (10–50 %) | 2293 | 4.7 % |
| severe (> 50 %) | 1015 | **2.1 %** |
| 100 % railed | 231 | 0.5 % |
| flat (`max−min < 0.1 V`) | 1961 | 4.0 % |

> **Key point.** `min_signal_spread` is shipped at `0.1` V (the DEAD-SIGNAL
> guard: every corpus window below 0.1 V is a channel pinned just inside a
> clamp, median level +9.1 V -- see §7 item 2), which removes **1434** of the
> 43 360 clips that survive the rail rule. The adopted **rail-touch rule removes
> every clip with ANY rail sample** — 5639 clips (11.5 %) — so no rail-valued
> target survives either; 41 926 clips in 1268 sessions remain. See
> [`resp_rail_touch_filter_results.md`](resp_rail_touch_filter_results.md).

### 3.3 The worst sessions (every clip > 50 % railed, 8 s hop)

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

### 3.4 Rail forensics — dead/pinned channel vs genuine clipped extreme

A `-10 V` sample has two possible causes. BOTH leave a rail-valued sample in the
LABEL (physically unrepresentable), so the adopted clip rule drops both — but the
split still matters, because it says what that removal COSTS:

1. **dead / pinned channel** (dead/disconnected belt, saturated amplifier, DC
   offset below range) → the recording carries no usable respiration, so losing
   it is a gain;
2. **genuine clipping** (the task provoked an extreme the recorder cannot
   represent, so the trough is flat-topped) → the rest of the window is valid, so
   losing it is the price of the policy (3565 clips; §3 of
   [`resp_rail_touch_filter_results.md`](resp_rail_touch_filter_results.md)).

`analysis/tir_resp/rail_forensics.py` separates them from the rail's *structure*
(no video decode) and its `dead_channel()` labels a session for the cost report;
nothing in the pipeline filters on it. Run it with::

    python analysis/tir_resp/rail_forensics.py --raw_root /work/projects/l0003511/test \
        --json $WORK_SCRATCH/rail_forensics.json

| signature | dead / pinned channel | genuine clipping |
|---|---|---|
| share of the file railed | high (often ~100 %) | low–moderate |
| longest contiguous rail run | long (5 s – whole file) | short (≤ ~3 s) |
| railed from `t = 0` | usually | rarely |
| waveform above the rail | pinned at the floor; only narrow upward spikes | normal breathing well above the floor |
| slope in / out of a run | arbitrary | **descends in, ascends out** (brackets the trough) |
| non-rail mean gap above `-10 V` | < 1.5 V (baseline sits at the rail) | median ≈ 3.9 V |

Visual confirmation (4 sessions, full + zoom):
`$WORK_SCRATCH/rail_forensics_examples.png`. `F074_T1` / `F001_T10` show a normal
breathing waveform whose **troughs** flat-top at `-10 V` (clipping); `F004_T3`
shows a baseline **pinned** at `-10 V` with narrow upward spikes (failure);
`F024_T8` is a flat line (dead channel).

**Corpus-wide result (272 sessions with any rail):**

| class | sessions | clips (8 s hop / **1 s hop**) | clips > 10 % railed |
|---|---|---|---|
| **dead / pinned channel** | **48** | 262 / **1909** | 218 |
| **genuine clipping** | **224** | 1065 / **7705** | 273 |

That split is what the adopted rule is designed for: both classes leave a
rail-valued, physically unrepresentable sample in the LABEL, so the rule drops
both (5639 of the 48 999 clips at the shipped geometry — 1765 from dead-channel
sessions and 3565 from genuine-clipping ones). The cost is reported in
[`resp_rail_touch_filter_results.md`](resp_rail_touch_filter_results.md) §3.

**Resolution — the rail-TOUCH rule (2026-10-06).** Drop a clip whose respiration
window contains ANY sample with `abs(x) >= rail_touch_v` (volts; `9.90` in the
shipped TIR-ROI/RESP configs, `0.0` = off). The recorder clamps at `+/-10 V`, so
`9.90` means "a reading reached within 0.1 V of either end of the range": that
removes a dead/pinned channel AND a clipped flat-topped trough, i.e. **no
rail-valued label survives in the training set**. `min_signal_spread` remains the
degenerate-window guard and is probed **after** the rail rule. Results and the
cost split by rail CAUSE are in
[`resp_rail_touch_filter_results.md`](resp_rail_touch_filter_results.md); §7 has
the details.

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

Levels (config convention; **8 s window / 1 s hop** -> 48 999 clips):

```
low       sessions 275/280   clips 11858   touched 1100   kept 10758   emptied 14   mean file rail 3.31%
moderate  sessions 535/560   clips 11771   touched 1335   kept 10436   emptied 36   mean file rail 3.02%
high      sessions 551/560   clips 25370   touched 3204   kept 22166   emptied 34   mean file rail 3.65%
```

(`touched` is the adopted rule's symmetric `abs(x) >= 9.90 V` test; `emptied` is
the number of sessions whose EVERY clip touches the rail. The non-overlapping
8 s-hop numbers — 1619 / 1705 / 3394 clips — plus the 0.1 V margin analysis are
in [`resp_rail_touch_filter_results.md`](resp_rail_touch_filter_results.md) §2.)

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
checks passed, `input_size 112`, `roi_padding 0.2` (that job ran with
`min_signal_spread 0.01`, i.e. before the guard was raised to 0.1):

```
F001_T2  clip 0    roi 103x160px   resp raw[-10.000, -4.665]V  11.7% railed (whole file)
F001_T3  clip 2    roi  85x 94px   resp raw[ -7.509, -1.131]V   0.0%
F001_T4  clip 11   roi  82x 91px   resp raw[ -8.932, -3.223]V   0.0%
F001_T7  clip 13   roi 103x102px   resp raw[-10.000, -4.355]V  13.3%
F001_T10 clip 18   roi 172x102px   resp raw[-10.000, -1.712]V  36.9%
```

`F001_T8` yields **0 clips** — its only 8 s window hits a `(0,0)` sentinel line.
Under the **adopted** `rail_touch_v = 9.90` the same low+moderate selection yields
**22 clips from 3 sessions** at a 4 s hop and **88 from 4 sessions** at the
shipped 1 s hop (137 with the rule OFF): `T10` (36.9 % rail) and `T7` (13.2 %)
are emptied/decimated, `T2` (11.7 %) loses part of its clips, and the clean
`T3`/`T4` are untouched.

---

## 7. Recommendations before Stage-2 pre-training

The corpus is **89 % clean at clip level**, so a default run is viable, but:

1. **Rail TOUCH clip drop — IMPLEMENTED 2026-10-06 (the adopted knob).**
   `data/tir_resp_dataset.BP4DPlusTIRRespDataset` gained an opt-in
   `rail_touch_v` (VOLTS, default `0.0` = off; `9.90` in the shipped
   TIR-ROI/RESP configs) that drops a clip whose respiration window contains ANY
   sample with `abs(x) >= rail_touch_v`. Exposed as `--rail_touch_v` on every
   entry point that can build the path (`run_pretrain.py`, `run_waveform.py`,
   `run_finetune.py`, `run_inspect_tir_resp.py`, `run_survey_tir_resp.py`) and
   logged by both builders. With the shipped 8 s / **1 s** geometry it drops
   **5639 / 48 999 clips (11.5 %)** and empties **84** sessions -> **43 360
   clips in 1277 sessions** remain, with **92.5 %** of the dead-channel-session
   clips removed (36 of 47 such sessions fully excluded). It also removes the
   3565 clips of genuine-clipping sessions whose label is flat-topped at the
   clamp — the deliberate, stricter trade. On top of it the **spread guard**
   (§7 item 2) removes another 1434 clips / 9 sessions (41 926 clips in 1268
   sessions remain in total).
2. **`min_signal_spread` (shipped `0.1`, not `0.01`) is the DEAD-SIGNAL guard.**
   The **rail-touch test runs FIRST**, so `clips_dropped_rail` is attributed
   correctly; the guard then catches what the rail test cannot see -- a channel
   PINNED just inside a clamp without touching it. Measured: every corpus window
   whose spread is below 0.1 V sits at a median level of **+9.1 V** (0.9 V under
   the positive clamp) with a 0.05-0.09 V ripple, and **none** is at a plausible
   breathing level; genuine breathing is >= 0.3 V p-p. At `0.1` it removes
   **1434 clips (3.3 % of the 43 360 that survive the rail rule)** and empties
   **9** sessions -> **41 926 clips in 1268 sessions** remain. It is not "noise
   filtering": `max−min` is a RANGE, so a single spike inflates it, and a loose
   belt that *swings* gives a LARGE artifact that no spread threshold sees.

Whichever is chosen, keep `rail_touch_v` **identical in the Stage-2,
Stage-3 and K400-vs-SSV2 runs** so the comparison is not confounded.

---

## 8. Artifacts

| path | content |
|---|---|
| `$WORK_SCRATCH/tir_resp_survey/` | full per-session records + aggregate + warnings (rail-prevalence run, job 55340713, 8 s hop) |
| `$WORK_SCRATCH/tir_resp_survey_touch/` | the same survey with the ADOPTED touch rule (job 55347699, 8 s hop) |
| `$WORK_SCRATCH/tir_resp_survey_1s/` | the SHIPPED geometry: the touch rule at an 8 s window / **1 s hop** (job 55347758) |
| `$WORK_SCRATCH/rail_forensics_all.json` | per-session rail structure + `dead_channel` classification |
| `$WORK_SCRATCH/rail_filter_impact{,_1s}.json` | the rule-vs-cause impact tables (`rail_filter_impact.py`) |
| `$WORK_SCRATCH/inspect_tir_resp_lowmod/` | F001 low+moderate per-clip figures + JSONs |
| `code/logs/survey_tir_resp_55340713.out` | the human-readable rail-prevalence report |

**Caveat.** Rail *prevalence* (§2–§5) is measured against the `-10.0000 V`
threshold (float32-exact, i.e. the dataset's `RESP_RAIL_V`). The underlying files
use exactly `-10.0000` for the dead-channel marker, so that threshold is not
sensitive; the ADOPTED clip filter is deliberately wider and symmetric
(`abs(x) >= 9.90 V`), which additionally catches 63 clips (24 in the
`[-10.0, -9.9)` band and 39 at the POSITIVE clamp) — see
[`resp_rail_touch_filter_results.md`](resp_rail_touch_filter_results.md) §2.1.
Frame counts come from an OpenCV *container probe* (no decode), matching what
the dataset itself uses.
