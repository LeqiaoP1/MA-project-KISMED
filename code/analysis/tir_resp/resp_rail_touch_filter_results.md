# Respiration clip filter — the rail-TOUCH rule (one rule, corpus results)

**Date:** 2026-10-06
**Jobs:** Slurm **55348993** — the SHIPPED policy (8 s window, **1 s hop**,
`rail_touch_v 9.90` + `min_signal_spread 0.1`) — **55347758** (the same geometry
with the guard at 0.01), **55347699** (reference geometry: 8 s hop =
non-overlapping) and **55347966** (the threshold sweep of §5). All `deflt_short`,
16 workers, ~30 s.
**Tools:** `runners/run_survey_tir_resp.py`
(`--clip_stride 1.0 --rail_touch_v 9.90 --flat_spread 0.1
[--rail_touch_sweep/--spread_sweep]`),
`analysis/tir_resp/rail_forensics.py`, `analysis/tir_resp/rail_filter_impact.py`
(`--sweep`)
**Artifacts:** `$WORK_SCRATCH/tir_resp_survey_1s/` (shipped),
`.../tir_resp_survey_touch/` (reference) and `.../tir_resp_survey_sweep/` (§5),
each with `tir_resp_survey.json` + `..._sessions.csv`;
`$WORK_SCRATCH/rail_forensics_all.json`;
`$WORK_SCRATCH/rail_filter_impact{,_1s,_sweep}.json`; logs
`code/logs/survey_tir_resp_5534775{8,9}.out`, `..._55347966.out` and
`..._55348993.out`
**See also:** §3.4 of [`resp_data_quality.md`](resp_data_quality.md) for the
two-cause analysis of the rail.

---

## 1. The two rules

Both are per-WINDOW and both must match between Stage 2 and Stage 3. The rail
test runs FIRST, then the spread guard (so the two drop counters never overlap).

### 1.1 Rail: drop a clip whose reading EVER TOUCHED the rail

> The clip's 8 s respiration window contains any sample with
> `abs(x) >= rail_touch_v`, with **`rail_touch_v = 9.90 V`** (`0.0` = off).

The recorder clamps at `+/-10 V`, so `9.90` is "within 0.1 V of either end of the
range". Implementation notes:

| aspect | choice | why |
|---|---|---|
| shape of the test | **magnitude**, `abs(x) >=` | the clamp is **two-sided**; a negative-only test would keep the **279** clips that touch the positive rail at the shipped hop |
| granularity | **per window**, ANY sample | it is a *touch* test — one sample is enough; any criterion that measures a SHARE of the window would let a partly-railed label through |
| what it is applied to | the **interpolated window that becomes the target** | "touched" == "the label contains a rail-valued sample" |
| threshold representation | quantised to **float32** once, in `__init__` | the samples are float32 (`-9.9000` -> `-9.8999996`), so a raw float64 `>= 9.9` test would silently MISS a sample valued exactly at the threshold — the trap that also hides `-9.999` from a `<= -9.999` test |

**Why a touch test.** The rail has two causes (§3.4, `rail_forensics.py`): a
**dead/pinned channel** (dead belt, stuck DAQ, DC offset out of range) and a
**genuine extreme expiration** whose trough is driven into the clamp. Both leave
a rail-valued, physically unrepresentable sample in the LABEL, and the rule keys
on exactly that — so both go. §3 reports the cost of removing the second kind.

`min_signal_spread` (0.1 V, §1.2) is the second guard and is probed AFTER the
rail test, so `clips_dropped_rail` keeps the attribution.

### 1.2 Spread: drop a clip whose window is a pinned / dead signal

> The window's spread `max(x) - min(x)` is **below `min_signal_spread = 0.1 V`**
> (`0.0` = off).

This is the **complement** of §1.1, not a duplicate: the rail test catches what
*reaches* a clamp, this one catches a channel **pinned just inside** it. Measured
on the 43 360 clips that survive the rail rule:

* every window below 0.1 V sits at a **median level of +9.1 V** — 0.9 V under the
  positive clamp — with a **0.05-0.09 V ripple** (disconnected belt / loose
  contact / sensor resting on a surface);
* **no** window in that band is at a plausible breathing level (-2..+2 V), and
  the 0.05-0.10 V part of the band is *even more* strongly pinned (88.8 % above
  +8 V) than the 0-0.05 V core (76.5 %);
* genuine breathing is >= 0.3 V p-p (corpus p10 ~0.75 V), so the guard does not
  reach even shallow respiration;
* it is not "noise filtering": `max - min` is a RANGE, so one spike inflates it
  (a pinned channel with a single 0.5 V glitch passes any threshold), and a loose
  belt that *swings* gives a LARGE artifact — no spread value sees either.

Why it matters beyond removing dead data: under `target_norm: clip` (Stage 2) and
`signal_norm: zscore` (Stage 3) a 0.06 V ripple is **rescaled to unit variance**,
so it would enter training as a full-amplitude label that is pure artifact.

### 1.3 Geometry: the 8 s window moves with a 1 s hop

`clip_stride` is `1.0` in all 7 TIR-ROI/RESP configs (it was `4.0` in Stage 2 and
`2.0` in Stage 3), so **7 of every 8 consecutive windows overlap**. The intent is
to sample the **transient breathing dynamics** an emotional episode shows — a 4 s
hop averages them away. Consequences:

* ~4x the clips (and the epoch time) for the same recordings;
* consecutive clips are ~87 % correlated, so the *effective* sample size is far
  smaller than the clip count: read a per-clip count as a count of WINDOWS, not
  of independent observations;
* `clip_stride` decides **which windows exist**, so it is part of the corpus
  contract — `run_waveform.py` prints a note when it (or `rail_touch_v`) differs
  from the Stage-2 checkpoint's value.

```bash
# from code/ -- both artifacts below come from these two commands
python analysis/tir_resp/rail_forensics.py \
  --raw_root "$RAW_DATA_PATH" --json $WORK_SCRATCH/rail_forensics_all.json
SURVEY_WORKERS=16 SURVEY_OUTPUT_DIR=$WORK_SCRATCH/tir_resp_survey_1s \
  sbatch -p deflt_short -t 00:25:00 -c 16 scripts/hpc/submit_survey_tir_resp.sbatch \
         --clip_stride 1.0 --rail_touch_v 9.90 --flat_spread 0.1
python analysis/tir_resp/rail_filter_impact.py \
  --survey $WORK_SCRATCH/tir_resp_survey_1s/tir_resp_survey.json \
  --forensics $WORK_SCRATCH/rail_forensics_all.json
```

## 2. Corpus result

| | 8 s hop (reference, job 55347699) | **1 s hop (shipped, job 55348969)** |
|---|---|---|
| sessions discovered | 1400 | 1400 |
| sessions usable | 1353 (47 skipped) | **1361 (39 skipped)** |
| **clips (8 s windows)** | 6718 | **48 999** |
| dropped by `rail_touch_v` (§1.1) | 786 (11.7 %) | **5639 (11.5 %)** |
| dropped by `min_signal_spread` (§1.2, on top) | – | **1434 (3.3 % of the survivors)** |
| **clips kept (both rules)** | – | **41 926 (85.6 %)** |
| sessions emptied (rail + spread) | 106 | **84 + 9 = 93** |
| **sessions remaining** | 1247 | **1268** (of 1361) |

The finer hop both **enlarges** the corpus (7.3x more windows) and **softens** the
session-level effect: fewer sessions are skipped (47 -> 39; `all_clips_dropped`
25 -> 17) and fewer are emptied by the rail rule (106 -> 84), because a longer
recording now has many more windows, so it is far less likely that *every* one of
them is railed. The **fraction** of clips that touch the rail is essentially
unchanged (11.7 % -> 11.5 %): the rail is a property of the channel and
overlapping windows each see it.

Skip reasons at the 1 s hop: `all_clips_dropped` 17, missing IRFeatures 15, resp
too short 6, missing Resp_Volts 1.

| level | sessions | clips | rail-dropped | kept | spread-dropped | **kept (both)** | sessions emptied (rail+spread) |
|---|---|---|---|---|---|---|---|
| low (T2,T3) | 275 | 11 858 | 1100 | 10 758 | 537 | **10 221** | 14 + 3 |
| moderate (T4,T7,T8,T10) | 535 | 11 771 | 1335 | 10 436 | 184 | **10 252** | 36 + 5 |
| high (T1,T5,T6,T9) | 551 | 25 370 | 3204 | 22 166 | 713 | **21 453** | 34 + 1 |
| **total** | **1361** | **48 999** | **5639** | **43 360** | **1434** | **41 926** | **84 + 9** |

### 2.1 What the 0.1 V margin buys

| measurement | 8 s hop | **1 s hop** |
|---|---|---|
| contain a sample exactly at `-10.0 V` | 723 | 5199 |
| touch the **negative** rail (`<= -9.90 V`) | 747 | 5360 |
| touch the **positive** rail (`>= +9.90 V`) | 39 | **279** |
| **touch either rail (the rule)** | **786** | **5639** |

A `<= -10.0 V` test would keep **63** clips at the 8 s hop and **440** at the 1 s
hop: those whose most extreme negative reading sits in `[-10.0, -9.90)` plus the
ones touching the positive clamp. Both are genuine rail excursions.

## 3. Cost: what the drop removes, split by the cause of the rail

`rail_forensics.dead_channel()` classifies a **file** as a dead/pinned channel
when `rail_pct > 90 %` **or** a rail run `>= 5 s` **or** the non-rail baseline
sits `< 1.5 V` above the floor. It is a description of the session, not a filter
(nothing in the pipeline applies it to the data).

| session class | sessions | clips (usable) | dropped by the rule | kept | sessions fully emptied |
|---|---|---|---|---|---|
| **dead / pinned channel** | 48 (47 usable) | 1909 | **1765 (92.5 %)** | 144 | 36 |
| **genuine clipped trough** | 224 (216 usable) | 7705 | **3565** | 4140 | 45 |
| not in the forensics set¹ | — | — | 309 | — | — |
| **total** | — | **48 999** | **5639** | **43 360** | **84** |

¹ sessions whose file never reaches the exact `-10.0 V` floor (so the forensics
sweep does not list them) but does come within 0.1 V of a clamp — the clips the
9.90 margin adds over a `<= -10.0 V` test.

So **31.3 %** of the drop is dead-channel material and **63.2 %** is genuine
clipped troughs. That is the deliberate price of the policy "no rail-valued
target sample anywhere in the training set": the retained corpus contains no
clipped label at all, which is what removes the unrepresentable targets — and it
costs 3565 clips whose label was a real, if clipped, expiration.

## 4. Known limitation of a rail-CONTACT test

The rule can only see what reaches the clamp. 11 dead-channel sessions never
do — their baseline is pinned *just above* the floor (non-rail gap
0.06–1.46 V) — so **144 clips (0.29 % of the corpus) remain**:

| session | file rail % | longest run (s) | non-rail baseline gap (V) | clips | kept |
|---|---|---|---|---|---|
| `F054_T3` | 1.9 | 0.92 | 0.79 | 67 | **55** |
| `F082_T2` | 9.2 | 2.52 | 1.37 | 31 | **19** |
| `M005_T9` | 7.7 | 0.97 | 1.35 | 44 | **13** |
| `M033_T9` | 34.5 | 16.02 | 1.43 | 109 | **10** |
| `M002_T8` | 29.4 | 3.07 | 1.46 | 38 | **9** |
| `M001_T8` | 15.2 | 2.32 | 1.42 | 37 | **9** |
| `F033_T9` | 12.7 | 1.16 | 0.66 | 85 | **8** |
| `F046_T10` | 2.0 | 0.35 | 1.24 | 11 | **8** |
| `F024_T1` | 1.1 | 0.21 | 0.80 | 11 | **6** |
| `M039_T2` | 0.7 | 0.06 | 1.14 | 10 | **6** |
| `M002_T7` | 22.4 | 1.85 | 1.34 | 22 | **1** |

These are small-amplitude channels (`F054_T3`, the worst, has a mean breathing
amplitude below 0.8 V), not constant ones, so the `min_signal_spread` guard does
not reach them either (their spread is 0.33-6.9 V). §5 prices the two knobs that
COULD remove them.

## 5. Trade-off curve: what each threshold would cost

Measured in ONE survey pass (`--rail_touch_sweep` / `--spread_sweep`, job
**55347966**). The sweep only CLASSIFIES — it never filters the corpus it builds.
`vs 9.9` is the change relative to the shipped value; `residual left` counts the
windows still kept out of the **144** of §4.

### 5.1 The rail threshold `rail_touch_v`

| V | dropped | vs 9.9 V | kept | sessions emptied | residual left | extra drop: dead / clipped / near-rail¹ |
|---|---|---|---|---|---|---|
| 9.99 | 5225 | −414 | 43 774 | 81 | 159 | −26 / −97 / −291 |
| 9.97 | 5274 | −365 | 43 725 | 81 | 152 | −13 / −69 / −283 |
| 9.95 | 5377 | −262 | 43 622 | 82 | 147 | −3 / −46 / −213 |
| **9.90 (shipped)** | **5639** | **0** | **43 360** | **84** | **144** | 0 / 0 / 0 |
| 9.80 | 5880 | +241 | 43 119 | 92 | 115 | 29 / 98 / 114 |
| 9.70 | 6053 | +414 | 42 946 | 97 | 93 | 51 / 134 / 229 |
| 9.50 | 6746 | +1107 | 42 253 | 109 | 66 | 78 / 295 / 734 |
| 9.00 | 8694 | +3055 | 40 305 | 166 | 21 | 123 / 686 / 2246 |
| 8.00 | 14 213 | +8574 | 34 786 | 301 | 0 | 144 / 1626 / 6804 |

¹ sessions whose file never reaches the exact `-10.0 V` floor (so the forensics
sweep does not classify them) but whose windows DO reach the threshold — the
positive clamp and the near-rail negatives. They are clip-like excursions too,
and they dominate the extra drop below 9.5 V.

The curve is flat at the safe end and steepens fast: the 0.1 V margin
(`9.99 -> 9.90`) costs 414 clips, and each further step down costs 241 (9.80),
414 (9.70), 1107 (9.50), 3055 (9.00).

| target | setting | residual removed | corpus cost | among the extra drop: dead / clipped / near-rail | residual clips per 1000 clips lost |
|---|---|---|---|---|---|
| −20 % of the residual | `9.80` | 29 / 144 | 241 (0.5 %) | 29 / 98 / 114 | **120** |
| −35 % | `9.70` | 51 / 144 | 414 (0.8 %) | 51 / 134 / 229 | **123** |
| −54 % | `9.50` | 78 / 144 | 1107 (2.3 %) | 78 / 295 / 734 | 70 |
| −85 % | `9.00` | 123 / 144 | 3055 (6.2 %) | 123 / 686 / 2246 | 40 |
| −100 % | `8.00` | 144 / 144 | 8574 (17.5 %) | 144 / 1626 / 6804 | 17 |

### 5.2 The degenerate-window guard `min_signal_spread`

Applied ON TOP of the shipped rail rule (the guard runs second):

| spread < V | dropped | kept | residual left | among the drop: dead / clipped / other² | residual clips per 1000 clips lost |
|---|---|---|---|---|---|
| 0.01 | 0 | 43 360 | 144 | 0 / 0 / 0 | — |
| 0.10 | 1434 (3.3 %) | 41 926 | 144 | 0 / 1 / 1433 | **0** |
| 0.25 | 2639 (6.1 %) | 40 721 | 144 | 0 / 14 / 2625 | **0** |
| 0.50 | 3908 (9.0 %) | 39 452 | 138 | 6 / 54 / 3848 | 1.5 |
| 1.00 | 6448 (14.9 %) | 36 912 | 112 | 32 / 120 / 6296 | 5.0 |
| 1.50 | 8783 (20.3 %) | 34 577 | 84 | 60 / 218 / 8505 | 6.8 |
| 2.00 | 11 568 (26.7 %) | 31 792 | 45 | 99 / 327 / 11 142 | 8.6 |
| 3.00 | 17 397 (40.1 %) | 25 963 | 42 | 102 / 598 / 16 697 | 5.9 |

² sessions that are neither dead-channel nor clipping-labelled: the drop is
ordinary low-amplitude BREATHING, i.e. the cost of the guard is paid mostly on
valid data.

**Reading.** Below 0.5 V the guard removes *none* of the residual (its smallest
window spread is 0.33 V) while already costing 1434–2639 clips of normal data;
the window into which it starts biting is the same one that thins the whole
corpus (p10 of the surviving spread is 0.75 V). Per residual clip removed it
costs **15–80x more** than lowering `rail_touch_v` (§5.1), and most of what it
drops is legitimate respiration.

So the two knobs answer different questions: `rail_touch_v` targets
rail-adjacent windows (the residual included), `min_signal_spread` targets
low-amplitude ones whatever their cause — and the residual is the former, not
the latter. The SHIPPED guard is `0.1` (§1.2): it takes the 1434 clips of the
first two rows (`max - min` is a range, so it is the only value on this table
that removes the pinned class without touching anything that moves), leaving
**41 926 clips in 1268 sessions** after both rules. The rail threshold stays at
`9.90` unless the ~0.29 % residual of §4 is judged worth the prices in §5.1.

## 6. Where the rule applies (audited) + end-to-end spot check

| stage | entry point | path to the dataset | flags |
|---|---|---|---|
| 2 — masked pretraining | `runners/run_pretrain.py` | `build_pretraining_dataset` -> `build_tir_roi_pretrain_dataset` | `--rail_touch_v`, `--min_signal_spread` |
| 3 — waveform | `runners/run_waveform.py` | `build_dataset` -> `build_tir_roi_finetune_dataset` | both |
| 3 — classification | `runners/run_finetune.py` | `build_dataset` -> `build_tir_roi_finetune_dataset` | both |
| clip inspection | `runners/run_inspect_tir_resp.py` | `BP4DPlusTIRRespDataset` | both |
| corpus survey | `runners/run_survey_tir_resp.py` | `BP4DPlusTIRRespDataset` built UNFILTERED on purpose (it CLASSIFIES with the rules) | `--rail_touch_v`, `--flat_spread` |
| dataset self-test / ROI parity | `data/tir_resp_dataset.py`, `check_view_parity()` | same class | both |

Real dataset build for **F001, low + moderate** (`input_size 64`, `roi_padding
0.2`, `min_signal_spread 0.1`, `rail_touch_v 9.90`, no dev caps), only the hop
and the spread guard changed:

```
clip_stride = 4.0, rail_touch_v = 0.0   -> 36 clips / 5 sessions
clip_stride = 4.0, rail_touch_v = 9.90  -> 22 clips / 3 sessions   (T7, T10 emptied)
clip_stride = 1.0, rail_touch_v = 0.0   -> 137 clips / 5 sessions  (T2 15, T3 68, T4 10, T7 33, T10 11)
clip_stride = 1.0, rail_touch_v = 9.90  ->  88 clips / 4 sessions  (T2 15->9, T3 68, T4 10, T7 33->1, T10 emptied)
```

Which matches the per-file rail% of `resp_data_quality.md` §6: `F001_T10`
(36.9 %) and `F001_T7` (13.2 %) are emptied (`T7` keeps one clip at the 1 s hop),
`F001_T2` (11.7 %) loses 6 of 15, and the clean `T3`/`T4` are untouched. The
spread guard changes none of these (F001 is a rail problem, not a pin problem).

## 7. Verification

* `pytest -q` -> **111 passed**:
  * `tests/test_tir_resp_rail_filter.py` (12) — fully railed channel, pinned
    baseline, a clipped trough at 0.5/1.0/2.5/4.0 s (**dropped**), a single
    rail-valued sample in both polarities, a `-9.90 V` sample exactly at the
    threshold (**dropped** — the float32 regression), `-9.89 V` (kept),
    `10.5 V` (nothing kept), rule-off, attribution order vs `min_signal_spread`,
    and validation;
  * `tests/test_rail_rule_plumbing.py` (11) — the DISPATCH layer
    (`build_pretraining_dataset` for Stage 2, `build_dataset` for Stage 3) hands
    `rail_touch_v` to the dataset, an absent key means off *and printed*, every
    entry point exposes the flag, and all 7 configs ship `rail_touch_v: 9.90` +
    `min_signal_spread: 0.1` + `clip_stride: 1.0`.
* Two guards make a silently-off rule impossible: both builders print
  `[data] stage2|stage3 corpus cleaning: rail_touch_v=... V (DROP ... | OFF)`,
  and `run_waveform.py` lists `rail_touch_v` **and** `clip_stride` in its
  Stage-2-vs-Stage-3 corpus note alongside `task_set` / `tasks` /
  `min_signal_spread`.
* `compileall` clean; the 7 configs load with `rail_touch_v=9.9`,
  `clip_stride=1.0`; `tests/test_config_loading.py` asserts both.

## 8. Artifacts

| path | content |
|---|---|
| `runners/run_survey_tir_resp.py` | survey; `--rail_touch_v`, `--clip_stride`, `--rail_touch_sweep` / `--spread_sweep` (the §5 curve) |
| `analysis/tir_resp/rail_forensics.py` | per-session rail structure + `dead_channel()` (a description, not a filter) |
| `analysis/tir_resp/rail_filter_impact.py` | this report's cost table + `--sweep` for §5 (survey x forensics) |
| `$WORK_SCRATCH/tir_resp_survey_1s/` | `tir_resp_survey.json`, `..._sessions.csv` (shipped geometry) |
| `$WORK_SCRATCH/tir_resp_survey_sweep/` | the same + the sweep columns / curves of §5 |
| `code/logs/survey_tir_resp_55347758.out` | the human-readable survey report |
| `code/logs/survey_tir_resp_55347966.out` | the trade-off curve printout |

**Caveat.** The clip counts come from the survey's *classification* of windows
the dataset builds UNFILTERED (it builds with the rule off and counts
afterwards), because a survey of `clip_touched` must not depend on the dataset's
own filtering — so the retained-corpus numbers are predictions, not a post-filter
census. The §5 spot check confirms them on real data.
