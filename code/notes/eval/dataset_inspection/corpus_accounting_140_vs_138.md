# Corpus accounting: why BP4D+ (140 subjects) evaluates as 138

> Measured 2026-10-25 against `$RAW_DATA_PATH=/work/projects/l0003511/test`
> Reproduces the pooled k-fold artifact exactly:
> `output/eval/resp_kfold/kfold_resp_tir_roi_hpc.json` -> **138 subjects /
> 1260 sessions / 21 231 clips**.
>
> Scope: the TIR-ROI -> respiration (Stage-3) path, i.e.
> `data/tir_resp_dataset.py::BP4DPlusTIRRespDataset`, built with the config
> `configs/finetune/resp_tir_roi_hpc.yaml` (`clip_duration 8.0`, `fps 25.0`,
> `resp_fs 1000.0`, `clip_stride 2.0`, `min_signal_spread 0.1`,
> `rail_touch_v 9.9`, `physio_norm session`).

## Verdict

There is no counting bug. Discovery sees the whole corpus -- **140 subjects x
10 tasks = 1400 sessions** -- and then the dataset's own data-quality gates drop
140 sessions. Exactly **two subjects lose all 10 of their sessions**, so the
evaluated set is 138 subjects / 1260 sessions.

| subject | cause | sessions lost |
|---|---|---|
| **M049** | no `IRFeatures` track at all (`IRFeatures/M049_*.txt` absent) -> `missing_ir_features` | 10 / 10 |
| **F004** | every 8 s window touches the respiration rail -> `all_clips_dropped` (rail rule) | 10 / 10 |

F004 is a **sensor failure, not a threshold that is merely too strict**: see
"F004 evidence" below.

## Raw inventory

| tree | entries | note |
|---|---|---|
| `Thermal/` | 140 subject dirs, 1400 videos (10 per subject) | discovery keys on the VIDEO |
| `IRFeatures/` | **1385** `F<nnn>_T<k>.txt` | 15 sessions never released |
| `Physiology/` | 140 subject dirs, 1399 `Resp_Volts.txt` | 1 session missing the file |
| `2D+3D/` | 140 zips | not used by this arm |

## Skipped sessions (140 total)

| reason | sessions | detail |
|---|---|---|
| `all_clips_dropped` -- rail (`\|x\| >= 9.9 V`) | 90 | F004's 10 are all here |
| `all_clips_dropped` -- `(0,0)` sentinel landmark line | 18 | any single bad IRFeatures line kills the window |
| `all_clips_dropped` -- near-constant window (`< 0.1 V`) | 10 | |
| `missing_ir_features` | 15 | M049 x10, F016_T2/T3/T4, F054_T10, M045_T2 |
| `resp_too_short` (`< 8000` samples = 8 s @ 1 kHz) | 6 | F010_T10, F029_T4, F030_T4, F042_T7, F061_T4, F081_T10 |
| `missing_resp_volts` | 1 | F082_T1 |
| **total** | **140** | 1400 - 140 = **1260** |

Session-level counts only; individual clip windows rejected by the rail / flat /
sentinel rules are recorded per session in `dataset.sessions`
(`clips_dropped_rail`, `clips_dropped_flat`, `clips_dropped_sentinel`).

## F004 evidence: the respiration channel is railed

`Physiology/F004/<T>/Resp_Volts.txt`, 1000 Hz:

| task | n samples | min | max | mean | std | share \|x\| >= 9.9 V |
|---|---|---|---|---|---|---|
| T1 | 43 017 | -10.000 | -3.661 | -8.692 | 1.588 | **40.3 %** |
| T3 | 74 952 | -10.000 | -6.382 | -9.751 | 0.617 | **79.6 %** |
| T9 | 80 820 | -10.000 | -4.064 | -9.337 | 1.303 | **67.3 %** |

Reference subjects for contrast: `F001_T1` min -10.000 / max +0.520 / std 2.466
(touches the rail only occasionally, 7 of 10 F001 sessions survive);
`M007_T3` min +2.784 / max +5.973 / std 0.940 (never near the rail).

So F004 is not "one clipped trough" -- the channel sits pinned at the -10 V
limit for most of every recording (a saturated/dead recorder channel on a
negative pedestal). Under the config's TOUCH TEST (one sample at
`|x| >= rail_touch_v` drops the window) **no** window of **any** F004 session
can survive. Recovering this subject would require treating the railed span as
*masked missing data*, not relaxing the threshold -- and the excluded span is
the majority of the signal.

## Where the gates live

`code/data/tir_resp_dataset.py::BP4DPlusTIRRespDataset._build_entries`:

1. `ir_file is None` -> `missing_ir_features`;
2. `resp_file is None` -> `missing_resp_volts`;
3. `_resp_start_limit(...) < 0` -> `resp_too_short`;
4. `missing_frame_mask(ir, self.target_idx)` -- any `(0,0)`/NaN target landmark
   in the window drops it (counted as `clips_dropped_sentinel`);
5. rail test (`rail_touch_v`, evaluated **before** the flat test) and
   `min_signal_spread` -- counted as `clips_dropped_rail` / `clips_dropped_flat`;
6. if no start survives -> `all_clips_dropped` with the per-reason counts.

All of this runs in `__init__`, so it is independent of `--roi_cache`.

## Not the cause: the ROI cache

The cache (`/work/scratch/.../tirroi_nose_mouth_pad0.20_minmax_src726x480_native_v2_...`)
holds **1385 shards + 15 skipped** because it is keyed on the 1385
`IRFeatures` files (pixels only -- it knows nothing about `Resp_Volts`, clip
windows or the cleaning rules). It therefore cannot explain the 1260, and it is
not "140 subjects x 10 tasks" as sometimes quoted: M049 contributes zero shards.

## Per-subject roll-up (54 subjects affected)

| subject | sessions kept | skipped | reasons |
|---|---|---|---|
| F001 | 7 | 3 | RAIL 2, SENT 1 |
| F004 | **0** | 10 | RAIL 10 |
| F010 | 9 | 1 | SHORT 1 |
| F011 | 9 | 1 | RAIL 1 |
| F016 | 7 | 3 | IR 3 |
| F017 | 6 | 4 | RAIL 4 |
| F018 | 9 | 1 | RAIL 1 |
| F019 | 8 | 2 | FLAT 2 |
| F020 | 8 | 2 | RAIL 2 |
| F021 | 7 | 3 | RAIL 3 |
| F023 | 7 | 3 | FLAT 3 |
| F024 | 3 | 7 | RAIL 7 |
| F025 | 8 | 2 | RAIL 2 |
| F029 | 8 | 2 | RAIL 1, SHORT 1 |
| F030 | 8 | 2 | RAIL 1, SHORT 1 |
| F031 | 9 | 1 | SENT 1 |
| F032 | 9 | 1 | SENT 1 |
| F034 | 8 | 2 | RAIL 2 |
| F038 | 4 | 6 | RAIL 5, SENT 1 |
| F039 | 9 | 1 | RAIL 1 |
| F042 | 8 | 2 | SENT 1, SHORT 1 |
| F043 | 5 | 5 | RAIL 3, SENT 2 |
| F045 | 7 | 3 | RAIL 3 |
| F049 | 5 | 5 | RAIL 5 |
| F051 | 9 | 1 | SENT 1 |
| F054 | 9 | 1 | IR 1 |
| F059 | 7 | 3 | RAIL 2, SENT 1 |
| F060 | 8 | 2 | RAIL 1, SENT 1 |
| F061 | 7 | 3 | RAIL 2, SHORT 1 |
| F066 | 9 | 1 | RAIL 1 |
| F072 | 9 | 1 | SENT 1 |
| F073 | 9 | 1 | SENT 1 |
| F074 | 9 | 1 | RAIL 1 |
| F076 | 8 | 2 | SENT 2 |
| F077 | 7 | 3 | RAIL 3 |
| F081 | 8 | 2 | RAIL 1, SHORT 1 |
| F082 | 8 | 2 | RAIL 1, RESP 1 |
| M002 | 7 | 3 | RAIL 3 |
| M003 | 3 | 7 | RAIL 7 |
| M005 | 4 | 6 | RAIL 6 |
| M016 | 8 | 2 | FLAT 1, RAIL 1 |
| M018 | 9 | 1 | RAIL 1 |
| M021 | 9 | 1 | RAIL 1 |
| M023 | 8 | 2 | RAIL 1, SENT 1 |
| M024 | 8 | 2 | FLAT 1, RAIL 1 |
| M025 | 7 | 3 | FLAT 3 |
| M029 | 9 | 1 | SENT 1 |
| M037 | 9 | 1 | RAIL 1 |
| M039 | 9 | 1 | RAIL 1 |
| M044 | 9 | 1 | RAIL 1 |
| M045 | 8 | 2 | IR 1, RAIL 1 |
| **M049** | **0** | 10 | IR 10 |
| M051 | 9 | 1 | SENT 1 |
| M053 | 9 | 1 | SENT 1 |

Legend: IR = `missing_ir_features`, RESP = `missing_resp_volts`,
SHORT = `resp_too_short`, RAIL = rail contact, FLAT = near-constant window,
SENT = `(0,0)` sentinel landmark line. The remaining 86 subjects keep all 10
sessions.

## Reproduce

From `code/` (login node is fine: no pixel decode, containers only):

```bash
source scripts/env_hpc.sh
../.env/bin/python - <<'PY'
import collections, re
from data.tir_resp_dataset import BP4DPlusTIRRespDataset
ds = BP4DPlusTIRRespDataset(clip_seconds=8.0, fps=25.0, resp_fs=1000.0,
                            clip_stride=2.0, min_signal_spread=0.1,
                            rail_touch_v=9.9, norm='session', allow_empty=True)
print(len(ds.skipped) + len(ds.sessions), len(ds.sessions),
      len(ds.entries), len({s.split('_')[0] for s in ds.sessions}))
print(collections.Counter(re.sub(r':.*', '', s['reason']) for s in ds.skipped))
PY
```

Expected: `1400 1260 21231 138` and
`all_clips_dropped 118, missing_ir_features 15, resp_too_short 6,
missing_resp_volts 1`.

## Consequences for reporting

- Report the evaluated corpus as **138 subjects / 1260 sessions / 21 231 clips**
  and state the two exclusions explicitly (M049: no thermal landmark track;
  F004: railed respiration channel). Do not describe it as "all 140 subjects".
- The 5-fold CV folds are 27/28/28/28/27 subjects over those 138 -- the two
  excluded subjects are absent from every fold, so no fold's test block is
  short by accident.
- Thin subjects are legitimate but their per-subject medians rest on few clips:
  M003 and F024 keep 3 sessions, M005 and F038 keep 4, F043 and F049 keep 5.
