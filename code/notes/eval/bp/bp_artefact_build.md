# Blood-pressure session artefact — build accounting (2026-11-20)

What is measured here: the **artefact** (what each shard contains) and the build's
accounting. No GPU time is spent and no model is involved; every number is a
dataset statistic.

Artefact contract: `code/notes/design/ADR-004-bp-session-cache-artefact.md`.
Build order of record: `code/notes/plan/ADR-004-bp-session-cache-plan.md`.

---

## 1. Build

`2026-11-20`, `raw_root=/work/projects/l0003511/test`, single process, CPU only,
**no torch import** (`--check-imports` → `torch loaded: False`).

```
bp_artefact/bpsession_src1000_range30-250_flat200ms_rail30ms_seg4s_v1_a89590f953
```

| quantity | value |
| :--- | :--- |
| session directories discovered | **1400** (matches the ADR exactly) |
| shards written | **1400** |
| skipped | **0** |
| missing | **0** |
| wall time | **34.1 s** (0.6 min; budget < 10 min) |
| per-session cost | 0.00–0.09 s, mean **0.024 s** |
| payload bytes | **380 155 846 B = 380.2 MB** (budget < 600 MB uncompressed; ≈ 374 MB predicted) |

Frozen identity, as written to `params.json`:

```json
{"format_version": 1, "corpus": "847a9166c7c4fbda", "fs_in": 1000, "fs_out": 1000,
 "gate1_min_mmhg": 30.0, "gate1_max_mmhg": 250.0, "flatline_max_ms": 200,
 "rail_dwell_min_ms": 30, "min_segment_s": 4.0,
 "store": "signal+mask+cause+segments:float32,bool,uint8,int64,float64"}
```

A 20-session subset build ran first (0.7 s) and was bit-identical after the full
build — the builder is resumable and skips an existing shard unless `--overwrite`.

## 2. Validity accounting

`manifest.json` over the 1400 shards, 62 405 029 samples:

| cause | samples | share |
| :--- | ---: | ---: |
| `valid` | 61 762 663 | **98.97 %** |
| `gate1_range` | 485 945 | 0.78 % |
| `short_segment` | 156 421 | 0.25 % |
| `gate1_nonfinite` | 0 | 0.00 % |
| `gate2_flatline` | **0** | **0.00 %** |
| `gate3_rail` | **0** | **0.00 %** |

`valid_frac`: mean **0.9902**, median **1.000**, min **0.000**, max **1.000**.
**1349 / 1400 shards (96.4 %) are 100 % valid.**

## 3. Segments

1442 valid segments over 1394 shards:

| distribution | value |
| :--- | :--- |
| segments per session | 1 × 1354, 2 × 34, 3 × 4, 4 × 2 |
| length median / p10 / p90 | **33.1 s / 15.4 s / 74.7 s** |
| length mean / max | 43.5 s / **206.1 s** |
| segments shorter than 8 s | **2.1 %** |

### The six zero-segment shards are not skips

| session | `n_src` | cause mix | why |
| :--- | ---: | :--- | :--- |
| `F030_T10` | 12 863 | `gate1_range` 100 % | no sample at or above 30 mmHg |
| `M049_T10` | 20 773 | `gate1_range` 100 % | no sample at or above 30 mmHg |
| `F016_T10` | 13 790 | range 78.6 %, short 21.4 % | only sub-4 s stretches survived |
| `M016_T1` | 11 525 | range 78.7 %, short 21.3 % | only sub-4 s stretches survived |
| `F001_T4` | 17 278 | range 5.5 %, short 94.5 % | dropouts chopped the trace below 4 s |
| `F042_T7` | 2 114 | `short_segment` 100 % | the whole session is 2.1 s long |

All six remain **shards with a full schema** (all-`NaN` signal, empty segment
table in its fixed dtypes). The artefact never rules a session unusable; only a
consumer does. Note the split: **Gate 1 alone wipes 2**, the **4 s rule costs the
other 4** — and `F042_T7` is simply a 2.1 s recording, not corrupt data.

## 4. Parity gate

`runners/run_build_bp_artefact.py --check 200`, seeded random sample of the 1400
shards: **PASS — 200/200 bit-identical on `signal` (NaN-aware), `mask`, `cause`
and the segment table**, in 4.6 s. The sample includes fully valid sessions,
multi-segment sessions (`M045_T9`: 4 segments, 66.6 % valid) and one
zero-segment session. Report: `check200.json` beside the artefact.

## 5. The ADR §1 corpus statistics, re-measured

`analysis/bp/corpus_stats.py` reads the raw traces only and writes nothing into
the artefact. Per-session rows: `corpus_stats.json` beside the artefact.

| ADR §1 claim | re-measured | verdict |
| :--- | :--- | :--- |
| 1400 sessions | **1400** readable | matches |
| 62 405 029 samples, 17.3 h | **62 405 029**, 17.33 h | matches |
| 59 sessions with a sample below 30 mmHg | **59** | matches |
| 41 with a negative sample | **41** | matches |
| dropouts near −4 to −7 mmHg | global min **−7.2827** | matches |
| longest sub-30 plateau 52.53 s | **52.532 s** (`F039_T8`) | matches |
| 4 sessions above 250 mmHg, max 414.90 | **4**, max **414.897** | matches |
| longest run of exactly equal samples 11 ms | **11.0 ms** | matches |
| no non-finite samples | **0** | matches |

One fact the ADR does not state: **2 sessions have no sample at or above
30 mmHg at all** (`F030_T10`, `M049_T10`) — these are the two shards Gate 1 wipes
completely.

## 6. ADR rollback observables

| observable | ADR rollback condition | measured |
| :--- | :--- | :--- |
| share of samples with cause `2` | (a) Gate 1 rejects genuine extremes | **0.78 %**; wipes **2** sessions entirely |
| sessions that cause `2` alone kills | (a) | **2** (both have zero in-range samples) |
| share of samples with cause `4` | (b) Gate 3 rejects normal pulse morphology | **0.00 %** |
| longest equal run (Gate 2 input) | (b)-adjacent | **11 ms** vs a 200 ms threshold |
| longest dwell at a raw extremum (Gate 3 input) | (b)-adjacent | **5 ms** vs a 30 ms threshold |
| share of samples with cause `5` | (c) 4 s minimum does not fit supported clips | **0.25 %**; costs **4** sessions |
| segments shorter than 8 s | (c) | **2.1 %** |
| segments shorter than the shortest clip a consumer asks for | (c) | **unknown — no consumer exists yet** (plan §9 item 3) |

**Gates 2 and 3 remove nothing on this corpus** — the ADR's "safety nets, not
filters" claim is now measured rather than inherited: the longest invariant run
is 11 ms against a 200 ms limit, and the longest extremum dwell is 5 ms against a
30 ms limit. Gate 1 and the 4 s rule are the only rules that ever fire, and the 4 s
rule costs five times as many sessions (4 vs 2) as the range gate.

## 7. What must not be read from this note

- **Not a model result.** No stream, no encoder, no loss is involved.
- **`valid_frac` is not signal quality.** A session can be 100 % "valid" by these
  gates and still be a noisy or non-pulsatile trace; the gates are validity
  plumbing, not a quality score.
- **The 0.78 % `gate1_range` share is a policy cost, not an error rate.** Those
  samples are removed *by decision* (ADR §4), including 2 sessions' worth of them.
- **The six zero-segment shards will yield no clip for any consumer.** That is
  correct behaviour, and it is a consumer-visible exclusion of 6/1400 sessions
  (0.43 %), not a build defect.
- **The `--check 200` pass says the build is deterministic, not that the gates are
  right.** Correctness of the thresholds is the ADR's decision, tested in
  `tests/test_bp_gates.py`, not something parity can establish.
