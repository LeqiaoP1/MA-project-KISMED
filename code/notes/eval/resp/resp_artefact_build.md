# Respiration session artefact — build accounting (2026-11-20)

The respiration artefact is **modality-independent**: one artefact serves every
consumer of the `resp` stream (`tir_roi_resp`, `rgb_roi_resp`, the rgb2d/trimodal
fusions, and the Stage-3 waveform evaluators), because it stores a whole-session
grid and a per-sample validity mask and knows nothing about any visual stream.
This note therefore lives here, not under a visual lineage.

What is measured: the **artefact** (what each shard contains) and the build's
accounting. No GPU time is spent; every number is a dataset statistic, not a
model result.

Artefact contract: `code/notes/design/ADR-003-resp-session-cache-artefact.md`.
Build order of record: `code/notes/plan/features/ADR-003-resp-session-cache-plan.md`.
The **TIR-specific** clip accounting of an earlier artefact lineage remains in
`code/notes/eval/archive/tir_resp/resp_preprocess_impact.md` (that note is about the TIR
dataset's clip counts at its 8 s / 2 s hop, and is not comparable with this one).

---

## 1. Build

`2026-11-20`, `raw_root=/work/projects/l0003511/test`, single process, CPU only:

```
respgrid_src1000_fs100_sat9.95-1s_flat0.3-5s_run8s_v1_aedd3ff84e
```

| quantity | value |
| :--- | :--- |
| subject-tasks discovered | **1400** (140 subjects) |
| shards written | **1399** |
| skipped | **1** — `F082_T1`, `missing_resp_volts` |
| missing | 0 |
| wall time | **33.7 s** |
| shard bytes | 41.7 MB (50 MB on disk with `params.json` + manifest) |
| per-session cost | 0.01–0.10 s |

## 2. Validity accounting

`manifest.json` over the 1399 shards:

| statistic | value |
| :--- | :--- |
| `valid_frac` mean / median | **0.850 / 1.000** |
| `valid_frac` min / max | 0.000 / 1.000 |
| wholly invalid shards (`valid_frac == 0`) | **149** (`F004_*`, `F010_*`, …) |
| invalid samples — flatline | **9.2 %** (571 823 / 6 243 308) |
| invalid samples — saturation | **2.8 %** (171 811) |
| invalid samples — short run | **2.6 %** (164 285) |
| usable share | **85.4 %** (5 331 389) |

The 149 all-invalid sessions are **shards, not skips**: the artefact never rules a
session unusable (ADR Criterion 7). They will produce no clips downstream, which
is a consumer decision, not a build decision.

## 3. Parity gate

`runners/run_build_resp_cache.py --check 200`, seeded random sample of the 1399
shards: **PASS** — 200/200 bit-identical on `signal` (NaN-aware), `mask` and
`cause`, in 6.7 s. The sample includes sessions at 100 %, 69 %, 26 % and 0 %
validity.

## 4. Measured source rate — the ADR's open question, answered negatively

Step 27a of the plan; per-session table:
`code/notes/eval/resp/resp_rate_spread.csv`. Measurement:
`f_meas = N_src / duration_s` with `duration_s` taken from the paired thermal
video's **metadata** (`num_frames / fps`, no pixel decode).

| quantity | value |
| :--- | :--- |
| sessions measured | **1384 / 1400** (16 have no readable video metadata) |
| `f_meas` min / median / max | **38.8 / 390.5 / 1599.6 Hz** |
| IQR | 11.0 Hz |
| sessions in 300–600 Hz | **1046** |
| sessions in 900–1100 Hz | **317** |
| sessions above 1100 Hz | **20** |
| sessions within 2 % of 1000 Hz | **0** |

**The thermal timeline is not a usable duration of record.** The ratio is
bimodal: ~75 % of sessions imply ≈390 Hz and ~23 % imply ≈1000 Hz, i.e. the video
covers a window ~2.57× longer than the physiology for most sessions. No tolerance
can be derived from this source, so **ADR Criterion 6's reopening tolerance stays
`[TODO: verify]`** — nothing was invented to fill it.

**Which convention is right is settled by physiology, not by the metadata.** On a
random 38 sessions, the dominant spectral peak of the trace, read at each
candidate rate:

| rate assumed | median peak | share inside 8–30 bpm |
| :--- | :--- | :--- |
| **1000 Hz (nominal, shipped)** | **14.6 bpm** | **100.0 %** |
| 390 Hz (video-derived) | 5.7 bpm | 10.5 % |

5.7 bpm is not a resting breathing rate; 14.6 bpm is. The nominal 1000 Hz
convention is therefore the physiologically correct one, which is what ADR §2.3
assumed when it excluded the measured rate from the artefact identity — now with
evidence rather than assertion. The measurement stays provenance-only.

## 5. Points clarified while implementing

* **The flatline probe is 5001 samples, not 5000.** `round(5 s × 1 kHz)` is even,
  and SciPy only aligns a rolling window to its START for an odd size. At 0.02 %
  longer this changes no decision; the alternative (a window silently offset by
  one sample) is worse.
* **"A 0.05 Hz breath survives" is amplitude-dependent.** The peak-to-peak spread
  over a window $w$ at an extremum of a sine of frequency $f$ and amplitude $A$ is
  $A\,(1-\cos(\pi f w)) = 0.293A$ at $f = 0.05$ Hz, $w = 5$ s. The 0.3 V probe
  therefore spares 0.05 Hz only for **$A > 1.02$ V**. Real belts swing several
  volts (the rail is at 10 V), so this is not a practical limit — but it is the
  boundary, and it is pinned by a test instead of assumed away.
* **`find_cache` separates mismatch from absence**: a candidate present with the
  wrong fields raises `ValueError` with a field-by-field diff; nothing present
  raises `FileNotFoundError`. Only the first means "you asked for another grid".
* Two build constraints the artefact adds: `fs_out` must **divide** `fs_in`
  (integral decimation is what lands every run on exact master-grid instants), and
  `taps_per_phase >= 10` (the run-edge pad and the discarded output count are
  derived from the filter's real support — 100 source samples at 1000 → 100 Hz,
  not a guessed 20).

## 6. What must not be read from this note

* **No model result.** The build spends no GPU time; every number is a dataset
  statistic.
* **No A/B against the previous artefact lineage.** Those `v2` builds were
  deleted, so the delta in clips/sessions cannot be recomputed. The label space
  also changed with the implementation (same run-mean-removed volts, but a
  correctly padded resampler and a mask with no support dilation), so any retrain
  needs a new run prefix.
* `valid_frac` is a **coverage** statistic. A high value says nothing about
  whether the remaining signal is respiration.
* A small-subset "spread" quoted against the thermal timeline is a spread **at the
  level of 390 Hz**, not evidence that 390 Hz is right — see §4.
