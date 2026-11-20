# ADR-004: Precomputed Blood-Pressure Session Artefact (Validity Mask + Segment-Level DC Removal)

- **Status**: Accepted
- **Date**: 2026-11-20
- **Author**: Ne95ocyg
- **Related Spec/Plan**: `N/A` (implementation-defined; no runtime dependency)
- **Supersedes**: `N/A`

**Decision.** Clean the blood-pressure trace **once, offline**, into one cached
per-session artefact containing a whole-session signal grid, a per-sample validity
mask, and a per-sample cause code. Apply the three validation gates per sample at
the nominal 1000 Hz source rate; invalidate admitted segments shorter than 4 s;
remove the mean of each valid segment; and store the result on the unchanged
1000 Hz source grid. Consumers read the artefact and cut windows from it; they
never clean the raw trace again.

**Scope.** The artefact is self-contained: it depends on no other modality,
artefact, configuration, or code path. Tier-2 detrending and standardization read
the cache but are out of scope here.

--------------------------------------------------------------------------------

## 1. Context & Problem Statement

Each session carries one continuous trace, `BP_mmHg.txt`, sampled at a nominal
**1000 Hz**, in **mmHg**, with no time column. Three failure modes can corrupt
individual samples or stretches of samples:

1. **Out-of-range values.** Sensor-off stretches sit on a low pedestal, not at
   zero. In **59 / 1400** sessions at least one sample is below **30 mmHg**;
   **41** contain a negative sample, and the dropouts are near-constant plateaus
   around **−4 to −7 mmHg**, the longest lasting **52.53 s**. At the other end,
   **4** sessions contain samples above **250 mmHg** (maximum **414.90 mmHg**).
2. **Frozen channels.** A disconnected transducer, a held ADC value, or a
   recalibration failure produces an exact invariant stretch.
3. **Amplifier rail clipping.** A channel pinned at its maximum or minimum holds
   the exact extremum for a measurable dwell; a real pulse does not.

A whole-clip verdict is too coarse: one corrupt sample would discard the clean
remainder of the same window. The rejection unit is therefore the **sample**; the
cache unit is the **whole session**, stored as a grid plus a mask; and a clip is a
consumer-side window that must lie inside one valid segment. The DC level belongs
to the segment, not to a clip that happens to overlap it.

Without one shared cache, every consumer cleans the trace independently, so a
training label and an evaluation reference can become two different cleanings of
the same session. Precomputing the grid once removes that possibility: the cache is
the sole source of BP samples for training, evaluation and analysis.

Corpus reference (all 1400 sessions, 62,405,029 samples, 17.3 h): the longest run
of exactly equal consecutive samples anywhere in the corpus is **11 ms**. The
200 ms and 30 ms rules are therefore safety nets against dead or railed channels,
not filters that remove ordinary pulsatile data in the current corpus.

## 2. Decision & Parameter Exposure

**Build one shard per session, offline. Apply Gates 1 → 2 → 3 per sample at
1000 Hz. Form maximal continuous segments from the surviving samples; invalidate
segments shorter than 4 s; subtract each valid segment's mean; and store the
whole-session signal, validity mask, and cause codes on the 1000 Hz source grid.
Consumers read the shard and cut windows from it. An all-valid window lies inside
one valid segment and therefore carries that segment's offset removal.**

### 2.1 What is stored

Each session produces one independent shard containing:

- **signal** — float32, in mmHg, on the 1000 Hz grid at absolute session positions;
  each valid segment's mean is removed; every invalid sample is **NaN**;
- **mask** — boolean, the sole authority on usability: `True` exactly for samples
  that are admitted by all three gates **and** belong to a valid segment;
- **cause** — uint8 per sample: `0` valid, `1` non-finite, `2` range, `3` flatline,
  `4` rail, `5` short segment. A rejected sample carries the cause of the first
  gate that rejected it;
- **segments** — a table of valid segments: cache-grid start, cache-grid stop,
  removed mean in mmHg and length in seconds;
- **provenance** — source sample count, both sample rates, number of valid
  segments, valid fraction, non-finite count, session id and source file name.

Every shard satisfies `np.isnan(signal) == ~mask`. Samples in short segments are
NaN and masked false; they are never silently stored as admitted values.

### 2.2 Build order — the order is part of the decision

| Step | Operation | Why it is here |
|---|---|---|
| **1. Gate per sample** | On the **raw 1000 Hz mmHg** trace, apply Gate 1 → Gate 2 → Gate 3. | The verdict is made once, per sample, on the raw trace. |
| **2. Form segments** | Find maximal continuous stretches of admitted samples. | Segments define where a DC estimate is allowed to run. |
| **3. Invalidate short segments** | A segment shorter than **4.0 s** becomes invalid: every sample in it is masked false with cause `short_segment` and stored as NaN. | Validity is a property of the segment, and a short segment cannot support a stable DC estimate. |
| **4. Remove each valid segment's mean** | Subtract the valid segment's arithmetic mean from every sample of that segment. | Per-segment subtraction introduces no transient and preserves waveform shape and pulse amplitude. No high-pass filter, no detrending. |
| **5. Emit** | Store each valid segment at its absolute source-sample positions. Write the grid, mask, cause, segment table, and provenance. | No resampling is applied, no segment value crosses a boundary, and no consumer needs the raw trace again. |

### 2.3 Gate definitions

- **Gate 1 — dynamic range.** A sample is rejected if it is **non-finite**
  (cause `gate1_nonfinite`) or if `y_i < 30.0 mmHg` or `y_i > 250.0 mmHg`
  (cause `gate1_range`). The verdict is per sample; neighbours are unaffected.
- **Gate 2 — flatline.** On the Gate-1 survivors, any invariant run that
  **exceeds 200 ms** (strictly more than 200 samples at 1 kHz) has every sample in
  the run rejected (cause `gate2_flatline`). A run of exactly 200 ms is accepted;
  201 ms is rejected. A run is counted only across trace-adjacent samples that
  both survived Gate 1; a Gate-1-rejected sample breaks the run.
- **Gate 3 — amplifier rail clipping.** On the Gate-2 survivors, take the surviving
  extrema `y_min` and `y_max`. Any dwell of **30 ms or longer** (at least 30
  samples at 1 kHz) at `y_max` or at `y_min` has every sample in the dwell rejected
  (cause `gate3_rail`). A dwell of 29 ms is accepted; 30 ms is rejected; an
  interior plateau is not a Gate 3 rejection. A dwell is counted only across
  trace-adjacent samples that both survived Gate 2; a Gate-2-rejected sample breaks
  the dwell.

The non-finite clause is required because comparisons with `NaN` are false and
would otherwise pass the range test. The corpus contains no non-finite samples
today; the rule is a correctness guard.

### 2.4 Sequential evaluation

The gates are applied in the fixed order **1 → 2 → 3**:

1. Gate 1 assigns `gate1_nonfinite` or `gate1_range`;
2. Gate 2 evaluates the Gate-1 survivors and assigns `gate2_flatline`;
3. Gate 3 evaluates the Gate-2 survivors and assigns `gate3_rail`;
4. every remaining sample is admitted.

A sample is evaluated by at most one rejecting gate and carries exactly one cause.
The order fixes both the early-exit cost and the recorded cause. It is mandatory
because each later gate evaluates only the preceding gate's survivors. If a gate
receives no survivors, the cascade ends and the session yields no valid segment.

Gate 3 uses the extrema of the Gate-2 survivors. Using extrema from the original
trace would make Gate 3 unable to fire at a value already removed by Gate 1, such
as a 300 mmHg sample rejected as out of range.

### 2.5 Segment and DC rules

- A segment is a maximal continuous stretch of admitted samples.
- A segment is **valid** only if it lasts at least **4.0 s**; a shorter segment is
  **invalid**.
- Every sample of an invalid segment is masked false (`short_segment`) and stored
  as NaN.
- Each valid segment's own arithmetic mean is removed from that segment.
- No mean is computed over a clip, and no mean crosses a segment boundary.
- Valid segments are stored at their original sample positions; no resampling is
  applied, and invalid samples are NaN.
- A sample carries the cause of the first gate that rejected it. `short_segment`
  is assigned only to samples that passed all three gates and whose segment is
  shorter than 4.0 s.

### 2.6 Cache identity and storage

The cache is a **pure derived artefact**. Its identity is the frozen parameter set:

| Field | Value |
|---|---|
| `format_version` | **1** |
| `corpus` | digest of the discovered session list |
| `fs_in` | **1000 Hz nominal** |
| `fs_out` | **1000 Hz** (same as the source; the cache is full-rate, with no resampling) |
| `gate1_min_mmhg` | **30.0** |
| `gate1_max_mmhg` | **250.0** |
| `flatline_max_ms` | **200** |
| `rail_dwell_min_ms` | **30** |
| `min_segment_s` | **4.0** |
| `store` | signal (float32) + mask (bool) + cause (uint8) + segment table |

The cache key is a readable slug plus a hash of this entire set. One shard is
written per session, atomically. A manifest is derived from the shards, never from
a separate index. A parity check re-preprocesses a sample of sessions from raw and
requires the shard to be **bit-identical** on signal (NaN-aware), mask, cause and
segment table.

A change to any identity field produces a different cache; a stale shard cannot be
read silently.

### 2.7 Consumer contract

| ID | Requirement |
|---|---|
| **C1** | Consumers read the cache; they never read `BP_mmHg.txt` or re-clean it. |
| **C2** | The mask is the sole authority. A window of `n > 0` output samples at index `s` is usable iff `0 <= s`, `s + n <= N_out` and every mask sample in it is true. |
| **C3** | Because every segment boundary and every short segment is masked false, an all-true window lies inside exactly one valid segment and carries that segment's offset. |
| **C4** | Window indices are `s = round(t_start * fs_out)` and `n = round(clip_seconds * fs_out)`. |
| **C5** | The training label and the evaluation reference are read from the **same** shard, so no reported number compares two cleaning spaces. |
| **C6** | A consumer verifies the cache key before reading and refuses a mismatch. |
| **C7** | Clip length, hop, split, fold, task subset and normalisation are consumer choices and are absent from the shard. |
| **C8** | Tier-2 detrending and standardization read usable windows from this cache; they do not modify it. |

### 2.8 What never enters the cache

| Excluded | Reason |
|---|---|
| Clip length and hop | Consumer choices; baking them in forces one rebuild per consumer. |
| Split, fold and task subset | They select windows; they do not change a stored sample. |
| Normalisation | It rescales; the label normalisation stays downstream and must remain single. |
| Any other modality or artefact | This cache is self-contained. |
| Any existing configuration or cache | This cache depends on none. |
| Per-clip offsets | The offset is a segment property, not a clip property. |

### 2.9 Requirements

| ID | Requirement |
|---|---|
| **R1** | Validity is decided once, on raw mmHg at 1000 Hz, before any offset removal or storage. |
| **R2** | The gates are applied in the order 1 → 2 → 3; a rejected sample is never evaluated by a later gate. |
| **R3** | Gate 2 runs and Gate 3 dwells are evaluated on the preceding gate's survivors; a rejected sample breaks a run or dwell. |
| **R4** | Maximal continuous admitted samples form segments; a segment shorter than 4.0 s is invalid. |
| **R5** | The DC level is removed per valid segment, never per clip and never across a segment boundary. |
| **R6** | Invalid samples are NaN and masked false, never zero. |
| **R7** | One shard per session, written atomically; a manifest is derived from the shards. |
| **R8** | The cache key hashes the full identity in §2.6; a mismatch is refused, never read. |
| **R9** | Consumers never clean the raw trace; labels and references come from the same shard. |
| **R10** | The cache encodes no consumer choice: clip length, hop, split, fold, task subset and normalisation are absent. |
| **R11** | The stage is self-contained: it reads only the raw trace and the frozen parameters in §2.6. |
| **R12** | Any change to a gate, the minimum segment or a rate creates a new cache lineage and requires a rebuild and a retrain. |

### 2.10 Implementation freedom

Module boundaries, file names, command spellings, the shard container and the
manifest format are implementation details. The contract is the identity in §2.6,
the gates in §2.3–§2.4, the segment rule in §2.5 and the consumer contract in §2.7.

### Target Modules, Runner(s), Config File(s), Exposed Parameters

- **Target Modules**: `N/A` — implementation-defined; §2.9 fixes behaviour only.
- **Target Runner(s)**: `N/A` — implementation-defined.
- **Config File(s)**: `N/A` — the cache has no configuration dependency.
- **Exposed Parameters**: the identity fields in §2.6. Their external spelling is
  implementation-defined.

--------------------------------------------------------------------------------

## 3. Falsifiable Criteria

- **Criterion 1 — Gate 1 is per sample.** A sample at `29.99 mmHg` is rejected as
  `gate1_range`; `30.00` is admitted by Gate 1. `250.01` is rejected; `250.00` is
  admitted. A non-finite sample is rejected as `gate1_nonfinite`. Neighbours are
  unaffected.

- **Criterion 2 — Gate 2 is per sample.** An invariant run of **201 ms** has its
  samples rejected as `gate2_flatline`; **200 ms** is accepted, and samples outside
  the run are unaffected. A Gate-1-rejected sample inside an otherwise invariant
  stretch breaks the run.

- **Criterion 3 — Gate 3 is per sample.** A dwell of **30 ms** at the Gate-2
  survivors' `y_max` or `y_min` has its samples rejected as `gate3_rail`; **29 ms**
  is accepted. An interior plateau is not rejected, and samples outside the dwell
  are unaffected. A Gate-2-rejected sample breaks the dwell.

- **Criterion 4 — order, survivors and one cause.** A sample rejected by Gate 1 is
  never considered by Gate 2 or Gate 3. A trace containing a `300 mmHg` sample has
  that sample rejected by Gate 1, and Gate 3 evaluates the extrema of the remaining
  samples. Every rejected sample carries exactly one cause.

- **Criterion 5 — segment validity is exact.** A segment of **3.999 s** is invalid:
  every sample is masked false with cause `short_segment`, stored as NaN, and
  absent from the segment table. A segment of **4.000 s** is valid and appears in
  the table.

- **Criterion 6 — DC removal is per segment.** A synthetic trace with two valid
  segments, means 70 and 110 mmHg, separated by rejected samples, must produce two
  stored segments whose means are **0 ± 1e-3 mmHg** in the grid values. No sample
  may carry the other segment's offset.

- **Criterion 7 — the mask is the sole authority and windows do not cross
  segments.** A window is usable iff it is in bounds and all its mask samples are
  true. An all-true window lies inside one valid segment; a window that touches a
  segment boundary or any sample of a short segment is unusable. On every shard,
  `np.isnan(signal) == ~mask`.

- **Criterion 8 — one shard per session.** All **1400** readable sessions produce a
  shard (0 skips, 0 missing); each shard carries signal, mask, cause, segment table
  and provenance, with `n_src` equal to the source sample count and both recorded
  rates equal to 1000 Hz.

- **Criterion 9 — identity and parity.** Rebuilding with the same identity is
  bit-identical on signal (NaN-aware), mask, cause and segment table; changing any
  gate, the 4 s minimum, `fs_in` or `fs_out` produces a different key; a shard
  whose `params` do not match the requested identity is refused.

- **Criterion 10 — no consumer choice in the artefact.** Rebuilding with a
  different clip length or hop leaves every stored array unchanged, and no shard
  contains a window, stride, split, fold, task subset or normalisation field.

- **Criterion 11 — self-containment.** A build performed with only the raw trace
  and the identity parameters present produces the same shard as a build with the
  full corpus available; no read outside the trace and those parameters is
  permitted.

- **Criterion 12 — smoke and resources.** A 5-session subset build produces 5
  shards, and a rebuild of the same 5 is bit-identical. The build uses CPU only, no
  GPU and no accelerator library. A full 1400-session build must complete under
  **10 min** and under **600 MB uncompressed** (the signal + mask + cause grid is
  about 375 MB). VRAM threshold: `N/A`.

- **Evaluation Command(s)**: `N/A` at this level; the implementation defines its
  invocation. The requirements are Criteria 1–12.
- **Acceptance Artifact(s)**: the per-session shards, the manifest, and the parity
  report produced by Criterion 9.

--------------------------------------------------------------------------------

## 4. Consequences & Trade-offs

### Positive

- The trace is cleaned once, offline. Every consumer reads the same grid, so a
  training label and an evaluation reference cannot diverge.
- One corrupt sample no longer discards the clean remainder; the clean stretches
  survive as separate valid segments.
- The mask is the sole authority, and an all-true window cannot cross a segment
  boundary or enter a short segment.
- The DC level is removed per segment, so overlapping windows share one offset and
  no estimate crosses a break.
- Only a segment constant is removed, so waveform shape and pulse pressure are
  preserved.
- The cache identity is hashed and every shard is self-verifying; a stale or
  foreign grid is refused rather than read silently.
- The cache is self-contained: no other modality, artefact, configuration or code
  path is required or changed.

### Negative / Technical Debt

- **The cache must be rebuilt when a gate changes.** Gate values, the 4 s minimum
  or a rate are identity fields; any change is a new lineage and a retrain.
- **Segments shorter than 4 s are lost.** Two clean stretches of 3 s separated by
  one rejected sample are both invalid.
- **Gate 1 rejects genuine extremes by policy.** 59 sessions contain a sample below
  30 mmHg and 4 contain a sample above 250 mmHg; any such sample is removed, even
  if the rest of the trace is usable.
- **Near-constant stretches with any variation are not rejected.** Gate 2 only
  catches exact invariant runs longer than 200 ms; a noisy but frozen channel can
  pass it.
- **Gate 3 is context-dependent.** Its extrema are the Gate-2 survivors' extrema,
  so the same value can be an extremum in one session and an interior value in
  another.
- **Gates 2 and 3 remove nothing on the current corpus.** The longest invariant run
  anywhere is 11 ms, so the 200 ms and 30 ms rules act as safety nets rather than
  filters today.
- **The stored level is segment-relative.** A valid segment carries its own mean
  removal; there is no per-clip or per-session offset.
- **Tier-2 is out of scope.** This cache stops at the DC-removed grid; detrending
  and standardization remain downstream.

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A`. This is a build-time data decision; the corpus
  measurements used no GPU and no scheduler job.
- **Evidence Artifacts**: the corpus statistics in §1, measured over all 1400 raw
  files on **2026-11-20**. Method: per-session parse; min/max and non-finite count;
  contiguous runs of exactly equal samples.
- **Tests / Smoke Checks**: Criteria 1–12, in particular the per-sample boundary
  probes of Criteria 1–3, the survivor-recomputation probe of Criterion 4, the
  segment-validity probe of Criterion 5, the two-segment DC probe of Criterion 6,
  the mask/window probe of Criterion 7, and the parity check of Criterion 9.
- **Rollback / Rejection Condition**: reopen if **(a)** Gate 1's 30 / 250 mmHg
  bounds reject too much genuine data, **(b)** Gate 3's 30 ms dwell rejects normal
  pulse morphology, **(c)** the 4 s segment minimum does not fit the supported clip
  lengths, or **(d)** the nominal 1000 Hz source rate is shown to be wrong. Any
  change to a gate, the minimum segment or a rate creates a new cache and requires
  a rebuild and a retrain; it must never be an in-place edit.
