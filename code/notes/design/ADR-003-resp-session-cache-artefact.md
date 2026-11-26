# ADR-003: Precomputed Respiration Session Artefact (Whole-Session Grid + Validity Mask)

- **Status**: Accepted
- **Date**: 2026-11-20
- **Author**: Ne95ocyg
- **Related Spec/Plan**: `code/notes/plan/features/ADR-003-resp-session-cache-plan.md`
- **Supersedes**: `N/A`

**Scope.** This ADR makes one decision: **whether to clean respiration once, offline, into a cached per-session artefact, or to clean it repeatedly when each consumer loads a clip.** The rail and flatline thresholds, per-run offset removal, resampling, and output grid are fixed inputs to this decision; they are not reopened here. **One deviation is recorded: the raw source rate remains the nominal 1000 Hz instead of being measured** (§2.3). Missing values are marked `[TODO: verify]` rather than guessed.

--------------------------------------------------------------------------------

## 1. Context & Problem Statement

Each session contains one continuous 1 kHz voltage trace from a contact respiration sensor. Some parts of the trace are not respiration: the sensor can detach, the cable can come loose, or the recorder can hit its ±10 V clamp. A DC estimate over the whole session is then dominated by the clamped baseline. In one corpus session, the signal is at −10 V for 40–80 % of the samples. The offset must therefore be removed separately for each valid run, and a clip must never cross a break.

Two consequences fix the shape of the decision.

1. **Decide validity on the raw signal first**, before offset removal or normalisation. A break is then a property of the session, not of whichever window a consumer happens to cut. The result is one boolean mask per session, and that mask is the sole authority on validity.
2. **Use contiguous runs only internally.** A short or absent run creates a hole in the artefact. It does not redefine the session start or which windows exist.

This stage does not choose clips. Clip length, stride, split, and normalisation remain consumer decisions. One artefact can therefore serve pretraining with a 2.0 s hop, fine-tuning with a 1.0 s hop, and other clip lengths without rebuilding it.

## 2. Decision & Parameter Exposure

**Clean each session once, offline, into a cached artefact containing a whole-session signal grid and a per-sample validity mask in absolute session time. Consumers read the artefact and cut windows from it; they never clean the raw trace again.**

### 2.1 What is stored

Each session has one shard containing:

- the offset-removed signal on the output grid, at its absolute sample positions;
- a boolean validity mask;
- a per-sample cause code identifying the rejection rule (rail, flatline, run too short, or non-finite); and
- build provenance: the source sample count and both sample rates.

This information makes a shard self-verifying without reopening the raw trace. Invalid signal samples are **NaN**, never zero.

### 2.2 The staged contract — the order is part of the decision

| Step | What happens | Why it is in this position |
|---|---|---|
| **1. Decide validity** | Inspect the **raw trace at the source rate**. Reject rail events where the magnitude exceeds 9.95 V for at least 1 s, and flatline regions covered by any 5 s window with a peak-to-peak spread below 0.3 V. | This produces one boolean mask per session, and that mask **alone** defines validity. Deciding before filtering keeps the result independent of later processing. |
| **2. Group into contiguous runs** | Find maximal valid stretches. A stretch shorter than the 8 s minimum usable run becomes a hole. | Runs are an internal mechanism and are not stored. They prevent offset estimation across a break. The session is still emitted. |
| **3. Remove each run's offset, then resample** | Subtract each run's mean, then anti-alias resample that run onto the output grid. | No high-pass filter is used. Subtracting a constant introduces no transient, so no edge guard is needed and waveform shape and amplitude are preserved. |
| **4. Emit** | Write the grid and mask in absolute session time. | Nothing is dropped or tested again. Invalid stretches are holes, not missing clips. |

### 2.3 The one change: the source rate stays at the nominal 1000 Hz

The trace is treated as sampled at **1000 Hz nominal**, a corpus-wide constant. **The measured rate is deliberately excluded from the artefact identity.** This has two reasons, and one known gap:

- the frame-to-sample mapping between respiration and thermal time is a corpus-wide convention defined against the nominal rate. Correcting the rate would redefine that mapping for every session;
- each shard records its source sample count and both rates, so the artefact can be verified without the raw file. A value that is not recorded cannot be part of the artefact identity.

**The measurement itself is not recorded.** There is no per-session value or corpus-wide spread for the measured rate. Only the decision not to use it is recorded. The tolerance that would reopen this decision is therefore undefined: `[TODO: verify: record the measured rate per session, publish the corpus spread, and fix the reopening tolerance]`.

The **output grid** is different. The artefact is emitted on this grid at absolute session positions, and consumers address windows on it. The grid is part of the artefact contract, not a consumer choice.

### 2.4 What may never enter the artefact

| Excluded | Why |
|---|---|
| Clip length and hop | These are consumer choices; baking them in would force one rebuild per consumer and destroy the single-artefact property (a 2.0 s hop and a 1.0 s hop must share one build). |
| Split, fold, task subset | They select windows; they do not change a stored sample. |
| Normalisation | It rescales; the label normalisation is applied downstream and must remain single. |

A consumer cannot request a clip **longer** than the minimum run, because that window could cross a break. Clip-length flexibility is therefore bounded by the minimum run; changing that bound requires a new artefact.

### 2.5 Consumer contract

| ID | Must hold |
|---|---|
| **I1** | Validity is decided once, on raw volts, before any offset removal or normalisation; the mask is the only statement of validity. |
| **I2** | No offset is estimated across a break: the offset is removed per contiguous run. |
| **I3** | Every readable session gets a shard. The artefact never rules a session unusable; only a consumer may. |
| **I4** | Nothing is dropped: invalid stretches and runs shorter than the minimum are holes, and a hole is NaN, never 0, so an unmasked read fails loudly. |
| **I5** | A window of `n` output samples starting at index `s` is usable iff every mask sample inside it is valid, with `s` and `n` derived from the output rate and the requested start time and length. |
| **I6** | A consumer verifies the artefact's identity before using it and refuses a mismatch rather than reading a different grid. |
| **I7** | The respiration artefact and the pixel artefact have disjoint identities and independent versions: neither rebuild forces the other, and a dataset uses the intersection of their coverage, joined by session name. |
| **I8** | The evaluation reference is read from the **same** artefact as the training label, so no reported number compares two preprocessing spaces. |
| **I9** | Any change to the validity thresholds, the minimum usable run, the source rate or the output rate changes the artefact's identity, so a stale artefact cannot be reused silently; such a change is a new lineage and needs a retrain. |

### 2.6 Exposed quantities

No new training option is introduced. The artefact fixes and records the following about itself, so a reader can test it without reading an implementation:

| Quantity | Value | Role |
|---|---|---|
| Source rate | **1000 Hz nominal** | **The change**: corpus constant, part of the identity (§2.3) |
| Output grid rate | fixed when the artefact is built, recorded in each shard | Windows are addressed in this grid: `s = round(t_start · fs_out)`, `n = round(clip_seconds · fs_out)` |
| Rail threshold / alarm duration | 9.95 V / at least 1 s | Event-based: a sample is invalid only inside a rail event at least this long, so a brief graze is kept |
| Flatline probe | 5 s window, 0.3 V peak-to-peak | Long enough that a 0.05 Hz breath (10 s period) cannot be mistaken for a dead channel |
| Minimum run | 8 s | Valid runs shorter than this become holes; the session is still emitted |
| Invalid representation | NaN, never 0 | An unmasked read fails loudly |

These are the quantities the artefact is built under, and each of them changes a stored sample, so a change to any one of them produces a different artefact rather than a tweak.

### Rationale

- **Validity must be decided before a window exists.** If the window is cut first, its validity is a property of the cut, and every consumer that cuts differently gets a different answer. Deciding on the whole session makes the answer a property of the session.
- **A per-run offset is the only offset that means anything.** A whole-session mean is dominated by the clamp (one session reads −10 V for 40–80 % of its samples); a high-pass filter would remove the break but ring at every edge. Subtracting each run's own constant does neither.
- **One artefact for every consumer is the point.** Clip length, hop, split, task subset and normalisation are all consumer decisions, so they must not be inside the artefact — that is exactly what lets one build serve pretraining, fine-tuning and evaluation. It is also what guarantees reference and label live in one space (I8).
- **Holes, not zeros.** A zero would silently become a training target; NaN cannot. Invalidity therefore propagates as an error until a consumer explicitly masks it.
- **The artefact never rules a session unusable.** Usability depends on the question being asked — a session with one clean minute is useless for long clips and useful for short ones. Only the consumer knows.
- **Fail-loudly identity.** Silent data drift — a run reading a slightly different grid — is the one failure mode that invalidates every comparison while looking healthy at every step.
- **1000 Hz nominal.** §2.3: the frame-to-sample mapping is a corpus-wide convention, and the artefact identity carries one recorded source rate.

### Rejected alternatives

- **Clean at clip load time.** Rejected because training labels and evaluation references would be cleaned separately and could diverge. Cleaning once on the raw signal makes the mask a property of the session.
- **Remove one whole-session DC offset.** Rejected because the clamped baseline dominates the estimate, leaving the real signal at the wrong level.
- **Use a high-pass filter to remove the offset.** Rejected because it introduces a transient at every break and changes waveform shape, which is the quantity measured by the thesis.
- **Store invalid samples as zero, or drop unusable sessions during the build.** Rejected because both choices are silent: zero can become a training target, while dropping moves the usability decision into the builder.
- **Bake clip length, hop, or normalisation into the artefact.** Rejected because each consumer would require a rebuild and the reference and label spaces could diverge.
- **Adopt a measured per-session or corpus-wide source rate.** Rejected under §2.3 because frame-to-sample mapping is a corpus-wide convention and the artefact identity carries one recorded source rate. The measured rate is therefore deliberately excluded.
- **Resample the raw trace to exactly 1000 Hz at ingest.** Rejected because it adds an interpolation stage whose transient would need guarding at every run edge, unlike per-run offset removal. It would correct a rate whose measured size is not recorded.

--------------------------------------------------------------------------------

## 3. Falsifiable Criteria

- **Criterion 1 — the mask is the sole authority.** A window of `n` samples starting at index `s` is usable if and only if every mask sample in the window is true. The indices are `s = round(t_start · fs_out)` and `n = round(clip_seconds · fs_out)`. Stored values are volts minus the mean of their run, so amplitude is preserved. A window that overlaps a hole is unusable, and no invalid sample may be silently replaced with zero. Holes are NaN, never 0, so an unmasked read fails loudly.
- **Criterion 2 — rail rejection is event-based.** A sample is invalid only when it lies inside a rail event with magnitude above 9.95 V lasting at least 1 s. Under the current contract, a **0.5 s** graze is kept and a **1.5 s** block is removed as saturation. Both probes must reproduce these results: no sample in a sub-second contact is rejected, and no sample in a multi-second block survives.
- **Criterion 3 — the flatline probe does not remove slow breathing.** A sample is invalid if it lies in any 5 s window whose peak-to-peak spread is below 0.3 V. A 0.05 Hz breath (10 s period) must remain valid, while a frozen or barely ringing channel must not. The 0.3 V operating point is intentional; the more costly 0.5 V alternative was rejected. This criterion fixes the operating point without claiming that the probe physically separates the two cases.
- **Criterion 4 — per-run offset removal introduces no transient.** Remove each contiguous valid run's mean, then verify that the resampled run has no edge artefact. The recorded analytic thresholds are an out-of-band residual at 130 Hz of **2.4e-4 V** and a run-edge gradient below **1e-3 V**. Both must hold after a rebuild, so no additional invalid guard is added.
- **Criterion 5 — the artefact encodes no consumer choice.** Rebuilding with only a different clip length or hop must leave stored samples unchanged. One artefact must therefore serve both the 2.0 s-hop and 1.0 s-hop stages. A shard must contain no window, stride, split, or normalisation choice.
- **Criterion 6 — the source rate.** Every artefact records **1000 Hz nominal** as its source rate, and the measured rate is excluded from its identity. The measured-rate spread is **not recorded**, so no tolerance can be quoted yet: `[TODO: verify: record the measured rate per session, publish the corpus spread, and fix the tolerance beyond which this decision is reopened]`.
- **Criterion 7 — the stage never rules a session unusable.** Every session with a readable trace must produce a shard, including a session whose whole grid is invalid. Only a consumer may decide that the session is unusable. The threshold is zero readable sessions missing from a build.
- **Criterion 8 — smoke and resource.** A build restricted to a small session subset must produce a shard for every readable session. Every shard must load and satisfy Criterion 1. This stage runs before training and uses no GPU time, so the resource threshold is `N/A`.

- **Evaluation Command(s)** (downstream): the session, k-fold, and subject evaluators. Each reads the reference from the same artefact as the label.
- **Acceptance Artifact(s)**: the per-session and per-subject reports produced by those evaluators. These reports have no fixed acceptance threshold. Because the label space is run-mean-removed volts, earlier numbers belong to a different lineage; the reports are **descriptive** here. The acceptance gates for this ADR are Criteria 1–8.

--------------------------------------------------------------------------------

## 4. Consequences & Trade-offs

### Positive

- **Validity is decided once on raw volts**, before offset removal or normalisation. The mask is the sole authority, so later stages cannot quietly overrule it.
- **Nothing is dropped.** Invalid stretches and short runs become holes in the artefact rather than missing clips.
- **One artefact serves every consumer.** Clip length, stride, split, and normalisation remain downstream, so the 2.0 s-hop and 1.0 s-hop stages share one build.
- **Labels and references come from the same artefact**, so reported numbers do not compare two preprocessing spaces.
- **No transient is introduced.** Per-run constant subtraction does not require a high-pass filter or edge guard, preserving waveform shape and amplitude.
- **Invalidity is explicit.** Holes are NaN, so an unmasked read fails instead of silently training on zeros.
- **The two artefacts remain independent.** Pixel and respiration artefacts have separate identities and versions. Rebuilding one does not force a rebuild of the other, and a dataset uses the intersection of their coverage.

### Negative / Technical Debt

- **The label space becomes run-mean-removed volts.** Earlier numbers belong to a different lineage, so retraining is required. The data work uses no GPU time, but it still takes time.
- **The validity thresholds are policy choices, not measurements.** The 0.3 V peak-to-peak threshold is an operating point; the more costly 0.5 V alternative was rejected after measuring its impact. Changing a threshold creates a new artefact rather than a small adjustment.
- **Some short valid data is lost.** A valid run shorter than 8 s becomes a hole even if the sensor was working, so a session can lose an otherwise usable tail.
- **Clip length is bounded by the minimum run.** A window must fit inside one valid run. A consumer requesting longer clips gets fewer usable windows, not a longer artefact.
- **The measured rate is not corrected**, and the size of the mismatch remains unquantified (§3, Criterion 6).
- **A wholly invalid session remains in the corpus.** The stage never declares a session unusable, so the shard remains and every consumer must tolerate it.

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A`. This is a data-pipeline decision recorded before training. It is verified before the first run and uses no GPU time.
- **Evidence Artifacts**: the built artefact and its build accounting, recorded in `code/notes/eval/resp/resp_artefact_build.md` (re-homed 2026-11-20: this artefact is **modality-independent** — one grid serves `tir_roi_resp`, `rgb_roi_resp` and the rgb2d/trimodal fusions — so its evidence does not belong under a visual lineage. The TIR dataset's clip accounting stays in `code/notes/eval/archive/tir_resp/resp_preprocess_impact.md`, where every number depends on that dataset's hop and geometry); plus the current contract values: a 0.5 s rail graze kept, a 1.5 s block removed, an out-of-band residual of 2.4e-4 V at 130 Hz, and a run-edge gradient below 1e-3 V.
- **Tests / Smoke Checks**: Criterion 1 (mask and NaN), Criteria 2 and 3 (rail and slow-breath probes), Criterion 4 (analytic residual and run-edge gradient), Criterion 5 (rebuild invariance under a different hop or clip length), Criterion 7 (no readable session missing), and the subset build in Criterion 8.
- **Rollback / Rejection Condition**: Reopen this decision if **(a)** the measured-rate spread in Criterion 6 is large enough to change window arithmetic, **(b)** a consumer needs a window that cannot fit inside one valid run, or **(c)** the validity thresholds no longer describe the sensor's failure modes. Any change to those thresholds, the minimum run, or the source rate creates a new artefact **and** requires retraining; it must never be an in-place edit. Drop the artefact only if load-time cleaning is shown equivalent in subject-level evaluation, not merely because it is more convenient.
