# Respiration Session Preprocessing (cached artefact)

Cleans one continuous 1000 Hz `Resp_Volts.txt` session and caches the result as a **whole-task signal grid plus a per-sample validity mask, in absolute session time**. It chooses no clips: clip length, stride, split and normalisation stay with the consumer (§3).

## 1. Why this stage exists

Sensor detachments, cable unclips and the recorder's $\pm 10\text{ V}$ clamp corrupt parts of a session. A DC estimate taken over a whole session is dominated by the clamped baseline (one corpus session reads $-10\text{ V}$ for 40-80 % of its samples), so the offset has to be removed per valid run instead, and a clip window must never straddle a break. Validity is therefore decided on the RAW signal first. Contiguous runs are an internal device only: a short or absent run leaves a hole in the artefact, it never redefines where a session starts or which windows exist.

---

## 2. Fundamental workflow

1. **Decide validity** (raw signal, source rate). 
    - Saturation: A sample is invalid if it touches the rail ($|V| \gt 9.95\text{ V}$) as saturation alarm and within a saturation alarm window lasting at least for 1 second.
    - Flatline: A sample is invalid if it is in 5s-window whose peak-to-peak spread is $< 0.3\text{ V}$ (disfunctioning or fronzen sensor). 
    
    The result is one boolean mask per session, and that mask alone defines validity.

2. **Group into contiguous runs** (internal). Each run is offset-corrected on its own, so a window must never straddle a break. Too short run **Runs shorter than 8 s** (current "clip duration" configurated as 8 seconds) marked unusable; the session itself is still emitted.

3. **Remove the DC offset per run, then resample.** Subtract each run's own mean and anti-alias resample it onto the output grid. No high-pass filter is used: subtracting a constant introduces no transient, so there is no edge guard and waveform shape and amplitude are preserved.

4. **Emit the session artefact**: the DC-removed signal on the whole-task output grid at its absolute sample positions, plus the mask.

---

## 3. Using the stage

- Read the session's signal and mask. A window of `n` samples starting at index `s` is usable iff every `mask[s : s + n]` is true, with `s = round(t_start * fs_out)` and `n = round(clip_seconds * fs_out)`. Values are volts minus their run's mean, so amplitude is preserved.
- Invalid samples are `NaN` in the signal, never `0`, so an unmasked read fails loudly.
- Normalisation (`physio_norm`), `clip_stride`, `task_set` and the subject split stay downstream. One cache therefore serves Stage 2 (2.0 s hop) and Stage 3 (1.0 s hop), and any clip length, without a rebuild.
- Every session with a readable `Resp_Volts.txt` gets a shard; this stage never rules a session unusable, only a consumer may.
- It joins the TIR-ROI cache by session name `<SUBJECT>_<TASK>`. The two caches have disjoint keys and independent versions, so neither rebuild forces the other, and the dataset uses the intersection of their coverage. Thermal frame `f` maps to respiration sample `round((f / fps) * fs_out)`.

---

## 4. Guardrails

- **Validity is decided on raw volts**, before any DC removal or normalisation; the mask is the only statement of validity.
- **No offset is estimated across a break**: DC is removed per run. Nothing is dropped -- invalid stretches and runs shorter than 8 s are holes in the artefact, not missing clips.
- **No window, stride, split or normalisation enters the artefact**, which is what keeps one cache valid for every consumer.


---

## 5. Implementation status and resolutions (2026-10-31)

**Implemented.** `data/resp_preprocess.py` (validity mask, per-run DC removal,
anti-alias resample) + `data/resp_cache.py` (key, shard IO, fail-loudly lookup),
built by `runners/run_build_resp_cache.py`; consumed by
`data/tir_resp_dataset.py` (`--resp_cache`), the six HPC/local drivers, and the
session / k-fold / subject evaluators (the reference is read from the same
artefact). Build order and the conflict table against the shipped pipeline:
`plan/RESP_session_preprocess_implementation_plan.md`. Measured output:
`eval/tir_resp/resp_preprocess_impact.md`.

Resolutions of the open items:

* **R1 — `sat_min_s` semantics.** EVENT-based (decision D5): a sample is invalid
  only when it sits in a rail event at `|V| > sat_v` lasting `>= sat_min_s`.
  The design's 2026-10-31 revision set `sat_min_s = 1 s` and
  `flat_ptp_v = 0.3 V`; measured at that contract, a **0.5 s** graze is KEPT and
  a **1.5 s** block is removed with cause 1. Why the flatline probe sits at
  0.3 V (and what 0.5 V cost): `eval/tir_resp/resp_preprocess_impact.md` §1.2.
* **R2 — slow breath vs the flat probe.** The flatline window is **5 s** (not
  2 s), so a 0.05 Hz breath (10 s period) cannot be mistaken for a dead channel.
* **R3 — run-edge transient.** The run's own edge values are replicated by
  `RESAMPLE_HALF_TAPS = 20` source samples before `resample_poly`, and the
  conservative grid mask excludes the filter support around every break. The G2
  analytic gate measures an out-of-band (130 Hz) residual of 2.4e-4 V, and the
  run-edge gradient stays below the 1e-3 V threshold, so no extra invalid guard
  was added.
* **R4 — nominal `fs_in`.** `fs_in = 1000` is kept (the frame <-> sample mapping
  is a corpus-wide convention); the measured rate is deliberately not in the key.
  Each shard stores `n_src` / `fs_in` / `fs_out`, so it is self-verifying without
  the raw file.
* **R5 — `physio_norm: session`.** Statistics are now `nanmean` / `nanstd` over
  the stored grid (valid samples only), documented in `_resp_full`.
* **R6 — retrain required.** The label space changed to run-mean-removed volts;
  the plan stops at "data pipeline verified" and spends no GPU time.
