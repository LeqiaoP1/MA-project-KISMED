# Respiration Session Preprocessing (cached artefact)

Cleans one continuous 1000 Hz `Resp_Volts.txt` session and caches the result as a **whole-task signal grid plus a per-sample validity mask, in absolute session time**. It chooses no clips: clip length, stride, split and normalisation stay with the consumer (§3).

## 1. Why this stage exists

Sensor detachments, cable unclips and the recorder's $\pm 10\text{ V}$ clamp corrupt parts of a session. A DC estimate taken over a whole session is dominated by the clamped baseline (one corpus session reads $-10\text{ V}$ for 40-80 % of its samples), so the offset has to be removed per valid run instead, and a clip window must never straddle a break. Validity is therefore decided on the RAW signal first. Contiguous runs are an internal device only: a short or absent run leaves a hole in the artefact, it never redefines where a session starts or which windows exist.

---

## 2. Fundamental workflow

1. **Decide validity** (raw signal, source rate). 
    - Saturation: A sample is invalid if it touches the rail ($|V| \gt 9.95\text{ V}$) as saturation alarm and within a saturation alarm window lasting at least for 2 seconds.
    - Flatline: A sample is invalid if it is in 5s-window whose peak-to-peak spread is $< 0.05\text{ V}$ (disfunctioning or fronzen sensor). 
    
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

