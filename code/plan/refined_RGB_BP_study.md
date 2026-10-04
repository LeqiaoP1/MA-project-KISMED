# Refined study: RGB visual signals vs. the arterial blood-pressure waveform in BP4D+

**Date:** 2026-09-27
**Scope:** visible-light (RGB) facial video → arterial blood-pressure waveform
**Data:** BP4D+, 40 sessions, 4 subjects (F001–F004 × T1–T10)
**Status:** detection established; measurement (heart rate) not achieved

---

## 1. Summary

Facial RGB video contains a **genuine, skin-borne, motion-independent cardiac signal**
that is detectable in **90 % of sessions**, with a mismatched-subject control firing
at the nominal 3–5 % rate. The visual pipeline is therefore validated **by a positive**,
not merely left unfalsified.

The same signal, however, does **not** yield an accurate heart rate: through an
estimator that reaches 3.4 BPM on the arterial pressure waveform, the face lands at
5.0 BPM against a 5.0 BPM constant-rate baseline (skill ≈ 0).

**Verdict: detection — yes. Measurement — no.**

---

## 2. Research question

> Which 1-D signals can be predicted from which visual 2-D signals, with or without
> landmark-defined regions of interest?

This document answers the **cardiac half** of that question, narrowed to the arterial
pressure waveform. The respiratory half is documented separately
(`SpanMask_PhysioSignals.md`, `TirROI_Resp_plan.md`) and is negative.

---

## 3. Data

| stream | detail |
|---|---|
| RGB frames | `2D+3D/<S>/<T>/%04d.jpg`, 1392 rows × 1040 cols, 25 fps, 8-bit JPEG |
| coverage | 40 sessions, **45 867 frames**, all index-contiguous from 0 |
| arterial BP | `Physiology/<S>/<T>/BP_mmHg.txt`, **1000 Hz continuous pulsatile** waveform |
| heart rate | `Physiology/<S>/<T>/Pulse Rate_BPM.txt` (quantised, ~90 distinct values/session) |
| landmarks | `2DFeatures/<S>_<T>.mat` — 49 (x, y) points + head pose (radians), 25 fps |

### 3.1 The BP target is a real waveform, not a cuff reading

For `F001_T1`: 64 597 samples, range **51.1–119.0 mmHg**, sd 13.5, **2 218 distinct
values** — a finger-cuff arterial trace. Validation:

* band-passed 0.9–3.0 Hz, its dominant frequency is **1.656 Hz = 99.4 BPM**, against
  `Pulse Rate_BPM` mean 95.9 — the target and the spectral machinery agree;
* systolic-peak detection finds **69 peaks against 66 expected** (ratio 1.057,
  sd 0.122 over 40 sessions) — beat detection is sound;
* instantaneous HR from inter-beat intervals correlates only **+0.408** with
  `Pulse Rate_BPM`, which is expected (beat-to-beat HR is noisy) but means the
  instant-by-instant trace must **not** be used as a smooth comparison truth.

### 3.2 Frame alignment

JPEG indices are contiguous and start at 0; landmark `frame` fields run 1..N
monotonically. In 39 of 40 sessions the counts match exactly. `F003_T8` disagrees
(1518 JPEGs vs 1559 landmark frames) and is excluded from this study per instruction.

---

## 4. Signals under test

### 4.1 Regions of interest

* **`face_plain`** — the bounding box of all 49 landmarks, padded 20 %, held static
  per session (the median landmark box). This reproduces the naive "crop the face"
  behaviour and covers eyes, mouth, brows, and any hair or background inside the box.
* **`face_skin`** — the same box with a two-stage skin mask:
  1. **per-frame landmark exclusion** of the brows (1–10), both eyes (20–25, 26–31)
     and the mouth (32–49), each box padded 15–25 % and **tracked with the head**,
     because those regions carry motion rather than blood;
  2. **skin-colour gate** in YCrCb: `133 ≤ Cr ≤ 173`, `77 ≤ Cb ≤ 127`, `Y ≥ 40`,
     which drops hair, background and shadow.
* **`bg`** — a face-free patch (top-left corner, 180 × 180), the negative control.
* **`full`** — the whole frame, the illumination/exposure control.

**Mask behaviour (F001_T1):** retains a mean **34.4 %** of the face box across the
corpus (per-session means 13.4–51.2 %) — selective, not a pass-through. Retained
pixels are **brighter** than the plain box (R 144.9 vs 118.8, G 83.3 vs 69.2,
B 68.9 vs 58.9), because shadow, hair and eye sockets are excluded. Channel sanity:
face is `R > G > B`; background is dark and slightly blue-dominant (R 23.8, G 22.1,
B 34.6) — i.e. not skin.

### 4.2 Signal extraction

Each ROI's per-channel time series is normalised by its session mean (AC/DC) and
band-limited to **0.9–3.0 Hz** (54–180 BPM; 25 fps Nyquist is 12.5 Hz, so this is
comfortably resolved). Three combinations are compared:

| name | formula | source |
|---|---|---|
| `GR` | $G_n - R_n$ | naive baseline |
| `CHROM` | $X - \frac{\sigma_X}{\sigma_Y}Y$, with $X = 3R_n - 2G_n$, $Y = 1.5R_n + G_n - 1.5B_n$ | de Haan & Jeanne (2013) |
| `POS` | $S_1 + \frac{\sigma_{S_1}}{\sigma_{S_2}}S_2$, with $S_1 = G_n - B_n$, $S_2 = -2R_n + G_n + B_n$ | Wang et al. (2017) |

$\sigma$ denotes the standard deviation over the session. This is a **global** AC/DC
normalisation, not the sliding-window form published with CHROM/POS — see §9.

---

## 5. Statistical framework

This is the part that determines whether any of the results can be believed.

**Statistic.** $\max |r|$ between the band-passed BP waveform and the candidate, over
lags of **±2 s in 2-frame steps** (51 lags). The lag sweep is phase-agnostic and
covers the pulse transit time. Reduced to a **scalar** (max over lags *and* channels)
identically for observation and null.

**Null.** 200 **phase-scrambled BP surrogates** — same amplitude spectrum, randomised
phase, DC preserved — evaluated over the **same lag set**, so the multiplicity of the
max is matched exactly. Phase scrambling is the appropriate null because the
physiological claim is *phase locking*, not merely shared spectral content.

**Controls, in increasing order of strength:**

| # | control | what it rules out |
|---|---|---|
| 1 | `bg` and `full` patches | illumination / exposure / sensor artifact |
| 2 | **mismatched-subject pairing** — this session's face signal against a *different subject's* BP | any generic statistical artifact; cannot be gamed by ROI placement |
| 3 | **motion regression** — project out head pose (3), landmark centroid (2), log inter-ocular scale, full-frame mean | head/camera motion masquerading as a pulse |
| 4 | **synthetic pulse ceiling** — a pulse whose instantaneous frequency *is* the measured rate | estimator incapacity |

Control 2 is the decisive one. A real physiological coupling is destroyed by pairing
the wrong subject; only an artifact survives.

---

## 6. Results

### 6.1 Detection — established

`max |r|` vs BP at p < 0.05. Chance is 2.0 of 40.

| signal / method | matched rate | median \|r\| | **mismatched control** |
|---|---|---|---|
| `face_plain` / `GR` | 27/40 (68 %) | 0.49 | **3 %** |
| `face_plain` / `CHROM` | 33/40 (83 %) | 0.62 | **4 %** |
| `face_plain` / `POS` | 32/40 (80 %) | 0.61 | — |
| `face_skin` / `GR` | 27/39 (69 %) | 0.4 | — |
| **`face_skin` / `CHROM`** | **35/39 (90 %)** | **0.6** | **5 %** |
| **`face_skin` / `POS`** | **35/39 (90 %)** | **0.5** | **3 %** |
| `bg` / `GR` *(control)* | 3/40 (8 %) | 0.17 | — |
| `bg` / `CHROM` *(control)* | 4/40 (10 %) | 0.17 | **5 %** |
| `full` / `CHROM` | 32/40 (80 %) | 0.64 | — |

*(n = 130 mismatched pairings per row; one session dropped from the skin rows.)*

**Three findings:**

1. **CHROM/POS beat the naive chrominance difference by ~22 points** (68 % → 90 %).
   This is the largest single improvement in the study.
2. **Skin masking adds ~7 points** on top (83 % → 90 % with CHROM).
3. **The controls are clean and calibrated.** Every mismatched control sits at 3–5 %,
   i.e. nominal, confirming the statistic is sound. The background fires at 8–10 %,
   an order of magnitude below the face.

Matched 90 % against mismatched 3–5 % is an **18–30× effect**.

### 6.2 Mechanism — it is blood, not motion

Projecting head pose (3), landmark centroid (2), inter-ocular scale and the
full-frame mean out of the skin signal:

| signal | raw | after motion regression |
|---|---|---|
| `face_skin` / `CHROM` | 34/39 | **33/39** |
| `face_skin` / `POS` | 34/39 | **32/39** |

The signal **survives** removal of everything kinematic. Independently, the motion
route on its own — head pose and landmark coordinates vs BP — gives
**4/40, median |r| 0.234**, i.e. chance.

**Conclusion:** the signal is **skin colour change at the heart rate = blood volume**.
This is genuine remote photoplethysmography. It is not motion, not illumination, and
not a sensor artifact.

### 6.3 Measurement — not achieved

Heart-rate estimation in sliding windows (spectral peak with parabolic sub-bin
interpolation) against `Pulse Rate_BPM`. `skill = 1 − MAE/baseline`; **skill ≤ 0 means
worse than predicting a constant**.

| window | n | SYNTH (ceiling) | BP (positive control) | `face_skin`/POS | `face_skin`/CHROM | `face_plain`/GR | `bg` |
|---|---|---|---|---|---|---|---|
| 20 s | 26 | **+0.278** (MAE 3.58) | **+0.305** (3.44) | +0.003 (4.94) | −0.117 (5.54) | −1.060 (10.21) | −5.5 … −5.8 (~33) |
| 30 s | 16 | +0.249 (4.05) | +0.255 (4.01) | −0.023 (5.51) | −0.221 (6.58) | −0.703 (9.17) | −5.0 … −5.4 |
| 40 s | 12 | +0.143 (3.08) | +0.025 (3.50) | −0.124 (4.03) | −0.928 (6.92) | −1.380 (8.54) | −7.2 … −8.5 |

Constant-rate baselines: 4.96 / 5.39 / 3.59 BPM. Correlations at 20 s: SYNTH +0.649,
BP +0.628, `face_skin`/CHROM +0.431, `face_skin`/POS +0.429, `face_plain`/GR +0.416.

**Reading:**

* The **estimator is capable and the positive control works.** A clean synthetic pulse
  scores +0.278, and the **arterial pressure waveform through the identical estimator
  scores +0.305** at 3.4 BPM. The whole chain BP → windowed spectral estimate → rate
  is validated, so this is a fair test.
* **Skin masking plus CHROM/POS moved the face from 2× worse than a constant to
  exactly as good as one** (−1.06 → +0.003). A large, real improvement.
* It lands on **zero**. The face detects the pulse but cannot measure its rate.
* The background is catastrophically worse (MAE ~33 BPM, skill −5 to −8), confirming
  the metric is skin-specific.

**Interpretation:** the facial signal has enough cardiac content for *detection*
(§6.1) but not enough narrow-band stability for *rate estimation*. Detection is a
broadband/phase-locked test; rate estimation demands a stable, low-variance spectral
peak. The face supplies the former and not the latter.

---

## 7. Bugs found and corrected

Recorded because each one could have produced a wrong published number.

| # | bug | impact | fix |
|---|---|---|---|
| 1 | Background control boxes were sized to the **face box** (690 × 690 px) and parked in the frame corners, where they overlap each other **and cover the face's left/right halves** | made "background" fire 24/40 vs the face's 27/40 — the control was measuring the face | park controls **strictly outside** the session's landmark bounding box; verify before trusting |
| 2 | `cv2.imread` returns **BGR**, but column 0 was treated as red | the signal labelled `G−R` was actually `G−B`; **CHROM/POS had R and B swapped** | reverse to RGB at storage. Detection unaffected (27/40 both ways); CHROM/POS are the affected results. Permanent sanity check: face ROI must read `R > G > B` |
| 3 | `y.std() < 1e-12` is **False for NaN**, so a NaN session was not skipped and `mean(null >= NaN) = 0.0` scored it as **significant** | would inflate the rate | audit found this touched **1 session**; 35/40 → 35/39, result unchanged |
| 4 | `np.stack` of the three exclusion groups (brows 10, eyes 12, mouth 18 landmarks) — different shapes | worker crash | keep one array per group |
| 5 | `nan_to_num` on landmarks would place lost-landmark boxes at the origin | would zero a corner of the face | keep NaN and skip |
| 6 | Instantaneous peak-detected HR used as the comparison truth | beat-to-beat noise made every skill ≈ 0, wrongly implying the estimator was useless | use one consistent **smoothed** reference for every row |
| 7 | `ndarray.ptp()` removed in numpy 2 | self-test crash | use `np.ptp(arr)` |

Bugs 1–3 each had the potential to reverse a conclusion. Bug 1 did produce a wrong
intermediate result, which was caught only because a second, independent control
(mismatched pairing) disagreed with it.

---

## 8. What is and is not established

**Established (40 sessions, 4 subjects, 0.9–3.0 Hz):**

* Facial RGB carries a **skin-borne, motion-independent** cardiac component.
* It is detectable in **90 % of sessions** with CHROM/POS and skin masking.
* It is **face-specific**: matched 90 % vs mismatched-subject 3–5 %, background 10 %.
* **CHROM/POS substantially outperform** a naive chrominance difference.
* Skin masking provides a smaller, consistent additional gain.
* Heart-rate **measurement is not achieved**: skill ≈ 0 against +0.31 for the BP
  positive control.

**Not established:**

* Any statement about BP4D+ as a dataset — **4 subjects only**.
* Behaviour outside 0.9–3.0 Hz, or at frame rates other than 25 fps.
* Whether a sliding-window CHROM/POS implementation (published form) would do better
  than the global normalisation used here.
* Per-subject skin-colour calibration effects (no tuning was done).
* Absolute timing: BP is a **pressure** trace and the visual pulse lags it by the
  pulse transit time; this tests phase locking within ±2 s, not latency.

---

## 9. Caveats

1. **Only 4 subjects.** The strongest external-validity limit in this document.
2. **One band.** 0.9–3.0 Hz only.
3. **Global AC/DC normalisation.** Published CHROM/POS normalise in sliding windows;
   the global form is more susceptible to slow illumination drift.
4. **Estimator ceiling is low.** Even a clean synthetic pulse reaches only skill
   +0.278. The rate result is therefore bounded by the estimator as much as by the video.
5. **Reference HR is quantised** (~90 distinct values per session, ~1.5 BPM steps).
6. **Compression.** 8-bit JPEG at 25 fps is aggressive for colour-change measurements;
   this is a plausible reason the *rate* is unstable even where the *pulse* is detectable.
7. **Remaining control gap.** The face-free background patches are ~3.5× darker than
   the face, so their 8–10 % firing rate is not fully explained. The mismatched-subject
   control sidesteps this (it uses the face signal), but a **brightness-matched
   non-skin** patch would close it.

---

## 10. Conclusions

1. **The central positive result:** facial RGB video carries a real cardiac signal in
   these recordings, and it is blood-derived rather than motion- or illumination-derived.
   This is the positive control the project previously lacked — a reviewer's objection
   that "the pipeline may simply not work" is now answered with a measurement.
2. **Pipeline design matters enormously.** The same pixels give 68 % or 90 % detection
   depending on whether CHROM/POS and skin masking are used. Any negative result in
   this area must state which extraction was used.
3. **Detection and measurement are different problems.** The facial signal is reliably
   *present* and unreliably *quantifiable*. Reporting a single "it works / doesn't work"
   verdict for facial cardiac sensing conflates the two.
4. **Controls are load-bearing.** Bugs 1–3 (§7) each could have reversed a conclusion.
   A single control was not sufficient; the mismatched-subject design caught an error
   that a background patch alone would have missed.

---

## 11. Reproduction

All analysis code, logs and signal caches are committed under
**`code/analysis/rgb_bp/`** (7.6 MB).

| artefact | file | log |
|---|---|---|
| skin-masked CHROM/POS extraction + gate + HR | `rppg_skin_chrom.py` | `rppg_skin.log` |
| NaN audit, corrected gate, mismatched-subject control | `verify_skin_gate.py` | — |
| blood-vs-motion regression | `blood_or_motion.py` | — |
| HR estimator validation (synthetic ceiling) | `hr_estimator_validation.py` | — |
| comparable HR table (single consistent truth) | `hr_comparable.py` | — |
| earlier gate, n=40 (superseded) | `rppg_gate40_v3.py` | `rppg40v3.log` |
| respiration face-ROI mechanism decomposition | `rgb_face_mechanism.py` | `face_mech.log` |
| landmark reader + 2D ROI presets | `code/data/rgb_features.py` | — |

**Signal caches** (per-frame ROI statistics for all 40 sessions):

* `rppg_skin_v1.npz` — skin-masked pipeline; `face_plain`, `face_skin`, `bg`, `full`
  per-channel means plus the per-frame skin fraction. Feeds §6.1–6.3.
* `rgb_roi_means_v3.npz` — the earlier five-background-patch extraction (bug 1
  present, retained for the bug record).

Each script resolves its cache from a `CACHE` constant near the top, pointing at
`/tmp`. Re-running therefore needs either the cache copied back to `/tmp`, or
`CACHE` edited to the local path. With the cache present, every analysis in this
document re-runs in **seconds**; only a **new** ROI or per-pixel quantity requires
the ~4-minute parallel JPEG decode (12 processes).

Environment: `.venv` at the repo root; `scipy`, `h5py`, `opencv`, `numpy 2`, `pandas`.


---

## 12. Open next steps

1. **Better rate estimator** — sliding-window POS with overlap-add and harmonic
   tracking. The synthetic ceiling is only +0.278, so the estimator, not the video,
   currently caps the measurement result. This is the single highest-value next step.
2. **Brightness-matched non-skin control** to close caveat 7.
3. **Per-subject skin calibration**, since a fixed YCrCb box cannot suit all skin tones
   and lighting.
4. **Skin masking is the only factor that helps beyond CHROM/POS** — quantify whether
   the effect is brightness normalisation or genuine region selection.

---

## 13. Relationship to the wider project

The respiratory counterpart of this study is **negative from every visual input**
(thermal ROI 0/8 sessions; RGB face 0/11 after controls; the background fires *more*
often than the face). Reading the two together:

* the **sensor and pipeline are sound** — demonstrated here by a validated cardiac
  positive, and for the thermal stream by a motion control passing 8/8 at p = 0.000;
* the **respiratory absence is a property of the face-centric recording geometry**,
  not of the instrumentation;
* and low-frequency illumination artifact is the dominant contaminant in the
  respiratory band, whereas the cardiac band is comparatively clean — the background
  control fires 6/11 in 0.1–0.5 Hz but 0/11 in 0.9–3.0 Hz.
