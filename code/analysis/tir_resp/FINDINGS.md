# TIR-ROI → vital-signal: findings (2026-10-09)

Concise summary of the re-opened TIR-ROI investigation. Full numbers live in the
tool JSONs and the HPC logs; this is the verdict and the reasoning.

## Verdict

**The thermal FACE ROI does not reliably carry respiration — nor BP.** The
apparent whole-frame "detections" are driven by OFF-TARGET pixels (chest / hair /
a fixed background band = the recording rig), not by the face. The line stays
closed for the face, now with a mechanism.

## What was measured

The gate is the same statistic throughout: *max over (pixel × lags) of |r|
against a reference, judged by a phase-scrambled null*, evaluated in the model's
own input space (the ROI cache) and on the whole frame. Controls: `motion_x`
(positive control), off-face background boxes, and local-motion partialling.

## Findings

1. **Face ROI ↔ respiration:** detected in most sessions and it survives local-
   motion partialling — but the *strongest* belt pixel in the frame is
   **off-face**, and on the face the strongest reference is **motion, not the
   belt**. So the face result is real-but-modest and confounded, not facial
   respiration.
2. **The task label is not the lever:** detection is near-uniform across tasks
   and non-monotonic across task groups.
3. **Whole-frame winners are rig/body artifacts** (chest in one session, hair in
   another) — out of BP4D+ scope. A CONFOUND to exclude, not a region to exploit.
4. **BP (2nd vital signal) is weaker, not stronger:** at corpus scale its
   detection rate collapses well below the belt's, and its pulse-band hits are
   concentrated on one task — an artifact signature, not a physiological one.
5. **The data are representation-limited:** 8-bit palettised false-colour, the
   face near the saturated end of the palette, and a frontal view — so the
   exhaled-air plume has almost no thermal contrast against the (equally warm)
   skin.

## Why (mechanism)

- The respiration band **overlaps the activity / auto-gain band**, and the face's
  top signal is head motion — so "belt detected on the face" is expected from
  motion/global coupling alone.
- The breath plume is only visible **against a cold background**; frontally it
  sits in front of skin at the same temperature, is low-emissivity, and is
  quantised away by the palette.
- The off-face winners are **chest-wall motion, head/hair motion, and a fixed hot
  vertical band** — properties of the subject's body and the rig.

## Recommendation

- **Close the line as a controlled negative** and document the mechanism (this
  note). Let the Stage-3 fine-tune finish as a baseline/ablation; do not invest
  further compute.
- **Do NOT** pursue a chest/body ROI (out of scope), a whole-frame arm (its
  winner is the rig), or a BP pivot (weaker and task-concentrated).
- **Keep `IRFeatures`** — it remains the face anchor, the motion reference, and
  the sentinel/missing-frame mask.
- **Optional, bounded last attempt** (only if a positive is required): the
  ON-FACE nostril/lip tiles, cleared by all three of a rank/Spearman gate
  (palette-monotonicity-immune), local-motion partialling, and the
  gradient/sign-flip test (thermal vs mechanical). Low expected yield.

## Tools (`code/analysis/tir_resp/`)

- `gate_model_input.py` — the gate in the model's input space.
  `--refs belt,motion_x,bp`, `--bp_band`, `--partial-motion`.
- `frame_argmax.py` — WHERE the whole-frame winner is (`--selftest`).
- `roi_sweep.py` — disjoint-tile region sweep with detrending.
- `native_resolution_gate.py` — the shared statistic / null / signal loaders.
