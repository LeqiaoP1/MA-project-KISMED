# RESP reconstruction from the three-stream checkpoint — held-out subjects (2026-11-24)

- **Checkpoint**: `checkpoint-0149.pth`, epoch 149, fold 0
  (`/work/scratch/ne95ocyg/pretrain/adr005_3stream_fold0`)
- **Split**: `--split test` of the fold-0 record (`dc80d46c…`) — subject-disjoint
  from training and validation
- **Population**: **1,594 clips from 201 sessions / 27 subjects**. The fold has 28
  held-out subjects; `F027` contributes no usable clip at the corpus gates, so 27
  are scored. Every clip is an 8 s window with a 4 s hop.
- **Probe**: `runners/run_eval_recon.py`, plan pinned (not the training draw),
  `--resp_probe hidden` so the scored positions can only be reconstructed from
  the visual streams, `--repeats 5`, `--seed 0`, `--batch_size 16`, `--preload`.
- **Target / metric**: the dataset's per-clip respiration z-score; Pearson and
  MSE are computed on the **masked positions only** (`r_masked`). Each clip is
  averaged over its 5 mask draws before aggregation.
- **Subject aggregation**: median over the subject's clips, then a bootstrap
  95 % CI over subjects (2,000 resamples). The headline protocol's extra
  per-session step is not applied (see §7).

## 1. What was run

All arms keep the ADR-005 constant budget (350 visible tokens) and the same
checkpoint. The two virtual-sensing arms differ **only in how the 350 tokens are
split between the two visual streams**, so their per-clip comparison is a
within-model allocation contrast.

| Arm | Requested | Realised kept tokens | Resolved mode |
|---|---|---|---|
| **A — TIR-max** | `--kept_tir 1.0 --kept_rgb2d 0.0` | `tir 225` (9 tubes), `rgb2d 125` (5), `resp 0` | 1 |
| **B — RGB2D-max** | `--kept_tir 0.0 --kept_rgb2d 1.0` | `tir 125` (5), `rgb2d 225` (9), `resp 0` | 1 |
| **Reference — respiration visible** | `--kept_resp 0.5 --resp_probe random` | `tir 150`, `rgb2d 75`, `resp 125` | 2 |

A literal zero-token visual is not expressible: the policy's 2-tube floor is
mandatory. 225 tokens (9 of 36 tubes) is therefore the maximum any visual stream
can be given, and the 9 + 5 / 5 + 9 split is the widest contrast available.

**Neither arm is “TIR only” or “RGB2D only”; both streams are present in both
arms.** With respiration hidden, the 350 tokens must be 14 visual tubes, and one
stream can supply at most 9 tubes (the 25 % cap = 225 tokens). The other stream
must therefore supply at least 5 tubes (125 tokens):

$$K_{\text{tir}} + K_{\text{rgb2d}} = 350,\qquad K_v \le 225
\;\Longrightarrow\; \min(K_{\text{tir}}, K_{\text{rgb2d}}) \ge 125.$$

A “TIR-only, sensor-free” arm would need 14 TIR tubes (39 % of the stream), which
violates the 25 % cap, and would also leave the 2-tube RGB2D floor unsatisfied.
It cannot be expressed by this policy at this budget. The two extremes above are
therefore the strongest allocation contrast the architecture admits:

- **A** answers “what is the best a TIR-heavy allocation can do?”
- **B** answers “what happens if the second visual gets the budget instead?”

The `--kept_*` fractions are requests clamped by floor and cap; the residual is
spent on the visuals, which is why A still receives 125 RGB2D tokens even though
`--kept_rgb2d 0.0` was requested.

The reference arm was run on three subjects (F001, F003, F005; 210 clips) as a
positive control; A and B were run on all 27.

## 2. Q1a — with the sensor visible, reconstruction is essentially exact

| Metric (masked positions) | Reference (210 clips, 3 subjects) |
|---|---:|
| `r_masked` mean / median / sd | **+0.9959 / +0.9968 / 0.0041** |
| `mse` (z²) | **0.0090** |
| per-clip `r > 0.9` | **100 %** |
| per-subject median `r` | 0.9964, 0.9964, 0.9976 |

When 125 of the 250 respiration tokens are visible (`scattered`, the training
placement), the model interpolates the masked 4 s gaps almost perfectly. This
establishes two things: the decoder, the target normalisation and the scoring are
sound, and the probe is sensitive to respiration content.

## 3. Q1b — with the sensor absent, the signal is real but weak

Fully hidden respiration (Mode 1), all 27 subjects / 1,594 clips:

| Metric | A — TIR-max | B — RGB2D-max |
|---|---:|---:|
| `r_masked` mean | **+0.1091** | **+0.0800** |
| `r_masked` median | +0.1126 | +0.0832 |
| `r_masked` sd over clips | 0.2792 | 0.2692 |
| subject median `r` [95 % CI] | **+0.1129** [+0.0775, +0.1363] | **+0.0919** [+0.0705, +0.1085] |
| `mse` (z²) mean | 1.0408 | 1.0590 |
| `mae` (z) mean | 0.8458 | 0.8546 |
| clips with `r > 0.3` | **25.8 %** | 22.1 % |
| clips with `r < −0.3` | 8.1 % | 8.8 % |

**Both arms are above chance**: the subject-level 95 % CI excludes zero in both
(by a wide margin for A). The effect is also **asymmetric in the tails** — two to
three times more clips exceed $+0.3$ than fall below $-0.3$ — which is the
signature of genuine tracking in a subset of clips, not of noise.

But it is **not a reliable per-clip reconstruction**:

- the mean correlation is ≈ 0.1, i.e. about 1 % of variance explained;
- `mse ≈ 1.04` is the variance of the target, so the predicted **amplitude is
  heavily shrunk**: the model recovers some phase/shape, not the waveform level;
- roughly one clip in four reaches `r > 0.3`, and one in twelve is anti-correlated
  at that strength;
- the effect is heterogeneous across subjects (in the paired A→B contrast, 19 of
  27 subject medians move down, 8 move up).

So the honest answer to “is it obviously and reliably reconstructed?” is:
**inpainting yes, virtual sensing no — it is a real but small and clip-dependent
cross-modal signal.** The single-clip panel (`multimae_style_kept_density.png`)
shows the mechanism clearly; §3 is what happens over the population.

## 4. Q2 — does RGB2D help? No: reallocating budget to it hurts

A and B score exactly the same 1,594 clips and use the same five repeat seeds.
The plans differ, so the realised placements differ, but each clip is averaged
over the same five draws before the paired contrast:

| Quantity | Value |
|---|---|
| `r_masked`, A (TIR-max) | +0.10913 |
| `r_masked`, B (RGB2D-max) | +0.07995 |
| Paired Δ (B − A) | **−0.0292 ± 0.0044 (s.e.)** |
| $t$ | **−6.62** (n = 1,594) |
| Clips B worse / better | **876 / 718** |
| Subjects with negative median Δ | **19 / 27** |
| `mse`, A → B | 1.0408 → 1.0590 (worse) |

Moving 100 of the 350 visual tokens from TIR to RGB2D **reduces** the
virtual-sensing correlation by 27 % relative (0.109 → 0.080) and worsens the
masked MSE. There is no allocation at which RGB2D helps: the arm that gives it
the *most* budget is the worse one, consistently across clips and subjects.

That answers the budget question. It does not yet show whether RGB2D's **content**
is used at all — an allocation test can only show that more RGB2D is not better.
§5 tests content directly by corrupting RGB2D while holding the plan fixed.

## 5. Q2b — is RGB2D's content used at all? (input ablation)

A literal presence/absence arm cannot be expressed: the visual floor keeps at
least two RGB2D tubes on every step, and a matched `tir+resp` training arm does
not exist. The available test is an **inference-time information ablation** on
the same checkpoint: the plan, the token counts and the mask-placement seeds are
identical to arm B, and only the RGB2D input tensor is altered.

- `roll` — circularly shift the 50-frame RGB2D time axis by 16 frames (2.56 s).
  Every frame and the frame statistics are preserved; the temporal alignment with
  TIR and with respiration is destroyed.
- `zero` — blank the stream. This is the stronger “no content at all” condition,
  but it is out of distribution, so a drop would be less interpretable than the
  roll.
- The same roll applied to **TIR** is the positive control: the probe must react
  to TIR's content, otherwise a null on RGB2D is uninformative.

All four runs use arm B's plan (`tir 125`, `rgb2d 225`, `resp 0`), the same five
repeat seeds and the same placement draws, so intact vs ablated is paired at the
mask level as well as at the clip level.

| Run (all: `tir 125 / rgb2d 225 / resp 0`) | `r_masked` mean | median | subject median [95 % CI] | MSE (z²) | `r`>0.3 / <−0.3 |
|---|---:|---:|---|---:|---:|
| **B intact** | **+0.07995** | +0.08324 | +0.0919 [+0.0705, +0.1085] | 1.0590 | 22.1 / 8.8 % |
| **B, RGB2D rolled** | +0.07872 | +0.07950 | +0.0720 [+0.0492, +0.1097] | 1.0578 | 22.0 / 8.6 % |
| **B, RGB2D zeroed** | +0.07710 | +0.07264 | +0.0586 [+0.0432, +0.1033] | 1.0781 | 21.3 / 7.8 % |
| **B, TIR rolled** (control) | **+0.02605** | +0.02686 | +0.0233 [−0.0181, +0.0483] | 1.1009 | 15.7 / 11.7 % |

Paired over the same 1,594 clips:

| Contrast | Δ (ablated − intact) | s.e. | $t$ | clips worse / better |
|---|---:|---:|---:|---:|
| RGB2D rolled | **−0.0012** | 0.0026 | **−0.47** | 836 / 758 |
| RGB2D zeroed | **−0.0028** | 0.0064 | **−0.45** | 816 / 778 |
| TIR rolled | **−0.0539** | 0.0054 | **−9.96** | 948 / 646 |

**Reading.** Rolling RGB2D — which carries 9 of the 14 visual tubes, i.e. 64 % of
the visual budget — leaves the masked-position correlation unchanged to three
decimals; blanking it entirely changes the mean by −0.003 (n.s.). Rolling TIR,
which carries only 5 tubes, removes two-thirds of it, and the remaining
subject-level correlation is no longer distinguishable from zero (CI spans 0). The
virtual-sensing signal the decoder uses is therefore carried by **TIR's five
tubes**, not by RGB2D's nine; this is the mechanism behind the allocation result
of §4.

The blanking run does show a **larger per-clip dispersion** of the difference
(sd 0.256 vs 0.105 for the roll) with no mean shift: RGB2D perturbs individual
clips in both directions without systematically helping or hurting them. That is
what “not used for this target” looks like, as opposed to “used and necessary”.

## 6. Interpretation

1. **The 3-stream model has a genuine sensor-free respiration pathway, but it is
   weak.** The subject-level CI excludes zero and the per-clip tails are
   asymmetric, so this is not noise. It is, however, an order of magnitude below
   the inpainting regime (≈ 0.11 vs ≈ 0.996) and the predicted amplitude is
   collapsed (MSE ≈ 1 z²). It is a *lead* for the representation, not a usable
   waveform.
2. **The pathway lives in TIR, not in RGB2D.** Giving RGB2D nine of the fourteen
   visual tubes makes the reconstruction significantly worse (Δ = −0.029,
   t = −6.6), and the content ablation of §5 shows that neither rolling nor
   blanking RGB2D changes the masked correlation materially, while rolling TIR
   removes two-thirds of it. Whatever the encoder uses, it is better served by
   thermal tokens. This is not evidence that RGB2D is useless in principle — the
   RGB2D tokenizer is random-initialised (§7) and a training-time benefit cannot
   be excluded without a 2-stream arm.
3. **The curriculum is doing something the mask statistics alone cannot show.**
   Mode 1 is 49 % of the run and the probe is exactly Mode 1, so the model was
   trained on the deployment condition for half of pre-training. Whether that is
   what produced the positive r cannot be separated here: there is no
   `E_warm = 0` or `p_target = 0.10` control run.
4. **Do not over-read the absolute number.** `r_masked ≈ 0.11` is a zero-shot,
   per-clip z-scored, masked-position correlation on 8 s windows. It is not the
   band-limited subject-level metric of ADR-005 §3.2, and no Stage-3 fine-tuning
   has been applied.

## 7. What this does not establish

- **Zero-shot only.** These are frozen Stage-2 decoders, not Stage-3 fine-tuned
  waveforms. No band-limited Pearson (0.1–0.6 Hz), no rate error, no
  subject-level headline of ADR-005 §3.2 has been computed.
- **Fold 0 only**, one checkpoint, one seed, one checkpoint-selection rule.
- The target is a **per-clip z-score**; `r_masked` is scale-invariant, but the
  MSE “≈ 1” statement is in those per-clip z units.
- **27 of 28** held-out subjects; `F027` has no usable clip under the corpus gates.
- The RGB2D adapter is **randomly initialised** (ADR-005 I2): the shared
  transformer body inherits VideoMAE, but the RGB2D patch embed does not. A
  conclusion about RGB2D is therefore conditional on that tokenizer.
- The **allocation** contrast (A vs B) and the **content** ablation (§5) answer
  different questions; neither is a substitute for a matched 2-stream training
  run. An off-policy all-TIR / no-RGB2D diagnostic is in
  `diag_full_tir_no_rgb2d.md`: it collapses to `r ≈ 0` at both 900 and 350
  tokens, which shows the pathway is allocation-specific, not that a 2-stream
  model would fail.
- The content ablation uses **one** temporal shift (16 of 50 frames) and one
  blanking variant. A static RGB2D cue would survive a roll, but respiration is
  dynamic, and the TIR control shows the probe is sensitive to content; a
  different shift or a spatial corruption was not tested.
- Subject CIs use the median over clips, not the protocol's per-session step; the
  clip counts per subject (1–115) are reported in the artifact.
