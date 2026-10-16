# Why does the AU-occurrence probe exist? — motivation review

> **Purpose of this file.** A critical reading of the *motivation* for the
> AU-occurrence probe, not a change to it. It answers "for what reason do we need
> this at all?" and records where the stated justification holds and where it
> does not. It does **not** supersede the protocol of record
> (`AU_OccurenceProbe.md`) — it comments on why that protocol is worth running.

Compiled 2026-10-16 on branch `au_probing`. Companion to:
`AU_OccurenceProbe.md` (protocol of record),
`AU_OccurenceProbe_implementation.md` (what was built),
`au_tir_roi_hpc.md` (the run and its negative result).

---

## 1. The stated reason

`plan/ImplementationPlan.md` §5, verbatim:

> *"A **regression-only read-out is a weak witness of representation quality**,
> so this probe tests whether the frozen Stage-2 features **linearly separate
> FACS facial-action semantics**."*

`README.md`, the AU-probe section:

> *"Its claim: Stage-2 unsupervised masked pre-training **should make FACS
> facial-action semantics linearly decodable** from the (frozen) shared
> encoder."*

Unpacked, the argument is a four-step chain:

1. **Stage 2 is a masked-reconstruction (MAE) objective.** The classic failure
   mode of any MAE is reaching a low reconstruction loss through a **shortcut** —
   copying neighbouring pixels, exploiting low-level statistics — while learning
   no *semantic* structure.
2. **The only other read-out is a regression**, and a regression wearing a
   physiological label is a **weak witness**: it cannot distinguish "learned
   something real" from "found a cue that happens to correlate with the target".
   This investigation subsequently *did* find such a cue — the off-face
   chest/hair channel identified by `frame_argmax`, plus a model that only ties a
   single-tone template (49.1 % win rate).
3. **So an orthogonal read-out is needed**: ask a **classification** question
   instead of a regression one — categorical, semantic, with different failure
   modes from the thing being audited.
4. **AU is the chosen classification axis because it is a standard.** BP4D + FACS
   has a codified protocol, known base rates and a published baseline structure,
   so numbers are comparable across encoders. The probe is imported wholesale
   from the SSL-probing literature (MAE/VideoMAE linear probes).

So the probe is **not** intended to measure "how good is the encoder at its job".
It is intended to audit **whether the latent space contains semantics, or whether
the low loss is a shortcut.**

## 2. Where that argument is sound

The motivating distrust was justified and this investigation vindicated it. The
Stage-3 respiration result is confounded by a global/motion channel, and the model
merely ties a single-tone template — i.e. **the loss was being reduced by
something other than respiration.** In that context a *falsification test for
"the encoder learned faces"* is a legitimate instrument, and a negative result is
informative: it says the encoder does **not** hold lower-face action semantics,
which **corroborates** the low-level / global-motion story rather than
contradicting it. That is the one framing under which the AU probe earns its keep.

## 3. Where the argument breaks down

### 3.1 The proxy axis is not the task axis
AU decodability is **neither necessary nor sufficient** for good respiration
regression. An encoder could predict respiration from subtle thermal/colour
change with *zero* AU semantics; and it could be AU-decodable and still useless
for respiration. A *pass* would not validate the Stage-3 number, and this *fail*
does not invalidate it. The probe measures a **different capability**.

### 3.2 The borrowed benchmark's validity assumption fails
In the SSL papers this probe comes from, the encoder is trained on data from the
**same distribution** as the probe task (RGB faces/video). Here the encoder saw a
**thermal**, 8-bit **palettised**, **~3 % of the frame** crop, trained purely to
regress respiration. Probing it with RGB-face AU semantics is an **out-of-domain**
test of a **narrow-purpose** encoder — so failure is the *a priori expected*
outcome, not evidence of a defect. The observed result (prior-only head,
`corr(presence rate, F1) = +0.984`) is exactly what would be predicted without
running anything.

### 3.3 The docs already demote it below decision-bearing
Same `ImplementationPlan.md` §5, a few lines further down:

> *"this probe is a **biased witness** … Judge Stage 1 on the Stage-3 ladder below
> and report the AU probe as a **secondary diagnostic**."*

And the protocol's own §1:

> *"report it **alongside** the Stage-3 respiration number, **never instead of
> it**."*

So the document that motivates the probe also states that it does not drive the
main decision. Additionally §5's acceptance ladder is internally **stale**: it
expects `C2 ≥ C1 ≫ C0`, but **C0 was removed** and `--finetune` is now a hard
error (the protocol's §0 conflict table flags §5 as stale on exactly
C0/ImageNet/`videomae:base`).

## 4. The test to apply: what decision changes?

| AU outcome | Decision |
|---|---|
| F1 > trivial on a **majority** of Tier-A AUs | The encoder is a general face representation → it can be reused for other facial tasks; don't retrain for them. |
| **Not** (the observed case: 2/7) | Don't invest in semantic reuse of this encoder; and the latent space is low-level, consistent with the global-motion confound. |

Both branches yield a real, if narrow, decision — and the current answer is the
second. Note the crucial property: **the decision is independent of the respiration
result.** Nothing about Stage-2 or Stage-3 changes either way.

## 5. Verdict

**Why it exists:** to guard against the reconstruction shortcut, on the theory
that a regression-only evidence base is too weak — an *orthogonal read-out* rather
than another view of the same number.

**Whether that reason is sufficient:** only as an **audit of what the latent space
contains**, and only because this project has a live confound question about
whether the respiration signal is real. As a **representation-quality score** for
the pipeline: no — it is off-axis (§3.1), out-of-domain (§3.2), and the docs
themselves call it a biased secondary witness (§3.3).

**Recommendation.**

1. **Reframe in the write-up** from *"representation quality"* to
   *"semantic-content audit / falsification test"*, with the decision attached
   explicitly: *the encoder is not a general face model → do not reuse it for
   other facial tasks; its latent space is low-level, consistent with the
   global-motion confound.*
2. **If no decision can be named, do not run it.** It costs GPU time and, worse,
   it invites the over-reading the docs themselves warn against — quoting a
   macro-F1 of 33.04 as though it said something about the respiration number.
3. **For the shortcut question specifically**, prefer the diagnostics that bear
   *directly* on the claimed output: the **global-motion control** (off-face
   `frame_argmax`), the **region gate**, and the **single-tone template baseline**.
   AU probing bears on a different output and cannot substitute for them.
4. **Keep it strictly secondary**, which is what the protocol already says.

## 6. Provenance

| Claim | Source |
|---|---|
| "a regression-only read-out is a weak witness" | `plan/ImplementationPlan.md` §5 |
| "should make FACS … linearly decodable" | `README.md`, AU-probe section |
| "biased witness … secondary diagnostic" | `plan/ImplementationPlan.md` §5 |
| "alongside … never instead of it" | `AU_OccurenceProbe.md` §1 |
| stale `C0` ladder; §5 flagged stale | `ImplementationPlan.md` §5 + `AU_OccurenceProbe.md` §0 |
| off-face chest/hair channel; template tie 49.1 % | `analysis/tir_resp/frame_argmax.*`; `au_tir_roi_hpc.md` §7 |
| 2/7 majority-rule failure; r = +0.984 | `au_tir_roi_hpc.md` §4–§6 |

_This is a review/commentary document. The protocol of record remains
`AU_OccurenceProbe.md`._
