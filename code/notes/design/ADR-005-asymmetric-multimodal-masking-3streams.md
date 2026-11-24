# ADR-005: Three-Stream Constant-Budget Asymmetric Multi-Modal Masking with a Virtual-Sensing Curriculum (`tir`, `rgb2d`, `resp`)

- **Status**: **Superseded by `ADR-006-three-regime-virtual-sensing-masking.md`** (2026-11-24). ADR-006 is the sole design of record for this policy and restates everything it needs (§2.2–§2.4, §2.7, §2.9–§2.10 of that document). This file is retained as the record of the two-mode decision and of the measurements that motivated its successor; it is **not** an authority, and nothing should be implemented from it.
- **Date**: 2026-11-23
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/design/Evaluation_RespWaveform_protocol.md` — normative for the definition and aggregation of the headline metric used in §3. `code/notes/plan/features/ADR-005-asymmetric-multimodal-masking-3streams-plan.md` records the build order and gates; it is not an authority. Everything else in this ADR is self-contained.
- **Supersedes**: `code/notes/design/ADR-002-asymmetric-multimodal-masking.md` — the four-stream arm, now **obsolete**. This ADR is the sole design of record for the masking policy: the counting, alignment and placement arithmetic it inherits is stated normatively here (§§2.3–2.5), together with the declared stream set, the tokenizer geometry, the initialisation contract and the virtual-sensing curriculum. No clause of this document depends on the superseded record.

**Normative status of this document.** *Superseded.* While it was live, this ADR was the authority for the masking policy it defines, and it was written to stand on its own: the policy, its units, its arithmetic, its feasibility conditions, its initialisation requirements, its observability requirements, its parameter surface and its acceptance criteria are all specified here. It no longer governs: **ADR-006 supersedes it**, and where the two differ ADR-006 wins. Appendix A is the only place that maps the specification onto the present toolchain, it is marked informative, and it may drift without affecting any decision.

## 1. Context & Problem Statement

### 1.1 Objective and scope

The pipeline this ADR governs has one objective: **reconstruct the respiration waveform of an unseen subject from the thermal face ROI with the contact sensor absent**. The headline metric, its band, its per-session → per-subject aggregation and its uncertainty estimate are fixed by `code/notes/design/Evaluation_RespWaveform_protocol.md` (§3, §4, §6); this ADR does not redefine them.

**In scope.** Which tokens of which streams reach the shared encoder on a pre-training step, how many, and with what probability; the units the policy is defined over; the tokenizer geometry those units are derived from; the initialisation the geometry must admit; the conditioning a fully masked stream must receive; the observability the policy must provide; and the pre-registered acceptance criteria.

**Out of scope.** Encoder and decoder internals beyond the conditioning contract of §2.3(7); data selection, pre-processing, ROI construction and caching; the Stage-3 fine-tuning recipe; the measurement implementation. A decision about any of these is taken by its own document.

### 1.2 Design pressures

Pre-training a multimodal masked autoencoder for this target failed to learn it. The five pressures below are what ADR-005 answers. They are stated as *behaviour of the previous design*, deliberately without reference to code, because the implementation may be replaced and the pressures would not change.

**Predecessor arm and its measured outcome.** The predecessor policy carried four streams — two visual (`tir`, `rgb2d`) and two 1-D physiological (`bp`, `resp`) — under the same constant 350-token budget, and was pre-trained for 150 epochs on one subject-disjoint fold. The two physical mechanisms it was asked to track are different in kind: subcutaneous blood-volume pulsation in the visible band (remote photoplethysmography) and boundary-layer airflow temperature at the nostrils in the thermal band. Its visual reconstruction improved monotonically as more tubes were left visible, while respiration reconstruction tracked only the respiration tokens that were themselves visible: with the sensor fully masked the waveform collapsed to the clip mean, scoring a masked-position Pearson correlation of $+0.0062$ (budget on the thermal stream) and $+0.0140$ (budget on the texture stream) with a masked-position MSE of about $1$ in z units. Those numbers and that monotone visual curve are what the five pressures below explain.

1. **Collapse of the target under complete sensor loss.** With zero visible respiration tokens the reconstruction fell to the clip mean, and the per-clip correlation tails were symmetric — the signature of noise, not of partial tracking. A policy that exercises the deployed failure mode on half of all steps therefore produced no transferable representation.
2. **No conditioning path for a fully masked stream.** A stream with zero visible tokens cannot be reconstructed from anything if its decoder has no route to the other modalities' latents. The mode then exists in name only: its reconstruction contributes no gradient to the shared encoder, so nothing ever asks the encoder to carry the target in the visual latents.
3. **Cold start from random initialisation.** A backbone width with no published video-MAE initialisation — the predecessor arm used a ViT-Small width, for which no checkpoint of that geometry exists — forces the tokenizer and the encoder to learn micro-scale physiological variation from scratch, on a cohort far too small to support it.
4. **A physiological channel the tokenizer cannot represent.** Continuous arterial blood pressure is carried by sub-100 ms landmarks — the systolic upstroke and the dicrotic notch — that a 320 ms temporal token low-passes away. The stream becomes an under-determined reconstruction that contributes gradient variance without contributing to the target.
5. **A mode probability pinned at its target value from the first step.** Sampling the deployment regime at its final rate before any cross-modal structure exists removes the reference signal from exactly the steps that must establish one.

### 1.3 Requirements

The decision of §2 satisfies all six of the following. They are the criteria by which the design — not an implementation of it — is to be judged.

- **R1 — Constant visible budget.** The number of visible encoder tokens is identical on every step, so no input shape varies and no padding or placeholder token is ever needed.
- **R2 — Asymmetric, two-sided bounds in kept units.** Any stream can be masked *completely*; no visual stream may exceed a bounded kept fraction; both limits are expressed as kept counts, never as a mask ratio alone.
- **R3 — Gradient path for a fully masked stream.** A stream with zero visible tokens remains a reconstruction target and its loss must reach the shared encoder with non-zero gradient.
- **R4 — No random visual tokenizer.** The visual tokenizer geometry must admit a publicly available video-MAE kernel verbatim, with no inflation or surgery, so pre-training never starts from random visual features at the tokenizer.
- **R5 — Exactly one physiological target, representable by the tokenizer.** The objective carries a single 1-D physiological target whose bandwidth the token geometry can resolve.
- **R6 — Scheduled, not constant, probability of complete sensor loss.** The probability of the pure virtual-sensing mode is a function of training progress, and it starts below its target value.

## 2. Decision & Parameter Exposure

**Adopt the constant-budget asymmetric masking policy over a three-stream set `{tir, rgb2d, resp}`, on a `2×16×16` tubelet at a decimation of 4, with a two-phase virtual-sensing curriculum.**

### 2.1 Declared stream set: exactly two visual, exactly one physiological

The active set is $\mathcal{M} = \{\text{TIR}, \text{RGB2D}, \text{RESP}\}$: two visual streams and one 1-D physiological stream. The set is not a preference — it is forced by the budget arithmetic of R1 and R2. A single visual stream caps at 225 tokens and cannot supply the whole budget when the physiological stream is masked; three visual streams cannot survive physiological saturation, because $100$ visual tokens would remain while three 2-tube floors already need $150$. Exactly two visual streams and one or two physiological streams is the only closed form, and R5 selects the one-stream case.

*Why the second physiological stream leaves the objective (pressure 1.2(4)).* Two physiological streams would consume up to $2C_p = 250$ of the 350 tokens and would halve the visual evidence on every saturation step. The blood-pressure channel would also be trained against a token that cannot resolve its morphology, so its loss would act as gradient noise on a shared encoder. Removing it makes the objective single-target, which is what makes a negative result interpretable: the ADR records this as a design assumption, not as a measured result.

*Why exactly these two visual streams.* TIR is the deployed sensing modality — the target must ultimately be recoverable from the thermal ROI alone — so it is the primary visual stream. RGB2D is the auxiliary texture stream over the same facial region, cropped from the same frames, sharing `input_size` and `tubelet`, so the two streams occupy the same spatial token grid and their kept tubes are interchangeable in the budget.

### 2.2 Masking units and tokenizer geometry

| quantity | value | derivation |
| :--- | :--- | :--- |
| decimated frames | 50 | `round(clip_duration · fps / temporal_stride) = round(8 · 25 / 4)` |
| temporal tokens per tube `T` | 25 | `50 / tubelet_t = 50 / 2` |
| tube span | 320 ms | `tubelet_t · temporal_stride / fps = 2 · 4 / 25` |
| spatial tubes | 36 | `input_size / patch = 96 / 16`, squared (`6 × 6`) |
| tokens per visual stream | 900 | `36 · 25` |
| signal tokens per stream | 250 | `seq_len / sig_kernel = 8000 / 32` |
| signal tokens per visual token `R` | 10 | `250 / 25`; `10 · 32 ms = 320 ms = T` |

A **visual masking unit** is one spatial tube: one spatial patch together with *all* of its temporal positions, so keeping a unit adds a whole temporal column and a visual count is always a multiple of `T`. A **physiological masking unit** is one temporal token. This unit choice is what keeps masked video a dense rectangle (R1) and makes the visual budget a small integer number of tubes (§2.4).

Two derived quantities are fixed by the table and are load-bearing:

- `T = 25` — the visual temporal axis. The per-sample tube placement lives on the *spatial* index only, so the temporal axis is never ragged and every visual count is divisible by 25.
- `R = 10` — signal tokens per visual token, an exact integer, and `R · 32 ms = 320 ms = T`. One token index space can therefore hold both stream kinds without drift; a non-integer `R`, or a mismatch between `R · sig_kernel / fs` and `tubelet_t · temporal_stride / fps`, is an infeasible geometry and must be rejected when the model is built, not discovered during training.

**What this geometry does *not* change.** A 320 ms token at 25 fps is realisable as `(tubelet_t, temporal_stride) = (4, 2)` — 12.5 fps, decimation 2, the predecessor geometry — or as `(2, 4)` — 6.25 fps, decimation 4. Both give 25 temporal tokens over the 8 s window, so the *token rate* is identical. The second is required by R4/§2.6, and its consequence is that intra-tubelet sampling changes from four samples 80 ms apart to two samples 160 ms apart, i.e. a within-tubelet Nyquist of 3.125 Hz (187.5 breaths per minute). The respiratory band of interest runs from 0.15 Hz to 0.50 Hz (9 to 30 breaths per minute), so that cutoff is more than six times the highest frequency to be resolved, and un-aliased capture of the nostril boundary-layer signal is therefore guaranteed **at the sampling level**. **This ADR claims no improvement in temporal resolution: the token rate is unchanged.** Any observed difference between the predecessor arm and this one must be attributed to the stream set, the initialisation, the conditioning or the curriculum — never to temporal resolution. The one capability deliberately given up is the fine morphology of the blood-pressure waveform, which is the reason that channel leaves the objective (§2.1).

### 2.3 Normative contracts

These nine contracts are the policy. Each is checkable; §3 states how.

1. **Constant visible budget.** With $K_m$ the visible tokens of stream $m$:
   $$\sum_{m \in \mathcal{M}} K_m(t) = n_{\text{budget}} \quad \forall t, \qquad K_{\text{tir}} + K_{\text{rgb2d}} + K_{\text{resp}} = 350.$$
   The encoder therefore receives a fixed-length sequence on every step, containing no `[MASK]` token and no padding (R1).
2. **Asymmetric kept caps.** A visual stream keeps at most $c_v = 0.25$ of its units, i.e. $\lfloor 0.25 \cdot 36 \rfloor = 9$ tubes $=225$ tokens, so at least 75 % of it is a reconstruction target. The physiological stream keeps at most $c_p = 0.50$, i.e. 125 of 250 tokens (R2).
3. **Visual visibility floor.** Each visual stream keeps at least $m = 2$ tubes ($50$ tokens, 5.6 % of its 900). Without a floor the visual streams could round to zero on a high-physiology step and leave the encoder without visual evidence; $m = 2$ is the smallest allocation that still carries spatial context, and it is what makes the budget solvable at physiological saturation.
4. **Two modes, separated by the physiological count.** **Mode 1 (pure virtual sensing)** keeps $K_{\text{resp}} = 0$: the contact sensor is absent and the visual streams supply the whole budget. **Mode 2 (waveform inpainting)** keeps a drawn number of respiration tokens and inpaints the rest (R2, R6). A step belongs to Mode 1 **iff** $K_{\text{resp}} = 0$; if a Mode-2 draw happens to produce $K_{\text{resp}} = 0$ the step is a Mode-1 step and must be logged as such, because the two are indistinguishable to the model.
5. **Per-step counts, per-sample placement.** The counts $K_m$ are drawn once per step and shared by the whole mini-batch; the kept *indices* are drawn independently for every sample. Sharing placements would give a step the statistical weight of one mask; sharing counts keeps the batch a dense tensor (R1).
6. **1-D placement policy.** Respiration indices are drawn uniformly without replacement (`scattered`) or as contiguous spans (`span`). The choice is a property of the *run*, fixed before the first step, and recorded with it; training and evaluation must use the same choice unless a change is itself the object of study.
7. **Conditioned reconstruction for a fully masked stream.** A stream with $K_m = 0$ has no encoder tokens of its own, so its decoder sees only mask tokens and their positions. Its reconstruction must nevertheless condition on the encoder latents, by one of exactly two permitted mechanisms:
   (a) **prefixing** the full encoder latent sequence into that stream's decoder input before self-attention, or
   (b) **cross-attention** from that stream's decoder queries to the encoder latents.
   The mechanism is fixed for a run, recorded with it, and used identically in training and in evaluation. Contract check: for a stream with $K_m = 0$, $\partial \mathcal{L}_m / \partial z \neq 0$ (R3). A policy can satisfy contracts 1–6 in full and still fail pressure 1.2(2), which is why this contract is stated as a *checkable gradient condition* rather than as an architectural suggestion.
8. **Batch uniformity and determinism.** Every sample of a step carries the same $K_m$. The counts must be reproducible from `(seed, epoch, step)` alone and must be identical on every rank without a collective; placements must differ per sample and per rank. A run must therefore be replayable step-for-step from its seed and its schedule.
9. **Loss-key stability.** Every declared stream reports a loss term on every step and on every rank, as an explicit zero when it has no masked position. Metric synchronisation issues one collective per key, so a key set that depended on the sampled mode would desynchronise a distributed run — and Mode 1 depends on this: the physiological stream contributes no encoder token yet remains a reconstruction target.

### 2.4 Normative apportionment algorithm

The following is the complete algorithm. It contains no literal that is not one of the parameters of §2.9, and any implementation must reproduce it exactly.

```
# parameters: N = n_budget, T = tokens per visual unit, m = visual floor (tubes),
#             M = visual cap (tubes), C = physiological cap (tokens), p(e) = mode probability

function SPLIT_TUBES(S_total):            # clamped-uniform split over the two visual streams
    lo = max(m, S_total - M)
    hi = min(M, S_total - m)
    S_first = uniform_integer(lo, hi)     # inclusive; lo == hi is a single-valued draw
    return S_first, S_total - S_first

function SAMPLE_STEP(e):                  # e = epoch index, 0-based
    if uniform_real(0, 1) < p(e):
        K_resp = 0                        # Mode 1: pure virtual sensing
        S_total = N / T
    else:                                 # Mode 2: waveform inpainting
        K_tilde = uniform_integer(0, C)   # inclusive
        R_m = (N - K_tilde) mod T         # remainder that would break tube divisibility
        if K_tilde + R_m <= C:
            K_resp = K_tilde + R_m        # absorb it upward, inside the cap
        else:
            K_resp = K_tilde - (T - R_m)  # otherwise move down to the next multiple
        S_total = (N - K_resp) / T
    S_tir, S_rgb2d = SPLIT_TUBES(S_total)
    K_tir, K_rgb2d = T * S_tir, T * S_rgb2d
    assert K_tir + K_rgb2d + K_resp == N
    return mode, (K_tir, K_rgb2d, K_resp), (S_tir, S_rgb2d)
```

Consequences of the algorithm, stated so that they can be tested without reading it:

- **Mode 1** yields $S_{\text{total}} = 350/25 = 14$ tubes and, with `m = 2`, `M = 9`, a split $S_{\text{tir}} \in [\max(2, 14-9), \min(9, 14-2)] = [5, 9]$.
- **Mode 2** yields $K_{\text{resp}} \in \{0, 25, 50, 75, 100, 125\}$ and $S_{\text{total}} \in [9, 14]$, with the split interval widened to $[\max(2, S_{\text{total}}-9), \min(9, S_{\text{total}}-2)]$.
- **The remainder is charged to the only physiological stream**, so the predecessor's rule — charging it to the first physiological stream, which handed that stream up to 24 extra visible tokens on non-aligned steps — disappears entirely. The alignment does bias the *distribution*: enumerating the 126 draws of Mode 2 gives $E[K_{\text{resp}} \mid \text{Mode 2}] = 74.4$, above the raw mean 62.5, because the upward branch fires more often than the downward one.
- **`K_tilde = 0` is the one draw where Mode 2 produces a Mode-1 step** (1 of 126 draws). Contract 4 resolves it: log it as Mode 1.

### 2.5 Feasibility preconditions

The preconditions are properties of the *numbers*, not of any draw, so they must be validated when the configuration is resolved and an infeasible tuple must be rejected before the first step. With $T$ tokens per visual unit, floor $m$, cap $M = \lfloor c_v \cdot \text{units}_v \rfloor$ and physiological cap $C$:

1. $n_{\text{budget}} \bmod T = 0$ — Mode 1 must map the whole budget onto whole visual units.
2. $2m \le n_{\text{budget}} / T \le 2M$ — Mode 1 must be splittable inside both the floor and the cap.
3. $n_{\text{budget}} - C \ge 2mT$ — physiological saturation must still leave both visual floors.

With the parameters of this ADR ($T = 25$, $m = 2$, $M = 9$, $C = 125$, $n_{\text{budget}} = 350$): $350 \bmod 25 = 0$; $350/25 = 14 \in [4, 18]$; and $350 - 125 = 225 \ge 100$. All three hold, the third with a margin of 125 tokens — a design carrying two physiological streams satisfies it only with equality. That margin is the reason a three-stream arm can afford a 25 %-capped pair of visual streams on every step.

### 2.6 Initialisation and tokenizer-compatibility contract

R4 is a constraint on the geometry, not on the software, so it is stated as four requirements:

- **I1 — Verbatim kernel.** The visual tokenizer's temporal kernel must equal the pre-trained kernel, `(2, 16, 16)`. The projection is one 3-D convolution (`patch_embed.proj`) whose pre-trained weight has shape `[768, 3, 2, 16, 16]` at a 768-wide encoder, so matching the tubelet makes the inherited tensor fit bit-for-bit. A configuration whose tubelet differs is **infeasible** and must be rejected when the model is built. Neither temporal replication nor 2-D boxcar inflation is permitted: both destroy the temporal prior the initialisation is being borrowed for, and both make the claim "pre-trained" false.
- **I2 — One designated recipient.** A video-MAE checkpoint carries exactly one 3-channel patch embed, so exactly one visual adapter can receive it. The recipient must be **designated explicitly, not resolved by adapter ordering**, defaults to `tir` (the deployed modality of §1.1), and is written into the run record. The other visual adapter is then randomly initialised, and the run record must say so.
- **I3 — Encoder geometry.** The encoder depth, width and head count must match the checkpoint; the shipped decision uses the VideoMAE **ViT-Base** geometry — 768-d, 12 blocks, 12 heads. Another width is not forbidden, but it requires a checkpoint of that width, which R4 makes a precondition rather than a preference. Positional embeddings cannot be inherited (a pre-trained 2-D `pos_embed` over a 14×14 grid is meaningless over `G_t·6·6` tubelet tokens), so they must be re-initialised from a deterministic space–time prior rather than left random; the 1-D streams take the same temporal prior on their own token grid.
- **I4 — Honest labelling.** If no pre-trained checkpoint exists at the chosen width, the run is a random-initialisation run, must be labelled as one, and cannot support any claim of R4. A width is chosen *for* R4, not the other way round.

*Technical debt accepted by I2.* The auxiliary texture stream starts from a random Conv3d, so a null result for the third stream cannot distinguish "the stream carries nothing" from "its tokenizer never converged". Any claim about the second visual stream requires a run that routes the kernel to it, and that is a separate decision with its own document.

### 2.7 Virtual-sensing curriculum

Over the **0-based** epoch index $e$, with $E_{\text{warm}}$ the number of leading warm-up epochs:

$$p(e) = \begin{cases} p_{\text{warm}} & e < E_{\text{warm}} \quad (\text{Phase I — grounding}) \\[2pt] p_{\text{target}} & e \ge E_{\text{warm}} \quad (\text{Phase II — deployment regime}) \end{cases}$$

with $p_{\text{warm}} = 0.10$, $p_{\text{target}} = 0.50$, $E_{\text{warm}} = 50$.

| phase | epochs | $p(e)$ | Mode 1 | Mode 2 | primary learning objective |
| :--- | :--- | ---: | ---: | ---: | :--- |
| I — grounding | $e < 50$ | 0.10 | 10 % | 90 % | align visual latents with real respiration tokens |
| II — deployment | $e \ge 50$ | 0.50 | 50 % | 50 % | carry respiration dynamics in the visual latents alone |

**Why the warm-up value is 0.10 and not 0.50.** Mode 1 gives the respiration decoder no encoder-side respiration token, so from a cold start it has no physiological reference against which the visual latents can be aligned. At $p_{\text{warm}} = 0.10$ at least one respiration token survives on $0.9 \times (125/126) \approx 89.3\%$ of steps — the excluded mass is the single Mode-2 draw $\tilde K_{\text{resp}} = 0$, which contract 4 re-labels as Mode 1 — while the remaining ~10.7 % of steps still exercise the deployment regime, so it is never unseen.

Normative properties of the schedule:

- **It is a parameter of the policy, not of the training script.** $p(e)$ is evaluated once per epoch from the three parameters of §2.9 and applied before the epoch's first step. All three are validated on the same closed interval as the mode probability itself ($[0, 1]$), and a resolved tuple with $p_{\text{warm}} > p_{\text{target}}$ is a configuration error.
- **Validation masks follow the phase's own $p$**, drawn from a fixed seed sequence. The validation curve is therefore a *within-phase* observable: it must not be pooled or plotted as one continuous curve across $e = E_{\text{warm}}$, and no comparison may set an across-boundary value beside a within-phase one. Selection must not read a metric at all, precisely because the metric is discontinuous by construction.
- **A two-launch chain is not an acceptable substitute.** Splitting the run into a warm-up launch and a target launch cannot express "stop after $E_{\text{warm}}$" (a launch does not decide its own end), the learning-rate schedule is indexed by the epoch counter and would have to be declared identically in both launches to remain continuous, and the record must show one schedule rather than two.
- **Open question, not blocking.** The curriculum boundary and the learning-rate warm-up are independent schedules. Whether they should coincide is a tuning decision belonging to the experiment plan; this ADR fixes only that $p = p_{\text{warm}}$ strictly below $E_{\text{warm}}$, that both schedules are recorded, and that the boundary is reported with the result.

### 2.8 Normative observability

Whether the policy is running as decided must not require reading an implementation. Every run must therefore emit:

- **per step:** the epoch and step index, the active $p(e)$, the resolved mode *according to contract 4*, the kept token count **and** kept unit count of every declared stream, and the encoder sequence length;
- **per stream:** the masked reconstruction loss of every declared stream on every step, explicit zero included (contract 9), so the log's key set is a function of the configuration and never of the sampled mode;
- **per epoch:** the Mode-1 fraction and the mean kept counts, which must reproduce the values implied by $p(e)$;
- **per run:** the record of the placement policy (contract 6), the conditioning mechanism (contract 7), the designated kernel recipient (I2), and the full resolved parameter tuple of §2.9.

A run whose records cannot answer §3 Criterion 1 has not executed this decision, however well it trained.

### 2.9 Parameter surface

This is the complete parameter surface of the policy. An implementation must expose every row, and a resolved tuple must be re-validated against §2.5 before the first step.

| parameter | symbol | type | default | admissible values | meaning |
| :--- | :--- | :--- | ---: | :--- | :--- |
| `n_budget` | $N$ | int | 350 | any value for which §2.5 holds | constant visible encoder tokens per step |
| `tokens_per_unit` | $T$ | int | 25 | derived from the geometry (§2.2) | encoder tokens per visual spatial tube |
| `video_cap_ratio` | $c_v$ | float | 0.25 | $(0, 1]$ with $M = \lfloor c_v \cdot \text{units}_v \rfloor \ge m$ | kept-fraction ceiling per visual stream |
| `physio_cap_ratio` | $c_p$ | float | 0.50 | $(0, 1]$ with $C = \lfloor c_p \cdot \text{units}_{1d} \rfloor$ | kept-fraction ceiling for respiration |
| `video_min_tubes` | $m$ | int | 2 | $[1, M]$ | kept-tube floor per visual stream |
| `p_virtual_sensing` | $p_{\text{target}}$ | float | 0.50 | $[0, 1]$ | Mode-1 probability from $E_{\text{warm}}$ onwards |
| `virtual_sensing_warmup_p` | $p_{\text{warm}}$ | float | 0.10 | $[0, p_{\text{target}}]$ | Mode-1 probability strictly below $E_{\text{warm}}$ |
| `virtual_sensing_warmup_epochs` | $E_{\text{warm}}$ | int | 50 | $[0, \text{epochs}]$ | number of leading epochs at $p_{\text{warm}}$; `0` disables the curriculum |
| `physio_placement` | — | enum | `scattered` | `scattered`, `span` | 1-D placement policy (contract 6) |
| `physio_span_tokens` | — | int | 0 | required iff `span` | length of one contiguous span |
| `spatial_patches` | $\text{units}_v$ | int | 36 | must equal the tokenizer's tube count | units per visual stream the policy is defined over |
| `physio_tokens` | $\text{units}_{1d}$ | int | 250 | must equal the tokenizer's signal-token count | units for the physiological stream |

Geometry the policy is defined over, stated because §2.2 derives the unit counts from it and because R4 constrains it — none of these is tuned by this ADR: `clip_duration = 8.0 s`, `fps = 25`, `temporal_stride = 4`, `tubelet = (2, 16, 16)`, `input_size = 96`, patch `16`, `fs = 1000 Hz`, `sig_kernel = 32`.

Three notes on the surface. First, `virtual_sensing_warmup_p` and `virtual_sensing_warmup_epochs` are the **only** additions this policy makes to the parameter surface of §§2.3–2.4, and they exist solely because R6 cannot be expressed with a single probability. Second, the two `units` rows are assertions, not knobs: they must equal what the tokenizer actually produces, and a mismatch is an infeasible configuration rather than a parameter to adjust. Third, the visual cap $M$ and the physiological cap $C$ are **derived** from their ratios rather than exposed as independent knobs (an earlier design exposed them directly, as `video_max_tubes` and `resp_max_tokens`); the derivation is kept because a cap that can be set independently of its ratio invalidates the feasibility proof of §2.5, while any cap value that is actually required can still be produced exactly by the ratio that yields it.

$E_{\text{warm}}$ is expressed relative to the run's total epoch budget rather than as an absolute epoch number: the shipped arm trains for 150 epochs, so the split is 50 warm-up epochs followed by 100 at the target probability. This ADR fixes the split, not the total.

### 2.10 Rejected alternatives

- **Four streams, keeping the blood-pressure channel.** Rejected: two physiological streams consume up to 250 of the 350 tokens, halving the visual evidence at saturation; the pressure of 1.2(4) means that channel's loss is gradient noise on a shared encoder; and a two-target objective makes a negative result uninterpretable.
- **A single probability of 0.50 from the first step.** Rejected by R6 and pressure 1.2(5): it removes the physiological reference from exactly the steps that must establish the cross-modal alignment.
- **A `4×16×16` tubelet (12.5 fps, decimation 2).** Rejected by R4: no public video-MAE kernel matches it, and the temporal token count would be identical anyway, so nothing is gained.
- **Temporal replication or 2-D boxcar inflation of the pre-trained kernel.** Rejected: it destroys the temporal prior being borrowed and makes the "pre-trained" label false (I1).
- **A narrow backbone without a published initialisation, trained from scratch.** Rejected by R4 and pressure 1.2(3); such a run may be performed but must be labelled random-init (I4).
- **Symmetric proportional allocation.** Rejected: variable visible counts, no domain-asymmetric bounds, and no way to express complete sensor loss (R1, R2).
- **An uncapped physiological kept ratio.** Rejected: a fully visible waveform gives the encoder a self-contained route to the target and lets it ignore the visual streams entirely.
- **Prescribing cross-attention as the only conditioning mechanism.** Rejected as over-specification: both mechanisms of contract 7 satisfy R3, and which one was used changes the objective, so it must be *recorded* rather than *mandated*.
- **A two-launch curriculum.** Rejected in §2.7.
- **A per-step varying visible budget.** Rejected by R1: ragged batches lose the dense shape guarantee and make the workload non-deterministic.

## 3. Falsifiable Criteria

Every criterion below is evaluable from the run records required by §2.8; none of them requires reading an implementation.

### 3.1 Criterion 1 — policy conformance (behavioural; no learning required)

Established from the records of §2.8 before any scientific reading is attempted. Failure of any item means the decision was not executed, and the run may not be interpreted.

1. **Exact sum.** Over at least 10⁴ sampled steps (or the run's whole step count if smaller): $K_{\text{tir}} + K_{\text{rgb2d}} + K_{\text{resp}} = 350$ on 100 % of steps, so the encoder input is exactly `[B, 350, D]` on every step and its sequence length is 350 on 100 % of steps.
2. **Bounds and discreteness.** $K_{\text{tir}}, K_{\text{rgb2d}} \in \{50, 75, \dots, 225\}$ — multiples of $T$ inside $[mT, MT]$ — and $K_{\text{resp}} \in \{0, 25, \dots, 125\}$, with the corresponding tube counts inside $[2, 9]$.
3. **Mode consistency.** A step is Mode 1 *iff* $K_{\text{resp}} = 0$ (contract 4); no Mode-1 step carries a respiration token, and the share of Mode-2 draws landing on $K_{\text{resp}} = 0$ is the predicted $1/126$, i.e. about $0.8\%$.
4. **Schedule.** The Mode-1 fraction is $0.10 \pm 0.02$ over the first $E_{\text{warm}}$ epochs and $0.50 \pm 0.02$ thereafter, each reported with its binomial interval. This is the only direct test of R6.
5. **Batch uniformity and replay.** All samples of a step share $K_m$ while their placements differ; replaying the count sequence from $\texttt{(seed, epoch, step)}$ reproduces it exactly and does not depend on the world size.
6. **Gradient path.** On a step with $K_{\text{resp}} = 0$, $\partial \mathcal{L}_{\text{resp}} / \partial z \neq 0$ (contract 7). One forward/backward on a synthetic batch suffices; this is the operational form of R3 and the test that pressure 1.2(2) is cured.
7. **Feasibility rejection.** Every infeasible tuple of §2.5 is rejected when the configuration is resolved rather than at the first step, and so is a tubelet that differs from the pre-trained kernel (I1).

### 3.2 Criterion 2 — the scientific claim (headline gate)

- **Metric.** Band-limited Pearson (0.1–0.6 Hz), the headline of the evaluation protocol §4. Amplitude metrics are excluded by protocol §4.1 and must not appear in a comparison.
- **Unit and aggregation.** Per session → **median per subject** → median with a **bootstrap 95 % CI over subjects** (≥ 2000 resamples). Each subject is one independent observation; clips within a subject are not.
- **Split.** Subject-disjoint folds, **test role only**, pooled across folds. A validation figure is never the reported number.
- **Baselines, in ascending order of strength.** (i) the constant/mean line (Pearson 0 by construction); (ii) the single-tone template at the reference's own peak frequency — the best a rate-only model can do (protocol §5); (iii) the matched predecessor arm — same fold set, same fine-tuning recipe — which is the arm this ADR exists to improve.
- **Threshold, pre-registered here.** The pooled median band-limited Pearson must be **at least 0.02 above baseline (iii)** *and* the beat-template fraction must exceed **0.5** against baseline (ii). The 0.02 margin is fixed by this ADR as the smallest difference the thesis would act on; changing it requires a superseding ADR, and the CI must always be reported so that the margin can be judged independently of it.
- **Reported, not gating.** The respiration-rate block (when the rate target is active), `hf_power_rel` as an artefact indicator, and the per-task / per-group breakdowns.
- **Explicitly non-gating.** Any Stage-2 masked-position reconstruction score. It measures the pretext task, and protocol §1 judges pre-training only by whether it moves the headline above.

### 3.3 Criterion 3 — smoke run and resource record

- **Smoke.** One epoch, warm-up disabled, corpus capped at roughly 100 clips, batch 8, single device, development partition. Pass = completes without an assertion failure or a `NaN` loss, prints a geometry matching §2.2 (36 tubes, 250 signal tokens, $T = 25$, $R = 10$), and emits every record of §2.8.
- **Resource.** This ADR sets no memory ceiling, but it requires the measurement: record peak allocated and reserved device memory on the smoke run and on one step at the intended production batch, together with the precision mode. The geometry changed relative to the predecessor (wider encoder, two video streams, half the frame count), so the predecessor's figure does not transfer and may not be reused in the launch plan.

### 3.4 Evaluation commands (informative)

The evaluation sequence is: pre-train one run per fold → fine-tune with the **identical** recipe used by the compared arm, so the comparison isolates pre-training → pool the folds. Concrete invocations are toolchain facts and are given in Appendix A.

### 3.5 Acceptance artefacts

Per fold: the partition record, the selection record (rule, selected epoch, criterion value) and the per-epoch metric log. Per run: a pooled fold JSON carrying `overall` with the subject bootstrap CI, the per-subject and per-task blocks and the rate block when active; a per-subject metrics file; and the masking record of §2.8. Artefacts are outputs of a run, not part of this repository.

## 4. Consequences & Trade-offs

- **Positive.**
  - R4 is satisfied at no cost to the token geometry: the pre-trained kernel is used verbatim and the token count stays at 25 per 8 s window, so nothing downstream of the tokenizer changes.
  - The masking arithmetic is unchanged (§§2.3–2.4), so the decision is one configuration plus two curriculum parameters — and Criterion 1 can be cleared before any accelerator time is spent.
  - The three-stream budget satisfies the saturation precondition with a 125-token margin instead of the equality a two-physiological-stream design is forced into, so the visual floors hold even in the worst case.
  - The objective is single-target, which is what makes a negative result interpretable: the failure mode of §1.2(1) becomes visible instead of being absorbed by a second physiological channel.
- **Negative / Technical Debt.**
  - **Encoder-side physiological evidence shrinks by roughly 1.8×.** Enumerating the Mode-2 alignment: $E[K_{\text{resp}} \mid \text{Mode 2}] = 74.4$, so at $p_{\text{target}} = 0.50$ the expected kept respiration tokens per step is $37.2$ of 350 (≈ 10.6 %). The same enumeration for a two-physiological-stream design gives $E[K_{\text{bp}} + K_{\text{resp}} \mid \text{Mode 2}] = 136.6$, i.e. $68.3$ per step (≈ 19.5 %). The shared encoder therefore sees about half the waveform it used to, which shifts more of the burden onto the cross-modal path of contract 7. If pre-training stalls, the physiological cap or the Mode-2 draw range is the first thing to revisit, and that is a new decision rather than a tuning run.
  - **The second visual stream starts from a random tokenizer** (I2), so a null result for it is uninterpretable; any claim about that stream needs a run that routes the kernel to it.
  - **The curriculum makes the validation loss discontinuous** at $e = E_{\text{warm}}$ by construction, so only within-phase readings are legitimate (§2.7).
  - **Attribution is partial.** The stream set, the encoder width, the tokenizer geometry and the curriculum move together, so a positive result establishes “this arm reaches the headline”, not which change produced it. Separating the contributions needs an ablation grid, not this run.
  - **Blood pressure leaves the objective.** If BP remains a thesis target it needs its own tokenizer study; it cannot be recovered from this arm.
  - **Cost.** A wide encoder with two video adapters at 96 px, against a narrow backbone at twice the frame count.
- **Rollback / Rejection Condition.** Reject this decision if any of the following holds — the Stage-2 design of record must then be re-decided by a new ADR, because the four-stream arm it replaces is obsolete and cannot simply be restored:
  (i) Criterion 1 fails on the first full run, which means the policy was not executed and no result can be attributed to it;
  (ii) the smoke run cannot build the geometry of §2.2, or cannot produce the constant-budget shape after the tubelet change;
  (iii) the headline does not beat the template baseline in a majority of segments, i.e. the arm reproduces the collapse instead of curing it;
  (iv) the second-visual-stream question turns out to be decisive, in which case I2 must be re-opened before any modality claim is written.

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A` — this decision has not been executed. The measurements that motivate §1.2 and the arm this ADR is compared against are recorded in the evaluation notes listed in Appendix A.
- **Evidence Artifacts**: Appendix A carries the implementation mapping and the motivating measurements. Both are **informative**: no clause of §§1–4 depends on them, and they may be superseded, moved or deleted without amending this ADR.
- **Tests / Smoke Checks**: the acceptance procedure is §3.1 (policy conformance) followed by §3.3 (smoke and resource record). Where the present toolchain already covers part of it, Appendix A names the suites that exercise the invariant arithmetic.
- **Rollback / Rejection Condition**: §4.
- **Deferred, non-blocking decisions.** (1) Whether the curriculum boundary should coincide with the learning-rate warm-up (§2.7). (2) Whether the pre-trained kernel should also be routed to the second visual stream (I2, §4). (3) The production batch size and precision mode, which follow from the Criterion-3 measurement. None of these changes the policy defined in §2.

--------------------------------------------------------------------------------

## Appendix A. Mapping to the present toolchain (informative, non-normative)

This appendix exists for traceability only. Nothing in §§1–5 depends on it, it is expected to drift as the software changes, and when it does this ADR is unchanged and only the appendix is stale.

**Reference implementation of the policy.** The counting, alignment, feasibility and placement arithmetic of §§2.3–2.5 lives in `code/core/budget_masking.py` (counts) and `code/core/mask_placement.py` (indices) and is consumed by `code/core/multimae.py`; the encoder, the per-stream decoders and the conditioning of contract 7 under mechanism (a) (prefixing) are in the same module. Invariant coverage: `code/tests/test_budget_masking.py` (exact sum over 10 000 draws, caps, floors, replay determinism, infeasible tuples, and the single-physiological-stream case) and `code/tests/test_mask_placement.py` (whole temporal columns, per-sample placement, rank-independent counts).

**Initialisation path (I1–I4).** `code/models/pretrained.py` resolves the video-MAE sources, and `code/core/multimae.py::load_pretrained_encoder` maps the checkpoint into the encoder and into one visual adapter. The *explicit designation* required by I2 is not yet a property of that code: the recipient is currently resolved by adapter ordering, which happens to yield `tir` for this stream set but is not written into the run record. Closing that gap is implementation work, not a change to this decision.

**Configuration.** Both Stage-2 arm configs carry the I1 tokenizer geometry (`tubelet: 2,16,16`, `temporal_stride: 4`, `input_size: 96`, `sig_kernel: 32`, `fs: 1000`) and the §2.9 units (`streams: tir,rgb2d,resp`, `n_budget: 350`, the caps). `code/configs/pretrain/stage2_budget_3stream_curriculum.yaml` is this decision's arm: it adds `virtual_sensing_warmup_p` / `virtual_sensing_warmup_epochs` and designates `patch_stream: tir`. `code/configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml` omits both curriculum keys and therefore keeps `E_warm = 0`, i.e. `p(e) = p_target = 0.50` from the first step — the **no-curriculum comparator** with which R6 is isolated. The remaining parameters of §2.9 are options of `code/runners/run_pretrain.py` and `code/runners/_common.py`, where the YAML supplies defaults and explicit flags override them.

**Artefacts.** The geometry is load-time/model-level, so changing the tubelet and the decimation does not invalidate the ROI or signal caches — their keys carry the clip length, the frame rate and the sample rate, not the stride — and the signal sample rate is unchanged at 1000 Hz. The window hop (`clip_stride`, 4 s in both arms) only selects which windows exist, so it too stays outside the cache keys.

**Reference invocations.** Pre-training: `code/runners/run_pretrain.py`. Fine-tuning: `code/runners/run_finetune.py`. Pooled headline: `code/runners/run_evaluate_kfold.py`. Stage-2 diagnostic probe: `code/runners/run_eval_recon.py`. Per-step masking records: `code/engines/pretrain.py`.

**Motivating measurements (§1.2).** `code/notes/eval/baselines/tir_rgb2d_bp_resp/resp_recon_arms_ab.md`, `code/notes/eval/baselines/tir_rgb2d_bp_resp/multimae_style_panel.md`, `code/notes/eval/rgb2d/resp_recon_zeroshot.md`; the predecessor experiment plan `code/notes/plan/experiments/exp_20261121_fourstream_vits_random.md`; the metric protocol `code/notes/design/Evaluation_RespWaveform_protocol.md`; the session-cache decisions `code/notes/design/ADR-003-resp-session-cache-artefact.md` and `code/notes/design/ADR-004-bp-session-cache-artefact.md`; the terminology record `code/notes/GLOSSARY.md`; and an earlier, now retired draft of this decision, whose surviving clauses are enumerated under "Deviations from the earlier draft" below.

**Terminology note.** `code/notes/GLOSSARY.md` retires "AMMM" as a name for this policy — it denotes the removed per-step Dirichlet scheme. The canonical term for what this ADR defines is *constant-budget asymmetric multi-modal masking*.

**Deviations from the earlier draft (recorded so that nothing is changed silently).** Where the earlier draft of this decision differs, the difference is deliberate: (i) the draft was titled and numbered `ADR-003`, which this repository already assigns to the respiration session-cache artefact, and scoped that supersession to "the dedicated respiration recovery pipeline", whereas this ADR supersedes the four-stream arm outright; (ii) the draft counted epochs from 1 (`1 ≤ e ≤ 20`) and its reference implementation tested `epoch <= threshold` on a 0-based index — which yields 21 warm-up epochs — whereas §2.7 fixes a 0-based index with `e < E_warm`, and the warm-up length itself is raised from 20 to 50 epochs, so the draft's range matches neither the index convention nor the duration of this decision; (iii) the draft's Mode-1 split hard-coded the integer set `{5, …, 9}` and the constant `14`, whereas §2.4 derives both from `m`, `M` and `N/T`; (iv) the draft stated that the respiration decoder *cross-attends directly* to the visual latents, whereas contract 7 permits either conditioning mechanism and requires the one actually used to be recorded; (v) the draft claimed that saturating respiration "guarantees at least 9 spatial tubes" for the visual streams — 9 is the *total* tube count at that point, while the guaranteed per-stream floor is `m = 2`, i.e. 4 tubes in total (§2.4, §2.5); (vi) the draft gated acceptance on a Stage-2 masked-position correlation of `0.35`, which the evaluation protocol forbids as a pre-training gate and which is roughly two orders of magnitude above the measured predecessor values, so §3.2 gates on the subject-level headline instead and keeps the Stage-2 score as a non-gating diagnostic; (vii) the draft recorded the status `Accepted` under a different author label, while this document stays `Proposed` until the criteria of §3 are met.
