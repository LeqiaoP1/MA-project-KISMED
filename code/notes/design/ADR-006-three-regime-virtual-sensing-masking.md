# ADR-006: Three-Regime Constant-Budget Masking with Single-Modality Virtual Sensing (`tir`, `rgb2d`, `resp`)

- **Status**: Proposed
- **Date**: 2026-11-24
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/design/Evaluation_RespWaveform_protocol.md` — normative for the headline metric of §3.2 and not redefined here. Build order and gates: `code/notes/plan/features/ADR-006-three-regime-masking-plan.md` (written 2026-11-24; describes how this decision is built, never what it is).
- **Supersedes**: `code/notes/design/ADR-005-asymmetric-multimodal-masking-3streams.md`. The stream set, the tokenizer geometry, the constant budget, the caps and floors, the alignment arithmetic, the batch-determinism rules, the loss-key rules and the initialisation contract I1–I4 are **inherited verbatim** from ADR-005 and are restated here only where this decision changes them. Where this document is silent, ADR-005 governs.

**Normative status.** This ADR is the authority for the masking policy it defines. The policy, its units, its arithmetic, its feasibility conditions, its schedule, its parameter surface, its observability requirements and its acceptance criteria are all specified here, so the decision survives any change to the software that implements it. No statement in §§1–5 depends on a particular source file, configuration file, checkpoint or run, and none depends on another design document. Appendix A is the only place that maps the specification onto the present toolchain and is informative; Appendix B lists the formal corrections applied to the earlier (now withdrawn) draft of this decision.

## 1. Context & Problem Statement

### 1.1 Objective and scope

The objective is unchanged from ADR-005: reconstruct the respiration waveform of an unseen subject with the contact sensor absent. What changes is **which inputs the model is pre-trained to be able to do that from**. Three downstream evaluation targets are declared:

- **Target A — bimodal fusion** ($\text{tir} + \text{rgb2d} \to \text{resp}$);
- **Target B — thermal-only sensing** ($\text{tir} \to \text{resp}$), the deployed modality;
- **Target C — visible-only sensing** ($\text{rgb2d} \to \text{resp}$), commodity-camera operation.

**In scope.** Which tokens of which streams reach the shared encoder on a pre-training step, how many, and with what probability; the step classes the schedule is partitioned into; the arithmetic that keeps the budget constant across all of them; the conditioning contract for a stream that keeps nothing; the parameter surface; and the pre-registered acceptance criteria.

**Out of scope.** Encoder and decoder internals beyond the conditioning contract of §2.3(7); data selection, pre-processing, ROI construction and caching; the Stage-3 fine-tuning recipe; the measurement implementation.

### 1.2 Design pressures

ADR-005's two modes never produce a **single-camera** input. Its contract 3 gives every visual stream a floor of $m = 2$ tubes, and its Mode-1 split distributes 14 tubes over two visual streams. A model pre-trained under that policy has therefore never seen the encoder input that Target B and Target C require, and deploying it to either is a distribution shift that pre-training never rehearsed — in token composition and in the positional-embedding mix, if not in tensor shape, which is `[B, 350, D]` in every case. The three pressures below are what this ADR answers.

1. **The single-carrier regime is unreachable.** No combination of ADR-005 parameters produces a step on which one visual stream holds the whole budget and the other holds none. The capability that Targets B and C depend on is therefore never trained.
2. **The 25 % visual cap forbids the arithmetic.** A single stream can supply at most 9 of 36 tubes ($=225$ tokens) under $c_v = 0.25$, which is 125 short of the 350-token budget. Single-carrier operation is not a tuning question; it is excluded by the cap.
3. **A stream that keeps nothing needs a route to the loss.** When a visual stream keeps zero tokens its decoder sees only mask tokens and positions. Unless the conditioning contract gives that decoder a route to the other streams' latents, the reconstruction contributes no gradient to the shared encoder and the regime exists in name only.

### 1.3 Requirements

- **R1 — Constant visible budget.** The visible encoder token count is identical on every step: no varying shape, no padding, no `[MASK]` token in the encoder input.
- **R2 — Asymmetric bounds in kept units.** Any stream may be masked completely; no visual stream may exceed a bounded kept fraction; both limits are expressed as kept counts.
- **R3 — Single-carrier expressibility.** The policy must be able to allocate the entire budget to exactly one visual stream, with the other visual streams at zero, on a declared fraction of steps.
- **R4 — Gradient path for a zero-token stream.** A stream with zero visible tokens remains a reconstruction target and its loss must reach the shared encoder with non-zero gradient.
- **R5 — Verbatim visual tokenizer.** The visual tokenizer geometry must admit a published video-MAE kernel unchanged (§2.2 for the geometry, §2.10 for the initialisation contract).
- **R6 — Scheduled, not constant, regime probabilities.** Both the sensor-absent probability and the single-carrier probability within it are functions of training progress, and the single-carrier class is absent from the early phase.
- **R7 — Determinism and batch uniformity.** Counts are reproducible from `(seed, epoch, step)` and identical across ranks; placements differ per sample; every declared stream reports a loss key on every step.

## 2. Decision & Parameter Exposure

**Adopt a three-regime constant-budget policy over `{tir, rgb2d, resp}` that admits a single-carrier step class, with a two-phase curriculum that introduces it only after grounding.**

### 2.1 Declared stream set and naming

$\mathcal{M} = \{\text{tir}, \text{rgb2d}, \text{resp}\}$: two visual streams and one 1-D physiological stream. `tir` is the primary and deployed visual stream; `rgb2d` is the auxiliary texture stream over the same facial region, cropped from the same frames and sharing `input_size` and `tubelet`, so both occupy one spatial token grid and their kept tubes are interchangeable in the budget.

**Why exactly two visual streams.** Two are *required*, not preferred. Regime 3 is defined by one stream carrying the whole budget **while another is zeroed**, so a set with a single visual stream could not express it — the encoder input would be the same either way — and contract 2's $c_v = 0.25$ means a stream can hold all 350 tokens only when it is the sole carrier. A third visual stream is rejected because it cannot be given a meaningful share on a saturation step ($350 - 125 = 225$ tokens, minus three 2-tube floors, leaves 75 to divide across three streams) and because every additional visual adapter is a randomly initialised tokenizer under I2 that the budget would have to pay for.

**Naming, of record.** The stream called "RGB" in the draft is the declared stream **`rgb2d`** (the 2D-texture ROI at 112 px, resized to `input_size`); it is not the full-frame `rgb` stream. This ADR uses `rgb2d` throughout. The draft's step-class names are kept, and map to the inherited ones as follows — implementations keep the existing numeric mode values:

| this ADR | ADR-005 / `core.budget_masking` | $K_{\text{resp}}$ | carriers |
| :--- | :--- | ---: | :--- |
| **Regime 1** — multimodal inpainting | `MODE_INPAINTING = 2` | $>0$ | both visuals |
| **Regime 2** — bi-visual virtual sensing | `MODE_VIRTUAL_SENSING = 1` | $0$ | both visuals |
| **Regime 3** — single-modality virtual sensing | *new* | $0$ | exactly one visual |

A step's regime is resolved from its counts, never from the draw that produced them. Regime 3 subdivides into **3A** (`rgb2d` only) and **3B** (`tir` only), at equal probability within the class.

### 2.2 Masking units and tokenizer geometry

Geometry (unchanged from the two-mode policy this ADR supersedes, and stated here in full because ADR-005 no longer governs): decimation 4 → 50 frames; `tubelet (2,16,16)` → $T = 25$ temporal tokens per spatial tube; `input_size 96` → $6\times6 = 36$ spatial tubes; 900 tokens per visual stream; `sig_kernel 32` at `fs 1000` → 250 signal tokens; $R = 250/25 = 10$ signal tokens per visual token, with $R \cdot 32\,\text{ms} = 320\,\text{ms} =$ one tube. A **visual masking unit** is one spatial tube carrying its whole temporal column; a **physiological masking unit** is one temporal token. A 320 ms token spans two decimated samples 160 ms apart, i.e. a within-tubelet Nyquist of 3.125 Hz against a 0.15–0.50 Hz respiratory band. **This ADR claims no change in temporal resolution.**

### 2.3 Normative contracts

1. **Constant visible budget.** $K_{\text{tir}} + K_{\text{rgb2d}} + K_{\text{resp}} = n_{\text{budget}} = 350$ on every step (§6 below proves it for all three regimes). No encoder `[MASK]` token, no padding (R1).
2. **Kept caps, regime-indexed.** Let $c_v = 0.25$ be the multi-carrier cap: a visual stream that shares the budget keeps at most $\lfloor c_v \cdot 36 \rfloor = 9$ tubes ($=225$ tokens), so at least 75 % of it is a target. A visual stream that is the **sole** carrier may keep up to $14$ tubes ($=350$ tokens, $38.9\%$ kept, $61.1\%$ masked). The physiological cap stays $c_p = 0.50$ → 125 of 250 tokens (R2, R3).
3. **Visual visibility floor.** On Regime 1 and Regime 2 steps every visual stream keeps at least $m = 2$ tubes ($=50$ tokens). On a Regime 3 step the floor applies to **one** stream only: the carrier keeps all 14 tubes and every other visual stream keeps exactly 0. The floor exists so the encoder always has visual evidence; a carrier holding 350 tokens satisfies that purpose, so the floor is not relaxed, it is satisfied by the carrier.
4. **Regime resolution.** A step is Regime 1 iff $K_{\text{resp}} > 0$; Regime 2 iff every physiological stream keeps 0 and every declared visual stream keeps $>0$; Regime 3 iff every physiological stream keeps 0 and exactly one visual stream keeps $>0$. A Regime-1 draw that lands on $K_{\text{resp}} = 0$ is resolved and logged as Regime 2.
5. **Per-step counts, per-sample placement.** Counts are drawn once per step and shared by the mini-batch; kept indices are drawn independently per sample (R7).
6. **1-D placement policy.** Respiration indices are `scattered` or contiguous `span`; the choice is a property of the run, fixed before the first step, recorded with it, and identical in training and evaluation.
7. **Conditioned reconstruction for a zero-token stream.** A stream with $K_m = 0$ keeps its loss and must condition on the encoder latents by exactly one of: (a) **prefixing** the full encoder latent sequence into its decoder input before self-attention, or (b) **cross-attention** from its decoder queries to the encoder latents. The mechanism is fixed for a run and recorded; checkable requirement: for $K_m = 0$, $\partial\mathcal{L}_m/\partial z \neq 0$ (R4). Regime 3 exists only if this holds for the zeroed visual stream.
8. **Batch uniformity and determinism.** Every sample of a step carries the same $K_m$; counts are a function of `(seed, epoch, step)` alone and are identical on every rank without a collective; placements differ per sample and per rank (R7).
9. **Loss-key stability.** Every declared stream reports a loss term on every step and every rank, as an explicit zero when it has no masked position — Regime 3 depends on this: the zeroed visual stream contributes no encoder token yet remains a target.

### 2.4 Normative regime definitions

Parameters: $N = 350$, $T = 25$, $m = 2$, $M = 9$ (multi-carrier cap in tubes), $C_p = 125$, $S_v = 14$ (single-carrier allocation in tubes).

**Regime 1 — multimodal inpainting ($K_{\text{resp}} > 0$).**
1. Draw $\tilde K \sim \text{Uniform}\{0,\dots,C_p\}$; let $R = (N - \tilde K) \bmod T$; then
$$K_{\text{resp}} = \begin{cases}\tilde K + R & \text{if } \tilde K + R \le C_p\\ \tilde K - (T - R) & \text{otherwise}\end{cases} \;\Longrightarrow\; K_{\text{resp}} \in \{0,25,50,75,100,125\}.$$
$K_{\text{resp}} = 0$ resolves to Regime 2 (contract 4).
2. $S_{\text{total}} = (N - K_{\text{resp}})/T \in [9, 13]$ tubes, split clamped-uniform:
$$S_{\text{tir}} \sim \text{Uniform}\big[\max(m,\,S_{\text{total}} - M),\; \min(M,\,S_{\text{total}} - m)\big], \qquad S_{\text{rgb2d}} = S_{\text{total}} - S_{\text{tir}}.$$

**Regime 2 — bi-visual virtual sensing ($K_{\text{resp}} = 0$, both visuals $> 0$).**
$S_{\text{total}} = N/T = 14$ tubes, split by the same clamped-uniform rule, which with $m = 2$, $M = 9$ gives
$$S_{\text{tir}} \sim \text{Uniform}\big[\max(2, 14-9),\; \min(9, 14-2)\big] = \text{Uniform}\{5,\dots,9\}, \qquad S_{\text{rgb2d}} = 14 - S_{\text{tir}} \in [5, 9].$$

**Regime 3 — single-modality virtual sensing ($K_{\text{resp}} = 0$, one carrier).**

| submode | $S_{\text{carrier}}$ | tokens | $K_{\text{resp}}$ | other visual |
| :--- | ---: | ---: | ---: | ---: |
| 3A `rgb2d` only | 14 | 350 | 0 | $S_{\text{tir}} = 0$ |
| 3B `tir` only | 14 | 350 | 0 | $S_{\text{rgb2d}} = 0$ |

Each submode has probability $\tfrac12$ within Regime 3. On a 3A step the `tir` and `resp` decoders must reconstruct from `rgb2d` latents; on a 3B step the `rgb2d` and `resp` decoders must reconstruct from `tir` latents.

### 2.5 Normative apportionment algorithm

No literal appears below that is not a parameter of §2.9.

```
# N = n_budget, T = tokens per visual unit, m = visual floor (tubes),
# M = multi-carrier cap (tubes), C = physiological cap (tokens),
# S_single = single-carrier allocation (tubes), p(e), p_s(e) = schedules

function SPLIT_TUBES(S_total):
    lo = max(m, S_total - M)
    hi = min(M, S_total - m)
    return uniform_integer(lo, hi)          # inclusive; lo == hi is a single draw

function SAMPLE_STEP(e):
    if uniform_real(0,1) >= p(e):                       # Regime 1
        K_t_tilde = uniform_integer(0, C)
        R_m = (N - K_t_tilde) mod T
        K_resp = K_t_tilde + R_m if K_t_tilde + R_m <= C else K_t_tilde - (T - R_m)
        if K_resp > 0:
            S_total = (N - K_resp) / T
            S_first = SPLIT_TUBES(S_total)
            return REGIME_1, (S_first, S_total - S_first, K_resp)
    S_total = N / T                                     # Regimes 2 and 3
    if uniform_real(0,1) < p_s(e):                      # Regime 3
        if uniform_real(0,1) < 0.5: return REGIME_3A, (0, S_single, 0)
        else:                       return REGIME_3B, (S_single, 0, 0)
    S_first = SPLIT_TUBES(S_total)                      # Regime 2
    return REGIME_2, (S_first, S_total - S_first, 0)
```

Consequences, stated so they can be tested without reading the algorithm: Regime 1 does not draw $K_{\text{resp}}$ uniformly on $\{25,\dots,125\}$ — the alignment biases the distribution upward because the upward branch fires more often than the downward one, giving $E[K_{\text{resp}} \mid \text{Regime 1}] = 74.4$ by enumeration over the 126 draws; Regime 2 yields $S_{\text{tir}} \in \{5,\dots,9\}$; Regime 3 yields exactly one stream at 14 tubes and the other two at 0.

### 2.6 Feasibility preconditions

Validated when the configuration is resolved; an infeasible tuple is rejected **before the first step**, not discovered during training.

1. $N \bmod T = 0$ — Regimes 2 and 3 must map the whole budget onto whole visual units. ($350 \bmod 25 = 0$ ✓)
2. $2m \le N/T \le 2M$ — Regime 2 must be splittable inside floor and cap. ($350/25 = 14 \in [4, 18]$ ✓)
3. $N - C \ge 2mT$ — physiological saturation must leave both visual floors. ($225 \ge 100$ ✓)
4. $S_{\text{single}} = N/T$ and $S_{\text{single}} \le \text{units}_v$ — the carrier must be able to hold the whole budget in its own tube grid. ($14 \le 36$ ✓)
5. The carrier's kept fraction $S_{\text{single}}/\text{units}_v$ must not exceed the single-carrier cap of contract 2 ($14/36 = 38.9\%$ ✓). Since that cap is *defined* as $S_{\text{single}}/\text{units}_v$, rule 5 cannot be violated independently of rules 1 and 4: it is the statement that the kept fraction stays causal, and it is retained so that a configuration in which the two are separated is a detectable error rather than a silent one.
6. The declared set has exactly two visual streams — Regime 3 needs one to carry the whole budget and one to be zeroed, and the budget arithmetic of §2.1 leaves no other closed form.
7. The tubelet equals the pre-trained kernel (I1, §2.10).

### 2.7 Curriculum

Over the **0-based** epoch index $e$, with $E_{\text{warm}}$ the number of leading warm-up epochs:

$$p(e) = \begin{cases} p_{\text{warm}} & e < E_{\text{warm}}\\ p_{\text{target}} & e \ge E_{\text{warm}}\end{cases}, \qquad p_s(e) = \begin{cases} p_{s,\text{warm}} & e < E_{\text{warm}}\\ p_{s,\text{target}} & e \ge E_{\text{warm}}\end{cases}$$

with $p_{\text{warm}} = 0.10$, $p_{\text{target}} = 0.50$, $p_{s,\text{warm}} = 0.00$, $p_{s,\text{target}} = 0.20$, $E_{\text{warm}} = 50$.

**A Mode-2 draw that resolves to $K_{\text{resp}} = 0$ is sensor-absent, and takes the $p_s$ decision too.** In §2.5 the `p_s` test sits after both draws, so that class comprises the Mode-1 draws *and* the $1/126$ of Mode-2 draws whose aligned respiration count is zero (contract 4). The single-carrier probability is therefore a share of the **resolved** sensor-absent mass $p + (1-p)/126 = 0.50397$, not of $p$ alone; §3.2 and the plan's expected regime shares use that value.

| phase | epochs | $p$ | $p_s$ | step mix | objective |
| :--- | :--- | ---: | ---: | :--- | :--- |
| I — grounding | $e < 50$ | 0.10 | 0.00 | 90 % R1, 10 % R2, 0 % R3 | surviving respiration tokens anchor the visual latents |
| II — tri-target | $e \ge 50$ | 0.50 | 0.20 | 50 % R1, 40 % R2, 5 % 3A, 5 % 3B | complete respiratory dynamics from bimodal, thermal-only and visible-only input |

Normative properties: both probabilities are parameters of the policy, evaluated once per epoch, validated on $[0,1]$; a resolved tuple with $p_{\text{warm}} > p_{\text{target}}$ or $p_{s,\text{warm}} > p_{s,\text{target}}$ is a configuration error. The $p_{s,\text{warm}}$ ordering is checked **unconditionally**; the $p_{\text{warm}}$ ordering is checked only while Phase I exists ($E_{\text{warm}} > 0$), because its $0.10$ default would otherwise make a legal $p_{\text{target}} = 0.0$ configuration unbuildable — an exemption the single-carrier pair does not need, since $p_{s,\text{warm}}$ defaults to $0.00$. Validation masks follow the phase's own probabilities from a fixed seed sequence, so the validation curve is a *within-phase* observable and must not be pooled or plotted across $e = E_{\text{warm}}$. $E_{\text{warm}}$ is expressed relative to the run's epoch budget, not as an absolute epoch number; this ADR fixes the split, not the total.

### 2.8 Observability

- **per step:** epoch, step, the active $p(e)$ and $p_s(e)$, the resolved regime *according to contract 4* including the 3A/3B submode, kept token and unit counts of every declared stream, encoder sequence length.
- **per stream:** masked reconstruction loss on every step, explicit zero included (contract 9).
- **per epoch:** the realised fraction of each regime and the mean kept counts, which must reproduce the values implied by $p(e)$ and $p_s(e)$.
- **per run:** placement policy (contract 6), conditioning mechanism (contract 7), the designated kernel recipient (I2, §2.10), the full resolved tuple of §2.9.

A run whose records cannot answer §3.1 has not executed this decision, however well it trained.

### 2.9 Parameter surface

| parameter | symbol | type | default | admissible | meaning |
| :--- | :--- | :--- | ---: | :--- | :--- |
| `n_budget` | $N$ | int | 350 | such that §2.6 holds | constant visible tokens per step |
| `tokens_per_unit` | $T$ | int | 25 | from the geometry | tokens per visual spatial tube |
| `video_cap_ratio` | $c_v$ | float | 0.25 | $M = \lfloor c_v\cdot 36\rfloor \ge m$ | multi-carrier kept cap |
| `physio_cap_ratio` | $c_p$ | float | 0.50 | $(0,1]$ | respiration kept cap |
| `video_min_tubes` | $m$ | int | 2 | $[1, M]$ | carrier/multi-carrier floor |
| `single_carrier_tubes` | $S_v$ | int | `0` → resolves to 14 | `0` (derive) or exactly $N/T$ | allocation to the sole carrier; `0` is a derivation sentinel, and the resolved policy value is $N/T$ |
| `p_virtual_sensing` | $p_{\text{target}}$ | float | 0.50 | $[0,1]$ | sensor-absent probability, Phase II |
| `virtual_sensing_warmup_p` | $p_{\text{warm}}$ | float | 0.10 | $[0, p_{\text{target}}]$ | sensor-absent probability, Phase I |
| `p_single_modality` | $p_{s,\text{target}}$ | float | 0.20 | $[0,1]$ | single-carrier probability *within* the sensor-absent class, Phase II |
| `single_modality_warmup_p` | $p_{s,\text{warm}}$ | float | 0.00 | $[0, p_{s,\text{target}}]$ | as above, Phase I |
| `virtual_sensing_warmup_epochs` | $E_{\text{warm}}$ | int | 50 | $[0,\text{epochs}]$ | leading epochs at the warm-up probabilities; `0` disables |
| `physio_placement` | — | enum | `scattered` | `scattered`, `span` | 1-D placement policy |
| `physio_span_tokens` | — | int | 0 | required iff `span` | span length |
| `spatial_patches` | $\text{units}_v$ | int | 36 | must equal the tube count | assertion, not a knob |
| `physio_tokens` | $\text{units}_{1d}$ | int | 250 | must equal the signal-token count | assertion, not a knob |

Geometry the policy is defined over — not tuned here: `clip_duration 8.0 s`, `fps 25`, `temporal_stride 4`, `tubelet (2,16,16)`, `input_size 96`, patch 16, `fs 1000`, `sig_kernel 32`.

### 2.10 Initialisation and tokenizer-compatibility contract (I1–I4)

R5 is a constraint on the geometry, not on the software. These four clauses are normative **here**:
ADR-005 no longer governs, so they are stated in full rather than by reference, and no clause of this
ADR depends on a document outside it.

- **I1 — Verbatim kernel.** The visual tokenizer's temporal kernel must equal the pre-trained kernel,
  `(2, 16, 16)`. The projection is one 3-D convolution whose pre-trained weight has shape
  `[768, 3, 2, 16, 16]` at a 768-wide encoder, so matching the tubelet makes the inherited tensor fit
  bit-for-bit. A configuration whose tubelet differs is **infeasible** and must be rejected when the
  model is built. Neither temporal replication nor 2-D boxcar inflation is permitted: both destroy
  the temporal prior the initialisation is borrowed for and make the claim "pre-trained" false.
- **I2 — One designated recipient.** A video-MAE checkpoint carries exactly one 3-channel patch
  embed, so exactly one visual adapter can receive it. The recipient must be designated
  **explicitly, not resolved by adapter ordering**, defaults to `tir` (the deployed modality of
  §1.1), and is written into the run record. The other visual adapter is then randomly initialised,
  and the run record must say so.
- **I3 — Encoder geometry.** Encoder depth, width and head count must match the checkpoint; the
  shipped decision uses the VideoMAE **ViT-Base** geometry — 768-d, 12 blocks, 12 heads. Another
  width is not forbidden, but it requires a checkpoint of that width, which makes R5 a precondition
  rather than a preference. Positional embeddings cannot be inherited, so they are re-initialised
  from a deterministic space–time prior rather than left random; the 1-D streams take the same
  temporal prior on their own token grid.
- **I4 — Honest labelling.** If no pre-trained checkpoint exists at the chosen width, the run is a
  random-initialisation run, must be labelled as one, and cannot support any claim of R5.

*Technical debt accepted by I2, and load-bearing for Regime 3.* The auxiliary visual stream starts
from a random Conv3d, so any claim about its **content** is conditional on that tokenizer. This ADR
makes that debt sharper than its predecessor did: on submode 3A the sole carrier is `rgb2d`, i.e. the
randomly initialised adapter, so a null for Target C cannot distinguish "visible light carries
nothing" from "the tokenizer never converged". Re-opening Target C needs an arm that routes the
kernel to `rgb2d`, and that is a separate decision.

### 2.11 Rejected alternatives

- **Two modes only (ADR-005).** Superseded: R3 cannot be expressed, so Targets B and C are never rehearsed (§1.2).
- **Relaxing the cap globally to $14/36$.** Rejected: on Regime 1 and Regime 2 steps the 25 % cap is what keeps at least three quarters of each visual stream a reconstruction target, and those are the steps that carry the cross-modal alignment.
- **A single carrier with the other stream at its 2-tube floor** (i.e. near-single-modality). Rejected: it is not a single-camera input, so it does not rehearse the deployment condition, and it spends 50 tokens that the carrier needs to reach 350.
- **A third visual stream.** Rejected by the arithmetic of §2.1: on a saturation step the three 2-tube floors leave 75 tokens to divide across three streams, and every extra visual adapter is a random tokenizer under I2.
- **Introducing Regime 3 from the first step.** Rejected by R6 and pressure 1.2(3): with no visual reference at all on those steps, a cold start has nothing to align them against.
- **Mandating cross-attention (mechanism (b)) as the only conditioning route.** Rejected as over-specification: both mechanisms of contract 7 satisfy R4, and which one is used changes the objective, so it must be recorded rather than prescribed.
- **A per-step varying visible budget.** Rejected by R1.

## 3. Falsifiable Criteria

### 3.1 Criterion 1 — policy conformance (behavioural; no learning required)

1. **Exact sum.** Over ≥ 10⁴ sampled steps: $K_{\text{tir}} + K_{\text{rgb2d}} + K_{\text{resp}} = 350$ on 100 % of steps; encoder input `[B, 350, D]` on 100 % of steps.
2. **Bounds and discreteness.** On Regime 1 and 2 steps $K_{\text{tir}}, K_{\text{rgb2d}} \in \{50,\dots,225\}$ in multiples of $T$ and $K_{\text{resp}} \in \{0,25,\dots,125\}$. On Regime 3 steps the carrier is exactly 350 and the other visual exactly 0.
3. **Regime consistency.** The resolved regime follows contract 4 on 100 % of steps; the Regime-1 draws landing on $K_{\text{resp}} = 0$ are the predicted $1/126$.
4. **Schedule.** Realised regime shares match $p(e)$ and $p_s(e)$ within a binomial interval: Phase I $\approx$ 90/10/0, Phase II $\approx$ 50/40/5/5 (R6).
5. **Uniformity and replay.** Counts shared across the batch and reproducible from `(seed, epoch, step)`, independent of world size; placements differ per sample.
6. **Gradient path.** On a Regime-3 step, $\partial\mathcal{L}_{m}/\partial z \neq 0$ for the zeroed visual stream and for `resp` (R4, contract 7). One synthetic forward/backward suffices.
7. **Feasibility rejection.** Every infeasible tuple of §2.6 is rejected at configuration-resolve time.

### 3.2 Criterion 2 — scientific claim

- **Headline metric.** Band-limited Pearson (0.1–0.6 Hz) per `Evaluation_RespWaveform_protocol.md`; per session → median per subject → median with bootstrap 95 % CI over subjects; subject-disjoint test roles. Amplitude metrics are excluded from comparisons by protocol §4.1.
- **Tier 1 — go/no-go, relative.** Each arm is compared against the **ADR-005 arm fine-tuned with the identical Stage-3 recipe** on the same folds, and must exceed it by **≥ +0.02** with the CI reported. This is the gate: it decides whether the policy is adopted.
- **Tier 2 — thesis targets, absolute.** Reported per target, not gating: Target A $r \ge +0.45$ and MAE $\le 1.8$ BPM; Target B $r \ge +0.40$ and MAE $\le 2.0$ BPM; Target C $r \ge +0.30$ and MAE $\le 2.5$ BPM. These are the numbers the draft pre-registered; they are recorded here as targets whose attainment is reported per target.
- **Reference point on record.** The only measured sensor-free values on fold 0 are masked-position correlations from frozen Stage-2 decoders: subject median **+0.1129** [+0.0775, +0.1363] for the TIR-heavy allocation and **+0.0919** [+0.0705, +0.1085] for the RGB2D-heavy allocation, against +0.9959 for the same probe with respiration visible (`code/notes/eval/baselines/tir_rgb2d_resp/`). They are not the headline metric and must not be reported as one, but they are what the Tier-2 targets will be read against.
- **Non-gating.** Any masked-position reconstruction score, including everything above, measures the pretext task.

### 3.3 Criterion 3 — smoke and resource record

One epoch, warm-up disabled, corpus capped at ~100 clips, batch 8, single device: completes with no assertion failure or `NaN`, prints a geometry matching §2.2, emits every §2.8 record, and produces at least one step of each regime when the single-modality probability is forced to 1 and 0 in two short runs. Record peak allocated and reserved device memory at the production batch together with the precision mode; the geometry is unchanged from ADR-005, but the third regime changes the token composition and the figure must be re-measured.

### 3.4 Acceptance artefacts

Per fold: partition record, selection record, per-epoch metric log. Per run: pooled fold JSON with the subject bootstrap CI, per-subject metrics, and the masking record of §2.8.

## 4. Consequences & Trade-offs

- **Positive.** Targets B and C become rehearsed during pre-training rather than only at fine-tuning (§1.2, R3). Regimes 1 and 2 keep ADR-005's arithmetic exactly, so the tensors, caches and fine-tuning path are untouched and the change is a schedule plus one allocation branch. Criterion 1 can be cleared before any accelerator time is spent.
- **Negative / Technical debt.**
  - **The single-carrier step is a weaker pretext.** $61.1\%$ masked against ADR-005's $\ge 75\%$: 10 % of Phase-II steps train at lower difficulty, and that is the regime the deployment targets depend on.
  - **Only 10 % of Phase-II steps are single-carrier.** Whether that is enough to move Targets B and C is unmeasured; the mix is a parameter, but changing it after the run would be a new decision.
  - **Submode 3A rests on a random-init tokenizer.** Under I2 (§2.10) the VideoMAE kernel is routed to `tir`, so the `rgb2d` adapter is randomly initialised. On 3A steps the sole carrier is that adapter, so a null for Target C cannot distinguish "visible light carries nothing" from "the tokenizer never converged". Any claim about Target C needs an arm that routes the kernel to `rgb2d`.
  - **Ten percent of steps are spent on the stream with the weakest measured support.** The fold-0 measurements favour TIR over `rgb2d` in both allocation and content (§3.2 reference point). If 3A is inert, the cost is 5 % of Phase-II compute and a recorded null.
  - **The curriculum makes the validation loss discontinuous** at $e = E_{\text{warm}}$ by construction.
  - **Attribution is partial.** The third regime and the schedule move together; separating them needs a one-clause arm.
- **Rollback / Rejection Condition.** Reject this decision if (i) Criterion 1 fails on the first full run — the policy was not executed and no result is attributable; (ii) the smoke run cannot produce a Regime-3 step with the constant-budget shape; (iii) Tier 1 fails, i.e. the arm does not beat the ADR-005 arm by +0.02 on the headline, in which case ADR-005's two-mode policy is restored rather than patched; (iv) contract 7 cannot be shown to give the zeroed visual stream a non-zero gradient, in which case Regime 3 has no mechanism and must be withdrawn.

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A` — not executed.
- **Evidence Artifacts**: `code/notes/eval/baselines/tir_rgb2d_resp/README.md`, `resp_recon_arms_ab.md`, `kept_ratio_distribution.md` (the ADR-005 arm's measurements, cited in §3.2 only). Informative; no clause of §§1–4 depends on them.
- **Tests / Smoke Checks**: §3.1 as unit tests over the apportioner (regime resolution, exact sum, both cap regimes, zero-carrier floor, replay determinism, infeasible tuples) and one gradient test over `multimae`; then §3.3.
- **Rollback / Rejection Condition**: §4.
- **Deferred, non-blocking decisions.** (1) The proportion of single-carrier steps within Phase II. (2) Whether the `rgb2d` tokenizer should receive a pre-trained kernel before any Target-C claim (I2 debt). (3) Whether $E_{\text{warm}}$ should coincide with the learning-rate warm-up. None changes the policy defined in §2.

--------------------------------------------------------------------------------

## Appendix A. Mapping to the present toolchain (informative)

**Apportioner.** `code/core/budget_masking.py::ConstantBudgetApportioner` implements Regimes 1 and 2 today (`_mode2` and `_mode1` respectively, i.e. the inherited `MODE_INPAINTING`/`MODE_VIRTUAL_SENSING` values). This decision adds: a Regime-3 branch with the 3A/3B draw, the single-carrier cap in `_plan`'s invariant re-check, feasibility rules 4–5 of §2.6, and the `p_s` schedule alongside `p_at`. `_validate`'s existing `n_visual != 2` rejection is what makes contract 3's two-visual requirement explicit.

**Model.** `code/core/multimae.py::MultiModalMAE.forward` already skips the adapter of a stream with `plan.kept_tokens == 0` and conditions its decoder by prefixing `z` (mechanism (a)), which is the contract-7 path Regime 3 needs; the sentence there that a zero-kept stream "still has a gradient path" becomes a regime requirement rather than a Mode-1 edge case. Placements come from `code/core/mask_placement.py` unchanged.

**Runner and record.** `code/runners/run_pretrain.py` exposes the §2.9 masking flags and writes the masking record (`write_masking_manifest`); the new keys of §2.9 must be added to both, and `mode_by_counts`/regime resolution must report the 3A/3B submode.

**Configs.** One new sibling of `code/configs/pretrain/stage2_budget_3stream_curriculum.yaml` carrying the four schedule keys of §2.7 and `single_carrier_tubes`, with every other key identical — the delta discipline that config already uses against its no-curriculum comparator.

**Initialisation.** The I1–I4 contract of §2.10 is normative in this document; the debt recorded in §4 (3A rides a random-init `rgb2d` adapter because `patch_stream: tir`) follows from I2, not from a document outside this one.

## Appendix B. Deviations from the earlier draft (recorded so that nothing changes silently)

This decision was formalised from an earlier draft that was **withdrawn on 2026-11-24 and removed from the tree**; it is superseded by this document and nothing should be implemented from it. The deviations are listed here so that no change between the draft and this record is silent. Where the draft differs, the difference is deliberate and formal:

1. **Number and supersession.** The draft claimed `ADR-006` while superseding `ADR-002`, which ADR-005 already supersedes. This document supersedes **ADR-005**, and states which clauses it changes rather than restating all of them.
2. **Status.** The draft declared `Accepted` before any run; this document is `Proposed` until §3 is met.
3. **Title.** "AMMM" is retired by `code/notes/GLOSSARY.md` (it names the removed per-step Dirichlet policy). The title names the policy instead.
4. **Stream name.** The draft's "RGB" is the declared stream `rgb2d`; no stream named `rgb` is trained by this arm (§2.1).
5. **Epoch convention.** The draft wrote "Epochs 1–20" and its reference implementation tested `epoch <= warmup_epochs` on a 0-based index, which yields 21 warm-up epochs. §2.7 fixes a 0-based index with `e < E_{\text{warm}}` and sets $E_{\text{warm}} = 50$ — deliberately the same length as the arm this policy is compared against, so the comparison isolates the step class rather than a change of schedule.
6. **Step-class names.** The draft's "Regime 1/2" invert ADR-005's Mode 1/Mode 2 numbering (draft Regime 1 = `MODE_INPAINTING`). §2.1 keeps the draft's names and fixes the mapping in a table, and implementations keep the existing numeric values.
7. **Hidden constants.** The draft's §7 hard-coded `14`, `randint(5, 10)` and the split bounds; §2.5 contains no literal that is not a §2.9 parameter, and §2.9 exposes the four new schedule keys.
8. **Cap coherence.** The draft stated a 25 % cap and a 38.9 % "saturation capacity" without reconciling them. Contract 2 makes the cap regime-indexed and says which regime each value applies to.
9. **Floor wording.** The draft's §4.1 argued the floors keep "at least 9 spatial tubes"; 9 is the total tube count there, the per-stream guarantee being $m = 2$. Contract 3 states the floor per stream and states explicitly how Regime 3 satisfies it.
10. **Conditioning.** The draft mandated cross-attention; contract 7 permits either mechanism and requires the one used to be recorded.
11. **Acceptance.** The draft gated on Stage-2 masked-position correlations ($+0.45/+0.40/+0.30$). The evaluation protocol forbids those as pre-training gates, so §3.2 gates on the subject-level headline against the matched ADR-005 arm (Tier 1) and keeps the draft's absolute numbers as reported per-target figures (Tier 2).
12. **Missing sections added.** Feasibility (§2.6), parameter surface (§2.9), observability (§2.8), determinism and loss-key contracts (2.3 clauses 8–9), and the inherited initialisation contract (Appendix A).
