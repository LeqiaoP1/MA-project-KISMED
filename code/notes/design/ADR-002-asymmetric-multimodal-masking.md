# ADR-002: Asymmetric Multi-Modal Masking with a Constant 350-Token Encoder Budget

- **Status**: Accepted
- **Date**: 2026-11-19
- **Author**: Refined Study MA KISMED Research Team
- **Related specification**: `N/A`
- **Supersedes**: `N/A`

**Scope.** This ADR decides one thing: **how many tokens of each input stream reach the encoder on a training step, and which ones**. Model architecture, tokenization, initialisation, data preprocessing and the evaluation protocol are out of scope; they appear here only as the fixed quantities the masking policy is defined over.

--------------------------------------------------------------------------------

## 1. Context and Problem Statement

The masking policy decides which evidence the shared encoder may use on each step, so it determines what the model can be asked to reconstruct. Four requirements follow from the task itself:

1. **Complete sensor loss must be trainable.** In deployment the contact sensors are the components that fail while the cameras keep running, so the policy must be able to mask a physiological stream entirely and supervise its reconstruction from vision alone.
2. **Limits are needed in both directions, expressed in kept units.** A quantity that only says how much to mask cannot guarantee that a stream is ever fully masked, nor bound how much of a visual stream survives. Both limits are needed, and both must be stated as kept counts.
3. **The two kinds must be bounded differently.** A visual stream carries more than three times as many tokens as a physiological one, so a single symmetric rule lets visual tokens dominate the visible set on one step and disappear on the next.
4. **The visible-token count must be fixed.** If the total changes between steps, the encoder input changes shape on every step. The total can be constrained instead.

## 2. Decision and Parameter Exposure

Adopt the **Asymmetric Multi-Modal Masking (AMMM) policy with a constant 350-token visible budget**. This ADR is the decision of record and is self-contained: the rules below are sufficient to implement the policy and to test it.

### Masking units and per-stream totals

The policy operates on counts of *masking units*. A visual unit is one **spatial tube**: one spatial patch together with all of its temporal positions, so keeping a unit adds a whole temporal column. A physiological unit is one **temporal token**. The totals below are properties of the tokenizer, supplied to the policy as `spatial_patches` and `physio_tokens`:

| stream kind | masking unit | units per stream | tokens per unit | tokens per stream |
| :--- | :--- | ---: | ---: | ---: |
| visual (RGB, TIR-ROI) | spatial tube | 36 | 25 | 900 |
| physiological (BP, Resp) | temporal token | 250 | 1 | 250 |

Hence a visual count is always $K_v = 25\,S_v$, and any visual allocation is a multiple of 25.

### Core contracts

1. **Constant visible budget.** The visible tokens of all streams sum to $N_{\text{budget}} = 350$ on every training step:
   $$\sum_{m \in \{\text{RGB},\text{TIR},\text{BP},\text{Resp}\}} K_m^{(t)} = 350 \quad \forall t.$$
   The encoder therefore receives a fixed-length sequence with no `[MASK]` placeholders and no padding.
2. **Asymmetric kept caps.** Each visual stream keeps at most 25% of its tokens — at most 9 tubes, i.e. 225 tokens, so at least 75% of the visual stream is masked. Each physiological stream keeps at most 50%, i.e. 125 tokens.
3. **Visual visibility floor.** Each active visual stream keeps at least 2 tubes, i.e. 50 tokens (5.56% of its 900).
4. **Two modes.** Each step samples one of two modes, default probability 0.50 each:
   - **Mode 1 — pure virtual sensing:** $K_{\text{BP}} = K_{\text{Resp}} = 0$; the visual streams supply the whole budget, so every physiological token is a reconstruction target.
   - **Mode 2 — waveform inpainting:** each physiological stream receives an independent allocation in $[0, 125]$; the visual streams take the remainder in whole tubelets.
5. **Declared stream set.** The policy is defined for exactly two visual streams and one or two physiological streams. The budget arithmetic requires it: a single visual stream caps at 225 tokens and cannot supply Mode 1's 350, and three visual streams cannot survive physiological saturation, where 100 visual tokens remain while three 2-tube floors already require 150.
6. **Counts per step, placement per sample.** Counts are drawn once per step and shared by every sample of the mini-batch; the kept *indices* are drawn independently per sample. Sharing placements would give a step the statistical weight of a single sample.
7. **Conditioned reconstruction for a fully masked stream.** A stream with $K_m = 0$ has no encoder tokens of its own, so its decoder sees only mask tokens and positions. Its reconstruction must still condition on the encoder latents — by cross-attending to them or by receiving them in its decoder sequence. Otherwise Mode 1 reconstructs the waveform with no visual evidence and contributes no gradient to the encoder, and the mode exists in name only.

### Normative allocation rules

Every implementation and test must follow these rules.

- **Mode 1 visual split.** Draw $S_{\text{RGB}}$ uniformly from the integers $\{5,6,7,8,9\}$ and set $S_{\text{TIR}} = 14 - S_{\text{RGB}}$. Each stream keeps 5 to 9 tubes (125 to 225 tokens), inside its cap and floor.
- **Mode 2 physiological draw.** Draw $\tilde{K}_{\text{BP}}$ and $\tilde{K}_{\text{Resp}}$ independently and uniformly from the integers $[0,125]$.
- **Mode 2 tubelet alignment.** With
  $$R = \left[350 - (\tilde{K}_{\text{BP}} + \tilde{K}_{\text{Resp}})\right] \bmod 25,$$
  add $R$ to BP when $K_{\text{BP}} + R \le 125$; otherwise add it to Resp when $K_{\text{Resp}} + R \le 125$; otherwise reduce BP by $25 - R$. The remaining visual budget then maps to an integer tube count $S_{\text{total}} \in [4,14]$. Note that the remainder is always charged to BP first, so BP absorbs up to 24 extra tokens on non-aligned steps: a known, deliberate asymmetry.
- **Mode 2 visual split.** Draw $S_{\text{RGB}}$ uniformly from the integers in
  $$\left[\max(2,\,S_{\text{total}}-9),\; \min(9,\,S_{\text{total}}-2)\right]$$
  and set $S_{\text{TIR}} = S_{\text{total}} - S_{\text{RGB}}$, so both streams satisfy the 2-tube floor and the 9-tube cap.
- **Placement.** Video: choose $S_v$ spatial indices per sample and keep their full temporal column. 1-D: choose $K_p$ temporal indices per sample, either as contiguous spans or uniformly at random without replacement; the choice is fixed for a run and recorded with it.

### Feasibility precondition

The engine validates the conditions below at construction and rejects the configuration otherwise, because they are properties of the chosen numbers rather than of any particular draw. With $T$ tokens per visual unit, $m$ the visual floor in tubes, $M=\lfloor c_v \cdot \text{units}_v \rfloor$ the visual cap in tubes and $C_p=\lfloor c_p \cdot \text{physio\_tokens} \rfloor$ the per-stream physiological cap:

1. $N_{\text{budget}} \bmod T = 0$ — Mode 1 must map the whole budget onto whole visual units;
2. $2m \le N_{\text{budget}}/T \le 2M$ — Mode 1 must be splittable between the two visual streams inside both the floor and the cap;
3. $N_{\text{budget}} - 2C_p \ge 2mT$ — physiological saturation must leave at least the two visual floors.

With the defaults of this ADR ($T=25$, $m=2$, $M=9$, $C_p=125$): $350/25 = 14 \in [4,18]$ and $350 - 250 = 100 = 4$ units, so all three conditions hold, the third with equality.

### Invariants

Beyond the budget equation, every implementation must satisfy:

- **Batch uniformity.** $\forall b, b' \in \text{batch}_t:\ K_m^{(t)}(b) = K_m^{(t)}(b')$ — one count per stream per step, shared across the batch.
- **Encoder shape.** The tensor fed to the encoder is asserted to be `[B, 350, D]` on every step.
- **Loss-key stability.** Every declared stream reports its loss term on every step and on every rank, as an explicit zero when it has no masked position, because metric synchronisation issues one collective per key; a key set that depended on the sampled mode would desynchronise distributed runs. Mode 1 relies on this: the physiological streams contribute no encoder tokens yet remain reconstruction targets.
- **Mode-1 gradient path.** For a stream with $K_m = 0$, the gradient of its reconstruction loss must reach the shared encoder — $\nabla_z \mathcal{L}_m \neq 0$. This is the operational form of contract 7: without it, Mode 1 is a reconstruction with no visual evidence behind it.

### Rationale

The numbers above are chosen so that both operational extremes are representable and every intermediate draw stays legal. The argument is mechanical, not empirical.

- **Why 350 tokens.** The budget must satisfy both saturation cases fixed by contracts 1-4. At full physiological saturation the visual streams need their two 2-tube floors, i.e. 100 tokens, so the budget cannot be below $250 + 100$. At full sensor loss the two visual streams must supply the entire budget, so their combined cap of 450 must exceed it or Mode 1 is unsatisfiable. A 340-token budget fails both tests: 90 visual tokens is neither a multiple of 25 nor above the 100-token floor sum.
- **Why the visual cap is 25%.** The evidence this task depends on is micro-scale — capillary colour oscillation for rPPG, thermal boundary-layer dissipation at the nostrils — spread over a grid of only 36 units. A much higher mask ratio keeps roughly three patches and can drop the anatomical region of interest entirely; a much lower one leaves enough local correlation for interpolation shortcuts. Masking at least 75% of a visual stream destroys those shortcuts while keeping spatial anchors, and the cap also stops two 900-token grids from dominating the 250-token physiological streams inside a shared encoder.
- **Why the 2-tube floor.** It is the smallest allocation that still carries spatial context, and it is what makes the budget solvable at physiological saturation ($4 = 2 + 2$ tubes). Without a floor the visual streams could round to zero on a high-physiology step and leave the encoder with no visual evidence.
- **Why physiology is capped at 50%.** A fully visible waveform gives the encoder a self-contained route to the reconstruction target and lets it ignore the visual streams during cross-modal attention. Half-visible still leaves enough tokens for within-waveform interpolation and cross-talk.
- **Why both modes, at equal default probability.** Pure virtual sensing alone never trains intra-waveform interpolation, because no physiological token reaches the encoder; inpainting alone never trains complete waveform generation from vision, because a partly visible waveform short-cuts part of it. Splitting the steps gives both regimes comparable exposure.
- **Why a fully masked stream must still be conditioned.** Mode 1 masks every physiological token, so those decoders have no input tokens of their own. If their prediction cannot read the encoder latents, the waveform is reconstructed without visual evidence and the step contributes no gradient to the encoder — the mode would exist in name only. Contract 7 therefore makes the conditioning path explicit rather than implied.
- **Why the masking unit is a full temporal tube.** Masked video must remain a dense rectangle with no padding, so a visual unit keeps all temporal positions of one spatial patch. Counts then stay multiples of 25, and the visual budget is a small integer number of tubes — which is what makes the Mode-2 discreteness rule tractable.
- **Why counts are per step and placements per sample.** Sharing placements would give a step the statistics of a single mask and make the rest of the batch redundant; sharing counts keeps the batch a dense tensor. The two properties are independent, so contract 6 states both.

### Rejected alternatives

- **Symmetric proportional allocation.** Rejected: variable visible-token counts, no domain-asymmetric bounds, and no way to express complete sensor loss.
- **Uncapped physiological kept ratio.** Rejected: a fully visible physiological stream lets the encoder ignore the visual streams. Capped at 50% for this reason.
- **340-token budget.** Rejected as infeasible: physiological saturation leaves $340-250=90$ visual tokens, which is not a multiple of 25, and the two 2-tube floors already require $2 \times 50 = 100$ tokens. No allocation satisfies the budget, the tubelet discreteness and the visual floors at once.
- **Constant budget with three or more visual streams.** Rejected by contract 5: at physiological saturation 100 visual tokens remain, below three 2-tube floors (150).

### Components

1. **Apportionment engine.** Draws one kept-unit count per declared stream per step and returns the mode and the counts. It holds no tensors, no modality data and no token grids, and takes an injectable random-number generator so a step sequence can be replayed. Inputs: the per-stream unit counts, the caps, `video_min_tubes`, `n_budget`, `p_virtual_sensing` and the tubelet-discreteness rule. It validates the budget, caps and floors before returning, and rejects a step that violates them.
2. **Placement layer.** Turns the step's counts into per-sample masks — tube placement for video, temporal-index placement for 1-D — independently of the counts.

The masking policy's only interface to the model is the set of visible indices per stream; it produces no placeholder tokens and no padding.

### Exposed parameters

Configuration files provide defaults; explicit command-line arguments take precedence.

- `n_budget` / `--n_budget` (int, default `350`, allowed `[100, 450]`): constant visible-token budget.
- `video_cap_ratio` / `--video_cap_ratio` (float, default `0.25`, allowed `(0.05, 0.50]`): kept-ratio ceiling per visual stream.
- `physio_cap_ratio` / `--physio_cap_ratio` (float, default `0.50`, allowed `(0.0, 1.0]`): kept-ratio ceiling per physiological stream.
- `video_min_tubes` / `--video_min_tubes` (int, default `2`, allowed `[1, 9]`): kept-tube floor per active visual stream.
- `p_virtual_sensing` / `--p_virtual_sensing` (float, default `0.50`, allowed `[0.0, 1.0]`): probability of Mode 1.
- `spatial_patches` / `--spatial_patches` (int, default `36`, allowed `[36, 64]`): unit count per visual stream (tokenizer input).
- `physio_tokens` / `--physio_tokens` (int, default `250`, allowed `[25, 250]`): unit count per physiological stream (tokenizer input).

--------------------------------------------------------------------------------

## 3. Falsifiable Criteria

- **Criterion 1: the policy samples the intended operating regimes.** Over a logged window of at least 1000 consecutive steps, the fraction of Mode-1 steps must match `p_virtual_sensing` within its binomial sampling interval; every Mode-1 step must log $K_{\text{BP}} = K_{\text{Resp}} = 0$ with the visual streams supplying the full budget; and every Mode-2 step must log a physiological allocation inside $[0,125]$ with visual counts in whole tubelets. This criterion is behavioural: it tests that the policy delivers the operating spectrum it was designed for.
- **Criterion 2: invariants hold under sampling.** A unit test over at least $10^4$ draws must confirm, for both modes: the exact-sum invariant; the per-stream caps and floors; visual counts that are multiples of 25; identical counts across a batch with independently drawn placements; and deterministic replay from a fixed seed. A forward/backward check must confirm that a $K_m = 0$ stream's loss leaves a non-zero gradient on the shared encoder. A 10-epoch smoke run with `--max_clips 100 --batch_size 8 --n_budget 350` must complete without assertion failures, token-count deviations or NaN losses, and log every stream count and the sampled mode.
- **Criterion 3: the budget is actually constant.** Logging the encoder sequence length on every step must show exactly `n_budget` for 100% of steps, and the per-step per-stream counts must match the logged mode.

### Evaluation Commands

- Pre-train with the policy at its defaults (`n_budget=350`, `p_virtual_sensing=0.50`, `video_cap_ratio=0.25`, `physio_cap_ratio=0.50`), logging the sampled mode and the per-stream counts on every step.
- Report the downstream virtual-sensing metric on the five subject-disjoint folds (`--task virtual_sensing --split groupkfold`) as a descriptive result; it is not an acceptance gate for the policy.

### Acceptance Artifacts

- The invariant test suite for the apportionment engine and the placement layer.
- Per-step masking logs (mode, per-stream counts, encoder length).
- `metrics_virtual_sensing_groupkfold.json`, reported for context rather than as a pass/fail threshold.

--------------------------------------------------------------------------------

## 4. Consequences and Trade-offs

### Positive

- Half of the steps train the exact deployed failure mode: complete physiological sensor loss with the cameras still running.
- The asymmetric caps bound both failure directions — visual tokens cannot swamp the encoder, and physiology cannot be handed to the model fully visible.
- The 2-tube floor keeps anatomical context on every step, while ≥75% visual masking suppresses local copy-paste shortcuts.
- A constant visible-token count removes per-step shape variation and makes the encoder workload deterministic.
- Counts shared per step and placements per sample preserve dense batch execution without collapsing a step to one mask.

### Negative and Technical Debt

- **The visual and physiological budgets are coupled.** Because the total is fixed, physiology reaching its caps forces both visual streams onto their 50-token floors; the policy cannot be generous to both kinds at once.
- **The mode is a step-level coin flip.** With $p = 0.50$, half the steps carry no encoder-side physiological evidence, so each step's loss value is not comparable across steps; only aggregated metrics are.
- **The remainder rule favours BP.** Mode-2 alignment charges $R$ to BP before Resp, giving BP up to 24 extra visible tokens on non-aligned steps.
- **The constants are interdependent.** The three feasibility conditions of §2 hold only for the shipped numbers; changing the budget, a cap, the floor or either unit count can make the policy unsatisfiable, and the engine must reject such a configuration at construction rather than at the first step.

--------------------------------------------------------------------------------

## 5. Reproducibility Record

- **Run IDs:** `N/A`. No run under this policy is recorded yet; the Slurm job IDs are recorded once the jobs are submitted.
- **Evidence artifacts:** the apportionment-engine and placement-layer implementations, the masking configuration generated from the parameters in §2, and the per-step masking logs.
- **Tests and smoke checks:** the invariant suite of Criterion 2; the encoder-length check of Criterion 3; the 10-epoch smoke run.
