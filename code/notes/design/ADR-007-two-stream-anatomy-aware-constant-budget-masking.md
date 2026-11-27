# ADR-007: Two-Stream Constant-Budget Masking with Anatomy-Aware Tubelet Sampling (`tir`, `resp`)

- **Status**: Implemented (Pending Eval)
- **Date**: 2026-11-25
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/plan/features/ADR-007-two-stream-anatomy-aware-masking-plan.md` (build order and conformance gates), `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md` (normative for the headline metric, frequency band, and subject-level aggregation used in §3), `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md` (normative for subject-disjoint 5-fold partitioning and validation-only model selection), `code/notes/GLOSSARY.md` (canonical terminology). All masking, geometry, stratification, curriculum, and feasibility rules in this ADR are self-contained.
- **Supersedes**: `code/notes/design/ADR-006-three-regime-virtual-sensing-masking.md` for the dedicated Phase-1 two-stream (`tir`, `resp`) pre-training milestone. Where multi-camera bimodal or visible-only pre-training (`tir`, `rgb2d`, `resp`) is later reopened, ADR-006 remains the record for the three-regime fusion policy; for the single-camera thermal-infrared respiration foundation, this ADR is the sole design of record.

**Normative status.** This ADR is the self-contained architectural specification for the two-stream constant-budget, anatomy-aware masking policy. Every contract, mathematical definition, feasibility precondition, curriculum rule, observability requirement, and acceptance criterion is stated independently of any software implementation, module layout, or configuration file, so the decision survives arbitrary rewrites of the training stack.

---

## 1. Context & Problem Statement

### 1.1 Objective and scope

The objective of Stage-2 multimodal masked pre-training under this decision is to establish a verifiable, high-signal cross-modal representation that reconstructs the continuous respiration waveform (`resp`) of an unseen subject from thermal-infrared facial video (`tir`) alone under complete contact-sensor absence ("virtual sensing"). This serves as the **Phase-1 foundation** before re-introducing auxiliary visible-light or secondary physiological modalities.

The downstream scientific headline metric, its respiratory passband ($0.1\text{--}0.6\text{ Hz}$), its session-assembly and per-session $\to$ per-subject aggregation, and its bootstrap confidence interval are governed by `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md` and are not redefined here.

**In scope.** The declared two-stream modality set (`tir`, `resp`); the spatiotemporal masking units and token alignment contract; the constant 200-token visible encoder budget; the anatomy-aware stratified spatial tube sampling policy; the two operational step modes and their discrete apportionment rules; the two-phase virtual-sensing curriculum; the encoder initialization compatibility contract; observability requirements; and pre-registered falsifiable acceptance criteria.

**Out of scope.** Specific software architectures, module boundaries, or configuration file schemas beyond the decoder conditioning contract of §2.4(8); raw data curation, ROI landmark extraction, and offline cache storage formats; Stage-3 fine-tuning objectives and head architectures; and multi-camera fusion policies (deferred until Phase-1 succeeds).

### 1.2 Design pressures

Predecessor multi-stream constant-budget designs (the four-stream policy of ADR-002 and the three-stream curriculum and three-regime policies of ADR-005 and ADR-006) coupled `tir` with visible-light 2D-texture (`rgb2d`) — and originally blood pressure (`bp`) — under a 350-token visible encoder budget. Empirical baseline measurements and theoretical analysis identify three structural pressures that prevent clean validation of the `tir` $\to$ `resp` pathway in that multi-stream setting:

**Measured predecessor context (evidence).** On fold 0 of the ADR-005 three-stream (`tir`, `rgb2d`, `resp`) 150-epoch Stage-2 baseline (`code/notes/eval/baselines/tir_rgb2d_resp/README.md`), frozen zero-shot masked-position respiration correlation reached $+0.996$ when respiration tokens were partially visible, but dropped to a subject median of $+0.1129$ ($[+0.0775, +0.1363]$) under TIR-heavy virtual sensing and $+0.0919$ ($[+0.0705, +0.1085]$) under `rgb2d`-heavy virtual sensing. Input-content ablation showed that destroying `tir` temporal alignment significantly degraded correlation ($\Delta = -0.0539$, $t = -9.96$), whereas rolling or zeroing `rgb2d` produced no measurable effect ($\Delta = -0.0012$ and $-0.0028$). Meanwhile, pooled 5-fold Stage-3 fine-tuned evaluation on the TIR-ROI lineage (`code/notes/eval/archive/tir_resp/resp_5fold_cv_96respgrid.md`) achieved a subject-level band-limited Pearson median of $+0.542$ ($[+0.493, +0.576]$) against a $+0.537$ template baseline, confirming that strong respiratory information exists in the TIR ROI.

1. **Auxiliary visual gradient dilution and uninitialized tokenizer debt (hypothesis + evidence).** Coupling an auxiliary visible-light stream (`rgb2d`) with `tir` forces the shared encoder to devote capacity and gradient updates to visible appearance, identity, and reflectance reconstruction, while mechanical respiratory motion in facial RGB is typically sub-pixel and heavily attenuated by $16 \times 16$ spatial patch pooling. Furthermore, because a single-stream pretrained video encoder carries only one 3-channel patch projection, any second visual stream starts from random tokenizer weights, creating an asymmetric optimization handicap whose content contribution cannot be disentangled from tokenizer under-convergence.
2. **Spatial blindspot under uniform tubelet sampling (mathematical property).** Over a $96 \times 96$ TIR mouth-nose ROI partitioned into $16 \times 16$ spatial patches, the spatial grid has $6 \times 6 = 36$ tubes. The convective airflow thermal plume — the primary physical signature of inhalation cooling and exhalation warming in TIR video — is localized to roughly 2 to 4 patches covering the nostril apertures and subnasal philtrum. Under unconstrained uniform random sampling without replacement of $S_{\text{tir}}$ tubes from 36, the probability of observing at least one patch from a 4-patch nasal core is:
   $$P(\text{core hit} \mid S_{\text{tir}}) = 1 - \frac{\binom{36 - 4}{S_{\text{tir}}}}{\binom{36}{S_{\text{tir}}}}$$
   which evaluates to **$39.0\%$** at $S_{\text{tir}} = 4$, **$53.5\%$** at $S_{\text{tir}} = 6$, and **$65.2\%$** at $S_{\text{tir}} = 8$. Across $35\%\text{--}61\%$ of training samples, uniform masking hides the thermal plume completely, forcing the encoder to predict respiration from passive cheek or lip background.
3. **Budget oversizing for a two-stream task (mathematical property).** Retaining a 350-token visible budget (originally sized for three to four streams) when only `tir` and `resp` remain would force the sole visual stream to keep $350 / 25 = 14$ of 36 spatial tubes ($38.9\%$ kept, $61.1\%$ masked) under pure virtual sensing. High spatiotemporal redundancy in facial video requires substantially higher masking ratios ($75\%\text{--}85\%$, matching standard video masked autoencoding) to prevent trivial local interpolation and compel temporal reasoning.

### 1.3 Requirements

The decision in §2 satisfies seven verifiable architectural requirements:

- **R1 — Strict two-stream focus.** Pre-training operates exclusively on `{tir, resp}` so that every visual encoder token and gradient update serves thermal-infrared physiological sensing with no auxiliary-stream interference.
- **R2 — Right-sized constant visible budget.** The total visible token count entering the shared encoder is fixed at $N = 200$ on every training step, yielding dense rectangular `[B, 200, D]` tensors with zero padding, no encoder `[MASK]` tokens, and a high visual masking ratio ($\ge 77.8\%$).
- **R3 — Guaranteed anatomical core visibility.** On every training sample, the visual tube selection guarantees at least one visible spatial tube drawn from the designated nasal-plume core subgrid while preserving peripheral spatial context.
- **R4 — Asymmetric kept bounds and clean mode separation.** Pure virtual sensing ($K_{\text{resp}} = 0$) and waveform inpainting ($K_{\text{resp}} > 0$) are strictly separated, with the physiological stream capped at $c_p = 0.40$ ($100$ of $250$ tokens) and visual counts always exact multiples of the temporal token count per tube.
- **R5 — Gradient path under complete sensor blackout.** When $K_{\text{resp}} = 0$, the respiration decoder remains active and its reconstruction loss must produce a non-zero gradient with respect to the shared visual encoder latents ($\partial \mathcal{L}_{\text{resp}} / \partial z \neq 0$).
- **R6 — Verbatim pretrained visual tokenizer compatibility.** The visual stream geometry admits a published 3-channel $(2, 16, 16)$ 3-D video-MAE patch kernel verbatim with no temporal inflation, channel duplication hacks, or uninitialized secondary visual adapters.
- **R7 — Scheduled curriculum, determinism, and batch-uniform shapes.** Virtual-sensing probability follows a two-phase epoch curriculum; per-step token counts are deterministic from `(seed, epoch, step)` and uniform across all samples and distributed ranks; spatial and temporal mask placements vary independently per sample.

---

## 2. Decision & Parameter Exposure

**Adopt a dedicated two-stream (`tir`, `resp`) constant-budget masking policy with a fixed 200-token visible encoder budget, anatomy-aware stratified spatial tube sampling over the $6 \times 6$ TIR ROI patch grid, discrete tube-aligned remainder apportionment, and a two-phase virtual-sensing curriculum.**

> **Abstraction boundary.** In accordance with this ADR's high-level design mandate, no existing source modules, runner scripts, or configuration file paths are cited below. The parameter surface in §2.10 defines the normative logical knobs that any conforming implementation must expose and record.

### 2.1 Declared stream set

The active stream set is $\mathcal{M} = \{\text{tir}, \text{resp}\}$:

- **`tir`** (visual, 3 channels, `[B, 3, T_f, H, W]`): cropped $96 \times 96$ thermal-infrared mouth-nose ROI represented in its native 3-channel false-colour rendering.
- **`resp`** (1-D physiological, 1 channel, `[B, 1, L]`): continuous respiration waveform aligned to the identical temporal clip window and normalized per clip.

Dropping `rgb2d` and `bp` from Phase 1 removes two confounders simultaneously: (i) multi-stream gradient competition between appearance/pulse and airflow temperature, and (ii) the random-initialization tokenizer debt inherent to having more visual streams than pretrained patch kernels.

### 2.2 Spatiotemporal topology, masking units, and alignment invariant

| Quantity | Value | Derivation / Contract |
| :--- | :--- | :--- |
| Clip duration | $8.0\text{ s}$ | Fixed temporal window shared by `tir` and `resp` |
| Source video frame rate / decimation | $25\text{ fps}$ / stride $4$ | $200$ source frames $\to 50$ decimated frames ($6.25\text{ fps}$, $\Delta t = 160\text{ ms}$) |
| Visual spatial input / patch size | $96 \times 96\text{ px}$ / $16 \times 16\text{ px}$ | $G_h \times G_w = 6 \times 6 = 36$ spatial patch positions |
| Visual 3-D tubelet | $(P_t, P_h, P_w) = (2, 16, 16)$ | $T = 50 / 2 = 25$ temporal tokens per spatial tube; $320\text{ ms}$ per tubelet |
| Visual masking unit | 1 spatial tube ($T = 25$ tokens) | Full temporal column at one spatial position; $36 \times 25 = 900$ total visual tokens |
| Physiological token count | $250$ temporal tokens | $32\text{ ms}$ per token over $8.0\text{ s}$ (e.g., $8000$ samples at $1000\text{ Hz}$ with $32$-sample non-overlapping tokens, or any anti-aliased grid yielding $32\text{ ms}$ per token) |
| Physiological masking unit | 1 temporal token | $250$ total physiological tokens |
| Space-time alignment ratio $R$ | $10$ | $R = 250 / 25 = 10$ signal tokens per visual tubelet step; $10 \times 32\text{ ms} = 320\text{ ms}$ |
| Encoder visible token budget $N$ | $200$ tokens | $K_{\text{tir}}(t) + K_{\text{resp}}(t) \equiv 200$ on every step |

**Sampling adequacy.** At $6.25\text{ fps}$ effective visual sampling, the within-tubelet Nyquist frequency is $3.125\text{ Hz}$ ($187.5\text{ breaths/min}$), which is more than $6\times$ the upper respiratory passband edge ($0.50\text{--}0.60\text{ Hz}$, $30\text{--}36\text{ breaths/min}$). **Hard alignment invariant:** $R = N_{\text{resp}} / T$ must be an exact positive integer and $R \cdot (\text{signal token duration}) = (\text{visual tubelet duration})$, validated at model construction time.

### 2.3 Anatomy-aware stratified spatial partitioning

Index the $6 \times 6$ spatial patch grid of the $96 \times 96$ TIR ROI by row $r \in \{0, \dots, 5\}$ and column $c \in \{0, \dots, 5\}$. Partition the 36 spatial tube positions into two disjoint subsets:

1. **Anatomical Core ($\mathcal{A}_{\text{core}}$):** The configurable subgrid targeting the nostril apertures and subnasal philtrum where cyclic convective airflow temperature modulation occurs. Default bounds:
   $$\mathcal{A}_{\text{core}} = \{(r, c) \mid r \in \{2, 3\},\; c \in \{2, 3\}\}, \qquad |\mathcal{A}_{\text{core}}| = 4\text{ tubes.}$$
2. **Peripheral Tissue Context ($\mathcal{A}_{\text{periph}}$):** All remaining spatial tubes covering surrounding facial anatomy (nasal bridge, cheeks, upper lip, chin):
   $$\mathcal{A}_{\text{periph}} = \{(r, c) \mid (r, c) \notin \mathcal{A}_{\text{core}}\}, \qquad |\mathcal{A}_{\text{periph}}| = 32\text{ tubes.}$$

**Stratified selection rule.** Given a step's visual tube budget $S_{\text{tir}} \in \{4, 5, 6, 7, 8\}$:

1. Draw the core tube count per step:
   $$S_{\text{core}} \sim \text{Uniform}\{S_{\text{core}}^{\min}, \dots, S_{\text{core}}^{\max}\} \qquad (\text{default } \{1, 2\}).$$
2. Assign the remaining visual budget to peripheral context:
   $$S_{\text{periph}} = S_{\text{tir}} - S_{\text{core}}.$$
3. **Independently for each sample in the mini-batch**, sample $S_{\text{core}}$ distinct spatial tubes without replacement from $\mathcal{A}_{\text{core}}$ and $S_{\text{periph}}$ distinct spatial tubes without replacement from $\mathcal{A}_{\text{periph}}$.

**Consequence.** Every sample is guaranteed to expose $1$ or $2$ tubes from $\mathcal{A}_{\text{core}}$ ($P(\text{core hit}) \equiv 1.0$) while keeping $2$ to $3$ core tubes masked as high-value reconstruction targets and retaining $2$ to $7$ peripheral context tubes for spatial reference.

### 2.4 Normative contracts

These ten contracts completely define the policy:

1. **Constant visible budget.** $K_{\text{tir}}(t) + K_{\text{resp}}(t) = N = 200$ on every step $t$. The encoder input shape is strictly `[B, 200, D]` with no padding and no `[MASK]` tokens (R2).
2. **Asymmetric kept caps.** The visual stream keeps at most $S_{\text{tir}}^{\max} = N / T = 8$ of $36$ tubes ($22.22\%$ kept; $77.78\%$ masked). The physiological stream keeps at most $C_p = \lfloor c_p \cdot 250 \rfloor = 100$ tokens ($c_p = 0.40$; $\ge 60\%$ masked) (R2, R4).
3. **Visual visibility floor.** At maximum physiological allocation ($K_{\text{resp}} = 100$), the visual stream retains $S_{\text{tir}}^{\min} = (200 - 100) / 25 = 4$ tubes ($100$ tokens, $11.11\%$ kept), which strictly exceeds $S_{\text{core}}^{\max}$ and preserves peripheral spatial context on every step (R3, R4).
4. **Two strictly separated modes.**
   - **Mode 1 (Pure Virtual Sensing):** $K_{\text{resp}} = 0$, $S_{\text{tir}} = 8$ tubes ($K_{\text{tir}} = 200$).
   - **Mode 2 (Waveform Inpainting):** $K_{\text{resp}} > 0$ aligned to a non-zero multiple of $T$ within $[T, C_p]$, and $S_{\text{tir}} = (N - K_{\text{resp}}) / T$.
   A step's realized mode is always resolved from its counts: Mode 1 **iff** $K_{\text{resp}} = 0$, Mode 2 **iff** $K_{\text{resp}} > 0$ (R4).
5. **Anatomy-aware spatial stratification.** Visual tube placement obeys §2.3 on every step and every sample unless explicitly ablated (`spatial_sampling = uniform` for controlled comparisons) (R3).
6. **Per-step counts, per-sample placement.** The tuple $(K_{\text{tir}}, K_{\text{resp}}, S_{\text{tir}}, S_{\text{core}}, S_{\text{periph}})$ is drawn once per step and shared across the mini-batch; the kept visual tube indices (within $\mathcal{A}_{\text{core}}$ and $\mathcal{A}_{\text{periph}}$) and kept 1-D respiration indices are drawn independently per sample (R2, R7).
7. **1-D placement policy.** Kept respiration indices are drawn either uniformly without replacement (`scattered`) or as contiguous temporal blocks (`span` of length `physio_span_tokens`). The choice is fixed per run, recorded in the run manifest, and held identical between training and evaluation unless placement is the variable under study. *(Extended by [`ADR-017`](ADR-017-masked-span-physio-placement.md), which adds a third value, `masked_span`, bounding the MASKED runs instead of the kept ones. The rest of this contract — one choice per run, recorded in `masking.json`, matched at evaluation — still holds.)*
8. **Conditioned reconstruction under complete sensor loss.** On every Mode-1 step ($K_{\text{resp}} = 0$), the respiration decoder receives no visible tokens of its own and must condition on the shared encoder latents $z$ via exactly one declared mechanism: (a) **decoder prefixing** of the full encoder latent sequence prior to decoder self-attention, or (b) **cross-attention** from decoder queries to $z$. Checkable invariant: $\partial \mathcal{L}_{\text{resp}} / \partial z \neq 0$ when $K_{\text{resp}} = 0$ (R5).
9. **Batch uniformity and cross-rank determinism.** Step counts are a pure deterministic function of `(seed, epoch, step)` and are identical on all distributed ranks without collective communication; per-sample placements differ across samples and ranks (R7).
10. **Loss-key stability.** Both streams (`tir` and `resp`) emit a scalar masked-reconstruction loss entry on every step and every rank (emitting an explicit zero if a stream ever has zero masked tokens), preventing distributed synchronization hangs across modes.

### 2.5 Normative apportionment rules & algorithm

Parameters: $N = 200$, $T = 25$, $C_p = \lfloor c_p \cdot 250 \rfloor = 100$, $S_{\text{core}}^{\min} = 1$, $S_{\text{core}}^{\max} = 2$, schedule $p(e)$.

**Mode 1 (Pure Virtual Sensing, probability $p(e)$):**
- $K_{\text{resp}} = 0$.
- $S_{\text{tir}} = N / T = 8$ tubes $\implies K_{\text{tir}} = 200$ tokens ($77.78\%$ masked).

**Mode 2 (Waveform Inpainting, probability $1 - p(e)$):**
1. Draw raw physiological count $\tilde{K} \sim \text{Uniform}\{1, 2, \dots, C_p\}$.
2. Compute tube-divisibility remainder $r = (N - \tilde{K}) \bmod T$.
3. Align to an exact tube-compatible count within $[T, C_p]$:
   $$K_{\text{resp}} = \begin{cases} \tilde{K} + r & \text{if } \tilde{K} + r \le C_p \\ \tilde{K} - (T - r) & \text{otherwise} \end{cases}$$
   with the guard that if $K_{\text{resp}} = 0$ (possible only when $C_p < T$, excluded by §2.6), it clamps to $T$.
4. Assign $S_{\text{tir}} = (N - K_{\text{resp}}) / T$ tubes and $K_{\text{tir}} = S_{\text{tir}} \cdot T$.

**Exact Mode-2 distribution property.** Under the shipped parameters ($N = 200$, $T = 25$, $C_p = 100 = 4T$), $\tilde{K} + r \le 100$ holds for all $\tilde{K} \in \{1, \dots, 100\}$. Every value in $\{25, 50, 75, 100\}$ receives exactly 25 of the 100 draws ($25\%$ probability each), giving an **exact discrete uniform distribution** with $E[K_{\text{resp}} \mid \text{Mode 2}] = 62.5$, zero Mode-2-to-Mode-1 leakage, and $S_{\text{tir}} \in \{7, 6, 5, 4\}$ tubes ($K_{\text{tir}} \in \{175, 150, 125, 100\}$ tokens).

**Language-agnostic reference algorithm** (contains no hardcoded literal outside §2.10):

```text
# Step-level count draw (shared across batch; seeded by (seed, epoch, step))
function SAMPLE_STEP_COUNTS(epoch, step, seed):
    rng_step = deterministic_rng(seed, epoch, step)
    p_v = p_warm if (warmup_epochs > 0 and epoch < warmup_epochs) else p_target

    if uniform_real(rng_step, 0, 1) < p_v:
        mode = MODE_1_VIRTUAL_SENSING
        K_resp = 0
        S_tir = N / T
    else:
        mode = MODE_2_INPAINTING
        K_tilde = uniform_integer(rng_step, 1, C_p)      # inclusive
        rem = (N - K_tilde) mod T
        if K_tilde + rem <= C_p:
            K_resp = K_tilde + rem
        else:
            K_resp = K_tilde - (T - rem)
        S_tir = (N - K_resp) / T

    K_tir = S_tir * T
    assert K_tir + K_resp == N

    if spatial_sampling == "anatomy_aware":
        S_core = uniform_integer(rng_step, S_core_min, S_core_max)
        S_periph = S_tir - S_core
    else:                                                 # "uniform" ablation
        S_core = 0
        S_periph = S_tir

    return (mode, K_tir, K_resp, S_tir, S_core, S_periph)

# Sample-level placement draw (independent per sample b in 0..B-1)
function SAMPLE_ITEM_PLACEMENT(counts, rng_sample):
    if spatial_sampling == "anatomy_aware":
        kept_core   = sample_without_replacement(A_core,   counts.S_core,   rng_sample)
        kept_periph = sample_without_replacement(A_periph, counts.S_periph, rng_sample)
        kept_tir_tubes = sort(kept_core U kept_periph)
    else:
        kept_tir_tubes = sort(sample_without_replacement(A_core U A_periph, counts.S_tir, rng_sample))

    kept_resp_tokens = place_1d(units_resp, counts.K_resp, physio_placement, physio_span_tokens, rng_sample)
    return (kept_tir_tubes, kept_resp_tokens)
```

### 2.6 Feasibility preconditions & mathematical proofs

Every precondition below must be validated before training step 0; any infeasible configuration must fail loudly at startup:

1. **Exact budget divisibility:** $N \bmod T = 0$. *(Check: $200 \bmod 25 = 0$ ✓)*
2. **Non-trivial inpainting cap:** $T \le C_p < N$. *(Check: $25 \le 100 < 200$ ✓)*
3. **Visual grid capacity:** $S_{\text{tir}}^{\max} = N / T \le \text{units}_v$. *(Check: $8 \le 36$ ✓)*
4. **High-masking ceiling:** $S_{\text{tir}}^{\max} / \text{units}_v \le c_v^{\max}$ (visual kept fraction remains in the high-masking regime). *(Check: $8 / 36 = 22.22\% \le 25\%$ ✓)*
5. **Stratified feasibility at minimum visual allocation:** Let $S_{\text{tir}}^{\min} = \lfloor (N - C_p) / T \rfloor$. Require $1 \le S_{\text{core}}^{\min} \le S_{\text{core}}^{\max} \le |\mathcal{A}_{\text{core}}|$, $S_{\text{tir}}^{\min} \ge S_{\text{core}}^{\max} + m_{\text{periph}}$ (with peripheral floor $m_{\text{periph}} \ge 1$), and $|\mathcal{A}_{\text{periph}}| \ge S_{\text{tir}}^{\max} - S_{\text{core}}^{\min}$. *(Check: $S_{\text{tir}}^{\min} = 4 \ge 2 + 1 = 3$; $2 \le 4$; $32 \ge 8 - 1 = 7$ ✓)*
6. **Disjoint partition coverage:** $\mathcal{A}_{\text{core}} \cap \mathcal{A}_{\text{periph}} = \emptyset$ and $|\mathcal{A}_{\text{core}}| + |\mathcal{A}_{\text{periph}}| = \text{units}_v$. *(Check: $4 + 32 = 36$ ✓)*
7. **Space-time alignment:** $R = \text{units}_{\text{resp}} / T \in \mathbb{Z}_{>0}$ and $R \cdot \Delta t_{\text{resp}} = \Delta t_{\text{tubelet}}$. *(Check: $250 / 25 = 10$; $10 \times 32\text{ ms} = 320\text{ ms}$ ✓)*
8. **Schedule ordering:** $0 \le p_{\text{warm}} \le p_{\text{target}} \le 1$ whenever $E_{\text{warm}} > 0$, and $0 \le E_{\text{warm}} \le E_{\text{total}}$. *(Check: $0 \le 0.10 \le 0.50 \le 1$ ✓)*

**Budget invariance proof.**
- In Mode 1: $K_{\text{tir}} + K_{\text{resp}} = (N / T) \cdot T + 0 = N$ (by Precondition 1).
- In Mode 2: $r = (N - \tilde{K}) \bmod T \implies (N - (\tilde{K} + r)) \equiv 0 \pmod T$ and $(N - (\tilde{K} - (T - r))) \equiv 0 \pmod T$. Since $T \le C_p$, $K_{\text{resp}} \in [T, C_p]$ and $T \mid (N - K_{\text{resp}})$, hence $K_{\text{tir}} + K_{\text{resp}} = \frac{N - K_{\text{resp}}}{T}\cdot T + K_{\text{resp}} = N$. $\blacksquare$

### 2.7 Two-phase virtual-sensing curriculum

Over the **0-based** epoch index $e \in \{0, 1, \dots, E_{\text{total}} - 1\}$, with $E_{\text{warm}}$ leading warm-up epochs:

$$p(e) = \begin{cases} p_{\text{warm}} = 0.10 & \text{if } e < E_{\text{warm}} \quad (\text{Phase I: Multimodal Grounding}) \\ p_{\text{target}} = 0.50 & \text{if } e \ge E_{\text{warm}} \quad (\text{Phase II: Virtual-Sensing Pre-Training}) \end{cases}$$

| Phase | 0-Based Epochs (default $E_{\text{warm}} = 20$) | Mode 1 (Virtual Sensing) | Mode 2 (Inpainting) | Intended Role |
| :--- | :--- | :---: | :---: | :--- |
| **Phase I — Grounding** | $e \in \{0, \dots, 19\}$ ($20$ epochs) | $10\%$ | $90\%$ | Surviving respiration tokens provide phase anchors while visual latents adapt from generic video priors to facial thermal plumes. |
| **Phase II — Virtual Sensing** | $e \ge 20$ | $50\%$ | $50\%$ | Forces the shared encoder and conditioned respiration decoder to infer oscillatory respiratory dynamics solely from `tir` latents. |

**Validation observability rule.** Validation masking replays a fixed-seed sequence evaluated at the current epoch's $p(e)$. Consequently, validation loss is a **within-phase** observable and exhibits an expected discontinuity at $e = E_{\text{warm}}$; model selection and learning curves must not pool validation losses across the phase boundary.

### 2.8 Initialization & tokenizer-compatibility contract (I1–I4)

- **I1 — Verbatim 3-D kernel.** The visual tokenizer's convolutional patch kernel must match the pretrained VideoMAE kernel $(P_t, P_h, P_w) = (2, 16, 16)$ over $C = 3$ input channels (`[D, 3, 2, 16, 16]`). Because `tir` is natively a 3-channel false-colour rendering, the kernel transfers bit-for-bit with no 2-D inflation and no artificial grayscale-to-RGB channel replication.
- **I2 — Sole visual recipient (zero auxiliary tokenizer debt).** Because `tir` is the sole visual stream, $100\%$ of visual tokens pass through the pretrained 3-D tokenizer. No visual adapter starts from random initialization.
- **I3 — Matched encoder geometry.** The shared encoder uses the ViT-Base geometry ($D = 768$, $12$ blocks, $12$ attention heads) matching published VideoMAE ViT-Base checkpoints. Positional embeddings are re-initialized from a deterministic 3-D space-time sinusoidal prior (with the 1-D respiration stream using the aligned temporal sinusoidal prior).
- **I4 — Honest random-init labelling.** If an ablation deliberately runs without pretrained weights (e.g., for smaller backbones), it must be explicitly recorded as random-initialized and cannot be cited as a test of I1–I3.

### 2.9 Observability contract

Any run executing this ADR must record:

- **Per step:** epoch, step, active $p(e)$, realized mode (Mode 1 vs Mode 2), $K_{\text{tir}}$, $K_{\text{resp}}$, $S_{\text{tir}}$, $S_{\text{core}}$, $S_{\text{periph}}$, encoder sequence length ($= 200$), and per-stream masked reconstruction losses (`tir`, `resp`).
- **Per epoch:** realized Mode-1 / Mode-2 step fractions, mean kept counts per stream, and mean $S_{\text{core}}$ — verified against the theoretical expectations of $p(e)$ and §2.5.
- **Per run (manifest):** full resolved parameter tuple of §2.10, core patch coordinate set $\mathcal{A}_{\text{core}}$, 1-D placement policy, decoder conditioning mechanism (prefixing vs cross-attention), and initialization provenance (I1–I4).

### 2.10 Parameter surface

These logical parameters completely specify the policy. Any implementation must expose them without hidden hardcoded constants and record their resolved values in the run manifest:

| Logical Parameter | Symbol | Type | Default | Admissible Values | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `streams` | $\mathcal{M}$ | set[enum] | `{tir, resp}` | exactly `{tir, resp}` | Active two-stream modality set |
| `n_budget` | $N$ | int | `200` | satisfies §2.6 | Constant visible tokens entering the shared encoder |
| `tokens_per_unit` | $T$ | int | `25` | derived from geometry | Temporal tokens per visual spatial tube |
| `physio_cap_ratio` | $c_p$ | float | `0.40` | $T / \text{units}_{\text{resp}} \le c_p < N / \text{units}_{\text{resp}}$ | Maximum kept fraction of respiration tokens ($C_p = 100$) |
| `spatial_sampling` | — | enum | `anatomy_aware` | `anatomy_aware`, `uniform` | Stratified core+periphery sampling vs uniform ablation |
| `core_row_bounds` | — | tuple[int, int] | `(2, 3)` | inclusive within `[0, G_h - 1]` | Row span of $\mathcal{A}_{\text{core}}$ on the spatial patch grid |
| `core_col_bounds` | — | tuple[int, int] | `(2, 3)` | inclusive within `[0, G_w - 1]` | Column span of $\mathcal{A}_{\text{core}}$ on the spatial patch grid |
| `core_tubes_min` | $S_{\text{core}}^{\min}$ | int | `1` | $\ge 1$, $\le S_{\text{core}}^{\max}$ | Minimum visible tubes drawn from $\mathcal{A}_{\text{core}}$ |
| `core_tubes_max` | $S_{\text{core}}^{\max}$ | int | `2` | $\le |\mathcal{A}_{\text{core}}|$, satisfies §2.6(5) | Maximum visible tubes drawn from $\mathcal{A}_{\text{core}}$ |
| `p_virtual_sensing` | $p_{\text{target}}$ | float | `0.50` | `[0.0, 1.0]` | Mode-1 probability in Phase II ($e \ge E_{\text{warm}}$) |
| `virtual_sensing_warmup_p` | $p_{\text{warm}}$ | float | `0.10` | `[0.0, p_target]` | Mode-1 probability in Phase I ($e < E_{\text{warm}}$) |
| `virtual_sensing_warmup_epochs` | $E_{\text{warm}}$ | int | `20` | `[0, epochs]` | Leading 0-based epochs in Phase I (`0` disables warm-up) |
| `physio_placement` | — | enum | `scattered` | `scattered`, `span` (plus `masked_span` from `ADR-017`) | 1-D kept-token placement strategy |
| `physio_span_tokens` | — | int | `0` | $> 0$ iff `span`, else `0` | Contiguous block length for `span` placement |
| `physio_gap_min_tokens` / `physio_gap_max_tokens` | — | int | `0` | `>= 1` iff `masked_span`, else `0` | See `ADR-017`: shortest / longest MASKED run |
| `spatial_patches` | $\text{units}_v$ | int | `36` | must equal $G_h \cdot G_w$ | Construction-time geometry assertion (not tuned) |
| `physio_tokens` | $\text{units}_{\text{resp}}$ | int | `250` | must equal $R \cdot T$ | Construction-time geometry assertion (not tuned) |

Fixed geometry assumed by the policy (not tuned here): `clip_duration = 8.0 s`, `fps = 25`, `temporal_stride = 4`, `tubelet = (2, 16, 16)`, `input_size = 96`, `patch_size = 16`, `signal_token_duration = 32 ms` ($250$ tokens).

### 2.11 Rejected alternatives

- **Retaining a 350-token budget for two streams.** Rejected: forces $14 / 36 = 38.9\%$ visual retention ($61.1\%$ masked) in Mode 1, weakening the masked video pretext task and violating R2.
- **Unconstrained uniform spatial tube sampling as the default.** Rejected as default (retained only as a controlled ablation): misses the 4-patch nasal core on $34.8\%\text{--}61.0\%$ of samples depending on $S_{\text{tir}}$, violating R3.
- **Continuing three- or four-stream pre-training before validating `tir` $\to$ `resp`.** Rejected for Phase 1: mixes unresolved auxiliary visual tokenizer debt and multi-stream gradient dilution with the core thermal-respiration question, violating R1 and I2.
- **Hard-pinning $p(e) = 0.50$ from epoch 0.** Rejected: removes the phase-anchoring warm-up needed to prevent early virtual-sensing collapse, violating R7.
- **Batch-shared spatial tube indices.** Rejected: sharing kept indices across all samples in a mini-batch reduces effective spatial mask diversity to one mask per step; only token *counts* are shared per step (Contract 6).
- **Dynamic/padded encoder lengths.** Rejected: breaks dense rectangular `[B, 200, D]` execution and violates R2.

---

## 3. Falsifiable Criteria

### 3.1 Criterion 1 — Policy & mathematical conformance (pre-training gate; zero GPU hours required)

All seven checks must pass before launching any full training run:

1. **Exact budget & shape invariance:** Over $\ge 10^5$ synthetic sampled steps across both curriculum phases, $K_{\text{tir}} + K_{\text{resp}} = 200$ on $100\%$ of steps and the encoder input tensor shape is strictly `[B, 200, 768]` with zero assertion failures.
2. **Exact mode & bound compliance:** On $100\%$ of Mode-1 steps, $K_{\text{resp}} = 0$ and $S_{\text{tir}} = 8$ ($K_{\text{tir}} = 200$). On $100\%$ of Mode-2 steps, $K_{\text{resp}} \in \{25, 50, 75, 100\}$ (each occurring at $25\% \pm 1\%$ over $10^5$ Mode-2 draws) and $S_{\text{tir}} \in \{7, 6, 5, 4\}$.
3. **Guaranteed anatomical stratification:** Across $\ge 10^5$ drawn sample masks, every sample contains $S_{\text{core}} \in \{1, 2\}$ distinct indices from $\mathcal{A}_{\text{core}}$ ($P(\text{core hit}) = 1.000$) and $S_{\text{periph}} = S_{\text{tir}} - S_{\text{core}}$ distinct indices from $\mathcal{A}_{\text{periph}}$, with zero duplicates and zero cross-stratum leakage.
4. **Curriculum schedule fidelity:** Over repeated simulated epochs, realized Mode-1 share matches $0.10$ for $e \in \{0, \dots, 19\}$ and $0.50$ for $e \ge 20$ within a binomial $99\%$ confidence interval.
5. **Determinism & per-sample diversity:** Replaying `(seed, epoch, step)` reproduces identical counts across independent processes/ranks, while placements within a mini-batch differ across samples.
6. **Zero-sensor gradient path:** On a synthetic Mode-1 forward/backward pass ($K_{\text{resp}} = 0$), $\|\partial \mathcal{L}_{\text{resp}} / \partial z\| > 0$ strictly (Contract 8).
7. **Feasibility rejection:** Each violated precondition in §2.6 triggers an immediate startup validation error before step 0.

### 3.2 Criterion 2 — Scientific claim (respiration evaluation protocol)

All scientific claims follow `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md` and `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md`:

- **Split & aggregation:** Subject-disjoint 5-fold cross-validation ($70\%$ train / $10\%$ val / $20\%$ test by subject per fold; model selection strictly on `val`, one-shot evaluation on `test`). Clip predictions are assembled into continuous per-session waveforms via Hann overlap-add and scored against the shared respiration-cache reference.
- **Headline metric (gating):** Band-limited Pearson correlation ($0.1\text{--}0.6\text{ Hz}$), aggregated per session $\to$ median per subject $\to$ pooled median across held-out subjects with bootstrap $95\%$ CI. Secondary diagnostic metric: breath-rate MAE (`bpm`). Amplitude metrics are excluded from gating by protocol.
- **Tier 1 — Go/No-Go gate (relative, Stage-3 downstream):**
  - *Primary comparison (pre-training utility):* After identical Stage-3 TIR-only fine-tuning on the same 5 folds, an ADR-007 pre-trained encoder must exceed both (a) a matched **uniform-sampling** two-stream comparator (`spatial_sampling = uniform`, same 200-token budget and schedule) and (b) the matched multi-stream comparator (ADR-005/006) on the pooled subject-level band-limited Pearson median by **$\ge +0.02$**, with bootstrap $95\%$ CIs reported.
- **Tier 2 — Reference benchmarks & thesis targets (reported, non-gating):**
  - *Verified repository reference points:* Pooled 5-fold Stage-3 TIR-ROI baseline median **$+0.542$** ($[+0.493, +0.576]$) and classical template baseline **$+0.537$** (`code/notes/eval/archive/tir_resp/resp_5fold_cv_96respgrid.md`).
  - *Non-gating Stage-2 zero-shot pretext diagnostics (fold 0 held-out test, $K_{\text{resp}} = 0$, $K_{\text{tir}} = 200$):*
    1. **Conditional-mean-trap check:** Predicted waveform dispersion $\sigma(\hat{y}) \ge 0.50 \cdot \sigma(y_{\text{ref}})$ on held-out test clips, verifying non-collapsed oscillatory output rather than flat clip-mean prediction.
    2. **Pretext correlation & rate progression:** Report subject-level zero-shot correlation and breath-rate MAE against the fold-0 ADR-005 zero-shot reference ($+0.1129$ $[+0.0775, +0.1363]$, `code/notes/eval/baselines/tir_rgb2d_resp/README.md`). Pre-registered stretch target for zero-shot Stage-2 tracking: subject-level $r \ge +0.40$ and rate $\text{MAE} \le 2.0\text{ bpm}$ (explicitly non-gating for Stage-3 adoption, per evaluation protocol).

### 3.3 Criterion 3 — Smoke execution & resource record

- **Smoke run:** A single-device capped smoke run (1 epoch, warm-up disabled, corpus capped at $\le 100$ clips, small batch size) must complete without assertion errors or `NaN` losses, verify the `[B, 200, 768]` encoder shape, and emit every observability field of §2.9 for both Mode 1 and Mode 2.
- **Resource threshold:** `[TODO: verify peak allocated/reserved VRAM, batch size, device model, and precision mode on first 200-token ViT-Base execution]`. Reducing the encoder sequence length from $350$ to $200$ tokens ($-42.9\%$) reduces quadratic self-attention compute by approximately $(200/350)^2 \approx 0.33\times$ (a $\sim 67\%$ reduction in encoder attention FLOPs per step), which must be confirmed empirically rather than assumed.

### 3.4 Acceptance artefacts

- **Per fold:** Immutable fold partition record (with subject-disjoint digest), validation selection record, per-epoch training/validation metric log, and run masking audit manifest (§2.9).
- **Per experiment:** Pooled 5-fold evaluation summary JSON containing per-subject metrics, pooled median band-limited Pearson with bootstrap $95\%$ CI, breath-rate MAE, and paired fold-level deltas against the matched comparators.

---

## 4. Consequences & Trade-offs

- **Positive:**
  - **Eliminates multi-stream confounders:** Every visual token and gradient comes from `tir` through a verbatim pretrained 3-D tokenizer, making Phase-1 outcomes cleanly attributable.
  - **Eliminates the nasal-plume blindspot:** Guarantees $100\%$ core-region visibility per sample instead of dropping the thermal plume on $35\%\text{--}61\%$ of samples.
  - **Restores VideoMAE-regime masking difficulty & efficiency:** $77.8\%$ Mode-1 visual masking suppresses trivial spatiotemporal interpolation while cutting encoder sequence length from $350$ to $200$ tokens (substantially reducing memory and attention FLOPs).
  - **Clean Mode-2 arithmetic:** Under $C_p = 100 = 4T$, discrete remainder alignment produces an exact uniform distribution over $\{25, 50, 75, 100\}$ with zero Mode-2-to-Mode-1 leakage.
- **Negative / Technical Debt:**
  - **Static geometric proxy for anatomy:** Defining $\mathcal{A}_{\text{core}}$ as the fixed central $2 \times 2$ subgrid (`rows 2..3, cols 2..3`) assumes the ROI crop reliably places the nostrils near the ROI centre and that subject sway remains bounded. If the mouth-nose crop shifts nostrils into upper rows (e.g., row 1), the fixed subgrid could partially miss the true airflow plume. Mitigation: keep `core_row_bounds` and `core_col_bounds` configurable and verify empirical core-patch overlap against landmark heatmaps before scaling to 5 folds.
  - **Potential spatial-attention shortcutting:** Always supplying 1–2 central tubes could bias the encoder toward over-relying on central patch positional embeddings. Mitigation: retain $2\text{--}3$ masked core tubes per sample as reconstruction targets and benchmark directly against the `spatial_sampling = uniform` ablation (§3.2).
  - **Deferral of visible-light fusion (`rgb2d`) and pulse (`bp`):** Phase 1 cannot answer whether multimodal visible+TIR fusion improves robustness under thermal occlusion or cross-modality transfer; those questions require a subsequent Phase-2 decision once Phase 1 clears its gate.
  - **Non-stationary validation curve at $e = E_{\text{warm}}$:** Switching $p(e)$ from $0.10$ to $0.50$ introduces an intentional jump in validation loss at epoch 20; automated checkpoint selection must respect phase boundaries.
- **Rollback / Rejection Condition:** Reject or revert this decision if: (i) Criterion 1 fails any invariant or gradient check; (ii) empirical landmark-to-grid verification shows the nostrils systematically fall outside $\mathcal{A}_{\text{core}}$ (requiring re-centering of `core_row_bounds`/`core_col_bounds` before training); (iii) Stage-2 virtual sensing collapses to the conditional mean ($\sigma(\hat{y}) < 0.50 \cdot \sigma(y_{\text{ref}})$) despite passing the gradient check; or (iv) Tier-1 downstream 5-fold evaluation fails to beat the uniform-sampling comparator or matched baseline by the pre-registered margin, in which case anatomy-aware stratification is disabled (`spatial_sampling = uniform`) or prior policies are revisited.

---

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A` — proposed decision; not yet executed.
- **Evidence Artifacts**:
  - `code/notes/eval/baselines/tir_rgb2d_resp/README.md` (measured fold-0 three-stream baseline and input-ablation evidence motivating single-stream focus).
  - `code/notes/eval/archive/tir_resp/resp_5fold_cv_96respgrid.md` (measured 5-fold Stage-3 TIR-ROI respiration reference $+0.542$ and template baseline $+0.537$).
  - `[TODO: link Phase-1 fold-0 and 5-fold evaluation artefacts once executed]`.
- **Tests / Smoke Checks**:
  - Policy unit verification covering all seven Criterion-1 checks (§3.1): exact 200-token sum over $10^5$ steps, Mode-2 uniformity, core-vs-periphery stratification invariants, curriculum schedule transitions, cross-rank determinism vs per-sample placement diversity, zero-sensor gradient flow, and infeasible-tuple rejection.
  - Single-device capped smoke run (§3.3).
  - `[TODO: record exact verification suite results and peak VRAM measurements after implementation]`.
- **Rollback / Rejection Condition**: Defined in §4.
- **Deferred, non-blocking decisions:** (1) Empirical landmark-heatmap audit confirming whether `core_row_bounds = (2, 3)` vs `(1, 2)` maximizes nostril-plume capture on the ROI crop; (2) choice of `E_warm = 20` vs `E_warm = 50` if total epoch budgets vary across experiments; (3) Phase-2 re-introduction of `rgb2d` once `tir` $\to$ `resp` passes Criterion 2.
