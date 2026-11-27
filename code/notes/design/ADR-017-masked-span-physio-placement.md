# ADR-017: Gap-Controlled Masked-Span Placement for the 1-D Respiration Stream (`masked_span`)

- **Status**: Proposed
- **Date**: 2026-11-27
- **Author**: KISMED MA research team
- **Related Spec/Plan**: [`ADR-017-masked-span-physio-placement-plan.md`](../plan/features/ADR-017-masked-span-physio-placement-plan.md) (build order and test gates); [`ADR-007`](ADR-007-two-stream-anatomy-aware-constant-budget-masking.md) (counts, modes, curriculum — unchanged); [`ADR-008`](ADR-008-composite-objective-and-cross-modal-alignment.md) (Stage-2 objective — unchanged); [`ADR-009`](ADR-009-resp-waveform-evaluation-protocol.md) (headline metric and subject-level aggregation); [`ADR-001`](ADR-001-subject-disjoint-split-and-selection.md) (folds and selection); [`GLOSSARY.md`](../GLOSSARY.md).
- **Supersedes**: In part, `ADR-007` §2.4 contract 7 and the two placement rows of §2.10 (the 1-D placement policy is now governed here; counts, modes and every other ADR-007 rule stay normative). Reactivates the span-masking idea of the archived note [`archive/SpanMask_PhysioSignals.md`](archive/SpanMask_PhysioSignals.md), re-expressed on top of the constant-budget policy.

> **Abstraction boundary.** This ADR specifies **which** respiration tokens a Mode-2 step hides and
> **why**. It does not change **how many** tokens are kept (ADR-007), the loss (ADR-008), or the
> evaluation metric (ADR-009). Target modules, YAML/CLI wiring and test gates live in the paired plan.

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope

ADR-007 fixes, per step, how many respiration tokens are kept: Mode 1 keeps $K_\text{resp} = 0$;
Mode 2 keeps $K_\text{resp} \in \{25, 50, 75, 100\}$ of $N_s = 250$ tokens (32 ms each, 8 s clip).
It leaves the **placement** of those tokens to a run setting (`scattered` or `span`). This ADR
governs that placement for Mode-2 steps. The goal is that **every masked respiration gap is long
enough that it cannot be filled from the visible respiration samples beside it**, so the encoder has
to supply the missing waveform from `tir`.

**In scope.** The 1-D placement rule for the respiration stream; the gap-length parameters; their
feasibility against ADR-007's admissible Mode-2 counts; how placement is matched between training
and the Stage-2 inpainting probe; and the criteria for adopting the rule.

**Out of scope.**

- Kept counts, modes, the visual stratification and the curriculum: these are ADR-007's.
- The loss terms, including any mask-weighted spectral loss: these are ADR-008's.
- Stage-3 recipe (ADR-015) and the evaluation protocol (ADR-009).
- Time-aligned gaps across several 1-D streams. The rule is defined on one token axis: the $N_s$
  respiration tokens of the single physiological stream the model declares.

### 1.2 Motivation / Design Pressures

**P1 — Scattered Mode-2 inpainting is solvable by interpolation (measured, current geometry).**
On the fold-0 ADR-008 two-stream checkpoint, the scattered inpainting allocations reach a
subject-median `r_masked` of **+0.9989** (`tir`/`resp` = 100/100) and **+0.9986** (125/75). With
respiration hidden (Mode 1), the same model reaches only **+0.3256 [+0.2444, +0.4061]**
([`multimae_style_panel.md`](../eval/baselines/tir_resp/multimae_style_panel.md) §4). The
inpainting figure sits at the local-interpolation ceiling and says nothing about sensing. Under the
ADR-008 arm, Mode 2 makes up **90 %** of Phase-I steps and **50 %** of Phase-II steps
(`p_warm = 0.10`, `p_target = 0.50`, `E_warm = 20`). Those steps therefore spend most of their
respiration loss on a task the visible respiration samples already solve.

**P2 — The geometry alone guarantees the shortcut (computed from the placement code, no data).**
`core.mask_placement.place_signal` was run at ADR-007 geometry ($N_s = 250$, 32 ms tokens; 400
samples per row, seed 0, 2026-11-27):

| Placement | $K_\text{resp}$ | Median masked gap | Masked gaps < 1 s | Median longest gap per clip |
| :--- | ---: | ---: | ---: | ---: |
| `scattered` | 25 / 50 / 75 / 100 | 0.22 / 0.13 / 0.06 / 0.06 s | 96.8 / 99.9 / 100 / 100 % | 1.06 / 0.61 / 0.38 / 0.29 s |
| `span`, kept block 25 tokens | 25 / 50 / 75 / 100 | 3.58 / 2.11 / 1.41 / 0.96 s | 7.6 / 19.9 / 33.0 / 52.8 % | 4.83 / 3.26 / 2.24 / 1.60 s |
| `span`, kept block 125 tokens (4 s) | 25 / 50 / 75 / 100 | 3.58 / 3.20 / 2.78 / 2.40 s | 7.6 / 8.6 / 11.2 / 13.4 % | 4.83 / 4.29 / 3.74 / 3.20 s |

At the respiratory passband of 0.10–0.60 Hz (one breath every 1.7–10 s), a scattered gap of
≤ 0.3 s is a small fraction of a breath cycle. The existing `span` mode does not fix this.
`physio_span_tokens` sets the length of the **kept** blocks, so the masked gaps are whatever slack is
left over. Depending on $K_\text{resp}$, 8–53 % of the gaps are still shorter than 1 s.

**P3 — Gap length, not masked fraction, is what removes the shortcut (measured, earlier geometry).**
In the archived note (100 Hz, 160 ms tokens, 50 % masked), a no-video linear interpolator gave
these masked MSE values (z²), compared with a constant predictor at about 1.0:

| Gap pattern | Masked MSE (z²) | Beats constant? |
| :--- | ---: | :--- |
| scattered | 0.13 | yes |
| 4 × 1 s | 0.84 | yes |
| 2 × 2 s | 1.37 | no |
| 1 × 4 s | 1.46 | no |

Interpolation stops beating the constant once **individual gaps** reach roughly 0.5–1 breath cycle
([`archive/SpanMask_PhysioSignals.md`](archive/SpanMask_PhysioSignals.md) §4.2). These floors are
geometry-specific and must be re-measured at the current geometry (Criterion 3a).

**P4 — The probe already shows the cost (measured, predecessor arm).** On the 96 px three-stream
ViT-B checkpoint, trained with scattered placement, the scattered 50 % probe gave `r_masked`
**+0.9975**. A single contiguous 4 s hidden block at the same budget gave **+0.5419**
([`resp_recon_zeroshot.md`](../eval/dataset_inspection/rgb2d/resp_recon_zeroshot.md)). A model
trained only on interpolatable gaps loses most of its inpainting skill once the gaps are long.

**What the archived note asked for that ADR-007 now covers.** Its "ramp to 100 % masked" concern —
a fully hidden stream has no gradient path to the encoder — is resolved by ADR-007 Mode 1 together
with contract 8 (decoder prefixing, $\partial\mathcal{L}_\text{resp}/\partial z \neq 0$). This ADR
therefore controls only **Mode-2 gap geometry**. It adds no masking-ratio ramp.

### 1.3 Requirements

- **R1 — Bounded gaps.** On every Mode-2 sample, every maximal run of masked respiration tokens has
  a length in $[g_\text{min}, g_\text{max}]$ tokens.
- **R2 — Counts untouched.** The kept count $K_\text{resp}$ is exactly the ADR-007 step count,
  identical across the batch (ADR-007 contract 6). Encoder input stays `[B, 200, D]`.
- **R3 — Per-sample placement, deterministic replay.** Gap structure varies per sample. It is a pure
  function of the placement generator, `(seed, epoch, step, rank)`.
- **R4 — Mode 1 untouched.** $K_\text{resp} = 0$ consumes **no** placement random numbers. This keeps
  the paired Mode-1 and remote-sensing evaluations (same masked positions across arms) valid.
- **R5 — Neutral default.** With the default placement (`scattered`), masks, generator stream and
  losses are bit-identical to the pre-ADR code. The legacy kept-block `span` stays available and
  unchanged, so archived probe numbers remain reproducible.
- **R6 — Fail at startup, not mid-run.** If the gap bounds cannot tile any admissible Mode-2 masked
  count, model construction is refused. The Stage-2 probe's pinned kept count is not restricted to
  $\mathcal{K}_2$, so a pinned count the bounds cannot tile is refused as well, naming the feasible
  counts — R7's caption honesty covers a *different* placement, not an infeasible one.
- **R7 — Matched probe.** The Stage-2 inpainting probe can reproduce the training gap rule from the
  checkpoint record. A probe whose placement differs from training must say so in its report caption.

---

## 2. Decision & Parameter Exposure

**Add a third 1-D placement, `masked_span`. On every Mode-2 step it hides the $M = N_s - K_\text{resp}$
masked respiration tokens as $n$ contiguous gaps, each between $g_\text{min}$ and $g_\text{max}$
tokens long, separated by kept blocks of at least one token. Kept counts, modes and Mode 1 stay
exactly as ADR-007 defines them. A matched two-stream ADR-008 arm that differs only in placement is
the experiment that decides adoption (§3). `scattered` stays the default until Criterion 3c passes.**

### 2.1 Normative Definitions & Mathematical Specification

**Notation.**

| Symbol | Meaning |
| :--- | :--- |
| $N_s$ | number of respiration tokens (250 at ADR-007 geometry) |
| $K$ | the step's $K_\text{resp}$ |
| $M = N_s - K$ | number of masked tokens |
| $\mathcal{K}_2$ | the set of Mode-2 kept counts the active policy can draw (ADR-007 §2.5: $\{25, 50, 75, 100\}$) |
| $\mathcal{M}_2 = \{N_s - K : K \in \mathcal{K}_2\}$ | the corresponding masked counts ($\{225, 200, 175, 150\}$) |

$\mathcal{K}_2$ is **enumerated from the apportioner** (`ConstantBudgetApportioner`), never
transcribed into a literal: a Mode-2 count is the tube-modulus snap of a
`Uniform{physio_draw_min..C_p}` draw, so the set moves with `n_budget`, `tokens_per_unit` and
`physio_cap_ratio`. For the ADR-008 arm's tuple it resolves to $\{25, 50, 75, 100\}$.

**Admissible gap counts.** For a masked count $M$ and kept count $K$:

$$\mathcal{N}(M, K) = \{\, n \in \mathbb{Z}_{\ge 1} \;:\; n\,g_\text{min} \le M \le n\,g_\text{max},\;\; n - 1 \le K \,\}.$$

**Per-sample placement algorithm** (all draws from the step's placement generator, on CPU):

1. If $K = 0$ (Mode 1): every token is masked and **no random number is drawn** (R4).
   If $M = 0$ (every token kept — unreachable for $K \in \mathcal{K}_2$, but reachable from a
   policy or pin whose cap is $N_s$, e.g. `physio_cap_ratio: 1.0`): nothing is masked, every index is
   returned, and **no random number is drawn**.
2. Draw $n \sim \mathrm{Uniform}(\mathcal{N}(M, K))$.
3. **Gap lengths.** Set $g_i = g_\text{min} + e_i$ for $i = 1, \dots, n$, where $\sum_i e_i = M - n\,g_\text{min}$
   and $0 \le e_i \le g_\text{max} - g_\text{min}$. Draw the $e_i$ one at a time. With $R$ the
   remainder still to assign and $s$ the number of later gaps, draw
   $e_i \sim \mathrm{Uniform}\{\max(0, R - s\,(g_\text{max} - g_\text{min})), \dots, \min(g_\text{max} - g_\text{min}, R)\}$;
   the last gap takes what is left. Then apply a uniform random permutation to $(g_1, \dots, g_n)$,
   so position in the clip does not bias gap length.
4. **Kept blocks.** Form $n + 1$ kept blocks $b_0, \dots, b_n$:
   - interior blocks $b_j = 1 + f_j$ for $1 \le j \le n - 1$;
   - edge blocks $b_0 = f_0$ and $b_n = f_n$, so a gap may touch a clip edge.

   $(f_0, \dots, f_n)$ is a uniformly random weak composition of $X = K - (n - 1)$ into $n + 1$
   parts, drawn by the stars-and-bars construction: pick a uniform $n$-subset of the $X + n = K + 1$
   bar positions (a random permutation's first $n$ entries, sorted → $s_0 < \dots < s_{n-1}$) and
   set $f_0 = s_0$, $f_j = s_j - s_{j-1} - 1$ for $1 \le j \le n - 1$, $f_n = K - s_{n-1}$. The
   composition must be distribution-uniform: a sequential remainder split — the procedure step 3
   prescribes for the masked side — would bias the kept side, which is the one that must stay
   position-neutral.
5. **Assemble** the token axis as $[b_0\ \text{kept}]\,[g_1\ \text{masked}]\,[b_1\ \text{kept}] \cdots [g_n\ \text{masked}]\,[b_n\ \text{kept}]$
   and return the ascending kept indices, shape `[B, K]`.

**Invariants.**

- **I1 (exact count):** each row of the returned index tensor has exactly $K$ distinct, ascending
  entries in $[0, N_s)$. The batch shares $K$.
- **I2 (gap bounds):** in each row, the maximal masked runs number $n \in \mathcal{N}(M, K)$, and
  each has length in $[g_\text{min}, g_\text{max}]$.
- **I3 (separation):** consecutive gaps are separated by at least one kept token, so gaps never
  merge into a run longer than $g_\text{max}$.
- **I4 (RNG silence on the two degenerate counts):** for $K = 0$ (every token masked) and for
  $M = 0$ (every token kept), the generator state is unchanged after the call.
- **I5 (no encoder leak):** in the forward pass, every gathered visible respiration token has
  `mask == 0`.
- **I6 (feasibility):** the model declares a single physiological stream (`len(signal_streams) == 1`,
  per §1.1), $1 \le g_\text{min} \le g_\text{max} \le N_s$, and
  $\mathcal{N}(M, N_s - M) \neq \emptyset$ for every $M \in \mathcal{M}_2$. This is checked at model
  construction against the counts the apportioner can actually draw.

**Default bounds at ADR-007 geometry** (arm values; the parameter defaults stay neutral):

- $g_\text{min} = 63$ tokens = **2.016 s** (0.2–1.2 cycles of the 0.10–0.60 Hz passband);
  $g_\text{max} = 125$ tokens = **4.0 s** (0.4–2.4 cycles). That spans 0.2–2.4 passband cycles and
  is at least the 2 s gap that broke interpolation in P3.
- Feasibility: $\mathcal{N} = \{2\}$ for $M \in \{150, 175\}$ and $\{2, 3\}$ for $M \in \{200, 225\}$ (I6 ✓).
- Gaps of 2–4 s at 32 ms resolution leave 2 to 3 masked gaps and 1 to 4 kept blocks per clip, against
  roughly $K$ boundaries under `scattered`. The ADR-008 spectral splice therefore sees at most 6
  visible/masked transitions instead of up to ~100
  ([`archive/SpanMask_PhysioSignals.md`](archive/SpanMask_PhysioSignals.md) §5.3).

**Effect on ADR-008 terms.**

- Masked token counts are unchanged, so $M_\text{pts} = n_\text{masked} \cdot k$ is identical.
- $\mathcal{L}_\text{MSE}$ and $\mathcal{L}_\text{Pearson}$ are computed on the same number of
  masked samples.
- The InfoNCE gate ($K_\text{resp} > 0$) fires on exactly the same steps.

The only change is which samples are hidden.

### 2.2 Logical Parameter Surface

| Parameter | Status | Type | Default | Allowed / Constraints | Role |
| --- | --- | --- | --- | --- | --- |
| `physio_placement` | existing (extended enum) | enum | `scattered` | `scattered`, `span`, `masked_span` | 1-D placement. `scattered` is neutral (R5); `span` is the legacy kept-block mode (unchanged); `masked_span` is proposed by this ADR |
| `physio_span_tokens` | existing (unchanged) | int | `0` | `> 0` iff `span`, else `0` | Legacy kept-block length (unchanged meaning) |
| `physio_gap_min_tokens` | **proposed** | int | `0` | `>= 1` iff `masked_span`, else `0`; `<= physio_gap_max_tokens` | $g_\text{min}$: shortest masked gap |
| `physio_gap_max_tokens` | **proposed** | int | `0` | `>= physio_gap_min_tokens` iff `masked_span`, else `0`; `<= N_s` | $g_\text{max}$: longest masked gap |

Rules:

- All four values come from YAML or CLI flags. No gap length is hard-coded (`AGENTS.md` §4).
- Lengths are **token counts**, consistent with the constant-budget surface. The resolved seconds
  ($g \cdot$ `sig_kernel` / `fs`) and the per-$M$ sets $\mathcal{N}$ are printed in the `[mask]`
  banner and written to the run manifest.
- The knobs are exclusive **in both directions**: `physio_span_tokens` must be `0` unless
  `physio_placement` is `span`, and both gap bounds must be `0` unless it is `masked_span`. A stray
  non-zero value is refused rather than ignored.
- The two gap bounds are written to the run manifest **only when non-neutral**, like the ADR-006 and
  ADR-007 keys: the resume guard compares the whole `policy` block, so an unconditional write would
  make every pre-ADR-017 run un-resumable. Manifest entries in `policy` must use JSON-native scalar
  types (and `str(M)` keys with `list[int]` values if $\mathcal{N}(M)$ is stored) so the reload
  comparison on resume is type-invariant.
- `masked_span` requires a single declared physiological stream (`len(signal_streams) == 1`, I6);
  multi-physiology support is deferred (§1.1).
- On resume, a change to any of the four values is refused, as for every other masking key.
- The matched arm config (`stage2_budget_tir_resp_2stream_anatomy_adr017.yaml`, to be added in the
  paired implementation plan) sets `physio_placement: masked_span`, `physio_gap_min_tokens: 63`,
  and `physio_gap_max_tokens: 125`, and is otherwise byte-identical to
  `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`. 63/125 are the
  Criterion-3a hypothesis, not a frozen value: the floor probe sweeps the feasible bounds first
  (paired plan, Phase 5) and the run protocol records the point actually adopted.

### 2.3 Rejected Alternatives

- **Reuse the existing `span` mode.** It bounds kept blocks, not masked gaps. P2 shows 8–53 % of
  masked gaps under 1 s, so the shortcut survives at the counts ADR-007 draws most often.
- **Redefine `span` in place.** That would silently change archived probe results (the `span50`
  figure in `resp_recon_zeroshot.md` and the four-stream `resp_recon_arms_ab.md`). A new enum value
  keeps those results reproducible (R5).
- **A single gap per clip ($n \equiv 1$).** With $M \ge 150$ this needs $g_\text{max} \ge 225$ tokens
  (7.2 s), which is nearly Mode 1. Mode 2 would then lose the "visible context on both sides" case it
  exists to train, and Mode 1 already covers full blackout.
- **Ramp the masked fraction toward 100 % (the archived ladder).** ADR-007 Mode 1 together with
  decoder-prefix conditioning already provides the fully masked case with an encoder gradient.
  Adding more short gaps would not lengthen any gap (P3).
- **Seconds-valued knobs.** These would round differently at each `fs`/`sig_kernel`. Tokens are the
  unit the policy and the checks operate on, and seconds are logged alongside.
- **A gap-length curriculum.** Deferred. It adds a schedule with no evidence behind it. Fixed bounds
  come first; a ramp is a follow-up only if Criterion 3b shows Mode-2 optimisation failure.
- **Mask-weighted or MR-STFT spectral loss in Stage 2.** The objective belongs to ADR-008 and is not
  changed here.

---

## 3. Falsifiable Criteria

- **Criterion 1 (Conformance / Invariants, zero GPU hours).**
  - I1–I4 hold over $\ge 10^4$ seeded draws for every $K \in \mathcal{K}_2 \cup \{0, N_s\}$; the
    $K = N_s$ case is the $M = 0$ state of step 1 and must consume no random number, like $K = 0$.
  - I5 holds in a forward pass on a `masked_span` model.
  - I6 rejects at construction:
    - `masked_span` on a model with more than one declared physiological stream;
    - $g_\text{min} > g_\text{max}$, or $g_\text{max} > N_s$;
    - bounds that cannot tile some $M \in \mathcal{M}_2$ (for example $g_\text{min} = g_\text{max} = 100$ with $M = 175$);
    - nonzero gap bounds with any placement other than `masked_span`;
    - a nonzero `physio_span_tokens` with `masked_span`.
  - Replaying `(seed, epoch, step, rank)` reproduces identical indices; samples within a batch differ.
  - The default `scattered` placement yields bit-identical masks and generator state to the pre-ADR
    code (R5).
  - The legacy `span` mode yields identical indices to the pre-ADR code.
- **Criterion 2 (Smoke / Pipeline).**
  - Command (from `code/`, once the paired plan lands):
    `python -u runners/run_pretrain.py -c configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml --exp_root /tmp/adr017_smoke --fold 0 --dev_smoke --epochs 1 --warmup_epochs 0 --virtual_sensing_warmup_epochs 0 --max_entries 100 --batch_size 8 --num_workers 0 --output_dir /tmp/adr017_smoke/out`
  - Must hold: no `NaN`, encoder `[B, 200, 768]` on every step, both modes observed, and the `[mask]`
    banner plus manifest record `masked_span`, both bounds, their seconds, and $\mathcal{N}(M)$ for
    each $M \in \mathcal{M}_2$.
- **Criterion 3 (Scientific / Resource Gate).**
  - **3a — Shortcut removed (data-only, zero GPU; must pass before any training run).**
    - Setting: fold-0 **validation** role, all $K \in \mathcal{K}_2$, ≥ 5 placement draws per clip.
    - Measure: masked MSE of the **no-video linear interpolator** (which uses only visible
      respiration samples via `np.interp`: linear between visible neighbours, constant-clamped to
      the nearest visible sample whenever a gap touches a clip edge, $b_0 = 0$ or $b_n = 0$) under
      `masked_span`, against the **constant-zero predictor**.
    - Pass: the interpolator is worse than the constant (`mse_interp > mse_const`) for every $K$.
    - Sanity check: the same probe under `scattered` must show `mse_interp ≪ mse_const`.
    - If it fails: widen $g_\text{min}$ (and $g_\text{max}$ if I6 requires it) and repeat. Do not train.
  - **3b — Encoder contributes in Mode 2 (Stage-2 fold-0 `test`, 150 epochs, non-gating).**
    - Under the matched `masked_span` inpainting probe, the model's per-clip `r_masked` must exceed
      the same `np.interp` no-video interpolator's `r_masked` on the **same masks**.
    - Statistics: paired within clip; aggregated per subject as in ADR-009; 95 % bootstrap CI of the
      subject-median difference must be > 0.
    - Also report the Mode-1 (`--resp_probe hidden`) subject-median `r_masked` against the scattered
      ADR-008 arm (+0.3256 [+0.2444, +0.4061]). Flag a drop of more than 0.02.
  - **3c — Adoption gate (five-fold Stage 3, ADR-009 protocol).**
    - Metric: subject-level band-limited Pearson (0.10–0.60 Hz) of Stage-3 models fine-tuned from the
      `masked_span` arm versus the matched scattered ADR-008 arm (same folds, seeds and Stage-3 recipe).
    - **Adopt** `masked_span` as the arm default if the pooled subject-median difference is ≥ +0.02
      and its 95 % CI excludes 0.
    - **Reject** if 3a cannot be met at any feasible bounds, or if the difference is ≤ −0.02.
    - Anything in between keeps `scattered`: no evidence of benefit, so the simpler rule stays.
  - **Resource bound.** Placement adds < 5 ms per step at B = 16 on the host CPU. Model, VRAM and
    tokens per step are unchanged by construction.

- **Acceptance Artifacts.** The Criterion-1 test module; the Criterion-3a floor report (JSON +
  markdown under `code/notes/eval/`); the fold-0 `masked_span` inpainting and Mode-1 reports with
  paired contrasts; the five-fold `kfold_*.json` for both arms.

---

## 4. Consequences & Trade-offs

- **Positive.**
  - Mode 2 — 50–90 % of training steps under the ADR-008 curriculum — becomes an informative
    cross-modal pretext instead of a local interpolation task.
  - The Stage-2 inpainting probe can tell models apart again (P1, P4).
  - The ADR-008 spectral term sees a few splice boundaries instead of dozens.
  - Counts, encoder shape, losses' masked-point arithmetic and Mode 1 are unchanged, so the
    remote-sensing and Mode-1 comparisons stay paired across arms.
  - The legacy modes remain reproducible.
- **Negative / Technical Debt.**
  - A full retrain is required (fold 0, then five folds) for the matched arm.
  - Mode-2 respiration loss will rise, so `loss` / `mse_resp` / `grad_ratio` are comparable only
    within one placement.
  - The ADR-008 gradient-balance band will need re-reading for this arm.
  - Gap bounds are geometry-specific (32 ms tokens, 8 s clips) and must be re-derived through
    Criterion 3a whenever `fs`, `sig_kernel`, `seq_len` or the clip length change.
  - Contiguous visible blocks change what the InfoNCE respiration pool summarises (one or a few
    blocks instead of the whole clip). Alignment difficulty may shift; this is monitored, not
    controlled.
  - A third placement enum adds surface area to `place_signal`, the runner, the evaluator, the
    manifest and the resume guard.

---

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A` (proposed; no training run yet).
- **Evidence Artifacts**:
  - [`../eval/baselines/tir_resp/multimae_style_panel.md`](../eval/baselines/tir_resp/multimae_style_panel.md) (P1)
  - [`../eval/dataset_inspection/rgb2d/resp_recon_zeroshot.md`](../eval/dataset_inspection/rgb2d/resp_recon_zeroshot.md) (P4)
  - [`archive/SpanMask_PhysioSignals.md`](archive/SpanMask_PhysioSignals.md) §1–§4 (P3, earlier geometry)
  - The P2 table, computed with `core.mask_placement.place_signal` at $N_s = 250$, 400 samples per
    row, seed 0. The snippet is in the plan, Phase 0.
- **Tests / Smoke Checks**: `code/tests/test_adr017_masked_span_placement.py` (to be added);
  existing `code/tests/test_mask_placement.py`, `code/tests/test_two_stream_anatomy_masking.py` and
  `code/tests/test_budget_masking.py` must stay green unchanged.
- **Rollback / Rejection Condition**: Set `physio_placement: scattered` and both gap bounds to `0`.
  This is the neutral default and is bit-identical to the pre-ADR behaviour. If Criterion 3a fails at
  every feasible bound, or Criterion 3c rejects, mark this ADR `Superseded`/`Deferred` and move it,
  with its plan, to `archive/`.
