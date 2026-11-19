# ADR-002 — constant-budget masking implementation plan

> **Scope.** Implements the masking policy decided in
> `code/notes/design/ADR-002-asymmetric-multimodal-masking.md`: the apportionment engine
> (per-step counts, two modes), the placement layer, the model-side integration, the seven
> exposed parameters, and the three falsifiable criteria. Model architecture, tokenization,
> initialisation, data preprocessing and the evaluation protocol stay out of scope, except as
> the fixed quantities the policy is defined over (§5).
>
> **State.** This is a **new implementation**. Nothing here preserves, adapts or stays
> bit-compatible with the existing masking paths (`masking: ratio`, `masking: ammm`,
> `code/core/ammm_masking.py` and its knobs); those are superseded by this design and are removed
> or replaced in the same change. Nothing of this ADR is implemented yet.
>
> **Judged by** the ADR's own criteria (§8): invariants under sampling, the operating-regime
> distribution, and a constant encoder length. Downstream accuracy is reported, not gated.

---

## 1. Definition of done

1. A pure apportionment engine produces the ADR's counts for both modes and refuses an infeasible
   configuration at construction.
2. Its invariant suite passes over at least $10^4$ draws per mode.
3. A smoke pre-training run logs the sampled mode, the per-stream counts and the encoder length,
   and shows exactly `n_budget` visible tokens on 100% of steps.
4. The seven parameters are reachable from YAML and from the command line.
5. The configuration surface carries no residue of the superseded policy: no `mask_ratio_*`
   fallback, no Dirichlet knob, no second masking policy to select between.

## 2. Target layout

| piece | new location | note |
| :--- | :--- | :--- |
| apportionment engine (policy only) | `code/core/budget_masking.py` | counts only; no tensors, no modality data, no token grids |
| placement layer | `code/core/mask_placement.py` | counts → per-sample masks; no policy |
| model integration | `code/core/multimae.py` | the encoder consumes the visible indices the policy emits |
| pre-training driver | `code/runners/run_pretrain.py` | seven parameters, per-step mode/count logging |
| arm config | `code/configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml` | declares **two visual streams** (`tir`, `rgb2d`) plus `resp` — and `bp` for the 4-stream arm — because contract 5 requires two visual streams; replaces the two `stage2_ammm_*.yaml` masking blocks |
| tests | `code/tests/test_budget_masking.py`, `code/tests/test_mask_placement.py` | gates of §8 |

Superseded and to be removed or rewritten in the same change: `code/core/ammm_masking.py`,
`code/tests/test_ammm_masking.py`, the masking keys of both `code/configs/pretrain/stage2_ammm_*.yaml`
(`masking`, `dirichlet_alpha`, `video_kept_cap`, `physio_kept_cap`, `min_kept`, `modality_dropout`,
`kept_ratio_override`, and the inert `mask_ratio_*` block), the matching flags in
`code/runners/run_pretrain.py`, the `--val_kept_<stream>` flags and their `kept_override` plumbing
(validation no longer pins ratios, §6) including their consumer in `code/engines/pretrain.py`
(`evaluate_pretrain`), and the eight docstring references to the deleted
`Asymmetric_MultiModal_Masking.md`.

## 3. Phase 1 — apportionment engine

`code/core/budget_masking.py`:

- `StreamSpec(name, kind, n_units, tokens_per_unit)` — `kind` is `visual` or `signal`.
- `BudgetPlan(mode, kept_tokens, kept_units)` — the step's counts, nothing else.
- `ConstantBudgetApportioner(streams, n_budget, video_cap_ratio, physio_cap_ratio,
  video_min_tubes, p_virtual_sensing, generator=None)` — the declared streams arrive as
  `StreamSpec` instances, so `spatial_patches` and `physio_tokens` reach the engine through
  `n_units` rather than as separate globals; every parameter name matches the ADR.
- **Construction** validates the declared set (exactly two visual streams, one or two signal
  streams), the caps and the floor, then the three feasibility conditions with
  $P = |\mathcal{P}_{\text{decl}}|$ in place of the factor 2 in condition 3
  ($N_{\text{budget}} - P\,C_p \ge 2mT$; the ADR's written form is the conservative $P = 2$
  case), each raising a `ValueError` that names the failing condition and the numbers behind it.
- **`sample_step()`** draws the mode, then the counts:
  - Mode 1: `total = n_budget // T`; draw $S_{v_1}$ uniformly on
    $[\max(m, total-M), \min(M, total-m)]$ and set $S_{v_2} = total - S_{v_1}$.
  - Mode 2: draw each physiological count integer-uniform on $[0, C_p]$.
  - **Alignment, general over $P$:** with $R = [N_{\text{budget}} - \sum_p \tilde{K}_p] \bmod T$,
    add $R$ to the **first declared** physiological stream when the result stays $\le C_p$,
    otherwise to the next one, otherwise subtract $T - R$ from the first and assert it stays
    $\ge 0$. With the ADR's numbers $C_p = 125$ is an exact multiple of $T = 25$, so for $P = 1$
    the first branch always applies: $\tilde{K} + R$ is the next multiple of 25 and cannot exceed
    125.
  - Split the visual residual over the two declared visual streams with the same clamped-uniform
    rule, $[\max(m, s-M), \min(M, s-m)]$.
  - Before returning: assert the exact sum, the per-stream caps and floors, and the tube multiple.
- The ADR's literal figures ($14$, $\{5..9\}$, $350$) enter only as the defaults of
  $(N, T, m, M, C_p)$; every bound is derived from them so the engine is defined for any feasible
  tuple.

Gate **G1**: at the ADR's defaults the engine reproduces its numbers exactly — Mode-1 splits
$5..9$, Mode-2 $S_{\text{total}} \in [4,14]$, physio in $[0,125]$.

## 4. Phase 2 — placement layer

`code/core/mask_placement.py`, written fresh (no reuse of the current mask builders):

- `place_video(counts, n_units, tokens_per_unit, generator)` — per sample, draw $S_v$ spatial
  indices and emit the flattened indices of their full temporal columns.
- `place_signal(counts, n_units, generator, mode)` — per sample, draw $K_p$ temporal indices,
  either scattered (uniform without replacement) or as contiguous spans; the choice is a run
  setting recorded with the run.
- Both return index tensors (or a boolean keep mask) and **never** touch the policy; a count of 0
  yields an empty index set for that stream, which the loss path still scores in full.
- The placement generator is rank-distinct and collision-free: one `torch.Generator` per rank,
  seeded once with `base_seed + rank` and advanced sequentially, so no arithmetic stride can
  collide however long the run is (a 64-bit hash of `(base_seed, rank, epoch, step)` is an
  equivalent alternative; a fixed multiplier such as `rank * 10000` collides once the step count
  exceeds it). Counts stay rank-invariant (§6).

Gate **G10a**: for both modes the per-sample kept counts equal the plan's counts, and placements
differ across samples of one batch.

## 5. Phase 3 — the geometry contract the policy is defined over

The policy takes unit counts as inputs, so this phase only has to make the ADR's units
representable. It is a property of the new model, not an accommodation of the old one.

1. **Token spans.** A visual token is one temporal tube covering $P_t$ frames; a physiological
   token covers `sig_kernel` samples. The model must accept an integer ratio
   $R = n_{\text{signal}}/G_t$ with `sec_per_signal_token * R == sec_per_visual_token`
   ($R = 10$ at the ADR's geometry: 320 ms tubelet, 32 ms signal token). This is a new
   construction-time rule; no ratio-1 special case is preserved.
2. **Visual units.** `input_size 96`, tubelet $4\times16\times16$, 100 frames per window give
   $G_t = 25$ steps and `spatial_patches = 36`; a visual unit is therefore 25 tokens.
3. **Signal units.** `physio_tokens = 250` requires a 1000 Hz signal grid over the 8 s window
   (kernel 32, stride 32, i.e. 8,000 samples per window). The existing artefact is written at
   100 Hz, so it is **invalidated and recomputed** on the new grid. Cache invalidation is an
   accepted cost of this design, not a constraint to design around: the new cache key records the
   1000 Hz grid identity, so a stale artefact cannot be reused silently and the rebuild is a
   scheduled step of Phase 3.
4. **Initialisation.** A 4-frame tubelet means no patch-embed transfer from a 2-frame checkpoint;
   the new model is initialised for its own geometry. Tracked outside this plan; the masking
   policy is agnostic to it.

Gate **G12**: the model constructs at the ADR's geometry and rejects a non-integer ratio.

## 6. Phase 4 — model integration

- `MultiModalMAE` constructs the apportioner from its parameters; there is **one** masking policy,
  so no `masking` selector remains.
- `make_masks`: one `sample_step()` per training step, then per-stream placement; the returned
  plan is exposed for logging.
- `forward`: gather each stream's visible indices, concatenate, and assert
  `z.shape == [B, n_budget, D]` before the encoder.
- Loss: every declared stream emits its key on every step and rank. Mode-1 physiological streams
  have $K = 0$ and are scored over all their tokens; their adapters receive no gradient on those
  steps, so `find_unused_parameters` stays enabled.
- **Decoder conditioning is in scope.** Today a 0-kept stream's decoder input is built from mask
  tokens and positions alone (`dec = mask_token + pos_embed`) and self-attends through
  `dec_blocks`, so in Mode 1 the physiological prediction would be unconditional on `z` and
  $\nabla_z \mathcal{L}_{\text{physio}} = 0$. Mode 1 therefore requires a conditioning path for
  $k = 0$ streams — cross-attention onto `z`, or prefixing `z` into the decoder sequence — and
  Phase 4 owns that change. Without it Mode 1 trains no virtual sensing, so the ADR's first
  contract would be inert while every masking invariant still passed.
- **Rank-invariant counts.** The apportioner runs on a dedicated generator seeded from a
  rank-independent key (`base_seed + epoch * steps_per_epoch + step`), so each rank derives the
  same plan independently: no per-step collective, no serialization and no host synchronization.
  The global seed set by `init_env` (`seed + rank`) must not be used for it, or one rank could
  draw Mode 1 while another draws Mode 2. Rank agreement is asserted in Gate G13, not in the
  training loop.
- **Validation is a deterministic sequence.** At the start of every validation epoch the
  validation generator is reset to a fixed seed, so each epoch replays the same sequence of modes
  and counts. The curve stays comparable without hard-coding counts and without pinning
  per-stream kept fractions.

Gate **G10b**: a two-rank smoke run through both modes completes without a collective mismatch
and with the shape assertion armed.

## 7. Phase 5 — parameters, config, logging

- The runner exposes exactly the ADR's seven keys with its defaults and ranges: `n_budget`,
  `video_cap_ratio`, `physio_cap_ratio`, `video_min_tubes`, `p_virtual_sensing`,
  `spatial_patches`, `physio_tokens`. YAML keys mirror the flag names.
- `spatial_patches` and `physio_tokens` are tokenizer-derived
  ($(input\_size/P_h)(input\_size/P_w)$ and $seq\_len/sig\_kernel$). When either is passed
  explicitly, `run_pretrain` asserts the identity and fails loudly on a mismatch; the default path
  derives `StreamSpec.n_units` from the tokenizer, so the redundancy is never the user's problem.
- One arm config at `code/configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml`, carrying the
  geometry of §5, the two visual streams contract 5 requires (`tir`, `rgb2d`) and the seven
  values; it contains no legacy masking block to switch back to.
- Per step, log `mode`, `kept_<stream>`, the realised ratio, the total and the encoder length;
  aggregate per epoch. These are the artefacts Criteria 1 and 3 read.
- The checkpoint records the seven parameters, the unit counts and the geometry, so a downstream
  stage can reconstruct the same policy without guessing.

## 8. Phase 6 — tests, gates, smoke

| gate | check | ADR criterion |
| :--- | :--- | :--- |
| G1 | defaults reproduce the ADR's numbers | — |
| G2 | exact sum $=N_{\text{budget}}$ over $10^4$ draws, both modes | Criterion 2 |
| G3 | per-stream caps and floors, both modes | Criterion 2 |
| G4 | visual counts are multiples of $T$ | Criterion 2 |
| G5 | Mode 1 ⇒ physio exactly 0, visuals sum to the budget | Criterion 1 |
| G6 | Mode 2 ⇒ $S_{\text{total}} \in [4,14]$; split bounds degenerate correctly at both ends | — |
| G7 | Mode-1 frequency matches `p_virtual_sensing` within the binomial interval | Criterion 1 |
| G8 | identical counts across a batch with differing placements; replay from a fixed seed | Criterion 2 |
| G9 | rejection: one visual stream, three visual streams, $N \bmod T \neq 0$, infeasible 340, cap/floor inconsistent | Feasibility precondition |
| G10 | counts → masks equality; `[B, n_budget, D]`; a 0-kept stream is scored **and** its loss backpropagates — `losses['resp'].backward()` leaves non-zero grads on the visible streams' adapters and on the encoder; two-rank smoke | Criteria 2–3 |
| G11 | encoder length is exactly `n_budget` on 100% of logged steps | Criterion 3 |
| G12 | model constructs at the ADR geometry, rejects a non-integer token ratio | §5 |
| G13 | two ranks independently derive an identical plan (mode + counts) on every step | Contract 6 |
| G14 | the alignment remainder goes to the first physiological stream (BP), never Resp first | §2 allocation rules |
| G15 | a single physiological stream runs end to end, exercising the $P=1$ residual branch | Contract 5 |
| G16 | Mode 1: $\nabla_z \mathcal{L}_{\text{physio}} \neq 0$ — the loss reaches the encoder, proving the decoder is conditioned on `z` | Contract 7 |
| G17 | placement seeds are distinct for every (rank, step) pair of a run, with no fixed stride | Contract 6 |

Smoke (Criterion 2): 10 epochs with `--max_clips 100 --batch_size 8 --warmup_epochs 0` (the LR
scheduler asserts below a full warmup). Criterion 3 is read from the same log.

### BDD acceptance scenarios

```gherkin
Feature: ADR-002 Constant-Budget Masking Policy

  Background:
    Given two visual streams "tir" and "rgb2d" with 36 spatial patches and 25 tokens per tube (T=25)
    And one or two physiological streams with 250 tokens (C_p=125)
    And a constant visible budget n_budget = 350
    And video_min_tubes = 2 (m=2) and video_cap_ratio = 0.25 (M=9)

  Scenario: Mode 1 — Pure Virtual Sensing Apportionment
    Given the apportioner samples Mode 1
    When sample_step() is evaluated
    Then kept physiological tokens K_bp and K_resp must equal exactly 0
    And total visual tokens must sum to exactly 350 (14 tubes)
    And visual tube counts S_tir and S_rgb2d must each fall in [5, 9]
    And S_tir + S_rgb2d == 14

  Scenario: Mode 2 — Two-Signal Inpainting and BP Alignment Priority
    Given declared physiological streams ["bp", "resp"]
    And the apportioner samples Mode 2
    When raw counts K_bp_tilde = 100 and K_resp_tilde = 101 are drawn (sum = 201)
    Then the remainder R = (350 - 201) % 25 = 24
    And because K_bp_tilde + 24 <= 125, R is added to bp, yielding K_bp = 124
    And K_resp remains 101
    And remaining visual budget is 350 - 225 = 125 tokens (S_total = 5 tubes)
    And S_tir and S_rgb2d must fall in [2, 3] and sum to 5
    And the total visible tokens across all streams must equal 350

  Scenario: Mode 2 — Single Physiological Stream (P = 1) Residual Alignment
    Given declared physiological streams ["resp"]
    And the apportioner samples Mode 2
    When any raw count K_resp_tilde in [0, 125] is drawn
    Then the remainder R = (350 - K_resp_tilde) % 25 is added to resp
    And aligned K_resp must be a multiple of 25 and remain <= 125
    And remaining visual tokens must equal 350 - K_resp in whole tubes
    And the total visible tokens must equal 350

  Scenario: Construction Rejection on Infeasible Parameterization
    When ConstantBudgetApportioner is constructed with any of:
      | invalid_configuration                                               | failing_condition  |
      | exactly one visual stream ["tir"]                                   | stream count       |
      | three visual streams ["tir", "rgb", "rgb2d"]                        | stream count       |
      | n_budget = 340 with T = 25                                          | condition 1 (mod)  |
      | n_budget = 500 with T = 25 (2*M*T = 450)                           | condition 2 (cap)  |
      | n_budget = 300, 2 physio streams (C_p=125, leaves 50 < 2*m*T=100)  | condition 3 (sat)  |
    Then construction must raise ValueError with the failing parameter details

  Scenario: Batch Uniformity with Independent Sample Placements
    Given a batch of size B = 8
    When make_masks() generates masks for the step
    Then the visible token count for each stream must be identical across all 8 samples
    And the total visible tokens for each sample must be exactly 350
    And spatial tube placement indices must differ across the 8 samples

  Scenario: Mode-1 Gradient Path to Visual Encoder
    Given a step sampled in Mode 1 where K_resp = 0
    When forward() executes on the batch and loss = losses["resp"]
    And loss.backward() is called
    Then the gradient of loss with respect to encoder latent z (d_loss / d_z) must be non-zero
    And visual adapter gradients (adapters["tir"].weight.grad) must be non-zero
    And physio input adapter gradients (adapters["resp"].weight.grad) must be None

  Scenario: Deterministic Rank-Independent Count Derivation
    Given two DDP ranks 0 and 1 with different rank seeds (seed + rank)
    When each rank generates a BudgetPlan using the dedicated step generator (base_seed, epoch, step)
    Then rank 0 and rank 1 must derive identical mode and per-stream counts
    And neither rank shall invoke collective communication to synchronize the plan
```

## 9. Phase 7 — documentation

- `code/notes/design/ADR-002-asymmetric-multimodal-masking.md` is the only design record; no
  derivation document is written alongside it.
- Rewrite the masking section of `code/notes/spec/02_model_architecture.md` from the new code, and
  regenerate/adjust `code/scripts/project/write_specs.py` to match.
- Remove the new plan from the repository or mark it complete once §1 holds; a plan that outlives
  its implementation is a stale source of truth.

## 10. Order and dependencies

`3 (geometry) ∥ 1 → 2 → 4 → 5 → 6 → 7`. Phase 1 is independent of everything and lands first with
its tests. Phase 3 also carries the signal-cache rebuild of §5.3. Phase 4 needs §5.1 and §5.3 to
run at the ADR's defaults, but its **wiring** can be exercised earlier with any feasible tuple —
e.g. the current 50-token tube, where $350/50 = 7$ tubes is legal — to prove counts → masks →
`[B, n_budget, D]` before the new grid exists. Such a run validates the plumbing only; the ADR's
token figures come from §5.

## 11. Resolved decisions

- **Validation masking:** a fixed-seed deterministic draw sequence per validation epoch (§6) — no
  hand-pinned counts and no `val_kept_*` pin for this policy.
- **Arm config:** `code/configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml`, declaring `tir`
  and `rgb2d` as the two visual streams plus `resp`, adding `bp` only for the 4-stream arm.
- **Data:** §5.3 rebuilds the signal cache at 1000 Hz rather than preserving the 100 Hz artefact.

Nothing blocks Phase 1.

## 12. Non-goals

No change to the loss functional form, model width, initialisation strategy, data selection, folds
or evaluation protocol. No new metrics. No second masking policy kept "for comparison" — the
superseded one is removed, and the ADR's criteria do not depend on a head-to-head result. The one
architectural change this plan owns is the decoder conditioning of §6, without which Mode 1 has
no gradient path from the waveform loss to the visual encoder.
