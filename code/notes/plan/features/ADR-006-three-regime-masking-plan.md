# ADR-006 — three-regime single-modality masking implementation plan

> **Authority.** `code/notes/design/ADR-006-three-regime-virtual-sensing-masking.md` is the contract;
> this document only says how it is built and in what order. Where the two disagree, the ADR wins and
> this plan is wrong.
>
> **Scope.** Adds the Regime-3 step class (single-carrier virtual sensing, submodes 3A `rgb2d`-only
> and 3B `tir`-only), its probability schedule `p_s(e)`, the regime-indexed visual cap, the extended
> `BudgetPlan` label, the runner keys and one arm configuration. Model architecture, tokenization,
> data preprocessing, the Stage-3 recipe and the evaluation protocol stay out of scope.
>
> **Compatibility guarantee (this is the point of the plan).** The landing is **additive and
> neutral** on top of the ADR-005 implementation: no module is forked, no existing key changes
> meaning, and a configuration that mentions none of the three new keys produces the **bit-identical
> mask sequence** it produces today for a fixed `(mask_base, seed, epoch, step)`. Neutrality is
> achieved by *defaults plus draw gating*, not by a second engine — the ADR-005 warning applies
> unchanged: a forked apportioner would drift from the invariants and the 10⁴-draw suite would guard
> only one branch. Concretely, the single-carrier branch is unreachable when `p_single_modality = 0`,
> and **no random number is drawn on the way to it** (§3.1), so the RNG stream does not move.
>
> **State (2026-11-24).** Phases 1–5 are **implemented and verified**, including the
> data-dependent smoke of §3.3 on a real 16-clip development partition (§6.2): the full suite is
> green at 844 passed and the arm reaches its first step with the constant budget on both carrier
> submodes. Phase 6 (measurement) belongs to `code/notes/plan/experiments/`.
>
> **Judged by** ADR-006 §3: Criterion 1 (policy conformance) is a gate for this plan; Criterion 2
> (the subject-level headline) is the *arm's* gate and is recorded in `code/notes/eval/`, never here.

---

## 1. Definition of done

1. With the three new keys absent (or `p_single_modality = 0`), the kept-count sequence over 50 000
   draws at a fixed `(mask_base, epoch, step)` is **bit-identical** to the reference fixture captured
   from the current engine.
2. A Regime-3 step allocates exactly `tir 350 / rgb2d 0 / resp 0` (3B) or `rgb2d 350 / tir 0 /
   resp 0` (3A), and the encoder input is `[B, 350, D]` on it.
3. `p_s_at(epoch)` implements ADR-006 §2.7, is non-decreasing in the epoch, and is validated at
   construction, not at the first step.
4. `plan.mode` and `plan.mode_by_counts` keep their **exact** ADR-005 meanings; Regime 3 is exposed
   as a *new, additional* label. Nothing that currently reads `mode` is redefined.
5. The zeroed visual stream keeps its loss key and receives non-zero gradient on a Regime-3 step.
6. The feasibility rules 4–5 of ADR-006 §2.6 reject an infeasible tuple at configuration-resolve
   time.
7. Every ADR-006 §2.8 record is emitted, and `masking.json` carries the new tuple.
8. The three-regime config resolves through `_check_config_keys` and reaches its first step with
   `[B, 350, D]`.
9. Gates G1–G10 pass; `code/tests/test_budget_masking.py`, `code/tests/test_mask_placement.py` and
   `code/tests/test_curriculum_schedule.py` are **unchanged** and green — that is the compatibility
   proof.

## 2. What already satisfies ADR-006 (nothing to build)

Verified against the code before planning; this is why the delta is small.

| ADR-006 requirement | where it already holds |
| :--- | :--- |
| exactly two visual + one physiological stream (§2.6 rule 6) | `ConstantBudgetApportioner._validate` |
| Regimes 1 and 2 arithmetic (§2.4) | `_mode2` / `_mode1` in `code/core/budget_masking.py` |
| $T = 25$, $R = 10$, `tubelet (2,16,16)`, 36 tubes (§2.2) | `MultiModalMAE._check_space_time_alignment`, already the config values |
| contract 7 conditioning — mechanism (a), prefixing (§2.3) | `MultiModalMAE.forward`: the decoder prefix is `torch.cat([z, mask_tokens])` |
| a 0-kept stream skips its adapter and keeps a loss key (§2.3 clauses 7, 9) | `MultiModalMAE.forward` (the `plan.kept_tokens[s] == 0` branches) |
| `mode_by_counts` resolution (§2.3 contract 4) | `_plan`, `BudgetPlan.mode_by_counts` |
| per-step counts / per-sample placement, replay determinism (§2.3 clause 8) | `code/core/mask_placement.py` |
| explicit kernel designation (ADR-005 I2) | `--patch_stream`, `write_masking_manifest` |
| `kept_*`, `kept_units_*`, `mode`, `mode_resolved`, `is_virtual`, `p_virtual` logging | `code/engines/pretrain.py::train_one_epoch` |

**What is genuinely missing** is therefore: (i) a single-carrier allocation branch, (ii) the draw
that selects it, (iii) an invariant exemption for the zeroed stream, (iv) three runner keys, (v) a
`BudgetPlan` label, (vi) an eval-pin path that can express 350/0 without the floor clamp.

## 3. Phase 1 — apportioner (the only hot-path change)

**3.1 `code/core/budget_masking.py`**

- New module constants, additive only: `REGIME_INPAINTING = 'inpainting'`,
  `REGIME_BI_VISUAL = 'bi_visual'`, `REGIME_SINGLE = 'single'`, `REGIMES = (...)`. **Do not** touch
  `MODE_VIRTUAL_SENSING` / `MODE_INPAINTING`: Regime 3 is a refinement of Mode 1, not a replacement.
- `__init__` gains, with the **exact** names of ADR-006 §2.9 (so the YAML key, the flag dest and the
  Python keyword are one string and `parser.set_defaults(**cfg)` reaches the apportioner without a
  translation layer):
  - `single_carrier_tubes: int = 0` — `0` means "derive as `n_budget // tokens_per_unit`"; a
    non-zero value must **equal** that derivation, otherwise construction raises (an assertion
    expressed as an option, the same pattern as `spatial_patches`).
  - `p_single_modality: float = 0.0` — `p_{s,target}`.
  - `single_modality_warmup_p: float = 0.0` — `p_{s,warm}`.
- `_validate` additions:
  - `0 ≤ p_s_target ≤ 1`, `0 ≤ p_s_warm ≤ 1`;
  - `p_s_warm ≤ p_s_target` **unconditionally** — ADR-006 §2.7 and §2.9 admit
    `[0, p_single_modality]` with no exemption. There is no `E_warm = 0` carve-out here, because none
    is needed: `p_warm`'s exemption exists only because its 0.10 default would otherwise make a legal
    `p_target = 0.0` run unbuildable, while this value defaults to 0.00;
  - `single_carrier_tubes ∈ {0} ∪ {N // T}`, where `0` is the **raw sentinel** meaning "derive" and
    the **resolved policy value** is `S_v = N // T` (= 14 at this geometry) — the number ADR-006
    §2.9 names as the default. The resolved value must satisfy `1 ≤ S_v ≤ min(n_units)` (ADR-006
    §2.6 rule 4); rule 5 then holds **by construction**, because the sole-carrier cap *is*
    `S_v / units_v` (14/36 = 38.9 %), so the kept fraction cannot exceed it;
  - `p_s_target > 0` requires exactly two visual streams (already enforced, asserted separately so
    the failure names the *single-modality* tuple rather than the general stream-set rule).
- New `p_s_at(epoch: Optional[int]) -> float` — mirrors `p_at` exactly: `p_s_target` when
  `epoch is None` or `epoch >= E_warm`, else `p_s_warm`.
- `sample_step(generator=None, epoch=None)`: signature unchanged, and `epoch` now reaches **both**
  branches, exactly as ADR-006 §2.5 specifies:

  ```
  if u < self.p_at(epoch):     return self._mode1(g, epoch)
  return self._mode2(g, epoch)
  ```

  **Both branches must reach the same single-carrier decision.** ADR-006 §2.3 contract 4 makes a
  Mode-2 draw whose aligned `K_resp` is zero a *sensor-absent* step, and §2.5 routes it through the
  same `p_s(e)` test as a sampled Mode-1 step. Implemented as one shared
  `_sensor_absent(g, epoch, mode)` helper: `_mode1` delegates to it, and `_mode2` calls it with
  `mode = MODE_INPAINTING` when `sum(aligned K_resp) == 0`, **before** it splits the visual budget.
  Consequences to preserve:

  - the sampled `mode` stays `MODE_INPAINTING` on that path (ADR-005 contract 4); only the resolved
    `mode_by_counts` and `regime` change;
  - the `_plan` invariant for `REGIME_SINGLE` must therefore check **`mode_by_counts`**, not the
    sampled `mode`;
  - the extra `p_s` draw happens *after* the `K_tilde` draw, so the RNG order is the ADR's;
  - with `p_s = 0` the helper consumes no draw at all, so this path is bit-identical to the
    pre-ADR-006 engine.

  `_mode1` and `_mode2` both default `epoch` to `None`:
  `_mode1(g, epoch: Optional[int] = None)`, `_mode2(g, epoch: Optional[int] = None)`.
  `code/tests/test_curriculum_schedule.py::_pre_change_draw` calls `app._mode1(gen)` **and**
  `app._mode2(gen)` with one positional argument, and that suite must stay unmodified (DoD 9), so a
  required second parameter turns the compatibility proof into a `TypeError`.

  ```
  p_s = self.p_s_at(epoch)
  if p_s > 0.0:                       # <-- the neutrality gate: no draw when disabled
      if uniform(g) < p_s:
          carrier = primary if uniform(g) < 0.5 else secondary
          return self._single_carrier(carrier)
  total = self.n_budget // self.tokens_per_unit
  s1, s2 = self._split_tubes(total, g)
  ```

  **The `p_s > 0.0` guard is the whole compatibility mechanism.** It must be a guard, not
  `uniform(g) < 0.0` on an already-drawn sample: any consumed sample would shift every subsequent
  count and placement draw of the run.
- New `_single_carrier(name)` — builds the plan directly, never calling `_split_tubes`:
  `units = {carrier: S_v, other: 0, ** {signal: 0}}`, `tokens = {carrier: S_v*T, other: 0, ...}`.
- `_plan` must **branch its invariant re-check on the regime**, because the current loop asserts
  `video_min_tubes ≤ kept ≤ video_cap_units[stream]` for *every* visual stream and 14 > 9 would
  fire on a legitimate Regime-3 plan:
  - Regime 1/2: unchanged.
  - Regime 3: `kept_units[carrier] == S_v` and every other stream exactly 0; the floor and the
    multi-carrier cap both skip.
  `_plan` therefore needs the regime; pass it in explicitly rather than re-deriving it from the
  counts (the counts of a 3A step and a hypothetical mis-specified 2-stream Mode-1 step are
  indistinguishable to the label logic, and a regime-specific invariant must not be inferred).

**3.2 `plan_from_counts`** (used by eval pins) infers `regime`: `inpainting` iff any signal keeps
`>0`; else `single` iff exactly one visual keeps `>0`; else `bi_visual`. The invariant exemptions of
§3.1 apply identically here, or every existing evaluation pin would need the new label.

## 4. Phase 2 — label, payload and observability

**4.1 `BudgetPlan` gains two additive fields** — `regime: str = REGIME_BI_VISUAL` and
`carrier: Optional[str] = None`, both validated in `__post_init__`, both defaulted so positional
construction in existing tests keeps working:

- `mode` and `mode_by_counts` are **not touched**. A Regime-3 step continues to report
  `mode_by_counts == MODE_VIRTUAL_SENSING`, which is correct (no physiological token) — and it is why
  the ADR-005 tests that assert on `mode_by_counts` stay valid.
- `carrier` is the name of the sole carrier on a `single` plan and `None` otherwise.

**4.2 `code/core/multimae.py`** — `make_masks` and `forward` already forward `epoch` to
`sample_step`, so no signature changes are needed. Additions:
- `forward`'s returned dict gains, **for every step**, `'regime': plan.regime` and
  `'carrier': plan.carrier` (`None` outside Regime 3), plus `'p_single':
  float(self.apportioner.p_s_at(epoch))` — the active `p_s(e)` that ADR-006 §2.8 requires per step,
  alongside the existing `'p_virtual'`. The raw `regime`/`carrier` labels are the per-step record;
  the numeric indicators of §4.3 are **additional aggregates, not replacements**.
- `[mask]` provenance print gains the resolved `(p_s_warm, p_s_target)` and `S_v`.
- `plan_from_counts` needs nothing beyond §3.2.

**4.3 `code/engines/pretrain.py`** — inside the existing `if isinstance(out, dict):` block, add the
numeric indicators the per-epoch record needs (`metric_logger` averages numbers, so the regime must
be encoded numerically, exactly as `is_virtual` was): `is_single = 1 if out['regime'] == REGIME_SINGLE
else 0`, plus `is_single_<stream> = 1` when `out['carrier'] == stream` so the 3A/3B split is
observable per epoch. Import the constants; no literal `'single'` in the engine. Add the same
sample-weighted accumulation for `is_single` / `is_single_<stream>` to `evaluate_pretrain`'s payload,
DDP merge and return value, alongside the existing `mode_resolved` accumulation.

**4.4 `code/runners/run_pretrain.py`** — register on the existing masking block of `get_args()`, all
three defaulting to the ADR-005 behaviour:

| flag | type | default | note |
| :--- | :--- | ---: | :--- |
| `--single_carrier_tubes` | int | `0` | `0` = derive `n_budget // tokens_per_unit` |
| `--p_single_modality` | float | `0.0` | `p_{s,target}` |
| `--single_modality_warmup_p` | float | `0.0` | `p_{s,warm}` |

Forward them through `build_pretraining_model(args)` with the same `getattr` pattern as the other
masking keys, and add them to the `[mask]` print. `_check_config_keys` in `code/runners/_common.py`
fails on any YAML key that is not a parser `dest`, so the config of Phase 4 is unusable until these
exist; `_common.py` itself needs no edit. Extend `write_masking_manifest`'s `policy` block with the
three keys, the resolved `S_v` and the resolved `(p_s,warm, p_s,target)` pair — a resumed run whose
record disagrees must be rejected, which is the existing behaviour for the rest of the tuple.

**4.5 `code/runners/run_eval_recon.py`** — `pin_plan_from_flags` clamps every visual count to
`[video_min_tubes, video_cap_units]` and then fills the residual to the constant budget, so it cannot
express 350/0 (this is why Arm A of the ADR-005 baseline realised 225/125 when `--kept_rgb2d 0.0` was
requested). Add `--pin_single_carrier {none,rgb2d,tir}` (default `none`): when set, that stream is
pinned to `n_budget` and every other visual to 0, bypassing the floor/cap clamp, and the caption
records `regime=single carrier=<name>`. Without this the ADR-006 §3.2 Target-B and Target-C probes
cannot be run at all; with it, they are one flag.

## 5. Phase 3/4 — the arm configuration

New file `code/configs/pretrain/stage2_budget_3regime_single_modality.yaml`, derived from
`code/configs/pretrain/stage2_budget_3stream_curriculum.yaml`. Four keys change; everything else —
including `streams`, `n_budget`, the caps, the floor, the tokenizer geometry, `patch_stream: tir`,
`epochs: 150`, `batch_size: 16`, `blr`, the task groups, `physio_placement` — is identical, so the
pair differs only by the policy and the conflation of a curriculum-length change is explicit:

| key | base (ADR-005 arm) | ADR-006 arm | why |
| :--- | :--- | :--- | :--- |
| `virtual_sensing_warmup_epochs` | `50` | `50` | **unchanged** — the step class is the only difference |
| `p_single_modality` | — | **`0.20`** | §2.7 `p_{s,target}` |
| `single_modality_warmup_p` | — | **`0.00`** | §2.7 `p_{s,warm}` (Phase I has no Regime 3) |
| `single_carrier_tubes` | — | **`14`** | §2.4 `S_v`; equals `n_budget / T`, asserted at construction |

Operational notes: the two ROI caches and the respiration cache stay valid (their keys carry clip
length, fps and fs, not the temporal stride or the hop); `clip_stride: 4.0` is unchanged and must
stay unchanged or the comparison against the ADR-005 arm is confounded; `--warmup_epochs` must stay
below `epochs` for the cosine schedule. `E_{\text{warm}}` is deliberately **not** changed: with it
held at 50 the arm differs from the ADR-005 arm by the step class alone, so a comparison between
the two isolates the policy instead of confounding it with a schedule change. No second config is
needed for that attribution, and none is registered.

## 6. Phase 5 — tests

Existing suites are **not modified** (DoD 9). New: `code/tests/test_regime_single_carrier.py`.

| gate | assertion |
| :--- | :--- |
| **G1** | Neutrality: with the three new keys absent, and separately with `p_single_modality = 0` but the other two set, the kept-count sequence over 50 000 draws at a fixed `(mask_base, epoch, step)` equals the pre-change fixture — captured once and committed. Also asserts the *generator state* advances identically (a consumed-then-ignored draw would pass a counts-only test only by luck, so advance the generator twice and compare the third). |
| **G2** | `p_s_at(e) == p_s_target` for `e ≥ E_warm` and for `None`; `== p_s_warm` below; non-decreasing in `e`. |
| **G3** | Construction rejects `p_s_warm > p_s_target` **for every `E_warm`, including `E_warm = 0`**; it also rejects `single_carrier_tubes ∉ {0, N//T}` and `p_s_target > 0` with a stream set that is not two-visual. The `E_warm = 0` exemption exists only for `p_warm` (its 0.10 default), never for the single-carrier pair (ADR-006 §2.7). |
| **G4** | Over 10⁴ draws at `p_s_target = 0.20`, `E_warm = 50`, `p_virtual_sensing = 0.50`: the Phase-I single-carrier share is exactly 0; in Phase II the sensor-absent mass is `p + (1-p)/126 = 0.50397` (not `0.50`, because a zero-count Mode-2 draw is sensor-absent), so `is_single` is `0.10079 ± 0.012`, the 3A:3B split is `504 ± 63` of the ≈1008 single steps, and the regime shares are `0.4960 / 0.4032 / 0.1008` — the rounded `50 / 40 / 5+5` target, tested against those exact probabilities (4σ binomial), never as the literal `0.50 / 0.40 / 0.05 / 0.05`. |
| **G5** | Allocation exactness: every `single` plan has `carrier ∈ {tir, rgb2d}`, `kept_units[carrier] == 14`, all other streams 0, and `total_tokens == 350`; `_split_tubes` is not called (assert the two streams are never both non-zero). |
| **G6** | Label compatibility: on a 3A step `mode_by_counts == MODE_VIRTUAL_SENSING` (unchanged meaning) **and** `regime == 'single'` **and** `carrier == 'rgb2d'`; on a Regime-2 step `regime == 'bi_visual'`, `carrier is None`. Existing `mode`/`mode_by_counts` assertions in `test_budget_masking.py` are untouched and green. |
| **G7** | Gradient path: on a Regime-3 step, `∂L_m/∂z ≠ 0` for the zeroed visual stream and for `resp` (ADR-006 §3.1 clause 6), and the encoder input is `[B, 350, D]`. |
| **G8** | `plan_from_counts({'tir': 350, 'rgb2d': 0, 'resp': 0})` yields `regime == 'single'`, `carrier == 'tir'` and passes the invariants; `plan_from_counts` on a Regime-2 allocation still yields `bi_visual`. |
| **G9** | `_check_config_keys` accepts the Phase-4 configs; `masking.json` contains the three keys, the resolved `S_v` and the regime probabilities; two short runs with `--p_virtual_sensing 1.0` and `--p_single_modality 1.0` vs `0.0` produce at least one Regime-3 and zero Regime-3 steps respectively. The forced `p_virtual_sensing` is required: at the config's `0.50` a two-step epoch hits the Regime-3 branch on only about half its steps, so the assertion would be a coin flip. |
| **G10** | `code/tests/test_config_loading.py` (parses **every** YAML with its real runner's `get_args()`) passes with the two new configs — it is a gate of its own here, and it fails until Phase 2.4 lands. |

Kernel routing (ADR-005 I1/I2) is unchanged by this plan and stays covered by the existing suite.

```text
cd code
python -m pytest tests/test_budget_masking.py tests/test_mask_placement.py \
                 tests/test_curriculum_schedule.py tests/test_regime_single_carrier.py \
                 tests/test_config_loading.py -q

# G9/G10 — smoke the new arm with the curriculum disabled, regime class FORCED ON.
python -u runners/run_pretrain.py \
    -c configs/pretrain/stage2_budget_3regime_single_modality.yaml \
    --exp_root /tmp/adr006_smoke --fold 0 --dev_smoke --epochs 1 --warmup_epochs 0 \
    --virtual_sensing_warmup_epochs 0 --p_virtual_sensing 1.0 --p_single_modality 1.0 \
    --max_entries 16 --batch_size 8 --num_workers 0 \
    --output_dir /tmp/adr006_smoke/out \
    --roi_cache <ROI_CACHE> --rgb2d_cache <RGB2D_CACHE> --resp_cache <RESP_CACHE> \
    --min_signal_spread 0 --rail_touch_v 0

# G9/ADR-006 §3.3 — the SAME arm with the class FORCED OFF: zero Regime-3 steps,
# is_single = 0, and the encoder input still [B, 350, D].
python -u runners/run_pretrain.py \
    -c configs/pretrain/stage2_budget_3regime_single_modality.yaml \
    --exp_root /tmp/adr006_smoke_off --fold 0 --dev_smoke --epochs 1 --warmup_epochs 0 \
    --virtual_sensing_warmup_epochs 0 --p_virtual_sensing 1.0 --p_single_modality 0.0 \
    --max_entries 16 --batch_size 8 --num_workers 0 \
    --output_dir /tmp/adr006_smoke_off/out \
    --roi_cache <ROI_CACHE> --rgb2d_cache <RGB2D_CACHE> --resp_cache <RESP_CACHE> \
    --min_signal_spread 0 --rail_touch_v 0

# G8 — single-carrier eval pin on the existing ADR-005 checkpoint (no floor clamp).
python -u runners/run_eval_recon.py --ckpt <ADR005_CKPT> --split test \
    --resp_probe hidden --pin_single_carrier tir --repeats 5
```

`--max_entries` is the per-SPLIT cap (`--max_clips` is per session and `--max_sessions` is not
forwarded by the ROI lineage), and `--min_signal_spread 0 --rail_touch_v 0` are required whenever
`--resp_cache` is attached.

### 6.1 BDD-style acceptance scenarios

```gherkin
Feature: Three-Regime Constant-Budget Masking with Single-Modality Virtual Sensing (ADR-006)

  Background:
    Given streams (tir, rgb2d, resp) with N = 350, T = 25, visual cap c_v = 0.25 (M = 9 tubes),
      visual floor m = 2 tubes, physio cap c_p = 0.50 (C = 125 tokens), S_v = 14 tubes

  Scenario: Neutral count sequence when the regime is disabled (Gate G1, DoD 1)
    Given an apportioner constructed without the three ADR-006 keys
    When 50,000 steps are sampled with fixed (epoch, step) generators
    Then the kept-token sequence is bit-identical to the committed pre-change fixture
    When one further draw is taken from each generator
    Then it equals the fixture's recorded next draw
    And tests/test_curriculum_schedule.py::test_e_warm_zero_replays_the_pre_change_sequence
        passes unmodified
    # A live comparison against "the previous engine" is impossible once the code has changed;
    # the fixture plus the next-draw check is what actually detects a consumed-then-ignored sample.

  Scenario: Single-carrier allocation and invariant exemption (Gate G5, ADR §2.3 Contract 3, §2.4)
    Given p_virtual_sensing = 1.0 so that every step is sensor-absent
    And p_single_modality = 1.0 and single_carrier_tubes = 14
    When a step is sampled
    Then exactly one visual stream keeps 14 tubes and 350 tokens
    And every other visual stream and the respiration stream keep 0 tokens
    And the sum over streams equals 350 and the plan passes the invariant re-check

  Scenario: The carrier draw is subordinate to the sensor-absent draw (ADR §2.7)
    Given p_virtual_sensing = 0.50 and p_single_modality = 1.0
    When 10,000 steps are sampled at an epoch past the warm-up boundary
    Then the single-carrier fraction is 0.50397 = p + (1 - p)/126 -- the RESOLVED
        sensor-absent mass, which includes the 1/126 zero-count Mode-2 draws --
        within a 4-sigma binomial tolerance; it is neither 1.0 nor exactly 0.50
    And every single-carrier step also satisfies mode_by_counts == MODE_VIRTUAL_SENSING

  Scenario: Regime labels are additive, not redefinitions (Gate G6, ADR §2.3 Contract 4)
    Given a Regime-3 step with carrier rgb2d
    Then plan.mode_by_counts is MODE_VIRTUAL_SENSING (its ADR-005 meaning)
    And plan.regime is 'single' and plan.carrier is 'rgb2d'
    Given a Regime-2 step
    Then plan.regime is 'bi_visual' and plan.carrier is None
    And plan.mode and plan.mode_by_counts carry the values they carried before ADR-006

  Scenario: Schedule introduces the regime only after grounding (Gate G2, G4, ADR §2.7)
    Given p_virtual_sensing = 0.50 with virtual_sensing_warmup_p = 0.10 for the phase probability
    And p_single_modality = 0.20, single_modality_warmup_p = 0.00, virtual_sensing_warmup_epochs = 20
    When steps are sampled at epoch = 10
    Then no Regime-3 step is produced
    When steps are sampled at epoch = 60
    Then 20 percent of the sensor-absent steps are Regime 3
    And the 3A and 3B submodes occur with equal frequency

  Scenario: Zeroed carrier keeps a gradient path (Gate G7, ADR §2.3 Contract 7)
    Given a MultiModalMAE forward on a Regime-3 plan with carrier rgb2d
    Then the encoder input has shape [B, 350, D]
    And the gradient of the tir reconstruction loss with respect to the encoder latents is non-zero
    And the gradient of the respiration loss with respect to the encoder latents is non-zero

  Scenario: Infeasible tuples are rejected before the first step (Gate G3, ADR §2.6)
    Given single_carrier_tubes = 10 with n_budget = 350 and T = 25
    When the apportioner is constructed
    Then construction raises ValueError naming the carrier allocation rule
    Given p_single_modality = 0.0, single_modality_warmup_p = 0.9 and E_warm = 0
    When the apportioner is constructed
    Then construction raises ValueError naming single_modality_warmup_p, because
        that ordering is unconditional -- E_warm = 0 excuses p_warm only

  Scenario: Explicit evaluation pins resolve the regime (Gate G8)
    Given plan_from_counts with counts {tir: 350, rgb2d: 0, resp: 0}
    Then the plan's regime is 'single' and its carrier is 'tir'
    And neither the multi-carrier floor nor the multi-carrier cap is applied to it
    Given plan_from_counts with a bi-visual allocation {tir: 175, rgb2d: 175, resp: 0}
    Then the regime is 'bi_visual' and the carrier is None

  Scenario: Config resolution, manifest and forced-regime smoke (Gate G9, G10)
    Given stage2_budget_3regime_single_modality.yaml
    When it is parsed by the runner's get_args() and passed to _check_config_keys
    Then it resolves, and single_carrier_tubes equals n_budget // tokens_per_unit
    And masking.json carries p_single_modality, single_modality_warmup_p,
        single_carrier_tubes, the resolved S_v and the regime probabilities
    When a 1-epoch smoke runs with --p_virtual_sensing 1.0 --p_single_modality 1.0
    Then every step logs is_single = 1 and the encoder input is [B, 350, D]
    When a 1-epoch smoke runs with --p_single_modality 0.0
    Then every step logs is_single = 0
```

### 6.2 Executed record (2026-11-24)

Environment: repo venv (`.env`), one local Tesla T4; caches `$WORK_SCRATCH/tir_roi_cache`,
`rgb2d_roi_cache`, `resp_cache`; `--dev_smoke --max_entries 16 --batch_size 8 --num_workers 4`.

- `python -m pytest -q` over the whole `code/` tree: **844 passed** (was 822 before this change),
  including the new `tests/test_regime_single_carrier.py` (**22 passed**, G1–G10) and the four
  pre-existing suites **unmodified**: `test_budget_masking.py`, `test_mask_placement.py`,
  `test_curriculum_schedule.py`, `test_multimae_rgb2d.py` — that is the DoD 9 compatibility proof.
- G10 is exercised by `tests/test_config_loading.py`, which parses every YAML with its runner's
  `get_args()`; the two new configs resolve with the three new keys present as `dest`s.
  `tests/test_task_group_configs.py` additionally required the two configs to be registered in its
  inventory (`TIR_ROI_CONFIGS`) — done.
- **G1 in detail.** For three constructor variants (keys absent; `single_carrier_tubes = 14` only;
  the full arm tuple) and four epoch contexts (`None`, `0`, `E_warm`, `149`), the 500-step plan
  sequence equals the pre-ADR-006 selection path plan-for-plan, **and the generator's next raw draw
  is bit-identical** — the check that a consumed-then-ignored sample would fail.
- **§3.3 smoke, both directions — passed.** (2 steps each, ~3.4 s/it, precision `none`.)
  (a) **Class forced on** (`--p_virtual_sensing 1.0 --p_single_modality 1.0`): `[mask] … p_single=1
  S_v=14`. Step 0 realised **3A** (`kept_rgb2d 350 = 14 tubes`, `tir 0`, `resp 0`,
  `is_single_rgb2d 1`), step 1 realised **3B** (`tir 350`, `is_single_tir 1`), `n_batch = 350` on
  both, `mode`/`mode_resolved = 1` (ADR-005 meanings intact). All three `mse_*` terms are present on
  both steps **including the zeroed streams** (`mse_tir 1.99` on the step where `tir` held 0 tokens)
  — contract 7 on real data, not just on a synthetic batch. `masking.json`, `metrics.jsonl` and
  `selection.json` written; validation ran (5.143007 over 16 samples). Resource: 5.74 GB allocated /
  6.49 GB reserved at batch 8.
  (b) **Class forced off** (`--p_virtual_sensing 1.0 --p_single_modality 0.0`): every step is
  bi-visual (`tir 225 / rgb2d 125` and `tir 150 / rgb2d 200`), **zero** Regime-3 steps,
  `is_single = 0`, `p_single = 0`, `n_batch = 350`. Resource: 5.75 GB allocated / 6.84 GB reserved.
  An independent draw of 2 000 plans at `p_s = 0` and 500 at `p_s = 1` gave `regimes =
  {bi_visual}` and `{single}` respectively with `total_tokens = {350}` in both.
- **End-to-end neutrality A/B, same binary and same caps.** `stage2_budget_3stream_curriculum`
  (the ADR-005 arm) vs `stage2_budget_3regime_single_modality` with `--p_single_modality 0.0`,
  both at `--virtual_sensing_warmup_epochs 0`: **identical** epoch loss `5.482074737548828`,
  identical per-stream `mse_*`, identical `kept_*` / `kept_units_*` (100.0 / 212.5 / 37.5 tokens,
  4.0 / 8.5 / 37.5 units), identical validation loss `5.091627`, and `is_single = 0` throughout.
  `grad_norm` differs only in the 8th significant digit (`26.390928…` vs `26.390929…`), i.e. GPU
  reduction order, not policy.
- **Manifest neutrality, checked on the artefacts.** The ADR-005 config's `masking.json` `policy`
  block carries **no** ADR-006 key; both ADR-006 runs carry all three. That is the guard of §4.4
  working: an in-flight ADR-005 run resumed on this code compares equal key-for-key and is not
  rejected.
- **Production geometry, data-free smoke.** Built the model from
  `configs/pretrain/stage2_budget_3regime_single_modality.yaml` and ran one forward on a synthetic
  batch with a pinned plan: `[mask] … p_single=0.2 p_s_warm=0 E_warm=50 S_v=14
  units=[('tir',36,25),('rgb2d',36,25),('resp',250,1)]`; the pinned allocation
  `{'tir': 350, 'rgb2d': 0, 'resp': 0}` resolved to `regime=single carrier=tir`; the forward
  returned `[B, 350, D]` with a finite loss and both zero-token streams still contributing a loss
  term. This is G5/G6/G7/G8 at the real 96 px geometry; the §3.3 run over real clips, with its
  VRAM measurement, is the entry above.

## 7. Phase 6 — measurement (not this plan)

The fold protocol, the fold orchestration and the Tier-1 comparison against the ADR-005 arm belong to
`code/notes/plan/experiments/`. Measured numbers land in `code/notes/eval/`. Metric definitions are
fixed by `code/notes/design/Evaluation_RespWaveform_protocol.md`. The ADR-005 comparator is the arm
already measured at `code/notes/eval/baselines/tir_rgb2d_resp/`, whose Stage-3 fine-tune does not yet
exist and must be launched with the identical recipe before the Tier-1 gate can be read.

## 8. Traps

1. **Consuming a draw on the disabled path.** Writing `if uniform(g) < p_s` with `p_s = 0.0` still
   consumes a sample and shifts every later draw of the run — the counts change, G1 fails, and worse,
   any in-flight ADR-005 job resumed on the new code replays a different mask sequence. The guard must
   be `p_s > 0.0` *before* the draw.
2. **Redefining `mode` or `mode_by_counts`.** Regime 3 is a refinement of Mode 1. Relabelling Mode 1
   would flip ≈ 0.8 % of inpainting draws and break the ADR-005 assertions for every existing run.
3. **Reusing `_split_tubes` for the carrier.** It clamps to `[m, M]` and would silently return
   `9 / 5` instead of `14 / 0`. The single-carrier branch must build the plan directly.
4. **Leaving `_plan`'s per-stream invariant loop un-branched.** It asserts `kept ≤ video_cap_units`
   for every visual stream, so a correct 14-tube plan raises at the first Regime-3 step — in
   production, not in a test, because no existing test samples `p_s > 0`.
5. **Deriving the regime from the counts inside `_plan`.** A 3A plan and a mis-specified two-stream
   plan have the same counts; the regime-specific invariant must be given, not inferred.
6. **Assuming the eval pin can express single-modality.** `pin_plan_from_flags` clamps to the floor
   and cap and then fills the residual; a 350/0 request silently becomes 225/125 (§4.5).
7. **A non-neutral default for `single_carrier_tubes`.** Anything but `0` (derive) or `14` (this
   arm) must be rejected; a "helpful" default of `14` in the signature would make the option look
   mandatory in every config.
8. **Resume determinism.** The apportioner is not an `nn.Module` and is not in the checkpoint. A run
   resumed with different masking keys produces a different sequence; `masking.json` (§4.4) is the
   only detector, which is why the three keys must be in its stable fields.
