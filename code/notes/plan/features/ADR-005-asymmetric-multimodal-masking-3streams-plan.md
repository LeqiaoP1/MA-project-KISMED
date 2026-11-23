# ADR-005 — three-stream curriculum masking implementation plan

> **Authority.** `code/notes/design/ADR-005-asymmetric-multimodal-masking-3streams.md` is the
> contract; this document only says how it is built and in what order. Where the two disagree,
> the ADR wins and this plan is wrong.
>
> **Scope.** Adds the virtual-sensing curriculum (§2.7), the resolved-mode label (contract 4), the
> explicit kernel designation (I2) and the observability of §2.8 to the existing constant-budget
> policy, and introduces the three-stream arm configuration. Model architecture, tokenization,
> data preprocessing, the Stage-3 recipe and the evaluation protocol stay out of scope.
>
> **Parallel-operation requirement.** The ADR-002 arm must keep running unchanged while this lands.
> The implementation is therefore **additive and neutral**: no module is forked, no existing
> parameter changes meaning, and a configuration that does not mention a new key behaves exactly as
> it did before. "Parallel" is achieved by *defaults*, not by a second implementation — a forked
> engine would drift from the invariants and the 10⁴-draw suite would guard only one branch.
>
> **State (2026-11-23).** Nothing implemented. Phases 1–3 are additive code; Phase 4 is the arm
> configuration; Phase 5 is measurement and belongs to `code/notes/plan/experiments/`.
>
> **Judged by** ADR-005 §3: Criterion 1 (policy conformance) is a gate for this plan; Criterion 2
> (the subject-level headline) is the *arm's* gate and is recorded in `code/notes/eval/`, never
> here.

---

## 1. Definition of done

1. With no ADR-005 key set, the ADR-002 arm produces the **bit-identical** mask sequence it
   produced before the change, for a fixed `(mask_base, seed, step)`.
2. `p_at(epoch)` implements ADR-005 §2.7, is monotone in the epoch, and is validated at
   construction rather than at the first step.
3. Validation evaluates at the phase's own `p`, with the mask RNG sequence unchanged.
4. The resolved mode of contract 4 is observable **without** changing the meaning of the existing
   `mode` label — `MultiModalMAE.forward` returns `mode_resolved` and `p_virtual` in its payload
   next to `mode`, and the engine publishes the Mode-1 fraction they imply (§2.8).
5. The pre-trained kernel recipient (I2) is designated explicitly and recorded with the run.
6. Every record required by ADR-005 §2.8 is emitted, and a manifest captures the parameter tuple.
7. The three-stream curriculum configuration exists, resolves through `_check_config_keys`, and
   reaches its first step with `[B, 350, D]`.
8. Gates G1–G8 pass; `code/tests/test_budget_masking.py` and `code/tests/test_mask_placement.py`
   are **unchanged** and green — that is the parallel-operation proof.

## 2. What already satisfies ADR-005 (nothing to build)

Verified against the code before planning; this is why no fork is needed.

| ADR-005 requirement | where it already holds |
| :--- | :--- |
| exactly two visual + one physiological stream | `ConstantBudgetApportioner._validate` in `code/core/budget_masking.py`; covered by `test_single_physiological_stream_stays_aligned_and_inside_the_cap` |
| the counting / alignment / placement arithmetic (§§2.3–2.5) | `code/core/budget_masking.py` + `code/core/mask_placement.py` |
| `tubelet (2,16,16)` | already the default of `--tubelet`; `--temporal_stride 4`, `--input_size 96`, `--fs 1000`, `--sig_kernel 32` are values, not code |
| `T = 25`, `R = 10` space–time alignment | `MultiModalMAE._check_space_time_alignment` |
| contract 7 conditioning (mechanism (a), prefixing) | `MultiModalMAE.forward` |
| I1 verbatim kernel, rejection on mismatch | `MultiModalMAE.fit_visual_patch_embed` |
| per-step counts / per-sample placement, replay determinism | `code/core/mask_placement.py::counts_generator` / `placement_generator` |
| `kept_<stream>`, `mse_<stream>`, `mode`, `n_batch` logging | `code/engines/pretrain.py::train_one_epoch` |

## 3. Phase 1 — neutral plumbing (the only hot-path change)

The four edits below are the whole risk surface. Each has a default that reproduces today's
behaviour exactly.

**3.1 `code/core/budget_masking.py`**

- `ConstantBudgetApportioner.__init__` gains `virtual_sensing_warmup_p: float = 0.10` and
  `virtual_sensing_warmup_epochs: int = 0` — the **exact** names of ADR-005 §2.9, so the YAML key
  and the Python keyword are identical and `parser.set_defaults(**cfg)` reaches the apportioner
  without a translation layer. (`p_virtual_sensing_warmup` appears nowhere.)
- `_validate` adds: `0 ≤ p_warm ≤ p_target`; `E_warm ≥ 0`. (`E_warm = 0` must be legal — it is the
  compatibility default and disables the curriculum.)
- New `p_at(epoch: Optional[int]) -> float` — ADR-005 §2.7, on the **0-based** index:
  `p_target if epoch is None or epoch >= E_warm else p_warm`.
- `sample_step(generator=None, epoch=None)` uses `self.p_at(epoch)` instead of
  `self.p_virtual_sensing` directly. Positional/keyword compatibility is preserved because
  `generator` stays first and `epoch` is appended as a keyword.

**3.2 `code/core/multimae.py`**

- `MultiModalMAE.__init__` gains `virtual_sensing_warmup_p: float = 0.10` and
  `virtual_sensing_warmup_epochs: int = 0`, stores them and passes both to the
  `ConstantBudgetApportioner` it builds at lines 372–377. Without this the apportioner keeps the
  `0` signature default and the YAML key is inert.
- `build_pretraining_model(args)` (lines 874–968) reads both keys off `args` with
  `getattr(args, ..., default)` — exactly the pattern of the seven ADR-002 keys it already
  forwards — and adds them to the `MultiModalMAE(...)` call. Extend the `[mask]` provenance print
  with the resolved `(p_warm, E_warm)`.
- `make_masks(..., epoch: Optional[int] = None)` and `forward(..., epoch: Optional[int] = None)`
  forward `epoch` to `apportioner.sample_step`. `plan=None` and `epoch=None` remain the defaults,
  so every existing caller (evaluation pins, `run_eval_recon.py`, `run_visualize_pretrain.py`, the
  unit-test helpers) is untouched.
- `forward`'s returned dictionary (lines 854–865) gains two keys next to `mode`:
  `'mode_resolved': int(plan.mode_by_counts)` and
  `'p_virtual': float(self.apportioner.p_at(epoch))`. `plan` and `epoch` are both already in scope
  there. Without these the engine cannot log §2.8's resolved mode or the active `p(e)`.

**3.3 `code/engines/pretrain.py`**

- `train_one_epoch` already receives `epoch` — pass it as `model(..., epoch=epoch)`.
- `evaluate_pretrain(model, data_loader, device, *, mask_base)` gains `epoch: int = 0` and passes
  it to the model. **The mask RNG must not move:** keep `counts_generator(seed, 0, batch_idx)` with
  `seed = val_mask_seed(mask_base, batch_idx)` exactly as it is; only the *threshold* `p` changes
  with the epoch. Its docstring line "the same sequence of modes, counts and placements every
  epoch" becomes "…every epoch **within a phase**".

**3.4 `code/runners/run_pretrain.py`**

- Register `--virtual_sensing_warmup_p` (float, default `0.10`) and
  `--virtual_sensing_warmup_epochs` (int, default `0`) in `get_args()` beside `--p_virtual_sensing`
  (line 321). `code/runners/_common.py::_check_config_keys` fails with `SystemExit` on any YAML key
  that is not a parser `dest`, so the config of Phase 3 is unusable until these exist. `_common.py`
  itself needs no edit.
- Pass `epoch=epoch` at the validation call (line 717):
  `evaluate_pretrain(model, data_loader_val, device, mask_base=mask_base, epoch=epoch)`. The epoch
  is in the loop scope but is **not** currently forwarded, so validation would otherwise pin at
  `epoch=0` forever (Trap 2).
- No other change here; Phase 2.3 adds the `--patch_stream` flag on the same parser.

## 4. Phase 2 — observability, designation, contract 4

**4.1 `BudgetPlan` gains `mode_by_counts`** (`code/core/budget_masking.py`) — a fourth frozen
field `mode_by_counts: int`, equal to `MODE_VIRTUAL_SENSING` iff every declared signal stream keeps
0 tokens, else `MODE_INPAINTING`; `__post_init__` validates it against `MODES`. `BudgetPlan`
currently carries no stream kinds, so it cannot evaluate the rule itself: `_plan` computes the value
from `self.signal` and passes it in (preferred over adding a `signal_streams` field, since `_plan`
is the single construction site — lines 315–317 — and `plan_from_counts` already derives the same
label from its inferred `mode`). `mode` keeps its current meaning ("the sampled mode"), so:

- existing tests (`test_ten_thousand_draws_hold_the_exact_sum`,
  `test_probability_extremes_are_absorbing`) keep asserting the draw and stay untouched;
- ADR-005 contract 4 is satisfied by logging `mode_by_counts` as a **second** key.

Do **not** redefine `mode`: relabelling would flip ≈ 0.8 % of Mode-2 draws (the `K̃_resp = 0` case)
and break those two assertions for every ADR-002 run as well.

**4.2 `code/engines/pretrain.py` logging** — inside the existing `if isinstance(out, dict):` block
of `train_one_epoch` (lines 126–141), add per step: `p_virtual` (the resolved `p(e)`),
`kept_units_<stream>` (already in the forward payload as `kept_units`, just unpublished) and
`mode_resolved` (`plan.mode_by_counts`). Key set stays a function of the configuration, so the DDP
collective pattern of contract 9 is preserved.

Also add the contract-4 indicator `is_virtual = 1 if out['mode_resolved'] == MODE_VIRTUAL_SENSING
else 0`. §2.8's **per-epoch** requirement is the *Mode-1 fraction*, and `metric_logger` averages
whatever it is given: the mean of `mode ∈ {1, 2}` is `2 − f` (≈ 1.5), not `f`. The mean of
`is_virtual` is exactly `f`, which must land in `0.10 ± 0.02` / `0.50 ± 0.02` (Criterion 1.4).
`mode`, `mode_resolved` and `is_virtual` are all logged; none of them is redefined.

**4.3 `--patch_stream`** (`code/runners/run_pretrain.py::get_args`, default `''`) — thread it to the
call site in `run_pretrain.py::init_stage1_encoder` (currently line 507):
`load_pretrained_encoder(model, ckpt, patch_stream=args.patch_stream)`. The default reproduces
today's adapter-ordering resolution, so no existing run changes; the ADR-005 arm sets `tir`
explicitly (I2). The flag must exist on the parser before the config of Phase 3 is valid
(`_check_config_keys`).

**4.4 Run manifest** — write `masking.json` into `args.output_dir` before the first step, guarded by
`if is_main_process():` (`code/utils/dist.py`) so the ranks never race on one path, carrying the
ADR-005 §2.9 tuple, `physio_placement`, `physio_span_tokens`, the conditioning mechanism, the
kernel recipient, `mask_base`, `epochs` and the resolved `(p_warm, p_target, E_warm)`. It is the
only record that makes a resumed run's mask sequence reproducible, because the apportioner is not
an `nn.Module` and therefore is **not** inside the checkpoint.

## 5. Phase 3 — the arm configuration

New file `code/configs/pretrain/stage2_budget_3stream_curriculum.yaml` *(new)*, derived from
`code/configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml`. Delta only:

| key | base value | ADR-005 value | why |
| :--- | :--- | :--- | :--- |
| `tubelet` | `4,16,16` | `2,16,16` | I1: the pre-trained kernel is `(2,16,16)`; the base value is infeasible |
| `temporal_stride` | `2` | `4` | keeps `T = 25` at the new tubelet (ADR-005 §2.2) |
| `virtual_sensing_warmup_p` | — | `0.10` | §2.7 |
| `virtual_sensing_warmup_epochs` | — | `50` | §2.7 |
| `p_virtual_sensing` | `0.50` | `0.50` | unchanged; now means "from `E_warm` onwards" |
| `epochs` | `150` | `150` | unchanged; Phase II is the remaining 100 |
| `patch_stream` | — | `tir` | I2 |
| `streams`, `n_budget`, caps, floor, `input_size`, `fs`, `sig_kernel`, `clip_duration`, `fps` | — | unchanged | already the ADR-005 geometry |

Two operational notes for the header comment: the two ROI caches and the respiration cache stay
valid (their keys carry clip length, fps and fs, not the stride), and `--warmup_epochs` must stay
below `epochs` for the cosine schedule.

## 6. Phase 4 — tests

`code/tests/test_budget_masking.py` and `code/tests/test_mask_placement.py` are **not modified**
(G8). New: `code/tests/test_curriculum_schedule.py` *(new)*.

| gate | assertion |
| :--- | :--- |
| **G1** | With `E_warm = 0` (and with the new keys absent), the kept-count sequence over 50 000 draws is identical to a reference sequence captured from the pre-change engine at the same `(mask_base, epoch, step)`. Captured once, committed as a fixture. |
| **G2** | `p_at(e) == p_target` for every `e ≥ E_warm`, `p_at(None) == p_target`, `p_at(e) == p_warm` for `e < E_warm`, and `p_at` is non-decreasing in `e` (R6). |
| **G3** | Construction rejects `p_warm > p_target` and `E_warm < 0`; `E_warm = 0` is accepted and yields `p_at(e) == p_target` for all `e`. |
| **G4** | Over 10⁴ draws at `E_warm > 0`, the Mode-1 fraction (from `mode`, and separately from `mode_by_counts`) is `p_warm ± 4σ` below the boundary and `p_target ± 4σ` above it. |
| **G5** | `mode_by_counts` equals `MODE_VIRTUAL_SENSING` iff all signal counts are 0; the `K̃ = 0` draw is labelled Mode 1 by `mode_by_counts` and Mode 2 by `mode` (contract 4 vs the draw). |
| **G6** | Model-level: a forward with `epoch=0` and with `epoch=E_warm` on the same batch and the same `counts_generator` yields plans with the expected mode fractions; `[B, 350, D]` holds in both. |
| **G7** | Gradient path: with `K_resp = 0`, `∂L_resp/∂z ≠ 0` (contract 7, ADR-005 §3.1(6)). |
| **G8** | The two existing suites pass unmodified, and a 1-epoch smoke of the ADR-002 config reproduces its pre-change loss to the last printed digit. |

```text
cd code
python -m pytest tests/test_budget_masking.py tests/test_mask_placement.py \
                 tests/test_curriculum_schedule.py -q

# G8 — ADR-002 regression: the untouched arm must reproduce its pre-change loss.
python -u runners/run_pretrain.py -c configs/pretrain/stage2_budget_tir_roi_resp_rgb2d.yaml \
    --exp_root /tmp/adr005_regress --fold 0 --dev_smoke --epochs 1 --warmup_epochs 0 \
    --max_clips 100 --batch_size 8 --num_workers 0 --output_dir /tmp/adr005_regress/out \
    --roi_cache <ROI_CACHE> --rgb2d_cache <RGB2D_CACHE> --resp_cache <RESP_CACHE>

# Criterion 3.3 — ADR-005 smoke: prints 36 tubes, 250 signal tokens, T=25, R=10
# under tubelet (2,16,16) / temporal_stride 4, and emits every §2.8 record.
python -u runners/run_pretrain.py -c configs/pretrain/stage2_budget_3stream_curriculum.yaml \
    --exp_root /tmp/adr005_smoke --fold 0 --dev_smoke --epochs 1 --warmup_epochs 0 \
    --virtual_sensing_warmup_epochs 0 \
    --max_clips 100 --batch_size 8 --num_workers 0 --output_dir /tmp/adr005_smoke/out \
    --roi_cache <ROI_CACHE> --rgb2d_cache <RGB2D_CACHE> --resp_cache <RESP_CACHE>
```

Both smokes use `--warmup_epochs 0` (the *learning-rate* schedule); the curriculum is disabled on
the second one with `--virtual_sensing_warmup_epochs 0`, which is the ADR §3.3 "warm-up disabled"
variant. The first command must be byte-identical in output to a run of the same config from the
pre-change commit.

### 6.1 BDD-style acceptance test scenarios

The scenarios below specify the behavioural contracts to be implemented in
`code/tests/test_curriculum_schedule.py` and runner verification tests.

```gherkin
Feature: Three-Stream Constant-Budget Masking with Virtual-Sensing Curriculum (ADR-005)

  Background:
    Given a declared stream set with two visual streams (tir, rgb2d) and one signal stream (resp)
    And a total budget N = 350 visible tokens with visual unit size T = 25 tokens
    And visual cap c_v = 0.25 (M = 9 tubes), visual floor m = 2 tubes, physio cap c_p = 0.50 (C = 125 tokens)

  Scenario: Neutral backwards-compatible count sequence for ADR-002 runs (Gate G1, DoD 1)
    Given an apportioner initialized with virtual_sensing_warmup_epochs = 0
    When 50,000 steps are sampled with fixed (mask_base, epoch, step) generators
    Then the generated sequence of kept token counts is bit-identical to the pre-change baseline fixture
    And the visible token count sum(K_m) equals 350 on every step.

  Scenario: Curriculum schedule resolution and monotonicity across epochs (Gate G2, ADR §2.7)
    Given an apportioner configured with virtual_sensing_warmup_p = 0.10, p_virtual_sensing = 0.50, and virtual_sensing_warmup_epochs = 50
    When p_at(epoch) is evaluated for epoch indices from 0 to 100 and for epoch is None
    Then p_at(epoch) evaluates to exactly 0.10 for all epoch < 50 (Phase I — grounding)
    And p_at(epoch) evaluates to exactly 0.50 for all epoch >= 50 (Phase II — deployment)
    And p_at(None) evaluates to 0.50 (un-epoched evaluation fallback)
    And p_at(epoch) is monotonically non-decreasing over all epoch >= 0.

  Scenario: Construction-time parameter validation and feasibility bounds (Gate G3, ADR §2.5, §2.7)
    Given invalid curriculum parameters where virtual_sensing_warmup_p > p_virtual_sensing or virtual_sensing_warmup_epochs < 0
    When ConstantBudgetApportioner is constructed
    Then construction raises ValueError naming the failing bound before any training step
    Given valid baseline parameters with virtual_sensing_warmup_epochs = 0
    When ConstantBudgetApportioner is constructed
    Then construction succeeds and p_at(epoch) evaluates to p_virtual_sensing for all epoch.

  Scenario: Statistical mode fractions match target probabilities across phase boundaries (Gate G4, ADR §3.1.4)
    Given an apportioner with virtual_sensing_warmup_p = 0.10, p_virtual_sensing = 0.50, and virtual_sensing_warmup_epochs = 50
    When 10,000 steps are sampled at epoch = 10 (Phase I)
    Then the empirical Mode-1 fraction from mode is within 0.10 ± 4σ (0.10 ± 0.012)
    And the empirical Mode-1 fraction from mode_by_counts is within (0.10 + 0.90 * 1/126) ± 4σ
    When 10,000 steps are sampled at epoch = 60 (Phase II)
    Then the empirical Mode-1 fraction from mode is within 0.50 ± 4σ (0.50 ± 0.020)
    And the empirical Mode-1 fraction from mode_by_counts is within (0.50 + 0.50 * 1/126) ± 4σ.

  Scenario: Contract 4 mode classification and resolution of Mode-2 K_tilde = 0 draw (Gate G5, ADR §2.3 Contract 4)
    Given a sampled step where the inpainting draw (Mode 2) produces K_tilde_resp = 0
    When the resulting BudgetPlan is created
    Then plan.mode remains MODE_INPAINTING (preserving sampled draw identity)
    And plan.mode_by_counts evaluates to MODE_VIRTUAL_SENSING (satisfying Contract 4)
    Given a sampled step where the physiological stream has kept_tokens > 0
    When the resulting BudgetPlan is created
    Then plan.mode_by_counts evaluates to MODE_INPAINTING.

  Scenario: Deterministic within-phase validation mask sequence and RNG stability (Plan §3.3, ADR §2.7)
    Given evaluate_pretrain called with fixed mask_base and a deterministic validation loader
    When validation is executed at epoch = 1 and at epoch = 2 (both in Phase I, p = 0.10)
    Then the drawn mode, stream token counts, and spatial tube placements across all batches are identical bit-for-bit
    When validation is executed at epoch = 50 (Phase II, p = 0.50)
    Then the underlying PRNG sequence matches epoch 1
    And all validation steps with uniform draw u < 0.10 remain Mode 1 with identical visual tube splits
    And steps with 0.10 <= u < 0.50 transition deterministically from Mode 2 to Mode 1.

  Scenario: Model forward diagnostics and latent gradient path under complete sensor loss (Gate G6, G7, ADR §2.3 Contract 7, §3.1.6)
    Given a MultiModalMAE instance with curriculum apportioner and a multimodal input batch of shape [B, ...]
    When forward(x, epoch=epoch) is executed on a step where K_resp = 0
    Then the combined visible token sequence has shape [B, 350, D]
    And the forward output dictionary contains mode_resolved == MODE_VIRTUAL_SENSING and p_virtual == p_at(epoch)
    When the respiration loss L_resp is backpropagated
    Then the gradient with respect to the encoder latent representations ∂L_resp/∂z is non-zero (satisfying Contract 7 prefixing).

  Scenario: Config resolution, CLI flag registration, and manifest creation (Plan §3.4, §4.3, §4.4)
    Given the configuration configs/pretrain/stage2_budget_3stream_curriculum.yaml
    When parse_args_with_config resolves the CLI arguments
    Then _check_config_keys accepts virtual_sensing_warmup_p, virtual_sensing_warmup_epochs, and patch_stream without error
    And patch_stream resolves to 'tir'
    And masking.json is written to output_dir containing the §2.9 tuple, placement, conditioning, and recipient
    And multi-rank distributed launches guard the file write with is_main_process().

  Scenario: VideoMAE patch embed kernel compatibility and designated routing (ADR §2.6 I1, I2)
    Given a model with tubelet (2, 16, 16) and a VideoMAE ViT-Base pretrained checkpoint
    When load_pretrained_encoder is called with patch_stream = 'tir'
    Then the Conv3d(3, D, (2, 16, 16)) weight tensor is loaded verbatim into adapters.tir.patch_embed.weight
    And adapters.rgb2d.patch_embed.weight remains randomly initialized
    Given a model configured with an incompatible tubelet (4, 16, 16)
    When load_pretrained_encoder is called
    Then fit_visual_patch_embed raises ValueError rejecting the tubelet mismatch (satisfying I1).
```

## 7. Phase 5 — measurement (not this plan)

The eight-fold protocol, the fold orchestration and the headline comparison belong to
`code/notes/plan/experiments/`. The comparison arm is the retired four-stream run described in
`code/notes/plan/experiments/exp_20261121_fourstream_vits_random.md`; measured numbers land in
`code/notes/eval/`. Metric definitions are fixed by
`code/notes/design/Evaluation_RespWaveform_protocol.md`.

## 8. Traps

1. **Non-neutral defaults.** If `virtual_sensing_warmup_epochs` defaults to anything but `0`, every
   existing configuration silently acquires a curriculum the moment the code lands. The ADR's `50`
   is the *shipped arm's* value and lives in the config, not in the signature.
2. **Validation epoch.** `evaluate_pretrain` currently hardcodes the epoch slot at `0`, and its
   call site (line 717) does not forward the loop's `epoch`. Without both halves of §3.3/§3.4,
   validation evaluates every epoch at `p_warm`, so the tracked curve never reflects the phase
   after the boundary.
3. **Mutating `p_virtual_sensing` per epoch.** Rejected: it bypasses the `(0, 1]` validation and
   makes the plan depend on mutable state, breaking replay from `(seed, epoch, step)`.
4. **Resume determinism.** The apportioner is not checkpointed. A run resumed with different
   masking keys produces a different mask sequence; the manifest of Phase 2.4 is what makes the
   mismatch detectable.
5. **Relabelling `mode`.** See §4.1 — it would change ADR-002 logs and two existing assertions.
6. **Config/parser coupling.** `code/runners/_common.py::_check_config_keys` rejects unknown YAML
   keys with `SystemExit`, so all three new keys (`virtual_sensing_warmup_p`,
   `virtual_sensing_warmup_epochs`, `patch_stream`) must be registered in
   `runners/run_pretrain.py::get_args` in the same change as the config. A missing flag is not a
   silent no-op — it aborts the run before step 0 — which is why the YAML name and the Python
   keyword must be spelled identically.
7. **Half-plumbed warm-up.** Adding the parameters to `ConstantBudgetApportioner` without threading
   them through `MultiModalMAE.__init__` and `build_pretraining_model` leaves the apportioner at
   `E_warm = 0`: the run looks healthy, logs `p(e) = 0.50` throughout, and silently drops the
   curriculum. The `[mask]` provenance print must therefore show the resolved `(p_warm, E_warm)`.
8. **Mean of `mode` is not the Mode-1 fraction.** See §4.2 — `mode` averages to `2 − f`; the
   per-epoch Criterion-1.4 check needs the `is_virtual` indicator.

## 9. Non-goals

- Migrating, rewriting or deleting the ADR-002 configuration — it stays as the parallel arm.
- Any change to the counting, alignment or placement arithmetic.
- Re-opening the kernel-routing decision (I2 defaults to `tir`; a second-visual-stream study needs
  its own ADR).
- Stage-3 recipe changes, evaluation changes, or any headline claim.
