# ADR-017 — Gap-controlled masked-span respiration placement: implementation plan

> **Authority.** [`ADR-017-masked-span-physio-placement.md`](../../design/ADR-017-masked-span-physio-placement.md)
> is the contract; this plan only says **how** it is built and **in what order**. Where the two
> disagree, the ADR wins.
>
> **Scope.** A third 1-D placement (`masked_span`) in the placement module, its two gap-bound knobs,
> construction-time feasibility against the apportioner's Mode-2 counts, runner/manifest/resume
> wiring, a matched inpainting probe in the reconstruction evaluator, a no-video floor probe for
> Criterion 3a, one arm config, and the Criterion-1 test suite.
> **Out of scope:** counts, modes and curriculum (ADR-007), the objective (ADR-008), the Stage-3 recipe,
> the run protocol (`code/notes/plan/experiments/`) and measured outcomes (`code/notes/eval/`).
>
> **State (2026-11-27).** Proposed — no phase started. Revised the same day after ADR/plan reviews:
> the count set is derived from the apportioner rather than transcribed, the kept-block sampler and
> the `M = 0` state are pinned, the single-signal guard is aligned with ADR-017 §1.1 / I6, the
> manifest writes are neutrality-guarded and JSON round-trip safe, the probe path clears stale
> attributes and validates pinned counts, the gap-bound sweep (up to `g_max = 150`, excluding global
> `n ≡ 1`) sits in Phase 5, and BDD scenarios are appended in §6.

---

## 1. Why this is additive

- `scattered` (default) and legacy `span` keep their code paths and generator consumption unchanged,
  so every existing config, checkpoint and archived probe reproduces bit-for-bit.
- Counts come from `ConstantBudgetApportioner` exactly as today; only the index draw for the
  respiration stream changes, and only when `physio_placement == 'masked_span'`.
- `place_signal` already returns early for `k == 0` before touching the generator, which is what
  ADR-017 I4 requires. Two different placements follow, and they are not in conflict: the
  *validation* of the gap bounds runs **before** that return (so a bad tuple fails on step 1 rather
  than on the run's first Mode-2 step), and the `masked_span` *placement branch* sits **after** it.

## 2. Definition of done

1. ADR-017 §3 Criterion 1 passes in `code/tests/test_adr017_masked_span_placement.py`.
2. The full pre-existing test suite passes with **one** intended edit: the new arm config is
   registered in `code/tests/test_task_group_configs.py::TIR_ROI_CONFIGS` (that test fails by design
   on an unregistered `task_groups:` config). Everything else — in particular
   `test_mask_placement.py`, `test_two_stream_anatomy_masking.py`, `test_budget_masking.py`,
   `test_adr008_*`, `test_config_loading.py` (which globs every shipped YAML) and
   `test_notes_adr_governance.py` — stays unchanged.
3. Criterion 2 smoke completes with the arm config.
4. The Criterion-3a floor probe exists and its fold-0 validation report is written under
   `code/notes/eval/` **before** any GPU training run is submitted.
5. **Hand-off:** Criteria 3b and 3c are GPU runs and are not this plan's deliverables. They are
   executed by an experiment protocol note under `code/notes/plan/experiments/` (AGENTS.md §3),
   authored from this plan once Phase 6 lands; that note owns the fold-0 `masked_span` run and the
   five-fold Stage-3 gate.

## 3. Phases

### Phase 0 — pre-flight (no code)

- Re-run the ADR-017 §1.2 P2 table with the same four columns, so the comparison is 1:1:

  ```bash
  cd code && python - <<'EOF'
  import torch
  from core.mask_placement import place_signal
  N, TOK = 250, 0.032
  for K in (25, 50, 75, 100):
      for kw in (dict(mode='scattered'), dict(mode='span', span_tokens=25),
                 dict(mode='span', span_tokens=125)):
          idx = place_signal(K, N, generator=torch.Generator().manual_seed(0),
                             batch_size=400, **kw)
          keep = torch.zeros(400, N, dtype=torch.bool); keep.scatter_(1, idx, True)
          gaps, longest = [], []
          for row in (~keep).int().tolist():
              run, best, runs = 0, 0, []
              for v in row + [0]:
                  if v: run += 1
                  elif run: runs.append(run); best = max(best, run); run = 0
              gaps.extend(runs); longest.append(best)
          g = torch.tensor(gaps, dtype=torch.float) * TOK
          L = torch.tensor(longest, dtype=torch.float) * TOK
          print(K, kw, f'median gap {g.median():.2f}s, <1s {(g < 1).float().mean():.3f}, '
                       f'median longest {L.median():.2f}s')
  EOF
  ```

- **Enumerate, do not transcribe, the Mode-2 kept counts** the arm's apportioner can draw. The
  ADR's `{25, 50, 75, 100}` is an *outcome* of `_mode2`'s tube-modulus snap (`n_budget 200`,
  `tokens_per_unit 25`, `C_p 100`), not a constant; this script is a throwaway preview of the
  public `mode2_counts` Phase 2 adds.

  ```bash
  cd code && python - <<'EOF'
  import collections, torch
  from core.budget_masking import ConstantBudgetApportioner, StreamSpec, VISUAL, SIGNAL
  app = ConstantBudgetApportioner(
      [StreamSpec('tir', VISUAL, 36, tokens_per_unit=25, spatial_grid=(6, 6)),
       StreamSpec('resp', SIGNAL, 250, tokens_per_unit=1)],
      n_budget=200, video_cap_ratio=0.25, physio_cap_ratio=0.40, video_min_tubes=4,
      p_virtual_sensing=0.5, virtual_sensing_warmup_p=0.10,
      virtual_sensing_warmup_epochs=20, spatial_sampling='anatomy_aware',
      core_row_bounds=(2, 3), core_col_bounds=(2, 3), core_tubes_min=1,
      core_tubes_max=2, periph_min_tubes=1, physio_draw_min=1)
  g, seen = torch.Generator().manual_seed(0), collections.Counter()
  for _ in range(20000):
      kept = app.sample_step(generator=g, epoch=30).kept_tokens['resp']
      if kept: seen[kept] += 1
  print('Mode-2 resp counts:', dict(sorted(seen.items())))
  EOF
  ```

- Cross-check the feasibility arithmetic by hand against those counts: with `g_min = 63` and
  `g_max = 125`, `N(M) = {2}` for `M ∈ {150, 175}` and `{2, 3}` for `M ∈ {200, 225}` (ADR-017 §2.1),
  and `n - 1 <= K` never binds.

### Phase 1 — `code/core/mask_placement.py`

- Add `MASKED_SPAN = 'masked_span'`, extend `PLACEMENT_MODES` (widening its `Tuple[str, str]`
  annotation to `Tuple[str, ...]`), and update the `place_signal` bullet in the module docstring.
- Add `gap_min_tokens` / `gap_max_tokens` as **keyword** arguments to `place_signal`, default `0` so
  every existing call site is untouched. Validate the pairing **before** the `k == 0` early return,
  so a bad tuple cannot survive every Mode-1 step and only fail mid-run: `gap_{min,max}_tokens` set
  iff `mode == MASKED_SPAN` (with `1 <= g_min <= g_max <= n_units`), and `span_tokens` set iff
  `mode == SPAN` (ADR-017 §2.2 symmetric exclusivity; existing `test_mask_placement.py` passes
  `span_tokens=None` for `SCATTERED` and stays green).
- Handle `M == 0` (every token kept) before any draw: return `arange(n_units)` per sample and consume
  no random number (ADR-017 §2.1 step 1).
- Implement ADR-017 §2.1 steps 2–5 with a fixed batch-level draw order: per sample draw `n`,
  sequential gap excesses, and gap permutation; then draw one batched `torch.rand(b, K + 1)` tensor
  and `argsort` it for the stars-and-bars kept-block composition (per sample, the first `n` ranks
  sorted give the bar positions; because width `K + 1` is `n`-independent, this step stays
  vectorised even though `n` varies per sample). Restricted to one sample this *is* §2.1's order;
  across samples it pins an interleaving the ADR leaves open, so do not "simplify" it back to
  per-sample draws — that would change every index while passing every invariant. Draw on the CPU
  generator the engine already passes — `engines/pretrain.py` calls `placement_generator(...)`
  without a `device`, so it *is* a CPU generator; keep it that way, because a CUDA generator puts a
  synchronisation inside the per-sample loop — then `.to(device)` once.
- Add a pure helper `admissible_gap_counts(M, K, g_min, g_max) -> tuple[int, ...]` used by both the
  sampler and the feasibility check.
- Post-condition checks inside the function (cheap, raise `RuntimeError`): exact count, ascending
  unique indices, exactly `n` masked runs for the drawn `n`, every run within bounds.

### Phase 2 — `code/core/budget_masking.py` + `code/core/multimae.py`

- **`budget_masking.py` (why this phase names a second file):** add a public
  `ConstantBudgetApportioner.mode2_counts(stream: str) -> Tuple[int, ...]` returning the sorted kept
  counts a Mode-2 step can draw for that stream. Restrict it to single-signal apportioners
  (`len(self.signal) == 1`, otherwise `ValueError`), enforcing ADR-017 §1.1 / I6: the rule is
  defined on one physiological token axis, and when multiple physiological streams share the
  tube-modulus remainder (`test_budget_masking.py` pins that alignment), a *per-stream* Mode-2 count
  set is not solely determined by one stream's snap.
- Enumerate deterministically in $O(C_p)$ over `range(self.physio_draw_min, C_p + 1)` by **calling
  the pure snap helper this phase extracts from `_mode2`**, never by re-implementing it: the
  downward branch (`raw - (T - r)` when the up-snap would exceed `C_p`, pinned by
  `test_alignment_subtracts_when_no_stream_can_absorb`) is part of the snap. Drop `0` — a zero draw
  is the `_sensor_absent` fall-through, i.e. Mode 1, not Mode 2. Nothing is transcribed: the arm
  resolves to `(25, 50, 75, 100)` because `n_budget 200`, `tokens_per_unit 25`, `C_p 100`, and any
  of those changes moves the set (D4).
- Test the enumeration in `code/tests/test_adr017_masked_span_placement.py`, not in
  `test_budget_masking.py` (DoD 2 admits exactly one edit to the existing suite): compare it against
  a `sample_step` histogram over ≥ 2·10⁴ seeds, so a future change to `_mode2` cannot desynchronise
  it silently.
- **`multimae.py`:** the constructor accepts `physio_gap_min_tokens` / `physio_gap_max_tokens`
  (default `0`), enforces `len(self.signal) == 1` when `physio_placement == 'masked_span'`
  (ADR-017 §1.1 / I6), validates the enum/knob pairing in **both** directions as for
  `physio_span_tokens`, and checks `1 <= g_min <= g_max <= self.physio_tokens` (I6).
- Feasibility (ADR-017 I6): take `K` from `apportioner.mode2_counts`, form `M = n_signal - K`, and
  refuse construction if `admissible_gap_counts` is empty for any of them.
- Factor the whole check into one helper, e.g.
  `validate_physio_placement(placement, span_tokens, gap_min, gap_max, n_signal, mode2_counts)`, and
  have the constructor call it. Phase 4 needs the same function: the evaluator assigns
  `model.physio_placement` after construction, where no validation runs today.
- `make_masks` forwards the two bounds to `place_signal`.
- `build_pretraining_model` reads both keys from `args`; its `[mask] policy=…` banner gains
  `physio_placement`, both bounds, their resolved seconds (`tokens * sig_kernel / fs`) and `N(M)`
  per masked count. Add the geometry guard the ADR's §4 trade-off implies: the bounds are
  token-valued, so a `sig_kernel` / `fs` / clip-length change silently re-scales them — print a loud
  warning when the resolved `g_min` falls below the ~1.5 s floor implied by the 0.10–0.60 Hz
  passband.

### Phase 3 — `code/runners/run_pretrain.py`

- `--physio_placement` choices gain `masked_span`; add `--physio_gap_min_tokens` and
  `--physio_gap_max_tokens` (default `0`, help text names ADR-017).
- Manifest `masking` block records both bounds (and seconds) **only when non-neutral**. This is not
  cosmetic: `_stable()` compares the whole `policy` dict, so two unconditional new keys make every
  in-flight pre-ADR-017 run refuse to resume — the trap the ADR-006, ADR-007 and ADR-008 keys each
  hit and guard against in `write_masking_manifest`. Mirror that guarded write using plain
  JSON-native types (`int` bounds, `float` seconds; if `N(M)` is stored in `policy`, use `str(M)`
  keys and `list[int]` values so JSON reload does not turn `int`/`tuple` into `str`/`list` and
  break `_stable()` on resume); the resume guard then refuses a changed bound for free, through the
  same comparison. Add the counts to the `[mask] wrote …` banner too, so the log and the manifest
  agree.
- This phase must land **before** Phase 6: `test_config_loading.py` pushes every shipped YAML
  through the real parser, so the arm config cannot precede its CLI flags.

### Phase 4 — `code/runners/run_eval_recon.py`

- `--resp_probe` gains `masked_span`: reads `physio_gap_{min,max}_tokens` from the checkpoint's
  recorded args by default, with explicit override flags; caption records
  `placement=masked_span (gaps g_min..g_max tokens)`.
- Route the override through Phase 2's `validate_physio_placement`, and **explicitly zero all
  inactive placement attributes** whenever `pin_masking` overrides `model.physio_placement`
  (`model.physio_span_tokens = 0` when not `span`; `model.physio_gap_min_tokens =
  model.physio_gap_max_tokens = 0` when not `masked_span`). Today `run_eval_recon.py` leaves stale
  `model.physio_span_tokens` behind when switching to `scattered`, which would trip ADR-017's
  symmetric exclusivity check when probing a `span`- or `masked_span`-trained checkpoint under
  another probe mode.
- The probe's kept count is `--kept_resp` clamped to the physio cap, which is not restricted to
  `𝒦₂`; a pinned count the bounds cannot tile must be **refused**, naming the feasible kept counts
  and their `N(M)` (ADR-017 R6, as amended). Silent clamping would fabricate a placement the ADR does
  not define.
- Leave the default probe and the legacy `span` probe untouched (archived reports must reproduce).
- Add an opt-in `--interp_baseline` that scores, on the same masks per clip, the no-video linear
  interpolator (`np.interp`, constant-clamped to the nearest visible sample at clip edges when
  `b_0 = 0` or `b_n = 0`) **and** the constant predictor — Criterion 3a's pass condition needs the
  pair, and 3b's paired table is only interpretable next to the constant. Both are computed in the
  space the model is scored in (`physio_norm: zscore`, `target_norm` pinned `'none'`), so P3's
  "≈ 1.0 z²" reference transfers.

### Phase 5 — Criterion-3a floor probe

- New runner `code/runners/run_probe_masking_baselines.py` (the archived note's §8 promotion): it
  loads the fold's **validation** role and draws ≥ 5 placements per clip per `K`. The role comes
  from `protocol.load_plan(<partition>).roles['val']` — there is no existing shortcut:
  `_held_out_subjects()` returns `(test, train)` only and `run_eval_recon --split` offers
  `('test', 'corpus')`, so the val role needs new plumbing. Keep it on val: the gate informs the
  design, and ADR-001 reserves `test` for the five-fold headline.
- Report masked MSE / `r_masked` of the constant and `np.interp` linear interpolators (constant-edge
  clamped whenever `b_0 = 0` or `b_n = 0`, matching Phase 4 and Criterion 3a/3b) for `scattered`,
  legacy `span`, and `masked_span` in the z-scored target space the model is scored in, so
  constant-zero is the per-clip mean and P3's "≈ 1.0 z²" reference holds.
- **Sweep, do not assume.** Run the same probe over the feasible `(g_min, g_max)` region up to
  `g_max = 150` — excluding bounds that force `n ≡ 1` across **all** $M \in \mathcal{M}_2$
  (`g_max ≥ 225`, ADR-017 §2.3) — not only at 63/125, and report the interpolator/constant ratio
  per point. Criterion 3a can fail at 63/125 and the remedy is boxed in: at `M = 150` (`K = 100`),
  forcing `n ≥ 2` caps `g_min ≤ 75` (2.4 s). Allowing `g_max = 150` (4.8 s) opens two fallbacks:
  (a) `g_min ≤ 75, g_max = 150` gives `N(150) = {1, 2}` (mixed 1- and 2-gap draws), and
  (b) `75 < g_min ≤ 87, g_max = 150` forces `N(150) = {1}` at `M = 150` only (leaving 3.2 s of
  visible context while `M ∈ {175, 200, 225}` keep `n ≥ 2`; `g_min ≤ 87 = floor(175 / 2)` keeps
  `n = 2` feasible at `M = 175`). Including `g_max = 150` in the sweep turns the Phase-6 numbers into
  a single-pass measured outcome rather than a guess.
- Output JSON + a markdown note under `code/notes/eval/baselines/tir_resp/` (measured outcomes only).

### Phase 6 — arm configuration

- `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml`: copy of the ADR-008
  arm differing **only** in `physio_placement: masked_span`, `physio_gap_min_tokens: <g_min>`,
  `physio_gap_max_tokens: <g_max>`; the header states that delta *and* the Phase-5 report that chose
  those numbers. 63/125 are the ADR's hypothesis, adopted only if Criterion 3a passes there.
- If the sweep adopts `g_max = 150` at `K = 100`, state the resulting `N(150)` explicitly in the
  config header (`{1, 2}` if `g_min ≤ 75`, or `{1}` if `75 < g_min ≤ 87`): neither violates
  ADR-017 §2.3's rejected global `n ≡ 1` (`g_max ≥ 225`), because `M ∈ {175, 200, 225}` still draw
  `n ≥ 2` and `M = 150` still retains 3.2 s of visible context.
- Register the new file in `code/tests/test_task_group_configs.py::TIR_ROI_CONFIGS`.
  `test_tir_roi_config_inventory_is_complete` fails by design on an unregistered `task_groups:`
  config, so this edit is required by DoD 2 and belongs to this phase.

### Phase 7 — tests and gates

`code/tests/test_adr017_masked_span_placement.py`:

- I1–I3 over ≥ 10⁴ seeded draws for each `K` in
  `ConstantBudgetApportioner.mode2_counts('resp')` — derived, never transcribed as
  `{25, 50, 75, 100}` (D4);
- I4: generator state unchanged for `K = 0` **and** for `M = 0` (`K = n_units`);
- determinism replay and per-sample diversity, including across samples with different `n` (the
  kept-block draw's width is `n`-independent, so replay must not depend on `n`);
- I6 rejections — a multi-signal model (`len(signal) > 1`) under `masked_span`, inverted bounds,
  non-tiling bounds, gap knobs without `masked_span`, non-zero `physio_span_tokens` without `span`
  (for both `scattered` and `masked_span`), and `g_max > n_units`; plus the probe-path refusal of a
  pinned `K` outside `𝒦₂` that the bounds cannot tile, and a cross-mode override test confirming
  `pin_masking` clears stale placement attributes;
- I5: forward pass on a tiny `masked_span` model — gathered visible tokens all have `mask == 0`;
- regression: `scattered` and legacy `span` indices and generator state identical to a frozen
  reference draw (a small golden tensor plus generator-state equality, with the torch pin recorded —
  CPU `rand`/`argsort` order is not a documented cross-version contract);
- runner: manifest records the bounds when non-neutral and omits them when neutral; JSON dump +
  reload `--resume` with identical bounds succeeds (`_stable()` type-invariant), and resume with
  different bounds is refused.

### Phase 8 — terminology and spec refresh

- `code/notes/GLOSSARY.md` §3 *Placement*: add `masked_span` and the two knobs (the entry currently
  reads "`scattered` … `span` … or — under ADR-007 — `anatomy_aware`").
- `code/scripts/project/write_specs.py` **itself** carries the spec prose (the parameter-table row
  naming `physio_placement` / `physio_span_tokens` and the `place_video/place_signal` snippet), so
  regenerating alone writes the stale text back. Edit the generator, then run
  `python code/scripts/project/write_specs.py`.
- Update `code/scripts/project/mask_policy_report.py` to print gap statistics for `masked_span`.
- The same enum list is repeated in `code/README.md` (the 1-D placement section),
  `.github/prompts/new-experiment.prompt.md` (the Stage-2 key list),
  `code/notes/plan/experiments/exp_template.md`, and the ADR-006/ADR-007 parameter tables (which
  ADR-017 supersedes in part): update them, or add a pointer to ADR-017.
- Set ADR-017's `Status` (`Proposed` → `Implemented (Pending Eval)` once Criteria 1–2 and 3a are
  green; `Accepted` only after Criterion 3c passes) and the matching status column in
  `code/notes/design/README.md`. `test_notes_adr_governance.py` G3 checks the registry pairing but
  not the status, so a stale or premature status would go unnoticed.

## 4. Decisions taken while planning

- **D1 — New enum value, not a redefinition of `span`** (ADR-017 §2.3).
- **D2 — Token-valued knobs, seconds logged** (ADR-017 §2.2).
- **D3 — Gap count `n` drawn per sample**, not per step: only the kept count must be batch-uniform.
- **D4 — Feasibility derived from the apportioner**, so a future change to ADR-007's count set is
  re-checked automatically. This is why Phase 2 adds `mode2_counts` to `budget_masking.py` instead of
  calling `admissible_gap_counts` against a literal set.
- **D5 — Two different composition samplers, on purpose**: the masked side's split is a *procedure*
  (step 3, deliberately non-uniform), the kept side's is a *distribution* (step 4, stars-and-bars).
  Pinning both is what makes the training/probe match (R7) checkable at all.
- **D6 — The `M = 0` state returns every index and draws nothing**, rather than raising: a policy or
  pin whose cap is `N_s` (e.g. `--physio_cap_ratio 1.0`, or a `plan_from_counts` pin) reaches it,
  and "mask nothing" is a legal placement.
- **D7 — Manifest keys written only when non-neutral**, mirroring the ADR-006/007/008 rule, so this
  change cannot make an in-flight run un-resumable.

## 5. Risks / non-goals

- Per-sample Python loop in placement: acceptable at B = 16, and the legacy `span` branch already
  loops the same way; vectorise only if the ADR-017 resource bound (< 5 ms/step) is violated. The
  kept-block draw is vectorised by construction (its width `K + 1` does not depend on `n`).
- Criterion 3a is this plan's real risk, not the code: the arm's bounds are a hypothesis and the
  feasible region may contain no point where interpolation loses to the mean. Phase 5's sweep
  decides, and Phase 6 must not fix the numbers before it reports.
- Not in this plan: gap-length curriculum, any loss change, and Criteria 3b/3c themselves (see DoD 5).

---

## 6. BDD Test Cases

```gherkin
Feature: Gap-controlled masked-span 1-D placement (ADR-017 I1-I4)
  Every Mode-2 respiration mask hides tokens in contiguous gaps bounded by [g_min, g_max],
  leaving kept counts, Mode 1, and legacy placements untouched.

  Scenario: Every Mode-2 draw obeys exact count, gap bounds, and gap separation
    Given the ADR-007 single-signal apportioner with N_s = 250 and mode2_counts('resp') = (25, 50, 75, 100)
    And gap bounds gap_min_tokens = 63 and gap_max_tokens = 125
    When place_signal is called in mode "masked_span" for 10000 seeded samples per K in mode2_counts('resp')
    Then every row has exactly K unique ascending indices in [0, 250)
    And the number of maximal masked runs n lies in N(250 - K, K)
    And every maximal masked run has length in [63, 125] tokens
    And consecutive masked runs are separated by at least 1 kept token

  Scenario: Degenerate counts K = 0 and M = 0 consume no RNG state
    Given a seeded CPU placement generator and valid bounds gap_min_tokens = 63, gap_max_tokens = 125
    When place_signal is called in mode "masked_span" with K = 0
    Then it returns shape [B, 0] and the generator state is byte-identical before and after the call
    When place_signal is called in mode "masked_span" with K = N_s (M = 0)
    Then it returns arange(N_s) per row and the generator state is byte-identical before and after the call

  Scenario: Determinism replay and per-sample diversity hold across varying gap counts
    Given a batch of B = 16 clips with K = 50 (where N(200, 50) = {2, 3})
    When place_signal is called twice with the same (seed, epoch, step, rank) placement generator
    Then both calls return identical index tensors
    And the B = 16 rows in the batch are not identical and exhibit both n = 2 and n = 3 over repeated seeds

  Scenario: Neutral default and legacy span mode stay bit-identical to pre-ADR behaviour
    Given fixed CPU generator seeds and frozen pre-ADR reference index tensors
    When place_signal runs in mode "scattered" (gap bounds = 0) and in mode "span" (span_tokens = 10, gap bounds = 0)
    Then both index tensors and final generator states match the pre-ADR references bit-for-bit

Feature: Apportioner enumeration and startup feasibility (ADR-017 I6)
  Infeasible bounds, stray knobs, or multi-signal configs fail loudly before step 0.

  Scenario: mode2_counts matches Monte Carlo draws from sample_step
    Given a single-signal ConstantBudgetApportioner with any valid ADR-007 tuple
    When mode2_counts('resp') is compared against 20000 sample_step draws (excluding K = 0)
    Then the deterministic enumeration equals the exact set of observed Mode-2 counts

  Scenario: Invalid placement tuples are rejected at construction and before K = 0 exit
    Given a MultiModalMAE build or a direct place_signal call (even at K = 0)
    When any of the following is passed:
      | violation | example |
      | multi-signal model with masked_span | streams = "tir,bp,resp" |
      | inverted gap bounds | gap_min = 126, gap_max = 125 |
      | gap_max exceeding N_s | gap_min = 63, gap_max = 251 |
      | non-tiling bounds for some M in M_2 | gap_min = 100, gap_max = 100 with M = 175 |
      | non-zero gap bounds without masked_span | mode = "scattered", gap_min = 63, gap_max = 125 |
      | non-zero span_tokens without span | mode = "masked_span" or "scattered", span_tokens = 25 |
    Then construction or place_signal raises ValueError naming the violated constraint

Feature: Encoder integrity under masked_span (ADR-017 I5)
  Scenario: Forward pass keeps constant budget and leaks zero masked tokens to the encoder
    Given a two-stream MultiModalMAE configured with physio_placement = "masked_span" (63..125)
    When forward runs on a Mode-2 step
    Then the shared encoder input has shape [B, n_budget, D]
    And every gathered visible respiration token satisfies mask == 0

Feature: Neutrality-guarded manifest and resume safety
  Scenario: Neutral runs omit ADR-017 keys so pre-ADR runs stay resumable
    Given a Stage-2 run with physio_placement = "scattered" and gap bounds = 0
    When write_masking_manifest writes masking.json
    Then the policy block contains neither physio_gap_min_tokens nor physio_gap_max_tokens

  Scenario: Non-neutral masked_span manifest survives JSON round-trip and rejects changes on resume
    Given a Stage-2 run with physio_placement = "masked_span", gap_min = 63, gap_max = 125
    When masking.json is written, reloaded from disk, and resumed with the identical config
    Then write_masking_manifest accepts the resume with _stable(new) == _stable(reloaded)
    When resumed with physio_gap_min_tokens = 64
    Then write_masking_manifest refuses the resume with a policy-mismatch error

Feature: Matched and overridden Stage-2 reconstruction probes (ADR-017 R6, R7)
  Scenario: Cross-mode probe overrides clear stale placement attributes
    Given a checkpoint trained with physio_placement = "masked_span" (63..125) or "span" (125)
    When run_eval_recon.pin_masking overrides --resp_probe to "random", "hidden", "span", or "masked_span"
    Then all inactive placement attributes on the model are reset to 0 before validation
    And the caption records the resolved placement and gap bounds honestly

  Scenario: An infeasible pinned count is refused rather than silently clamped
    Given --resp_probe masked_span with gap bounds 63..125 and an exempt pin keeping K = 200 (M = 50 < 63)
    When pin_masking validates the probe
    Then it exits with an error naming the infeasible count and the feasible kept counts

  Scenario: Linear interpolation baseline clamps clip edges identically in Phases 4 and 5
    Given a respiration clip whose masked gap touches the left or right boundary (b_0 = 0 or b_n = 0)
    When the --interp_baseline or run_probe_masking_baselines interpolator scores masked positions
    Then it uses np.interp (linear interior, constant-clamped to nearest visible sample at clip edges)
    And reports both mse_interp and mse_const in z-scored target space

Feature: Criterion-3a validation-role floor probe
  Scenario: Validation sweep confirms shortcut removal before any training job launches
    Given fold-0 partition.json and run_probe_masking_baselines.py on the validation role only
    When scored across K in mode2_counts('resp') with at least 5 placement draws per clip
    Then scattered placement satisfies mse_interp << mse_const (sanity check)
    And the sweep over feasible (g_min, g_max) up to g_max = 150 reports mse_interp / mse_const per point without touching the test role
```

