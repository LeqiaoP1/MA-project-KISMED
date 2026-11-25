# ADR-007 — two-stream anatomy-aware constant-budget masking: implementation plan

> **Authority.** `code/notes/design/ADR-007-two-stream-anatomy-aware-constant-budget-masking.md` is the
> contract; this document only says **how** it is built and **in what order**. Where the two disagree,
> the ADR wins and this plan is wrong.
>
> **Scope.** Adds a **single-visual-stream** instance of the constant-budget policy (`streams: tir,resp`,
> `n_budget = 200`), its anatomy-aware stratified spatial tube sampling (`A_core` / `A_periph`,
> `S_core ∈ {1,2}`), the non-zero lower bound of the Mode-2 physiological draw, the parameter surface of
> ADR-007 §2.10, one arm configuration, the observability fields of §2.9, and the §3.1 conformance suite.
> **Out of scope:** model architecture (unchanged), tokenization (unchanged), data preprocessing and cache
> formats (unchanged), the Stage-3 recipe, the evaluation protocol, and the run protocol itself — the
> launch/fold/checkpoint procedure belongs in `code/notes/plan/experiments/`, measured outcomes in
> `code/notes/eval/`, never here.
>
> **State (2026-11-25).** **Approved** (two review rounds, §8 BDD scenarios) and **Phases 0-7
> IMPLEMENTED + SMOKE-VERIFIED on fold 0 of the real corpus**. 876 tests pass (843 pre-existing + 33
> new); the 2-visual mask sequence is bit-identical to the Phase-0 fixture over 200 000 plans. §7 records
> one measured deviation from ADR-007 §3.3 (encoder FLOPs fall 43 %, not 67 %) — see there.
>
> **Judged by** ADR-007 §3.1 (Criterion 1 — policy & mathematical conformance) as this plan's gate.
> Criterion 2 (subject-level band-limited Pearson, Tier 1/2) is the **arm's** gate and is recorded in
> `code/notes/eval/`.
>
> **Review response (2026-11-25, first review folded in).** Blocking: the amended ADR-002 test row
> (**D11**, DoD 8); `StreamSpec.spatial_grid` (§1.1b); `--checkpoint` in G10; `kept_core` only on
> stratified checkpoints (§1.5, §4.4); the dropped `(N - C_p) % T == 0` requirement and the scoped
> `K → T` clamp (§1.2 rule 5, §1.3; the clamp was later removed entirely, see §7 #4). Misalignments: `REGIME_SINGLE_VISUAL` (**D3**), the widened manifest
> guard + `core_coords` (§4.2), plan-level parameter disclosure and the off-ADR banner (**D10**, §4.2),
> and the completed Phase-5 delta table + Stage-3 geometry gap (§5). Polish: the validation-logging path
> (§4.3), the `retain_grad()` requirement in G6, and the replay-report gate G13.
>
> **Second review (2026-11-25).** Verdict: ready to implement, with three cleanups folded in — the
> respiration-cache filter rule (§5, §6), counts kept **out** of `val['components']` (§4.3), and the
> `uniform`-manifest nulls + normative `_validate` ordering (§4.2, Phase 1.2). D3 now also distinguishes
> the two modes on a one-visual run.

---

## 1. Why this is not additive (read this first)

ADR-006 landed as a purely additive branch on top of ADR-005. **ADR-007 cannot**: it changes the
*shape of the stream set*, which the current engine refuses by construction.

| current constraint | where | ADR-007 needs |
| :--- | :--- | :--- |
| "exactly **TWO** visual streams" | `ConstantBudgetApportioner._validate` (`code/core/budget_masking.py`) | exactly **ONE** visual + one physiological |
| Mode-1 / Mode-2 visual split via `_split_tubes` | `_mode1`, `_sensor_absent`, `_mode2` | no split: the sole visual takes `N / T` (Mode 1) or `(N - K_resp) / T` (Mode 2) |
| Mode-2 physiological draw `Uniform{0..C_p}` with a zero-draw fall-through to sensor-absent | `_mode2` | `Uniform{1..C_p}`, **zero** Mode-2 → Mode-1 leakage (ADR-007 §2.5) |
| spatial tubes drawn uniformly without replacement | `mask_placement.place_video` | stratified `A_core` + `A_periph` draw, per sample (ADR-007 §2.3) |
| a 1-visual stream set is rejected by `_validate` | `ConstantBudgetApportioner._validate`; the retirement is *recorded* in the masking block of `configs/pretrain/stage2_hpc_tir_roi_resp.yaml` | the ADR-007 arm |

**Do not revive `stage2_hpc_tir_roi_resp.yaml`.** It is still parsed by
`test_config_loading.py` as the `STAGE2_HPC` geometry fixture (`data_set: tir_roi`, `sig_kernel 16`,
`fs 100`, `temporal_stride 2`, `clip_stride 2.0`) and its resolved values are asserted there. The ADR-007
arm is a **new** file with the ADR-007 geometry; touching the old one breaks that fixture and the
Stage-2/Stage-3 pairing it guards.

The delta is therefore a **new policy branch selected by the declared stream set**, not a new engine.
That choice is deliberate and is decision **D1** in §6: a forked apportioner would drift from the shared
invariants, and `code/tests/test_budget_masking.py` (10⁴-draw suite) would then guard only one branch.

**Neutrality requirement.** Every pre-existing run (two visual + one/two physiological) must keep a
**bit-identical** mask sequence for a fixed `(mask_base, seed, epoch, step)`. Achieved by: (i) the new
branch being reachable only when `len(visual) == 1`; (ii) every new knob defaulting to the pre-change
value (`spatial_sampling = 'uniform'`, `physio_draw_min = 0`); (iii) **no RNG draw on the way to a
disabled branch** — the ADR-006 `p_s > 0.0` guard pattern, reused verbatim.

---

## 2. Definition of done

1. `streams: tir,resp` builds: `StreamSpec(tir, visual, 36 tubes, T=25)` + `StreamSpec(resp, signal, 250, 1)`,
   and the encoder input is strictly `[B, 200, 768]` on **every** step of both phases.
2. Over 10⁵ simulated steps, `K_tir + K_resp == 200` on 100 % of steps; Mode 1 ⇒ `(K_tir, K_resp) = (200, 0)`;
   Mode 2 ⇒ `K_resp ∈ {25,50,75,100}` at 25 % ± 1 % each and `S_tir ∈ {7,6,5,4}`.
3. Every sampled sample mask contains `S_core ∈ {1,2}` **distinct** tubes of `A_core`
   (`P(core hit) = 1.000`) and `S_periph = S_tir - S_core` distinct tubes of `A_periph`, with no duplicates
   and no cross-stratum leakage.
4. `p(e)` follows ADR-007 §2.7 exactly (`p_warm` below `E_warm`, `p_target` from it), is validated at
   construction, and is observed per step.
5. On a Mode-1 step the respiration loss has a **strictly non-zero** gradient w.r.t. the encoder latents
   (`∂L_resp/∂z ≠ 0`), via the existing decoder-prefixing mechanism — no new conditioning code.
6. Every violated feasibility precondition of ADR-007 §2.6 raises at **construction**, naming the condition
   and the numbers.
7. Every §2.9 field is emitted per step / per epoch / per run, including `S_core`, `S_periph` and the
   resolved `A_core` coordinate set.
8. The two-visual mask sequence is bit-identical to the committed pre-change fixture (including the
   generator's next raw draw) and every pre-existing suite is green. **Exactly one test is amended on
   purpose** (**D11**): the ADR-002 row of
   `test_budget_masking.py::test_infeasible_configurations_raise` that asserts a 1-visual stream set is
   rejected — plus the needle of its 3-visual sibling, because the count message widens from
   "exactly TWO visual streams" to "one or two visual streams". The sampling, cap/floor, alignment and
   curriculum tests in that file are untouched and must pass unmodified; that is the neutrality proof.
9. `code/notes/spec/` is regenerated from the new code (`python code/scripts/project/write_specs.py`) and
   `code/notes/GLOSSARY.md` gains the new canonical names of Phase 7.

---

## 3. What already satisfies ADR-007 (nothing to build)

Verified against the code before planning; this is what keeps the delta small.

| ADR-007 requirement | where it already holds |
| :--- | :--- |
| constant visible budget, dense `[B, N, D]`, no `[MASK]`/padding (§2.4(1)) | `MultiModalMAE.forward` gather + the `[B, n_budget, embed_dim]` assert |
| two-phase `p(e)` curriculum (§2.7) | `ConstantBudgetApportioner.p_at`, `virtual_sensing_warmup_p/epochs` |
| per-step counts, per-sample placement (§2.4(6)) | `make_masks` → `plan` + `place_video` / `place_signal` |
| determinism from `(seed, epoch, step)`, rank-independent counts (§2.4(9)) | `mask_placement.counts_generator` / `placement_generator` / `step_generators` (SplitMix64 mix) |
| 1-D placement policy `scattered` / `span` (§2.4(7)) | `place_signal` + `physio_placement` / `physio_span_tokens` |
| decoder conditioning mechanism (a) — prefixing (§2.4(8)) | `forward`: decoder prefix is `torch.cat([z, mask_tokens])`; `masking.json` records `conditioning: prefix` |
| loss-key stability with a 0-kept stream (§2.4(10)) | `forward`'s `n_masked == 0` / `tokens[s] is None` branches (explicit zero, key always present) |
| `R = 250/25 = 10`, `T=25`, tubelet `(2,16,16)`, 36 tubes (§2.2) | `MultiModalMAE._check_space_time_alignment` at `fps 25 / stride 4 / fs 1000 / sig_kernel 32` |
| verbatim 3-D VideoMAE kernel on `tir`, no 2nd visual (§2.8 I1–I3) | `--patch_stream tir`, `load_pretrained_encoder`, `project_multimae_base` (ViT-B, D=768, 12 blocks) |
| per-stream loss entries, `kept_*`, `kept_units_*`, `mode*`, `p_virtual`, `n_batch` (§2.9) | `code/engines/pretrain.py::train_one_epoch` |
| `tir_roi` corpus (thermal ROI + `Resp_Volts.txt`) with TIR-ROI + respiration caches | `data/datasets.py::TIR_ROI_DATA_SETS`, `--roi_cache`, `--resp_cache` |

**What is genuinely missing:** (i) the one-visual apportionment branch, (ii) `physio_draw_min`, (iii) the
stratified `S_core` count in the plan and its placement, (iv) the new parameter surface through model,
runner, manifest and eval pin, (v) `kept_core` / `kept_periph` observability, (vi) the arm config and its
registrations, (vii) the conformance suite.

---

## 4. Phases

### Phase 0 — pre-flight (no code, ~15 min)

1. Confirm the geometry resolves to the ADR's numbers **before** touching anything:
   `fps 25`, `temporal_stride 4`, `clip_duration 8.0` ⇒ `num_frames = 50`, `Gt = 25`;
   `fs 1000`, `sig_kernel 32` ⇒ `n_signal = 250`; `R = 250/25 = 10`; `96/16` ⇒ `6×6 = 36` tubes.
2. Confirm the caches stay valid: the TIR-ROI cache keys on clip length / fps / input geometry (not the
   window hop), the respiration cache on `fs`. The arm's `clip_stride` is **4.0** (decision **D6**), so the
   sampled clip inventory is identical to the ADR-005/006 comparator's.
3. Snapshot the neutrality fixture for DoD 8: 50 000 kept-count plans from the **current** engine at a
   fixed `(mask_base=1234, epoch∈{None,0,50,149}, step)` for the ADR-006 arm's stream set, plus the
   generator's next raw draw. Commit as a test fixture (`code/tests/fixtures/`), produced **before** Phase 1.

### Phase 1 — `code/core/budget_masking.py`: the one-visual policy

**1.1 Constructor surface (exact ADR-007 §2.10 names — one string for YAML key, argparse dest and Python
keyword, as ADR-006 established):**

| new keyword | default | meaning |
| :--- | ---: | :--- |
| `spatial_sampling: str` | `'uniform'` | `'anatomy_aware'` \| `'uniform'` (neutral default) |
| `core_row_bounds: Tuple[int,int]` | `(2, 3)` | inclusive row span of `A_core` on the patch grid |
| `core_col_bounds: Tuple[int,int]` | `(2, 3)` | inclusive column span of `A_core` |
| `core_tubes_min: int` | `1` | `S_core^min` |
| `core_tubes_max: int` | `2` | `S_core^max` |
| `periph_min_tubes: int` | `1` | `m_periph` — peripheral floor (§2.6 rule 5) |
| `physio_draw_min: int` | `0` | lower bound of the Mode-2 physiological draw; `0` = pre-ADR-007 behaviour, `1` = ADR-007 §2.5 |

Two of the seven are **plan-level additions beyond ADR-007 §2.10** and are disclosed as such:
`periph_min_tubes` (a named constant instead of a literal inside the §2.6(5) inequality, per `AGENTS.md`
"zero hidden constants") and `physio_draw_min` (**D4**). A third, `video_min_tubes`, already exists as an
ADR-005 key and is reused here as an *assertion* (`== S_tir^min`), not as a new knob. The other four are
§2.10 verbatim.

Resolved at construction and published for the manifest: `core_indices` (the sorted `A_core` tube indices
of the `Gh×Gw` grid), `periph_indices`, `s_tir_min = (N - C_p) // T`, `s_tir_max = N // T`.

**1.1b `StreamSpec` gains the grid shape.** `A_core` is a 2-D subgrid (`core_row_bounds` ×
`core_col_bounds`) but `StreamSpec` carries only the scalar `n_units`, so the apportioner currently cannot
resolve `[0, G_h-1] × [0, G_w-1]`. Add an optional field
`spatial_grid: Optional[Tuple[int, int]] = None` (validated in `__post_init__`: both ≥ 1 when given),
populated from `(self.grid_h, self.grid_w)` in `multimae.py`. Under `anatomy_aware`, construction asserts
`G_h * G_w == spec.n_units` and derives
`core_indices = [r * G_w + c for r in rows for c in cols]` — the flatten order `place_video` already uses
(`s = h*Gw + w`), so the default bounds resolve to `[[2,2],[2,3],[3,2],[3,3]]` → `[14,15,20,21]`. A visual
spec without `spatial_grid` under `anatomy_aware` raises.

**1.2 `_validate` — a second feasibility block, selected by the declared stream set (D1):**

`n_visual` moves from `!= 2 → raise` to `not in (1, 2) → raise`; `len(visual) == 1` requires exactly one
physiological stream (ADR-007 §2.1 declares `{tir, resp}`) and runs the ADR-007 §2.6 checks:

| §2.6 | check | ADR-007 arm | failure text names |
| :--- | :--- | :--- | :--- |
| 1 | `n_budget % tokens_per_unit == 0` | `200 % 25 = 0` | the tube-divisibility condition |
| 2 | `tokens_per_unit ≤ C_p < n_budget` | `25 ≤ 100 < 200` | the inpainting-cap condition |
| 3 | `S_max = N // T ≤ units_v` | `8 ≤ 36` | the grid-capacity condition |
| 4 | `S_max / units_v ≤ video_cap_ratio` | `0.2222 ≤ 0.25` | the high-masking ceiling (see **D2**) |
| 5 | `video_min_tubes == S_tir_min = (N - C_p) // T` — **floor** division, no divisibility requirement | `100 // 25 = 4`, `4 == 4` | the stratified-feasibility / floor condition |
| 5b | `1 ≤ core_tubes_min ≤ core_tubes_max ≤ ‖A_core‖` and `S_tir_min ≥ core_tubes_max + periph_min_tubes` and `‖A_periph‖ ≥ S_tir_max - core_tubes_min` | `2 ≤ 4`, `4 ≥ 3`, `32 ≥ 7` | which stratified inequality failed |
| 6 | `A_core ∩ A_periph = ∅` and `‖A_core‖ + ‖A_periph‖ = units_v` | `4 + 32 = 36` | the partition-coverage condition |
| 7 | `R = units_resp / T ∈ ℤ_{>0}` (already enforced by the model) | `250/25 = 10` | delegated, not re-derived |
| 8 | `0 ≤ p_warm ≤ p_target ≤ 1`, `0 ≤ E_warm ≤ epochs` | `0.10 ≤ 0.50` | the existing schedule-ordering check, reused |

**Order is normative.** `_validate` evaluates rules `1 → 2 → 3 → 4 → 5 → 5b → 6` in table order, so a
multi-violating tuple always reports the first condition. This is what makes the **D11** amended row
deterministic: `[visual('tir'), signal('resp')]` at `n_budget = 350` and `video_min_tubes = 2` fails
rule 4 (`14/36 = 38.9 % > 0.25`) **before** rule 5 (`video_min_tubes 2 != s_tir_min 9`).

**Do not add `(N - C_p) % T == 0`.** §2.6(5) defines `S_tir^min` by floor division precisely so a
non-multiple `C_p` stays legal: with `c_p = 0.45 ⇒ C_p = 112`, `S_tir^min = 88 // 25 = 3`, the downward
branch of §2.5 realizes `⌊C_p/T⌋·T = 100`, and the realized minimum is `⌈(N - C_p)/T⌉ = 4 ≥ 3` — so
`_plan`'s bound check `video_min_tubes ≤ kept_units ≤ s_tir_max` still holds and the downward branch is
reachable instead of dead. A divisibility requirement would reject a legal ADR-007 tuple.

Additional guards, all resolved-time: `physio_draw_min ∈ [0, C_p]`; with `physio_draw_min ≥ 1` require
`C_p ≥ tokens_per_unit`; `spatial_sampling` ∈ the two literals; `core_row_bounds`/`core_col_bounds`
inside `[0, G_h-1]`/`[0, G_w-1]` with `lo ≤ hi` (needs `spatial_grid`, §1.1b); `p_single_modality > 0`
remains rejected with a single visual stream (the existing rule already fires — assert it separately so
the message names *this* tuple).

The stream-count message widens from `'exactly TWO visual streams'` to `'one or two visual streams'`,
because the rule itself widens. Those two rows of the ADR-002 infeasibility test are therefore the single
documented test amendment of **D11**; nothing else in that file moves.

Module constants (additive, no meaning change): `SPATIAL_UNIFORM = 'uniform'`,
`SPATIAL_ANATOMY_AWARE = 'anatomy_aware'`, `SPATIAL_SAMPLING = (...)`, `REGIME_SINGLE_VISUAL =
'single_visual'` appended to `REGIMES` (**D3**; `BudgetPlan.__post_init__` validates against the tuple, so
no model change is needed), and `SPATIAL_POLICY_SINGLE_VISUAL` / `SPATIAL_POLICY_DUAL_VISUAL` for the
branch label used in logs.

**1.3 Draw paths — branch at the top, share everything below:**

```text
sample_step(g, epoch):                       # unchanged envelope
    u = rand(g)                              # ONE draw per step, both branches
    return _mode1(g, epoch) if u < p_at(epoch) else _mode2(g, epoch)

_sensor_absent(g, epoch, mode=MODE_VIRTUAL_SENSING):
    if len(self.visual) == 1:                        # ADR-007 single-visual
        return _sole_visual(g, mode, k_resp=0)       # no split; p_s > 0 is
                                                     # rejected at construction,
                                                     # so NO p_s draw happens
    ... existing ADR-006 body unchanged ...

_mode2(g, epoch):
    raw = {s.name: _randint(self.physio_draw_min, C_p[s.name], g) for s in self.signal}
    ... alignment: remainder charged to the first signal that can absorb it; the
        §2.5 downward fallback K_tilde - (T - r) stays and the count is NEVER
        clamped (a clamp would break the modulo alignment). With
        physio_draw_min >= 1 the result cannot drop below T provided
        _validate enforces C_p >= T for every signal; with the neutral 0 the
        zero-count draw still falls through to _sensor_absent below (DoD 8) ...
    if sum(raw.values()) == 0:                       # reachable only when physio_draw_min == 0
        return _sensor_absent(g, epoch, MODE_INPAINTING)
    if len(self.visual) == 1:                        # ADR-007 single-visual
        return _sole_visual(g, MODE_INPAINTING, k_resp=sum(raw.values()))
    ... existing ADR-006 body unchanged ...

_sole_visual(g, mode, k_resp):
    s_tir = (self.n_budget - k_resp) // self.tokens_per_unit      # Mode1: 8, Mode2: 4..7
    units = {v.name: s_tir, sig.name: k_resp}
    tokens = {v.name: s_tir * T, sig.name: k_resp}
    core, periph = self._draw_stratification(s_tir, g)            # None,None when uniform
    regime = REGIME_SINGLE_VISUAL if k_resp == 0 else REGIME_INPAINTING   # D3
    return self._plan(mode, units, tokens, regime, core=core, periph=periph)
```

`_draw_stratification(s_tir, g)` — **one** draw per step, shared by the whole batch (§2.4(6)), taken
**after** the count draws exactly as §2.5's reference algorithm orders it:

```text
if spatial_sampling == 'anatomy_aware':
    s_core = randint(core_tubes_min, core_tubes_max, g)     # {1,2}
    return {visual: s_core}, {visual: s_tir - s_core}
return None, None                                          # 'uniform': zero draws, RNG untouched
```

The `'uniform'` path must consume **no** random number (guard on the literal, never a draw-then-ignore).

**1.4 `_plan` invariants** gain a per-branch check, additive to the existing exact-sum / token-multiple
re-checks:

- one-visual branch: `video_min_tubes ≤ kept_units[v] ≤ s_tir_max`, and
  `kept_tokens[v] == kept_units[v] * T`;
- anatomy-aware: `core_units[v] + periph_units[v] == kept_units[v]`,
  `core_tubes_min ≤ core_units[v] ≤ core_tubes_max`, `periph_units[v] ≥ periph_min_tubes`.

**1.5 `plan_from_counts(kept_tokens, kept_core=None)`** — evaluation pins and reconstruction probes must
be able to state a stratified allocation, and **only a stratified policy can consume one**:

- `spatial_sampling == 'anatomy_aware'`: `kept_core` defaults to `core_tubes_min` (deterministic,
  documented, recorded in the caption); an explicit value is re-validated against
  `[core_tubes_min, core_tubes_max]` and against `periph_min_tubes`.
- `spatial_sampling == 'uniform'`: `core_units`/`periph_units` stay `None`, and an **explicit** `kept_core`
  raises `ValueError` — this covers an ADR-006 checkpoint *and* the ADR-007 `uniform` Tier-1 comparator.
  Reporting `core_units = 1` on a uniform checkpoint would put a fabricated number in a figure caption.
- `_infer_regime` mirrors the sampler under **D3**: `REGIME_INPAINTING` iff a signal stream keeps `> 0`
  tokens; otherwise `REGIME_SINGLE_VISUAL` for a one-visual set and `REGIME_SINGLE` / `REGIME_BI_VISUAL`
  (by carrier count) for a two-visual set. `REGIME_SINGLE` can never be returned for a one-visual run.

`BudgetPlan` gains two **additive, defaulted** fields — `core_units: Optional[Dict[str,int]] = None`,
`periph_units: Optional[Dict[str,int]] = None` — validated in `__post_init__`; `mode`, `mode_by_counts`,
`regime` and `carrier` keep their exact ADR-005/006 meanings.

### Phase 2 — `code/core/mask_placement.py`: stratified tube placement

New public function, exported through `__all__`:

```python
place_video_stratified(counts_core, counts_periph, n_units, tokens_per_unit,
                       core_indices, generator=None, batch_size=None, device='cpu')
-> torch.Tensor                                    # [B, (S_core+S_periph)*T], ascending per row
```

- `core_indices` is the sorted tuple produced by the apportioner; `A_periph` is its complement in
  `range(n_units)` (validated: non-empty, in range, no duplicates).
- Per sample, in this fixed order (one generator, documented): `perm_core = rand(‖A_core‖).argsort()[:k_core]`,
  then `perm_periph = rand(‖A_periph‖).argsort()[:k_periph]`; the union of spatial positions is expanded to
  tube indices with the same `+ arange(T) * n_units` offsets as `place_video`, then sorted.
- Asserts per row: exact `k_core + k_periph` unique indices, `k_core ≥ 1`, and — the §2.3 consequence —
  `min(core hit) ≥ 1` for every sample.
- `place_video` / `place_signal` / the seeding helpers are **not modified** (DoD 8).

Note the count-scaling warning that belongs in the docstring: a per-sample *count* is still rejected
(`_uniform_counts`), because counts are a per-step property; only *placements* differ per sample.

### Phase 3 — `code/core/multimae.py`: wiring

1. `__init__` accepts and forwards `spatial_sampling`, `core_row_bounds`, `core_col_bounds`,
   `core_tubes_min`, `core_tubes_max`, `periph_min_tubes`, `physio_draw_min` to
   `ConstantBudgetApportioner`; publishes `self.spatial_sampling`, `self.core_indices`,
   `self.s_tir_min`, `self.s_tir_max`. The `[S_v resolved]` block of ADR-006 stays untouched. The visual
   `StreamSpec`s are built with `spatial_grid=(self.grid_h, self.grid_w)` (§1.1b), so the apportioner
   resolves `A_core` without importing model geometry.
2. `make_masks`: for a visual stream with `self.spatial_sampling == SPATIAL_ANATOMY_AWARE`, call
   `placement.place_video_stratified(plan.core_units[s], plan.periph_units[s], spec.n_units,
   spec.tokens_per_unit, self.core_indices, generator=placement_generator, batch_size=B, device=device)`;
   otherwise the existing `place_video`. `plan.kept_tokens` remains the authority for the total.
3. `forward`: unchanged — it already gathers `plan.kept_tokens[s]`; the `[B, n_budget, D]` assert is the
   §3.1(1) shape gate. Add to the returned dict, **on every step**:
   `'kept_core': {s: int(plan.core_units[s]) if ... else 0}`, `'kept_periph': {...}` (0 under the uniform
   ablation, so the key set stays a function of the config), and `'spatial_sampling': self.spatial_sampling`.
4. `build_pretraining_model`: `getattr`-forward the seven keys, `_parse_int_csv` for the two bounds, and
   assert a one-visual stream set is legal (the existing "≥1 video and ≥1 signal" check already holds);
   `_parse_int_csv` failures must name the flag.
5. `[mask]` provenance print: add `spatial_sampling`, the resolved `A_core` coordinates, `S_core∈[min,max]`,
   `periph_min`, `S_tir∈[min,max]`, `physio_draw_min`.

### Phase 4 — runner, manifest, engine, eval pin

**4.1 `code/runners/run_pretrain.py`** — register on the existing masking block of `get_args()` (all
defaults are the pre-change values, so `_check_config_keys` stays a pure schema check):

| flag | type | default | note |
| :--- | :--- | ---: | :--- |
| `--spatial_sampling` | str | `uniform` | `uniform` \| `anatomy_aware` |
| `--core_row_bounds` | str | `2,3` | parsed with `_parse_int_csv`; ignored when `uniform` |
| `--core_col_bounds` | str | `2,3` | idem |
| `--core_tubes_min` | int | `1` | `S_core^min` |
| `--core_tubes_max` | int | `2` | `S_core^max` |
| `--periph_min_tubes` | int | `1` | `m_periph` |
| `--physio_draw_min` | int | `0` | `1` for ADR-007 |

Forward through `build_pretraining_model(args)` with the same `getattr` pattern as the other masking keys;
extend the `[mask]` print.

**4.2 `write_masking_manifest`** (same file) — extend the `policy` block when **any** of
`len(model.visual) == 1`, `spatial_sampling != 'uniform'`, `physio_draw_min != 0` holds. The widened guard
is what keeps a 1-visual `uniform` ablation (§3.2(a)) fully recorded, while every 2-visual ADR-005/006
manifest stays **byte-identical** (the resume guard compares key sets, so an unconditional write would
make an in-flight run un-resumable). Fields: `adr` (`'ADR-007'` when anatomy-aware **and**
`physio_draw_min == 1`, else `'off-adr-007'`), `spatial_sampling`, `core_row_bounds`, `core_col_bounds`,
`core_coords` — **both** the 2-D coordinate set `[[2,2],[2,3],[3,2],[3,3]]` and the flat `core_indices`
`[14,15,20,21]`, because §2.9 records the coordinate set, not just the bounds — `core_tubes_min/max`,
`periph_min_tubes`, `physio_draw_min`, `s_tir_min`, `s_tir_max`, `signal_tokens_per_visual`.
`conditioning: {mechanism: prefix}` and `kernel.random_init_visuals` already satisfy §2.8 I1–I4 and §2.9's
decoder-provenance record. Under a `uniform` policy the ignored defaults are **not** serialized:
`core_coords: null` and `core_indices: []`, so no reader can mistake a default `(2, 3)` bound for a
realized `A_core`.

Also print a `[mask] OFF-ADR-007` banner in `run_pretrain.py` when `len(model.visual) == 1` while
`spatial_sampling == 'uniform'` or `physio_draw_min == 0` (**D10**). Without it, a bare
`--streams tir,resp --n_budget 200` invocation skips the YAML, runs `uniform` sampling with the
`Uniform{0..100}` draw, and looks like the ADR's arm in every log line and manifest field.

**4.3 `code/engines/pretrain.py`** — inside the existing `if isinstance(out, dict):` block, log
`kept_core_<stream>` and `kept_periph_<stream>` from `out['kept_core'] / out['kept_periph']` (they average
to the per-epoch mean `S_core` / `S_periph` that §2.9 asks for; Mode-1/Mode-2 fractions are already the
`is_virtual` mean). Add the same sample-weighted accumulation to `evaluate_pretrain`'s payload, DDP merge
and return value, symmetrically to the existing `mode_resolved` accumulation, so validation reports the
same observables as training.

**Keep `components` loss-only.** `evaluate_pretrain` already returns the masking diagnostics as *sibling*
keys (`kept`, `mode`, `resolved`, `single`), and
`test_stage2_protocol.py::test_evaluate_pretrain_records_its_components` pins
`set(result['components']) == {'mse_resp'}` — putting counts there would break a third test. The new
fields therefore go into a new sibling `stratification` dict (`core_tubes_<s>` / `periph_tubes_<s>`, in
TUBES, so they never mix with the token-valued `kept_<s>`), are added to the DDP `payload` and merged
key-by-key like the others, and are then forwarded in `run_pretrain.py` to (i) the `[val]` print and (ii)
`wandb_logger.log({f'val/{k}': ...})`, exactly as `components` is today. `make_eval_record` stays
unchanged — it records loss components and counts, not masking diagnostics.

**4.4 `code/runners/run_eval_recon.py`** — `pin_masking` must be able to state a stratified allocation:

- add `--pin_core_tubes` (int, `-1` = unset): `-1` leaves `kept_core=None`, which the apportioner resolves
  to `core_tubes_min` **only on an anatomy-aware checkpoint**; a value `>= 0` is forwarded explicitly and
  raises on a `uniform` checkpoint (ADR-006 checkpoints and the ADR-007 `uniform` comparator) rather than
  reporting a fabricated `S_core` (§1.5).
- `--pin_single_carrier` (ADR-006) stays valid, but on a 1-visual checkpoint it is only meaningful as
  `--pin_single_carrier <the sole visual>` — where it is exactly Mode 1 (`n_budget / 0`). Naming any other
  stream raises, because a one-visual policy has no second carrier to zero.
- the existing floor/cap clamp becomes correct automatically **because** `video_cap_units[v]` resolves to
  `S_tir_max = 8` and `video_min_tubes` to `S_tir_min = 4` under a one-visual checkpoint (**D2**), so
  `--kept_tir 0.0 --kept_resp 0.0` fills to exactly `200 / 0` and `--kept_resp 0.4` gives `100 / 100` with
  a zero residual. No floor-bypass path is needed: the ADR-006 bypass existed only because `350/25 = 14`
  tubes lay *above* the split cap of 9, whereas here the cap **is** the Mode-1 allocation.
- the caption records `spatial_sampling`, `S_core`, `S_periph`, `mode`, `n_budget` — §2.9's machine-readable
  companion for a figure.

### Phase 5 — the arm configuration

New file `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy.yaml`, derived from
`code/configs/pretrain/stage2_budget_3regime_single_modality.yaml` (same geometry, same caches, same
optimiser) with **only** these deltas:

| key | ADR-006 arm | ADR-007 arm | source |
| :--- | ---: | ---: | :--- |
| `data_set` | `tir_roi_resp_rgb2d` | **`tir_roi`** (raw-tree TIR ROI + `Resp_Volts.txt`) | §2.1 |
| `rgb2d_landmarks`, `rgb2d_store`, `rgb2d_cache_lru`, `rgb2d_min_valid_ratio`, `rgb2d_area_min_frac`, `rgb2d_max_sway_px` | present | **removed** — no `rgb2d` stream, and a stale key is a hard error in `_check_config_keys` | §2.1 |
| `streams` | `tir,rgb2d,resp` | **`tir,resp`** | §2.1 |
| `n_budget` | `350` | **`200`** | §2.4(1) |
| `physio_cap_ratio` | `0.50` | **`0.40`** (`C_p = 100`) | §2.4(2) |
| `video_min_tubes` | `2` | **`4`** (`= (N - C_p)/T`) | §2.4(3) |
| `video_cap_ratio` | `0.25` | `0.25` (ceiling; `S_max/36 = 22.2 %`) | §2.6(4) |
| `virtual_sensing_warmup_epochs` | `50` | **`20`** | §2.7 (**D7**) |
| `p_virtual_sensing` / `virtual_sensing_warmup_p` | `0.50` / `0.10` | unchanged | §2.7 |
| `spatial_sampling` | — | **`anatomy_aware`** | §2.3 |
| `core_row_bounds` / `core_col_bounds` | — | **`2,3` / `2,3`** | §2.3 |
| `core_tubes_min` / `core_tubes_max` | — | **`1` / `2`** | §2.3 |
| `periph_min_tubes` | — | **`1`** | §2.6(5) |
| `physio_draw_min` | — | **`1`** | §2.5 |
| `p_single_modality`, `single_modality_warmup_p`, `single_carrier_tubes` | `0.20 / 0.00 / 14` | **removed** (unreachable with one visual stream) | §2.11 |
| `min_signal_spread` / `rail_touch_v` | `0.1` / `9.900` | **`0.0` / `0.0`** — the cache lineage refuses them non-zero | design D6 |

**Respiration-cache lineage, not the raw-path filters.** `BP4DPlusTIRRespDataset._from_resp_cache`
**raises** when `--resp_cache` is attached while either filter is non-zero (the cache owns per-sample
validity; the content must not be filtered twice), so an arm that runs cache-first at `fs: 1000` must
ship `0.0 / 0.0` — not the ADR-006 arm's `0.1 / 9.900`. Without `--resp_cache` the reverse holds: the raw
path needs `0.1 / 9.900`, and a run that silently switched between the two would change the corpus.
`run_waveform.py` compares `min_signal_spread` / `rail_touch_v` **only when no cache is attached**
(otherwise it compares `resp_cache_key`), and `clip_stride` is always in that corpus contract — so the
Stage-3 sibling must repeat `clip_stride: 4.0`.

Everything else — `tir_channels: 3`, `tubelet 2,16,16`, `input_size 96`,
`fs 1000`, `fps 25`, `clip_duration 8.0`, `temporal_stride 4`, `sig_kernel 32`, `seq_len 0`,
`clip_stride 4.0`, `pos_init sincos3d`, `patch_stream tir`, `pretrained_encoder videomae:k400`,
`model project_multimae_base`, `signal_weight 0.5`, `physio_placement scattered`, `epochs 150`,
`blr 1.5e-4`, `batch_size 16`, the task groups and the `roi_*` cache keys — is **identical** to the
ADR-006 arm, so a comparison differs by the policy alone (plus `E_warm`, see **D7**).

Header comment must state the §2.5 arithmetic (Mode 1 `200/0`; Mode 2 `K_resp ∈ {25,50,75,100}` ⇒
`S_tir ∈ {7,6,5,4}`), the `A_core` grid position and the `77.8 %` Mode-1 masking ratio, and point at the
ADR-007 experiment protocol for the launch command.

**Stage-3 consequence — belongs in the experiment protocol, flagged here so it is not forgotten.**
`configs/finetune/resp_tir_roi_hpc.yaml` is geometry-matched to the *retired* arm (`temporal_stride 2`,
`sig_kernel 16`, `fs 100`), so it cannot fine-tune an ADR-007 encoder: the 1-D grid would silently become
`R = 2` instead of `R = 10`, and `head_hidden: 0` transfers `heads.resp` at the wrong width. Note that
`run_waveform.py::ROI_CONTRACT_KEYS` compares only `roi_landmarks / roi_padding / input_size /
crop_rule / max_sway_px` — **the 1-D grid is NOT in that guard**, so this mismatch does not ABORT; it just
trains on the wrong geometry. A `configs/finetune/resp_tir_roi_hpc_adr007.yaml` sibling
(`temporal_stride 4`, `sig_kernel 32`, `fs 1000`, `input_size 96`, `roi_padding 0.10`) is a prerequisite of
the Tier-1 comparison and is a tracked deliverable of the ADR-007 run protocol.

### Phase 6 — tests and gates

(The spec/glossary refresh is Phase 7 — it belongs *after* the code it documents.)

New suite `code/tests/test_two_stream_anatomy_masking.py`, plus a **mandatory** registration of the new
YAML in `code/tests/test_task_group_configs.py::TIR_ROI_CONFIGS`: the arm ships a `task_groups:` block, and
`test_tir_roi_config_inventory_is_complete` asserts that the list equals every `configs/*/*.yaml` containing
one — so the file cannot land without it. `test_config_loading.py` needs no edit (it globs
`configs/pretrain/*.yaml`, and the new config then flows through `_check_config_keys` automatically).
The **only** amendment of an existing test is the stream-count parametrization of `test_budget_masking.py`
described in **D11**; every other pre-existing suite is unmodified (DoD 8).

| gate | assertion |
| :--- | :--- |
| **G1** | Budget & shape: over 10⁵ sampled steps across both phases, `K_tir + K_resp == 200` on 100 %; a two-step model forward per phase yields `[B, 200, 768]` and the `assert` never fires. |
| **G2** | Mode/bound compliance: Mode 1 ⇒ `(200, 0)`, `S_tir = 8` and `regime = 'single_visual'` on 100 % of Mode-1 steps; Mode 2 ⇒ `K_resp ∈ {25,50,75,100}` each at `25 % ± 1 %` of 10⁵ draws, `S_tir ∈ {7,6,5,4}` and `regime = 'inpainting'`; **zero** Mode-2 draw resolves to `K_resp = 0`. |
| **G3** | Stratification: over 10⁵ drawn sample masks, every sample has `S_core ∈ {1,2}` distinct `A_core` indices (`P = 1.000`), `S_periph = S_tir - S_core` distinct `A_periph` indices, no duplicates, no cross-stratum leakage; `S_core = 1` and `= 2` each occur ≈ 50 %. |
| **G4** | Curriculum: realized Mode-1 share `0.10` for `e < 20`, `0.50` for `e ≥ 20`, within a binomial 99 % CI (4σ), and `p_at(None) = p_target`. |
| **G5** | Determinism & diversity: two independent processes replaying `(base_seed, epoch, step)` derive identical counts (and an identical next raw draw); within one batch the per-sample tube/token placements differ. |
| **G6** | Zero-sensor gradient: on a Mode-1 forward/backward, `∂L_resp/∂z ≠ 0` and `∂L_tir/∂z ≠ 0`. `z` is a **non-leaf intermediate**, so the test calls `z.retain_grad()` (forward hook on `enc_norm`) or asserts non-zero `enc_blocks`/`enc_norm` **parameter** grads after `losses_mse['resp'].backward()`. Both `mse_tir` and `mse_resp` keys exist and `mse_resp > 0`. |
| **G7** | Feasibility rejection: one case per §2.6 precondition (Phase 1.2 table) raises `ValueError` at construction, each message naming its condition; `spatial_sampling='anatomy_aware'` with degenerate bounds (`core_tubes_max > ‖A_core‖`, `periph_min_tubes` breaking rule 5, out-of-grid bounds, a visual spec with no `spatial_grid`) raises too; and a non-multiple `C_p` (`c_p = 0.45`) **constructs** and realizes `S_tir ≥ 3` — the regression guard for the dropped divisibility rule. |
| **G8** | Neutrality: for the ADR-006 stream set with none of the seven new keys set, 50 000 plans equal the Phase-0 fixture **and** the generator's next raw draw is bit-identical; `test_mask_placement.py`, `test_curriculum_schedule.py`, `test_regime_single_carrier.py` are green **unmodified**, and `test_budget_masking.py` is green with exactly the **D11** amendment and no other change. |
| **G9** | Config: the new YAML parses through `run_pretrain.get_args()` + `_check_config_keys`, resolves the §2.10 tuple, and `masking.json` contains `adr`, `spatial_sampling`, `core_coords` + `core_indices`, `s_tir_min/max`, `physio_draw_min`, `signal_tokens_per_visual = 10`. |
| **G10** | Eval pin: `pin_masking` on a one-visual checkpoint yields exactly `200 / 0` (Mode 1) for `--kept_resp 0.0` and `100 / 100` (S_tir = 4) for `--kept_resp 0.4`, with a zero residual; the caption names `S_core`/`S_periph`; `--pin_core_tubes 2` raises on a `uniform` checkpoint. |
| **G11** | Forced-mode smoke (ADR-007 §3.3): `--p_virtual_sensing 1.0` ⇒ every step logs `mode_resolved = 1`, `kept_tir = 200`, `S_core ∈ {1,2}`; `--p_virtual_sensing 0.0` ⇒ every step `mode_resolved = 2`, `kept_tir ∈ {175,150,125,100}`; no `NaN`, no assertion, all §2.9 fields emitted. |
| **G12** | Uniform ablation: `--spatial_sampling uniform` produces no `A_core` draw (RNG-neutral vs the non-stratified engine) and `kept_core = 0` on every step, so the Tier-1 §3.2(a) comparator is one flag — and its `masking.json` still carries the full policy block (`adr: off-adr-007`). |
| **G13** | Replay report: `scripts/project/mask_policy_report.py` runs against the arm with all of its own ADR-005 checks green (`exact_sum`, `visual_multiple_of_T`, `visual_inside_floor_cap`, `signal_inside_cap`, `mode_iff_signal_zero`), and its `mode2_k_tilde_zero_share.expected` becomes `0.0` when `physio_draw_min ≥ 1` (today it hard-codes `1/(C_p+1)`, true only for the `Uniform{0..C_p}` draw); add an `S_core` / `S_periph` section so §2.9's stratified observables are replayable without a GPU. |

```text
cd code
python -m pytest tests/test_two_stream_anatomy_masking.py tests/test_budget_masking.py \
                 tests/test_mask_placement.py tests/test_curriculum_schedule.py \
                 tests/test_regime_single_carrier.py tests/test_config_loading.py \
                 tests/test_task_group_configs.py -q

# G11 — forced-mode smoke, caches attached, curriculum disabled.
python -u runners/run_pretrain.py \
    -c configs/pretrain/stage2_budget_tir_resp_2stream_anatomy.yaml \
    --exp_root /tmp/adr007_smoke --fold 0 --dev_smoke --epochs 1 --warmup_epochs 0 \
    --virtual_sensing_warmup_epochs 0 --p_virtual_sensing 1.0 \
    --max_entries 100 --batch_size 8 --num_workers 0 \
    --output_dir /tmp/adr007_smoke/out \
    --roi_cache <TIR_ROI_CACHE> --resp_cache <RESP_CACHE> \
    --min_signal_spread 0 --rail_touch_v 0

# G10 — zero-shot virtual-sensing pin on the resulting checkpoint.
python -u runners/run_eval_recon.py --checkpoint /tmp/adr007_smoke/out/<last>.pth --split test \
    --resp_probe hidden --kept_resp 0.0 --kept_tir 0.0 --repeats 5
```

`--max_entries` is the per-SPLIT cap (`--max_clips` is per session). The `--min_signal_spread 0
--rail_touch_v 0` overrides are redundant with the arm YAML (`0.0 / 0.0` by design) and are kept here only
to make the cache-lineage requirement explicit; a run **without** `--resp_cache` must instead pass the
raw-path filters `0.1 / 9.900`, because the dataset raises when a cache is attached with either filter
non-zero.

### Phase 7 — terminology, spec and glossary refresh (after Phase 6)

`A_core` / `A_periph`, `S_core` / `S_periph`, `S_tir^min` / `S_tir^max`, `m_periph`,
`spatial_sampling`, `physio_draw_min`, `REGIME_SINGLE_VISUAL`, and a one-line statement that
**`video_cap_ratio` is a *cap* under two visual streams and a *ceiling* under one** (**D2**). Canonical
symbols follow ADR-007 §2.3/§2.10 verbatim; no new aliases. Regenerating `code/notes/spec/`
(`python code/scripts/project/write_specs.py`) is part of this phase: `02_model_architecture.md` gains the
single-visual stream-set contract, `03_engine_and_loss.md` the `physio_draw_min` and stratified-placement
rules.

---

## 6. Decisions taken while planning (review these first)

| id | decision | why / alternative rejected |
| :--- | :--- | :--- |
| **D1** | The policy branch is selected by `len(visual)`; **no** new `policy:` flag. | `streams` is already the ADR's authoritative knob (§2.1); a second selector would allow the two to disagree. Rejected: forking the apportioner (drifts from the shared invariants; the 10⁴-draw suite would guard one branch). |
| **D2** | Under one visual stream, `video_cap_ratio` is read as the **high-masking ceiling** (`S_max/units_v ≤ ratio`, §2.6 rule 4) and `video_cap_units[v]` resolves to `S_max = N/T = 8`; `video_min_tubes` must equal `S_tir^min = (N - C_p) // T` (**floor**, per §2.6(5)). | Keeps `run_eval_recon`'s floor/cap clamp semantically correct with no new eval flag, and keeps the floor an "assertion expressed as an option" (the `single_carrier_tubes` pattern). Rejected: reusing the floored cap `floor(0.25·36) = 9`, which would silently admit a 9-tube Mode-2 step and break the §2.5 {7,6,5,4} set. |
| **D3** | New **additive** constant `REGIME_SINGLE_VISUAL = 'single_visual'`, appended to `REGIMES`. On a one-visual run the step class is `single_visual` when `K_resp = 0` and `inpainting` when `K_resp > 0`; `REGIME_SINGLE` is never returned. `_sole_visual` and `_infer_regime` apply the same rule, so `regime == inpainting ⇔ some signal stream keeps > 0 tokens` holds in **both** stream topologies. | `REGIME_BI_VISUAL` is documented as "**both** visuals carry tokens", which a `tir`-only step is not — a `tir`-only Mode-1 step is not `bi_visual` and not `single` either. Keeping `inpainting` for Mode 2 preserves the ADR-005/006 invariant above, makes `is_single_visual` a genuine sensor-absent indicator instead of a constant 1.0, and leaves `mode`/`mode_by_counts` untouched as the authority. Requires widening the `REGIME_INPAINTING` docstring (its "both visuals" clause is topology-specific, its `K_resp > 0` clause is not). `mask_policy_report.py` reads only `mode`/`mode_by_counts`, so it is unaffected. |
| **D4** | New knob `physio_draw_min` (`0` neutral, `1` for ADR-007) rather than hard-coding the §2.5 draw. | §2.5 draws `Uniform{1..C_p}` and forbids Mode-2→Mode-1 leakage; hard-coding it would change every ADR-005/006 mask sequence (DoD 8). Default 0 preserves the old draw *and* the old fall-through. |
| **D5** | `S_core` is drawn once per **step** (after the count draw) and shared by the batch; only the *indices* are per sample. | §2.4(6) + §2.5's algorithm order. Note the consequence for G2/G3: the per-step `S_core` is uniform over `{1,2}`, so its batch mean is 1.5. |
| **D6** | `clip_stride: 4.0` (ADR silent) to match the ADR-005/006 comparator's clip inventory. | Tier-1 comparison requires a matched corpus; a different hop would confound the arm. Must be recorded in the experiment protocol. |
| **D7** | `E_warm = 20` per §2.7, while the ADR-005/006 arms run `E_warm = 50`. | §2.7 is normative here, but this **is** a confound for the Tier-1 multi-stream comparison. Mitigation: the matched uniform-sampling comparator (§3.2(a)) must also use `E_warm = 20`, so at least the anatomy-aware claim is clean; the multi-stream comparator's delta must be reported with the schedule difference named. Flag for review before launching 5 folds. |
| **D8** | `plan_from_counts` takes an optional `kept_core`, which defaults to `core_tubes_min` **only when the policy is `anatomy_aware`** (§1.5). | A pinned allocation cannot infer a random draw, so the value must be stateable — but on a `uniform` checkpoint `core_units` must stay `None` and an explicit `kept_core` must raise, otherwise a uniform ablation would report a fabricated `S_core`. |
| **D9** | The decoder conditioning stays mechanism (a) — prefixing. | §2.4(8) allows (a) or (b); (a) already exists and is recorded in `masking.json`. No new code, no new risk. |
| **D10** | A `[mask] OFF-ADR-007` banner + `adr: 'off-adr-007'` in the manifest whenever a 1-visual run is not (anatomy-aware ∧ `physio_draw_min = 1`). | The parser defaults are neutral for DoD 8, which means `--streams tir,resp --n_budget 200` typed without the YAML silently runs `uniform` sampling with the `Uniform{0..100}` draw. Nothing in a log would distinguish it from the ADR's arm, and that is how a wrong ablation gets published. |
| **D11** | Exactly one existing test is amended: `test_budget_masking.py::test_infeasible_configurations_raise`. Its 1-visual row moves from "stream count" to "high-masking ceiling" (at `n_budget = 350` the same tuple now fails §2.6 rule 4: `14/36 = 38.9 % > 0.25`), and the 3-visual row's needle becomes the widened message `'one or two visual streams'`. Everything else in that file — G1/G8 sampling, caps/floors/alignment, replay — is untouched. | Widening `n_visual != 2` to `n_visual in (1, 2)` necessarily changes which error a 1-visual tuple produces; keeping the old assertion would mean keeping the old rule. The alternative (leaving the message text alone) is rejected because it would misdescribe the engine. Both amended rows must be declared in the commit message so the neutrality claim stays auditable. |

---

## 7. Executed record (2026-11-25)

- **Phase 0 fixture:** `code/tests/fixtures/adr007_neutrality_fixture.json` — 4 epoch contexts
  (`None`, `0`, `50`, `149`) × 50 000 plans = **200 000** plan signatures + per-context SHA-256, first
  plan, trailing plan and the generator's **next raw draw**, captured on the pre-change engine. The
  capture/verify tool is `code/scripts/project/capture_adr007_neutrality.py`
  (`--check` re-runs it; it records the SHA-256 of `core/budget_masking.py` and
  `core/mask_placement.py` inside the fixture).
- **Amended test (D11):** in `tests/test_budget_masking.py::test_infeasible_configurations_raise` the
  1-visual row moved to the needle `'condition 4'` (at `n_budget = 350` it now fails the high-masking
  ceiling, `14/36 = 38.9 % > 0.25`) and the 3-visual row to `'one or two visual streams'`; a 0-visual row
  was added. Both rows carry the reason inline.
- **G1–G13:** all green — `tests/test_two_stream_anatomy_masking.py` (**31 passed**). Whole tree:
  **876 passed** (843 before this change). G8 additionally re-verified out-of-band with
  `capture_adr007_neutrality --check` → *"4 epoch context(s), 200000 plans and every trailing draw are
  bit-identical"*.
- **Measured numbers behind the gates:** Mode 1 = `200/0`, `S_tir = 8`, core `1-2`; Mode 2 = exactly
  `25/50/75/100` physiological tokens at 25 % each with `S_tir ∈ {7,6,5,4}` and **zero** Mode-2 → Mode-1
  leakage; `A_core` resolves to `[[2,2],[2,3],[3,2],[3,3]] → [14,15,20,21]`, `|A_periph| = 32`;
  77.78 % of `tir` tubes masked in Mode 1; respiration loss alone produces non-zero gradients on all
  encoder parameters at `K_resp = 0`.
- **Spec regeneration:** `write_specs.py` rewrote `02_model_architecture.md`; `03/04/05` came out
  byte-identical. `GLOSSARY.md` gained `A_core`/`A_periph`, `S_core`/`S_periph`, `m_periph`,
  `S_tir^min`, `spatial_sampling`, `physio_draw_min`, `REGIME_SINGLE_VISUAL` and the
  cap-vs-ceiling note on `video_cap_ratio`.
- **Smoke (G11) — PASSED on fold 0 of the real corpus** (2026-11-25, one local GPU, `--dev_smoke
  --max_entries 100 --batch_size 8 --num_workers 0`, caches `output/tir_roi_cache` + `output/resp_cache`,
  `--raw_root /work/projects/l0003511/test`, 1 epoch = 12 steps, 100 val samples):
  - **Mode 1 forced** (`--p_virtual_sensing 1.0`): every step `kept_tir = 200`, `kept_resp = 0`,
    `kept_units_tir = 8`, `kept_core_tir` mean **1.42** (∈ {1,2}), `kept_periph_tir` 6.58,
    `mode/mode_resolved/is_virtual = 1`, `is_single* = 0`, `n_batch = 200`, `mse_resp 1.82` with
    `K_resp = 0`. `[resource]` peak **3.62 GB allocated / 3.99 GB reserved**, precision `none`,
    2.19 s/it (1.27 s of it data).
  - **Mode 2 forced** (`--p_virtual_sensing 0.0`): `K_resp ∈ {25,50,75,100}` (epoch means 56.25),
    `kept_tir = 200 - K_resp`, `kept_core_tir` 1.58, `kept_periph_tir` 4.17, `mode_resolved = 2`,
    `n_batch = 200` on every step; peak **3.62 / 5.00 GB**; 2.00 s/it.
  - **`uniform` ablation** (`--spatial_sampling uniform`): the `[mask] OFF-ADR-007` banner fires,
    `kept_core_* = 0` on every step, and `masking.json` records `adr: 'off-adr-007'`,
    `core_coords: null`, `core_indices: []` — G12/D10 on real data.
  - **`masking.json` (arm):** `adr: 'ADR-007'`, `video_cap_units {'tir': 8}`, `physio_cap_tokens
    {'resp': 100}`, `core_coords [[2,2],[2,3],[3,2],[3,3]]`, `core_indices [14,15,20,21]`,
    `periph_tubes 32`, `s_tir_min/max 4/8`, `conditioning: prefix`, `signal_tokens_per_visual_token 10`,
    kernel `tir` with **148** loaded tensors (I1/I2: sole visual recipient).
- **Measured FLOP deviation from ADR-007 §3.3 (report, do not silently accept).**
  `torch.utils.flop_counter.FlopCounterMode`, ViT-B (D=768, 12 blocks), B=1:
  whole encoder **33.97 GFLOP at 200 tokens vs 59.45 GFLOP at 350 ⇒ 0.571×** (a 43 % reduction), not
  the `(200/350)² ≈ 0.33×` the ADR anticipates. The quadratic *term* does scale as 0.327×, but at this
  width and sequence length it is only ~4 % of block FLOPs — the linear projections (`8ND²` in attention,
  `16ND²` in the MLP) dominate. Analytic attention-only ratio: `12.80/24.33 = 0.526`. The ADR's
  "~67 % reduction in encoder attention FLOPs" is therefore **not** confirmed; the defensible claim is
  "≈43 % fewer encoder FLOPs per step, and a quadratic-term reduction of 3.1×". Flagged for an ADR §3.3
  correction before any thesis sentence quotes it.
- **Code-review follow-up (2026-11-25, post-implementation).** Seven findings, all fixed:
  - **#4 (Medium, correctness)** — the `raw[first] = t` clamp in `_mode2` could mutate a count *out of*
    the modulo alignment on a two-signal tuple that opted into `physio_draw_min > 0` (raising later inside
    the sampler), and could exceed a `C_p < T` cap. The clamp is **removed** (an out-of-range result now
    raises as a named invariant violation instead of being silently "fixed"), and `_validate` rejects
    `physio_draw_min > 0` unless **every** `C_p >= T` — on the two-visual topology too. Regression tests:
    `test_draw_floor_on_two_signals_never_breaks_the_alignment` (3 000 steps, sum/alignment/caps) and
    `test_draw_floor_rejects_a_cap_below_T_on_two_visuals`.
  - **#7 (Medium, coverage)** — added the integration boundary the gates only touched through
    sub-components: `test_g9_manifest_records_the_adr007_policy`,
    `test_g12_uniform_manifest_nulls_the_core_fields`,
    `test_g8_two_visual_manifest_keeps_its_key_set` (the resume guard compares key SETS),
    `test_phase43_validation_reports_stratification_not_components` +
    `test_phase43_uniform_run_reports_no_stratification` (which also re-assert
    `set(components) == {'mse_tir','mse_resp'}`), and `test_g13_report_main_writes_the_stratified_payload`
    (runs `mask_policy_report.main()` end-to-end over 200 replayed steps).
  - **#5** — `kept_core`/`kept_periph` are now keyed over `plan.core_units` only, so a 1-D stream never
    gets a `core_tubes_resp` metric and a `uniform` run reports an empty `stratification`. Verified on the
    real run: the validation line is now `kept tubes core_tir=1.40 periph_tir=6.60` (the `core_resp=0.00`
    noise is gone).
  - **#2** — `masking.json` nulls `core_row_bounds`/`core_col_bounds` under `uniform` (the tube-count knobs
    stay, they document what a stratified run would use).
  - **#3** — the unreachable key check in `BudgetPlan` was replaced by a **symmetry** check, moved ahead of
    the per-stream loop so an asymmetric `core_units`/`periph_units` pair reports the missing key instead of
    a misleading sum mismatch.
  - **#6** — `_resolve_core_grid` moved from `_validate` into `_validate_single_visual`, between rules 5 and
    5b, so a malformed `core_*_bounds` can no longer preempt rule 1.
  - **#1** — `_parse_int_csv` accepts an already-parsed `[2, 3]` sequence (a legal YAML spelling) and names
    the offending flag on failure.
  - Suite after the fixes: **886 passed** (+10), neutrality fixture still *200 000 plans bit-identical*,
    and the forced-Mode-1 smoke re-run on fold 0 reproduces the pre-fix numbers (1.42 core tubes,
    3.62/3.99 GB).
- **Still open:** none for this plan. The 5-fold arm itself is a run-protocol deliverable
  (`code/notes/plan/experiments/`), not part of these phases.
- **Rollback triggers** (ADR-007 §4): Criterion-1 failure; landmark audit showing the nostrils fall outside
  `A_core` (then re-centre `core_row_bounds`/`core_col_bounds` before training); Stage-2 collapse to the
  conditional mean (`σ(ŷ) < 0.50·σ(y_ref)`) despite a passing gradient check; or Tier-1 failing the
  `+0.02` margin against either comparator (then fall back to `spatial_sampling = uniform`).

---

## 8. BDD Test Cases

Executable acceptance scenarios mapping directly onto gates **G1–G13** (to be implemented in `code/tests/test_two_stream_anatomy_masking.py` and verified by the Phase-6 smoke checks).

**Scenario 1: Mode-1 pure virtual sensing maintains the 200-token budget and 77.8% visual masking (G1, G2)**
* **Given** a `ConstantBudgetApportioner` and `MultiModalMAE` configured with `streams=('tir', 'resp')`, `n_budget=200`, `spatial_grid=(6, 6)`, `tokens_per_unit=25`, `physio_cap_ratio=0.40`, `video_min_tubes=4`, `spatial_sampling='anatomy_aware'`, `physio_draw_min=1`, and `p_virtual_sensing=1.0`
* **When** `sample_step` is called for `100,000` steps and a forward pass runs on a mini-batch `[B, 3, 50, 96, 96]` and `[B, 1, 8000]`
* **Then** every step yields `K_tir == 200`, `K_resp == 0`, `S_tir == 8`, `mode_by_counts == MODE_VIRTUAL_SENSING`, and `regime == REGIME_SINGLE_VISUAL`
* **And** the encoder input tensor has exact shape `[B, 200, 768]` with `77.78%` (`28 / 36`) of `tir` tubes masked

**Scenario 2: Mode-2 waveform inpainting yields an exact uniform discrete distribution with zero leakage (G1, G2)**
* **Given** the ADR-007 apportioner configured with `p_virtual_sensing=0.0`, `physio_cap_ratio=0.40` (`C_p = 100`), and `physio_draw_min=1`
* **When** `sample_step` is called for `100,000` steps
* **Then** every step satisfies `K_tir + K_resp == 200`, `mode_by_counts == MODE_INPAINTING`, `regime == REGIME_INPAINTING`, and `K_resp > 0` — with `physio_draw_min = 1` the `K_tilde = 0` draw does not exist, so no Mode-2 step can leak into Mode 1
* **And** `K_resp` takes values strictly in `{25, 50, 75, 100}` at `25% ± 1%` each, while `S_tir` takes corresponding values in `{7, 6, 5, 4}`

**Scenario 3: Anatomy-aware stratified tube placement guarantees core visibility and per-sample diversity (G3, G5)**
* **Given** `core_row_bounds=(2, 3)` and `core_col_bounds=(2, 3)` on a `6 × 6` spatial grid (`A_core = {14, 15, 20, 21}`, `|A_periph| = 32`) with `core_tubes_min=1`, `core_tubes_max=2`, and batch size `B = 16`
* **When** `make_masks` draws a step plan from `counts_generator(seed, epoch, step)` and per-sample placements from `placement_generator(seed, epoch, step, rank)`
* **Then** all `B` samples share the exact same `(S_tir, S_core, S_periph)` counts with `S_core ∈ {1, 2}` and `S_core + S_periph == S_tir`
* **And** every sample's kept spatial tube set contains exactly `S_core` distinct indices from `A_core` (`P(core hit) == 1.0`) and `S_periph` distinct indices from `A_periph` with zero duplicates and zero cross-stratum leakage
* **And** replaying `(seed, epoch, step)` on an independent process reproduces identical counts while kept tube indices vary across samples within the batch

**Scenario 4: Two-phase virtual-sensing curriculum switches at 0-based epoch `E_warm = 20` (G4)**
* **Given** the ADR-007 apportioner with `virtual_sensing_warmup_p=0.10`, `p_virtual_sensing=0.50`, and `virtual_sensing_warmup_epochs=20`
* **When** `10,000` steps are sampled at epoch `19`, epoch `20`, and `epoch=None`
* **Then** `p_at(19) == 0.10`, `p_at(20) == 0.50`, and `p_at(None) == 0.50`
* **And** realized Mode-1 fractions match `0.10` at epoch `19` and `0.50` at epoch `20` within a 4σ binomial interval

**Scenario 5: Mode-1 respiration reconstruction propagates non-zero gradient to visual encoder latents (G6)**
* **Given** a `MultiModalMAE` built with `streams=('tir', 'resp')`, `n_budget=200`, and `p_virtual_sensing=1.0`
* **When** a synthetic forward pass runs with a `retain_grad()` hook on the shared encoder output `z` (after `enc_norm`) and `out['losses_mse']['resp'].backward()` is executed
* **Then** both `'tir'` and `'resp'` keys are present in `out['losses_mse']` with `out['losses_mse']['resp'] > 0`
* **And** `z.grad` is strictly non-zero (`‖∂L_resp/∂z‖ > 0`), confirming cross-modal conditioning via decoder prefixing under complete sensor blackout

**Scenario 6: Construction-time feasibility enforces §2.6 in normative rule order and accepts non-multiple `C_p` (G7, D11)**
* **Given** a 1-visual + 1-signal stream set `('tir', 'resp')` with `spatial_grid=(6, 6)` and `tokens_per_unit=25`
* **When** `ConstantBudgetApportioner` is constructed with each violated §2.6 precondition (e.g., `n_budget=210`, `physio_cap_ratio=0.05`, `n_budget=350` at `video_cap_ratio=0.25` and `video_min_tubes=2`, `core_tubes_max=5`, missing `spatial_grid` under `anatomy_aware`, or `p_single_modality=0.20`)
* **Then** construction raises `ValueError` naming the first violated condition in rule order `1 → 2 → 3 → 4 → 5 → 5b → 6` (specifically failing rule 4 high-masking ceiling on `n_budget=350, video_min_tubes=2`)
* **And** constructing with a non-multiple physiological cap (`physio_cap_ratio=0.45`, `C_p=112`, `video_min_tubes=3`) succeeds and realizes `S_tir ≥ 4` via the downward remainder-alignment branch without runtime errors

**Scenario 7: Two-visual ADR-006 mask sequences remain bit-identical and RNG-neutral (G8)**
* **Given** the Phase-0 pre-change fixture of `50,000` ADR-006 (`tir`, `rgb2d`, `resp`) plans and trailing generator draw
* **When** the updated `ConstantBudgetApportioner` is constructed with the ADR-006 stream set and default values for all seven new ADR-007 parameters
* **Then** all `50,000` sampled plans and the generator's next raw draw are bit-identical to the fixture
* **And** `test_mask_placement.py`, `test_curriculum_schedule.py`, and `test_regime_single_carrier.py` pass without modification

**Scenario 8: Evaluation pinning enforces stratified vs uniform checkpoint contracts (G10)**
* **Given** an ADR-007 `anatomy_aware` checkpoint and a `uniform` checkpoint (either ADR-006 or the ADR-007 uniform ablation)
* **When** `pin_masking` in `runners/run_eval_recon.py` runs with `--kept_tir 0.0 --kept_resp 0.0 --resp_probe hidden` on the ADR-007 checkpoint
* **Then** it produces a valid zero-residual plan with `K_tir == 200`, `K_resp == 0`, `core_units['tir'] == 1`, `periph_units['tir'] == 7`, and records `spatial_sampling`, `S_core`, and `S_periph` in the caption
* **And** passing `--pin_core_tubes 2` against a `uniform` checkpoint raises `ValueError` instead of fabricating a core count

**Scenario 9: Uniform two-stream ablation records clean manifest metadata and flags off-ADR-007 runs (G9, G12, D10)**
* **Given** a one-visual run launched with `streams='tir,resp'`, `n_budget=200`, and `--spatial_sampling uniform` (or `--physio_draw_min 0`)
* **When** `run_pretrain.py` initializes the model, writes `masking.json`, and runs one training + validation step
* **Then** stdout prints the `[mask] OFF-ADR-007` banner and `masking.json` records `adr: 'off-adr-007'`, `spatial_sampling: 'uniform'`, `core_coords: null`, and `core_indices: []`
* **And** no core stratification RNG draw occurs, `kept_core['tir'] == 0`, and `evaluate_pretrain` reports tube counts in sibling `val['stratification']` while leaving `val['components']` strictly loss-only

**Scenario 10: Offline mask-policy replay verifies zero Mode-2 leakage and stratified counts (G13)**
* **Given** a checkpoint produced by `configs/pretrain/stage2_budget_tir_resp_2stream_anatomy.yaml` (`physio_draw_min=1`, `spatial_sampling='anatomy_aware'`)
* **When** `scripts/project/mask_policy_report.py` replays the run offline
* **Then** all invariant checks (`exact_sum`, `visual_multiple_of_T`, `visual_inside_floor_cap`, `signal_inside_cap`, `mode_iff_signal_zero`) pass
* **And** `mode2_k_tilde_zero_share.expected` equals `0.0` (matching observed `0.0`), and the report includes the replayed `S_core` / `S_periph` summary
