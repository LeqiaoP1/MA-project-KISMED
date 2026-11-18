# AMMM (Asymmetric Multi-Modal Masking) — implementation plan

> **Scope.** The Stage-2 masking engine described in
> `code/notes/design/Asymmetric_MultiModal_Masking.md`: up to 4 decoupled
> streams (`rgb`, `tir`, `bp`, `resp`) — but the engine must work for **any**
> valid subset, e.g. `tir,resp` (current study) or `rgb,bp` (next), with the
> 4-stream case as the design superset. Dirichlet budget allocation per step, a
> **kept-ratio cap on the visual streams** (0.25 in the design), uncapped 1-D
> physio kept ratios, a **visuo-physiological co-existence guard** and
> batch-level modality dropout.
> **This is the only plan for this study.** *Why* the components exist is the
> design doc; measurements belong in `code/notes/eval/`.
> **State.** **Phase 1 implemented (2026-11-08)**: `core/ammm_masking.py` (pure
> policy module), `MultiModalMAE` wiring behind `masking: ammm`, 44 tests in
> `tests/test_ammm_masking.py`, and the arm-A config
> `configs/pretrain/stage2_ammm_tir_roi_resp.yaml` (a 1-variable clone of
> `stage2_local_tir_roi_resp.yaml`: only the masking policy differs).
> `masking: ratio` is byte-identical and stays the default. Phases 2-5 are NOT
> started: `modality_dropout > 0` and `physio_mask: span` under `ammm` raise
> `NotImplementedError` on purpose. The arm-A smoke (1 epoch, 4 clips, GPU,
> ROI + respiration caches) runs and logs the plan:
> `kept_tir 1, kept_resp 21..48, n_batch 98..160`, `mse_tir 2.11, mse_resp 2.58`
> on the first step. Engine semantics are settled (design rev. 2026-11-08);
> only D5/D6 (§6.3) remain open.
> **k-fold harness (2026-11-08).** The HPC twin
> `configs/pretrain/stage2_ammm_hpc_tir_roi_resp.yaml` (96 px / 2 s hop /
> 150 epochs, a 1-variable clone of `stage2_hpc_tir_roi_resp.yaml`) is the arm
> of the 5-fold sweep; `scripts/hpc/submit_ammm_pretrain_kfold.sh` preflights
> and queues ONE job per fold (`--pilot <EPOCHS>` for the fold-0 timing run),
> and `notes/eval/tir_resp/ammm_pilot_report.py` turns a train log into epoch
> time, the 150-epoch projection and a flat/still-descending verdict.

---

## 1. Question

Does an asymmetric kept-ratio policy — video capped at 25 % kept, 1-D physio
uncapped, budget drawn per step from a Dirichlet — beat the current fixed
per-stream `mask_ratio_*` policy on the downstream reconstruction number?

The deliverable is a **new, switchable masking engine plus the geometry it
needs**, not a change to the existing one: `masking: ratio` stays the default and
every shipped config keeps its behaviour bit-for-bit.

**Judged by** the Stage-3 subject-level reconstruction metric
(`design/Evaluation_RespWaveform_protocol.md`), A/B against the shipped
`masking: ratio` lineage at equal compute — the masking knob is the only
variable.

---

## 2. Design vs. current code — what already exists

| design requirement | current code | status |
|---|---|---|
| 4 decoupled streams `rgb,tir,bp,resp` | `MultiModalMAE(streams=...)`, `STREAM_CHANNELS`; `PairedPretrainDataset` can serve all five | **OK** |
| **any** valid stream combination (`tir,resp` now; `rgb,bp`; all four later) | `streams` is a constructor argument; the Stage-2 contract only demands >=1 visual + >=1 signal | **OK** — the new engine must keep it that way |
| encoder receives only visible tokens (`[B,N_batch,D]`, zero `[MASK]`) | `forward` gathers `ids_keep` per stream and concatenates into `z` | **OK** |
| shared encoder + per-stream decode | `enc_blocks` / `dec_blocks` / `heads[s]` | **OK** |
| space-time alignment (`tubelet_t·stride/fps == sig_kernel/fs`, `Gt == n_signal`) | `_check_space_time_alignment` (hard `ValueError`) | **OK**, reusable as-is |
| tube masking (same spatial patches over all frames) | `_tube_mask` | **OK**, but parameterised by *mask* ratio |
| Dirichlet budget per step | — | **missing** |
| video **kept** cap 0.25 (i.e. mask ≥ 0.75) | `mask_ratio_*` is a *mask* ratio and is per-stream constant (`rgb 0.75 / tir 0.50 / bp 0.90`) | **missing** (semantics differ) |
| 1-D kept ratio uncapped (up to 100 % kept), and 0 kept allowed | `_random_mask` clamps masked ≤ `N-1` ⇒ always ≥ 1 visible, and is parameterised by *mask* ratio | **gap** (see §6 D4) |
| modality dropout + co-existence guard | — (a run's `streams` are fixed for the whole job) | **missing** |
| batch-dense identical token count | counts per stream are constants of the run ⇒ already true within a batch | **OK**, but they must stay constant *within* a step and become free *across* steps |
| RGB 128 px (8×8) **and** TIR 96 px (6×6) in one model | one `input_size` for every visual stream (`__init__`, `_tube_mask`, `positions`, `_init_pos_embeds`, `_targets`, `heads._flat`) | **missing** |
| `T=25` tokens (tubelet_t 4 @ 12.5 fps, `sig_kernel` 32 @ 100 Hz) | shipped lineage is `Gt=50` / `sig_kernel 16` | **config value** — differs per experiment, as intended |
| Stage-1 VideoMAE conv init with tubelet_t 4 | `fit_visual_patch_embed` **raises** when checkpoint tubelet ≠ model tubelet | **code gap** (see §6 D5) |

Three conclusions that shape the plan:

1. **The masking engine can land without touching geometry.** It needs only
   "how many tokens to keep per stream", and the existing `_tube_mask` /
   `_random_mask` / `_span_mask` already produce exact counts. Phase 1–2 is
   therefore low-risk and testable on today's `Gt=50` configs.
2. **The 128 / 96 split is a model-wide refactor.** One `input_size` per *run*
   is assumed in five places inside `MultiModalMAE` and mirrored in
   `core/waveform_model.py` / `core/au_probe.py`. That is a code limitation, not
   a value conflict — the second size has nowhere to be written yet.
3. **It ships as one stream-count-agnostic module** with the 4-stream design as
   the superset it must satisfy, never as a hardcoded 4-element vector: a
   2-stream run must exercise the same code path as the 4-stream run.

---

## 3. Contract (values are experiment config, not code defaults)

**Configuration policy.** Every number below is an *experiment parameter*: an
AMMM run declares its own geometry and masking values in its YAML and is
allowed to disagree with the shipped Stage-2 lineages, with cached artefacts
and with earlier AMMM runs. Nothing in this plan has to stay compatible with a
previous checkpoint — a new geometry simply means a new checkpoint, and its
Stage-3 sibling declares the same numbers.

Consequences that make this cheap:

* **No cache rebuild.** `data/roi_cache.py::cache_key` deliberately excludes
  `input_size` and `clip_stride` (they only resize / re-window), and
  `data/resp_preprocess.py` is independent of the video geometry. Moving from
  64/96 to 128/96 therefore reads the **existing** ROI and respiration caches.
  Only knobs that change WHICH PIXELS the hull spans (`roi_landmarks`,
  `roi_padding`, `crop_rule`, corpus, validity filters) would invalidate a
  cache, and AMMM does not touch them.
* **Code defaults stay legacy.** `masking: ratio`, `tubelet: 2,16,16`,
  `input_size: 64` remain the defaults so every existing config and test keeps
  its behaviour; AMMM values are only what `configs/pretrain/stage2_ammm_*.yaml`
  sets.
* **Loud mismatch, not silent drift.** `load_stage2_encoder` / the ROI-contract
  keys already abort when a Stage-3 sibling disagrees with its Stage-2
  checkpoint, so a stale sibling cannot silently train.

Geometry implied by the design (verified consistent with
`_check_space_time_alignment`):

| item | value | configured by |
|---|---|---|
| clip | 8.0 s, source 25 fps, `temporal_stride 2` → 12.5 fps, `num_frames 100` | `clip_duration`, `fps`, `temporal_stride` |
| tubelet | `(4, 16, 16)` → `Gt = 25` | `tubelet` |
| RGB | `128` px → `Gh=Gw=8` → 64 spatial patches → 1600 tokens | `input_size_rgb` (new) |
| TIR | `96` px → `Gh=Gw=6` → 36 spatial patches → 900 tokens | `input_size_tir` (new) |
| 1-D | `fs 100`, `sig_kernel 32`, `seq_len 800` → `n_signal 25` (320 ms/token) | `fs`, `sig_kernel`, `seq_len` |
| alignment | `4·2/25 = 32/100 = 0.32 s`; `Gt 25 == n_signal 25` ✔ | checked at construction |

Masking parameters (new CLI/YAML keys, all inert unless `masking: ammm`):

| key | default | meaning |
|---|---|---|
| `masking` | `ratio` | `ratio` = today's path (untouched); `ammm` = new engine |
| `dirichlet_alpha` | `1.0` | shared α over the *active* streams (float, or `{stream: alpha}`) |
| `video_kept_cap` | `0.25` | default cap for **every visual** stream; `{rgb: 0.25, tir: 0.15}` overrides per stream |
| `physio_kept_cap` | `1.0` | default cap for **every signal** stream (cap exists for ablations only) |
| `modality_dropout` | `0.0` | probability a stream is dropped from the step |
| `min_kept` | `0` | floor for **Dirichlet-drawn** streams only; `0` = a stream may be fully masked (default), `1` = the historical "every active stream reaches the encoder" variant. Overridden streams are exempt |
| `kept_ratio_override` | `{}` | fixed `{stream: r}`, used **verbatim** and outside the Dirichlet (validated `0 ≤ r ≤ cap_s`). Deterministic way to pin a stream at `0.0` (100 % masked) or `1.0`; exempt from `min_kept` |

All of these are keyed **by kind with per-stream overrides**, never by a fixed
4-element vector — a `tir,resp` run gets the same caps as an `rgb,tir,bp,resp`
run without touching code.

`Dirichlet(α=1)` almost never lands exactly on 0 for one component, so
"TIR fully masked, Resp visible" as a *reproducible arm* is expressed with
`kept_ratio_override: {tir: 0.0}`, not by hoping for a small draw. Stochastic
0-kept draws still happen on their own (`round(r·n)` rounds to 0 for small
`r`) — the override is for pinning an extreme deliberately.

Kept/masked conversion (design → code; `mask == 1` means *to reconstruct*).
This is the **4-stream superset** of the engine; the module derives every row
from the stream's own `n_units` (see Phase 1), it does **not** hardcode 64/36/25:

| stream | `n_units` | kept range (inclusive) | kept tokens | implied mask ratio range |
|---|---|---|---|---|
| rgb | 64 spatial | `0 … min(round(r·64), 16)` | 0 … 400 | 1.00 … 0.75 |
| tir | 36 spatial | `0 … min(round(r·36), 9)` | 0 … 225 | 1.00 … 0.75 |
| bp / resp | 25 | `0 … round(r·25)` | 0 … 25 | 1.00 … 0.00 |

Both ends are reachable on purpose: **0 kept = 100 % masked** (legal for any
modality, design §3.2.3), and full kept = 0 % masked (legal for a physio
stream, whose masked-MSE term is then an explicit 0).

Encoder upper bound `400 + 225 + 25 + 25 = 675` tokens.

> Note: AMMM is a **kept** cap, so on a video-dominant draw it can keep *more*
> than the shipped configs (which mask 0.75–0.90). This is a genuine
> behavioural change, not a renaming — it must be an arm of an experiment, not
> a silent rewrite.

---

## 4. Build order

### Phase 0 — write the AMMM config (the parameter set)

No code. `configs/pretrain/stage2_ammm_*.yaml` is the single place the design's
numbers live (geometry §3 + masking keys). Because values are per-experiment,
this file is free to disagree with `stage2_hpc_tir_roi_resp.yaml` etc.
Engine semantics are settled for D3/D4 (§6.2); only **D5** (Stage-1 conv init)
and **D6** (budget leakage) remain open, and neither blocks Phase 1.

### Phase 1 — the AMMM sampler (policy only, stream-agnostic)

**Design goal for this module: high cohesion + any modality combination.**
It answers exactly one question — *how many units does each stream keep this
step* — for **whatever streams the run declares**. `tir,resp` today,
`rgb,bp` next, `rgb,tir,bp,resp` later: same module, same code path, only the
`specs` list and the config values differ.

New module `code/core/ammm_masking.py`, with **no** torch modules, no
parameters, no tensor shapes and **no knowledge of modality names**:

```
@dataclass(frozen=True)
StreamSpec:
    name: str            # 'tir', 'resp', whatever the run declares
    kind: str            # 'visual' | 'signal'  (taxonomy owned by core/multimae.py)
    n_units: int         # masking unit count: spatial patches (visual) or tokens (signal)

@dataclass(frozen=True)
MaskPlan:
    active: tuple[str, ...]                 # streams present in this step
    kept_units: dict[str, int]              # per active stream, in [min_kept, limit]
    r: dict[str, float]                     # realised kept ratio after the cap
    dropped: tuple[str, ...]                # dropout result (before the fallback)

plan_step(specs: Sequence[StreamSpec], *, caps: Mapping[str, float],
          alpha: Mapping[str, float] | float, dropout_p: float = 0.0,
          min_kept: int = 0, generator: torch.Generator | None = None) -> MaskPlan
```

Kind-agnostic rules, in this order (nothing indexes `[0]/[2]` of a vector):

1. **dropout** — each stream dropped with `p` (`0.0` ⇒ all active);
2. **co-existence fallback** — if no `kind='visual'` stream survives, re-activate
   the FIRST visual in `specs` order; same for `kind='signal'`. The target
   exists by the declared-set precondition (design §3), which `plan_step` must
   validate itself and raise `ValueError` (never `IndexError`) —
   `MultiModalMAE.__init__` already rejects such a stream list, so the module
   check is the second line of defence the tests exercise. A single-member kind
   can therefore never be dropped; **log that** once so the effective dropout
   distribution is visible (design §3.1.3);
3. **Dirichlet** over the DRAWN streams only, `α_s` per stream, one draw per
   **step**. Let `drawn = active − override`:
   * `|drawn| ≥ 2` → `Dir(α_drawn)`, dimension = `|drawn|` (never 4);
   * `|drawn| == 1` → that stream is drawn `Uniform(0, 1)` (a 1-D Dirichlet is
     a point mass at 1.0 and would pin it to its cap every step); log once;
   * `|drawn| == 0` → no draw at all (an empty Dirichlet is undefined); every
     ratio is the configured override, and rule 6 must not depend on `λ`;
   `kept_ratio_override` values are used verbatim and are validated at
   construction (`0 ≤ r_s ≤ cap_s`, else `ValueError`);
4. **asymmetric mapping** — `r_s = min(u_s, cap_s)` where `u_s` is the drawn
   `λ_s` or the pinned override (already validated `≤ cap_s`); `cap_s` = the
   per-kind default (`video_kept_cap` / `physio_kept_cap`) overridden per stream;
5. **discretise** — `limit_s = n_units` for a signal stream and
   `min(n_units, floor(cap_s * n_units))` for a visual stream, then
   `drawn`: `clamp(round(r_s·n_units), min_kept, limit_s)`;
   `overridden`: `clamp(round(r_s·n_units), 0, limit_s)` — **`min_kept` does NOT
   apply**. `min_kept` defaults to `0`, and **0 kept is a legal value for any
   modality** (design §3.2.3): a stream can be 100 % masked and contributes no
   encoder token that step. `min_kept: 1` is an opt-in strict variant for
   *drawn* streams; an explicit `kept_ratio_override: {tir: 0.0}` must survive
   it, or the ablation is silently corrupted. Concretely: `tir` 36 → 0..9,
   `resp` 25 → 0..25, `rgb` 64 → 0..16 — the table in §3 follows from `n_units`;
6. **global visibility guard** — if every active stream rounds to 0, grant ONE
   unit to the stream with the largest `λ_s` (ties → declared order) so
   `Σ kept_s ≥ 1` and the encoder never receives an empty `[B, 0, D]`
   (design §3.2.3 / §5.7). When a stream's ratio comes from
   `kept_ratio_override` instead of the draw, there is no `λ_s` for it — rank by
   that pinned `r_s` instead; if NO stream has a draw or a ratio (all pinned to
   an equal value), take the FIRST active stream in declared order. The guard
   must therefore be total: it can never fail for lack of a comparable number.
   From the **draw alone** an all-zero step is arithmetically impossible here:
   `Σλ = 1` over `k` active streams gives `λ_max ≥ 1/k`, so that stream keeps
   `round(λ_max·n) ≥ round(n/k)` = 6 tokens for `k=4` at `n=25`. In practice the
   all-zero step comes from CONFIG — `kept_ratio_override: {tir: 0.0, resp: 0.0}`
   (an "everything masked" arm) or a `physio_kept_cap ≈ 0` — which is exactly why
   tiers 2–3 exist. It fires only in that degenerate case and is not a per-stream
   floor;
7. **density — constant within a step, free across steps** — the plan produces
   ONE count per stream per step, so every sample of the batch has the same
   `kept_s` (dense `[B, N_batch, D]`, no padding), while `kept_s` and therefore
   `N_batch = Σ kept_s` may change on the next step (new Dirichlet draw). Only
   the **count** is shared; the **placement** must stay independently random
   per sample (`torch.rand(B, n)` inside the mask builders), or the whole batch
   would share one mask and the step would have the statistics of one sample.
8. **determinism** — every draw goes through the injected `generator`, so a test
   can replay a sequence and a caller can isolate mask RNG from model RNG.

Cohesion boundary: this module returns **counts only**. Turning a count into a
mask tensor (tube placement for video, scattered/span placement for 1-D) stays
in `MultiModalMAE`, which is the only place that knows the token grid. That is
what keeps the module testable without a model and reusable if a third stream
*type* appears.

Wiring in `MultiModalMAE` (the only place that knows modality names and token grids):

* `__init__` parses the new knobs (`parse_kept_cap`, `parse_dirichlet_alpha`,
  mirroring `parse_mask_span`) and builds the spec list
  **from `self.streams`**, one entry per declared stream:
  `kind = 'visual' if s in self.visual else 'signal'`,
  `n_units = grid_h*grid_w` (visual) or `self.n_signal` (signal). No stream
  count, name or size is assumed anywhere;
* `make_masks(B, device)` keeps the legacy branch verbatim and dispatches to a
  new `_make_masks_ammm(B, device)`, which calls `plan_step(...)` once per step
  and expands the per-kind counts into masks:
  `_tube_keep_mask(B, n_spatial, kept)` for visual,
  `_random_keep_mask(B, N, kept)` (or `_span_keep_mask`) for signal;
  both are added next to `_tube_mask` / `_random_mask` (kept → mask via
  `mask = 1 - keep`), because `_tube_mask`'s `int(mask_ratio·n)` rounding drifts
  from the design's `round(r·S)`;
* the returned mask dict is keyed by *all* `self.streams` (so callers can index
  it without branching), but a stream absent from `MaskPlan.active` only gets a
  **placeholder** all-ones mask. Activity is NOT recoverable from the mask: an
  ACTIVE stream with `kept == 0` also has an all-ones mask, yet it IS scored
  (G16). `MaskPlan.active` is the single source of truth for "present in this
  step", and `forward` must skip inactive streams explicitly (Phase 2);
* `physio_mask='span'` stays available inside AMMM by deriving `span`/`n_spans`
  from the *sampled* ratio — optional, only if the ablation is wanted;
* new out keys: `out['active']`, `out['kept']`, `out['r']` (logged by
  `engines/pretrain.py` next to `mse_<stream>` / `spec_<stream>`).

### Phase 2 — modality dropout in `forward`

* one availability draw per step (design §3.1.1); a per-sample draw would break
  the dense-gather contract of §5.5;
* streams **not in `MaskPlan.active`** are skipped explicitly
  (`if s not in plan.active: continue`) in **two** loops — tokenise+gather and
  decode. The mask cannot decide this: an inactive stream and an active 0-kept
  stream both carry an all-ones mask. Skipping means no adapter call, no
  `enc_parts` entry, no decoder ⇒ no mask-only gradient ever;
* the **loss loop is NOT skipped** for an inactive stream: it emits
  `losses[s] = losses_mse[s] = 0.0` (detached) and `masked_<s> = 0`, because
  the metric-key set must be config-stable ("loss keys are a function of the
  CONFIG" below, G13). `preds` is the only dict that may omit it (no decoder
  ran). Key policy: `losses` and `losses_mse` carry one entry per declared
  stream (`losses_spectral` was removed with the Stage-2 spectral term on
  2026-11-09). Applies to any number of streams and any combination, including
  5- or 2-stream runs;
* within a step the per-sample visible count must be **identical per stream** —
  assert it (`torch.unique(mask_s.sum(1)).numel() == 1`) instead of silently
  trusting it: the current `k = (mask_s == 0).sum(1).max()` gather LEAKS masked
  tokens into the encoder when counts differ (see
  `design/SpanMask_PhysioSignals.md` §4.3). Keep the batch-max form, add the
  equality assertion in front of it;
* `N_batch = Σ kept_s` therefore varies BETWEEN steps. Nothing may cache a
  static token count (no fixed-size buffer, no shape-specialised kernel), and
  expect allocator churn from the changing sequence length; the model is not
  `torch.compile`d today, and if it ever is, this is a dynamic-shape axis;
* keep `z` non-empty: the **global visibility guard** (Phase 1 rule 6) — not a
  per-stream floor — guarantees `Σ kept ≥ 1`, so `torch.cat` of the per-stream
  gathers is never `[B, 0, D]`. Additionally `forward` asserts
  `z.shape[1] >= 1` immediately before the encoder, so a hand-built mask or a
  future mask source cannot bypass the invariant (AGENTS.md §6: keep explicit
  dimension assertions);
* a stream with `kept == 0` (100 % masked, legal for any modality) needs no
  special branch: `ids_keep` is `[B, 0]`, its `enc_s` slice is empty, and its
  decoder input is `n` mask tokens + positions. It contributes **no gradient to
  the encoder or `adapters[s]`** that step and its masked-MSE is computed over
  all `n` tokens. Assert `k == 0` is handled rather than skipped, and count it
  in `kept_<s>` / `masked_<s>`;
* `find_unused_parameters=True` becomes load-bearing (not just the default):
  `heads`/`adapters`/`positions` stay instantiated while a dropped or 0-kept
  stream gives them no gradient, and the encoder itself may be untouched. No
  `NativeScaler` change is needed (unused params never enter the grad list).
* **loss keys are a function of the CONFIG, never of the step or the rank**
  (design §5.9). Every declared stream emits `losses_mse[s]` and `masked_<s>`
  on every step: a detached `0.0` when the stream is inactive or has no masked
  position. This is not cosmetic — `MetricLogger.synchronize_between_processes`
  issues one `barrier()` + `all_reduce` **per meter**, and drops are drawn from
  the per-rank RNG (`init_env` seeds `seed + rank`), so a rank-dependent key
  set would deadlock the collectives. Emit the zero through the normal
  `losses_mse` path (a detached scalar), not by omitting the key.
* optional, recommended for log comparability: draw the `MaskPlan` on rank 0 and
  broadcast `(active, kept_units)` so all ranks run the same ratios (placement
  stays per-rank random). Not required for correctness once the key set is
  config-stable.

### Phase 3 — per-stream visual geometry (the code change that makes 128/96 expressible)

This phase adds the *knob*, not a value: the design's 128/96 is then just the
AMMM config's setting, and a run that configures one size behaves exactly as
today.

* `input_size` / `tubelet` accept `int` **or** a dict / `stream=value` CSV
  (same parser style as `parse_kept_cap`); new CLI spellings
  `--input_size_rgb / --input_size_tir`;
* the resolved geometry is recorded with the checkpoint (args + a
  `[geometry]` log line) so a Stage-3 sibling can echo it verbatim;
* introduce `self.visual_geom[s] = (Gt, Gh, Gw)` and `self.n_visual_tokens[s]`;
  keep `self.n_visual` / `self.grid_h` / `self.grid_w` as derived properties for
  the single-size case so `core/au_probe.py`, `core/waveform_model.py` and the
  tests do not break;
* update `_check_space_time_alignment` (loop over visual streams),
  `_init_pos_embeds` (3-D sincos per stream grid), `_tube_mask` /
  `_tube_keep_mask` (per-stream `Gh·Gw`), `_targets` (per-stream tubelet),
  `heads` `_flat` (per-stream tubelet), `forward` gather (already per stream);
* `core/waveform_model.py` (`MultiModalWaveformRegressor`,
  `load_stage2_encoder`) and `core/au_probe.py` resolve `input_size`/`tubelet`
  by stream name; Stage-3 stays single-stream so the dict is resolved once.

### Phase 4 — data: one arm per modality combination

The engine is combination-agnostic, so the arms differ **only in `data_set` +
`streams`**; three of the four need no new dataset at all:

| arm | `data_set` / `streams` | dataset | new code |
|---|---|---|---|
| **A — TIR-ROI + Resp** (current study) | `tir_roi` / `tir,resp` | `TirRoiRespPretrainDataset` (exists) | none — `stage2_{local,hpc}_tir_roi_resp.yaml` geometry |
| **B — RGB + BP** (next) | `rgb_roi` / `rgb,bp` | `RGBRoiPretrainDataset` (exists, `signals` = `bp` or `resp`) | none — `stage2_local_rgb_roi_bp.yaml` geometry |
| **C — two visual streams + physio** (design superset) | canonical / `rgb,tir,bp,resp` at one size | `PairedPretrainDataset` (exists) | none; geometry values only |
| **D — two visual streams, two sizes** (design-faithful) | combined ROI / `rgb,tir,bp,resp` with 128 / 96 | **missing** | new `data/combined_roi_dataset.py` (Phase 3 prerequisite) |

Recommended order: **A** (validates the policy on the lineage the project is
already measuring) → **B** (second combination, same code path, proves the
"any subset" claim) → C/D once the policy is settled.

No dataset serves RGB-ROI **and** TIR-ROI together, which is why arm D is the
only one with data work. Limits of the three existing loaders:
`PairedPretrainDataset` (canonical tree) uses ONE `input_size` for both videos;
`TirRoiRespPretrainDataset` serves `tir,resp` only; `RGBRoiPretrainDataset`
serves `rgb` + any physio at `input_size` 224 by default.

Arm D = new `data/combined_roi_dataset.py` (name TBD) joining the RGB face crop
(128) and the TIR mouth+nose crop (96) on `(subject, task, window)`; reuses
`rgb_roi_dataset` JPEG decode and `tir_resp_dataset` ROI crop + `roi_cache`, and
the `resp_cache` / physiology reader. Gate: both crops must be addressable by
the same time origin (`data/alignment.py::plan_clip`) or the streams
desynchronise.

### Phase 5 — configs, tests, docs

* configs, one per arm, each differing only in `data_set` / `streams` /
  geometry: `stage2_ammm_tir_roi_resp.yaml` (A, **shipped 2026-11-08**),
  `stage2_ammm_rgb_bp.yaml` (B), `stage2_ammm_bp4d4.yaml` (C, canonical tree at
  one size), `stage2_ammm_roi4.yaml` (D, after Phase 4). All must state
  `masking: ammm` explicitly and keep `physio_mask`, `target_norm`
  at their Stage-2 defaults (`target_norm: none` for the ROI
  lineages, pinned by `pin_roi_target_norm`);
* `tests/test_ammm_masking.py` (Phase 1, CPU-only, `_tiny_model` style)
  **parametrised over modality combinations** — at minimum `('tir','resp')`
  (the current study), `('rgb','bp')` (the next one), `('rgb','tir','bp','resp')`
  (the design's superset), plus `('tir','resp','bp')` for a 3-stream mix and one
  hypothetical extra stream name to prove the module is name-agnostic.
  Per combination: cap invariant, co-existence fallback, batch-identical counts,
  determinism from a fixed `torch.Generator`;
  extend `core.multimae.mask_self_test` with an `ammm` section so
  `python -m core.multimae` covers it;
* `tests/test_config_loading.py` (YAML→CLI plumbing for the 7 new keys),
  `tests/test_rail_rule_plumbing.py` (config counts if a config is added);
* update `code/notes/spec/02_model_architecture.md` §7.5 (masking table) **by
  hand and in `code/scripts/project/write_specs.py`** — the generator is stale
  (`/memories/repo/docs-and-specs.md`);
* Stage-3 sibling configs carry the SAME geometry values as their AMMM Stage-2
  run (they are experiment parameters, so they are copied, not inherited);
  `load_stage2_encoder` fails loudly on a mismatch, so a stale sibling is a loud
  error, not a silent one.

---

## 5. Invariants and gates

| id | invariant | test |
|---|---|---|
| G1 | visual streams: `kept_s ≤ floor(cap_s · n_units)` (tir 36 → ≤9, rgb 64 → ≤16) ⇒ mask ratio ≥ 0.75 | unit, 10⁴ draws |
| G2 | signal streams: `0 ≤ kept_s ≤ n_units` (no cap); both extremes legal for every stream | unit |
| G3 | co-existence: every step has ≥1 `visual` and ≥1 `signal` active, including `p=1.0` dropout, for **every** tested stream combination | unit |
| G4 | within a step: the per-sample visible count of EVERY stream is identical (`torch.unique(counts).numel() == 1`), and across steps the counts are allowed to differ; the encoder gather contains **no** masked token | unit (`mask_self_test` style) |
| G5 | `N_batch ≤ 675` for the 4-stream superset at its caps (a property of that config, not of the engine) | unit |
| G6 | a fully kept (0 % masked) stream yields a finite loss and an explicit `0.0` under its key, not a missing key | forward smoke |
| G7 | `masking: ratio` reproduces today's masks exactly (`make_masks` untouched) | existing `mask_self_test` + `test_video_input_norm` |
| G8 | dropout step runs under DDP (`find_unused_parameters=True`) on a 1-step smoke | `scripts/local/*.sh` smoke |
| G9 | geometry: `Gt == n_signal` per visual stream after Phase 3 | construction-time `ValueError` |
| G10 | **combination-agnostic**: `tir,resp`, `rgb,bp`, `tir,resp,bp` and `rgb,tir,bp,resp` all run the *same* code path; only `specs` + config differ | parametrised unit + 1-step forward per combination |
| G11 | the study's immediate target `tir,resp` trains for 1 step on the existing local data with `masking: ammm` | local smoke script |
| G12 | a declared set with no visual (or no signal) stream is rejected at construction **and** by `plan_step` with `ValueError` (never `IndexError`) | unit, both layers |
| G13 | every declared stream's loss key exists on every step and every rank; value is exactly `0.0` when inactive / zero-masked, with `masked_<s>` = 0 | unit (2 fake ranks) + `MetricLogger.synchronize_between_processes` smoke |
| G14 | the fallback picks the *first declared* stream of the missing kind (`tir` in a `tir,resp` run, not `rgb`) | unit |
| G15 | global guard: the encoder input is never `[B, 0, D]`; an all-zero draw grants exactly one unit to the max-λ stream, and `forward` asserts `z.shape[1] ≥ 1`; any single stream may still be 0-kept | unit + forward |
| G16 | a 100 %-masked stream: `k = 0` for that stream, all its tokens masked, loss computed on all of them, **no** gradient to `adapters[s]` / encoder from it, and the step survives DDP with `find_unused_parameters=True` | forward + 1-step smoke |
| G17 | placement is per-sample: within one step the per-sample masks are **not** all identical (equal counts, different indices) | unit, 2 draws |
| G18 | a **dropped** stream (not in `MaskPlan.active`) never reaches the adapter, `enc_parts`, or the decoder (`adapters[s]`/`heads[s]` grads are `None`), yet its loss keys still EXIST as zero (`losses[s] = losses_mse[s] = 0.0`, `masked_<s> = 0`) so G13/DDP hold — distinct from a **0-kept** stream, which IS decoded and scored (G16) | unit + 1-step forward |
| G19 | override precedence: `min_kept: 1` + `kept_ratio_override: {tir: 0.0}` ⇒ `tir` stays 0-kept; an override above the cap (`tir: 0.5`) raises `ValueError` at construction | unit |
| G20 | degenerate draw: with one stream pinned, the remaining free stream's ratio VARIES across steps (Uniform/Beta(1,1)), never a constant `1.0` pinned at its cap | unit, 200 draws |

Verification cadence per AGENTS.md §7: `python -m compileall code` →
`python -m core.multimae` → `python -m pytest` from `code/` → local smoke
script with `--max_entries 2 --num_workers 0`.

---

## 6. Decisions

### 6.1 Settled — parameter values are experiment config

* **D1 — geometry values.** The design's `Gt=25` (tubelet_t 4, `sig_kernel 32`)
  is adopted *as the AMMM config's values*, not as a code default. It is a new
  lineage (new checkpoint dir, new Stage-3 sibling) and that is accepted: no
  compatibility with the shipped `Gt=50` runs is claimed or wanted. Note the one
  value that is *not* free: the design's "≤ 675 tokens" bound only holds at
  `Gt=25` (at `Gt=50` the same 16 spatial patches are 800 rgb tokens), so the
  token budget is a property of this config, not of the engine.
* **D2 — per-stream visual size.** 128 (RGB) / 96 (TIR) is adopted as the AMMM
  config's values. The prerequisites are code-shaped only: Phase 3 (the knob)
  and Phase 4 (a dataset that can emit two different visual sizes). Cached
  artefacts are unaffected — `roi_cache.cache_key` excludes `input_size`.
  Arms A/B (existing datasets) test the policy first; arm D is the
  design-faithful one and needs no new value beyond those two numbers.

### 6.2 Resolved (review round 1 + owner decisions)

* **D3 — dropout granularity.** *Resolved.* The design now defines a
  **batch-level** availability vector $\vec{a}_{\text{batch}}$, drawn once per
  step (design §3.1.1), so specification and implementation agree: a
  per-sample vector would break the dense-gather contract of §5.5.
* **D4 — masking extremes.** *Resolved (owner decision 2026-11-08, supersedes
  the round-1 reviewer floor).* **Any modality may be 100 % masked** — a 0 kept
  ratio is a legal draw, so the clamp lower bound is `0`, not `1` (design
  §3.2.3). Two reporting obligations remain: (a) a fully-**kept** stream has no
  masked position, so its masked-MSE is exactly 0 and is reported as an
  **explicit detached `0.0`** under its usual key plus `masked_<s>` = 0 — never
  by omitting the key, because `MetricLogger.synchronize_between_processes` runs
  one collective per meter and a rank-dependent key set can deadlock a DDP run
  (design §5.9, gate G13); (b) a fully-**masked** stream reports `kept_<s>` = 0 /
  `masked_<s>` = `n_s` so the state is visible in the log. The only global floor
  is `Σ kept ≥ 1` (design §5.7) so the encoder never gets an empty sequence.

### 6.3 Open — code semantics (remaining)

* **D5 — Stage-1 init at tubelet_t 4.** `fit_visual_patch_embed` raises on a
  `(2,16,16)` VideoMAE kernel, which is a *code* gap, not a value conflict: the
  encoder blocks load fine at any tubelet, only the temporal conv does not.
  Recommendation: add an explicit, logged temporal-kernel adaptation
  (repeat-mean over the 2-frame kernel /2) behind
  `--pretrained_tubelet_adapt repeat`, and accept random adapter init as the
  documented alternative — the spec already says positions/adapter are
  initialised when a source cannot provide them. Do **not** silently slice the
  kernel.
* **D6 — Dirichlet budget leakage.** `r = min(λ, cap)` with a video-dominant
  draw (`λ_rgb = 0.7`) forfeits 0.45 of the budget, so the realised visible
  fraction is well below 1 and varies per step. Recommendation: accept it (it is
  the design's explicit formulation), log `kept_total` / realised `r` every step,
  and offer `dirichlet_renormalise: true` as an ablation knob later.

---

## 7. Not in scope

* The loss side (`signal_weight`, `loss_weights`, MR-STFT term): unchanged; the
  Stage-2 decision (spectral term retired) still holds.
* `_span_mask` semantics and `SpanMask_PhysioSignals.md`: independent of AMMM;
  reuse only as an optional physio pattern.
* Stage-3 (`core/waveform_model.py`): consumes geometry, never masks.
* Decoder redesign: the shared decoder + per-stream head stays.

## 8. BDD-Style Test Cases for AMMM Engine

**Scenario: Visuo-physiological co-existence fallback**
* **Given** a declared stream set of `['rgb', 'tir', 'bp', 'resp']`
* **When** the dropout step drops both `bp` and `resp` (leaving only visual streams)
* **Then** the engine must re-activate `bp` (the first declared signal stream)
* **And** `MaskPlan.active` must equal `('rgb', 'tir', 'bp')`

**Scenario: 1D Dirichlet degeneracy fallback**
* **Given** a declared stream set of `['tir', 'resp']`
* **And** the configuration pins `kept_ratio_override: {'tir': 0.25}`
* **When** `plan_step` is called over 1000 iterations
* **Then** `resp` must be drawn from `Uniform(0, 1)`
* **And** the realized $r_{resp}$ must vary across steps, rather than being statically pinned to `1.0`

**Scenario: Strict min_kept does not override explicit 0.0 ablation**
* **Given** the configuration sets `min_kept: 1`
* **And** the configuration pins `kept_ratio_override: {'tir': 0.0}`
* **When** `plan_step` is executed
* **Then** the `tir` stream must have exactly `0` kept tokens
* **And** any drawn streams must have $\ge 1$ kept tokens

**Scenario: Global visibility guard on all-zero draw**
* **Given** a declared stream set of `['tir', 'resp']`
* **And** the configuration pins `kept_ratio_override: {'tir': 0.0, 'resp': 0.0}`
* **When** `plan_step` is executed
* **Then** the engine must grant exactly `1` kept token to `tir` (first in declared order)
* **And** the total batch sequence length $N_{batch}$ must be exactly `1`

**Scenario: Inactive streams emit detached zero losses for DDP stability**
* **Given** a declared stream set of `['tir', 'resp']`
* **When** `tir` is dropped from `MaskPlan.active` for a specific step
* **Then** the forward pass must not execute the adapter or decoder for `tir`
* **But** the returned loss dictionary must contain `losses_mse['tir'] == 0.0` (detached)
* **And** the returned loss dictionary must contain `masked_tir == 0`

