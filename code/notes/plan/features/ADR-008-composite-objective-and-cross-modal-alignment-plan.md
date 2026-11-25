# ADR-008 — composite respiration objective + cross-modal InfoNCE alignment: implementation plan

> **Authority.** `code/notes/design/ADR-008-composite-objective-and-cross-modal-alignment.md` is the
> contract; this document only says **how** it is built and **in what order**. Where the two disagree,
> the ADR wins and this plan is wrong.
>
> **Scope.** Adds the three-term composite physiological objective (`L_MSE` + `L_Pearson` +
> `L_spectral`), the Mode-2-gated symmetric cross-modal InfoNCE regularizer, the physiological weight
> rebalance (`lambda_resp = 5.0`), the opt-in gradient-balance diagnostic, the §2.8 logical parameter
> surface, the §2.7 observability fields, one arm configuration, and the §3.1 conformance suite.
> **Out of scope:** the masking policy, geometry, curriculum and tokenizer (governed verbatim by ADR-007,
> unchanged here), data curation and cache formats, the Stage-3 recipe, the evaluation protocol, and the
> run protocol itself — launch/fold/checkpoint procedure belongs in `code/notes/plan/experiments/`,
> measured outcomes in `code/notes/eval/`, never here.
>
> **State (2026-11-25).** **IMPLEMENTED — Phases 0-7 landed, 940 tests pass** (886 pre-existing
> + 54 new across `tests/test_adr008_composite_objective.py` and `tests/test_adr008_alignment.py`).
> §8 records the measured numbers, the two review rounds and the one deviation found during
> implementation. ADR-008 itself remains `Proposed`: its **Criterion 2/3** gates need the matched
> pure-MSE ADR-007 Fold-0 run and the five-fold Stage-3 sweep, which are run-protocol deliverables
> (`code/notes/plan/experiments/`), not phases of this plan.
>
> **Decision sign-off (2026-11-25).** **D1, D3, D4 and D6 are APPROVED** by the reviewer and are
> normative for the phases below: the `L_MSE` term reuses the existing `MaskedMSELoss` call on the full
> tensors (bit-identity + `mse_resp` comparability), the two extra `beta > 0 or gamma > 0` preconditions
> stand (`target_norm in ('none','clip')`, exactly one signal stream), the manifest `objective` block is
> written/compared only when non-neutral, and the gradient-balance probe lives inside `forward` behind a
> `step=` keyword. The remaining decisions (D2, D5, D7, D8, D9, D10) are implementation-level choices
> with neutral defaults and stand as written; they need no further sign-off unless a phase contradicts
> one of them, in which case raise it before editing code.
>
> **Review response (2026-11-25, first review folded in).** Three blockers: **B1** the gradient-balance
> probe now differentiates w.r.t. `z` and slices the **gradient** (a slice view is not in the graph the
> loss consumed, so `autograd.grad(l_tir, z[:, :K_tir])` raises); **B2** the weighted term uses
> `self.loss_weights[s]` (`MultiModalMAE` never stores `self.signal_weight`); **B3** the InfoNCE gather is
> gated on `self.training` because `evaluate_pretrain` runs `drop_last=False`. Medium: **M1** no dtype
> round-trip in `symmetric_infonce` (FP32 end to end); **M2** `composite_active` includes `alpha != 1.0`;
> **M3** the single-signal guard covers `beta > 0` as well as `gamma > 0`, and is mirrored into ADR-008
> §2.6 rule 6; **M4** `grad_balance_freq` is recorded under `run`, not inside the resume-guarded
> `objective` block. Minor: the flat-index formula in §2.2, the `self.apply(self._init_weights)` ordering
> in §3.3, and the `D9` / `G12` command wording.
>
> **Second review (2026-11-25).** Four cleanups folded in: the `[objective] OFF-ADR-008` banner is now
> `alpha == 0.0 or beta == 0.0 or gamma == 0.0 or lambda_align == 0.0` (the previous predicate was blind
> to a `beta`-only / `gamma`-off run); the gradient-balance probe consumes `losses[visual]` /
> `losses[signal]`, so `--grad_balance_freq` also works on the **matched pure-MSE ADR-007 comparator**
> where `self.resp_loss is None` (exactly what ADR-008 §3.2's gradient-balance row compares against),
> with `g_* is None` guards before the gradient slice; **D3**'s guard is worded `beta > 0 or gamma > 0`
> in every place (G7, the D3 row, the sign-off block); and the two stray trailing `>` characters in this
> header blockquote are fixed. Plan is **green to execute Phase 0 → Phase 7**.
>
> **Judged by** ADR-008 §3.1 (Criterion 1 — mathematical, masking, precision and distributed
> conformance). Criterion 2 (§3.2 pretext anti-collapse) is **non-gating**; Criterion 3 (§3.3 five-fold
> Stage-3 band-limited Pearson, `>= +0.02` over the matched pure-MSE ADR-007 encoder) is the **arm's**
> gate and lands in `code/notes/eval/`.

---

## 1. Why this is additive (and where it is not)

Unlike ADR-007, ADR-008 does **not** change the shape of the stream set, the tokenizer, or the masking
draw. Every new behaviour is reached through a knob whose default reproduces today's run. That gives the
plan one hard structural obligation — **the neutral path must be the existing code path, not a
re-implementation of it**:

| neutral-default obligation (ADR-008 §2.4(7), R5) | how |
| :--- | :--- |
| `(alpha, beta, gamma, lambda_align) = (1, 0, 0, 0)` ⇒ zero added parameters | `self.resp_loss` / `self.align_proj` are **not constructed**; the forward keeps calling `self.loss_fn(pred, tgt, mask_s)` |
| bit-identical loss and gradients | the neutral branch must execute the *same statements* as today — no re-routed tensor, no different reduction order (see **D1**) |
| identical logged key set | `out['objective']` appears only when `objective_active`; `out['diagnostics']` only when `grad_balance_freq > 0`; both are functions of the **config**, never of the sampled mode (§2.4(6), ADR-005 contract 9) |
| identical `masking.json` | the `objective` block is written only when at least one ADR-008 parameter is non-neutral (§2.7) |

**What genuinely is missing:** (i) the two FP32 loss terms + their degenerate guards, (ii) the InfoNCE
heads/pooling/gather, (iii) the forward wiring + gradient diagnostic, (iv) the 12-key parameter surface
through model → runner → manifest → engine → validation, (v) the arm config and its registrations, (vi)
the conformance suite.

---

## 2. Definition of done

1. `streams: tir,resp` with the ADR-008 tuple builds and trains: encoder input strictly `[B, 200, 768]`
   on **every** step of both phases; the objective adds exactly **1,378,048** parameters
   (`2 x (768*768+768 + 768*128+128)`), no others.
2. `L_resp` is the exact §2.2 composite: `mse` from the *existing* `MaskedMSELoss` on the full
   `[B, N_s, k]` tensors, `pearson` from the masked-only gather, `spectral` from the detached-spliced
   full clip. `resp_mse` is bit-identical to the pure-MSE run's `mse_resp` for the same plan (**D1**).
3. On Mode 1 (`K_resp = 0`) a flat zero prediction yields `L_Pearson = 1.0` and `L_spectral = 1.0`
   within the §3.1 tolerances, with a **strictly non-zero** gradient w.r.t. the prediction; the exact
   target yields `< 1e-6` on all three terms.
4. On Mode 2 (`K_resp > 0`), perturbing predictions at `mask == 0` changes **no** term and **no**
   gradient (§2.4(2)/(3)).
5. `L_align > 0` on Mode-2 steps with `B_eff >= 2`, producing non-zero gradients on **both** encoder
   paths; on Mode-1 steps `loss_align == 0.0`, `align_active == 0`, and the heads receive exact **zero**
   gradient while staying in the autograd graph (zero-multiplier dummy term).
6. Every §2.6 precondition raises at **construction**, naming the condition and the numbers.
7. Neutral defaults reproduce the Phase-0 fixture bit-for-bit (loss values, per-stream `losses_mse`,
   parameter count, state-dict key set) and leave every pre-existing suite green **unmodified**.
8. §2.7 observability is emitted per step / per validation pass / per run, with key sets fixed by
   configuration.
9. An ADR-008 Stage-2 checkpoint transfers into Stage 3 with the alignment heads silently excluded
   (no `unexpected` keys), because the transfer is prefix-whitelisted (#3 below).
10. `code/notes/spec/` is regenerated and `code/notes/GLOSSARY.md` gains the Phase-7 names.

---

## 3. What already satisfies ADR-008 (nothing to build)

Verified against the code before planning; this is what keeps the delta small.

| ADR-008 requirement | where it already holds |
| :--- | :--- |
| masked MSE term shape/reduction (§2.2.1) | `core/criterion.py::MaskedMSELoss` — `((p-t)**2).mean(-1)` then `(l*m).sum()/(m.sum()+eps)`, `eps = 1e-6` = `eps_mask` |
| `sigma_resp` weighting, neutral `0.5` (§2.8) | `MultiModalMAE.loss_weights` + `--signal_weight`; `_tiny_model` / every arm already carry it |
| `K_tir`, `K_resp` per step, batch-uniform (§2.3.1 inputs) | `plan.kept_tokens` / `plan.kept_units` from `ConstantBudgetApportioner` (ADR-007) |
| encoder latents `Z: [B, N, D]` post-norm (§2.3.1) | `z = self.enc_norm(...)` in `MultiModalMAE.forward`, concatenated in `self.streams` order |
| visible-token indices for the anatomy weighting (§2.3.1) | `ids_keep[s]` returned by `make_masks`, and `model.core_indices` (ADR-007 `A_core`) |
| `A_core` empty under `spatial_sampling = uniform` (§2.3.1) | `model.anatomy_aware` / `model.core_indices == ()` |
| per-token target/sample width `k = 32`, `L = 8000` (§2.2) | `self._flat['resp'] = sig_kernel`, `self.seq_len`, `target_norm='none'` pinned by `pin_roi_target_norm` |
| per-clip z-scored target, `E[y]=0`, `Var[y]=1` (§1.2.1) | `data/tir_resp_dataset.TARGET_NORM` (the single label normalisation) |
| DDP skeleton, rank-independent plan, FP32 validation | `engines/pretrain.py::{train_one_epoch,evaluate_pretrain}`, `core/mask_placement.py` |
| downstream transfer isolation (R6, §2.4(9)) | `core/waveform_model.py::load_stage2_encoder` copies only `enc_blocks.*`, `enc_norm.*`, `adapters.*`, `positions.*` (+ optional `heads.<target>`) — a new `align_proj.*` prefix is **excluded by construction** |
| rebuild-from-`args` on the recon path | `runners/run_eval_recon.py::build_model` calls `build_pretraining_model(ckpt['args'])`, so new args re-create the heads; an old checkpoint's namespace lacks them and `getattr` defaults keep it head-free |
| gradient-balance anchor is opt-in, zero cost when off (§2.7) | new code, but `grad_balance_freq = 0` must add **no** autograd traversal |

---

## 4. Phases

### Phase 0 — pre-flight (no code, ~20 min)

1. Confirm the geometry resolves to the ADR's numbers **before** touching anything: `fs 1000`,
   `seq_len 8000`, `sig_kernel 32` ⇒ `N_s = 250`, `L = 8000`, `k = 32`; `K_resp in {0,25,50,75,100}`
   ⇒ `n_masked in {250,225,200,175,150}` ⇒ `M_pts in {8000,7200,6400,5600,4800}` (§2.2.2's set).
2. Confirm the spectral grid: `N_fft = 4 * 2**ceil(log2 8000) = 32768`, `df = 1000/32768 = 0.0305176 Hz`
   (≈ 1.83 bpm), bins with `0.10 <= k*df <= 0.60` ⇒ `k in [4, 19]` ⇒ **16 bins** (§2.2.3, §2.6(3)).
3. Confirm the parameter arithmetic: two `768 -> 768 -> 128` heads = **1,378,048** parameters
   (≈ 1.3 % of ViT-B), and that this is the only size change.
4. **Capture the neutrality fixture BEFORE Phase 3**: `code/scripts/project/capture_adr008_neutrality.py`
   builds the current `_tiny_model` **and** the real ADR-007 geometry, runs one forward+backward over a
   fixed `(mask_base, epoch, step)` (Mode 1 and Mode 2 pinned via `plan_from_counts`), and records to
   `code/tests/fixtures/adr008_neutrality_fixture.json`: `total`, every `losses[s]`, every
   `losses_mse[s]`, the parameter count, the sorted `state_dict` key list, and the SHA-256 of
   `core/criterion.py` + `core/multimae.py`. `--check` re-runs and diffs it (the ADR-007
   `capture_adr007_neutrality.py` pattern).
5. Record which suites must stay green unmodified: `test_budget_masking.py`, `test_mask_placement.py`,
   `test_two_stream_anatomy_masking.py`, `test_stage2_protocol.py`, `test_curriculum_schedule.py`,
   `test_regime_single_carrier.py`, `test_val_gather.py`, `test_resume_guard.py`,
   `test_multimae_rgb2d.py`.

### Phase 1 — `code/core/resp_composite.py` (new): the three-term objective

One new module. `core/waveform_losses.py` stays the **Stage-3** loss and is not touched (its
`PearsonLoss` is unrelated: unmasked, no gather, no variance floor, `[B, T]` only, `eps` outside the
norms) — see **D2**.

**1.1 Public surface**

```python
__all__ = ['MaskedPearsonLoss', 'BandLimitedSpectralLoss', 'CompositeRespLoss']

class MaskedPearsonLoss(nn.Module):
    def __init__(self, eps: float = 1e-8, var_floor: float = 1e-6): ...
    def forward(self, pred_masked, target_masked) -> torch.Tensor   # [B, M_pts] FP32 -> scalar

class BandLimitedSpectralLoss(nn.Module):
    def __init__(self, fs: float, n_fft: int = 0, f_min: float = 0.10,
                 f_max: float = 0.60, eps: float = 1e-8,
                 energy_floor: float = 1e-8): ...
    @property
    def n_fft_resolved(self) -> int
    @property
    def band_bins(self) -> torch.Tensor        # the K_resp index set
    def forward(self, pred_full, target_full) -> torch.Tensor   # [B, L] FP32 -> scalar

class CompositeRespLoss(nn.Module):
    def __init__(self, *, fs: float, sig_kernel: int, n_signal: int,
                 mse_weight: float = 1.0, pearson_weight: float = 0.0,
                 spectral_weight: float = 0.0, spectral_n_fft: int = 0,
                 spectral_f_min: float = 0.10, spectral_f_max: float = 0.60): ...
    def forward(self, mse_term, pred_masked, target_masked,
                pred_full, target_full, n_masked: int) -> dict
        # {'total', 'mse', 'pearson', 'spectral'}
```

**1.2 `MaskedPearsonLoss` — §2.2.2 exactly**

- Inputs arrive **already gathered and flattened** to `[B, M_pts]` in FP32
  (`reshape(gather_tokens(pred, ids_masked), [B, n_masked*k])`), so the module never sees a zero hole.
- `p = p - p.mean(-1, keepdim=True)`, `t = t - t.mean(-1, keepdim=True)`.
- Denominator `sqrt((p**2).sum(-1) + eps) * sqrt((t**2).sum(-1) + eps)` — **eps inside each sqrt**.
- `valid = (t**2).sum(-1) > var_floor`; `loss = (1 - r)[valid].mean()`; if `valid.sum() == 0` or
  `M_pts < 2` return an explicit `0.0` **on the right device/dtype** (a constant tensor, not `None`).
- Range assertion (debug only, `_DEBUG` env-guarded): `0 <= loss <= 2`.

**1.3 `BandLimitedSpectralLoss` — §2.2.3 exactly**

- `n_fft_resolved`: `0` ⇒ `4 * 2**ceil(log2(L))`; otherwise the given value (validated `>= L`).
  `L` is derived as `n_signal * sig_kernel` — never a magic `8000`.
- `band_bins`: `k * fs / n_fft` inside `[f_min, f_max]`, resolved **once** at construction into a
  registered `long` buffer; `len(band_bins) >= 1` asserted with the numbers in the message.
- `forward`: per-clip DC removal (`x - x.mean(-1, keepdim=True)`) → Hann window
  (`torch.hann_window(L, dtype=torch.float32, device=...)`, buffered once) → `torch.fft.rfft` →
  `P = re**2 + im**2` → slice `band_bins` → unit-sum normalise (`eps` in the denominator) →
  `|Pt - Pp|` summed over bins, averaged over the **valid** samples
  (`sum_j P_y > energy_floor`), `0.0` when none qualify.
- Everything is cast `.float()` on entry, so autocast cannot turn the FFT complex-half-precision
  (BF16 rFFT raises on some backends) — §2.4(4).

**1.4 `CompositeRespLoss`**

- `mse_term` is **passed in**, not recomputed: the caller hands the already-computed
  `MaskedMSELoss(pred, tgt, mask)` scalar, which is what keeps `mse_resp` comparable across arms and
  bit-identical at neutral (**D1**).
- `n_masked == 0` ⇒ `{'total': 0, 'mse': 0, 'pearson': 0, 'spectral': 0}` (all explicit zeros).
- Disabled terms (`weight == 0.0`) are **not evaluated** and reported as explicit `0.0`, so
  `beta = 0` costs nothing and `gamma = 0` never builds an FFT.
- Returns **unweighted** `pearson` / `spectral` (the §2.7 "raw unweighted" requirement) and a `total`
  that is `alpha*mse + beta*pearson + gamma*spectral`.

**1.5 Exports** — mirror the existing pattern in `core/__init__.py` (which already re-exports
`MaskedMSELoss, MaskedL1Loss, MaskedCrossEntropyLoss` and `PearsonLoss, MultiResolutionSTFTLoss,
WaveformJointLoss`): add `from .resp_composite import (MaskedPearsonLoss, BandLimitedSpectralLoss,
CompositeRespLoss)` next to the `criterion` import, and `from .align_infonce import (AlignProjection,
anatomy_weighted_pool, symmetric_infonce)` in Phase 2. No `@register_model` — neither module is a
model entrypoint.

### Phase 2 — `code/core/align_infonce.py` (new): the cross-modal regularizer

**2.1 Public surface**

```python
__all__ = ['AlignProjection', 'anatomy_weighted_pool', 'symmetric_infonce']

class AlignProjection(nn.Module):            # 2-layer MLP, D -> D -> d_proj
    def __init__(self, embed_dim: int, proj_dim: int = 128): ...

def anatomy_weighted_pool(z_tir, ids_keep, core_indices, n_spatial,
                          core_weight: float = 1.0) -> torch.Tensor     # [B, D]

def symmetric_infonce(v_tir, v_resp, temperature: float = 0.07,
                      gather: bool = True) -> torch.Tensor              # scalar
```

**2.2 `anatomy_weighted_pool`**

- `s(b, i) = ids_keep[b, i] % n_spatial` — both `place_video` and `place_video_stratified` emit the
  encoder index `t * n_units + s` (`t` = intra-tube position, `s` = spatial tube; see `place_video`'s
  docstring and its `offsets = arange(tpu) * n_units`), so the modulo recovers `s` exactly.
- `w = where(s in core_indices, core_weight, 1.0)`; `u = (w.unsqueeze(-1) * z_tir).sum(1) / w.sum(1)`.
- `core_indices == ()` (the `uniform` ablation, or a 2-D grid never declared) ⇒ `w == 1` everywhere,
  i.e. plain mean pooling; no special case in the arithmetic, so `core_weight = 1.0` and an empty
  `A_core` are the *same* code path (§2.3.1's neutrality claim).
- `core_weight > 0` validated upstream (§2.6(4)).

**2.3 `AlignProjection`**

`Linear(D, D) -> GELU -> Linear(D, d_proj)`; init by the model's `_init_weights` (xavier, zero bias) so
it matches house style. **No** normalisation inside (the L2 normalisation is a separate FP32 step, so
the module stays dtype-transparent).

**2.4 `symmetric_infonce`**

```python
def _gather_keep_local_grad(v, world, rank):
    if world == 1:
        return v
    outs = [torch.zeros_like(v) for _ in range(world)]
    torch.distributed.all_gather(outs, v.detach())   # no grad through remote slices
    outs[rank] = v                                   # local slice keeps its graph
    return torch.cat(outs, 0)
```

- **Everything in this function is FP32** (§2.4(4)): `v_tir, v_resp` are cast `.float()` **before** the
  `max(norm, eps)` normalisation, the gathered `V_tir, V_resp` stay FP32, and the logits `(v @ V.T) / tau`
  are computed in FP32. There is **no** cast back to the autocast compute dtype anywhere in the matmul or
  the cross-entropy path.
- `labels = arange(B) + rank * B`; `0.5 * (CE(logits_t2r, labels) + CE(logits_r2t, labels))`.
- `rows(V) < 2` ⇒ explicit `0.0` (the single-device `B = 1` case, §2.3.3).
- `gather=False` (or no `torch.distributed` init) skips the collective entirely — the `align_gather_ddp`
  knob, default `true`. **The caller passes `gather = (self.align_gather_ddp and self.training)`**, not the
  raw knob: `dist.all_gather` requires identical shapes on every rank, the training loader runs
  `drop_last=True` (`runners/_common.py::make_data_loader`) while `evaluate_pretrain` runs
  `drop_last=False`, so an unpadded final **validation** batch has a rank-dependent `B` and would deadlock.
  This is ADR-008 §2.3.3's `is_training` scope, not a new rule.

### Phase 3 — `code/core/multimae.py`: parameter surface, feasibility, forward wiring

**3.1 Constructor surface (exact §2.8 names; one string for YAML key, argparse dest and Python keyword,
as ADR-006/007 established).** Twelve new keywords, every default neutral:

| keyword | default | ADR-008 arm | validated |
| :--- | ---: | ---: | :--- |
| `resp_loss_mse_weight` | `1.0` | `1.0` | `>= 0` |
| `resp_loss_pearson_weight` | `0.0` | `2.0` | `>= 0` |
| `resp_loss_spectral_weight` | `0.0` | `1.0` | `>= 0` |
| `spectral_f_min` | `0.10` | `0.10` | `0 < f_min < f_max` |
| `spectral_f_max` | `0.60` | `0.60` | `f_max <= fs/2` |
| `spectral_n_fft` | `0` | `0` (⇒ 32768) | `0` or `>= L`; `|K_resp| >= 1` |
| `align_weight` | `0.0` | `0.5` | `>= 0` |
| `align_proj_dim` | `128` | `128` | `>= 16` |
| `align_temperature` | `0.07` | `0.07` | `> 0` |
| `align_core_weight` | `1.0` | `1.0` | `> 0` |
| `align_gather_ddp` | `True` | `True` | bool |
| `grad_balance_freq` | `0` | `50` | `>= 0` |

`signal_weight` (§2.8 `lambda_resp`) is **reused**, not re-added.

**3.2 `_validate_objective()` — a new private method called from `__init__` before the adapters are
built**, so a bad tuple fails with no parameters allocated. Checks, in this normative order, each
message naming the condition and the numbers (§2.6):

1. `alpha + beta + gamma > 0` and each `>= 0`.
2. `gamma > 0` ⇒ `0 < f_min < f_max <= fs/2`.
3. `gamma > 0` ⇒ `n_fft in {0} or n_fft >= L`, and `|K_resp| >= 1`.
4. `lambda_align > 0` ⇒ `tau > 0`, `d_proj >= 16`, `core_weight > 0`, **and exactly one visual + one
   physiological stream** (`len(self.visual) == 1 and len(self.signal) == 1`) — the §2.4(8) cardinality
   guard, which must fire for the 3-/4-stream ADR-005/006 arms rather than silently aligning a pair.
5. `grad_balance_freq >= 0`.
6. **(plan-level, D3 — mirrored into ADR-008 §2.6 rule 6)** `beta > 0 or gamma > 0` ⇒
   `target_norm in ('none', 'clip')` **and** `len(self.signal) == 1`. A per-token normalised target puts
   a step at every token boundary, so the spliced full clip would be a discontinuous waveform and the
   passband would describe an artefact. The single-signal half covers **both** `beta` and `gamma`:
   `out['objective']` is a SINGULAR dict keyed `resp_*`, so a second physiological stream would silently
   overwrite another stream's Pearson/spectral term in the logs (and the band is a *respiration* band —
   `bp` runs near 1 Hz). Neither check can reject the ADR-008 tuple (arm: `target_norm 'none'`, one
   signal stream).

**3.3 Constructed modules (only when active — R5)**

One predicate decides everything, and it must include `alpha != 1.0`: with `alpha = 0.5, beta = gamma = 0`
the objective is **not** neutral (the MSE term is rescaled) even though both other terms are off.

```python
composite_active = (alpha != 1.0 or beta > 0.0 or gamma > 0.0)
self.resp_loss = (CompositeRespLoss(..., mse_weight=alpha, ...)
                  if composite_active else None)
self.align_proj = None
if lambda_align > 0:
    self.align_proj = nn.ModuleDict({s: AlignProjection(embed_dim, d_proj)
                                     for s in (self.visual[0], self.signal[0])})
self.objective_active = bool(composite_active or lambda_align > 0)
```

**Ordering.** `self.align_proj` must be built **before** the `self.apply(self._init_weights)` call that
closes `__init__` (it is what gives the two new `Linear`s the house xavier / zero-bias init): place the
block immediately after the `self.heads` loop, before `self.apply(...)`.

Key prefix `align_proj.` ⇒ automatically outside the Stage-3 whitelist (R6) and recreated by
`run_eval_recon` from the checkpoint's own `args`. The **same** `composite_active` predicate is reused by
`_objective_non_neutral(model)` in Phase 4.2, so the manifest and the model cannot disagree.

**3.4 `forward` — the exact insertion points**

1. Inside the per-stream decode loop, capture for the **signal** stream the two tensors the composite
   needs: `pred_masked` (`[B, n_masked, k]`, already computed) and
   `target_masked = tgt.gather(1, ids_masked.unsqueeze(-1).expand(...))`. Nothing else in the loop moves.
2. `mse_term = self.loss_fn(pred, tgt, mask_s)` — **unchanged statement**, so `mse_resp` is bit-identical
   to the pure-MSE run for the same plan (DoD 2).
3. If `self.resp_loss is not None`:
   `p_full = reshape(mask_s.unsqueeze(-1) * pred + (1 - mask_s.unsqueeze(-1)) * tgt.detach(), [B, L])`,
   `t_full = reshape(tgt, [B, L])` (target detached once, before the splice), then
   `objective = self.resp_loss(mse_term, pm_flat, tm_flat, p_full, t_full, n_masked)` and
   `losses[s] = self.loss_weights[s] * objective['total']` — **not** `self.signal_weight`, which does not
   exist: the constructor folds `signal_weight` into `self.loss_weights` (`1.0` for visuals, the knob for
   physiological streams) and never stores it under its own name.
   Otherwise `losses[s] = self.loss_weights[s] * mse_term` (today's line).
4. **Alignment**, after `z = self.enc_norm(...)` and before the decoder — but the *loss* must be added
   after the decoder loop so the returned scalar is one graph:
   - slice `z` by the per-stream token counts in `self.streams` order
     (`offset`, `K_tir = plan.kept_tokens[visual]`, `K_resp = plan.kept_tokens[signal]`);
   - `u_tir = anatomy_weighted_pool(z_tir, ids_keep[visual], self.core_indices, self.spatial_patches,
     core_weight)`, `u_resp = z_resp.mean(1)`;
   - `v = l2_normalize_fp32(proj(u))` for both, then `symmetric_infonce(v_tir, v_resp, tau, gather)`;
   - gate: `active = int(K_resp > 0 and K_tir > 0)`; on `active == 0` set `l_align = 0.0` and, in
     training only, add `zero_touch = sum(p.sum() * 0.0 for p in self.align_proj.parameters())` to the
     total (contract 6; `find_unused_parameters` defaults to `True` but the ADR forbids relying on it).
5. **Gradient-balance diagnostic** (`self.training and torch.is_grad_enabled() and
   self.grad_balance_freq > 0 and step is not None and step % self.grad_balance_freq == 0`), computed
   **after** both losses exist and **before** returning:

   ```python
   def _dz_grad(term):                       # term: scalar; z: [B, N, D] encoder output
       if term is None or not term.requires_grad:
           return None
       return torch.autograd.grad(term, z, retain_graph=True, create_graph=False,
                                  allow_unused=True)[0]     # grad w.r.t. z, or None

   g_tir  = _dz_grad(losses[visual])        # == 1.0 * L_tir in every arm
   g_resp = _dz_grad(losses[signal])        # == lambda_resp * L_resp (composite OR pure MSE)
   # slice the GRADIENT rows, using the tracked visual offset
   n_tir  = (None if g_tir is None else
             g_tir[:, offset_tir:offset_tir + K_tir].detach().float().norm(2))
   n_resp = (None if g_resp is None else
             g_resp[:, offset_tir:offset_tir + K_tir].detach().float().norm(2))
   ```

   - **Use `losses[visual]` / `losses[signal]`, never `weight * r_out['total']`.** `losses[s]` is the
     per-stream weighted loss in **both** branches — composite (`loss_weights * composite_total`) and
     pure MSE (`loss_weights * mse_term`) — whereas `r_out` does not exist when `self.resp_loss is None`.
     Wanted consequence: `--grad_balance_freq` is meaningful on the **matched pure-MSE ADR-007
     comparator** too, which is precisely what ADR-008 §3.2's gradient-balance row is compared against.
   - **Differentiate w.r.t. `z`, never w.r.t. a slice of `z`.** `z[:, :K_tir]` is a *new* slice node that
     no loss consumed, so `autograd.grad(l_tir, z[:, :K_tir])` raises
     `RuntimeError: One of the differentiated Tensors appears to not have been used in the graph`.
     Take `dL/dz` for the whole `[B, N, D]` tensor, then slice the **gradient** rows with the tracked
     `offset_tir` accumulated in `self.streams` order exactly as the concatenation built it — never a
     hardcoded `:K_tir`.
   - `allow_unused=True` plus the `requires_grad`, `g_* is None` and `n_* is None` guards keep the probe
     quiet instead of fatal when a term happens to be detached (e.g. a 0-kept stream); a missing norm
     drops the ratio rather than emitting `nan`.
   - `forward` gains an optional `step: Optional[int] = None` keyword; `engines/pretrain.py` already
     computes `global_step`, so it passes it (extend `_accepts_epoch` into a generic
     `_accepts_kw(fn, name)` helper).
   - `autograd.grad` is non-destructive (no `.backward()`, no parameter `.grad` touch) and
     `create_graph=False`; the norms go into a **sibling**
     `out['diagnostics'] = {'grad_norm_tir', 'grad_norm_resp', 'grad_ratio_resp_to_tir'}`, where the ratio
     is `n_resp / n_tir` and is **omitted** when either norm is `None` or zero (never `inf`/`nan` in a
     log).
6. **Returned keys** (all config-determined, never mode-determined):
   - `losses_mse` unchanged (`mse_tir`, `mse_resp`);
   - `out['objective'] = {'resp_mse', 'resp_pearson', 'resp_spectral', 'loss_align', 'align_active'}`
     **iff** `objective_active` (values `0.0` for disabled terms);
   - `out['diagnostics']` **iff** `grad_balance_freq > 0`, present only on probe steps (identical on
     every rank because `step` is global).

**3.5 `build_pretraining_model`** — `getattr`-forward the twelve keys with the neutral defaults, plus
`bool(getattr(args, 'align_gather_ddp', True))`; extend the `[mask]`-style provenance print with one
`[objective]` line (`alpha/beta/gamma`, `lambda_resp`, passband, resolved `n_fft` + `|K_resp|`,
`lambda_align`, `tau`, `d_proj`, `core_weight`, gather, `target_norm`) and an
`[objective] OFF-ADR-008` banner when `alpha == 0.0 or beta == 0.0 or gamma == 0.0 or
lambda_align == 0.0` — i.e. whenever the run is not the full three-term objective **plus** the
regularizer (the ADR-007 plan's **D10** pattern: a bare CLI invocation that skips the YAML must not look
like the arm). Note this is **not** `not composite_active`: `beta = 2.0, gamma = 0.0, lambda_align = 0.5`
is `composite_active` yet has no spectral term, and must still be flagged.

### Phase 4 — runner, manifest, engine, validation

**4.1 `code/runners/run_pretrain.py::get_args()`** — register the twelve flags with the neutral
defaults (all defaults are the pre-change behaviour, so `_check_config_keys` stays a pure schema check).
`--align_gather_ddp` uses `type=str2bool, nargs='?', const=True` (the `--find_unused_parameters`
pattern) because its default is `True`.

**4.2 `write_masking_manifest`** — append an `objective` block **only when non-neutral**:

```python
if _objective_non_neutral(model):
    record['objective'] = {
        'adr': 'ADR-008',
        'resp_loss_mse_weight': float(model.resp_loss_mse_weight),
        'resp_loss_pearson_weight': ...,
        'resp_loss_spectral_weight': ...,
        'spectral_f_min': ..., 'spectral_f_max': ...,
        'spectral_n_fft': int(model.resp_loss.spectral.n_fft_resolved) if ... else int(model.spectral_n_fft),
        'spectral_band_bins': [int(i) for i in model.resp_loss.spectral.band_bins] if ... else [],
        'signal_weight': float(model.loss_weights[model.signal[0]]),
        'align_weight': ..., 'align_proj_dim': ..., 'align_temperature': ...,
        'align_core_weight': ..., 'align_gather_ddp': bool(...),
    }
```

The predicate is `model.composite_active or model.align_weight > 0` (Phase 3.3) — never `beta/gamma`
alone, so `alpha != 1.0` is recorded too.

**`grad_balance_freq` is NOT part of `objective`.** It is a read-only logging interval, not an objective
parameter: it goes into `record['run']` next to `epochs`
(`record['run']['grad_balance_freq'] = int(model.grad_balance_freq)`). `_stable()` compares only
`run.mask_base`, so changing the diagnostic cadence can never block resuming a preempted HPC run, while
the value is still recorded for provenance.

Resume guard: extend `_stable()` with `'objective': rec.get('objective')` **but** compare it only when
**either** side is present (i.e. skip the comparison when the current record has no `objective` key
**and** the stored one has none either). Concretely:

```python
def _stable(rec):
    base = {..., 'objective': rec.get('objective')}
    return base
# in the guard:
changed = [k for k in mine if not (k == 'objective' and mine[k] is None and theirs.get(k) is None)
           and mine[k] != theirs.get(k)]
```

This satisfies §2.7 ("resuming a run whose stored objective parameters disagree must fail loudly")
**and** keeps every in-flight ADR-007 run resumable (its record has no `objective` key and the current
record writes none either). Disclosed as **D4**.

**4.3 `code/engines/pretrain.py::train_one_epoch`**

- After the existing `losses_mse` loop, log the objective block on every step when present:
  `metric_logger.update(**{k: float(v) for k, v in (out.get('objective') or {}).items()})` — the keys
  are already the ADR's names (`resp_mse`, `resp_pearson`, `resp_spectral`, `loss_align`,
  `align_active`).
- Log the probe-step diagnostics: `for k, v in (out.get('diagnostics') or {}).items():
  metric_logger.update(**{k: float(v)})`. Probe steps are `step % freq == 0` with the **global** step,
  so every rank adds the same keys on the same steps (`MetricLogger.synchronize_between_processes` and
  `_merge_sums` are both safe).
- Pass `step=global_step` (already computed) through the same `_accepts_kw` guard used for `epoch`.

**4.4 `code/engines/pretrain.py::evaluate_pretrain`**

- Accumulate a **sibling** `objective` dict, sample-weighted, exactly like `stratification`
  (`objective.setdefault(k, 0.0); objective[k] += float(v) * batch_n`), add it to the DDP `payload`,
  merge with `_merge_sums`, divide by `n_samples`.
- **`components` must stay the MSE component set**: `test_stage2_protocol.py` pins
  `set(result['components']) == {'mse_resp'}`; the new values go to `result['objective']`, mirroring the
  ADR-007 `stratification` decision (§2.7's "dedicated objective summary block").
- Validation never probes the gradient balance (it runs under `inference_mode` and passes no `step`), so
  the validation key set stays stable.

**4.5 `run_pretrain.py` validation block** — print
`[val] objective  resp_mse=… resp_pearson=… resp_spectral=… loss_align=… align_active=…` when the block
is non-empty, and `wandb_logger.log({f'val/{k}': v ...})` on the same `wb_step` axis as `components`.

**4.6 `code/runners/run_eval_recon.py`** — no new flag. `build_model` rebuilds from `ckpt['args']`, so
the heads exist and load cleanly; `pin_masking` is untouched (the ADR does not change placement).
Add one assertion in the eval path's own smoke: the loaded state dict reports **0** unexpected keys for
an ADR-008 checkpoint.

### Phase 5 — the arm configuration

New file `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`, derived from
`stage2_budget_tir_resp_2stream_anatomy.yaml` with **only** these deltas:

| key | ADR-007 arm | ADR-008 arm | source |
| :--- | ---: | ---: | :--- |
| `signal_weight` | `0.5` | **`5.0`** | §2.1, §2.8 |
| `resp_loss_mse_weight` | — | **`1.0`** | §2.2.1 |
| `resp_loss_pearson_weight` | — | **`2.0`** | §2.2.2 |
| `resp_loss_spectral_weight` | — | **`1.0`** | §2.2.3 |
| `spectral_f_min` / `spectral_f_max` | — | **`0.10` / `0.60`** | §2.2.3, protocol §4 |
| `spectral_n_fft` | — | **`0`** (⇒ 32768) | §2.2.3 |
| `align_weight` | — | **`0.5`** | §2.3 |
| `align_proj_dim` / `align_temperature` / `align_core_weight` | — | **`128` / `0.07` / `1.0`** | §2.8 defaults |
| `align_gather_ddp` | — | **`true`** | §2.8 |
| `grad_balance_freq` | — | **`50`** | §2.8 |

Everything else — `data_set: tir_roi`, `streams: tir,resp`, `n_budget: 200`, `physio_cap_ratio: 0.40`,
`video_min_tubes: 4`, `video_cap_ratio: 0.25`, `p_virtual_sensing: 0.50`,
`virtual_sensing_warmup_p: 0.10`, `virtual_sensing_warmup_epochs: 20`, the whole `spatial_sampling`
block, `physio_draw_min: 1`, geometry (`tubelet 2,16,16`, `input_size 96`, `fs 1000`, `fps 25`,
`clip_duration 8.0`, `temporal_stride 4`, `sig_kernel 32`, `seq_len 0`, `clip_stride 4.0`), `pos_init`,
`patch_stream tir`, `pretrained_encoder videomae:k400`, `model project_multimae_base`,
`physio_placement scattered`, `target_norm` pin, `epochs 150`, `blr 1.5e-4`, `batch_size 16`, the task
groups, and the `roi_*` cache keys — is **identical**, so the comparison against the pure-MSE ADR-007
arm differs by the objective alone.

The header comment must state: the §2.2 arithmetic at the shipped geometry (`N_s = 250`, `k = 32`,
`L = 8000`, `M_pts ∈ {4800,5600,6400,7200}` / `8000`), the passband and resolved `K_resp` bin set
(16 bins, `df = 0.0305 Hz`), the `+1,378,048` parameters, the Mode-2 gate share (90 % Phase I, 50 %
Phase II), and the "compare on `mse_resp` + §3.2 diagnostics, never on total val loss" consequence of
ADR-008 §4.

**Registrations that the config cannot land without:**

- `code/tests/test_task_group_configs.py::TIR_ROI_CONFIGS` — the arm ships a `task_groups:` block and
  `test_tir_roi_config_inventory_is_complete` asserts `found == sorted(TIR_ROI_CONFIGS)`, so the literal
  list must gain `configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml` in its **sorted**
  position (immediately after `..._anatomy.yaml`).
- `code/tests/test_config_loading.py` needs no edit (it globs `configs/pretrain/*.yaml`, so the new file
  flows through `_check_config_keys` automatically — which is exactly the guard that makes an unregistered
  key a hard error).

### Phase 6 — tests and gates

New suite `code/tests/test_adr008_composite_objective.py`; **no existing test is amended** (DoD 7). A
companion `code/tests/test_adr008_alignment.py` is acceptable if the suite exceeds ~400 lines — split by
module, not by gate.

| gate | assertion (§3.1 clause) |
| :--- | :--- |
| **G1** | Neutral bit-identity (C1.1): with `(alpha, beta, gamma, lambda_align) = (1,0,0,0)` the model has **zero** `align_proj.*` keys, `model.resp_loss is None`, and the forward/backward on the Phase-0 fixture matches **bit-for-bit** (`total`, per-stream `losses`, `losses_mse`, parameter count, key list). Companion assertion for **M2**: `alpha = 0.5, beta = gamma = 0` **does** construct `resp_loss` (and `out['objective']`), and its `total` equals `0.5 * losses_mse[...]` while `resp_mse` stays the raw MSE. |
| **G2** | Anti-collapse (C1.2): a synthetic unit-variance `0.25 Hz` sinusoid (`L = 8000`, `fs = 1000`, `n_masked = 250`); flat `y_hat = 0` ⇒ `L_Pearson = 1.0 +- 1e-5`, `L_spectral = 1.0 +- 1e-4`, `grad_norm > 0`; `y_hat = y` ⇒ all three `< 1e-6`. Also at `0.10` and `0.60 Hz` (band edges) and at `0.05 / 0.90 Hz` (outside the band, where the loss must **not** fall). |
| **G3** | Mode-2 masking integrity (C1.3): for `K_resp in {25,50,75,100}`, perturbing `pred` at `mask == 0` changes neither term nor either gradient (exact `torch.equal`); a constant all-zero target returns finite `0.0` with no `NaN`/`Inf` in `pearson` **and** `spectral`. |
| **G4** | InfoNCE (C1.4): Mode 2, `B >= 2` ⇒ `loss_align > 0` and **both** `model.adapters[visual]` and `model.adapters[signal]` receive non-zero gradients; a synchronized positive pair scores strictly lower than a permuted one; `core_weight = 2.0` ⇒ core-token gradient norm `>` peripheral-token gradient norm; `rows(V) == 1` ⇒ `0.0`. |
| **G5** | Mode-1 gate + distributed safety (C1.4/§2.4(6)): with `p_virtual_sensing = 1.0`, `loss_align == 0.0`, `align_active == 0`, and every `align_proj.*` parameter has `grad is not None` and `grad.abs().max() == 0`; all objective keys still present. |
| **G6** | Mixed precision (C1.5): FP16 **and** BF16 autocast forward/backward produce finite losses/grads, and the spectral term runs (no complex-rFFT dtype failure). |
| **G7** | Startup rejection (C1.7 + §2.6): one case per precondition — `alpha+beta+gamma == 0`; `f_min <= 0`; `f_max > fs/2`; `n_fft > 0 and n_fft < L`; a passband with `|K_resp| == 0`; `tau <= 0`; `d_proj < 16`; `core_weight <= 0`; `grad_balance_freq < 0`; `align_weight > 0` on a **2-visual** and on a **2-signal** stream set; plus **D3**'s `beta > 0 or gamma > 0` with `target_norm='token'` **and** — separately for `beta > 0` (Pearson only) and for `gamma > 0` (spectral) — with two signal streams. Each message names the condition and the numbers. |
| **G8** | Observability (§2.7): the step dict's key set equals the expected set for (a) neutral, (b) `beta,gamma > 0`, (c) `+align`, (d) `+grad_balance_freq > 0`, on **both** modes; `evaluate_pretrain` returns `set(components) == {'mse_tir','mse_resp'}` **plus** `objective` with the five keys and `stratification` untouched; a two-rank simulated gather (the `test_val_gather.py` pattern) sums the objective block correctly. |
| **G9** | Gradient-balance diagnostic: with `grad_balance_freq = 2` and `step` passed, `out['diagnostics']` appears on `step % 2 == 0` only, `grad_norm_resp > 0`, `grad_ratio_resp_to_tir` is present and finite when both norms are `> 0` and **absent** (never `inf`/`nan`) when either is zero, the values equal a reference `autograd.grad(l, z)` sliced at `offset_tir`, parameter `.grad` buffers are **unchanged** by the probe (compare a deep copy before/after), and with `freq = 0` no `diagnostics` key and no extra `autograd.grad` call (monkeypatched counter). |
| **G10** | Manifest (§2.7): the arm writes `objective.adr == 'ADR-008'` with the resolved `n_fft = 32768` and the 16 band bins; a neutral run's `masking.json` is byte-identical to the pre-change record; a resume with a **different** `align_weight` raises, while a neutral re-run of a **stored legacy record** (no `objective` key) does not. |
| **G11** | Downstream isolation (C1.6, R6): `load_stage2_encoder` on an ADR-008 checkpoint loads the same encoder tensors as on the pure-MSE checkpoint, reports `0` shape mismatches, and **no** `align_proj.*` key is copied (assert on the returned `loaded`/`skipped` payload). |
| **G12** | Config + smoke: the new YAML parses through `run_pretrain.get_args()` + `_check_config_keys` and resolves the §2.8 tuple; the capped single-device smoke (§3.4: 1 epoch, `--warmup_epochs 0`, `--virtual_sensing_warmup_epochs 0`, `--max_entries 100`, `--batch_size 8`, `--num_workers 0`, Mode-1 probability `0.50` so both modes execute, **`--grad_balance_freq 2`** so a 12-step epoch produces several probe steps instead of at most one) completes with no `NaN`/assert, records the resolved tuple, prints the peak allocated/reserved memory + batch size + precision, and stays within **+5 %** of the pure-MSE run on the same hardware (§3.4's resource threshold — the `[TODO]` the ADR leaves open). |

```text
cd code
python -m pytest tests/test_adr008_composite_objective.py tests/test_adr008_alignment.py \
                 tests/test_budget_masking.py tests/test_mask_placement.py \
                 tests/test_two_stream_anatomy_masking.py tests/test_stage2_protocol.py \
                 tests/test_val_gather.py tests/test_resume_guard.py \
                 tests/test_config_loading.py tests/test_task_group_configs.py -q
python scripts/project/capture_adr008_neutrality.py --check      # G1
```

### Phase 7 — terminology, spec and glossary refresh

- `code/notes/GLOSSARY.md` gains the canonical names of §2.2/§2.3 — `L_MSE` / `L_Pearson` /
  `L_spectral` / `L_resp` / `L_align` at Stage 2, `alpha`/`beta`/`gamma` **scoped to Stage 2** (the
  Stage-3 entry already owns `alpha, beta, gamma` for time/Pearson/MR-STFT — the glossary must say
  which namespace each belongs to, per **D2**), `lambda_resp`, `lambda_align`, `K_resp` (the spectral
  bin set), `M_pts`, `f_min`/`f_max`, `N_fft`, `d_proj`, `tau`, `w_core`, `A_core` pooling, and a
  one-line statement that **Stage 2 now has a band-limited spectral term again** (the 2026-11-09 removal
  note is superseded for the physiological stream only — the *visual* streams still have no spectral
  term).
- Regenerate `code/notes/spec/` with `python code/scripts/project/write_specs.py`:
  `03_engine_and_loss.md` gains the composite objective, the masked-only/detached-splice rules, the
  InfoNCE gate and the gradient-balance diagnostic; `02_model_architecture.md` gains the two
  `AlignProjection` heads and their Stage-3 exclusion; `04_configs_and_execution.md` gains the twelve
  keys.

---

## 5. Decisions taken while planning (review these first)

| id | decision | why / alternative rejected |
| :--- | :--- | :--- |
| **D1** | **APPROVED (2026-11-25).** The `L_MSE` term is the **existing** `self.loss_fn(pred, tgt, mask_s)` call on the full `[B, N_s, k]` tensors, handed into the composite as a pre-computed scalar. | Criterion 1.1 demands bit-identity at neutral. Gathering first and re-reducing would change the summation order (per-token `mean` then token-sum vs one flat mean) and differ at ~1e-7 — enough to fail a bit-exact fixture. It also keeps `mse_resp` the *same number* in both arms, which §2.7 requires for comparability. Rejected: computing MSE inside `CompositeRespLoss` from the gathered tensors. |
| **D2** | A **new** module `core/resp_composite.py`; `core/waveform_losses.py` is not modified. | The Stage-3 `PearsonLoss` is a different object (unmasked, `[B, T]`, `eps` outside the norms, no variance floor) and the Stage-3 `alpha/beta/gamma` already name Huber/Pearson/MR-STFT. Reusing it would silently change Stage-3 semantics (forbidden by the research-python instructions) and conflate two symbol namespaces. |
| **D3** | **APPROVED (2026-11-25).** Two plan-level preconditions beyond §2.6: `beta > 0 or gamma > 0` requires `target_norm in ('none','clip')` **and** exactly one physiological stream. | §2.2.3's spliced full clip is only a waveform if the target is coherent over the clip; `'token'` normalisation inserts a step per token and the passband would then describe a splicing artefact. The `[0.10, 0.60] Hz` band is a *respiration* band, so a second signal stream (`bp`, ~1 Hz) must not share it — the §2.4(8) cardinality logic applied to the loss instead of the alignment — and `out['objective']` is a singular `resp_*` key set that a second signal stream would silently overwrite. The `beta > 0` half matters even though Pearson is band-free: it is equally respiration-specific and equally single-keyed. Neither check can reject a legal ADR-008 tuple (arm: `'none'`, one signal), so they are disclosed as additions, not as ADR edits. |
| **D4** | **APPROVED (2026-11-25).** The `objective` manifest block is written only when non-neutral, and the resume guard compares it only when **either** side has it. | §2.7 requires legacy manifests to stay unchanged **and** disagreement to fail loudly. An unconditional key would make every in-flight ADR-007 run un-resumable (the ADR-007 plan hit exactly this trap with the single-carrier keys). Rejected: always writing `objective: {}`. |
| **D5** | `out['objective']` and `out['diagnostics']` are **conditional on the config**, never on the sampled mode. | ADR-005 contract 9 / §2.4(6): a key set that depended on the mode desynchronises a distributed run and makes epoch means meaningless. `grad_balance` is additionally step-conditional, which is safe because `step` is global and identical on every rank. |
| **D6** | **APPROVED (2026-11-25; corrected in review).** The gradient-balance probe lives in `forward` and is driven by a new optional `step=` keyword; it differentiates w.r.t. **`z`** and slices the **gradient** rows by the tracked visual offset. | It needs `z` (a non-leaf intermediate) and both losses in the same graph; any external implementation would have to re-run the forward or attach hooks. `retain_graph=True, create_graph=False` makes it non-destructive. **Why not `z[:, :K_tir]`:** a slice view is not a node the loss consumed, so `autograd.grad(term, z[:, :K_tir])` raises `RuntimeError: One of the differentiated Tensors appears to not have been used in the graph`. Rejected: `z_tir.retain_grad()` + a post-backward read, which leaks a buffer per step and only works after the optimizer step. |
| **D7** | Alignment pooling uses `ids_keep % n_spatial` with an explicit `core_indices` mask, and `core_weight = 1.0` / empty `A_core` collapse to plain mean pooling with **no branch**. | §2.3.1's formula is exactly this, and a branch-free implementation makes the "uniform ablation + default weight = unweighted mean" claim testable rather than asserted. |
| **D8** | `--align_gather_ddp` is `str2bool` with `const=True`, not `store_true`. | Default is `True`; a bare `store_true` cannot express "off" in a YAML file (the `--find_unused_parameters` precedent). |
| **D9** | No new eval/recon flag; the arm's probe is the existing `--resp_probe hidden --kept_resp 0.0 --kept_tir 1.0` (equivalently `--pin_single_carrier tir`, which on a one-visual checkpoint *is* Mode 1). | The ADR changes the objective, not the placement or the pin semantics. `--kept_tir 1.0` is the explicit spelling of the ADR-007 `200 / 0` Mode-1 pin (a `0.0` fraction relies on the residual fill and reads like an ablation of the visual stream). `run_eval_recon` rebuilds the model from `ckpt['args']`, so the heads travel with the checkpoint and load without a flag. |
| **D10** | An `[objective] OFF-ADR-008` banner when `alpha == 0.0 or beta == 0.0 or gamma == 0.0 or lambda_align == 0.0` (i.e. not all three terms plus the regularizer). | Same failure mode ADR-007 **D10** guards: `--streams tir,resp` typed without the YAML looks identical in every log line while running pure MSE at `signal_weight 0.5`. `not composite_active` is insufficient: a `beta`-only run is `composite_active` but has no spectral term. |

---

## 6. Risks / non-goals carried forward

- **Non-comparable total validation loss** (ADR-008 §4): `val/loss` mixes composite terms and InfoNCE.
  Cross-arm claims must use `mse_resp`, the §3.2 pretext diagnostics, and §3.3's subject-level
  band-limited Pearson. The plan therefore does **not** wire `val_criterion` to anything new; the arm
  keeps `val_criterion: joint_loss` for checkpoint selection only.
- **Over-weighting can starve the visual path** (§4): `mse_tir` divergence and
  `grad_ratio_resp_to_tir < 0.05` are the two rollback triggers; G12 records the ratio baseline.
- **InfoNCE false negatives** (§4): same-subject clips in one batch are inherent; `align_gather_ddp`
  enlarges the pool but does not remove them. Not fixable at this layer — recorded, not hidden.
- **Deferred, non-blocking** (ADR-008 §5): the `w_core = 2.0` ablation, the `lambda_resp`/`lambda_align`
  sweeps, and the `[0.15, 0.50] Hz` clinical-band comparison. None of them belong in this plan.
- **Not in this plan:** the matched pure-MSE ADR-007 Fold-0 run, the Fold-0 pretext evaluation, the
  five-fold Stage-3 sweep, and any thesis text. All are run-protocol / `code/notes/eval/` deliverables.

---

## 7. BDD Test Cases

```gherkin
Feature: Stage-2 composite respiration objective
  The ADR-008 objective must not trade the flat conditional mean for a numerical accident.

  Scenario: A flat prediction is expensive, an exact one is free
    Given a unit-variance 0.25 Hz sinusoid of 8000 samples at 1000 Hz, all tokens masked
    When the prediction is identically zero
    Then L_Pearson is 1.0 and L_spectral is 1.0 within the ADR tolerances
    And the gradient of L_resp with respect to the prediction is non-zero
    When the prediction equals the target
    Then L_MSE, L_Pearson and L_spectral are each below 1e-6

  Scenario: Mode-2 placeholder zeros never reach the correlation or the spectrum
    Given Mode-2 masking with K_resp in 25, 50, 75, 100
    When predicted values at unmasked positions are perturbed
    Then no term and no gradient changes

  Scenario: Neutral defaults are the old run
    Given alpha=1, beta=0, gamma=0, lambda_align=0
    When the model is built and a fixed plan is forwarded and backwarded
    Then no alignment parameters exist
    And the loss, the per-stream MSE and the parameter count match the pre-change fixture exactly

Feature: Cross-modal InfoNCE alignment
  Scenario: Direct supervision reaches both encoders, only where a physiological latent exists
    Given Mode-2 masking with K_resp > 0 and a batch of at least two clips
    Then L_align is positive and both adapters receive non-zero gradients
    Given Mode-1 masking with K_resp = 0
    Then loss_align is 0.0, align_active is 0, and every projection parameter has an exact zero gradient

Feature: Startup feasibility
  Scenario: An invalid objective tuple never trains
    Given any violated precondition of ADR-008 section 2.6
    When the model is constructed
    Then a startup error names the condition and the numbers

  Scenario: Single-signal and coherent-target guards protect composite terms
    Given a two-signal stream set "tir,bp,resp" or per-token target_norm "token"
    When the model is constructed with resp_loss_pearson_weight > 0 or resp_loss_spectral_weight > 0
    Then construction raises a ValueError naming the violated precondition

Feature: Composite loss edge cases and numerical stability
  Scenario: Degenerate constant targets never produce NaN or Inf
    Given a batch whose respiration target is identically zero
    When L_Pearson and L_spectral are evaluated on Mode 1 or Mode 2
    Then both terms return finite 0.0 and backward produces finite gradients

  Scenario: Spectral loss is selective to the respiratory passband
    Given a target oscillating at 0.25 Hz inside [0.10, 0.60] Hz
    When the prediction oscillates at 0.05 Hz or 0.90 Hz outside the passband
    Then L_spectral stays near its unaligned ceiling and does not collapse toward 0.0

  Scenario: Non-unit alpha activates composite tracking without altering raw mse_resp
    Given alpha=0.5, beta=0, gamma=0, lambda_align=0
    When one step is forwarded
    Then model.resp_loss is constructed and out['objective'] is present
    And weighted resp loss uses 0.5 * raw_mse while resp_mse stays equal to losses_mse['resp']

Feature: Anatomy-weighted InfoNCE pooling and distributed validation safety
  Scenario: Core-weighted pooling prioritizes anatomical core tubes
    Given an anatomy_aware two-stream model in Mode 2 with align_core_weight=2.0 and B >= 2
    When L_align is backwarded into the encoder latents
    Then the mean gradient norm on A_core visible tir tokens exceeds that on A_periph tokens
    And with align_core_weight=1.0 or spatial_sampling="uniform" pooling matches plain mean pooling

  Scenario: Single-sample batch and unpadded validation never deadlock distributed gather
    Given a two-stream ADR-008 model with align_gather_ddp=true
    When Mode 2 runs in eval mode (evaluate_pretrain, drop_last=False)
    Then symmetric_infonce skips dist.all_gather and computes local negatives only
    And evaluate_pretrain keeps set(components) == {'mse_tir', 'mse_resp'} while populating result['objective']
    When a single-device batch has B == 1
    Then L_align evaluates cleanly to 0.0

Feature: Mixed-precision FP16 and BF16 execution
  Scenario: Composite loss and InfoNCE remain finite under autocast
    Given a two-stream ADR-008 model running under FP16 or BF16 autocast
    When forward and backward execute on Mode 1 and Mode 2
    Then MaskedPearsonLoss, BandLimitedSpectralLoss (rFFT), and symmetric_infonce run in FP32
    And all losses and parameter gradients are finite with no complex-FFT dtype error

Feature: Non-destructive gradient-balance diagnostic
  Scenario: Probe slices dL/dz cleanly on both ADR-008 and pure-MSE ADR-007 runs
    Given a two-stream model (either ADR-008 or neutral pure-MSE ADR-007) with grad_balance_freq=2
    When training steps 0, 1, and 2 are forwarded with step=global_step
    Then out['diagnostics'] is emitted only on steps 0 and 2
    And grad_norm_tir and grad_norm_resp match a reference autograd.grad(losses[s], z) sliced at offset_tir (including when streams="resp,tir")
    And all model parameter .grad buffers remain byte-identical before and after the probe
    And when grad_balance_freq=0 no diagnostics key is emitted and zero extra autograd.grad calls occur

Feature: Manifest resume guard and Stage-3 transfer isolation
  Scenario: Objective changes block resume while read-only cadence and legacy resumes pass
    Given a run output directory with an existing masking.json
    When an ADR-008 run writes masking.json
    Then objective.adr is "ADR-008" with spectral_n_fft=32768 and 16 spectral_band_bins, and grad_balance_freq is stored under run
    When the ADR-008 run is resumed with a different align_weight
    Then startup aborts with a manifest mismatch error
    When the ADR-008 run is resumed with a different grad_balance_freq
    Then startup succeeds without a manifest error
    When a legacy neutral ADR-007 masking.json (no objective key) is resumed with neutral defaults
    Then startup succeeds and leaves masking.json untouched

  Scenario: Stage-3 weight transfer ignores auxiliary alignment heads
    Given a saved ADR-008 Stage-2 checkpoint containing align_proj.* tensors
    When load_stage2_encoder loads it into MultiModalWaveformRegressor
    Then all encoder tensors match the pure-MSE checkpoint with 0 shape mismatches
    And no align_proj.* key is copied or reported as an unexpected error
```

---

## 8. Executed record (2026-11-25)

- **Phase 0 fixture:** `code/tests/fixtures/adr008_neutrality_fixture.json`, produced by
  `code/scripts/project/capture_adr008_neutrality.py` **before** any `core/` edit.
  `tiny` = 321,416 parameters / 44 state keys; `adr007_geometry` (ViT-B, the shipped tokenizer) =
  102,528,032 parameters / 182 state keys. Both entries pin `total`, every `losses[s]`, every
  `losses_mse[s]`, five gradient norms (tiny only), `mode`/`mode_resolved`, the parameter count and the
  sorted key list, from `plan_from_counts` allocations (no random draw). `--check` reproduces them
  bit-for-bit in **6.4 s**.
- **Phase-0 arithmetic, verified before coding** (all confirmed on the live config):
  `N_fft = 32768`, `df = 0.0305176 Hz`, `K_resp = {4..19}` ⇒ **16 bins**, `M_pts ∈ {8000, 7200, 6400,
  5600, 4800}`, projection heads `2 x 689,024 = 1,378,048` parameters.
- **New modules:** `code/core/resp_composite.py` (`MaskedPearsonLoss`, `BandLimitedSpectralLoss`,
  `CompositeRespLoss`) and `code/core/align_infonce.py` (`AlignProjection`, `anatomy_weighted_pool`,
  `symmetric_infonce`). `core/waveform_losses.py` untouched (D2).
- **`core/multimae.py`:** the twelve §2.8 keywords; `_validate_objective()` (rules 1-6, normative
  order); `composite_active` / `objective_active` / `align_proj` / `resp_loss` constructed **before**
  `self.apply(self._init_weights)`; the composite in the decode loop (D1: the MSE term is still the
  original `self.loss_fn` call); per-stream `z` offsets; the InfoNCE gate; `_dz_norm()` (D6); the
  `objective` / `diagnostics` output blocks; the `[objective]` provenance line and the OFF-ADR-008
  banner; `build_pretraining_model` forwarding.
- **Runner / engine:** twelve flags (`--align_gather_ddp` as `str2bool`); `masking.json` gains
  `objective` (only when non-neutral) and `run.grad_balance_freq` (only when armed) with the D4 resume
  guard; `_accepts_kw()` + `step=` pass-through; per-step logging of the objective block and of the
  probe; `evaluate_pretrain` returns a sample-weighted, DDP-merged `objective` sibling (the
  `components` set is unchanged); `[val] objective …` print + W&B series.
- **Arm:** `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`, generated from
  the ADR-007 arm so that **only** the twelve objective keys differ (verified by a key-level diff), plus
  the `TIR_ROI_CONFIGS` registration.
- **Measured on the real arm geometry:** the built model is **103,906,080** parameters, i.e. exactly
  `+1,378,048` over the pure-MSE ADR-007 model (102,528,032); `n_fft 32768`, 16 bins,
  `lambda_resp {'tir': 1.0, 'resp': 5.0}`.
- **Measured objective behaviour (unit scale, `L = 8000`, `fs = 1000`):** flat prediction ⇒
  `L_Pearson = 1.000`, `L_spectral = 1.000`; exact prediction ⇒ all three terms `< 1e-6`; a wrong
  in-band rate (0.10 / 0.60 Hz vs a 0.25 Hz target) ⇒ `L_spectral > 0.5`; out-of-band 0.05 / 0.90 Hz ⇒
  `1.203` / `1.597`. On the tiny model: Mode 2 `loss_align ≈ 1.58` with non-zero gradients on both
  `align_proj` heads and `grad_ratio_resp_to_tir ≈ 16.8-21.3`; Mode 1 `loss_align = 0.0`,
  `align_active = 0` and **every** projection gradient exactly `0.0` (present, not `None`).
- **Gates:** G1-G5, G7-G12 green in the two new suites (54 tests); G6 green for **both** FP16 and BF16
  on CPU. G8's DDP-aggregation half rides the existing `test_val_gather.py` `_merge_sums` path, which
  is unchanged. G12's resource half (peak VRAM vs the pure-MSE run, `+5 %`) remains a **smoke-run**
  deliverable — it needs a GPU and the caches.
- **Deviation found during implementation (report, do not silently accept).** G6 exposed a
  **pre-existing** autocast defect in `MultiModalMAE.forward`: `pred` was allocated with
  `torch.zeros(...)` (float32) and then `pred.scatter_(..., pred_masked)` received a fp16/bf16 tensor,
  raising `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`. Every autocast run of
  the model would have hit it (the ADR-007 smoke ran `precision none`, so it was never exercised). Fixed
  by allocating `pred` with `dtype=pred_masked.dtype`; in FP32 this is byte-identical, and the Phase-0
  fixture still verifies bit-identity afterwards. **This is a change to a pre-existing code path that
  the plan did not anticipate** — flagged here rather than folded into a phase.
- **Still open:** the matched pure-MSE ADR-007 Fold-0 run (ADR-008 §3.2/§3.3 comparator), the Fold-0
  pretext evaluation, the five-fold Stage-3 sweep, and the §3.4 smoke with VRAM measurement.
- **Code-review follow-up (2026-11-25).** Two plan-vs-code gaps, both closed:
  - **Phase 1.5** — `code/core/__init__.py` now re-exports `MaskedPearsonLoss`,
    `BandLimitedSpectralLoss`, `CompositeRespLoss`, `AlignProjection`, `anatomy_weighted_pool` and
    `symmetric_infonce` alongside `criterion` / `waveform_losses` (leaf modules, import order unchanged).
  - **Phase 4.6** — `test_phase46_recon_builder_loads_an_adr008_checkpoint_cleanly` now exercises
    `runners.run_eval_recon.build_model` on a saved ADR-008 checkpoint and asserts **0 missing / 0
    unexpected** tensors, i.e. the recon path reconstructs the alignment heads from `ckpt['args']`
    instead of probing a partly random model. (G11's `load_stage2_encoder` check answers a different
    question: that Stage 3 *excludes* the heads.)
  - Suite after the follow-up: **942 passed** (940 + 2).
  - **Unchanged and still open:** the G12 GPU resource smoke (peak VRAM within `+5 %` of the pure-MSE
    run) needs a GPU and the dataset caches.
- **G12 smoke — EXECUTED 2026-11-25 on one Tesla T4 (15 GB), fold 0 of the real corpus**
  (`--dev_smoke --max_entries 100 --batch_size 8 --num_workers 0 --epochs 1 --warmup_epochs 0
  --virtual_sensing_warmup_epochs 0 --grad_balance_freq 2`, caches `output/tir_roi_cache` +
  `output/resp_cache`, `--raw_root /work/projects/l0003511/test`; 1 epoch = 12 steps, 100 val samples):
  - **Clean run:** no `NaN`, no assertion, `params = 103,906,080`, the `[objective]` line reports
    `alpha=1 beta=2 gamma=1  lambda_resp=5  n_fft=32768 (16 bins)  lambda_align=0.5  tau=0.07
    d_proj=128  w_core=1  gather=True  grad_balance_freq=2  target_norm=none`, and the
    `OFF-ADR-008` banner does **not** fire.
  - **Both modes executed** (Mode-2 probability 0.50, seed-dependent: 2 Mode-2 + 10 Mode-1 steps).
    Mode 2 logged `align_active = 1` with `loss_align ≈ 2.05`; Mode 1 logged `loss_align = 0.0` with
    `align_active = 0` — the §2.4(5) gate on real data.
  - **Objective terms per step (Mode 1 held-out draw):** `resp_mse 1.740`, `resp_pearson 0.993`,
    `resp_spectral 0.522`; the epoch means are `resp_mse 1.893`, `resp_pearson 0.995`,
    `resp_spectral 0.532`, `loss_align 0.345`, `align_active 0.167`. `mse_tir` fell 2.008 → 1.952
    (no visual divergence).
  - **Validation block emitted** as a sibling of `components`:
    `align_active=0.6000  loss_align=1.2458  resp_mse=1.5912  resp_pearson=1.0004
    resp_spectral=0.5257` over 100 samples.
  - **Resource gate (the §3.4 `+5 %` criterion): PASSED.** Peak device memory, identical caps and
    precision (`none`), same hardware:
    ADR-008 **3.80 GB allocated / 4.82 GB reserved** at 2.03 s/it vs. the matched pure-MSE ADR-007 arm
    **3.78 GB / 4.67 GB** at 1.97 s/it ⇒ **+0.53 % allocated, +3.2 % reserved**, both inside `+5 %`.
    The `[TODO]` in ADR-008 §3.4 can be closed with these numbers (batch 8, Tesla T4 15 GB, FP32).
  - **`masking.json` verified:** the ADR-008 record carries `objective.adr = 'ADR-008'`,
    `spectral_n_fft = 32768`, `spectral_band_bins = [4..19]`, `signal_weight = 5.0`, and
    `run.grad_balance_freq = 2`; the ADR-007 record has **no** `objective` key. `policy`, `geometry`
    and `kernel` are byte-equal across the two records, so the arms differ by the objective alone.
  - **Observed, needs a real run before it is judged (report, do not tune on it):** the
    `grad_ratio_resp_to_tir` epoch mean is **99.1** for ADR-008 vs **4.47** for the pure-MSE
    comparator (probe step 0: 97.2). ADR-008 §3.2's *success* band is `[0.05, 0.50]` and its
    *failure* threshold is `< 1e-3`, so both clear the failure condition and **both sit above the
    success band** — at step 12 from a random physiology head that is expected (the MSE term is still
    at its ~2.0 init value), but it is exactly the quantity §3.2 re-measures on Fold-0 held-out after
    convergence. If the converged ratio still exceeds `0.50`, ADR-008 §5's deferred
    `lambda_resp ∈ {2, 5, 10}` / `lambda_align ∈ {0.1, 0.5, 1.0}` sweep is the documented remedy.
  - Reproduce with:
    ```bash
    cd code && python -u runners/run_pretrain.py \
        -c configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml \
        --exp_root /tmp/adr008_probe --fold 0 --dev_smoke --epochs 1 --warmup_epochs 0 \
        --virtual_sensing_warmup_epochs 0 --max_entries 100 --batch_size 8 --num_workers 0 \
        --grad_balance_freq 2 --output_dir /tmp/adr008_smoke/out \
        --roi_cache <TIR_ROI_CACHE> --resp_cache <RESP_CACHE> --raw_root <RAW_ROOT>
    ```
- **Rollback triggers** (ADR-008 §4): any Criterion-1 failure; Fold-0 `Var(y_hat) < 0.10` or subject
  median `r < +0.10`; `grad_ratio_resp_to_tir < 1e-3`; `mse_tir > 2x` the pure-MSE ADR-007 run; or the
  five-fold band-limited Pearson failing the `+0.02` margin (then neutral defaults disable everything:
  `beta = gamma = lambda_align = 0`, `lambda_resp = 0.5`).
