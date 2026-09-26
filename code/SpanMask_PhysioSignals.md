# Span Masking for the 1-D Physiological Streams

### — and why the STFT (MR-STFT) loss becomes useful once you have it

**Status:** design document. **Nothing in this document is implemented yet** —
`core/multimae.py` still masks every non-visual stream with `_random_mask`
(scattered dropout), and `spectral_weight` is `0.0` in the shipped Stage-2
config. The measurements below were taken on the shipped TIR-ROI + RESP geometry
(2026-09-26); the probe scripts and commands are listed in §8 so every number can
be reproduced.

**Relationship to the other docs.** The plan already prescribes this masking
policy — `ImplementationPlan.md` §2.2 ("Physio: contiguous span masking, not
scattered dropout … use contiguous spans of >=1.5-2 s, `n_spans` 1-2, and ramp
the ratio 0.9 -> 1.0"), `simplifiedPlan.md` §4/§10 (`mask_span_s: 2.0`) and §11
item 5 ("contiguous **span** masking for the 1-D streams | `core/multimae.py`
(new `_span_mask`) | scattered dropout is locally solvable"). What was missing is
the implementation and the evidence. This document supplies the evidence
(§1-§3), the implementation contract (§4) and the answer to "why would the
spectral loss help then?" (§5-§6). The measured negative result for adding the
STFT term to the *current* scattered mask is in `TirROI_Resp_plan.md` §9.

---

## 0. TL;DR

1. **The physio stream is not learning** under the shipped objective. Measured:
   the trained model reaches `mse_resp 1.052` / `spec_resp ~5.0` — i.e. **worse
   than a constant predictor** (`mse 1.0` / `spec 4.7-5.1`) and 8x worse than a
   5-line interpolator.
2. **Why:** `_random_mask` masks 25 of the 50 resp tokens *scattered
   individually*. Each token covers 0.16 s and respiration has a 2.5-6.25 s
   period, so the stream is oversampled 16-39x and **every masked token has a
   visible neighbour 0.16-0.32 s away**. Interpolating them (using only the
   visible resp, zero video) achieves `mse 0.129` / `spec 0.98` — the task is
   solvable without the encoder, so nothing forces the encoder to represent
   respiration.
3. **The STFT term cannot fix that**, and did not: its optimum is already
   attained by that same interpolator, and at the weights tried it produced
   `spec_resp 3.7e8` and destroyed the model (`TirROI_Resp_plan.md` §7).
4. **Span masking removes the shortcut.** Measured: with a single 4 s
   contiguous span, the no-video interpolator becomes *worse than a constant*
   (`mse 1.459` vs `0.976`), and its spectrum degrades 2.4x
   (`spec 2.37` vs `0.98`). The gap can no longer be filled from the visible
   samples — the encoder must contribute.
5. **Only then does the spectral term become informative.** Under a long gap the
   characteristic failure is a *plausible but wrong* waveform (blurred, shrunk,
   or at the wrong rate/phase). Measured: 1 s boxcar smoothing reaches
   `MSE 0.090` while its spectrum is `2.33` — i.e. a magnitude loss sees the
   damage an amplitude loss forgives (§3). MR-STFT is therefore a **complement**
   to the MSE, not a replacement, and it must be re-introduced only as an
   auxiliary term with the safeguards in §5.6.
6. **Order of work:** implement `_span_mask` (§4) → verify the encoder starts
   beating the no-video floors (§7 criteria) → *then* re-evaluate the spectral
   term (§6 ladder).

---

## 1. The problem, precisely measured

Geometry (shipped `configs/pretrain/stage2_local_tir_roi_resp.yaml`): 8 s clips,
`fs 100`, `temporal_stride 2`, `sig_kernel 16` -> `L 800` samples, `N 50` resp
tokens, 160 ms per token; `mask_ratio_resp 0.5` -> 25 masked tokens; `tir` is
the other stream (`mask_ratio_tir 0.9`). `target_norm: clip`, `signal_weight
0.5`, `spectral_weight 0.0`.

| predictor (uses the **visible resp samples only** — no video, no encoder, no learning) | `mse_resp` (masked) | `spec_resp` (MR-STFT 64/128/256) |
| -------------------------------------------------------------------------------------- | ------------------- | -------------------------------- |
| constant 0 — the trivial floor                                                          | 0.9989              | 5.0932                           |
| **linear interpolation between the nearest visible samples**                             | **0.1299**          | **0.9709**                       |
| ground truth (upper bound)                                                              | 0.0                 | 0.0                              |
| *for reference:* the trained model (epoch 1, same geometry)                              | 1.052               | ~5.0                             |

24 real clips x 30 mask realisations, 388-clip corpus (`probe_mask_solvability.py`).

**Reading.** The objective's achievable region contains a solution (interpolation)
that needs *none* of the cross-modal structure Stage 2 is supposed to learn, and
the trained 102 M-parameter model is **8x worse** than it. This is not "the task
is hard"; it is "the task has an easy solution nobody is taking, because the
gradient path that would find it is not the one being optimised" — the encoder
gets no pressure from a target that its own visible half already determines.

## 2. Why scattered masking fails: the oversampling argument

* `_random_mask` (`core/multimae.py`) picks `k = max(1, min(N-1, int(ratio*N)))`
  tokens **independently** — no notion of contiguity.
* The respiration band is 0.16-0.4 Hz (period 2.5-6.25 s), the token grid is
  160 ms, and the sample grid is 10 ms. One breath spans 16-39 tokens and
  250-625 samples.
* Consequence: a masked token is almost always flanked by visible tokens
  (probability of being isolated in a long run is negligible at 50 % density),
  so linear reconstruction is near-free. The measured `mse 0.129` is essentially
  the second-difference energy of a low-pass signal at 0.32 s spacing.

Add to that the *loss* side: a blur is cheap under an amplitude loss. Measured,
replacing the whole clip with a **1 s boxcar smoothing of itself** gives
`MSE 0.090` — better than the interpolation floor — while its spectrum is
`2.33`, i.e. the signal's own spectral content is largely gone. So the scattered
objective does not merely *allow* the shortcut; it actively rewards blurring.

## 3. What each loss actually sees (measured)

Same true clip, controlled perturbations (mean over 24 clips,
`probe_loss_sensitivity.py`):

| prediction error vs the true clip | MSE | MR-STFT (64/128/256) | MR-STFT (128/256/512) |
| --------------------------------- | --- | -------------------- | --------------------- |
| zero (constant) | 1.0000 | 10.1850 | 10.4701 |
| amplitude x0.5 (shrink) | 0.2500 | 1.1911 | 1.1926 |
| time shift 1.00 s (=100 samples) | **2.5569** | **2.0158** | 2.1547 |
| time shift 0.25 s (=25 samples) | 0.3916 | 1.3880 | 1.4662 |
| smoothed with a 1.0 s boxcar | **0.0895** | **2.3300** | 2.5718 |
| linear chord across the clip | 1.5781 | 4.7414 | 5.0938 |
| sign-flipped (-z) | **4.0000** | **0.0000** | 0.0000 |

Three conclusions that shape the design:

1. **MSE forgives spectral damage.** 1 s smoothing is the *best* MSE score in the
   table and one of the *worst* spectra. If the reconstruction must carry the
   respiration rate/shape into Stage 3, an amplitude loss alone cannot police it.
2. **MR-STFT forgives phase/polarity.** A 1 s displacement costs `MSE 2.56` but
   only `spec 2.02` (its spectrum is intact), and a global sign flip is
   spectrally **free** (`spec 0.0000`) while being `MSE 4.0`.
   => A magnitude-spectral loss must **never** be the only term; it is
   insensitive to polarity and to time displacement by design.
3. **They are complementary, not redundant.** Amplitude terms fix *where* the
   waveform is; the spectrum fixes *what is in it* (rate, band, and — critically
   — whether the model has blurred the answer).

## 4. The span-masking design

### 4.1 Parameters — PER-MODALITY, not global

**Today neither number exists.** There is no span concept anywhere in `core/`
(`grep -rn span core/*.py` returns only the time-*span* alignment docstrings), and
`make_masks` masks every non-visual stream with `_random_mask`. The only
per-modality masking parameter that exists is the **ratio**:
`--mask_ratio_bp / _resp / _eda` (`run_pretrain.py:89-93`), collected into
`MultiModalMAE.mask_ratios`. So the honest status is "missing", not
"hardcoded".

**The two new knobs must be per stream**, because the masking threshold is set by
the *target's own correlation time* and the thesis' bands differ by ~1-2 orders
of magnitude (`bp` 1-2.5 Hz, `resp` 0.16-0.4 Hz, `eda` aperiodic with a 2-10 s
tonic scale). One global `mask_span_s` cannot be right for all three.

| stream | band (plan §2) | period / correlation time | `mask_span_s` default | tokens @160 ms |
| ------ | -------------- | ------------------------- | --------------------- | -------------- |
| `bp` | 1-2.5 Hz | 0.4-1.0 s | **1.0** | 6 |
| `resp` | 0.16-0.4 Hz | 2.5-6.25 s | **4.0** | 25 |
| `eda` | aperiodic, tonic 2-10 s | 2-10 s | **8.0** | 50 |

Proposed surface (additive, mirroring the existing per-stream conventions —
`MultiModalMAE` already takes `spectral_weights` as a **dict keyed by stream
name**, so use that form rather than a positional CSV):

```yaml
# one scalar = every physio stream, or a name-keyed mapping
mask_span_s: 4.0                     # all physio streams get a 4 s span
mask_span_s: {resp: 4.0, bp: 1.0}    # per modality (recommended for multi-stream runs)
mask_n_spans: 0                      # 0 = DERIVE (see below); ignored if set
```

**The three knobs are over-determined:**
`n_spans * span_s = ratio * clip_seconds`. Make `mask_span_s` (per stream) and
the existing `mask_ratio_<stream>` the primaries and **derive**
`n_spans = round(ratio * clip_seconds / span_s)`, then log both — a silent
mismatch means the effective ratio differs from the one you configured. This
also gives a hard feasibility bound:

```
span_s <= ratio * clip_seconds        (since n_spans >= 1)
```

At `ratio 0.5` a 4 s span is only reachable with an 8 s clip and `n_spans == 1`
— which is exactly the shipped configuration. A 4 s gap **with visible context on
both sides** needs a 12-16 s clip, and the `eda` default (8 s) is unreachable at
8 s clips, so the EDA branch would need longer windows.

**How to determine the values (a measurement, not a guess):** the design goal is
"the gap must not be fillable from its own visible edges". Pick the *smallest*
span for which the no-video interpolation baseline becomes **worse than the
constant predictor** (`mse_interp > mse_const` in the §4.2 table). For RESP at
this geometry the threshold sits between a 1 s and a 2 s **individual gap**, i.e.
at roughly **0.5-1 target cycle** — which is where the defaults above come from.
Two consequences visible in the §4.2 numbers:

* **What matters is the length of each individual gap, not the total masked
duration.** All three span rows mask 4.0 s of 8 s, yet 4x1 s stays fillable
(0.84), 2x2 s breaks it (1.37) and 1x4 s breaks it hardest (1.46). So `n_spans`
is not an independent difficulty dial: with the ratio fixed, *fewer spans =
longer gaps = harder*.
* The floors (§4.2) are **geometry- and modality-specific** — they depend on
`N`, `sig_kernel`, `fs`, the clip length and the signal's own spectrum. Re-measure
them per modality rather than importing the RESP numbers.

### 4.2 Measured difficulty by mask pattern

24 clips x 30 realisations, 25/50 tokens (4.0 s of 8 s) masked
(`probe_span_baselines.py`):

| mask pattern | `mse_const` | `mse_interp` | `spec_const` | `spec_interp` 64/128/256 | `spec_interp` 128/256/512 |
| ------------ | ----------- | ------------ | ------------ | ------------------------ | ------------------------- |
| scattered (shipped) | 1.0110 | **0.1288** | 5.1082 | **0.9822** | 0.9783 |
| **1 span (4.0 s)** | 0.9758 | **1.4585** | 4.6700 | **2.3665** | 2.2027 |
| 2 spans (~2 s each) | 0.9964 | 1.3697 | 4.3589 | 1.9945 | 1.8727 |
| 4 spans (~1 s each) | 1.0080 | 0.8411 | 4.3457 | 1.6034 | 1.4971 |

**This is the central result of the document.** Moving from scattered tokens to
one contiguous span flips the interpolator from *8x better than the trained
model* to *worse than a constant predictor* (`1.459 > 0.976`), and degrades its
spectrum by 2.4x. Difficulty then scales monotonically with span length
(4 spans 0.84 -> 2 spans 1.37 -> 1 span 1.46), which gives a controllable
difficulty dial: start at `n_spans 2` / `mask_span_s 1.5-2.0`, lengthen as the
model copes.

### 4.3 Invariants the implementation MUST respect

These come from `core/multimae.py::forward` and are not optional:

1. **Identical visible count per sample within a batch.** `forward` computes
   `k = int((mask_s == 0).sum(dim=1).max())` — a **batch max** — and gathers the
   first `k` entries of `argsort(mask)` as "visible". If one sample has 20
   visible tokens and another 25, the 20-visible sample contributes 5 *masked*
   tokens to the encoder: **masked tokens leak into the encoder** and the
   reconstruction becomes trivially self-referential. So either
   (a) draw the span length ONCE per step and share it across the batch (random
   *start* per sample is fine — it does not change the count), or
   (b) make `forward` per-sample aware. (a) is a 3-line change and is the
   recommendation.
2. **`max(1, min(N-1, int(ratio*N)))` forbids ratio 1.0**, so the plan's
   "ramp to 1.0" needs the clamp relaxed *and* `k=0` handled downstream
   (`enc_s` becomes an empty slice — `forward` already tolerates it, and the
   previous session verified `MaskedMSELoss` still has a valid all-ones mask).
   **Caveat measured earlier:** with `k = 0` the decoder input for that stream
   is `mask_token + pos_embed` only, i.e. *no* path to the encoder, so the
   "fully masked" case trains the decoder, not the encoder — see §5.5. The clamp
   that keeps >= 1 visible token is what preserves the encoder gradient path.
3. **Do not touch the geometry contract.** Span masking changes only *which*
   tokens are masked; `T`, `L`, `grid_t == n_signal` and the
   `sig_kernel = tubelet_t*temporal_stride/fps*fs` identity are untouched.
4. **Determinism / DDP.** Masks are drawn from the global RNG after `init_env`
   seeds `torch/np/random` per rank, so the same code path stays rank-consistent
   and comparable across runs. Any new helper must keep using `torch.rand` on the
   model's device (not `numpy`/`random`) to preserve that.

### 4.4 Implementation sketch

```python
# core/multimae.py -- new helper, same signature style as _random_mask
def _span_mask(self, B: int, device, N: int, mask_ratio: float,
               span: int, n_spans: int = 1):
    """Contiguous-block mask: each sample hides ``n_spans`` spans of ``span``
    tokens at uniform random starts.

    ``span`` and ``n_spans`` are passed IN (drawn once per step by the caller)
    so every sample hides exactly the same NUMBER of tokens -- ``forward``
    gathers visible tokens with a batch-max ``k``, and unequal counts would leak
    masked tokens into the encoder.
    """
    k = max(1, min(N - 1, span * n_spans))          # keep >= 1 visible
    start = torch.randint(0, N - k + 1, (B, 1), device=device)
    idx = (start + torch.arange(k, device=device)).clamp(max=N - 1)   # [B, k]
    m = torch.zeros(B, N, device=device, dtype=torch.long)
    m.scatter_(1, idx, 1)
    return m
```

Wiring and knobs (all additive; `_random_mask` stays the default so no existing
config changes behaviour):

* `make_masks`: use `_span_mask` for a non-visual stream when
  `self.physio_mask == 'span'`; otherwise `_random_mask` exactly as today.
* `MultiModalMAE.__init__`: `physio_mask: str = 'random'` ('random' | 'span'),
  `mask_span_s: float = 2.0`, `mask_n_spans: int = 1`, and conversion
  `span_tokens = max(1, round(mask_span_s * fs / sig_kernel))`.
* `build_pretraining_model` + `run_pretrain.py`: `--physio_mask`,
  `--mask_span_s`, `--mask_n_spans`; config keys of the same name.
* Config for the first experiment: add to `stage2_local_tir_roi_resp.yaml`
  ```
  physio_mask: span
  mask_span_s: 2.0
  mask_n_spans: 2
  mask_ratio_resp: 0.5      # 2 spans x 12 tokens = 24/50 ≈ 0.48
  ```
  (the ratio and the span geometry must be mutually consistent: `n_spans * span`
  should equal `round(mask_ratio * N)`, else the effective ratio silently differs
  from the configured one — log both).

### 4.5 Traps

* The batch-max leak (§4.3 item 1) is silent — it produces a *better* loss, not
  an error. Verify it by asserting, in a unit test, that every sample's gathered
  visible tokens are exactly the unmasked ones (`(mask.gather(1, ids_keep)==0).all()`).
* Keep `clip_grad: 5.0` on. Span masking makes the reconstruction *harder*, which
  is exactly the regime where the resp branch's occasional huge gradients appear
  (§7 of the TIR-ROI plan).
* Read the **last step** of an epoch, not the average.
* Ramp, don't jump: going from scattered 0.5 to a single 4 s span in one step
  raises the loss abruptly; a 1.5-2 s span is the gentle start.

### 4.6 Per-modality spans are safe in the code — one leak to decide about

* **Per-stream span lengths are structurally fine.** `forward` computes the
  visible-token count **per stream** inside the stream loop
  (`k = int((mask_s == 0).sum(dim=1).max())` at `core/multimae.py:636`), and the
  gather is per stream. A 6-token `bp` span and a 25-token `resp` span therefore
  coexist without interference; the only requirement is that *within one stream's
  own batch* the visible count is uniform (§4.3 item 1).
* **The physio streams share a time grid**, so masking them differently means the
  decoder sees, e.g., a fully visible pulse alongside a 4 s respiration gap — and
  the decoder's self-attention spans both. Respiration and heart rate are
  physiologically coupled, so a masked `resp` gap can be partly filled from the
  visible `bp` tokens. If the thesis claim is "reconstruct physiology from the
  **video** under sensor failure", then either
  (a) mask all physio streams with **time-aligned** spans (simplest and most
  faithful: the same gap in every 1-D stream), or
  (b) accept cross-physio context and state it explicitly as a leak in the
  write-up.
  The current TIR-ROI run has `streams: tir,resp`, so the question does not arise
  yet; it will as soon as a `rgb,tir,bp,resp` run uses span masking.
* **Video already has the analogous structure.** `_tube_mask` masks one spatial
  subset replicated across **all** time steps — a time-invariant "span" in
  space, which is why the plan calls it leak-free (a masked tube is unseen in
  every frame, so copy-from-the-neighbouring-frame is impossible). The physical
  analogue of the 1-D span length for video is therefore *the whole clip* — there
  is no duration to tune, only the count (`int(ratio * n_spatial)` patches).

### 4.7 Does span masking make `mask_ratio` obsolete for the 1-D streams?

**No — but its job changes, and it becomes partially redundant with the span
parameters.** Three things have to be separated.

**(a) What `mask_ratio_<stream>` still controls (two independent things).**

1. **The visible-token budget** `k = N - masked`. This is not just a supervision
   split: `k` is the only path by which that stream's loss reaches the *encoder*
   (the decoder's input for a masked token is `mask_token + pos_embed`, so with
   `k = 0` the stream's gradient to the encoder is exactly `0.0` — verified in a
   previous session). The ratio therefore decides *whether the physio stream
   trains the encoder at all*.
2. **How close the run sits to the Stage-3 condition** ("full sensor failure"),
   which is why the plan ramps it (`ImplementationPlan.md` §2.2: "ramp the ratio
   0.9 -> 1.0, since the downstream task is the fully-masked case").

**(b) What it stops doing.** The ratio alone no longer determines the
*difficulty*. Measured at the same `ratio 0.5` (all rows mask 4.0 s of 8 s):
`4x1 s` is still fillable (`mse_interp 0.841` < the constant's `1.008`), while
`1x4 s` is not (`1.459` > `0.976`). So "0.5 masked" can mean a trivially
solvable task or a task that requires extrapolation, depending only on the gap
length.

**(c) The redundancy.** With three knobs and two degrees of freedom, one is
always derivable:

```
n_spans * mask_span_s = mask_ratio * clip_seconds
```

Three coherent parameterisations:

| scheme | primary knobs | derived | keeps `mask_ratio`? | note |
| ------ | ------------- | ------- | ------------------- | ---- |
| **A (recommended)** | `mask_ratio_<stream>` + `mask_span_s` | `n_spans = round(ratio*clip/span_s)` | **yes, primary** | ratio stays the "how much", span_s the "how structured"; matches the existing CLI and the plan's ratio ramp |
| B | `mask_span_s` + `n_spans` | `ratio = n_spans*span_s/clip` | becomes derived (log it) | the span is the physical quantity; the ratio must then be *reported* because it is what the loss/`k` sees |
| C | `mask_span_s` only, `n_spans == 1` | `ratio = span_s/clip` | **yes – fully redundant** | the degenerate case: ratio and span_s are the same number in different units |

In an 8 s clip with `n_spans 1`, scheme C is the temptation (a 4 s span *is*
`ratio 0.5`), and then the ratio genuinely can be dropped. Two reasons not to:
it loses the "ramp the fraction" convenience across clip lengths (the same 4 s
span means `ratio 0.5` at 8 s but `0.25` at 16 s), and it breaks symmetry with
the video streams, whose `mask_ratio_*` flags remain meaningful.
**Recommendation: scheme A**, and log the derived `n_spans` *and* the effective
masked fraction next to the configured values.

**(d) The one trap this creates for the plan's ramp.** If you implement
"ramp 0.9 -> 1.0" by *adding spans* at a fixed short `mask_span_s`, every gap
stays short and the interpolation shortcut survives at 90 % masking — the ramp
would be cosmetic. Measured evidence: gap length, not the masked fraction, is
what breaks the shortcut. So the ramp must **grow the gap**, i.e. hold
`n_spans = 1` and let `mask_span_s` rise toward the full clip duration
(0.9 -> a 7.2 s span at 8 s clips), which also lands exactly on the fully-masked
limit case the plan is aiming at. Note that the final step (100 %) additionally
requires relaxing the `min(N-1, ...)` clamp, and at `k = 0` the physio stream
contributes no encoder gradient (§4.3 item 2) — so the last 2 % of the ramp
buys the Stage-3-aligned condition at the cost of the encoder path.

## 5. Why the STFT loss becomes helpful *under span masking*
The argument is not "spectral losses are for periodic signals, so use one". It is
that span masking changes the *failure mode* of the reconstruction from
"interpolation error" to "plausible-but-wrong waveform", and the magnitude
spectrum is the part of the objective that measures the latter.

### 5.1 It stops being trivially satisfiable

| | no-video interpolator `spec_resp` | no-video interpolator `mse_resp` |
| --- | --- | --- |
| scattered (shipped) | 0.98 | 0.13 |
| 1 span (4 s) | 2.37 | 1.46 |

Under scattered masking the spectral term has **nothing to add**: its optimum
(0) is approached to within 0.98 by a solution that uses no video, so the term
cannot discriminate between "the encoder understood the respiration" and "the
decoder copied the neighbours". Under a span mask the same shortcut is
spectrally 2.4x worse, i.e. the term now separates solutions that the MSE
separates only partly (1.46 vs 0.98 — both "bad", but the spectrum orders them).

### 5.2 It targets the failure mode of a long-gap reconstruction

Filling a 4 s gap invites three specific degeneracies, all of which are cheap
under an amplitude loss and expensive under a spectral one (§3):

* **blur / mean-reversion** — `MSE 0.090`, `spec 2.33`;
* **shrinking** the gap amplitude — `MSE 0.25`, `spec 1.19`;
* **a plausible waveform at the wrong rate** — a slow drift has low amplitude
  error and the wrong spectral peak. (The amplitude loss actively prefers this to
  a sharp but slightly mis-timed waveform, as the time-shift row shows:
  `MSE 2.56` for a 1 s displacement whose spectrum is intact.)

In other words: **the span-masked task's residual is dominated by spectral
error**, which is precisely the component an amplitude-only objective
under-penalises. That is the sense in which the STFT "makes sense" here — and it
is a *conditional* statement: it holds **only after** the mask removes the
interpolation shortcut. Before that, the term is not merely unhelpful, it is
misleading (it certifies a solution that ignores the video).

### 5.3 The boundary-mixing problem shrinks

Under scattered masking the assembled clip is a mosaic of 25 exact and 25
synthesised tokens, so every Hann-windowed spectrum sees a pattern of artificial
steps; the term largely measures *mask-boundary continuity* rather than
physiology. With `n_spans 1-2` there are **1-2 boundaries** instead of ~25, and
the windowed magnitude inside the gap is dominated by the reconstructed
waveform itself. Note the assembled signal is still a hybrid (visible halves are
exact copies), so the term remains a *consistency* term as well — see §5.6 for
how to sharpen it.

### 5.4 Window sizing must follow the span

The FFT windows must be **shorter than the gap** (otherwise a window contains no
information the model must invent) and **long enough to contain one target
period** (otherwise the magnitude loss measures local shape, not rate):

| `fft_sizes` | durations at `fs 100` | one respiration period (2.5-6.25 s)? |
| ----------- | --------------------- | ------------------------------------ |
| 64, 128, 256 (shipped) | 0.64 / 1.28 / 2.56 s | no; only the 256 window approaches it |
| **128, 256, 512** | 1.28 / 2.56 / 5.12 s | the 512 window does; `512 <= L 800` fits |

Measured, the two sets are **not comparable in absolute terms**, and the honest
reading is that switching is mostly a *scale shift*: for the same predictor the
longer windows report a *lower* value (1-span interpolator `2.20` vs `2.37`;
assembled constant `4.10` vs `4.67`), while the *relative* separation between the
bad and better solutions is essentially unchanged (`2.20/4.10 = 0.54` vs
`2.37/4.67 = 0.51`). So the change buys nothing measurable, and the reason to
prefer longer windows is physiological, not empirical: a 5.12 s window can
contain a full 0.2 Hz breath, a 0.64 s window cannot contain any breath.

> **Rule:** `spec_resp` is only comparable *within one `fft_sizes` setting* (the
> term averages over windows of different lengths and counts). Any baseline or
> acceptance threshold in this document is quoted for the shipped
> `64,128,256`; re-measure before comparing if you change it.

### 5.5 What the spectral term still cannot do

* **Polarity and time displacement are invisible to a magnitude spectrum**
  (§3: sign flip `spec 0.0`, 1 s shift `spec 2.02`). It must stay an *auxiliary*
  term next to the masked MSE.
* **It cannot create a gradient path where none exists.** With the physio stream
  fully masked (`k = 0`) the decoder's input for that stream is
  `mask_token + pos_embed` only — no path to the encoder — so a spectral term
  then trains the *decoder*, which Stage 3 discards. The "≥1 visible token"
  clamp is what preserves the encoder path, and it is also why the plan's
  "ramp to 1.0" must be treated as a **Stage-3-aligned variant**, not as the
  default (measured earlier: all-physio-masked + MSE weight 0 ⇒ encoder gradient
  exactly `0.0`).
* **Its log-magnitude term is unbounded** (`log(P + 1e-6)`), which is what
  produced `spec_resp 3.7e8` under the scattered mask. If it is re-enabled, use
  one of: a larger floor (`1e-3`), a clamp on the log term, or the
  spectral-convergence term only (`||P-T|| / ||T||`, which is bounded and already
  in the implementation).

### 5.6 Refinements worth building *before* trusting the term

1. **Mask-weighted STFT (recommended first step).** Today the STFT is taken on
   the whole assembled clip, including the exact visible halves. Restrict it to
   windows that sit inside the masked span (or weight each window by its masked
   fraction). Then the term becomes a genuine "reconstruct the spectrum of what
   is missing" objective instead of a partially-satisfied consistency prior —
   and its gradient no longer comes mostly from copying visible content.
   Implementation: compute the sample-level mask from `mask_s`, build a
   per-window weight from the STFT framing, and scale `sc`/`lm` per window
   before the mean.
2. **Weight ramp** — turn `spectral_weight` on only after the MSE has passed the
   no-video floor (§7), and ramp it (e.g. 0 -> 0.05 over the first N epochs).
3. **Keep `clip_grad` on** and monitor `spec_resp`'s **epoch maximum**, not its
   mean, so a spike is visible (the printed `grad_norm` is pre-clip).
4. **Ablate the term's content**: `sc` only (bounded, scale-invariant) vs
   `sc + lm` (current). This isolates whether the instability comes from the
   log-magnitude term.

## 6. Experiment ladder

| # | change | config | question it answers |
| - | ------ | ------ | ------------------- |
| E0 | *(baseline, already run)* scattered 0.5, no STFT | shipped config | the reference point: `mse_resp 1.05`, `spec ~5.0` — worse than a constant |
| E1 | `physio_mask: span`, `mask_span_s {resp: 1.5}`, no STFT | + 3 keys | does a 1.5 s gap (0.6 breath) already break the shortcut? Target: `mse_resp < 0.84` (the 4x1 s floor) |
| E2 | as E1 but `mask_span_s {resp: 2.0}` -> 2 s gaps | 1 key | target `mse_resp < 1.37` and **`< 0.98`** to beat the constant |
| E3 | as E2 but `mask_span_s {resp: 4.0}` (needs `ratio 0.5` @ 8 s -> `n_spans` derives to 1) | 1 key | the hardest single-gap case: target `mse_resp < 1.46` **and** `< 0.98` |
| E4 | E3 + `target_norm: token` | 1 key | the position-wise-mean shortcut, now unblocked |
| E5 | E3 + `spectral_weight 0.05` (fft 128/256/512) | 2 keys | does the spectral term add anything **now**? |
| E6 | E5 + mask-weighted STFT (§5.6.1) | code | the "proper" version of the term |
| E7 | a multi-stream run (`rgb,tir,bp,resp`) with the per-modality table of §4.1 | 3 keys | does the per-modality span table work when several physio streams share the grid, and how big is the cross-physio leak of §4.6? |

## 7. Acceptance criteria and falsification

**A span-masked Stage 2 is working when, on a fixed held-out clip set:**

1. `mse_resp` beats **both** no-video floors of its own pattern — i.e. below the
   interpolation baseline (4 s span: `1.46`) **and** below the constant
   (`0.98`). If it cannot beat a constant under a 4 s gap, the encoder is not
   using the video and nothing downstream can be trusted.
2. `spec_resp` falls below **2.0** (the span interpolator's spectral level at the
   shipped `fft_sizes 64,128,256`) — this is the number that tells you the
   reconstruction has the right *content*, not just the right amplitude.
3. The **encoder gate** improves: pooled `h` from the unmasked encoder varies
   across clips by more than the previously measured collapsed value
   (`ImplementationPlan.md` §4; the VideoMAE-init reference and the failing
   anchor are recorded there). This is the criterion that actually predicts
   Stage-3 usefulness.
4. Only then is a Stage-3 thermal->resp fine-tune worth launching.

**Falsification of the spectral term (E4 vs E2):** if adding `spectral_weight`
at a small weight (a) does not lower `spec_resp` below the E2 value, (b) does not
improve the encoder gate, and (c) costs stability (any epoch whose mean `loss`
is >2x its last-step value), then the term is not earning its place and should be
dropped — the honest conclusion being "the amplitude objective plus span masking
already determines the spectrum".

**What would falsify span masking itself (E1/E2):** if the model reaches the
span-interpolator floors *and no better* (i.e. `mse_resp` ~1.4 on E2), that means
it has learned to produce a chord across the gap — a degenerate solution that
the *interpolation* baseline already characterises, and the next lever is then
not the loss but the **video side**: whether the TIR ROI (mouth/nose, 64 px,
mask 0.9) carries enough information to infer respiration at all. The measured
pipeline fact to check first is the Stage-3-era probe that found the pulse is
present in the RGB green channel (`ImplementationPlan.md` §4.7): the analogous
question for thermal/respiration is whether the ROI's temporal intensity tracks
the breathing cycle.

## 8. Provenance of the measurements

All numbers from the shipped TIR-ROI + RESP geometry
(`configs/pretrain/stage2_local_tir_roi_resp.yaml`, 8 s clips, `clip_stride 4.0`,
`temporal_stride 2`, `fs 100`, `input_size 64`, 388-clip local corpus), run from
`code/` with the project venv, 2026-09-26:

```bash
python /tmp/probe_mask_solvability.py    # §1: no-video floor, scattered mask
python /tmp/probe_span_baselines.py      # §4.2: scattered vs 1/2/4 spans
python /tmp/probe_loss_sensitivity.py    # §3: what MSE vs MR-STFT see
python /tmp/probe_stft_grad.py           # TirROI_Resp_plan.md §9.1: gradient split
```

These are throwaway scripts; if the span work goes ahead they should be promoted
to a permanent `runners/run_probe_masking_baselines.py` so the floors are
re-derivable after any geometry change (**the floors are geometry-specific**:
they depend on `N`, `sig_kernel`, `fs` and the clip length — re-measure them
whenever the geometry changes).

---

## Appendix A — glossary of the numbers used above

| symbol | value (this geometry) | meaning |
| ------ | --------------------- | ------- |
| `N` | 50 | resp tokens per clip |
| `sig_kernel` | 16 | samples per token (160 ms at `fs 100`) |
| `L` | 800 | resp samples per clip (8 s) |
| `mse_resp` | — | per-modality MASKED MSE (masked positions only), reported raw by the model |
| `spec_resp` | — | MR-STFT on the assembled clip (weighted by `spectral_weight` in the loss) |
| const floor | `mse 1.0` | a per-clip z-scored target scored by a constant predictor |
| interpolation floor | pattern-dependent | achievable with the visible resp samples only, zero video |
