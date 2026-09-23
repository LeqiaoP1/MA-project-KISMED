Additional Plan — Simplified RGB-only Configuration
===================================================

> **Status: DECIDED (2026-09-16).** This file is the authoritative record of the
> simplified configuration for the thesis runs. Where it disagrees with
> `ImplementationPlan.md`, **this file wins**; §12 lists exactly what it
> supersedes there. Both files are kept so the 4-stream (RGB+TIR) design remains
> on record as the superseded alternative.

---

### **1. Decisions of record (2026-09-16)**

| # | decision | value |
|---|---|---|
| 1 | dataset | **full BP4D+** (~140 subjects x 8 tasks ~= 1120 sessions, ~12-13 h of video) |
| 2 | initialization | **space-time weights, patch 16x16** -> load a **VideoMAE-B** checkpoint (see §7 for why the source matters) |
| 3 | Stage 2 | **SSL representation learning; encoder input = RGB video only**; decoder reconstructs **BP and RESP** |
| 4 | Stage 3 | **fine-tuning for BP and RESP waveform construction** — two branches |
| 5 | evaluation | **session-level assembly** of the Stage-3 clip predictions, then clinical metrics |
| 6 | TIR | **dropped from scope** (no TIR stream in Stage 2; TIR cells removed, §12) |
| 7 | RGB mask ratio | **0.90 default, 0.95 as an ablation** (both physically motivated, §4) |
| 8 | `temporal_stride` | **2** (effective 12.5 fps) |

Rationale for 6: an RGB-only Stage-2 encoder leaves `adapters.tir` untrained, so a
TIR Stage-3 cell would load a random adapter. The physically grounded cell set is
therefore reduced from three (`RGB->BP`, `RGB->RESP`, `TIR->RESP`) to two.

---

### **2. Geometry (locked)**

| parameter | value | note |
|---|---|---|
| `clip_duration` | **8.0 s** | >=2 respiratory cycles at 15 bpm (period ~4 s); ~8-10 cardiac cycles |
| `fps` | **25.0** | **source** rate — the dataset decimates. Do NOT set 12.5 here |
| `temporal_stride` | **2** | effective 12.5 fps; `num_frames = round(8.0*25/2) = 100` |
| `tubelet` | **(2,16,16)** | `G_t = 100/2 = 50` |
| `input_size` | **224** | 224/16 = 14 -> **196 patches/frame** |
| `fs` | **100.0** | |
| `sig_kernel` | **16** | forced by the alignment contract (below) |
| `seq_len` | **800** (`seq_len: 0` auto-derives) | `n_signal = 800/16 = 50` |
| `pos_init` | `sincos3d` | the only source of temporal structure at init |
| token span | **160 ms** for video tubelets **and** 1-D tokens | |
| `n_visual` | **9 800** = `G_t * 196` | per visual stream |
| `n_signal` | **50** | per physio stream |

**Alignment arithmetic** (`core/multimae.MultiModalMAE._check_space_time_alignment`):

```
sec_per_visual_token = tubelet_t * temporal_stride / fps = 2 * 2 / 25 = 0.160 s
sec_per_signal_token = sig_kernel / fs                   = 16 / 100  = 0.160 s   [equal ✓]
G_t = num_frames / tubelet_t                             = 100 / 2   = 50
n_signal = seq_len / sig_kernel                          = 800 / 16  = 50        [equal ✓]
num_frames % tubelet_t = 100 % 2 = 0 ; seq_len % sig_kernel = 800 % 16 = 0        [both ✓]
```

Leaving `sig_kernel` at its default 8 makes the model **raise at construction**
(80 ms != 160 ms). That is the guard working, not a bug. `sig_kernel: 16` must be
set explicitly in every config (the argparse default is 8).

---

### **3. Streams, and what the encoder actually sees**

* Stage-2 `streams`: **`rgb,bp,resp`** (3) — satisfies the contract (>=1 video AND >=1 1-D physio).
* Stage-3 `streams`: **`rgb`** (visual only — the sensor-failure protocol).

The encoder receives **only the visible (unmasked) tokens**. Masked tokens exist
only in the decoder, and the 1-D streams contribute **no encoder tokens** once
their mask ratio reaches 1.0:

| stream | total tokens | mask policy | masked | visible -> encoder |
|---|---|---|---|---|
| `rgb` | 9 800 | tube **0.90** | 176 patches x 50 = 8 800 | **20 x 50 = 1 000** |
| `rgb` | 9 800 | tube 0.95 (ablation) | 186 x 50 = 9 300 | 10 x 50 = 500 |
| `bp` | 50 | contiguous span, ratio 0.9 -> 1.0 | 45 -> 50 | 5 -> **0** |
| `resp` | 50 | contiguous span, ratio 0.9 -> 1.0 | 45 -> 50 | 5 -> **0** |

So a typical step is **1 000 RGB tokens + ~10 physio tokens**, and at the end of
the ramp the encoder is **RGB-only**. Description worth reusing: *the encoder sees
20 spatial traces sampled at 12.5 fps over 8 s* (20 patches/frame, identical
patches at every one of the 50 time steps, because the mask is a **tube**).

The 1-D signals are always the **reconstruction target** and (during the ramp)
visible **decoder-side** tokens; they are never encoder inputs past the ramp. The
physio loss still shapes the encoder through the cross-attention decoder, whose
queries are the masked `bp`/`resp` tokens.

**Visual-stream dropout is inactive** in this configuration (there is only one
visual stream, and the contract forbids dropping it). That is expected, not an
oversight — the sensor-failure simulation is supplied entirely by the 1-D side.

---

### **4. Masking policy (Stage 2)**

* **Video: tube masking at 0.90 (default).** One random spatial subset per
  clip-sample, replicated across all `G_t` steps (`_tube_mask`) — deliberately
  leak-free: a masked tube is unseen at *every* time step, so the "copy from the
  neighbouring frame" solution is impossible.
* **Why 0.90-0.95 is defensible** (VideoMAE ablation, ViT-B, 16-frame, 800 epochs):
  tube 75 % -> 68.0 / 79.8, **tube 90 % -> 69.6 / 80.0**, random 90 % -> 68.3 / 79.5,
  frame 87.5 % -> 61.5 / 76.5 (SSv2 / K400 top-1). Images prefer 75 % (MAE);
  **video prefers 90-95 %** because temporal redundancy otherwise makes the task
  too easy (shortcut features). Their finding: "even 95 % can achieve good
  performance".
* **0.90 vs 0.95 is a *signal* decision, not a compute one** (both cost <= 2.4 h per
  800 epochs, §5). Visible budget at 0.90 is 20 patches/frame (good spatial
  coverage of the face); at 0.95 only 10 (~5 % of frame area), which weakens the
  effective spatial averaging of the pulse and risks capacity spent on
  conditional-mean inpainting of the masked appearance.
  **Decide with the downstream Stage-3 metrics + the video-dependence probe — NOT
  with the reconstruction loss** (which looks better at 0.95 simply because there
  is less to reconstruct, amplified by per-token normalisation).
* **Physio: contiguous span masking, ramping 0.9 -> 1.0.** Scattered independent
  dropout (`_random_mask`) leaves the target locally recoverable from the signal's
  own low-dimensional dynamics (global phase/frequency fitting; near-linear
  interpolation for RESP). Use spans of >=1.5-2 s (`n_spans` 1-2) — at 160 ms per
  token that is 10-13 tokens per span.
* **Two code constraints** (both currently violated by the helpers, §11):
  `max(1, min(N-1, int(ratio*N)))` forbids ratio 1.0; and the ratio must be drawn
  **once per step/batch** because `k = int((mask_s == 0).sum(dim=1).max())` is a
  batch max (a per-sample visible count would leak masked tokens into the encoder).
* **Augmentation must preserve colour** (spatial crop/flip, temporal shift); no
  colour jitter — rPPG lives in the chromatic change. MAE's own ablation shows
  colour jitter *degrades* results.

---

### **5. Compute and I/O budget**

Assumptions: ViT-B (D=768, 12 layers), B=1, ~39 k clips/epoch at `clip_stride 1 s`,
8x A100 at ~40 % MFU (~1 PFLOP/s effective). Per-sample FLOPs = 3 x forward.

| run | encoder tokens | TFLOP/sample | per epoch | 800 / 100 epochs |
|---|---|---|---|---|
| **Stage 2, stride 2, mask 0.90** | **1 000** | **0.28** | ~11 s | **~2.4 h** (800) |
| Stage 2, stride 2, mask 0.95 | 500 | 0.11 | ~4.4 s | ~1 h (800) |
| Stage 2, stride 1, mask 0.90 (old) | 2 000 | 0.78 | ~30 s | ~7 h (800) |
| Stage 2, stride 1, mask 0.75 (old) | 4 900 | 3.5 | ~137 s | ~30 h (800) |
| **Stage 3, unmasked** | **9 800** | **12.3** | ~8 min | **~13 h** (100) per branch |
| Stage 3, tube mask 0.90 | 1 000 | 0.28 | ~11 s | ~20 min (100) per branch |
| Stage 3, evaluation at ratio 0 | 9 800 | 12.3 (fwd only) | — | ~1 min (whole eval set) |

**I/O.** `temporal_stride` decimates on the index grid **before** decode, so stride 2
halves the decoded frame count:

| | frames/clip | epoch decode (39 k clips, single-core) | `.npy` store at 224 px |
|---|---|---|---|
| stride 1 | 200 | ~20 h | ~178 GB |
| **stride 2** | **100** | **~10 h** (~9.5 min at 64 workers) | **~89 GB** |

Consequences:
* Stage-2 **compute is now cheap**; the **RGB JPEG decode is the sole Stage-2
  bottleneck**. The materialised uint8 `.npy` store (page-cache resident, shared
  across workers/ranks) is the **first prerequisite**, before any long run.
* **Stage-3 unmasked becomes the next cost centre** (~13 h/branch). Adopt the
  Stage-3 masking mixture (§8) — it is both the regulariser and the main remaining
  compute lever. `F.scaled_dot_product_attention` is **mandatory** at 9 800 tokens
  (naive `q @ k^T` materialises 384 M scores/head/layer).
* The mask ratio does **not** change the loader (all frames are decoded before
  masking).

---

### **6. Sampling artefacts to be aware of**

1. **12.5 fps is comfortably above the BP Nyquist** (1.0-2.5 Hz needs > 5 Hz) and
   trivial for RESP — but `alignment.frame_indices_at_target_rate` **point-samples**
   (`floor(t*fps_src)`), it does not low-pass first. Source content above 6.25 Hz
   (compression noise, luminance flicker, tremor) can therefore **alias into the
   pulse band**.
2. The **anti-aliased alternative is `tubelet: 4,16,16` with `temporal_stride: 1`** —
   four consecutive frames averaged into the same 160 ms token, **identical token
   geometry and cost**. It is rejected here only because a `(4,16,16)` Conv3d
   **breaks the exact VideoMAE `(2,16,16)` patch-embed transfer**. Keep it on
   record as a free A/B if aliasing ever looks like a problem.
3. **BP morphology granularity.** 160 ms video tokens span the pulse fundamental
   and its low harmonics; the very high-frequency dicrotic-notch band is
   attenuated. HR and phase are unaffected — state this if waveform *morphology*
   fidelity is claimed.
4. **1-D granularity coarsens**: 50 tokens of 160 ms (was 100 of 80 ms), so
   `heads.<signal> = Linear(D, 16)`. The assembled output is still 100 Hz, but a
   per-token normalised target would span 0.16-0.4 of a BP cycle — which is why
   the per-clip/per-session z-score change (§9) matters *more* at stride 2, not less.

---

### **7. Initialization, and the checkpoint lock**

**Source matters at patch 16.** With `tubelet (2,16,16)` a **VideoMAE-B** checkpoint
has `patch_embed.proj = Conv3d(3, 768, (2,16,16))` — the **exact** shape — so the
tokenizer weights load **verbatim** instead of being boxcar-inflated from a 2-D
filter. This is what makes "space-time initialization" true for the *tokenizer* and
not just the blocks. MAE/ImageNet ViT-B remains the fallback (blocks + `norm` +
inflated patch embed). Requires the loader fix in §11.8.

**What is and is not inherited** (`core/multimae.load_pretrained_encoder`):

| inherited | not inherited |
|---|---|
| `enc_blocks.*` (12 blocks) | all `positions.*` (incl. the time basis) |
| `enc_norm.*` | `adapters.bp`/`adapters.resp`, the decoder, all `heads.*` |
| `adapters.rgb.patch_embed.*` — **verbatim for VideoMAE** (3-D tubelet kernel included: NOT motion-blind at init); boxcar-inflated from a 2-D MAE filter otherwise | the TIR adapter (still random) |

**Stride 2 is a project-wide lock.** `positions.<s>` is `[1, G_t*Gh*Gw, D]` =
`[1, 9800, 768]` for video and `[1, 50, D]` for physio — different shapes from the
stride-1 values, so **stride-1 and stride-2 checkpoints are NOT interchangeable**.
Set it identically in `configs/pretrain/*`, `configs/finetune/{bp,resp}.yaml`, the
**AU-probe configs**, and local smoke configs. Existing local checkpoints and the
AU-probe C2 control are stride 1 and become a **separate lineage** (the AU probe's
C2 control needs a stride-2 Stage-2 checkpoint).

---

### **8. Stage 3 — two branches**

* **Two runs** (`bp`, `resp`), `streams: rgb`, output `[B, 800]` = one 8 s clip.
  Branching avoids gradient interference between BP (1.0-2.5 Hz) and RESP
  (0.16-0.4 Hz).
* **Matrix = 2 cells**: `rgb -> bp`, `rgb -> resp` (the TIR cells are out, §12).
* **Visual masking at Stage 3 is augmentation, not task** — and now also the main
  compute lever:
  * sample the ratio **per step** from a mixture centred on the Stage-2 regime and
    **including 0**, e.g. `{0, 0.75, 0.90, 0.95}`; **tube** type (not per-frame
    random, which destroys temporal coherence of the visible set);
  * **evaluate the headline at ratio 0** (dense video = the deployment condition).
    VideoMAE pre-trains at 90-95 % tube and fine-tunes on dense clips, so this
    pairing is proven;
  * an optional short ratio-0 polish at the end gives the best headline numbers.
* **Loss** (unchanged): `L_joint = alpha*L1 + beta*(-Pearson) + gamma*MR-STFT`
  (MR-STFT windows 64/128/256 at fs = 100 Hz). Note at `sig_kernel = 16` each
  output segment is 16 samples, so the MR-STFT windows span multiple segments.
* **Head transfer:** `heads.<signal>` is `Linear(D, 16)` and `waveform_head` is
  `Linear(D, output_len // G_t)` = `Linear(D, 800/50 = 16)` — **identical shape**.
  Extend `load_stage2_encoder()` (which currently skips `heads.*`) to map
  `heads.<signal> -> waveform_head`.
* **Unmasked Stage-3 cost** is ~13 h/branch for 100 epochs; with the mixture it is
  ~20 min/branch.

---

### **9. Session-level assembly and evaluation**

Stage 3 emits **per-clip** waveforms; the current Tier-1/2 path macro-averages over
clips, and Tier-3 already expects **one continuous 1-D signal**. The assembly
module does not exist yet. Requirements:

* **Scale consistency is the real issue, not the seams.** With a **per-clip**
  z-score target each prediction is defined only up to a *per-clip* affine map.
  Naive concatenation injects a step at every boundary (broadband energy at the
  clip rate -> corrupts session-level Welch/MR-STFT) and destroys **between-clip**
  amplitude information (RIAV, baseline wander, tonic trend).
  **Fix: train with a per-session z-score** (a constant per training session, known
  from the label). Fallback: per-clip affine calibration over the overlap,
  `min_{a,b} ||a*y_k - y_{k+1}||^2`, before cross-fading. Only *consistency* across
  clips matters — Pearson r, PSD shape and RR intervals are affine-invariant
  within a session.
* **Hann overlap-add with window-sum normalisation.** Do NOT crop `D - stride` per
  clip; with `D = 8 s`, `clip_stride = 1 s` each sample is covered by ~8 windows, so
  the window weighting handles the low-context edges.
* **Time base**: stitch on the dataset's common grid (decimated by
  `temporal_stride`), mapping `t_start` -> `offset = round(t_start * fs / temporal_stride)`.
* **Plumbing**: `PairedSessionDataset.entries[idx] = (session_meta, t_start)`, but the
  loader yields only `(samples, target)` and `evaluate_waveforms` reads `batch[:2]`.
  Either walk `ds.entries` directly (the pattern `run_inspect_data` uses) or extend
  the collate to return the dataset index. Build the session **reference** from
  `_sig_all[session][name]` on the same grid and normalise prediction *and*
  reference with **one** per-session affine map.
* **Two-level reporting, explicitly labelled:** (i) *per-clip* Tier-1/2
  macro-averaged (the training/validation signal); (ii) *session-level post-stitch*
  Tier-1/2 on the stitched waveform **plus Tier-3**. Only (ii) supports the
  "restore the complete waveform" claim.
* **Tier-3 guards.** `extract_hrv_metrics`'s `len >= 2*fs` is far too loose:
  RMSSD/pNN50/MedianNN are interval statistics and an 8 s clip yields 7-9 RR
  intervals. Require **>=30 s** (NaN + warn below), compute on fixed **30-60 s
  sub-windows** then aggregate (median/IQR), and always report the number of
  detected RR intervals plus the **fraction of windows where peak detection
  succeeded**.
* **Corpus limit to state:** BP4D+ sessions are 21.9-64.6 s, so most sessions yield
  a single such window and none yield clinical-grade HRV. Frame Tier-3 as
  short-term HRV **proxies** on the longest sessions, and give RESP its own offline
  spectral/rate measures rather than HRV language.

---

### **10. Config skeletons**

Marked `[new]` = needs the code change in §11; everything else exists today.

**`configs/pretrain/stage2_rgb_bp_resp.yaml`**

```yaml
model: project_multimae_base        # 768-d / 12-layer / 12-head
streams: rgb,bp,resp               # 1 visual + 2 physio (contract: >=1 video AND >=1 physio)
input_size: 224
tubelet: 2,16,16                    # G_t = 50
clip_duration: 8.0
fps: 25.0                           # SOURCE rate
temporal_stride: 2                  # [new] effective 12.5 fps
fs: 100.0
sig_kernel: 16                      # REQUIRED at stride 2
seq_len: 0                          # -> 800
clip_stride: 1.0
pos_init: sincos3d
mask_ratio_rgb: 0.90                # 0.95 = ablation
mask_ratio_bp: 0.90                # [new] span masking, ramp 0.9 -> 1.0
mask_ratio_resp: 0.90               # [new] span masking, ramp 0.9 -> 1.0
signal_weight: 0.5                  # lambda_rgb = 1.0; physio = 0.5
loss_weights: ''                    # or e.g. 1.0,0.5,0.5
pretrained_encoder: <videomae-b .pth>   # [new] 3-D source support (§11.8)
target_norm: session                # [new] per-session z-score (§9)
mask_span_s: 2.0                    # [new] contiguous span length in seconds
epochs: 800
num_workers: 16
```

**`configs/finetune/bp.yaml` / `resp.yaml`**

```yaml
model: project_multimae_base
streams: rgb
input_size: 224
tubelet: 2,16,16
clip_duration: 8.0
fps: 25.0
temporal_stride: 2
fs: 100.0
sig_kernel: 16
seq_len: 0                          # -> output_len 800
target: bp                         # resp in the other run
signal_norm: zscore                 # [new] per-SESSION, not per-clip
finetune: <stage2 ckpt>             # heads.<target> -> waveform_head transfer [new]
train_mask_ratios: 0,0.75,0.90,0.95 # [new] Stage-3 augmentation mixture
eval_mask_ratio: 0.0                # headline numbers on dense video
```

---

### **11. Code changes required (checklist)**

| # | change | where | why |
|---|---|---|---|
| 1 | materialised uint8 `.npy` RGB store (~89 GB at 224 px on the stride-2 grid) | `data/video_io.py` + loader | **prerequisite** — JPEG decode is the sole Stage-2 bottleneck (~10 h/epoch single-core) |
| 2 | `F.scaled_dot_product_attention` instead of the materialised `q @ k^T` | `core/blocks.py::Attention` | **mandatory** at 9 800 tokens (Stage 3); ~2 lines, torch >= 2.1 already pinned |
| 3 | cross-attention (query-based) decoder | `core/multimae.py` | required for 224 px; also removes `mask_token`/`ids_shuffle`/`ids_restore` and makes the 1-D head match the Stage-3 head |
| 4 | allow mask ratio 1.0 (relax `min(N-1, ...)`) and draw the ratio **once per step** | `core/multimae.py::_tube_mask`/`_random_mask`, `make_masks` | the 1-D ramp to 1.0 and the batch-max `k` invariant |
| 5 | contiguous **span** masking for the 1-D streams | `core/multimae.py` (new `_span_mask`) | scattered dropout is locally solvable |
| 6 | per-**session** z-score target (instead of per-token `_targets()` normalisation) | `data/paired_dataset.py::_signal_at`, `core/multimae.py::_targets` | objective alignment with Stage 3 **and** the stitching requirement (§9) |
| 7 | zero-init modality embeddings + `no_weight_decay()` over `positions.*`/`modality_embeds.*` | `core/multimae.py` | `bp`/`resp` are shape- and init-identical; `positions.*` currently sits inside the weight-decay group |
| 8 | ~~accept a 3-D (Conv3d) `patch_embed` source; wire `inflate_rgb_patch` to a CLI flag~~ **DONE 2026-09-17**: 3-D sources copied verbatim (`fit_visual_patch_embed`), HF `transformers` VideoMAE layout mapped (`canonicalise_vit_state_dict`), tubelet mismatch raises, `--inflate_rgb_patch` wired into run_pretrain/run_au_probe, `videomae:base`/`videomae:large` are the new `DEFAULT_SOURCE` | `core/multimae.py::load_pretrained_encoder`, `models/pretrained.py` | a VideoMAE checkpoint used to be silently skipped under `shape_mismatch` |
| 9 | xavier-init the tubelet/signal convs (MultiMAE convention) | `core/multimae.py::_init_weights` | convs currently keep PyTorch defaults; matters most from scratch |
| 10 | Stage-3 masking support (relax the `n_visual` assertion, gather visible tokens, mean over visible patches only) | `core/waveform_model.py` | the Stage-3 masking mixture (§8) |
| 11 | `heads.<signal> -> waveform_head` transfer | `core/waveform_model.py::load_stage2_encoder` | identical shapes; avoids a random Stage-3 head |
| 12 | session-level assembly module + session ids through inference | new (`evaluation/assemble.py`?) + evaluator | §9 |
| 13 | Tier-3 guard >= 30 s, RR count + peak-detection success rate | `evaluation/clinical.py` | §9 |
| 14 | optional: `target_norm: session` also for the AU probe's Stage-2 checkpoint (stride-2 lineage) | configs | §7 |

**No longer needed in this configuration** (they were TIR-only blockers in
`ImplementationPlan.md` §4.7): forwarding `tir_channels` in
`build_pretraining_dataset`, and replacing the eager all-session TIR RAM cache
(`use_tir=False` already skips TIR entirely).

---

### **12. What this supersedes in `ImplementationPlan.md`**

| ImplementationPlan section | change |
|---|---|
| §1.2 | drop the `TIR -> RESP` row; the difficulty-ordering claim reduces to 2 cells |
| §1.3 | add `TIR -> RESP` to out-of-scope, reason: RGB-only Stage 2 |
| §2.1 | **VideoMAE-B** becomes the primary source (exact `(2,16,16)` transfer); MAE = fallback |
| §2.2 | streams -> `rgb,bp,resp`; patch 16; **stride 2 / sig_kernel 16**; mask **0.90 default, 0.95 ablation**; new cost table; remove the visual-stream-dropout bullet (inactive) |
| §2.3 | matrix -> 2 runs (`rgb -> bp`, `rgb -> resp`); add the Stage-3 masking mixture + "evaluate at ratio 0" |
| §4.5 | drop the TIR nostril-ROI probe; the RGB rPPG-SNR probe becomes the critical one |
| §4.6 | remove the TIR-cache discussion; the frame store is the sole Stage-2 bottleneck |
| §4.7 | close blockers 1-2 (TIR channels, TIR RAM cache); keep the rest (SDPA, decoder, mask clamps, weight decay) |
| §5 (AU probe) | unchanged in method, but its C2 control needs a **stride-2** Stage-2 checkpoint (§7) |

Everything else in `ImplementationPlan.md` — the physical-basis evidence, the
Stage-1 analysis, the A/B/C "is Stage 2 necessary" question, the evaluation tiers,
the blockers that remain — still applies.

---

### **13. Open decisions and risks**

1. **0.90 vs 0.95** — run both (2.4 h vs 1 h per 800 epochs) and decide on
   Stage-3 subject-disjoint metrics + the video-dependence probe, not on the
   reconstruction loss (§4).
2. **Is Stage 2 necessary at all?** The simplified config makes option **B
   (collapse 2+3)** more attractive than before: RGB-only + video (dense or masked)
   + 2 waveform heads is a legitimate single-stage design, and the contact
   waveform is the label for every session. Run the decisive experiment
   (`S3`-only vs short `S2`->`S3` vs long `S2`->`S3` at equal total wall-clock).
3. **Stage-3 unmasked cost** (~13 h/branch) — resolved by the masking mixture, but
   only if change #10 lands before the Stage-3 runs.
4. **Stitching scale convention** — per-session z-score is the recommendation; if
   per-clip training is kept, the affine calibration fallback must be implemented
   and validated on a session with a known-good reference.
5. **HRV claim strength** — 21.9-64.6 s sessions cap Tier-3 at short-term proxies;
   decide how the thesis words this before running the evaluation.
6. **Aliasing** — if the video-dependence probe or the spectral metrics show
   unexplained high-frequency energy, test the anti-aliased `tubelet_t: 4` variant
   (§6.2), at the cost of the VideoMAE patch-embed transfer.
