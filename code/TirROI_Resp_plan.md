# Stage 2 with **TIR-ROI** (visual) + **RESP** (1-D) — procedure plan

Status: **implementation + local verification DONE 2026-09-26** (see §7 for the
recorded results). This file is the procedure of record for this experiment: it
states what is fed to Stage 2, how the geometry is kept consistent with the
existing runs, what was changed, what was deliberately left untouched, and which
checks a run has to pass.

---

## 1. Goal and scope

Run **Stage-2 multimodal masked pre-training** with exactly two streams:

| stream | role | tensor per sample | source |
| ------ | ---- | ----------------- | ------ |
| `tir`  | visual (2-D) | `float32 [3, T, H, W]`, `[0, 1]` | thermal `.wmv` **cropped to the mouth+nose ROI** |
| `resp` | 1-D waveform (the future Stage-3 target) | `float32 [1, L]`, raw volts | `Resp_Volts.txt` |

This is the **thermal-only** counterpart of the `rgb,bp[,resp]` runs: the same
encoder, the same masked-MAE objective, the same loss — but the only visual
information is the thermal rendering of the respiration region (nostrils, nose
bridge, lips), and the only waveform is respiration.

**Non-goals.** No model change beyond the Stage-1 init routing in §5.3. No
change to Stage 3, no session-level assembly, no RGB/TIR fusion in this run (that
stays the `rgb,tir,...` variant), no landmark-prediction head.

**Why "ROI only" is the interesting variant.** The full thermal frame is 726x480
with a burned-in degC legend and a large background; the respiration signal
lives in a ~100x100 px region. Cropping to the landmark-derived box removes the
legend/background and makes the thermal patch comparable in size to the Stage-2
`input_size`, so the tube-masking budget is spent on the face rather than on the
scene.

---

## 2. Where the data comes from — and why not the canonical layout

This run reads the **raw tree** (`$RAW_DATA_PATH`, in-repo `data/raw/BP4D`):

```
<Thermal>/<S>/<T>.wmv                25 fps, 726x480, false-colour rendering
IRFeatures/<S>_<T>.txt               28 (x, y) landmark pairs per frame
Physiology/<S>/<T>/Resp_Volts.txt    1000 Hz, one value per line
```

The canonical (`data/processed/bp4d_canonical/<S>_<T>/`) layout used by every
other Stage-2/3 run **cannot** serve this variant, for one concrete reason:

* `PairedSessionDataset` decodes each session's TIR **once** into
  `_tir_cache` at `input_size` (`resize_center_crop` → e.g. 64x64). A
  clip-dependent ROI crop cannot be taken out of an already center-cropped
  64x64 frame, and caching the native 726x480 frames instead costs
  **1.05 MB/frame = 1.7 GB for F001_T1 alone** per dataset instance.
* The ROI is defined by `IRFeatures`, which lives only in the raw tree
  (`<S>_<T>.txt`), not next to the canonical session directories.

So the TIR stream is decoded **per clip** at native resolution and cropped
(`data/video_io.py::read_range`, seek-verified frame-exact against a full
sequential decode). The 1-D stream is resampled from the raw 1000 Hz file to the
model's `fs` (100 Hz default). Everything else — the geometry contract, the
model, the loss, the masking — is shared with the existing runs.

> The canonical `tir.wmv` is byte-identical to the raw `.wmv` (md5-verified
> 2026-09, §4.5 of `ImplementationPlan.md`), so this is a *source* choice, not a
> data-quality choice.

---

## 3. Clip → ROI → tensor procedure

Reused verbatim from `data/tir_resp_dataset.py` (already verified, see
`data/tir_resp_dataset.py::main`):

1. **Enumerate clips.** Sessions = thermal videos joined with an `IRFeatures`
   file and `Resp_Volts.txt`. Windows: `start = 0, stride, 2*stride, ...` while
   `start + T <= n_frames`, over `n_common = min(video_frames, IR_lines)` and
   limited by the respiration length.
2. **Drop invalid windows.** Any `(0, 0)` landmark line inside the window
   (the corpus' undocumented missing-data sentinel: whole frames where the
   tracker lost the face) drops the **whole clip**; the window hop moves on to
   the next one. Untracked sequences without `IRFeatures` are skipped.
3. **ROI box (one per clip, static).** For the 1-indexed labels
   `[9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26]` (nose bridge, nostrils,
   mouth corners, lips, lip centres) take `x_min/x_max/y_min/y_max` over **all
   frames of the clip**, extend each side by `roi_padding * extent`
   (`roi_padding: 0.2` → +40 % overall) and clamp to the frame. All `T` frames
   are cropped with that **identical** box → no per-frame spatial jitter.
4. **Resize** each crop to `input_size x input_size` (`cv2.INTER_AREA` when
   downscaling) and scale to `[0, 1]` (`/255`). Result: `[3, T, H, W]`.
5. **Respiration.** From the clip's raw window, sample the model grid by TIME:
   `idx = arange(L)/fs * resp_fs`, `np.interp` → `[L]` raw **volts**. The
   dataset deliberately returns **raw** values: the model's `target_norm: clip`
   z-scores per clip inside the forward pass, exactly as for `bp`/`rgb` runs. Do
   not z-score twice.
6. **Decimation (optional).** `temporal_stride N > 1` keeps every N-th ROI frame
   (12.5/6.25/3.125 fps) and shortens the respiration window to the kept
   duration, mirroring `build_pretraining_model`'s own arithmetic.
   **It REQUIRES `sig_kernel = tubelet_t * N / fps * fs`** (2*2/25*100 = 16 for
   N=2): the identity below is a hard model check, and forgetting this rescale
   is the one way to break the config (the model refuses to build and says
   which sig_kernel to use). A stride change is also a NEW geometry -- the video
   token count changes (8 s: 1600 tokens at N=1 vs 800 at N=2), so the pos-embed
   shape differs and checkpoints are not interchangeable across strides. Note
   the sentinel rule is applied to ALL source frames of the window, not only to
   the decimated ones (the ROI box uses every frame), which is the conservative
   choice.

### Geometry contract (identical to the existing runs)

```
T        = round(clip_duration * fps / temporal_stride), rounded DOWN to a multiple of tubelet_t
kept     = T * temporal_stride / fps                  (dropped tail shortens the window)
L        = round(kept * fs)                           (per-clip, not clip_duration*fs)
grid_t   = T / tubelet_t                              must equal n_signal = L / sig_kernel
```

With the shipping local geometry (`clip_duration 8.0`, `clip_stride 4.0`,
`fs 100`, `tubelet 2,16,16`, `input_size 64`) at the two decimations:

```
temporal_stride 1: sig_kernel 8   T=200  grid_t=100  n_visual=1600  L=800  n_signal=100  80 ms/token
temporal_stride 2: sig_kernel 16  T=100  grid_t= 50  n_visual= 800  L=800  n_signal= 50 160 ms/token
```

(The SHIPPED config uses stride 2 -- half the visual tokens, i.e. ~4x cheaper
attention at 8 s, which is ample for respiration: 12.5 fps vs a <1 Hz signal.
`ImplementationPlan.md` §4.7's Nyquist rule bounds stride 2 for bvp and stride 8
for resp/eda, so 2 is safe here and 4 would also be safe for a resp-only run.)

`MultiModalMAE._check_space_time_alignment()` enforces this as a hard
`ValueError`, so a mismatch fails at model build rather than silently training on
desynchronised streams.

---

## 4. Compatibility with the previous experiments (RGB, BVP)

The requirement is that `rgb,bp` / `rgb,bp,resp` / `rgb,tir,...` runs behave
**exactly** as before. What that means concretely:

| component | change | backward compatibility |
| --------- | ------ | ---------------------- |
| `data/paired_dataset.py` | **none** | untouched |
| `data/datasets.py::build_pretraining_dataset` | dispatch added on `args.data_set`; default branch unchanged | `bp4d+`/`paired`/absent → `PairedPretrainDataset` exactly as before |
| `core/multimae.py::build_pretraining_model` | **none** | geometry/loss/masking code untouched |
| `core/multimae.py::load_pretrained_encoder` | patch-embed destination is now *resolved* instead of hardcoded `adapters.rgb` | with `rgb` in the stream set the resolved destination **is** `adapters.rgb` → identical result (verified: 148 tensors loaded, as before) |
| `runners/run_pretrain.py` | new optional flags (`--data_set`, `--raw_root`, `--roi_padding`) | existing configs never set them → same defaults; `data_set` defaults to `bp4d+` |
| existing configs (`stage2_local_scratch/_pretrained`) | **none** | untouched; the new experiment is a NEW config file |
| `configs/pretrain/stage2_local_*.yaml` twin invariant | **none** | still differ in exactly `pretrained_encoder`/`inflate_rgb_patch`/`output_dir` |

The new run is therefore **additive**: a new dataset class, a new config, and one
resolution step in the Stage-1 init path.

---

## 5. Procedures

### P1 — dataset

`data/tir_resp_dataset.py::TirRoiRespPretrainDataset` (new class, same module as
the ROI primitives) wraps `BP4DPlusTIRRespDataset(norm='none')` and returns

```python
{'tir': float32 [3, T, S, S], 'resp': float32 [1, L]}
```

i.e. exactly the stream keys/ranks `MultiModalMAE` expects. `data_set: tir_roi`
(alias `tir_roi_resp`) selects it through
`data/datasets.py::build_pretraining_dataset`.

### P2 — model

No change: `streams: tir,resp` satisfies the Stage-2 modality contract (≥1 video
+ ≥1 1-D physio). `STREAM_CHANNELS['tir'] = 3` is already correct (the thermal
stream is a false-colour rendering, not gray).

### P3 — Stage-1 init routing

`--pretrained_encoder videomae:base` with a TIR-only stream set used to leave
`adapters.tir` **random** (the checkpoint's `patch_embed.proj` was routed to the
non-existent `adapters.rgb` and counted as "skipped"). `load_pretrained_encoder`
now resolves the destination to the first visual adapter that EXISTS
(`rgb` → `tir` → …), so the VideoMAE `Conv3d(3, 768, (2,16,16))` tubelet is
copied **verbatim** into `adapters.tir` — the same 3-channel, same-tubelet
geometry, i.e. this run inherits the temporal kernel as well as the 12 encoder
blocks.

```bash
# local config ships the VideoMAE init; the from-scratch A/B twin is a CLI override:
python runners/run_pretrain.py -c configs/pretrain/stage2_local_tir_roi_resp.yaml
python runners/run_pretrain.py -c configs/pretrain/stage2_local_tir_roi_resp.yaml \
    --pretrained_encoder '' --output_dir ../output/pretrain/stage2_local_tir_roi_resp_scratch
```

### P4 — masking and loss policy for this stream set

**The objective as shipped** (verified numerically: `0.343 + 0.5*1.048 = 0.867`
= the logged loss, and `losses_spectral == {}`):

```
loss = 1.0 * mse_tir + 0.5 * mse_resp          # per-modality MASKED MSE, masked positions only
```

The **STFT term is NOT in this objective**: `spectral_weight: 0.0` makes
`_spectral_streams == []` and `spectral_fn is None`, so `MultiResolutionSTFTLoss`
is never even constructed — no extra compute and no extra gradient. It is an
OPT-IN term, folded into the physio stream's contribution as
`contrib = lambda_s·MSE_s + w_s·MR-STFT_s` (so a positive weight gives
`loss = mse_tir + 0.5·mse_resp + w·spec_resp`), computed on the ASSEMBLED clip
waveform (`pred.reshape(B, -1)`; legal because `SignalEmbed` uses
kernel == stride, so the concatenated tokens ARE the clip in order). It is
defined on 1-D waveforms only (a positive weight on a video stream raises) and
requires `target_norm: clip` (it raises with token norm). It was switched off
because it was the divergence trigger — see §7. Stage 3 keeps its own
`gamma·MR-STFT` term (`WaveformJointLoss`, gamma = 1.0 in the finetune configs),
so the periodicity prior still exists downstream.

* `mask_ratio_tir: 0.90` (as shipped) — the thermal stream is now the *primary*
  visual stream, so it gets the visual budget. NOTE `0.90` on a 4x4 patch grid
  leaves only 2 of 16 patches visible per frame; `0.75` is the safer choice at
  64 px and costs nothing to try.
* `mask_ratio_resp: 0.50` — NOT 0.90. A physio stream's gradient path to the
  encoder is its **visible** tokens only (`dec = cat([z_visible, mask_tokens])`),
  so 0.90 leaves ~5 of 50 tokens and 1.0 would zero the encoder path entirely.
* `signal_weight: 0.5` per-modality masked-MSE weight for `resp`; the visual
  stream stays at 1.0. `target_norm: clip` (mandatory if the spectral term is
  ever re-enabled, and it removes the per-token z-score shortcut that lets
  the decoder solve reconstruction from position alone — the standing
  collapse hypothesis, §7).

### P5 — local smoke run

```bash
cd code
python runners/run_pretrain.py -c configs/pretrain/stage2_local_tir_roi_resp.yaml \
    --epochs 1 --warmup_epochs 0 --max_entries 2 --num_workers 0 \
    --output_dir /tmp/tir_roi_probe
```

`--warmup_epochs 0` is **required** for a smoke: the config's `warmup_epochs: 5`
with `--epochs 1` trips the pre-existing `cosine_scheduler` assertion
(`utils/lr_sched.py`, "5 != 1") before any training happens.

Expected: `Model = project_multimae_base`, the geometry block prints
`num_frames 100 / seq_len 800 / n_visual 800 / n_signal 50` (stride 2 as
shipped; stride 1 would print `200 / 800 / 1600 / 100`), the init prints the
resolved patch-embed stream (`adapters.tir`), and the epoch logs both `mse_tir`
and `mse_resp`.

### P6 — real local run

```bash
cd code && source scripts/env_local.sh
python -u runners/run_pretrain.py -c configs/pretrain/stage2_local_tir_roi_resp.yaml \
    2>&1 | tee ../output/pretrain/stage2_local_tir_roi_resp/train.log
```

`mkdir -p` the output dir first: piping into `tee` for a directory that does not
exist kills the run with SIGPIPE and no traceback. **`-u` is not optional**: piped
stdout is block-buffered, so a Ctrl-C'd run leaves a `train.log` with the config
dump and `Start training for 40 epochs` but ZERO epoch lines (this happened on
2026-09-26 and hid a divergence for 11 epochs). `PYTHONUNBUFFERED=1` works too.

### P7 — read the result before trusting it

Stage-2 loss is a **monitor, not a metric** (pretrain-on-all, no validation
split). Because `target_norm: clip` z-scores each stream per clip, a trivial
constant predictor scores ≈ **1.0 per stream**; and `clip_grad: 0.0` is a real
"off" (after the 2026-09-23 `native_scaler` fix) — do not read a flat loss as
convergence, and remember that AdamW's decoupled weight decay moves weights with
**zero** gradients.

Check, in this order:

1. `mse_tir` must fall clearly below ~1.0 (a visual stream that cannot beat a
   constant is not learning appearance);
2. `mse_resp` vs ~1.0 — this is where the previous runs failed (bp/resp ended at
   or above the trivial baseline, see `ImplementationPlan.md` §4);
3. the **encoder gate**: pooled features of the unmasked encoder must not
   collapse to an input-independent constant (the `/tmp/eval_new_run.py`
   procedure: pooled-`h` variation across clips, with the VideoMAE-init value as
   the reference);
4. only then is a Stage-3 thermal→resp fine-tune meaningful.

**Read the LAST STEP of an epoch, not the epoch average** - a single spike can
dominate an average. §7 shows what a spike looks like and what caused one.

### P8 — HPC scaling

The recipe scales by data, not by code: `max_sessions`/`max_entries: 0`,
`clip_stride: 1.0-2.0`, and `input_size: 224` — but 224 px is **not** free with
this encoder: `input_size` 224 with `tubelet 2,16,16` gives 196 patches/frame →
`n_visual = 9800` tokens, and the MAE-style attention in `core/blocks.py`
materialises the full `N*N` matrix (12 heads x 9800^2 x 2 B = 2.3 GB per batch
item). Raise resolution only together with a geometry change that still
satisfies the alignment identity (e.g. `temporal_stride 2` + `sig_kernel 16`).

---

## 6. Verification gates (all pass locally)

| # | check | how | result |
| - | ----- | --- | ------ |
| 1 | dataset contract | `TirRoiRespPretrainDataset(...)[0]` | keys `{tir, resp}`, `[3,100,64,64]` + `[1,400]`, `float32`, TIR in `[0,1]`, resp in raw volts (-4.81..0.33), DataLoader collate `[2,3,100,64,64]` + `[2,1,400]` |
| 2 | geometry identity | dataset `T`/`L` vs `build_pretraining_model` | both `T=100`, `L=800`, `n_visual=800`, `n_signal=50` at the shipped stride 2 (`sig_kernel 16`), and `T=100/L=400/n_signal=50` at stride 1 (`sig_kernel 8`); `_check_space_time_alignment` passes in both |
| 3 | forward + loss | 1-epoch smoke (`--max_entries 2`) | `loss 5.07`, `mse_tir 2.189`, `mse_resp 4.273`, `spec_resp 7.478`, `grad_norm 79.6` |
| 4 | init routing (tir) | `load_pretrained_encoder` on a `tir,resp` model | `loaded 148, skipped 0, patch_stream='tir'`; `adapters.tir.patch_embed.weight` **bit-equal** to the checkpoint `Conv3d`, and its two tubelet time-slices DIFFER (a real temporal kernel, not boxcar) |
| 5 | **no RGB regression** | same call on an `rgb,bp,resp` model | `loaded 148, skipped 0, patch_stream='rgb'`, `adapters.rgb` bit-equal — identical to before the change |
| 6 | **no RGB/BVP run regression** | `stage2_local_pretrained.yaml`, 1 epoch, 2 entries | runs as before: `mse_rgb 2.082 / mse_bp 2.683 / mse_resp 4.514 / spec_bp 3.777`, 148 tensors -> `adapters.rgb` |
| 7 | ROI + data rules | `python data/tir_resp_dataset.py --clip_seconds 4 --n_check 3` | PASS: 431 clips / 40 sessions / 0 skipped / 9 dropped; box containment, tensor==crop, resp==raw slice; synthetic untracked session skipped with 0 clips |

---

## 7. Failure modes seen in the first real run (2026-09-26)

The first real run (`clip_duration 8.0`, `clip_stride 4.0`, `batch_size 4`,
`mask_ratio_tir 0.90`, `lr 1e-3`, `clip_grad 0.0`, `spectral_weight 0.1`)
**diverged**, which the user read as "loss is high and reduces slowly".
Diagnosis, in order of discovery:

1. **The loss SCALE is expected and is not comparable to 0.** The loss
   decomposition was verified exactly: `loss = mse_tir + 0.5*mse_resp +
   0.1*spec_resp` (init: `2.152 + 0.5*3.406 + 0.1*8.595 = 4.715`). Under
   `target_norm: clip` a CONSTANT predictor scores exactly **1.0 per stream**,
   so the realistic envelope is `4.7 -> ~1.4`, not `-> 0`. 54 % of the initial
   loss is the two respiration terms, whose floor is where the documented
   `physio` collapse sits.
2. **The run really did blow up.** `checkpoint-0010` evaluated on 24 fixed
   clips: `mse_tir 11.96`, `mse_resp 556 017`, `max|pred_resp| = 1371` (all 24
   clips, so not one bad sample). At `checkpoint-0000` it was already
   unhealthy: `mse_tir 0.418` (good) but `mse_resp 6.76` (worse than the 3.41
   init). `182/182` optimizer `exp_avg` were non-zero, so gradients WERE
   flowing - this is not the old `clip_grad 0` zero-gradient bug.
3. **It was an ACTIVATION runaway, not a weight explosion.** Every parameter
   group stayed within ~2x of init (`enc_blocks` 266 -> 520, `positions.tir`
   784 -> 723, `heads.resp` 3.97 -> 2.58). The residual stream amplified instead:
   at `checkpoint-0010` the encoded `tir` tokens reach 60.7 (vs ~1-2 at init)
   and the prediction 1371. Note the target itself is ALWAYS bounded,
   `|z| <= 1` (`(x-mean)/sqrt(var+eps)`), so a masked MSE of 556 017 can only
   mean the PREDICTION exploded - that is a useful 1-line diagnostic.
4. **The MR-STFT term is the trigger.** `spec_resp` reached **3.7e8** in single
   steps with `grad_norm ~5e8` (reported pre-clip). Removing it
   (`spectral_weight 0.0`) eliminated the spikes completely: the epoch average
   tracks the last step (`0.877` vs `0.857` avg/step) and `grad_norm` is O(1-7)
   instead of 1e8.

**Recipe that is stable (measured at the same 8 s / batch-4 geometry):**

```
clip_grad:        5.0      # was 0.0 = OFF; 5-10 clips the rare resp spikes
spectral_weight:  0.0      # was 0.1 = the trigger; re-add later, small, once healthy
lr:               3.0e-4   # was 1e-3
```

2 epochs of the fixed config: epoch 0 `loss 2.69 / mse_tir 0.483 /
mse_resp 4.418`, epoch 1 `loss 0.901 / mse_tir 0.376 / mse_resp 1.052`,
`grad_norm` avg 2.91, **67 s/epoch** (385 clips, batch 4) -> 40 epochs ~ 45 min.
The last step of an epoch is the number to read; a huge epoch AVERAGE with a
sane last step still means a spike happened (the printed `grad_norm` is measured
BEFORE clipping, so it shows the spike even though the applied step was clipped).

**Still open after the fix: `mse_resp = 1.05` is exactly the constant-predictor
floor.** The visual stream learns (`mse_tir 0.38`, far below 1.0) but the
respiration stream does not beat a constant -- the same Stage-2 physio collapse
documented in `ImplementationPlan.md` §4, now reproduced in the thermal-only
setting. Turning the STFT off has a second benefit: it UNBLOCKS
`target_norm: token` (the model refuses a spectral weight with token norm), which
is the standing hypothesis for the collapse (per-token normalisation lets the
decoder reconstruct from position alone). That is the next experiment.


## 8. Implementation record

**Files added**

* `data/tir_resp_dataset.py::TirRoiRespPretrainDataset` +
  `build_tir_roi_pretrain_dataset(args)`.
* `configs/pretrain/stage2_local_tir_roi_resp.yaml` (local, VideoMAE-inherited
  init, `streams: tir,resp`).
* this document.

**Files changed (additive only)**

* `data/datasets.py` — `build_pretraining_dataset` dispatches on
  `args.data_set` (`tir_roi` / `tir_roi_resp` → the new builder).
* `core/multimae.py` — `load_pretrained_encoder` resolves the patch-embed
  destination (report included in the returned counts dict).
* `runners/run_pretrain.py` — `--data_set`, `--raw_root`, `--roi_padding`.

**Local corpus (re-inventoried 2026-09-26: the raw tree grew from 11 thermal
videos to 40 -- `F001..F004` x `T1..T10`).** Stats: **45 858 thermal frames /
30.6 min**, an `IRFeatures` file for EVERY video, `lines == video frames` on all
40 sessions, `Resp_Volts.txt` for all 40, and 3 sessions containing `(0,0)`
sentinel lines (F001_T8 112/227, F002_T7 2/1111, F003_T3 70/1852). Clip budget:

| window | hop | clips | windows dropped (sentinels) |
| ------ | --- | ----- | --------------------------- |
| 4 s | none | 431 | 9 |
| 4 s | 2 s | 843 | 17 |
| 4 s | 1 s | 1663 | 34 |
| 8 s | none | 206 | 7 |

Measured throughput at `input_size 64`, `batch_size 2`: **0.26-0.32 s/step on a
local GPU** (32 steps/epoch) -> ~843 clips / 2 = 422 steps = **~2 min/epoch**, so
the shipped 40-epoch recipe is ~1.5 h (the ROI path decodes each clip's frames on
demand; `num_workers: 4` keeps the GPU fed). These counts are per WINDOW, so a
`temporal_stride` does not change them -- it only changes `T`/`L` and the visual
token count (`sig_kernel` must be rescaled with it, see §3).

**Open items**

* `resp` here is resampled from the raw 1000 Hz file onto the model grid, which
  is *not* bit-identical to the canonical `signals.csv` column (anti-aliased
  100 Hz from `prepare_bp4d.py`). If Stage-3 evaluation must line up with the
  canonical numbers exactly, add a `resp_source: canonical` mode.
* `roi_padding` is a genuine trade-off, measured: over F001_T1's 8 clips the
  static box is 84x98..125x111 px (2.5-4.5x the single-frame mouth+nose area),
  except the last clip where the subject turns away (118x162 px, 7.5x) — the
  patch then contains eyes+nose+mouth. Tightening means `roi_padding: 0`,
  shorter clips, or a spread threshold; **not** per-frame boxes (that
  reintroduces the spatial jitter the static box exists to avoid).
* The raw respiration rails at exactly `-10.0000 V` in some sessions
  (F001 T2/T6/T7/T8). It is fed through as measured; z-scoring hides the rail
  from the loss but not from the signal.

---

## 9. Research note — does a spectral term make sense on a 50 %-masked 1-D stream?

Asked before re-enabling `spectral_weight`. **Measured answer: no** — not as a
Stage-2 lever, and the measurement also exposes why the resp stream collapses.
Both probes were run on the shipped geometry (8 s clips, stride 2, `N=50` resp
tokens of 16 samples, `sig_kernel 16`, mask 0.5) over 24 real clips.

### 9.1 Where the spectral gradient goes (one forward pass, both terms split)

`spectral_weight 0.1`, `target_norm clip`, batch 4, effective training weights:

| parameter group | grad(MSE part) | grad(MR-STFT part) | STFT/MSE |
| --------------- | -------------- | ------------------ | -------- |
| `adapters.tir` | 4.90 | 0.99 | 0.20 |
| **`enc_blocks` (the shared encoder, all Stage 3 keeps)** | **13.09** | **2.66** | **0.20** |
| `dec_blocks` (discarded at Stage 3) | 24.00 | 2.39 | 0.10 |
| `heads.tir` | 2.28 | 0.00 | 0 |
| `heads.resp` | 6.60 | 0.94 | 0.14 |
| `mask_token` | 0.81 | 0.07 | 0.08 |

So the term is **not** confined to the decoder: it supplies ~17 % of the
encoder's gradient magnitude. But look at *what* it is computed on: the
assembled waveform is 50 % exact copies of the given (visible) signal and 50 %
decoder synthesis, and the STFT window (2.56 s at the largest FFT) always
straddles both. Its residual is therefore dominated by **mask-boundary
continuity**, i.e. it rewards a smooth interpolation across the gap — a
constraint the encoder can satisfy without encoding any cross-modal information.

### 9.2 Is the masked task solvable with ZERO video information?

| predictor (uses only the VISIBLE resp samples) | `mse_resp` (masked) | `spec_resp` (MR-STFT) |
| ---------------------------------------------- | ------------------- | --------------------- |
| constant 0 — the trivial floor | 0.9989 | 5.0932 |
| **linear interpolation between the nearest visible samples** | **0.1299** | **0.9709** |
| ground truth (upper bound) | 0.0 | 0.0 |
| *for reference:* the trained model (epoch 1, same geometry) | 1.052 | ≈5.0 |

**A five-line linear interpolator — no video, no encoder, no learning — beats the
trained 102 M-parameter model by 8x on the masked MSE and 5x on the very spectral
metric the term is supposed to enforce.** The reason is structural: respiration
is band-limited (0.16-0.4 Hz, period 2.5-6.25 s) while the physio stream is
sampled at 100 Hz (`sig_kernel 16` = 0.16 s per token), i.e. oversampled
~30-60x, and `_random_mask` masks tokens **scattered individually**, never as
spans — so every masked token has visible neighbours 0.16-0.32 s away.

### 9.3 Conclusion

1. **The spectral optimum is reachable without the encoder**, so the term
   cannot force the encoder to represent periodicity — it is satisfied by the
   same within-modality interpolation shortcut that causes the collapse.
2. **The loss family is not the problem; the masking design is.** A scattered
   mask on an oversampled band-limited signal creates a reconstruction task
   whose easiest solution bypasses the video entirely.
3. **Scale pathology:** the log-magnitude L1 (floor `1e-6`) is unbounded, and at
   the weights tried it produced `spec_resp 3.7e8` / pre-clip `grad_norm 5e8`
   and destroyed the model (§7). An unbounded term whose optimum is already
   attained by interpolation is a bad trade.
4. **Precedent:** masked-autoencoder recipes (MAE, VideoMAE, MultiMAE,
   Audio-MAE) reconstruct masked patches with MSE/L1 — none add a spectral loss
   on the reconstructed waveform; Audio-MAE masks *spectrograms* yet still uses
   plain MSE. MR-STFT / spectral-convergence losses come from neural vocoders
   and speech enhancement (Parallel WaveGAN, MelGAN), where they supervise the
   FINAL generated waveform with no masking. rPPG works that use a
   frequency-domain loss supervise the final signal, usually via an
   HR/dominant-frequency or PSD-peak term against the reference rate, not a
   multi-resolution magnitude match. So this design has no precedent, and §9.2
   shows why.

### 9.4 What to do instead (ordered levers that attack the actual cause)

1. **Span masking for the physio stream.** Mask CONTIGUOUS blocks of resp
   tokens (ideally >= 1 breath, several seconds) so interpolation across the gap
   is impossible and reconstruction requires extrapolation from the video. Code
   change: `core/multimae.make_masks` uses `_random_mask` for every non-visual
   stream; a span/block variant is needed (a 90 % contiguous span was
   monkeypatched once to verify STFT/mask compatibility, never added to the
   model).
2. **Mask the physio stream completely** (the clamp allows `N-1` of `N`, i.e.
   49/50 ≈ 98 %), which makes Stage 2 the *same task* as Stage 3's "full sensor
   failure" protocol instead of a same-modality interpolation exercise.
3. **`target_norm: token`** — kills the position-wise-mean shortcut and is now
   UNBLOCKED because the spectral term is off. Cheap to test, but note it does
   not by itself defeat interpolation.
4. **Only reconsider a spectral term after the model actually beats the
   interpolator** (target: `mse_resp < 0.13` **and** `spec_resp < 0.97`). Below
   those numbers there is nothing for a spectral prior to regularise; above
   them the encoder is demonstrably not using the video.

