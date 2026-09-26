# Thesis codebase — scaffold

Target (ImplementationPlan.md): contactless **2D RGB + Thermal-IR video →
1D BP / RESP / EDA waveform** recovery via a 3-stage progressive pipeline
(ImageNet init -> multimodal masked pre-training on BP4D+ -> three task-specific
waveform fine-tuning branches).

Layout merging best practices from **MultiMAE** (`tmp/MultiMAE`) and
**VideoMAE** (`tmp/videomae`).

```
code/
├── configs/     # YAML configs (MultiMAE style: YAML sets argparse defaults)
├── core/        # Pure model code, no I/O / training logic (<= multimae/)
├── data/        # Datasets + masking generators (<= videomae/ datasets, masking_generator)
├── models/      # Thin create_model() factory over core registry
├── engines/     # Training/val loops OUT of run scripts (<= videomae/ engine_*.py)
├── evaluation/  # Offline Tier-1/2/3 waveform metrics (time, PSD, NeuroKit2 HRV)
├── utils/       # Helper hub package with __init__ re-exports (<= MultiMAE utils/)
├── runners/     # Thin entry points: run_pretrain.py / run_finetune.py / ...
└── scripts/     # HPC launchers, one .sh per dataset/variant (<= videomae/ scripts/)
```

## Source mapping (where to port the heavy parts from)

| This folder                   | Port from referenced projects (tmp/MultiMAE, tmp/videomae)                    | Notes                                           |
| ----------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------- |
| `core/registry.py`          | `tmp/MultiMAE/utils/registry.py`                                            | timm-style`@register_model`                   |
| `core/blocks.py`            | `tmp/MultiMAE/multimae/multimae_utils.py`                                   | Block / Attention / DropPath / trunc_normal_    |
| `core/input_adapters.py`    | `tmp/MultiMAE/multimae/input_adapters.py`                                   | per-modality input adapters                     |
| `core/output_adapters.py`   | `tmp/MultiMAE/multimae/output_adapters.py`                                  | reconstruction heads (Spatial/DPT/ConvNeXt/...) |
| `core/criterion.py`         | `tmp/MultiMAE/multimae/criterion.py`                                        | masked MSE/L1/CE                                |
| `core/model.py`             | `tmp/MultiMAE/multimae/multimae.py` + `tmp/videomae/modeling_finetune.py` | architecture + registered entrypoints           |
| `data/datasets.py`          | `tmp/MultiMAE/utils/datasets.py`, `tmp/videomae/datasets.py`              | dataset builders (modality specific)            |
| `data/masking_generator.py` | `tmp/videomae/masking_generator.py`                                         | Tube / Random masking                           |
| `utils/*`                   | `tmp/MultiMAE/utils/*`, `tmp/videomae/utils.py`                           | dist, logging, checkpoint, optim, EMA, scaler   |

## Conventions

- **Config**: YAML under `configs/`; run scripts accept `-c/--config` and merge it
  as argparse defaults; CLI flags override YAML (see `runners/run_pretrain.py`).
- **Model registration**: decorate factory functions with `@register_model` from
  `core.registry` and build them via `models.build.create_model`.
- **Engines**: keep training loops in `engines/`, keep `runners/` scripts thin.
- **Imports**: run scripts assume the CWD is `code/` (top-level packages
  `core`, `models`, `data`, `engines`, `utils`), same as MultiMAE/VideoMAE.
- **License headers**: keep the BSD provenance header when porting code blocks.

## Running

```bash
cd code
# (0) data: convert raw sessions once, then pinspect a few
python data/prepare_bp4d.py --raw_root ../data/raw/BP4D --out_root ../data/processed/bp4d_canonical --limit_sessions 8
python runners/run_inspect_data.py --data_path ../data/processed/bp4d_canonical --clip_duration 2 --input_size 64 --plot
# raw physiology only (min/max/length/estimated frequency, original sample rate)
python runners/run_inspect_physio.py --subject F001 --task T1 --channel all

# (1) Stage-2 multimodal masked pre-training (local milestone)
#   from-scratch smoke slice (base, 768-d) . configs/pretrain/stage2_local_scratch.yaml
#   VideoMAE/MAE ViT-Base inheritance (768-d) . configs/pretrain/stage2_local_pretrained.yaml
python runners/run_pretrain.py -c configs/pretrain/stage2_local_pretrained.yaml

# (2) Stage-3 waveform fine-tuning per branch (needs a Stage-2 encoder ckpt)
python runners/run_waveform.py -c configs/finetune/bp.yaml
python runners/run_waveform.py -c configs/finetune/resp.yaml
python runners/run_waveform.py -c configs/finetune/eda.yaml

# (3) offline multi-tier evaluation of saved predictions
python runners/run_evaluate.py --pred_path out.npy --target_path gt.npy \
    --fs 100 --tier 1,2,3 --waveform bp

# (4) OPTIONAL ADD-ON diagnostic: AU-occurrence probe on the Stage-2 encoder
python runners/run_au_probe.py -c configs/finetune/au_local.yaml

# (5) OPTIONAL ADD-ON Stage 2: thermal ROI (visual) -> RESPIRATION (1-D)
#     procedure of record: code/TirROI_Resp_plan.md
python runners/run_pretrain.py -c configs/pretrain/stage2_local_tir_roi_resp.yaml

# multi-GPU (HPC)
bash scripts/project/pretrain.sh
bash scripts/project/finetune.sh
```

### Stage-2 pre-training was silently training NOTHING (fixed 2026-09-23)

`clip_grad: 0.0` means "no gradient clipping", but the AMP branch of
`utils/native_scaler.py::NativeScalerWithGradNormCount.__call__` tested
`if clip_grad is not None:` and handed `0.0` straight to `torch.nn.utils.clip_grad_norm_`.
That call rescales every gradient by `clip_coef = 0.0 / (total_norm + eps) = 0`:

```python
clip_grad_norm_(max_norm=0.0) -> total_norm 1.732 ; grad now [0.0, 0.0, 0.0]
clip_grad_norm_(max_norm=1.0) -> grad now [0.577, 0.577, 0.577]
```

The returned `grad_norm` still looked healthy because it is measured *before* the
scaling, so the logs gave no hint. Both `run_pretrain.py` and `run_waveform.py`
construct a real scaler, so **every Stage-2/Stage-3 AMP run at the default
`--clip_grad 0.0` had its gradients zeroed**. The guard is now
`if clip_grad is not None and clip_grad > 0:` (mirroring the non-AMP branch).

Evidence from `output/pretrain/stage2_local_pretrained_BROKEN_clipgrad0/`: in
`checkpoint-0039.pth` the Adam buffers `exp_avg` / `exp_avg_sq` were **exactly zero
for all 188 parameters** (nothing ever reached the optimiser), and the 68 tensors
that *did* differ from `checkpoint-0000.pth` differed by exactly the same factor
`0.98280` (std `0.00000`) as the analytically predicted pure weight decay
`prod(1 - lr_t*wd) = 0.98273` — i.e. AdamW's decoupled weight decay, no learning.

**How to check any run for this class of bug:** load the checkpoint and assert that
some `optimizer['state'][pid]['exp_avg']` is non-zero. A non-zero `grad_norm` in the
log is NOT evidence of learning; a weight change is NOT either (weight decay moves
weights with zero gradients). Verify with a loss **drop**.

After the fix, the same 40-epoch config trains properly: loss `5.06 -> 2.21`,
`mse_rgb 1.48 -> 0.22`, `mse_bp 1.61 -> 1.06`, `mse_resp 3.59 -> 1.49`,
`spec_resp 6.94 -> 5.11`, 376/376 optimiser buffers non-zero, and the encoder sits
`0.0687` (relative mean |Δ|) from its Stage-1 init versus `0.0174` for the broken
run. The untrained run was moved to
`output/pretrain/stage2_local_pretrained_BROKEN_clipgrad0/`.

### Stage-2 pre-training loss (weighted per-modality masked MSE)

The Stage-2 (`core/multimae.py::MultiModalMAE`) reconstruction loss is a
**masked MSE**, computed **independently per modality over its masked positions
only** (`mask == 1` ⇒ to reconstruct; `core/criterion.py::MaskedMSELoss`), then
combined as a **weighted sum** over the streams:

```math
L = Σ_s λ_s·MSE_s ,      λ_{RGB} = λ_{TIR} = 1.0 ,      λ_{physio} = signal weight (default 0.5)
```

Visual streams (RGB/TIR) keep weight 1.0; every physio (1-D) stream uses
`--signal_weight` (default **0.5**). Tune the physio weight in ~[0.5, 1.0]
from the **normalized variance of the physio stream's masked target tokens**:
high variance ⇒ lean 0.5 so the 1-D signal does not dominate the loss gradient;
low variance ⇒ lean 1.0 so it is not ignored. `--loss_weights` is a full
per-stream override (one comma value per `--streams` modality, in order; empty
⇒ policy above). `MultiModalMAE.forward` returns `losses_mse` (raw per-modality
masked MSE) and `losses` (weighted contributions); `loss = Σ losses`.

#### Optional periodicity prior: multi-resolution STFT loss on the 1-D streams

A per-token masked MSE constrains **amplitude** only — a low-frequency
surrogate can lower it without ever modelling the cardiac/respiratory cycle.
Setting **`--spectral_weight`** (default **0.0 = off**) adds a
**multi-resolution STFT magnitude loss**
(`core/waveform_losses.MultiResolutionSTFTLoss`) on the **assembled** clip
waveform, so the shared encoder gets a direct gradient on periodicity:

```math
L = Σ_s ( λ_s·MSE_s + w_s·STFT_s ) ,      w_physio = spectral weight ,      w_RGB = w_TIR = 0
```

* **Assembled waveform:** `[B, n_signal*sig_kernel]` — the concatenated
  per-token windows (`SignalEmbed` uses kernel == stride), i.e. the whole clip.
* **`--target_norm clip` is REQUIRED** (the model raises under the default
  `token` norm): one mean/std per (sample, stream) instead of one per token, so
  the token windows reassemble into a *coherent* waveform. Under per-token
  normalization every token is independently rescaled, so its spectrum carries
  token-boundary artefacts instead of the physiological band the loss is meant
  to enforce. It also aligns the Stage-2 target space with Stage 3.
* **Knobs:** `--spectral_fft_sizes` (default `64,128,256` ⇒ 1.56 / 0.78 /
  0.39 Hz resolution at fs = 100 Hz, covering BP 1.0–2.5 Hz and RESP
  0.16–0.4 Hz), `--spectral_hop_ratio` (default 0.25 ⇒ 75 % overlap), and
  `--spectral_weights` — a FULL per-stream override (one comma value per
  `--streams` modality, same convention as `--loss_weights`; a positive value on
  a video stream is rejected). Start at ~0.1 and tune.
* **Diagnostics:** `forward` also returns `losses_spectral` (raw MR-STFT per
  stream) and the engine logs `mse_<stream>` / `spec_<stream>` per epoch, so the
  term's effect is visible without a separate probe.
* **Comparability:** enabling it switches the 1-D targets from per-token to
  per-clip normalization, so the reported loss scale is **not** comparable with
  runs made before it.

### Stage-2 streams — flexible modality contract
The pretraining modalities are configured with `--streams` (or `streams:` in
the YAML under `configs/pretrain/`) as a comma list. The **Stage-2 contract**
requires at least TWO streams: **≥1 video** (`rgb` and/or `tir`) **plus ≥1
physiological 1-D signal** (`bp`, `resp`, `eda` — the waveform later
regressed in Stage 3). Video-only, signal-only, single-modality, empty or
unknown lists are rejected consistently in `MultiModalMAE.__init__`,
`build_pretraining_model` (`core/multimae.py`) and
`PairedPretrainDataset`/`build_pretraining_dataset` (`data/`).

`PairedPretrainDataset` serves **exactly** the requested streams: e.g.
`streams: rgb,bp` returns only `{'rgb','bp'}`, while the default five-stream
configs return all of `{'rgb','tir','bp','resp','eda'}`. So trimming/ablating
modalities (while keeping the ≥1 video + ≥1 physio rule) is a YAML-only
change. Per-stream mask ratios: visual `mask_ratio_rgb/tir` 50–75 %, signals
`mask_ratio_bp/resp/eda` 90 %+.

Implemented and run so far (see the thesis-plan table below): canonical BP4D
conversion, the aligned `PairedSessionDataset` (+ overlapping windows via
`clip_stride`), the multimodal masked autoencoder `core/multimae.py`
(asymmetric per-stream masks + the weighted per-modality masked MSE above), its pretrain loader
`PairedPretrainDataset` and runner `run_pretrain.py` (explicit `--lr` +
step-level warmup/cosine schedule via `utils/lr_sched.py`), MAE ViT-Base
encoder inheritance (`load_pretrained_encoder`), and the Stage-3 waveform
scaffold (`run_waveform.py`, `WaveformJointLoss`, `evaluation/`).

**Second Stage-2 source (`data_set: tir_roi`).** `--data_set` selects the
dataset behind `build_pretraining_dataset`: the default `bp4d+` keeps
`PairedPretrainDataset` (canonical sessions), while `tir_roi` builds the
ADD-ON thermal-ROI + respiration dataset from the RAW tree and is the only
path whose visual stream is the landmark-derived mouth/nose crop. The first
such run is `configs/pretrain/stage2_local_tir_roi_resp.yaml` (`streams: tir,resp`);
its procedure of record is `code/TirROI_Resp_plan.md`. It needs no model
change, and with `streams: tir,...` the Stage-1 loader now routes the
checkpoint's tubelet into `adapters.tir` instead of leaving it random
(`load_pretrained_encoder` resolves the destination visual adapter; an
`rgb,...` run is unaffected). The canonical `rgb,bp*` configs and
`PairedPretrainDataset` are untouched.

Not yet implemented: a finer Stage-3 temporal decoder, and the *session-level*
reconstruction/stitching that turns per-clip predictions into one continuous
whole-session waveform (a planned offline inference step — per-clip training is
clip-level by design). (RESP/EDA streams in Stage 2 *are* implemented — the
local configs now train all five streams, see the flexible-modalities
subsection above.)

## Thesis plan alignment (ImplementationPlan.md)

> **ADD-ON (not a pipeline stage):** the *AU-occurrence probe* (semantic
> representation quality, `runners/run_au_probe.py`) is a separate diagnostic
> on the Stage-2 encoder — it is not part of the three-stage pipeline above.
> See the dedicated AU-probe subsection below and `code/ImplementationPlan.md` §5.

| Plan stage                                                                            | Supported here                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Still to port (thesis work)                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stage 1 — ImageNet init of ViT-Base encoder                                          | `core/model.py` entrypoints (`project_vit_base_patch16_224`)                                                                                                                                                                                                                                                                                                                                                                                                                     | official ImageNet-1K timm classifier converter; encoder inheritance IMPLEMENTED (`core/multimae.py::load_pretrained_encoder` + `canonicalise_vit_state_dict` -- 3-D tubelet kernel transferred verbatim for VideoMAE, 2-D filter boxcar-inflated for MAE, `--inflate_rgb_patch` to disable) via `--pretrained_encoder`; variant specs (`small |
| Stage 2 — multimodal masked pre-training on BP4D+ (RGB/TIR 50-75%, BP/RESP/EDA 90%+) | `core/input_adapters.py` (`SignalInputAdapter`), `data/masking_generator.py` (`MultiModalMaskingGenerator` asymmetric), `core/criterion.py` (`MaskedMSELoss`), configs `configs/pretrain/stage2_local_scratch.yaml` (from scratch) + `stage2_local_pretrained.yaml` (base + Stage-1 init) + HPC template `stage2_multimodal.yaml`; implemented local milestone: `core/multimae.py` (`MultiModalMAE`) + `PairedPretrainDataset` + `runners/run_pretrain.py` | separate deeper decoders; full-data HPC run at larger 224 geometry. Local five-stream milestone (rgb+tir+bp+resp+eda) is implemented & run:`core/multimae.py` (`MultiModalMAE`), `data/paired_dataset.PairedPretrainDataset`, `runners/run_pretrain.py`, `configs/pretrain/stage2_local{_scratch,_pretrained}.yaml`                         |
| Stage 3 — three branches BP, RESP & EDA, unified spatio-temporal-spectral loss       | `core/waveform_losses.py` (`WaveformJointLoss`: L1 + Pearson + MR-STFT; 64/128/256 for BP/RESP, 256/512/1024 for EDA), regression head (`ProjectViT(output_len=...)`, baseline CLS->seq), `runners/run_waveform.py`, configs `configs/finetune/{bp,resp,eda}.yaml`                                                                                                                                                                                                         | lightweight conv decoder over all tokens for finer temporal resolution; session-level whole-waveform reconstruction/stitching (offline inference, not yet implemented)                                                                                                                                                                                |
| Evaluation — Tier 1/2/3 post-processing                                              | `evaluation/metrics.py` (MAE/RMSE/Pearson; Welch PSD), `evaluation/clinical.py` (NeuroKit2 RMSSD/pNN50/MedianNN/ShanEn), `runners/run_evaluate.py`                                                                                                                                                                                                                                                                                                                             | —                                                                                                                                                                                                                                                                                                                                                    |

```bash
# Stage 3 example (needs a Stage-2 encoder ckpt)
python runners/run_waveform.py -c configs/finetune/bp.yaml
python runners/run_waveform.py -c configs/finetune/resp.yaml
python runners/run_waveform.py -c configs/finetune/eda.yaml
# Offline post-processing on saved predictions
python runners/run_evaluate.py --pred_path out.npy --target_path gt.npy \
    --fs 100 --tier 1,2,3 --waveform bp
```

### Whole-session waveform reconstruction (planned)

Stage-3 *training* is per clip (`video window -> waveform window`). The final
deliverable — the continuous 1-D waveform of a whole session (subject/task) —
is assembled **after** training by sliding the window over the session
(`clip_duration` + `clip_stride`) and overlap-adding (stitching) the per-window
predictions, then evaluated offline with Tier 1 (time), Tier 2 (spectral) and
Tier 3 (clinical/NeuroKit2, BP/HRV only). This stitching step is **not yet
implemented** (no training involved).

### Recorded data layout (asymmetric RGB jpg-seq + TIR .wmv)

```
<data_path>/<session>/
    rgb/            # ordered jpg frames  (25 fps nominal)
    tir.wmv         # single WMV ~60 s    (25 fps nominal)
    signals.csv     # header: [time,] bp, resp, eda   at fs Hz
```

Readers (`data/video_io.py`), temporal registration (`data/alignment.py`) and
the synchronised `PairedSessionDataset` (`data/paired_dataset.py`) handle this
layout; select it with `data_set: bp4d+` (alias `paired`). Per-session TIR fps
and signal sample rates are probed at runtime; both modalities are read on one
common time grid and the target waveform is resampled to `seq_len`. Set
data-set params in the YAML: `fs`, `fps`, `clip_duration`, `clip_stride`, `seq_len`,
`input_size`, `rgb_dir`, `tir_file`, `signals_file`, `train_ratio`,
`tir_channels`.

**TIR is not gray (verified 2026-09).** The BP4D thermal `.wmv` is a
false-colour (rainbow) thermal *rendering* with the camera's degC legend burned
into the right edge. Two independent decoders agree: `wmv3`/`yuv420p`, 3 planes
decoded, mean `|U-128| = 23-27` and `|V-128| = 16-20` on all 11 local sessions,
with the chroma plane following the scene (not noise, not a flat cast); the raw
`Thermal/<S>/<T>.wmv` and the canonical `tir.wmv` are byte-identical, so this is
source data rather than a preprocessing artefact. The pipeline therefore keeps
TIR as **3-channel** (`tir_channels: 3`, the default): the Stage-2 TIR adapter is
`Conv3d(3->D)`, `PairedSessionDataset` returns `[3+C, T, H, W]` = `[6, T, H, W]`
(RGB 3 + TIR 3) so Stage 3 needs `in_chans: 6`. `--tir_channels 1` restores the
legacy luma-only surrogate (`cvtColor(BGR2GRAY)`) and is only needed to match
Stage-2 checkpoints trained before this change - it does change the TIR adapter
geometry. Consequence to state wherever TIR is claimed to carry temperature:
this input is a *rendered* proxy whose palette depends on the camera's per-frame
auto-range, not radiometric temperature. (The legend itself is cropped away by
`resize_center_crop` at 64 px and at 224 px.)

Optional dev/quick-run caps (accepted by every training runner and by the
inspect runner): `max_sessions` bounds the number of decoded sessions up
front, `max_clips` bounds the number of windows taken from *each* session,
and `max_entries` bounds the global total number of clips in the dataset.
`clip_stride` (default `0` = non-overlapping) may be set to a value smaller
than `clip_duration` to generate overlapping windows and thus more samples per
session; combined with `max_clips` it keeps only the earliest windows of each
session.

### Data-loading throughput — RGB decode is the bottleneck, not the disk

Measured 2026-09 on 1392x1040 jpgs (`F004_T2`, cold session, 60 frames):

| step                                                                 | ms/frame |
| -------------------------------------------------------------------- | -------- |
| read the jpg bytes,**no decode**                               | 0.6      |
| read +**full decode** (the old path)                           | 11.6     |
| full decode of bytes already in RAM (**no filesystem at all**) | 9.2      |
| `IMREAD_REDUCED_COLOR_4` decode of bytes already in RAM            | 6.3      |

~90 % of the per-frame cost is libjpeg decoding a 1.4 MP image whose pixels are
then 97 % discarded for a 224 px target; the filesystem contributes well under
1 ms. A faster disk therefore does **not** fix it. One worker sustains ~0.35 of
a 10 s clip per second, i.e. **~2.6-2.9 s of single-core CPU per 10 s clip**, so
size `num_workers` against that budget. Stage 2 also decodes each session's TIR
**once** at dataset construction and keeps it in RAM (`_tir_cache`); only the RGB
grid is re-read per clip (250 jpgs for a 10 s window).

**Implemented (2026-09).** `data/video_io.py::read_image` asks OpenCV for a
DCT-scaled JPEG decode (`IMREAD_REDUCED_{COLOR,GRAYSCALE}_{8,4,2}`) chosen by
`_imread_flag` from the source size parsed out of the JPEG SOF header
(`_jpeg_size`, no decode): the coarsest scale that still covers `target_size`, so
the image is never upscaled. Non-JPEG inputs, `target_size=None` and sources
smaller than the scale keep the previous full decode.

| `target_size` | flag chosen (1392x1040 source) | mean abs diff vs full decode | speedup |
| --------------- | ------------------------------ | ---------------------------- | ------- |
| `None`        | `IMREAD_UNCHANGED`           | 0 (byte-identical)           | 1.00x   |
| 512             | `IMREAD_REDUCED_COLOR_2`     | 0.55/255                     | 1.15x   |
| 224             | `IMREAD_REDUCED_COLOR_4`     | 0.69/255                     | 1.38x   |
| 64              | `IMREAD_REDUCED_COLOR_8`     | 0.50/255                     | 1.34x   |

Shapes are unchanged for every `target_size` x `gray` combination, and a 10 s /
250-frame clip at 224 px drops from ~3450 ms to ~2630 ms (the clip-level gain is
smaller than the per-frame 1.4x because TIR slicing and the float conversion are
in the same loop).

**Not implemented (the remaining lever).** The decode itself is still there.
Materialising each session once into a uint8 `(T, S, S, 3)` mmap-able array
removes it: measured 4.5 ms/clip from page cache (71 ms cold) instead of
3450 ms, which takes the loader off the critical path entirely. Sizing: the
7.96 GB of jpgs (10 594 frames) collapse to 1.59 GB at 224 px (130 MB at 64 px);
building all 11 sessions takes ~2.5 min once. This is a *CPU* fix rather than a
disk fix, and it would also remove the per-rank startup TIR decode under DDP.

### Thermal -> respiration dataset (`BP4DPlusTIRRespDataset`, ADD-ON)

A second, independent read-out of the same raw corpus as the AU probe, and an
ADD-ON like it: **no Stage-1/2/3 file is touched**. One sample is one thermal
clip plus the respiration waveform over exactly the same time span:

```
'tir_video'    float32 [3, T, H, W]   T = clip_seconds * 25 fps / temporal_stride
'resp_signal'  float32 [L]            L = T * temporal_stride / 25 * 100 Hz
'subject_task' str                    e.g. 'F001_T1'
```

`temporal_stride` (8 s clip): 1 -> `T=200`, `L=800`, 1600 visual tokens
(`sig_kernel 8`); 2 (the shipped config) -> `T=100`, `L=800`, 800 tokens
(`sig_kernel 16`). A stride change is a **new geometry** (different pos-embed
token count, so checkpoints are not interchangeable) and **must** be matched by
`sig_kernel = tubelet_t * temporal_stride / fps * fs`, which the model enforces
with a hard error naming the value to use.

It reads the **raw** tree (no `prepare_bp4d.py` copy needed):

```
<raw_root>/Thermal/<S>/<T>.wmv                 # 25 fps, 726x480 false-colour TIR
<raw_root>/IRFeatures/<S>_<T>.txt              # 28 (x, y) PIXEL pairs per frame
<raw_root>/Physiology/<S>/<T>/Resp_Volts.txt   # 1000 Hz, one value per line
```

* **IRFeatures format.** One line per thermal frame, 56 floats = 28 `(x, y)`
  pairs in raw 726x480 pixel coordinates; **line `n` == frame `n` (1-based)**
  (verified: `F001_T1` has 1612 lines, the video decodes 1612 frames at
  25.000 fps). `(0, 0)` is the corpus' undocumented **missing-data sentinel**
  (written when the tracker loses the frontal fiducials -- the head is turned
  away), and it is all-or-nothing per frame: F001_T8 has 112/227 such lines.
* **Two skip rules.** (1) A session with **no** `IRFeatures` file is skipped
  gracefully (the guide ships 15 untracked, glasses-wearing sequences, e.g.
  `F016_T2..T4`, `M045_T2`, `M049_T1..T10`). (2) **Any clip whose frame range touches a `(0,0)` line is dropped
  entirely** and the next window is tried -- the sentinel would otherwise drag
  the ROI box to the image corner. Every skip and every dropped window is
  recorded (`ds.skipped`, `ds.stats['clips_dropped_sentinel']`) instead of
  being swallowed, and a file whose lines do NOT all carry 56 values raises,
  because then line/frame alignment can no longer be trusted.
* **ROI = one CLIP-STATIC box.** For the 12 mouth+nose landmarks (1-indexed
  user-guide labels `[9,10,11,12,13,20,21,22,23,24,25,26]` = nose bridge 9/20,
  nostrils 10/21, mouth corners 11/22, lips 12/13/23/24, lip centres 25/26) of
  all 200 frames: take the global `(x_min, x_max, y_min, y_max)`, extend each
  side by `roi_padding * extent` (0.2 -> +40 % overall), clamp to the frame, and
  crop **all 200 frames with that one box**, each resized to `input_size` with
  `cv2.resize`. A per-clip (not per-frame) box means the patch cannot jitter
  spatially -- the visible cost is that a moving head is covered by a slightly
  loose box (measured over F001_T1's 8 clips: 84x98 .. 125x111 px, i.e. 2.5-4.5x
  the mouth+nose area of a single frame, except the last clip where the subject
  turns away and it reaches 118x162 px = 7.5x, out of 726x480), which is what
  the padding is for. Reduce `roi_padding` (0 = the tight union) or shorten the
  clip to tighten it.
* **Alignment is by TIME, not a hand-tuned offset.** A clip starting at frame
  `f` covers `[f/25, (f+T)/25)` s and the respiration slice is that window on
  the `[L] = clip_seconds * 1000` grid; at the corpus' nominal rates the mapping
  is the exact integer slice `resp[f*40 : f*40 + 8000]`. `Resp_Volts.txt` holds
  64597 samples over the 64.48 s of `F001_T1` = 1001.8 Hz, i.e. the 1000 Hz
  nominal rate. Note the raw resp trace **rails at exactly -10.0000 V** in some
  sessions (F001 T2/T6/T7/T8) -- the dataset returns it as measured; drop or
  mask those windows before drawing conclusions about amplitude.
* **Normalisation.** Thermal frames -> `[0, 1]` by `/255`; respiration is
  z-scored **per clip** (`(y - mu) / (sigma + 1e-8)`, the default, matching the
  Stage-2 `target_norm: clip` convention). `norm='session'` uses whole-session
  statistics and `norm='none'` returns raw volts (needed for any clinical
  amplitude comparison).
* **Clips are non-overlapping by default** (`clip_stride=None`); pass
  `clip_stride` in SECONDS for a hop. **No split is applied here** -- this
  dataset exposes one stream per session, and the subject-disjoint split is the
  caller's job (do not split by clip).
* Cost: `CV2ClipReader.read_range` / `DecordClipReader.read_range` (new,
  2026-09-24) seek instead of decoding from frame 0, verified **frame-exact**
  against a full sequential decode on `F001_T2` (k = 0/1/100/200/399/400/550,
  maxdiff 0). A 64 px compile-free clip still costs ~0.7-2.4 s of decode, so the
  same decode-CPU argument as above applies; `preload=True` materialises all
  clip tensors at init (2.5 MB per 64 px clip) when epochs are re-read.

```bash
# verify the dataset: skip rules, shapes, [0,1] range, z-score, ROI box,
# tensor==crop, temporal alignment, resp window vs a raw-file slice
python data/tir_resp_dataset.py --n_check 3

# inspect one or more clips (figure + JSON per clip) via the wrapper
bash scripts/local/inspect_tir_resp.sh --list
SUBJECT=F001 TASK=T1 CLIPS=0,7 bash scripts/local/inspect_tir_resp.sh
```

Files: `data/tir_resp_dataset.py` (dataset + `parse_ir_features` +
`roi_box_from_landmarks` + `TirRoiRespPretrainDataset` for Stage 2 + `main`
self-test), `runners/run_inspect_tir_resp.py` (per-clip figure + JSON reports),
`scripts/local/inspect_tir_resp.sh`. Verified on the **40-session** local corpus
(4 subjects x T1..T10, 45 858 thermal frames = 30.6 min): 431 clips at 4 s
non-overlapping / 843 at a 2 s hop / 1663 at 1 s, 9-34 windows dropped for
sentinels depending on the hop, 0 sessions skipped, and every one of the 40
`IRFeatures` tracks has `lines == video frames`. The two negative paths are
asserted too -- `F001_T8` (112 sentinel lines) keeps only its one clean 4 s
window, and a SYNTHETIC tree with a thermal video but no `IRFeatures` is skipped
with 0 clips, so the rule is verified even though no shipped sequence is
untracked any more.

### AU-occurrence probe — Semantic Representation Quality (ADD-ON)

A **diagnostic control, not a Stage-1/2/3 step**. Its claim: Stage-2
unsupervised masked pre-training should make FACS facial-action semantics
linearly decodable from the (frozen) shared encoder. It is implemented as an
independent, self-contained module that re-uses the Stage-2 shared encoder
(identical module names/geometry); **no Stage code is modified**.

* **Task & label.** Per RGB **clip** (length == the Stage-2 `num_frames` =
  `clip_duration * fps`), predict ONE multi-label AU occurrence vector,
  supervised by the **centre frame** only (temporal context; centre-anchored).
  AU value `9` (unknown) is dropped — never treated as absent.
* **Data.** Canonical `rgb/` frames + raw BP4D+ `AUCoding/AU_OCC/<Session>.csv`
  (AU-coded tasks **T1/T6/T7/T8**, 140 subjects × 4). AU frame `f` ⇔ jpg
  number `f-1`. **Subject-disjoint** train/val splits (never frame/session).
* **Loss / metric.** Multi-label BCE; per-AU F1 @0.5 and macro-F1 over the
  target AU set (BP4D protocol). Report a per-AU *trivial baseline* too: coded
  segments are the "most expressive" ~15–28 s, so some AUs are near-constant.
* **Probe modes & controls** (identical protocol; only `--finetune` differs):
  * `--probe linear` (default) freezes the encoder ⇒ the formal diagnostic;
    `--probe ft` fine-tunes the whole model.
  * C0 random init (`--finetune ''`), C1 Stage-1 (`--finetune base` =
    `videomae:base`, or that checkpoint's path), **C2 Stage-2** (headline).
* **Configurable AU set.** `--au_list` (explicit, string or YAML list) **or**
  `--au_freq_topk N` (top-N by presence rate over the full AU corpus;
  `5` ⇒ AU6/7/10/12/14); default = the BP4D 12-AU subset. The head width,
  dataset label columns and per-AU table all follow the resolved list.
* **Two local configs = two probe variants of the same control.** Each of
  `configs/finetune/au_local.yaml` and `au_local_pretrained.yaml` probes ONE
  locally-trained Stage-2 encoder, so each must mirror that checkpoint's
  geometry: `au_local.yaml` -> `stage2_local_scratch` (768-d/12-layer from-scratch;
  clip 4.0 s -> num_frames 100, batch 4, full 12-AU subset),
  `au_local_pretrained.yaml` -> `stage2_local_pretrained` (MAE ViT-Base-inherited
  768-d/12-layer; clip 4.0 s -> num_frames 100, batch 2, top-5 frequent AUs
  `{6,7,10,12,14}`). Both use the SAME window length; only `batch_size` is
  smaller (2 vs 4) to keep the ~16x bigger 768-d encoder within a LOCAL GPU (see
  the `Geometry contract` bullet). Within one geometry, keep the protocol
  identical across C0/C1/C2 (C1 = the MAE 768-d checkpoint only fits the 768-d
  geometry). The two configs use DIFFERENT AU sets by design, so they are not a
  head-to-head A/B of the two encoders - fix one AU set before comparing across
  encoders.
* **Geometry contract.** `tubelet`, `input_size`, `clip_duration`,
  `enc_embed_dim`, `enc_depth`, `enc_num_heads`, `mlp_ratio` MUST match the
  probed Stage-2 checkpoint (the loader raises on mismatch instead of silently
  loading nothing). Visual-stream input only (no physio leakage).
* **Files.** `core/au_probe.py` (probe model + checkpoint loader),
  `data/au_dataset.py` (dataset + subject split + AU-set resolution),
  `engines/au_probe.py` (BCE train + per-AU F1 eval),
  `runners/run_au_probe.py`, configs `configs/finetune/au_local{,_pretrained}.yaml`,
  launcher `scripts/local/au_smoke.sh`. Details & status:
  `code/ImplementationPlan.md` §5.

```bash
# AU probe (linear, Stage-2 ckpt); local smoke via scripts/local/au_smoke.sh
python runners/run_au_probe.py -c configs/finetune/au_local.yaml
python runners/run_au_probe.py -c configs/finetune/au_local_pretrained.yaml
# controls / AU subset
python runners/run_au_probe.py -c configs/finetune/au_local_pretrained.yaml \
    --finetune base                                         # C1 Stage-1 (768-d)
python runners/run_au_probe.py -c configs/finetune/au_local.yaml \
    --au_list '' --au_freq_topk 5                            # top-5 frequent AUs
```

## Suggested port order

1. `utils/` (dist, logger, metrics, checkpoint, optim_factory, EMA, scaler, pos_embed)
2. `core/` (registry -> blocks -> adapters -> criterion -> model)
3. `data/` (datasets + masking)
4. `engines/` then `runners/` wiring + YAML
5. `scripts/*.sh` launchers

## Local development vs Lichtenberg HPC

The same `code/` tree is used for (a) quick single-GPU tests on a few samples
and (b) real multi-GPU training on the Lichtenberg cluster. Everything machine
specific is injected through environment variables — never hard-coded.

**Environment profiles** (source one from `code/`):

- `scripts/env_local.sh`  — single local GPU / WSL. Points at the repo data
  (`data/raw/BP4D` and `data/processed/bp4d_canonical`) and an `output/` dir.
- `scripts/env_hpc.sh`    — Lichtenberg (EDIT the marked values: `VENV`/`CONDA_ENV`,
  `RAW_DATA_PATH`, `DATA_PATH`, `OUTPUT_DIR`, `CODE_DIR`, `PARTITION`, `GPU_TYPE`,
  `GPUS_PER_NODE`). A `venv` is supported via `activate_project_env()`.

Env vars read by the runners: `DATA_PATH`, `OUTPUT_DIR`, `DATA_SET`,
`RAW_DATA_PATH` (converter only), `MODEL_PATH` (`--finetune`), `RESUME`,
`NUM_WORKERS`, `INITIAL_MODELS_DIR` (downloaded Stage-1 weights), `HF_ENDPOINT`
(HuggingFace mirror). CLI flags always take precedence over env/YAML.

**Initial (Stage-1) encoder weights.** `models.build.create_model` initialises
randomly; the Stage-1 spatial priors come from a public checkpoint. Instead of a
path you may pass a *variant spec* to `--finetune` / `--pretrained_encoder`; the
matching checkpoint is downloaded **once** into `<repo>/models/initial/`
(already git-ignored via `models/*`; relocate with `$INITIAL_MODELS_DIR` or
`--weights_dir`, e.g. a scratch volume on the HPC).

| spec                            | source                                                            | geometry (dim/depth/heads) |
| ------------------------------- | ----------------------------------------------------------------- | -------------------------- |
| `base` \| `videomae:base`   | **VideoMAE ViT-Base, tube-masked video MAE (Kinetics-400)** | 768/12/12                  |
| `mae:base`                    | MAE ViT-Base, self-supervised ImageNet-1k                         | 768/12/12                  |
| `large` \| `videomae:large` | VideoMAE ViT-Large                                                | 1024/24/16                 |
| `mae:large`                   | MAE ViT-Large                                                     | 1024/24/16                 |
| `timm:<model_id>`             | any timm/HF checkpoint, e.g.`timm:vit_base_patch16_224.mae`     | as named                   |

**VideoMAE is the default Stage-1 source for `base`/`large`, and it is the
better one.** Its `patch_embed.proj` is a `Conv3d(3, D, (2,16,16))` tubelet
filter -- the *exact* shape this repo's adapters use (`tubelet: 2,16,16`) -- so
the tokenizer transfers **verbatim**: the learned temporal kernel is inherited
instead of being faked by averaging two frames (a plain MAE source leaves the
model motion-blind at init), and its objective (tube-masked *video* MAE) is the
same family as Stage 2. Verified on the real checkpoint: 148 encoder tensors
loaded, 0 shape-mismatched, 102 pre-training/decoder keys dropped. Weights are
**CC-BY-NC 4.0** (fine for academic work -- state it if you redistribute).

Both on-disk layouts are accepted by `core.multimae.canonicalise_vit_state_dict`:
the official MCG-NJU/MAE key layout, and the HuggingFace `transformers` one
(`videomae.encoder.layer.N.layernorm_before.*` etc., where Q/K/V are re-fused
into one `attn.qkv` with MAE's zero-key-bias convention). `--inflate_rgb_patch 0`
skips the tokenizer transfer entirely (encoder blocks only) as an ablation.
Because a 5-D source kernel must equal the model's `tubelet`, a mismatch raises
instead of being quietly counted as a shape mismatch.

There is **no ViT-S MAE release** and no VideoMAE ViT-S/ViT-H on the Hub, so
only `base`/`large` have a built-in Stage-1 source (the DeiT-Small and MAE
ViT-Huge sources were dropped with the `small`/`huge` weight variants: the
project plan no longer uses those geometries). The `small`/`huge` *model*
entrypoints still exist for an explicit checkpoint path, and `timm:<model_id>`
reaches any other backbone.
For Stage 2 the model name sets the ViT geometry, so `model: project_multimae_base` pairs with `pretrained_encoder: base`
(`enc_embed_dim`/`enc_depth`/`enc_num_heads` remain explicit overrides). The
multimodal/probe loaders raise on mismatch instead of silently loading nothing.

```bash
python runners/run_download_weights.py --list        # variants + geometry
python runners/run_download_weights.py base          # pre-fetch (login node!)
python runners/run_download_weights.py mae:base      # plain MAE alternative
python runners/run_finetune.py --finetune large ...  # downloads on demand
python runners/run_au_probe.py -c configs/finetune/au_local_pretrained.yaml \
    --finetune base                                  # C1 Stage-1 (768-d)
```

Downloads stream to a `.part` file (HTTP-Range resume), are renamed atomically
and cached, so a second run is a no-op; under DDP only rank 0 downloads. On the
HPC the compute nodes have no internet - pre-fetch on a login node and keep
`$INITIAL_MODELS_DIR` on the shared filesystem.

**Canonical data layout.** The raw BP4D layout (`2D+3D/`, `Thermal/`,
`Physiology/*.txt`) is converted once into the per-session layout consumed by
`data/paired_dataset.py` using `data/prepare_bp4d.py`:
`<session>/{rgb/, tir.wmv, signals.csv (time,bp,resp,eda), meta.json}`.
Channel mapping: `bp <- BP_mmHg.txt`, `resp <- Resp_Volts.txt`,
`eda <- EDA_microsiemens.txt`; raw physiology `.txt` is anti-alias resampled to
`--fs` (default 100 Hz; raw rate `--phys_fs`, default 1000 Hz, or auto with `0`).

**Quick local smoke / data inspection (few sessions, data only):**

```bash
source scripts/env_local.sh
bash scripts/local/prepare_smoke.sh 2        # convert first 2 raw sessions
bash scripts/local/inspect_smoke.sh          # stats JSON + a PNG per clip
```

`scripts/local/inspect_smoke.sh` runs `runners/run_inspect_data.py`, which
writes `output/inspect_data/inspect_summary.json` and, per split, a PNG for
**every** clip of train and val into `output/inspect_data/figures/<split>/`
(one PNG = RGB + TIR middle frame, aligned signal windows vs. the dataset
target, plus `session/k<n>/split` metadata). The inspected scope is bounded by
the caps below; override them through environment variables (short names or
`INSPECT_*` aliases), e.g.:

```bash
# 2 s windows, one clip per session, across 6 sessions -> every clip gets a PNG
MAX_SESSIONS=6 MAX_CLIPS=2 CLIP_DURATION=2 bash scripts/local/inspect_smoke.sh
```

| Var (local)       | Alias (HPC)               | Meaning                                        | Default |
| ----------------- | ------------------------- | ---------------------------------------------- | ------- |
| `MAX_SESSIONS`  | `INSPECT_SESSIONS`      | sessions decoded (decode budget)               | 3       |
| `MAX_CLIPS`     | `INSPECT_MAX_CLIPS`     | max windows taken from each session            | none    |
| `CLIP_DURATION` | `INSPECT_CLIP_DURATION` | window length in s (`seq_len = fs*duration`) | 10 s    |
| `MAX_ENTRIES`   | `INSPECT_ENTRIES`       | optional per-split cap (`0` = no cap / all)  | 0       |
| `INPUT_SIZE`    | (HPC fixed to 64)         | frame short-side resize/crop                   | 64      |

Manual equivalent (runner defaults: 3 sessions, no per-session cap, no
`max_entries` limit, 10 s windows, `--plot`):

```bash
python runners/run_inspect_data.py --clip_duration 10 --plot
```

The same caps limit smoke **training** runs too: in `data/paired_dataset.py`
`max_sessions` caps decoding up front, `max_clips` caps the windows per
session, and `max_entries` caps the global dataset total (see the YAML under
`configs/pretrain/`).

**Raw physiology inspection (`Physiology/*.txt`, original sample rate):**

`runners/run_inspect_physio.py` looks at the RAW physiology channels *before*
the canonical conversion -- no alignment, no filtering, no resampling -- and
reports length (samples + duration), min/max (+ mean/median/std/percentiles) and
the estimated frequency, with one figure + JSON per channel:

```bash
source scripts/env_local.sh
# one session, every channel (BP | Resp | EDA | all, case-ctinsensitive)
python runners/run_inspect_physio.py --subject F001 --task T1 --channel all
python runners/run_inspect_physio.py --subject F001,F002 --task T1,T2 --channel Resp,EDA
python runners/run_inspect_physio.py --list          # what is on disk?
bash scripts/local/inspect_physio.sh                 # SUBJECT/TASK/CHANNEL env-style
```

`--list` checks PRESENCE, not just the directory tree: a session dir that holds
no channel file is marked `[empty]`, and a tree where nothing holds a channel
file raises a warning instead of listing sessions. A half-transferred raw root
(Lichtenberg: 1400 empty `Physiology/<subj>/<task>/` dirs, 0 files) otherwise
looks like a complete dataset until every single session is inspected and fails.

Output goes to `output/inspect_data/<subject>_<task>/`: `<Channel>.png` (the
full native-rate trace, a 10 s zoom, and the amplitude distribution),
`<Channel>.json`, `physio_summary.json` and an `overview.png` stacking the
channels. The per-channel figure shows the raw signal only -- no band-passed
overlay and no Welch PSD panel, since those describe how the frequency estimate
was derived and are reported through `frequency` in the JSON instead. All three
figures are deliberately numbers-free; every statistic is in the JSON files and
printed to stdout. `physio_index.json` indexes every inspected session. This
writes *alongside* `run_inspect_data.py`; the two do not share file names (the
session folders are only ever written by this runner, and
`run_inspect_data.py --force` wipes only `figures/` + `inspect_summary.json`).

**A channel pulls in its whole family of raw files.** Inspecting `bp` or `resp`
(explicitly or via `all`) also reads the session's other related raw files and
draws them together in `<FAMILY>_overview.png` (full session at the original
rate, 10 s zoom, plus a third panel). They are merged into the channel's JSON as
`family_components[]`, with `family_files[]` listing them and a warning if one is
missing:

| Channel  | Figure                | Raw files                                                                                  |
| -------- | --------------------- | ------------------------------------------------------------------------------------------ |
| `bp`   | `BP_overview.png`   | `BP_mmHg.txt`, `LA Systolic BP_mmHg.txt`, `LA Mean BP_mmHg.txt`, `BP Dia_mmHg.txt` |
| `resp` | `Resp_overview.png` | `Resp_Volts.txt`, `Respiration Rate_BPM.txt`                                           |

| Raw file                                   | Kind                                                | Unit    |
| ------------------------------------------ | --------------------------------------------------- | ------- |
| `BP_mmHg.txt`, `Resp_Volts.txt`        | continuous**waveform** (changes every sample) | mmHg, V |
| `LA Systolic` / `LA Mean` / `BP Dia` | vendor-derived**per-beat** value, step-held   | mmHg    |
| `Respiration Rate_BPM.txt`               | vendor-derived, step-held (updated every few s)     | BPM     |

The waveform/step-held distinction is real and measured: on `F001_T1` the
systolic series holds `114.433` for >300 samples (about a whole beat) before
jumping to `113.901`, so it is a beat-resolution envelope, not a second
waveform. For the BP family, overlaying them is what makes the figure useful --
the pulse waveform oscillates inside the systolic/diastolic envelope, and the
third panel is a box comparison of all four (box = IQR, whiskers = min/max).

**Units can differ inside a family, and the figure respects that.** The Resp
family pairs volts with breaths/min, so `Resp_overview.png` puts the rate on a
right-hand axis instead of drawing two scales on one, and its third panel is the
waveform's amplitude distribution rather than a cross-series box plot (which
would compare unlike quantities). The console prints each component's unit.

**Consequence for the numbers.** A spectral estimate is only meaningful for the
waveform. For the step-held series it is actively misleading: the staircase
spectrum is dominated by its flat plateaus, so the in-band Welch peak pins to
the LOWER edge of the search band -- measured on `F001_T1`, all three BP series
reported `0.6 Hz = 36.0/min` against a ~90/min pulse. They therefore get a
**`step_update_rate`** (count of value changes = the beat rate, plus
`update_interval_s_median` = how long the vendor held each value) instead of a
`frequency` block, and each record carries `derived`/`kind` so the two kinds are
never confused. The console marks them `(step-held)`.

**Dropout found in the derived files:** `LA Systolic` and `LA Mean` each contain
a 26-sample run of exactly `0.0` at line 29705 of `F001_T1` (a blood pressure of
0 mmHg is not a measurement). `stats.frac_exactly_zero` / `n_zero_samples`
reports this and it raises a warning. Neither file is read by
`prepare_bp4d.py`, so these zeros do **not** reach the canonical `signals.csv`.

Channels map through the same `data.prepare_bp4d.CHANNEL_FILES` table the
converter uses, so the two can never disagree: `BP <- BP_mmHg.txt` [mmHg],
`Resp <- Resp_Volts.txt` [V], `EDA <- EDA_microsiemens.txt` [uS]. The derived
`Pulse Rate_BPM.txt` / `Respiration Rate_BPM.txt` are deliberately excluded from
that mapping.

**`HR` -- the vendor heart rate (`Pulse Rate_BPM.txt` -> `HR.png`).** An extra,
inspector-only channel: `--channel HR` (aliases `heart_rate`, `pulse_rate`,
`bpm`) writes `HR.png` / `HR.json` for the session's `Pulse Rate_BPM.txt`, and
`--channel all` includes it. It is a per-beat series **held constant between
beats**, not a waveform, so it is analysed as a step signal: no spectral
estimate, but `step_update_rate` (value-change count, i.e. the beat rate, plus
`update_interval_s_median` = how long the vendor held each value) and the plain
min/max/mean BPM in `stats`.

`HR` is deliberately **not** added to `data.prepare_bp4d.CHANNEL_FILES`. That
table drives the canonical `signals.csv` in two ways -- its columns
(`['time'] + list(CHANNEL_FILES)`) and its fail-the-session behaviour when a
channel file is missing -- so adding HR there would silently add an `hr` column
to every `signals.csv` and break any session without a `Pulse Rate_BPM.txt`.
The needle lives in `CHANNEL_META['hr']['raw_file']` instead, resolved through
`channel_raw_file()`.

Sample rate: the raw files carry no time column, so `--phys_fs` (default 1000 Hz,
the BP4D nominal rate) sets the time axis, or `--phys_fs 0` estimates it per
session from the RGB frame count and `--fps_rgb` (`n_frames / fps`, listing only
-- no JPEG decode). The ORIGINAL samples are kept either way.

Frequency: `welch_peak_hz` is the headline estimate (in-band PSD maximum,
parabolically refined); `autocorr_hz` (autocorrelation peak) and
`cycle_count_per_min` (peak counting) are cross-checks. The runner DIAGNOSES
its own estimates rather than hiding failures: a peak sitting on the search-band
edge, a >25% spread between estimators, or a channel railed at a hard limit
(e.g. respiration pinned at exactly -10.0000 V for 10-45% of several sessions)
is reported as a warning in the JSON and printed. When the Welch peak is
flagged, `preferred_estimate` names the peak-counting rate as the value to use.
`--band lo,hi` (or `CHANNEL:lo,hi`) overrides the search band when a session is
flagged -- the cycle-count reference deliberately keeps the channel's own
physiological band, so it stays stable while the spectral search moves.
`--no-plot` writes JSON only.

Known caveats, measured on the 19 raw sessions: respiration estimates land at
14-26 /min where the cycle count and the raw waveform agree, but `T1`/`T2`
recordings are only 9-25 s long (vs ~65 s for `T6`-`T8`), which widens the
spectral bins, and the vendor `Respiration Rate_BPM.txt` is a coarse STEP
function (it holds one value for seconds at a time) that can read ~half the
cycle count -- do not use it as ground truth. `EDA` gets no frequency claim: it
is reported with a plain 20 s moving-average tonic/phasic split plus the LSB
step (an attempt to report SCR events/min was removed -- without NeuroKit2 it
produced meaningless 2-20 /min rates on traces whose whole range was ADC noise).

**Raw AU occurrence coding (`AUCoding/AU_OCC/*.csv`) -- corpus statistics:**

`runners/run_inspect_au.py` (+ `scripts/local/inspect_au.sh`) reads the raw
BP4D+ FACS **occurrence** csvs straight out of the raw tree -- the same files
`data/au_dataset.py` feeds to the AU probe -- and summarises the WHOLE corpus
instead of one session. Figures + JSON land in
`output/inspect_data/AUCoding/` (not in a `<subject>_<task>/` folder, because
the unit of analysis here is the annotation corpus):

```bash
source scripts/env_local.sh
python runners/run_inspect_au.py                       # 560 files, all figures
python runners/run_inspect_au.py --list                # what is on disk?
python runners/run_inspect_au.py --task T7 --no-plot   # one task, JSON only
python runners/run_inspect_au.py --au_list 6,7,10,12,14
TASK=T7 bash scripts/local/inspect_au.sh               # env-style overrides
```

| Figure                      | What it answers                                           |
| --------------------------- | --------------------------------------------------------- |
| `AU_presence.png`         | how often every AU fires (9 excluded) + its missing rate  |
| `AU_cooccurrence.png`     | pairwise Jaccard, all AUs and the default subset          |
| `AU_active_count.png`     | how many AUs are active in a frame (how degenerate it is) |
| `AU_segments.png`         | number and duration of activation segments per AU         |
| `AU_session_spread.png`   | per-session spread of the subset AUs, and by task         |
| `AU_task_presence.png`    | subset occurrence rate split by T1/T6/T7/T8               |
| `AU_session_timeline.png` | one session's coding as a raster (`--timeline_session`) |
| `AU_session_coverage.png` | where the coded blocks sit in the videos                  |

`AU_summary.json` carries every number behind the figures (per-AU table,
co-occurrence matrices, per-task and per-session rows); `AU_index.json` lists
what was inspected. `--au_list` selects which AUs are treated as the target
subset (highlighted in the figures, broken down per task/session) and defaults
to `data.au_dataset.DEFAULT_AU_LIST`; `--au_list all` uses all 34.

**Format and the AU99 trap.** A csv has a header row (a frame-column label, then
35 AU ids) and one row per coded video frame: `frame,au1,au2,...` with values
`0` absent / `1` present / `9` missing, and frame `f` mapping to canonical rgb
frame `f - 1`. **AU99 is not an AU -- it is a per-frame "coding unreliable"
flag**: the 944 frames of the corpus that contain a `9` are exactly the frames
with `AU99 == 1` (923 of them have *all* 34 real AU columns at `9`, 21 have a
partial `9`). The runner therefore drops AU99 and excludes those frames from
every occurrence rate, while `n_missing` still counts the `9` codes -- over ALL
frames, because that is where they live. The upshot: corpus missingness is a
whole-frame property (~0.48% uniformly on every AU), not a per-AU one.

Measured on all 560 local files (140 subjects x {T1,T6,T7,T8}, 82 F + 58 M,
197,875 coded rows = 131.9 min):

* **AU7 66.3%, AU10 64.8%, AU14 60.1%, AU12 57.9%, AU6 49.8%, AU11 41.1%,
  AU16 32.9%, AU23 16.7%, AU20 14.8%, AU17 13.0%, AU15 10.7%, AU1 9.7%,
  AU2 8.2%, AU4 5.8%, AU24 3.9%** (share of frames with the AU active);
  AU29/AU35/AU36 never fire at all, AU13/27/29/33-37 stay under 0.2%.
* the default 12-AU subset averages **3.67 active AUs per frame (30.6% of the
  subset)**, of 4.71 active of all 34, and 84.9% of frames have at least one AU.
* the expressive-block AUs co-fire: Jaccard **0.58-0.79** inside
  {6,7,10,12,14} and 0.53 for AU1-AU2, while cross-block pairs stay near 0.1.
* segments: AU16/AU23/AU17 are short-and-frequent (median 0.44-0.60 s at
  1200-1800 segments) while AU7/AU10/AU12 hold for ~3.6-3.9 s at a time
  (longest single activations 80-110 s).
* per-session activity is task-dependent: mean active subset AUs per frame
  **T1 4.38, T6 4.64, T7 4.10, T8 1.74** -- T8 is by far the calmest task.
* every session is ONE contiguous coding block of 83-700 frames (median 376),
  but it starts anywhere from frame 1 to frame 2248. The tasks differ: T1
  starts early (median frame 150), T6/T7 are spread over frames 200-1400, and
  **~73 of the 140 T8 codings start at frame ~1125**. Clip sampling must respect
  both video boundaries.

Caveat that matters for the probe: AU coding covers the MOST EXPRESSIVE part of
a task, so the frequent AUs are near-constant ON there. A degenerate
all-positive head therefore already scores a high macro-F1 on an AU-coded val
split -- keep reporting the trivial baseline next to the probe's F1 (the
`active/frame` and spread numbers above quantify exactly how degenerate it is).

**Lichtenberg HPC (sbatch):**

```bash
mkdir -p logs
sbatch scripts/hpc/submit_prepare.sbatch    # convert raw BP4D once (CPU)
sbatch scripts/hpc/submit_inspect.sbatch    # data smoke on 1 GPU
sbatch scripts/hpc/submit_inspect_physio.sbatch   # raw 1-D physiology (CPU, no GPU)
TARGET=bp sbatch scripts/hpc/submit_waveform.sbatch   # (later) multi-GPU Stage-3
```

Three cluster rules the `submit_*.sbatch` files already encode (verified with
`sbatch --test-only` on `lcluster`):

- **Submit from `code/`.** `sbatch` COPIES the script into the spool dir and runs
  that copy, so `$0` is `/var/spool/slurmd/job<id>/slurm_script` -- not the repo
  path. The scripts locate `scripts/env_hpc.sh` through `$SLURM_SUBMIT_DIR`, and
  abort with a message if it is not there.
- **`--mem-per-cpu`, never `--mem`** (the LUA `job_submit` plugin rejects `--mem`).
  Total memory is `--mem-per-cpu * cpus-per-task * ntasks`, and `DefMemPerCPU`
  is 3800 MB, so omitting the line is also fine on CPU partitions.
- **Partitions**: `deflt_short` (30 min) / `deflt` (1 day) / `long` (7 days) for
  CPU work, `acc_short` / `acc` / `acc_long` for GPU work -- there is no `gpu`
  partition on Lichtenberg II. `--account` may be omitted; the plugin bills your
  default project either way.

Multi-GPU jobs run one task per GPU through `srun`; `utils/dist.py` initialises
DDP from the SLURM environment (`SLURM_PROCID`/`SLURM_NTASKS`/`SLURM_LOCALID`)
and also works under `torchrun` for single-node local multi-GPU testing.
