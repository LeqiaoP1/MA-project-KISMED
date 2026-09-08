# Thesis codebase — scaffold

Target (ImplementationPlan.md): contactless **2D RGB + Thermal-IR video →
1D BVP / RESP / EDA waveform** recovery via a 3-stage progressive pipeline
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
# (0) data: convert raw sessions once, then inspect a few
python data/prepare_bp4d.py --raw_root ../data/raw/BP4D --out_root ../data/processed/bp4d_canonical --limit_sessions 8
python runners/run_inspect_data.py --data_path ../data/processed/bp4d_canonical --clip_duration 2 --input_size 64 --plot

# (1) Stage-2 multimodal masked pre-training (local milestone)
#   from-scratch small slice ............... configs/pretrain/stage2_local.yaml
#   MAE ViT-Base encoder inheritance ........ configs/pretrain/stage2_local_pretrained.yaml
python runners/run_pretrain.py -c configs/pretrain/stage2_local_pretrained.yaml

# (2) Stage-3 waveform fine-tuning per branch (needs a Stage-2 encoder ckpt)
python runners/run_waveform.py -c configs/finetune/bvp.yaml
python runners/run_waveform.py -c configs/finetune/resp.yaml
python runners/run_waveform.py -c configs/finetune/eda.yaml

# (3) offline multi-tier evaluation of saved predictions
python runners/run_evaluate.py --pred_path out.npy --target_path gt.npy \
    --fs 100 --tier 1,2,3 --waveform bvp

# (4) OPTIONAL ADD-ON diagnostic: AU-occurrence probe on the Stage-2 encoder
python runners/run_au_probe.py -c configs/finetune/au_local.yaml

# multi-GPU (HPC)
bash scripts/project/pretrain.sh
bash scripts/project/finetune.sh
```

### Stage-2 pre-training loss (weighted per-modality masked MSE)

The Stage-2 (`core/multimae.py::MultiModalMAE`) reconstruction loss is a
**masked MSE**, computed **independently per modality over its masked positions
only** (`mask == 1` ⇒ to reconstruct; `core/criterion.py::MaskedMSELoss`), then
combined as a **weighted sum** over the streams:

```math
L = Σ_s λ_s·MSE_s ,      λ_RGB = λ_TIR = 1.0 ,      λ_physio = signal_weight (default 0.5)
```

Visual streams (RGB/TIR) keep weight 1.0; every physio (1-D) stream uses
`--signal_weight` (default **0.5**). Tune the physio weight in ~[0.5, 1.0]
from the **normalized variance of the physio stream's masked target tokens**:
high variance ⇒ lean 0.5 so the 1-D signal does not dominate the loss gradient;
low variance ⇒ lean 1.0 so it is not ignored. `--loss_weights` is a full
per-stream override (one comma value per `--streams` modality, in order; empty
⇒ policy above). `MultiModalMAE.forward` returns `losses_mse` (raw per-modality
masked MSE) and `losses` (weighted contributions); `loss = Σ losses`.

### Stage-2 streams — flexible modality contract

The pretraining modalities are configured with `--streams` (or `streams:` in
the YAML under `configs/pretrain/`) as a comma list. The **Stage-2 contract**
requires at least TWO streams: **≥1 video** (`rgb` and/or `tir`) **plus ≥1
physiological 1-D signal** (`bvp`, `resp`, `eda` — the waveform later
regressed in Stage 3). Video-only, signal-only, single-modality, empty or
unknown lists are rejected consistently in `MultiModalMAE.__init__`,
`build_pretraining_model` (`core/multimae.py`) and
`PairedPretrainDataset`/`build_pretraining_dataset` (`data/`).

`PairedPretrainDataset` serves **exactly** the requested streams: e.g.
`streams: rgb,bvp` returns only `{'rgb','bvp'}`, while the default five-stream
configs return all of `{'rgb','tir','bvp','resp','eda'}`. So trimming/ablating
modalities (while keeping the ≥1 video + ≥1 physio rule) is a YAML-only
change. Per-stream mask ratios: visual `mask_ratio_rgb/tir` 50–75 %, signals
`mask_ratio_bvp/resp/eda` 90 %+.

Implemented and run so far (see the thesis-plan table below): canonical BP4D
conversion, the aligned `PairedSessionDataset` (+ overlapping windows via
`clip_stride`), the multimodal masked autoencoder `core/multimae.py`
(asymmetric per-stream masks + the weighted per-modality masked MSE above), its pretrain loader
`PairedPretrainDataset` and runner `run_pretrain.py` (explicit `--lr` +
step-level warmup/cosine schedule via `utils/lr_sched.py`), MAE ViT-Base
encoder inheritance (`load_pretrained_encoder`), and the Stage-3 waveform
scaffold (`run_waveform.py`, `WaveformJointLoss`, `evaluation/`).

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

| Plan stage                                                                             | Supported here                                                                                                                                                                                                                                                                                                                                                      | Still to port (thesis work)                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stage 1 — ImageNet init of ViT-Base encoder                                           | `core/model.py` entrypoints (`project_vit_base_patch16_224`)                                                                                                                                                                                                                                                                                                    | official ImageNet-1K timm classifier converter (MAE ViT-Base inheritance already implemented:`core/multimae.py::load_pretrained_encoder`, 2-D -> 3-D rgb tubelet inflation via `--pretrained_encoder`)                                                                                        |
| Stage 2 — multimodal masked pre-training on BP4D+ (RGB/TIR 50-75%, BVP/RESP/EDA 90%+) | `core/input_adapters.py` (`SignalInputAdapter`), `data/masking_generator.py` (`MultiModalMaskingGenerator` asymmetric), `core/criterion.py` (masked L1/MSE), config template `configs/pretrain/stage2_multimodal.yaml`; implemented local milestone: `core/multimae.py` (`MultiModalMAE`) + `PairedPretrainDataset` + `runners/run_pretrain.py` | separate deeper decoders; full-data HPC run at larger 224 geometry. Local five-stream milestone (rgb+tir+bvp+resp+eda) is implemented & run:`core/multimae.py` (`MultiModalMAE`), `data/paired_dataset.PairedPretrainDataset`, `runners/run_pretrain.py`, `configs/pretrain/stage2_local{,_pretrained}.yaml` |
| Stage 3 — three branches BVP, RESP & EDA, unified spatio-temporal-spectral loss       | `core/waveform_losses.py` (`WaveformJointLoss`: L1 + Pearson + MR-STFT; 64/128/256 for BVP/RESP, 256/512/1024 for EDA), regression head (`ProjectViT(output_len=...)`, baseline CLS->seq), `runners/run_waveform.py`, configs `configs/finetune/{bvp,resp,eda}.yaml`                                                                                      | lightweight conv decoder over all tokens for finer temporal resolution; session-level whole-waveform reconstruction/stitching (offline inference, not yet implemented)                                                                                                                            |
| Evaluation — Tier 1/2/3 post-processing                                               | `evaluation/metrics.py` (MAE/RMSE/Pearson; Welch PSD), `evaluation/clinical.py` (NeuroKit2 RMSSD/pNN50/MedianNN/ShanEn), `runners/run_evaluate.py`                                                                                                                                                                                                            | —                                                                                                                                                                                                                                                                                                |

```bash
# Stage 3 example (needs a Stage-2 encoder ckpt)
python runners/run_waveform.py -c configs/finetune/bvp.yaml
python runners/run_waveform.py -c configs/finetune/resp.yaml
python runners/run_waveform.py -c configs/finetune/eda.yaml
# Offline post-processing on saved predictions
python runners/run_evaluate.py --pred_path out.npy --target_path gt.npy \
    --fs 100 --tier 1,2,3 --waveform bvp
```

### Whole-session waveform reconstruction (planned)

Stage-3 *training* is per clip (`video window -> waveform window`). The final
deliverable — the continuous 1-D waveform of a whole session (subject/task) —
is assembled **after** training by sliding the window over the session
(`clip_duration` + `clip_stride`) and overlap-adding (stitching) the per-window
predictions, then evaluated offline with Tier 1 (time), Tier 2 (spectral) and
Tier 3 (clinical/NeuroKit2, BVP/HRV only). This stitching step is **not yet
implemented** (no training involved).

### Recorded data layout (asymmetric RGB jpg-seq + TIR .wmv)

```
<data_path>/<session>/
    rgb/            # ordered jpg frames  (25 fps nominal)
    tir.wmv         # single WMV ~60 s    (25 fps nominal)
    signals.csv     # header: [time,] bvp, resp, eda   at fs Hz
```

Readers (`data/video_io.py`), temporal registration (`data/alignment.py`) and
the synchronised `PairedSessionDataset` (`data/paired_dataset.py`) handle this
layout; select it with `data_set: bp4d+` (alias `paired`). Per-session TIR fps
and signal sample rates are probed at runtime; both modalities are read on one
common time grid and the target waveform is resampled to `seq_len`. Set
data-set params in the YAML: `fs`, `fps`, `clip_duration`, `clip_stride`, `seq_len`,
`input_size`, `rgb_dir`, `tir_file`, `signals_file`, `train_ratio`.

Optional dev/quick-run caps (accepted by every training runner and by the
inspect runner): `max_sessions` bounds the number of decoded sessions up
front, `max_clips` bounds the number of windows taken from *each* session,
and `max_entries` bounds the global total number of clips in the dataset.
`clip_stride` (default `0` = non-overlapping) may be set to a value smaller
than `clip_duration` to generate overlapping windows and thus more samples per
session; combined with `max_clips` it keeps only the earliest windows of each
session.

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
  * C0 random init (`--finetune ''`), C1 Stage-1 MAE/ImageNet
    (`../models/mae_pretrain_vit_base.pth`), **C2 Stage-2** (headline).
* **Configurable AU set.** `--au_list` (explicit, string or YAML list) **or**
  `--au_freq_topk N` (top-N by presence rate over the full AU corpus;
  `5` ⇒ AU6/7/10/12/14); default = the BP4D 12-AU subset. The head width,
  dataset label columns and per-AU table all follow the resolved list.
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
    --finetune ../models/mae_pretrain_vit_base.pth          # C1 Stage-1 (768-d)
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
`NUM_WORKERS`. CLI flags always take precedence over env/YAML.

**Canonical data layout.** The raw BP4D layout (`2D+3D/`, `Thermal/`,
`Physiology/*.txt`) is converted once into the per-session layout consumed by
`data/paired_dataset.py` using `data/prepare_bp4d.py`:
`<session>/{rgb/, tir.wmv, signals.csv (time,bvp,resp,eda), meta.json}`.
Channel mapping: `bvp <- BP_mmHg.txt`, `resp <- Resp_Volts.txt`,
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

**Lichtenberg HPC (sbatch):**

```bash
mkdir -p logs
sbatch scripts/hpc/submit_prepare.sbatch    # convert raw BP4D once (CPU)
sbatch scripts/hpc/submit_inspect.sbatch    # data smoke on 1 GPU
TARGET=bvp sbatch scripts/hpc/submit_waveform.sbatch   # (later) multi-GPU Stage-3
```

Multi-GPU jobs run one task per GPU through `srun`; `utils/dist.py` initialises
DDP from the SLURM environment (`SLURM_PROCID`/`SLURM_NTASKS`/`SLURM_LOCALID`)
and also works under `torchrun` for single-node local multi-GPU testing.
