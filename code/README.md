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

# multi-GPU (HPC)
bash scripts/project/pretrain.sh
bash scripts/project/finetune.sh
```

Implemented and run so far (see the thesis-plan table below): canonical BP4D
conversion, the aligned `PairedSessionDataset` (+ overlapping windows via
`clip_stride`), the multimodal masked autoencoder `core/multimae.py`
(asymmetric per-stream masks + per-stream masked L1), its pretrain loader
`PairedPretrainDataset` and runner `run_pretrain.py` (explicit `--lr` +
step-level warmup/cosine schedule via `utils/lr_sched.py`), MAE ViT-Base
encoder inheritance (`load_pretrained_encoder`), and the Stage-3 waveform
scaffold (`run_waveform.py`, `WaveformJointLoss`, `evaluation/`).

Not yet implemented: RESP/EDA streams in Stage 2, a finer Stage-3 temporal
decoder, and the *session-level* reconstruction/stitching that turns per-clip
predictions into one continuous whole-session waveform (a planned offline
inference step — per-clip training is clip-level by design).

## Thesis plan alignment (ImplementationPlan.md)

| Plan stage                                                                             | Supported here                                                                                                                                                                                                                                                                 | Still to port (thesis work)                                                                                                                                                      |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stage 1 — ImageNet init of ViT-Base encoder                                           | `core/model.py` entrypoints (`project_vit_base_patch16_224`)                                                                                                                                                                                                               | official ImageNet-1K timm classifier converter (MAE ViT-Base inheritance already implemented: `core/multimae.py::load_pretrained_encoder`, 2-D -> 3-D rgb tubelet inflation via `--pretrained_encoder`)                                         |
| Stage 2 — multimodal masked pre-training on BP4D+ (RGB/TIR 50-75%, BVP/RESP/EDA 90%+) | `core/input_adapters.py` (`SignalInputAdapter`), `data/masking_generator.py` (`MultiModalMaskingGenerator` asymmetric), `core/criterion.py` (masked L1/MSE), config template `configs/pretrain/stage2_multimodal.yaml`; implemented local milestone: `core/multimae.py` (`MultiModalMAE`) + `PairedPretrainDataset` + `runners/run_pretrain.py`                                             | RESP/EDA streams, separate deeper decoders, full-data HPC run at 224. Local rgb+tir+bvp milestone is implemented & run: `core/multimae.py` (`MultiModalMAE`), `data/paired_dataset.PairedPretrainDataset`, `runners/run_pretrain.py`, `configs/pretrain/stage2_local{,_pretrained}.yaml` |
| Stage 3 — three branches BVP, RESP & EDA, unified spatio-temporal-spectral loss       | `core/waveform_losses.py` (`WaveformJointLoss`: L1 + Pearson + MR-STFT; 64/128/256 for BVP/RESP, 256/512/1024 for EDA), regression head (`ProjectViT(output_len=...)`, baseline CLS->seq), `runners/run_waveform.py`, configs `configs/finetune/{bvp,resp,eda}.yaml` | lightweight conv decoder over all tokens for finer temporal resolution; session-level whole-waveform reconstruction/stitching (offline inference, not yet implemented)                                                                                                           |
| Evaluation — Tier 1/2/3 post-processing                                               | `evaluation/metrics.py` (MAE/RMSE/Pearson; Welch PSD), `evaluation/clinical.py` (NeuroKit2 RMSSD/pNN50/MedianNN/ShanEn), `runners/run_evaluate.py`                                                                                                                       | —                                                                                                                                                                               |

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

| Var (local)         | Alias (HPC)             | Meaning                                      | Default |
| ------------------- | ----------------------- | -------------------------------------------- | ------- |
| `MAX_SESSIONS`      | `INSPECT_SESSIONS`      | sessions decoded (decode budget)             | 3       |
| `MAX_CLIPS`         | `INSPECT_MAX_CLIPS`     | max windows taken from each session          | none    |
| `CLIP_DURATION`     | `INSPECT_CLIP_DURATION` | window length in s (`seq_len = fs*duration`) | 10 s    |
| `MAX_ENTRIES`       | `INSPECT_ENTRIES`       | optional per-split cap (`0` = no cap / all)  | 0       |
| `INPUT_SIZE`        | (HPC fixed to 64)       | frame short-side resize/crop                 | 64      |

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
