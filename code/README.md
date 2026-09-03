# Thesis codebase — scaffold

Target (docs/ImplementationPlan.md): contactless **2D RGB + Thermal-IR video →
1D BVP / RESP waveform** recovery via a 3-stage progressive pipeline
(ImageNet init -> multimodal masked pre-training on BP4D+ -> two task-specific
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

| This folder | Port from | Notes |
|---|---|---|
| `core/registry.py` | `tmp/MultiMAE/utils/registry.py` | timm-style `@register_model` |
| `core/blocks.py` | `tmp/MultiMAE/multimae/multimae_utils.py` | Block / Attention / DropPath / trunc_normal_ |
| `core/input_adapters.py` | `tmp/MultiMAE/multimae/input_adapters.py` | per-modality input adapters |
| `core/output_adapters.py` | `tmp/MultiMAE/multimae/output_adapters.py` | reconstruction heads (Spatial/DPT/ConvNeXt/...) |
| `core/criterion.py` | `tmp/MultiMAE/multimae/criterion.py` | masked MSE/L1/CE |
| `core/model.py` | `tmp/MultiMAE/multimae/multimae.py` + `tmp/videomae/modeling_finetune.py` | architecture + registered entrypoints |
| `data/datasets.py` | `tmp/MultiMAE/utils/datasets.py`, `tmp/videomae/datasets.py` | dataset builders (modality specific) |
| `data/masking_generator.py` | `tmp/videomae/masking_generator.py` | Tube / Random masking |
| `utils/*` | `tmp/MultiMAE/utils/*`, `tmp/videomae/utils.py` | dist, logging, checkpoint, optim, EMA, scaler |

## Conventions

- **Config**: YAML under `configs/`; run scripts accept `-c/--config` and merge it
  as argparse defaults; CLI flags override YAML (see `runners/run_pretrain.py`).
- **Model registration**: decorate factory functions with `@register_model` from
  `core.registry` and build them via `models.build.create_model`.
- **Engines**: keep training loops in `engines/`, keep `runners/` scripts thin.
- **Imports**: run scripts assume the CWD is `code/` (top-level packages
  `core`, `models`, `data`, `engines`, `utils`), same as MultiMAE/VideoMAE.
- **License headers**: keep the BSD provenance header when porting code blocks.

## Running (once datasets are implemented)

```bash
cd code
# single GPU (nothing implemented yet -> implement data/datasets.py first)
python runners/run_pretrain.py -c configs/pretrain/example.yaml
python runners/run_finetune.py -c configs/finetune/example.yaml
# multi GPU
bash scripts/project/pretrain.sh
bash scripts/project/finetune.sh
```

The supervised data path is implemented for the recorded layout via
`data/paired_dataset.py` (see "Recorded data layout" below). What still stops
the runners end-to-end is the masked Stage-2 pre-training loader plus the
spatio-temporal video model front-end (both documented as thesis work).

## Thesis plan alignment (docs/ImplementationPlan.md)

| Plan stage | Supported here | Still to port (thesis work) |
|---|---|---|
| Stage 1 — ImageNet init of ViT-Base encoder | `core/model.py` entrypoints (`project_vit_base_patch16_224`) | load official ImageNet-1K weights (timm / checkpoint converter); make patch-embed spatiotemporal (VideoMAE 3D tubelet) for RGB+TIR video |
| Stage 2 — multimodal masked pre-training on BP4D+ (visual 50-75%, signals 90%+) | `core/input_adapters.py` (`SignalInputAdapter`), `data/masking_generator.py` (`MultiModalMaskingGenerator` asymmetric), `core/criterion.py` (masked L1/MSE), config template `configs/pretrain/stage2_multimodal.yaml` | multimodal encoder+decoder with in-forward masking (port `tmp/MultiMAE/multimae/`) and the *masked* pre-training loader built on `data/paired_dataset.PairedSessionDataset` |
| Stage 3 — two branches BVP & RESP, unified spatio-temporal-spectral loss | `core/waveform_losses.py` (`WaveformJointLoss`: L1 + Pearson + MR-STFT 64/128/256), regression head (`ProjectViT(output_len=...)`, baseline CLS->seq), `runners/run_waveform.py`, configs `configs/finetune/{bvp,resp}.yaml` | lightweight conv decoder over all tokens for finer temporal resolution |
| Evaluation — Tier 1/2/3 post-processing | `evaluation/metrics.py` (MAE/RMSE/Pearson; Welch PSD), `evaluation/clinical.py` (NeuroKit2 RMSSD/pNN50/MedianNN/ShanEn), `runners/run_evaluate.py` | — |

```bash
# Stage 3 example (after implementing data/datasets.py + a Stage-2 ckpt)
python runners/run_waveform.py -c configs/finetune/bvp.yaml
python runners/run_waveform.py -c configs/finetune/resp.yaml
# Offline post-processing on saved predictions
python runners/run_evaluate.py --pred_path out.npy --target_path gt.npy \
    --fs 100 --tier 1,2,3 --waveform bvp
```

### Recorded data layout (asymmetric RGB jpg-seq + TIR .wmv)

```
<data_path>/<session>/
    rgb/            # ordered jpg frames  (25 fps nominal)
    tir.wmv         # single WMV ~60 s    (25 fps nominal)
    signals.csv     # header: [time,] bvp, resp [, eda]  at fs Hz
```

Readers (`data/video_io.py`), temporal registration (`data/alignment.py`) and
the synchronised `PairedSessionDataset` (`data/paired_dataset.py`) handle this
layout; select it with `data_set: bp4d+` (alias `paired`). Per-session TIR fps
and signal sample rates are probed at runtime; both modalities are read on one
common time grid and the target waveform is resampled to `seq_len`. Set
data-set params in the YAML: `fs`, `fps`, `clip_duration`, `seq_len`,
`input_size`, `rgb_dir`, `tir_file`, `signals_file`, `train_ratio`.

## Suggested port order

1. `utils/` (dist, logger, metrics, checkpoint, optim_factory, EMA, scaler, pos_embed)
2. `core/` (registry -> blocks -> adapters -> criterion -> model)
3. `data/` (datasets + masking)
4. `engines/` then `runners/` wiring + YAML
5. `scripts/*.sh` launchers
