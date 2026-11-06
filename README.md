# KISMED Multimodal Physiological Waveform Recovery

Research code for reconstructing contact-based physiological waveforms from
contactless RGB and thermal-infrared video. The project combines VideoMAE-style
visual initialization with MultiMAE-style multimodal masked pre-training on
BP4D+, followed by task-specific waveform fine-tuning.

The current downstream branches target:

- **BP**: blood-pressure waveform reconstruction
- **RESP**: respiration waveform reconstruction
- **EDA**: electrodermal-activity waveform reconstruction

An optional action-unit occurrence probe and thermal-ROI respiration pipeline are
also included.

> **Data and research status:** BP4D+ data and model checkpoints are not stored
> in this repository. Obtain access through the appropriate dataset and model
> providers before running the pipelines.

## What is included

| Path | Purpose |
| --- | --- |
| [`code/core/`](code/core/) | Multimodal models, adapters, losses, and model primitives |
| [`code/data/`](code/data/) | BP4D+ conversion, video/physiology loading, masking, and datasets |
| [`code/engines/`](code/engines/) | Training, validation, inference, and visualization loops |
| [`code/evaluation/`](code/evaluation/) | Temporal, spectral, and clinical waveform metrics |
| [`code/runners/`](code/runners/) | Command-line entry points |
| [`code/configs/`](code/configs/) | YAML experiment configurations |
| [`code/scripts/`](code/scripts/) | Local and Slurm/HPC environment and launch scripts |
| [`code/tests/`](code/tests/) | Pytest coverage for data, models, losses, and runners |
| [`code/notes/`](code/notes/) | Architecture specifications, design decisions, plans, and evaluations |
| [`notebooks/`](notebooks/) | Exploratory notebooks and pipeline guidance |

## Requirements

- Linux or WSL with Python **3.10 or newer**
- Python packages listed in [`code/requirements.txt`](code/requirements.txt)
- A CUDA-capable GPU for practical model training
- BP4D+ raw data for preprocessing and training
- Optional: a Slurm cluster for multi-GPU/HPC experiments

The repository does not package or download BP4D+ automatically. Keep raw data
outside version control.

## Installation

```bash
git clone https://github.com/LeqiaoP1/MA-project-KISMED.git
cd MA-project-KISMED

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

For local runs, configure the environment from the `code/` directory:

```bash
cd code
source scripts/env_local.sh
```

The local profile defaults to:

- raw BP4D+ data: `../data/raw/BP4D`
- canonical processed data: `../data/processed/bp4d_canonical`
- outputs: `../output`
- initial model cache: `../models/initial`

Override these locations before sourcing the profile when needed:

```bash
export RAW_DATA_PATH=/path/to/BP4D
export DATA_PATH=/path/to/bp4d_canonical
export OUTPUT_DIR=/path/to/output
source scripts/env_local.sh
```

## Quick start

All Python entry points are intended to be run from [`code/`](code/).

### 1. Prepare and inspect the data

The converter expects the raw BP4D+ trees `2D+3D/`, `Thermal/`, and
`Physiology/`. It writes one synchronized canonical directory per session.

```bash
cd code
python data/prepare_bp4d.py \
  --raw_root ../data/raw/BP4D \
  --out_root ../data/processed/bp4d_canonical \
  --limit_sessions 8

python runners/run_inspect_data.py \
  --data_path ../data/processed/bp4d_canonical \
  --clip_duration 2 \
  --input_size 64 \
  --plot
```

Remove `--limit_sessions` for a complete conversion. Use
`run_inspect_physio.py` to inspect raw physiology channels before training.

### 2. Run multimodal masked pre-training

Configs are loaded as argument defaults, and explicit command-line arguments
override YAML values.

```bash
python runners/run_pretrain.py \
  -c configs/pretrain/stage2_local_pretrained.yaml
```

The local configurations are intentionally small smoke or milestone runs.
Before a long experiment, check the selected data path, output directory,
streams, model geometry, and pretrained encoder checkpoint.

### 3. Fine-tune a waveform branch

Fine-tuning requires a compatible Stage-2 checkpoint. The Stage-3 geometry and
ROI settings must match the checkpoint’s recorded configuration.

```bash
python runners/run_waveform.py -c configs/finetune/bp.yaml
python runners/run_waveform.py -c configs/finetune/resp.yaml
python runners/run_waveform.py -c configs/finetune/eda.yaml
```

For the thermal-ROI respiration branch:

```bash
python runners/run_waveform.py \
  -c configs/finetune/resp_tir_roi_local.yaml \
  --val_subject F004
```

### 4. Evaluate saved predictions

The offline evaluator accepts `.npy` or `.npz` prediction and target arrays.

```bash
python runners/run_evaluate.py \
  --pred_path predictions.npy \
  --target_path targets.npy \
  --fs 100 \
  --tier 1,2,3 \
  --waveform bp \
  --out metrics.json
```

Tier 1 reports temporal MAE/RMSE/Pearson metrics. Tier 2 reports Welch PSD
consistency. Tier 3 reports clinical BP/HRV metrics and requires NeuroKit2.

## HPC and multi-GPU runs

Edit [`code/scripts/env_hpc.sh`](code/scripts/env_hpc.sh) for the cluster’s
virtual environment, data paths, output location, partition, account, and GPU
resource specification. Then submit the relevant Slurm job from `code/`:

```bash
cd code
source scripts/env_hpc.sh
sbatch scripts/hpc/<job>.sbatch
```

The HPC profile is a template and contains site-specific values that must be
reviewed before submission. Local scripts under `code/scripts/local/` are
single-process smoke wrappers; they do not require a DDP launcher.

## Testing

Run the test suite from `code/`:

```bash
cd code
python -m pytest
```

For a syntax-only check:

```bash
python -m compileall .
```

## Documentation and support

- [`code/README.md`](code/README.md): implementation layout and experiment
  commands
- [`code/notes/spec/`](code/notes/spec/): generated architecture, data, and
  training contracts
- [`code/notes/design/`](code/notes/design/): design decisions and review
  verdicts
- [`code/notes/plan/`](code/notes/plan/): current experiment plans
- [`code/notes/eval/`](code/notes/eval/): measurements and evaluation verdicts
- [`notebooks/TirROI_Resp_Guide.md`](notebooks/TirROI_Resp_Guide.md): thermal-ROI
  respiration workflow
- [`docs/`](docs/): project and thesis reference material

If a command or configuration is unclear, first check its `--help` output and
the corresponding YAML file. For reproducible bugs or proposed improvements,
open a GitHub issue with the command, configuration, environment, and concise
error output. Do not attach BP4D+ data or private checkpoints.

## Contributing and maintenance

This is a research/thesis repository maintained by the KIS*MED project team.
Contributions should remain focused and reproducible:

1. Create a branch for the change.
2. Keep runtime code changes under [`code/`](code/).
3. Update directly related configuration or documentation.
4. Add or update tests under [`code/tests/`](code/tests/) when behavior changes.
5. Run the relevant pytest tests and a local smoke command.
6. Open a pull request describing the data/configuration assumptions and
   validation performed.

Architecture changes should also update the relevant specification, design, or
evaluation note. See [`AGENTS.md`](AGENTS.md) for repository-wide development
conventions.

## Project structure

```text
data/          Raw, interim, and processed datasets (not committed)
models/        Initial and trained model weights
output/        Training outputs and evaluation artifacts
code/          Python package, runners, configs, tests, and scripts
notebooks/     Exploratory analysis
docs/          Reference material
report/        Thesis source
presentation/  Presentation source and assets
```
