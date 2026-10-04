# Global Repository Guidelines: PyTorch AI Model Development

## 1. Project Overview & Scope

- **Domain**: PyTorch-based Deep Learning research and development (Focus: MultiMAE / VideoMAE Multi-modal Vision).
- **Goal**: High-performance, modular, and reproducible neural network architectures and training pipelines.
- **AI Agent Context**: Code generation is driven by DeepSeek v4.1 Flash/Pro; code architecture, auditing, and reviews are handled by Claude 3.7 Sonnet. Maintain strict typing and modular OOP structure to ensure multi-agent context clarity.

## 2. Execution Environments & Workflows

The project operates under two distinct target environments, selected through the
environment profiles in `code/scripts/`:

- **Local Environment (WSL / single GPU)**:
  - **Purpose**: Rapid prototyping, dry-run smoke tests, debugging, and small-scale parameter checks.
  - **Setup**: Activate the repo venv at `.venv/` (deps: `code/requirements.txt`), then from `code/`: `source scripts/env_local.sh`.
  - **Data**: The canonical dataset is mounted as `DATA_PATH=$REPO/data/processed/bp4d_canonical` (raw at `RAW_DATA_PATH=$REPO/data/raw/BP4D`). There is **no** `data/dev/` split — keep runs small with the existing CLI caps `--max_sessions`, `--max_clips`, `--max_entries` (defined in `code/runners/_common.py`).
  - **Agent Rule**: Local smoke wrappers under `code/scripts/local/` run single-process. Do NOT require DDP launchers or a full-dataset mount for them.
- **HPC Cluster (Slurm, multi-GPU)**:
  - **Purpose**: Hyperparameter tuning, full-dataset training, ablation studies, and benchmark evaluation.
  - **Setup**: Edit `code/scripts/env_hpc.sh` (`VENV`/`CONDA_ENV`, `RAW_DATA_PATH`, `DATA_PATH`, `PROJ_DIR`, `OUTPUT_DIR`), then submit from `code/`: `sbatch scripts/hpc/<job>.sbatch`.
  - **Agent Rule**: Engines under `code/engines/` MUST stay DDP-compatible. Distributed bootstrap is centralised in `code/utils/dist.py` (`init_distributed_mode`, `get_rank`, `get_world_size`) and uses dynamic rank discovery (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`); it works under `torchrun` and `torch.distributed.launch`. Slurm batch templates live in `code/scripts/hpc/` as `.sbatch` files (there is no top-level `tools/slurm/`).

## 3. Directory & Scope Responsibilities

- **`code/`**: Core PyTorch codebase (Data, Core, Models, Engines, Runners, Utils). ALL code refactoring and updates MUST happen inside this directory.
- **`data/`**: Datasets and derived artifacts. Layout: `data/raw/`, `data/interim/`, `data/processed/`. Read-only for AI agents.
- **`report/`**: Academic LaTeX source files for papers. DO NOT modify unless explicitly instructed.
- **`presentation/`**: Slides and talk materials. DO NOT touch during code edits.
- **`models/` & `output/`**: Checkpoints, cached pretrained weights (`models/initial/`), and evaluation logs. Do not generate large output files directly into git tracking.
- **`docs/`**: Reference material. **`notebooks/`**: Exploratory notebooks (incl. `TirROI_Resp_Pipeline.ipynb`). **`tmp/`**: Scratch only — never a deliverable.
- **Environment**: Local runs use the Python `venv` at `.venv/` (workspace root). HPC runs use `VENV`/`CONDA_ENV` from `code/scripts/env_hpc.sh`. Dependencies are listed in `code/requirements.txt`.

## 4. Directory & Architecture Standards (under `code/`)

- `code/data/`: Datasets and preprocessing — `paired_dataset.py`, `rgb_roi_dataset.py`, `tir_resp_dataset.py`, `au_dataset.py`, `datasets.py`, `masking_generator.py`, `alignment.py`, `rgb_features.py`, `task_groups.py`, `video_io.py`, `prepare_bp4d.py`.
- `code/core/`: Architectures and primitives — `multimae.py`, `model.py`, `waveform_model.py`, `blocks.py`, `input_adapters.py`, `output_adapters.py`, `criterion.py`, `waveform_losses.py`, `au_probe.py`, `registry.py`.
- `code/models/`: Model factory and pretrained-weight loading — `build.py` (`create_model`, `is_model`, `list_models`) and `pretrained.py`; public re-exports in `code/models/__init__.py`.
- `code/engines/`: Training / evaluation / inference loops — `pretrain.py`, `finetune.py`, `waveform.py`, `au_probe.py`, `visualize.py`. (Note: singular `code/engine/` does not exist, and there is no `trainer.py` / `evaluator.py`.)
- `code/scripts/`: Shell wrappers — environment profiles `env_local.sh` / `env_hpc.sh` plus `local/`, `hpc/`, `project/` job scripts. They source the profile, select a config, and invoke the matching runner.
- `code/runners/`: Python CLI entry points — `run_pretrain.py`, `run_finetune.py`, `run_evaluate*.py`, `run_waveform.py`, `run_au_probe.py`, `run_inspect_*.py`, `run_download_weights.py`; shared argparse/YAML/DDP plumbing in `_common.py`.
- `code/configs/`: YAML experiment configs under `pretrain/` and `finetune/`. Configs are loaded as argparse *defaults* (`-c/--config`), so explicit CLI flags always override.
- `code/utils/`: Reusable helpers — `dist.py` (DDP + seed init), `logger.py` (`MetricLogger` / `SmoothedValue`, MultiMAE-style), `optim_factory.py`, `lr_sched.py`, `native_scaler.py` (AMP loss scaler + grad-norm), `model_ema.py`, `checkpoint.py`, `metrics.py`, `pos_embed.py`.
- `code/evaluation/`: Metric computation and reporting — `metrics.py`, `clinical.py`, `assemble.py`, `report.py`.
- `code/analysis/`, `code/design/`, `code/plan/`: Analysis scripts, design notes, and planning markdown — not runtime code.
- `code/tests/`: Reserved for the `pytest` suite (forward passes, tensor-shape flows, loss computation). **Currently empty** — no test files exist yet, so do not assume a working test runner.

## 5. PyTorch Engineering & Guardrails

- **Explicit Tensor Shapes**:
  - Every `forward()` pass and layer transformation MUST document expected tensor dimensions in docstrings or inline comments (e.g., Video: `[B, C, T, H, W]`, Tokens: `[B, N, D]`).
  - Do NOT remove explicit dimension assertion checks (`assert x.shape == ...`).
- **Memory & GPU Safety**:
  - Use `@torch.inference_mode()` or `with torch.no_grad():` during evaluation/testing.
  - ALWAYS call `.item()` when logging loss or metrics to prevent keeping the computational graph in GPU memory (prevents GPU OOM).
  - Mixed precision is handled by the custom AMP scaler in `code/utils/native_scaler.py` (loss scaling that also returns the gradient norm). There is currently **no** `torch.amp.autocast` usage in the codebase — if you introduce one, keep it opt-in and additive.
  - Device allocation goes through `code/runners/_common.py:init_env` (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
- **Model Design**:
  - Build models through `create_model(model_name: str, **kwargs)` in `code/models/build.py`; entry points are registered via `@register_model` in `code/core/registry.py` (`model_entrypoint`) and re-exported from `code/models/__init__.py`. (There is no `build_model(cfg)`.)
  - Prefer functional operations (`F.relu`, `F.scaled_dot_product_attention`) over unnecessary module instantiations.

## 6. Quality, Reproducibility & Testing Standards

- **Type Annotations**: Mandatory type hints for function arguments, return values, and tensor variables (`from torch import Tensor`, `from typing import Dict, Tuple, Optional`).
- **Reproducibility**: Seeding is centralised in `code/runners/_common.py:init_env`, which calls `torch.manual_seed`, `np.random.seed`, `random.seed` with `args.seed + get_rank()`. There is no standalone `set_seed()` helper — call `init_env(args)` from runners, and place any new seeding helper in `code/utils/`.
- **File Handling**: Prefer `pathlib.Path` in new code. Some existing modules still use `os.path` (e.g. `code/runners/_common.py`), so match the surrounding file rather than mass-rewriting.
- **Logging**: Training metrics are tracked through `code/utils/logger.py` (`MetricLogger`, MultiMAE-style). The stdlib `logging` module is NOT currently used in `code/`; prefer `MetricLogger` for trainer output and avoid adding ad-hoc `print()` in new engine code.
- **Dry-Run Check**: Before completing code refactoring, verify syntax (`python -m compileall code`, or import the touched module) and run the relevant `code/scripts/local/*.sh` smoke script. Run `pytest` once `code/tests/` is populated.

---

_Reconciled against the repository on 2026-10-04._
