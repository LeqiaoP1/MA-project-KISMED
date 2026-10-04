# Global Repository Guidelines: PyTorch AI Model Development

## 1. Project Overview & Scope

- **Domain**: PyTorch-based Deep Learning research and development (Focus: MultiMAE / VideoMAE Multi-modal Vision).
- **Goal**: High-performance, modular, and reproducible neural network architectures and training pipelines.
- **AI Agent Context**: Code generation is driven by DeepSeek v4.1 Flash/Pro; code architecture, auditing, and reviews are handled by Claude 3.7 Sonnet. Maintain strict typing and modular OOP structure to ensure multi-agent context clarity.

## 2. Execution Environments & Workflows

The project operates strictly under two distinct target environments:

- **Local Environment (Single GPU)**:
  - **Purpose**: Rapid prototyping, dry-run testing, debugging, and small-scale parameter checks.
  - **Constraints**: Limited VRAM, uses small data subsets (`./data/dev/`), reduced batch sizes, and single-GPU execution.
  - **Agent Rule**: Do NOT require DDP launchers or full-dataset mounts for basic test scripts under `code/tests/`.
- **HPC Cluster (Multi-GPU)**:
  - **Purpose**: Hyperparameter tuning, full-dataset training, ablation studies, and benchmark evaluation.
  - **Constraints**: Multi-GPU nodes running via Slurm schedulers with high-throughput full dataset mounts.
  - **Agent Rule**: Engine scripts under `code/engine/` MUST support DistributedDataParallel (`torchrun` / DDP) using dynamic rank discovery (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`). Store `.sh`/`.slurm` batch scripts under `tools/slurm/`.

## 3. Directory & Scope Responsibilities

- **`code/`**: Core PyTorch codebase (Models, Engine, Data, Utils). ALL code refactoring and updates MUST happen inside this directory.
- **`data/`**: Datasets, raw inputs, and video/image patches (relative to workspace root). Read-only for AI agents.
- **`report/`**: Academic LaTeX source files for papers. DO NOT modify unless explicitly instructed.
- **`presentation/`**: Slides and talk materials. DO NOT touch during code edits.
- **`models/` & `output/`**: Checkpoints and evaluation logs. Do not generate large output files directly into git tracking.
- **Environment**: Managed via Python built-in `venv` module at `.venv`. Dependencies listed in `code/requirements.txt`.

## 4. Directory & Architecture Standards (under `code/`)

- `code/data/`: `Dataset`, `DataLoader`, patchifying transforms, and preprocessing pipelines.
- `code/core/`: Core model architectures, encoders, decoders, adapters, blocks, losses, and masking logic.
- `code/models/`: Model factories, model registration/building utilities, and pretrained-model loading helpers.
- `code/engines/`: Training, evaluation, and inference execution loops (`trainer.py`, `evaluator.py`).
- `code/scripts/`: Shell wrappers for local, HPC, and project workflows; they source environment settings, select configs, and invoke the appropriate runner.
- `code/runners/`: Python CLI entry points for pre-training, fine-tuning, evaluation, inspection, visualization, and waveform workflows, including shared config and distributed-runtime setup.
- `code/configs/`: Hyperparameters and experiment configurations managed via YAML or dataclasses (Zero hardcoded constants in Python scripts).
- `code/utils/`: Reusable helpers (logging, seed management, metric calculators).
- `code/tests/`: `pytest` suite for testing forward passes, tensor shape flows, and loss computation.

## 5. PyTorch Engineering & Guardrails

- **Explicit Tensor Shapes**:
  - Every `forward()` pass and layer transformation MUST document expected tensor dimensions in docstrings or inline comments (e.g., Video: `[B, C, T, H, W]`, Tokens: `[B, N, D]`).
  - Do NOT remove explicit dimension assertion checks (`assert x.shape == ...`).
- **Memory & GPU Safety**:
  - Use `@torch.inference_mode()` or `with torch.no_grad():` during evaluation/testing.
  - ALWAYS call `.item()` when logging loss or metrics to prevent keeping the computational graph in GPU memory (prevents GPU OOM).
  - Retain native mixed precision (`torch.amp.autocast('cuda')`) support across model forward passes.
  - Dynamic device allocation: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Model Design**:
  - Implement and maintain model construction through the `build_model(cfg)` factory in `code/models/build.py` and its public exports in `code/models/__init__.py`.
  - Prefer functional operations (`F.relu`, `F.scaled_dot_product_attention`) over unnecessary module instantiations.

## 6. Quality, Reproducibility & Testing Standards

- **Type Annotations**: Mandatory type hints for function arguments, return values, and tensor variables (`from torch import Tensor`, `from typing import Dict, Tuple, Optional`).
- **Reproducibility**: Use a central `set_seed(seed: int)` helper setting seeds for `torch`, `numpy`, `random`, and `torch.cuda`.
- **File Handling**: Always use `pathlib.Path` instead of string operations with `os.path`.
- **Logging**: Use Python's `logging` module rather than `print()` statements for tracking training progress and metrics.
- **Dry-Run Check**: Before completing code refactoring tasks, verify syntax or run unit tests under `code/tests/`.
