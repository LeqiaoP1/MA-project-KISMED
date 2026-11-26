# Repository Guidelines: PyTorch Deep Learning Research (KISMED MA)

## 1. Core Mandate: Code as Ground Truth

- **Authority**: The executable PyTorch code under `code/` and active configurations under `code/configs/` dictate the final system architecture and parameters. Early proposals or outdated design notes are strictly superseded.
- **Zero Redundancy**:
  - Trace active execution flow only (`code/runners/run_*.py` -> `code/engines/` -> `code/core/`).
  - Prune or ignore abandoned experimental flags, legacy files, and dead branches.
  - Maintain exactly one authoritative document per topic under `code/notes/`.
- **Agents**: DeepSeek (Implementation & CLI execution), Gemini / Claude (Architecture audit, specs, thesis alignment).

## 2. Environments & Workflows

- **Local (WSL / 1 GPU)**: Fast smoke tests (`code/scripts/local/*.sh`, `source code/scripts/env_local.sh`). Use CLI caps (`--max_sessions`, `--max_clips`, `--max_entries` in `code/runners/_common.py`). No DDP required.
- **HPC (Slurm / Multi-GPU)**: Full pre-training and waveform fine-tuning (`sbatch code/scripts/hpc/<job>.sbatch`). Must use dynamic rank discovery via `code/utils/dist.py` (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`).

## 3. Directory Layout & Boundaries

- `code/`: ALL source modifications happen here (`data/`, `core/`, `models/`, `engines/`, `runners/`, `configs/`, `utils/`, `evaluation/`, `tests/`).
- `code/configs/`: Baseline parameters loaded as argparse defaults (`-c/--config`). Explicit CLI flags take precedence.
- `code/notes/spec/`: Reverse-engineered system contracts (`01_data_pipeline.md`, `02_model_architecture.md`, `03_engine_and_loss.md`). Regenerate directly from code using `python code/scripts/project/write_specs.py`.
- `code/notes/design/`: Architecture Decision Records (`ADR-NNN-<slug>.md`) indexed by `code/notes/design/README.md`; every active research aspect is governed by one ADR, and superseded ADRs move to `code/notes/design/archive/`.
- `code/notes/plan/`: Roadmap of record, split by scope — `experiments/` holds training-run protocols (configs, fold orchestration, gates); `features/` holds code implementation plans (`ADR-NNN-<slug>-plan.md`) paired 1-to-1 with active `design/ADR-*` records (superseded plans move to `features/archive/`). Measured outcomes land in `code/notes/eval/`, never in a plan.
- `report/`: Academic LaTeX files (`report/TUDaThesis/`). Do not modify during routine code edits.
- `data/`, `docs/`, `models/initial/`: Read-only.

## 4. PyTorch Engineering & Parameter Guardrails

- **Zero Hidden Constants**: All loss weights, masking ratios, dimensions, and learning rates must reside in `code/configs/` or CLI flags—no hardcoded magic numbers in `core/` or `engines/`.
- **Tensor Shape Assertions**: Retain explicit shape annotations and runtime assertions across all spatial-temporal transformations (Video: `[B, C, T, H, W]`, Tokens: `[B, N, D]`).
- **Memory & Dist**: Use `@torch.inference_mode()`, AMP scaler (`code/utils/native_scaler.py`), `.item()` for metric logging, and seed init via `code/runners/_common.py:init_env`.
- **Model Factory**: Register modules with `@register_model` in `code/core/registry.py` and instantiate via `create_model()` in `code/models/build.py`.
- **Terminology Grounding**: Refer to `code/notes/GLOSSARY.md` for canonical naming. Never introduce deprecated aliases or ad-hoc abbreviations in code or comments.