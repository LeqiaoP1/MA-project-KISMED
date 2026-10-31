# Global Repository Guidelines: PyTorch AI Model Development

## 1. Project Overview & Scope

- **Domain**: PyTorch-based Deep Learning research and development (Focus: MultiMAE and VideoMAE with multi-modal dataset).
- **Goal**: High-performance, modular, and reproducible neural network architectures and training pipelines.
- **AI Agent Context**:
  - Implemenation code is generated and execution script managed by DeepSeek v4.1 Flash
  - Desgin architecture, feature auditing, and reviews are handled by Gemini 3.1 Pro or Claude 4.6 Sonnet.

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
- **`docs/`**: **Literature and research reference only** (thesis handouts, `docs/references/*.pdf`). Read-only — never write project notes here.
- **`code/notes/`**: The project's own context for human and agent readers — `spec/` (contract, generated), `design/` (how it is built, plus the review verdict), `eval/` (measurements and verdicts), `plan/` (implementation roadmap). Not runtime code.
- **`notebooks/`**: Exploratory notebooks (incl. `TirROI_Resp_Pipeline.ipynb`). **`tmp/`**: Scratch only — never a deliverable.
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
- `code/notes/`: Context, **not runtime code** — `spec/` (generated contract), `design/` (how it is built), `eval/` (measurements + verdicts, including the probe scripts under `tir_resp/` and `rgb_bp/`), `plan/` (implementation roadmap).
- `code/tests/`: The live `pytest` suite (forward passes, tensor-shape flows, loss computation, AMP, config keys, ROI cache) plus `code/pytest.ini` (`testpaths = tests`). Run `python -m pytest` from `code/`.

## 5. Specification & Architecture Referencing

- The repository specifications are modularized under `code/notes/spec/`:
  - Architecture & Tensor Dimensions: Refer to `#file:code/notes/spec/02_model_architecture.md`
  - Data Pipeline & Masking: Refer to `#file:code/notes/spec/01_data_pipeline.md`
  - Training Engine & Loss: Refer to `#file:code/notes/spec/03_engine_and_loss.md`
- Always check the relevant spec file under `code/notes/spec/` before modifying existing PyTorch modules or adding new feature branches.
- The spec is a **dated snapshot** (`code/notes/spec/*.md`, reverse-engineered 2026-10-04). The code is the source of truth; regenerate the spec after architectural changes with `python code/scripts/project/write_specs.py` (read its warning first — it overwrites hand-edited content).

### Document flow for a change

`spec/` (what it must be) -> `design/` (how, plus the review verdict) -> implementation in `code/` -> `code/tests/` -> `eval/` (measurement and verdict).

- One file per topic in `code/notes/`. Delete a superseded file instead of keeping a second copy; do not create `archive/`, `STATUS.md` or index files.
- Authority when docs disagree: `eval/` measurements are facts; `design/` wins on intent; each study has exactly one plan in `code/notes/plan/` (`RGB_BP_plan.md`, `TirROI_Resp_plan.md`), which wins on code placement.
- Any doc a code comment or config points at **must exist** at the referenced path. If you delete or rename one, fix every reference in the same commit (`grep -rn "code/notes/" code/`).

## 6. PyTorch Engineering & Guardrails

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

## 7. Quality, Reproducibility & Testing Standards

- **Type Annotations**: Mandatory type hints for function arguments, return values, and tensor variables (`from torch import Tensor`, `from typing import Dict, Tuple, Optional`).
- **Reproducibility**: Seeding is centralised in `code/runners/_common.py:init_env`, which calls `torch.manual_seed`, `np.random.seed`, `random.seed` with `args.seed + get_rank()`. There is no standalone `set_seed()` helper — call `init_env(args)` from runners, and place any new seeding helper in `code/utils/`.
- **File Handling**: Prefer `pathlib.Path` in new code. Some existing modules still use `os.path` (e.g. `code/runners/_common.py`), so match the surrounding file rather than mass-rewriting.
- **Logging**: Training metrics are tracked through `code/utils/logger.py` (`MetricLogger`, MultiMAE-style). The stdlib `logging` module is NOT currently used in `code/`; prefer `MetricLogger` for trainer output and avoid adding ad-hoc `print()` in new engine code.
- **Dry-Run Check**: Before completing code refactoring, verify syntax (`python -m compileall code`, or import the touched module) and run the relevant `code/scripts/local/*.sh` smoke script. Run `pytest` once `code/tests/` is populated.

---

_Reconciled against the repository on 2026-10-04._
