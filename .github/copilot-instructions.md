# Global Repository Guidelines: PyTorch AI Model Development

## 1. Project Overview & Scope

- **Domain**: PyTorch-based Deep Learning and AI model research and development.
- **Goal**: High-performance, modular, and reproducible neural network architectures and training pipelines.
- **AI Agent Context**: Code generation is driven by DeepSeek v4.1, auditing/reviews by GPT-4.6 luna. Maintain clear, typed, and well-structured code to assist multi-agent parsing.

## 2. Tech Stack & Dependencies

- **Core Framework**: PyTorch with CUDA support.
- **Ecosystem Libraries**: dependent package installed via the "code/requirements.txt"
- **Tooling & Environment**: environment managed by python in-built venv module.  Path ".venv"

## 3. Directory & Architecture Standards

Ensure data in the path "./data" (relateive to the workspace root).

Ensure code in the path "./code" follows a clean separation of concerns across the project layout:

- `code/data/`: `Dataset` and `DataLoader` abstractions, transformations, and preprocessing pipelines.
- `code/engines/`: Training, evaluation, and inference loops (`trainer.py`, `evaluator.py`).
- `code/configs/`: Hyperparameters and experiment configurations managed via dataclasses or YAML.
- `code/utils/`: Reusable helpers (logging, seed management, metrics calculation).
- `code/tests/`: `pytest` suite testing forward passes, gradient flows, and shape transformations.
- 

## 4. PyTorch Engineering & Code Conventions

- **Explicit Tensor Shapes**: Every `forward()` pass and custom layer must document expected tensor dimensions in comments or docstrings (e.g., `x: Tensor [B, C, H, W] -> output: Tensor [B, N, D]`).
- **Memory & GPU Optimization**:
  - Use `with torch.no_grad():` or `@torch.inference_mode()` during validation/testing to avoid gradient graph allocation.
  - Avoid calling `.item()` or transferring tensors to CPU (`.cpu()`) inside training loops unless explicitly required for logging to avoid GPU synchronization bottlenecks.
  - Allocate devices dynamically: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Model Design**:
  - Prefer functional operators (`F.relu`, `F.scaled_dot_product_attention`) over stateless module instantiations where appropriate.
  - Implement a `build_model(cfg)` factory function in `models/__init__.py` to instantiate models from config objects.

## 5. Python & Quality Standards

- **Type Annotations**: Mandatory type hints for all function arguments, return values, and tensor variables (`from torch import Tensor`, `from typing import Dict, Tuple, Optional`).
- **Reproducibility**: Include a central `set_seed(seed: int)` helper in `utils/seed.py` that sets seeds for `torch`, `numpy`, `random`, and `torch.cuda`.
- **File Handling**: Always use `pathlib.Path` instead of string manipulations with `os.path`.
- **Logging**: Use Python's `logging` module rather than `print()` statements for tracking training loss, metrics, and warnings.
