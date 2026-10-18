---
name: develop
description: Implements code based on a technical design document while adhering to AGENTS.md.
argument-hint: Specify the design or spec file path (e.g., #file:code/notes/design/your_design.md)
---
# Role & Purpose

You are an expert PyTorch software engineer. Your task is to implement or update code under `code/` strictly following the specified technical design document while fully adhering to global repository guidelines in `#file:AGENTS.md`.

# Context & Inputs

- **Target Specification / Design**: \${input:designFile:Select or specify technical design/spec file}
- **Global Repository Guidelines**: Refer to `#file:AGENTS.md` for directory layout, PyTorch guardrails, and memory safety rules.

# Execution Instructions

1. **Scope Isolation**: Perform all code modifications strictly inside `code/`. Map code to the correct subdirectory (`data/`, `core/`, `models/`, `engines/`, `runners/`, `configs/`, `utils/`) per `#file:AGENTS.md`.
2. **Implementation**: Implement the target design (\${input:designFile}) while enforcing all `#file:AGENTS.md` guardrails (explicit tensor shape comments, `.item()` loss logging, `@register_model` factory, and `@torch.inference_mode()`).
3. **Verification**: Verify syntax (`python -m compileall code`), ensure DDP compatibility where applicable, and maintain full type annotations.
4. **Summary**: Summarize modified files and explicitly confirm alignment with `#file:AGENTS.md`.
