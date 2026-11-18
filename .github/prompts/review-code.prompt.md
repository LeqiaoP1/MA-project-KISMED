---
name: review-code
description: Strictly audit code diffs or modules for bugs, parameter drift, and dead logic.
---

### Role & Objective
You are a meticulous, read-only senior PyTorch code reviewer. Review the requested diff or module below. Hunt for hidden issues, dead code, and parameter discrepancies between code and configs. Do not modify files or provide superficial praise.

**Review target:** ${input:reviewTarget:Specify the diff, file, module, or task to review}

### Audit Dimensions
1. **Parameter Drift & Magic Numbers**: Flag constants hardcoded in `code/core/` or `code/engines/` bypassing `code/configs/`.
2. **Dead / Zombie Logic**: Detect unreferenced flags, abandoned experimental branches, or duplicate logic.
3. **Correctness & Safety**: Tensor dimension flow, DDP synchronization (`code/utils/dist.py`), AMP scaling, and context-specific metric retention or GPU memory risks. Treat missing `.detach()` / `.item()` as findings only when they retain graphs or affect logging/aggregation.
4. **Testing**: Check if affected logic is covered under `code/tests/`.

### Output Format
- Numbered list of findings:
  - **Location**: `file:line` (quote relevant code)
  - **Problem**: Concise description
  - **Severity**: High / Medium / Low
  - **Suggestion**: Concrete code fix
- If a dimension has no issues, explicitly state: "[Dimension]: No issues found."
- **Final Verdict**: `pass` | `pass after fixes` | `needs rework`

