---
description: Refactors code based on prior reviewer feedback.
---
# Instructions

You are an expert PyTorch developer. Look at the code review comments provided in the previous chat turn and update the target file accordingly:

1. **Fix Critical Bugs**: Resolve shape mismatches, missing `detach()` calls, or memory leaks.
2. **Preserve Rules**: Follow all guidelines in `.github/copilot-instructions.md`.
3. **Tests**: Add or update focused regression tests under `./code/tests/` using `pytest`.
   Tests must import project modules through `code.*` and be runnable from the
   repository root with `python -m pytest -q code/tests/`. 
4. **Format**: Apply edits directly to the file if permitted, or output clean updated code blocks.

Target file: \${input:file:Select file to update}
