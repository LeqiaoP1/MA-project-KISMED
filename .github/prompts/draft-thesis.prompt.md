---
name: draft-thesis
description: Draft or revise a thesis section in LaTeX strictly grounded in code, configs, and verified outputs.
argument-hint: Specify the target .tex file (e.g., #file:report/TUDaThesis/chapters/3_methodology.tex)
---

### Role & Protocol
Draft or revise the requested LaTeX section in `report/TUDaThesis/chapters/` strictly grounded in the codebase and verified outputs.

### Pre-Drafting Checks
1. **Methodology Chapters**: Inspect exact tensor ops and loss implementations in `code/core/` and active parameters in `code/configs/`.
2. **Results & Discussion**: Trace every reported metric and table directly to existing JSON/CSV/NPY outputs under `output/`. Never extrapolate or invent causation without recorded evidence.
3. **Citations**: Check `report/TUDaThesis/bib.bib`. If a key is missing, insert `% TODO-CITATION: [description]` instead of fabricating.

### Writing Constraints
- Write formal academic English for a graduate thesis.
- Preserve document preambles, cross-references (`\ref{...}`), and standard notation (RGB, TIR, MultiMAE, VideoMAE, BP, RESP).
- Mathematical formulas for losses must exactly reflect the PyTorch implementation (reduction types, weights, normalization).
- Report: (1) Inspected source files, (2) Modified `.tex` files, (3) Unresolved questions or missing citations.

