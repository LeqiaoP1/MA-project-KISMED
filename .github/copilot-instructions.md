# Thesis Documentation Guidelines (TUDaThesis)

Apply the repository-wide engineering principles, execution workflows, and directory boundaries defined in `AGENTS.md`.

## 1. Code-Grounding & Academic Integrity

- **Evidence-Based Writing**: Every architecture specification, hyperparameter, dataset split, and evaluation metric reported in the thesis must trace directly to active implementations under `code/` or verified experiment logs.
- **No Conceptual Drift**: The executable code in `code/core/` and active configurations in `code/configs/` are the ground truth. If earlier thesis drafts contradict current code, update the LaTeX text to match the real implementation.
- **Strict Verification**: Never invent citations, sample counts, or baseline comparison numbers. Use only verified keys in `report/TUDaThesis/bib.bib` (insert `% TODO-CITATION: ...` if a reference is missing).
- **Scope Discipline**: Edit only the designated `.tex` sections. Do not alter document templates or `main.tex` without explicit instructions.

## 2. LaTeX Scope & Structure

- **Target Root**: `report/TUDaThesis/`
  - Chapters: `chapters/` (e.g., `3_methodology.tex`, `4_experiments.tex`)
  - Figures: `gfx/`
  - Generated tables: `generated/`
  - Bibliography: `bib.bib`
- **Mathematical Exactness**:
  - Equations for composite losses (e.g., L1, Pearson correlation, temporal smoothness, waveform recovery) must reflect the exact PyTorch operations (reduction types, scaling factors, normalization) implemented in `code/core/criterion.py` and `code/core/waveform_losses.py`.
- **Consistent Terminology**:
  - Adhere strictly to standardized project terminology: standard RGB, thermal-infrared (TIR), masked autoencoding, asymmetric masking, zero-shot evaluation, and physiological waveform reconstruction (BP, RESP, EDA).

## 3. Working Protocol for Copilot

- Before drafting or editing methodology chapters, inspect the corresponding model definitions in `code/core/` and active parameters in `code/configs/`.
- Summarize the specific files inspected before making substantial LaTeX modifications.
- Do not run training scripts, submit HPC jobs, delete data, or modify git tracking unless explicitly instructed.
