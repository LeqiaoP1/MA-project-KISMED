---
name: review-thesis
description: Audit thesis LaTeX against active code, YAML configs, and experimental results.
argument-hint: Specify the .tex file to review
---

### Role & Objective
Review the specified thesis section without modifying files. Verify scientific accuracy, academic style, and absolute consistency with the executable implementation in `code/`.

### Verification Checklist
1. **Code & Config Grounding**: Does every reported hyperparameter, loss formulation, and dimension match `code/configs/` and `code/core/`?
2. **Result Verification**: Are numerical claims backed by recorded output files under `output/`?
3. **Literature & Citations**: Are claims backed by valid keys in `report/TUDaThesis/bib.bib`?
4. **Academic Tone & Precision**: Are assumptions, missing-modality cases (occlusion/absence), and limitations accurately qualified?

### Output Deliverable
1. **Consistency Matrix**:
   | Thesis Claim / Formula | Actual Code / Config Reality | Location (`file:line`) | Status (Match / Drift) |
2. **Categorized Findings**: Grouped by Critical Factual Issue, Major Scientific Gap, or Style Suggestion.
