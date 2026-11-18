---
name: review-thesis
description: Audit thesis LaTeX against active code, YAML configs, and experimental results.
argument-hint: Specify the .tex file to review
---

### Role & Objective
Review the specified thesis section below without modifying files. Verify scientific accuracy, academic style, canonical terminology, and consistency with the executable implementation in `code/`.

**Review target:** ${input:thesisTarget:Specify the .tex file or section to review}

### Verification Checklist
1. **Code & Config Grounding**: Does every reported hyperparameter, loss formulation, and dimension match `code/configs/` and `code/core/`?
2. **Result Verification**: Are numerical claims backed by recorded output files under `output/`?
3. **Literature & Citations**: Are claims backed by valid keys in `report/TUDaThesis/bib.bib`?
4. **Terminology & Scope**: Do names and symbols match `code/notes/GLOSSARY.md`, and is the section consistent with the thesis scope?
5. **Academic Tone & Precision**: Are assumptions, missing-modality cases (occlusion/absence), and limitations accurately qualified?

### Output Deliverable
1. **Consistency Matrix**:
   | Thesis Claim / Formula | Actual Code / Config Reality | Location (`file:line`) | Status (Match / Drift) |
2. **Categorized Findings**: Grouped by Critical Factual Issue, Major Scientific Gap, or Style Suggestion.
