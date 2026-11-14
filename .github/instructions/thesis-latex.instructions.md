---
name: Thesis LaTeX
description: Rules for editing LaTeX thesis source and bibliography.
applyTo: "report/TUDaThesis/**/*.tex, report/TUDaThesis/**/*.bib"
---

- Preserve the existing document class, preamble, macros, labels, and citation style unless explicitly asked to change them.
- Keep chapter content in `report/TUDaThesis/chapters/`; do not put whole chapters into `main.tex`.
- Use stable, unique labels such as `sec:`, `fig:`, `tab:`, and `eq:`.
- Reference figures and tables with `\ref{...}` or `\autoref{...}` consistently with the existing template.
- Do not manually copy numerical experiment results into prose or tables if a generated artifact already exists.
- Generated tables are included with `\input{generated/<file>.tex}`.
- Generated figures are stored under `report/TUDaThesis/generated/figures/`.
- Never invent BibTeX keys. Check `report/TUDaThesis/bib.bib` before citing.
- Avoid changing generated files unless the user explicitly requests regeneration or a manual correction.
- Keep LaTeX compilable; do not add packages without checking whether the template already loads them.
