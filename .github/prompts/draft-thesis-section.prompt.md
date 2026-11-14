---
name: Draft thesis section
description: Draft one evidence-grounded thesis section in LaTeX.
---

Draft or revise only the section specified in my request.

Before writing:
1. Read the relevant thesis notes under `code/notes/`.
2. Inspect the relevant implementation, configuration, experiment protocol, and results files that support the section.
3. Check `report/TUDaThesis/bib.bib` for citation keys that actually exist.
4. Identify missing information that prevents a reliable statement.

Writing requirements:
- Write formal academic English appropriate for a graduate thesis.
- Explain the scientific motivation, method, assumptions, and limitations where relevant.
- Clearly distinguish literature claims from this project's implementation and findings.
- Never invent code behavior, data properties, metrics, citations, or conclusions.
- Use only verified BibTeX keys. For a missing source, add `% TODO-CITATION: describe the source needed`.
- Do not turn an unverified hypothesis into a factual statement.
- Preserve existing labels and cross-references.
- Do not alter unrelated sections or the document preamble.

Edit the requested `.tex` file only after inspecting the evidence. Then report the files inspected, files changed, evidence used, and unresolved questions. Do not compile, regenerate results, commit, or push unless I explicitly ask.
