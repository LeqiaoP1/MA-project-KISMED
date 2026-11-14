---
name: Review thesis section
description: Review a thesis section for scientific accuracy, logic, and academic style.
---

Review the specified thesis section without editing it initially.

Check:
1. Technical accuracy against the actual source code and configuration.
2. Whether each experimental statement is supported by recorded results.
3. Whether literature claims have valid citations in `report/TUDaThesis/bib.bib`.
4. Logical flow, scope, terminology, and graduate-level academic style.
5. Whether limitations, assumptions, and missing-modality conditions are described precisely.
6. Whether figures, tables, equations, and references are used consistently.

Return findings grouped by severity: critical factual issue, major scientific issue, minor issue, style suggestion. For every factual concern, identify the relevant source file, configuration, result file, or missing evidence. Do not invent corrections. Do not edit files unless I subsequently ask you to apply specific changes.
