---
name: Analyze experiment for thesis
description: Explain an existing experiment using its recorded protocol and outputs.
---

Analyze the experiment identified in my request.

Inspect its configuration, evaluation code, logs, and recorded result files. Do not run training or evaluation unless explicitly requested.

Report:
- Research question and comparison being made.
- Dataset/split and modality condition, only where documented.
- Model/checkpoint and relevant configuration.
- Metrics and exact values from the recorded output.
- Supported interpretation, separated from hypotheses.
- Limitations, missing metadata, and reproducibility concerns.

Do not invent missing values or infer statistical significance from point estimates. Do not regenerate tables or figures. If a result file is ambiguous, ask for clarification rather than guessing.
