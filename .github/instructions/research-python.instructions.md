---
name: Research Python
description: Rules for experiment analysis and thesis artifact-generation Python scripts.
applyTo: "code/**/*.py"
---

- Keep experiment analysis deterministic and document input/output schemas.
- Do not change training or evaluation semantics while editing thesis-generation scripts.
- Do not silently drop rows, samples, or missing values. Validate inputs and fail with a useful message.
- Do not fabricate, smooth, or otherwise alter reported metrics for presentation.
- Keep generated artifacts reproducible from tracked scripts and source CSV/JSON files.
- Do not run expensive training or submit HPC jobs unless explicitly requested.
- Do not run thesis artifact-generation scripts unless explicitly requested.
