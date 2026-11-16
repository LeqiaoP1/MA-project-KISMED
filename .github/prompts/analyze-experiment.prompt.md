---
name: analyze-experiment
description: Explain an existing experiment using its recorded protocol and outputs.
argument-hint: Specify the experiment config or output directory
---

Analyze the experiment identified in the request. Inspect its YAML configuration, evaluation code, logs, and recorded `.npy`/`.json` outputs. Do not execute jobs.

### Report Structure
1. **Research Objective**: Question addressed and comparison being made.
2. **Dataset & Conditions**: Documented split, modality condition (`--streams rgb,tir`), and temporal stride.
3. **Model & Checkpoint**: Architecture, pre-trained weights, and geometry.
4. **Verified Metrics**: Exact values extracted from recorded output files (Tier 1/2/3).
5. **Supported Findings vs. Hypotheses**: Separate empirical evidence from theoretical interpretations.
6. **Limitations & Missing Metadata**: Reproducibility gaps or undocumented parameters.
