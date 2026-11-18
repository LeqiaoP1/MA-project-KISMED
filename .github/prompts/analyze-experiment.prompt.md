---
name: analyze-experiment
description: Explain an existing experiment using its recorded protocol and outputs.
argument-hint: Specify the experiment config or output directory
---

Analyze the experiment identified below. Inspect only its YAML configuration, relevant evaluation code, logs, and recorded `.npy`/`.json` outputs. Do not execute jobs or modify files.

**Experiment target:** ${input:experimentTarget:Specify the experiment config or output directory}

### Report Structure
1. **Research Objective**: Question addressed and comparison being made.
2. **Dataset & Conditions**: Documented split, modality condition, and temporal stride. Report the actual modalities found; do not assume RGB/TIR or any other stream.
3. **Model & Checkpoint**: Architecture, pre-trained weights, and geometry.
4. **Verified Metrics**: Exact values extracted from recorded output files. Label each value as directly recorded or derived from recorded values, and show the derivation for derived values.
5. **Supported Findings vs. Hypotheses**: Separate empirical evidence from theoretical interpretations.
6. **Limitations & Missing Metadata**: Reproducibility gaps or undocumented parameters.
