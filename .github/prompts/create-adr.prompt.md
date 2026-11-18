---
name: create-adr
description: Create a grounded architectural decision record for a new repository task
action-hint: Describe the proposed architectural or API change, including the motivation and expected outcome
agent: 'agent'
---

Create one new Architectural Decision Record (ADR) for the task described below.

**Task description:** ${input:taskDescription:Describe the proposed architectural or API change}

## Repository grounding

1. Read `AGENTS.md`, `.github/copilot-instructions.md`, and `code/notes/design/template.md` before drafting.
2. Trace the active execution path that controls the requested behavior. Follow the repository's code-as-ground-truth rule: inspect the relevant runner, config, engine, data, model, or evaluation implementation as needed.
3. Inspect the active YAML configuration files and argparse definitions for every parameter mentioned in the ADR. Do not invent CLI flags, YAML keys, defaults, module paths, metrics, or experiment results.
4. Check nearby design/spec/evaluation notes for terminology, protocol requirements, superseded decisions, and existing ADR numbering. If no numbering convention exists, use the number supplied by the user or leave the placeholder unchanged.

## Output requirements

1. Create exactly one Markdown file under `code/notes/design/` named according to the repository's existing convention. If no convention exists, use `ADR-[NUMBER]-[short-kebab-case-title].md`.
2. Start from `code/notes/design/template.md` and preserve its section structure.
3. Fill in:
   - status, date, author, related spec/plan, and superseded ADR;
   - the problem and research motivation;
   - target modules, runners, config files, and only parameters actually exposed by the runner's argparse parser;
   - falsifiable criteria with the exact split/fold, metric, aggregation unit, baseline, threshold, smoke command, and resource measurement method;
   - consequences, trade-offs, rollback/rejection condition, run IDs, evidence artifacts, and focused checks.
4. For respiration waveform work, follow `code/notes/design/Evaluation_RespWaveform_protocol.md`: use subject-level aggregation and the active headline metric rather than an unspecified validation Pearson value.
5. Keep evidence and hypotheses separate. Use `N/A` or an explicit `[TODO: verify ...]` marker where repository evidence is missing; never fabricate a result.
6. Do not modify source code, configurations, specifications, or existing notes. The task is to document the decision, not implement it.
7. Validate that the new Markdown file has no whitespace errors and that every referenced repository-relative path exists. Report the created path, inspected implementation/config files, and validation performed.
