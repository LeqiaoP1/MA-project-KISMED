# Thesis documentation instructions

This repository contains a graduate research project on multimodal AI.

## Working rules

- Treat source code, experiment configurations, and recorded outputs as evidence. Do not infer implementation details that are not present.
- Never invent metrics, dataset counts, citations, paper titles, author names, or experimental outcomes.
- Distinguish published findings, implementation facts, hypotheses, and results produced by this project.
- For thesis work, read the relevant files under `code/notes/` and inspect the relevant source/configuration/results before drafting.
- Make focused edits only to the files requested. Do not rewrite unrelated chapters or change the thesis structure without asking.
- Before editing, summarize the files you intend to inspect and modify when the task is substantial.
- After editing, summarize changed files and list unresolved factual questions or citation gaps.
- Do not run training jobs, submit HPC jobs, delete data, or overwrite experiment outputs unless explicitly requested.
- Do not run thesis result-generation scripts unless the user explicitly asks. Result generation is intentionally manual.
- Never commit changes or push to GitHub unless explicitly asked.

## LaTeX

- The thesis source is located in directory "report/TUDaThesis".
- Keep chapters in directory "report/TUDaThesis/chapters".
- Store figures in directory "report/TUDaThesis/gfx".
- Store generated tables in directory "report/TUDaThesis/generated".
- Use BibTeX citations as in file "report/TUDaThesis/bib.bib".
- Do not modify main.tex unless requested.

## Research integrity

- Use actual implementation details from the repository.
- Distinguish established literature from our own contributions.
- Clearly identify assumptions and limitations.
- Never fabricate references, metrics, or experimental findings.

## Academic writing

- Use formal, precise academic English.
- Prefer specific, testable statements over promotional language.
- Explain abbreviations at first use.
- Use consistent terminology for RGB, thermal, modality availability, missing modalities, masking, and occlusion.
- Do not claim that a method is robust, superior, or state-of-the-art without supporting evidence.
- Use only verified BibTeX keys already present in `report/TUDaThesis/bib.bib`. If a source is needed but unavailable, insert a clear `% TODO-CITATION: ...` comment instead of fabricating a citation.
- Preserve the institution's official thesis template and formatting requirements.
