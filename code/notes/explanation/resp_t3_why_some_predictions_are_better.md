# Why do some T3 respiration predictions look great and others look bad?

*Plain-English summary of the 2026-10-25 investigation. The technical versions,
with all numbers, tests and caveats, are
`code/notes/eval/tir_resp/resp_motion_confound.md` and
`code/notes/eval/tir_resp/resp_t3_spread_sources.md`. Everything here was
measured on the existing five cross-validation runs — nothing was retrained.*

## The short version

The original guess was that subjects moving their heads during the recording
explains why some T3 waveforms look convincing and others look wrong. **It
doesn't.** The real reason is much more boring: **some people are simply easier
to read than others, and that shows up on every task, not just T3.**

## What was compared

For every T3 session, the predicted waveform was scored against the breathing
belt recording (the headline number is the correlation of the breathing rhythm,
0.1–0.6 Hz, after removing slow drift). That score was then compared against
everything measurable about the session:

- how much the head moved, how abruptly, and how much of that movement sat in the
  breathing frequency range;
- how much the fixed image crop was inflated by that movement;
- how regular the subject's breathing was, how deep, how fast, how much the depth
  wobbled from breath to breath, how much slow baseline drift was present;
- how big the face was, how close the subject sat, how many image pixels the face
  actually occupied, and how much thermal detail the nose/mouth patch contained
  (read directly from the model's own cached input images);
- how many clips the session produced, and whether landmark tracking was clean;
- which of the five cross-validation models happened to predict that subject.

## Finding 1 — it is mostly about who the person is

Take a subject's score on their **other nine tasks** and use it to predict their
T3 score. The agreement is strong (correlation +0.67), and that single number
explains about **43 % of the entire T3 spread**. Nothing else measured adds
anything on top of it.

In plain terms: if you know how well the model did on someone's singing task or
pain task, you can already guess how well it did on their quiet-baseline task.
The "T3 spread" is therefore not really a T3 story — it is a **per-person** story
that repeats across all ten tasks. Some faces give the model a readable thermal
respiration signal; most do not; and it stays that way for that person.

That also explains why the *breathing* characteristics do not carry the effect:
how regular or how deep someone breathes is a property of a single recording,
whereas how well they are read at all is a property of the person.

### The numbers behind it

Figure: `code/notes/eval/tir_resp/resp_t3_subject_effect.svg`
(reproduce with `code/notes/eval/tir_resp/subject_effect_evidence.py`).

Slide figure for the additive model itself: `presentation/figures/
additive-model-slide.svg` (16:9-ish, 1500 x 760; regenerate with
`code/notes/eval/tir_resp/additive_model_slide.py`). It draws the equation
`observed = overall level + person + task + everything else` as five 12-subject
grids taken from the real fit, with a stacked bar underneath showing 36 % /
8.6 % / 55 %.

**Check 1 — how much of the whole evaluation is "which person" vs "which task".**
Fit a simple additive model to all 1260 session scores: one term per person, one
term per task.

| what it explains | share of the variation |
|---|---|
| **who the person is** | **36.0 %** |
| which task it was | 8.6 % |
| both together | 44.6 % |

So the identity of the subject is more than **four times** as informative as the
identity of the task.

**Check 2 — is that stable, or is it the same recording counted twice?**
Split each person's tasks into two *disjoint* halves (odd-numbered tasks
T1/T3/T5/T7/T9 vs even-numbered T2/T4/T6/T8/T10) and compare their median score
on each half. Different recordings, often different days, different activities:

- Spearman +0.69 across the two halves (n = 137).
- Corrected for the fact that each half is only five tasks, reliability
  **+0.82**. A person's "readability" is highly reproducible and is not an
  artefact of any single session.

**Check 3 — can we predict T3 without looking at T3?**
Use only the other nine tasks to guess the T3 score for all 129 subjects:

| | T3 score |
|---|---|
| bottom quarter of subjects (other-task median +0.09) | **+0.22** |
| second quarter (+0.21) | +0.50 |
| third quarter (+0.36) | +0.58 |
| top quarter (+0.55) | **+0.79** |

Spearman +0.67, Pearson +0.63 (R² 0.40). That is a **0.57 gap** in T3 quality
between the easiest and hardest quarter of subjects, obtained without ever
looking at the T3 recording.

**Check 4 — the figure.** Panel (a) is that scatter: points rise steadily from
bottom-left to top-right, and the red quartile medians trace the climb. Panel (b)
is the full 138-subject × 10-task score matrix, sorted by each subject's median.
If the task were what mattered, you would see vertical stripes. Instead the rows
read as **horizontal bands**: a person who scores poorly is poor nearly
everywhere, a person who scores well is good nearly everywhere.

## Finding 2 — the surprise: the five folds are all-women, then all-men

The fold assignment simply chops the sorted subject list into five equal pieces.
Because IDs run F001…F057 and then M001…M081, the pieces came out as:

| fold | test subjects | who |
|---|---|---|
| 0 | F001–F028 | 28 women |
| 1 | F029–F056 | 28 women |
| 2 | F057–M002 | 26 women, 2 men |
| 3 | M003–M030 | 28 men |
| 4 | M031–M058 | 27 men |

So what looked like "fold 1 has a weaker model" is really "fold 1 is the second
group of women". Men score clearly higher on T3 (about 0.71 vs 0.48), and the
same gap exists across all tasks.

The gap is **not** because men breathe more regularly, move less, or have cleaner
reference recordings — on all of those they are statistically identical. It is
the same "some people are easier to read" effect, just clustered by ID order.

Two consequences:

1. **The per-fold numbers in the existing CV report compare five different groups
   of people, not five repetitions of the same experiment.** The pooled number
   over all 1260 sessions is still valid; the per-fold column should not be used
   to judge model quality.
2. The fix is small: deal subjects out round-robin (or stratify on sex and number
   of sessions) instead of splitting the sorted list into blocks.

## Finding 3 — a small part really is the breathing signal

About 2 % extra, beyond the person effect: sessions where the breathing is
regular and clean, with steady depth and little slow drift, are reconstructed
better. Sessions where the depth wobbles from breath to breath are worse.

What does **not** matter at all: the breathing **rate**, and how many breaths fit
into the session. Slow or fast, it is the same.

## What was ruled out

| guess | result |
|---|---|
| head movement | essentially no effect (largest correlation ≈ 0.05–0.12) |
| crop inflated by movement | no effect |
| face size / distance from camera / pixels on the face | no effect |
| spatial contrast of the thermal image | no effect, despite being the most stable property of a person (nearly identical across all ten of their tasks) |
| session length, clip count, landmark tracking quality | no effect |
| breathing rate, number of breaths | no effect |
| which fold / model | real, but confounded with sex (Finding 2) |

The only weak thread on "why is this person readable" is how saturated the red
channel is in the nose/mouth patch — more saturated goes with better scores. That
is suggestive, not established, and I would not build a plan on it.

## What this means in practice

- **Stop looking for a way to filter out bad T3 sessions.** There isn't one: the
  sessions that fail are not distinguishable from the ones that succeed by
  anything measurable inside the session.
- **Report T3 per subject, not per session.** Two thirds of the variation is
  already fixed by who the person is. This is the quantitative justification for
  the subject-as-the-unit protocol the project already uses elsewhere.
- **The remaining room for improvement is the model**, not the data selection.
  Improving the T3 average means improving the encoders.
- **Fix the fold split** before comparing folds or using per-fold numbers to
  choose between models or hyper-parameters. Right now a "fold effect" is largely
  a sex/recording-cohort effect.

## Caveats worth remembering

- One session per subject per task, so there is no way to see the same person on
  the same task twice. The "person effect" is inferred from their other tasks,
  which is strong evidence but not a repeat measurement.
- Which model produced a subject's prediction cannot be separated from which
  person it was, because every session of a subject is scored by the same fold
  model. The findings above are consistent with either a partly weak encoder or a
  partly hard cohort; the design cannot tell them apart. Fixing the split would
  make that separable in the next sweep.
- "Quality" throughout means the correlation of the breathing rhythm after drift
  removal. A waveform can track the rhythm correctly and still have the wrong
  amplitude or shape; the separate amplitude numbers say the predictions are
  typically about three times too small.
- The previously highlighted example `M038_T3` looked excellent (+0.95) in the
  single earlier run but scores −0.26 in the cross-validated runs. Those are
  different models; only the cross-validated number is a fair estimate.
