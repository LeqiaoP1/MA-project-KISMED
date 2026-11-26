# ADR-001: Subject-Disjoint 5-Fold Split with 70/10/20 Roles

- **Status**: Accepted
- **Date**: 2026-11-18
- **Author**: Ne95ocyg
- **Related Spec/Plan**: `code/notes/plan/features/ADR-001-subject-kfold-plan.md`, `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md`
- **Supersedes**: `N/A`

## 1. Context and Problem Statement

The thesis claim concerns **subjects the model has never seen**. A run supports this claim only if
the subject being scored is never used for fitting and every training decision is based on data
from subjects disjoint from the test subjects. Without a validation role, the only available
signal is the test block, so using it for any decision amounts to selecting on the test set.

This protocol is independent of the downstream task, model, and data modality. It applies to every
stage of every run. Implementations may vary, but they must satisfy this protocol.

## 2. Decision

Use one protocol for every stage of every run: subject-disjoint 5-fold cross-validation with
train, validation, and test roles covering 70%, 10%, and 20% of the subject universe,
respectively. Validation evaluations are deterministic, and the selection rule is explicit and
recorded. The test role is evaluated once, at report time.

### 2.1 Constants

These are thesis-level properties, not hyperparameters. They must never be tuned. Changing either
constant requires a new ADR.

| Constant | Value | Meaning |
|---|---|---|
| Fold count `K` | **5** | The subject universe is divided into five near-equal blocks. Block `k` is the test role for fold `k`. |
| Role ratio | **70 / 10 / 20** (train / validation / test), by subject | The test role is one block (`1/K`). The validation role contains 10% of all subjects selected from the four non-test blocks. The train role contains the remainder. |

Rounding may make a role one subject larger or smaller than its nominal share. The realised counts
are recorded in the run artifact and used in the report.

### 2.2 Invariants

| ID | Must hold |
|---|---|
| **I1** | Roles are disjoint at the **subject** level. Every sample inherits its subject's role. |
| **I2** | The partition is a pure function of the usable subject universe, partition seed, stratifier, `k`, and `K`. It must not depend on caps, task selection, or filter thresholds. It is identical across arms and resumes of the same fold. |
| **I3** | For fold `k`, the test role is block `k`, the validation role contains 10% of the universe drawn from the other four blocks, and the train role contains the remainder. Realised role counts sum to the size of the subject universe. |
| **I4** | Every stage of fold `k` uses the same three roles. Training uses the train role, validation is the only selection signal, and the test role remains untouched until reporting. |
| **I5** | The test role is never loaded during training. The validation role is loaded only for evaluation and never contributes to a gradient update. |
| **I6** | Validation is deterministic. Every stochastic part of the forward path, such as random masking, is pinned; the seed and sample order are fixed; and the pinned settings are recorded for each evaluation. |
| **I7** | The selection rule is explicit and recorded: `fixed_epoch` (default) or `best_val`, which reads only the validation role. The selected epoch and criterion value are recorded. |
| **I8** | Validation values are diagnostics, not reported results. The reported metric is the headline metric defined for the task before the run. It is computed per subject and aggregated with a subject-level confidence interval. |
| **I9** | The validation block is fixed across the arms being compared and is used at most once per decision. |

### 2.3 Interface

| Parameter | Values | Meaning |
|---|---|---|
| `n_folds` | fixed at `5` | Protocol constant, not a tunable parameter. |
| `fold` | `0 ... K-1` | Selects the block used as the test role. |
| `split_seed`, `split_stratify` | integer, `none`, or `sex` | Determine the partition (I2). |
| `eval_freq` | integer; `0` = off | Number of epochs between validation evaluations. |
| `val_select` | `fixed_epoch` (default) or `best_val` | Selects the rule described in I7. |
| `val_kept_<stream>` | float in `[0, 1]` | Pins the masking fraction for the validation pass when the model uses stochastic masking (I6). |

Each run writes a **partition record** containing `k`, `K`, the seed, the stratifier, the three
role lists, and the realised counts. It also writes a **selection record** containing `rule`,
`eval_freq`, `selected_epoch`, and the criterion value. Each validation evaluation records
`{epoch, pinned settings, seed, n_samples, per-component loss and counts, validation criterion}`.

### 2.4 Rejected Alternatives

- *No validation role; monitor the test reconstruction curve*: This selects on the test block.
- *Validation subjects from the test fold, or validation at sample or session level*: This breaks subject-level disjointness.
- *A 20% validation block*: This removes twice as much training data without sufficient benefit.
- *Validation under the run's stochastic forward path, such as random masking*: The resulting curve is not comparable across epochs.
- *Any fold count other than five, or any other role ratio*: This requires a separate ADR and is not a run-time choice.

## 3. Falsifiable Criteria

1. **Split integrity.** Roles are pairwise disjoint, and their realised shares are within +/-2 percentage points of 70/10/20. Two independent constructions of the same partition are byte-identical. A partition built with development caps is identical to one built without caps. Reject any overlap or cap-dependent change.
2. **Evaluation determinism.** Two validation evaluations of the same checkpoint produce bit-identical metrics when the recorded settings and seed are the same.
3. **No leakage.** If a sample whose subject belongs to a held-out role reaches a gradient update, the run aborts.
4. **Cost.** A validation evaluation costs less than 5% of the run's wall-clock time and increases peak allocated VRAM by less than 5%. Both limits are measured on a short pilot.
5. **Selection validity.** In at least two fold-paired arm comparisons, the arm ranked higher by the final validation criterion must also be ranked higher by the headline metric on the test role. Use a paired bootstrap confidence interval on per-subject differences, excluding zero. If both comparisons disagree, `val_select` reverts to `fixed_epoch`, and validation remains diagnostic only.

## 4. Consequences and Trade-offs

- Every stage follows one auditable protocol. Selection is explicit, and the test role is read only at reporting time.
- The training corpus contains 70% of the subject universe instead of 80%, and the same 10% is unavailable to every stage. This is accepted because an unmonitored stage poses a greater risk to the thesis claim than a smaller training corpus.
- Results produced with another split are not comparable and must be rerun or explicitly labelled.
- Validation can be over-tuned; I9 limits this risk. Because `best_val` changes which epoch is delivered, a run is reproducible only when its selection rule is recorded.

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A` - the protocol was established before any run.
- **Evidence artifacts**: A partition record and selection record for each run, plus headline-metric artifacts from the reported evaluation.
- **Checks**: Split-integrity test (Criterion 1), determinism test (Criterion 2), leakage abort (Criterion 3), and short pilot for Criterion 4.
