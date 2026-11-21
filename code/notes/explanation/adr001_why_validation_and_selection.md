# Why does the protocol need a validation role, and what is `val_select` for?

*Plain-English rationale, written 2026-11-19, for the two questions that keep
coming back when reading `design/ADR-001-subject-disjoint-split-and-selection.md`:
what a validation set is actually for, and why the selection rule is an explicit,
recorded parameter rather than something the code just does. The normative text
is the ADR; the runbook is `plan/experiments/kfold_run_checklist.md`; the operational spec is
`spec/04_configs_and_execution.md`. Nothing here overrides those — this document
explains **why** they say what they say, and records the one place where the code
is currently looser than the ADR.*

---

## 1. The short version

A validation set exists for one reason: **it is the only data you are allowed to
look at more than once.** Everything else follows from that.

`val_select` exists because a multi-epoch run does not produce *one* model — it
produces `E` of them, and the thesis number belongs to exactly one. Naming and
recording the rule that picked it is what makes the reported artefact
identifiable and the run reproducible.

---

## 2. Why a validation role at all

Supervised training partitions data by *what you are permitted to do with it*:

| Set | Permitted use | Why the restriction |
|---|---|---|
| train | gradient updates | the loss is optimised against it, so its value is in-sample and biased optimistic |
| validation | any number of **observations**; never a gradient update | a proxy for the reported distribution that tolerates repeated peeking |
| test | **one** observation, at report time | the value is only honest if nothing was chosen by looking at it |

The train set cannot answer "is this working?", because it is the quantity being
minimised. The test set cannot answer it *repeatedly*, because each look is a
comparison and comparisons accumulate into selection — that is how test sets get
overfit. The validation role absorbs all of that looking so the test role can stay
a single clean shot. ADR-001's I5 ("the test role is never loaded during
training") and I9 ("the validation block … is used at most once per decision") are
the two halves of that firewall.

### What it buys, in order of importance

1. **Decision separation (why it is mandatory).** Every change you make — a loss
   weight, a masking ratio, an epoch budget, a head — must be paid for out of val,
   never test. Without a val role there is no way to iterate: either you never
   improve, or you improve by consulting the set you claim is unseen. ADR §1: with
   no validation role "the only available signal is the test block, so using it
   for any decision amounts to selecting on the test set."
2. **Selection.** `best_val` uses it to pick the delivered epoch. Under the
   default `fixed_epoch` this job is switched off: the curve is observed, but
   nothing is chosen by it.
3. **Monitoring / diagnostics.** In this thesis this is the *dominant live use*,
   because I8 says outright that validation values are diagnostics, not reported
   results. Concretely it buys:
   - **collapse detection** — the documented failure here is the constant
     predictor, where `mse_resp` plateaus at the target variance (~1.0); a flat
     val curve at that level says the head learned nothing, without spending a
     test read;
   - **generalisation gap** — val vs train loss is the only overfitting signal
     that exists when 97 subjects are trained on and 14 watched;
   - **per-stream diagnosis** — `mse_<stream>` / `kept_<stream>` separate "TIR is
     fine, respiration is dead" from "everything is dead", which one scalar hides;
   - **implementation bugs** — a val loss *below* train loss, or a
     suspiciously good val curve, is the cheapest detector of leakage and of
     double normalisation;
   - **cheap arm comparison** — comparing candidate configurations on val spends
     10 % of the data instead of the one test read.
4. **Licensing.** `best_val` is a *claim* that the val criterion predicts the test
   ranking; the val role is where Criterion 5 gathers the evidence for it.

### What it is not

Not the headline. I8 defines the reported metric as the task's pre-declared metric
on the **test** role, aggregated per subject with a subject-level interval. Val
values are not pooled and not cited as results. If a val number entered the
report, the reader would have to ask whether it was selected on, and the design
would collapse back into the leakage it exists to prevent.

### Why 10 %, and the caveat that follows

ADR §2.4 rejects 20 % ("removes twice as much training data without sufficient
benefit") and rejects monitoring on test outright. But 10 % of the 139-subject
universe is **14 subjects** (verified: 8 F / 6 M), so the val criterion is a
high-variance estimate — roughly
`sigma_val ≈ sigma_per-subject / sqrt(14) ≈ 0.27 * sigma_per-subject`. Consequences:

- as a **monitor**, 14 subjects are ample — you are looking for a curve that is
  flat, diverging, or off by a factor, not third-decimal differences;
- as a **selector**, 14 subjects are thin. This is exactly why `best_val` is not
  the default and why Criterion 5 demands agreement in *two* fold-paired
  comparisons with a bootstrap interval excluding zero. A rule that takes the
  argmin of a noisy 14-subject criterion will sometimes take noise.

### Why the val pass must be deterministic (I6, Criterion 2)

A monitoring signal is only interpretable if consecutive epochs are measured the
same way. With stochastic masking, "val loss went up" is ambiguous between the
model getting worse and a different mask being drawn. Hence the pinned
`val_kept_<stream>`, fixed sample order, FP32, and `mask_seed = base + batch_idx`
(identical masks every epoch). Criterion 2 protects the *interpretation of the
curve*, not just tidiness.

---

## 3. What `val_select` is, and why it is an argument

In standard ML vocabulary it is the **model-selection rule**, specifically the
*checkpoint*-selection policy.

| Value | Standard name | What it does |
|---|---|---|
| `fixed_epoch` | fixed budget / train-to-completion | deliver epoch `E`; no data-driven choice, so no selection cost |
| `best_val` | early stopping with restore-best | deliver the argmin/argmax of the val criterion over the `eval_freq` grid |

### Why it cannot be hardcoded

1. **The rule is evidence.** D11 requires every artefact to state its compliance;
   `selection.json` stores `rule`, `eval_freq`, `selected_epoch`, `criterion`,
   `direction`. A hardcoded rule makes that field a constant — the record would
   "prove" `fixed_epoch` by construction and prove nothing.
2. **Criterion 5 must be runnable.** If `best_val` were not expressible, the study
   could neither earn the rule nor reject it.
3. **The branches differ in artefacts.** `best_val` writes `best.pth`,
   `fixed_epoch` never does; `best_val` with `eval_freq: 0` is refused. Untestable
   if unexpressible.
4. **It is not a hyperparameter.** A hyperparameter changes how well the model
   does; `val_select` changes *what the reported number means* — which estimator
   produced it and which data it was allowed to consult. That is why it sits in
   §2.3's interface next to `n_folds`, not in an optimiser block.

`K` and `70/10/20` are constants because they must never vary. The selection rule
is different in kind: it is *allowed* to vary, conditional on evidence, and the
default value encodes the current state of that evidence.

### Why the default is the conservative one

Unrecorded selection is hidden multiplicity. Reporting the best of `E` epochs is
`E` implicit comparisons against data; if made on the test block, the estimate is
optimistically biased (winner's curse):

```
E[ min_e  theta_e(D_test) ]  <  min_e  E[ theta_e(D_test) ]
```

and the bias grows with the number of epochs tried. `val_select` makes the
multiplicity visible and bounded: either zero comparisons, or comparisons made on
`D_val` only. `fixed_epoch` costs no decision, so it needs no
justification; `best_val` spends one and only spends it well if the criterion is
*stable* (Criterion 2) and *predictive* (Criterion 5).

Recording the rule is also the ML analogue of pre-registration: it removes the
option to rationalise afterwards which epoch "really" counts. ADR §4 states the
minimum: "a run is reproducible only when its selection rule is recorded".

---

## 4. Status of the implementation (verified 2026-11-19)

- **`fixed_epoch`** is the default in `runners/_common.py::add_protocol_args`
  and in both stage runners; `best.pth` is written only by
  `protocol.SelectionTracker.writes_best_checkpoint`.
- **Criterion 5 is not built.** No arm study exists, so `best_val` is unearned and
  the headline does not use it.
- **Known looseness:** `runners/run_evaluate_kfold.py::gate_test_artefacts`
  enforces that the five folds agree on **one** rule, but never checks that the
  agreed rule is *licensed*. A five-fold `--val_select best_val` sweep would
  therefore score, pool and print. Compare `dev_smoke`, which the partition gate
  refuses outright. Until Criterion 5 exists, "`best_val` is not the headline" is
  enforced by the default value and by the ADR text, not by the pooler.
- Closing that would take a `POOLABLE_RULES = ('fixed_epoch',)` constant plus one
  refusal in the pooler — but note the veto is **stricter than the ADR**, which
  only says the rule *reverts* if Criterion 5 fails. Making it normative needs one
  sentence added to Criterion 5 in the ADR.
