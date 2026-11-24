# ADR-006: Visual-Only Respiration Conditioning and Primary-Stream Saturation in the Sensor-Absent Regime

- **Status**: **Alternative — not adopted.** The ADR of record is `ADR-006-three-regime-virtual-sensing-masking.md`. This file is kept only for its evidence analysis and its two pre-flight probe designs (its §3.1 P1/P2), which remain useful whatever policy is chosen.
- **Date**: 2026-11-24
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/design/Evaluation_RespWaveform_protocol.md` — normative for the headline metric of §3.2, unchanged by this ADR. Build order and gates: `code/notes/plan/features/ADR-006-visual-only-resp-conditioning-plan.md` (to be written; not an authority). Registered run protocol for the arms of §2.4: `code/notes/plan/experiments/exp_20261124_adr006_tir_primary_rgb2d_ablation.md` (to be written; not an authority).
- **Supersedes**: `code/notes/design/ADR-005-asymmetric-multimodal-masking-3streams.md` — **amdement scope, two clauses only**: §2.3 contract 7 (conditioning) and §2.4/§2.7 (Mode-1 allocation and the floor's scope). Everything else in ADR-005 — the stream set, the tokenizer geometry, the constant budget, the caps, the alignment arithmetic, the per-step/per-sample split, determinism, loss keys, initialisation contract I1–I4, and the acceptance protocol — is **inherited verbatim** and is not restated here. Where this document is silent, ADR-005 governs.

**Why an ADR at all.** The ADR-005 arm was trained and measured; the measurement says the sensor-free pathway is roughly an order of magnitude weaker than the sensor-present one, and that half the visual budget is spent on a stream whose content the decoder does not use. Both facts are traceable to two clauses of ADR-005, not to tuning. Amending those two clauses is a decision, so it is recorded as one.

## 1. Context & Problem Statement

### 1.1 The measured outcome of the ADR-005 arm

Fold 0, `checkpoint-0149.pth`, `project_multimae_base` (105,581,600 parameters), VideoMAE-k400 kernel routed to the TIR adapter, curriculum $p = 0.10$ for epochs 0–49 and $p = 0.50$ for 50–149, constant budget $n_{\text{budget}} = 350$. Scored on the fold-0 `test` role (subject-disjoint): 1,594 clips / 201 sessions / 27 subjects. All values are the masked-position Pearson $r_{\text{masked}}$ on the dataset's per-clip respiration z-score; subject aggregation is the median over a subject's clips with a bootstrap 95 % CI over subjects. Source: `code/notes/eval/baselines/tir_rgb2d_resp/` (`README.md`, `resp_recon_arms_ab.md`).

| Condition | Allocation (tokens) | `r_masked` mean | Subject median [95 % CI] | MSE (z²) |
| :--- | :--- | ---: | --- | ---: |
| Respiration visible, `scattered` | tir 150 / rgb2d 75 / resp 125 | +0.9959 | 0.9964, 0.9964, 0.9976 (3 subjects) | 0.0090 |
| Sensor absent, TIR-max | tir 225 / rgb2d 125 / resp 0 | +0.1091 | **+0.1129** [+0.0775, +0.1363] | 1.0408 |
| Sensor absent, RGB2D-max | tir 125 / rgb2d 225 / resp 0 | +0.07995 | **+0.0919** [+0.0705, +0.1085] | 1.0590 |

Mode 1 is 50.2 % of the sampled steps and its mean split is `tir 174.7 / rgb2d 175.3`, i.e. the training distribution is centred on the 7+7-tube split, which is neither of the two measured extremes.

Three measured facts drive this decision:

1. **The sensor-free pathway is real but weak.** The subject CI excludes zero, and the per-clip tails are asymmetric (25.8 % of clips at $r > 0.3$ against 8.1 % at $r < -0.3$), so it is not noise. It is also an order of magnitude below the sensor-present condition and its predicted amplitude is collapsed ($\text{MSE} \approx 1$ z², i.e. about the variance of the target).
2. **Reallocating budget to the second visual stream hurts.** Paired over the same 1,594 clips and the same five mask draws: $\Delta(\text{B}-\text{A}) = -0.0292 \pm 0.0044$, $t = -6.62$; 876 clips worse, 19 of 27 subject medians worse.
3. **The second visual stream's content is not used.** Holding the plan and masks fixed and corrupting only the input: rolling RGB2D by 16 of 50 frames changes $r_{\text{masked}}$ by $-0.0012$ (s.e. 0.0026, $t = -0.47$) and blanking it by $-0.0028$ (s.e. 0.0064, $t = -0.45$), while the **positive control** — the same roll applied to TIR, which carries 5 of the 14 visual tubes — removes two-thirds of the signal ($-0.0539$, s.e. 0.0054, $t = -9.96$) and leaves the remaining subject-level correlation indistinguishable from zero.

### 1.2 The two clauses this produces

**Bottleneck H1 — the decoder prefix makes the inpainting step intra-modal.** ADR-005 contract 7 permits a stream's decoder to prefix *the full encoder latent sequence*, which includes that stream's own visible tokens. With 125 of 250 respiration tokens visible the decoder reconstructs the masked gaps at $r = 0.996$ — it does not need the visual latents to do so. The consequence is that on the 49.8 % of steps where the respiration target is partly visible, the respiration loss reaches $z_{\text{visual}}$ only through a weak attention path that has a strictly better intra-modal alternative; only Mode 1 steps exert real pressure on the visual streams, and they are also the steps whose loss is hardest to reduce. Half the training signal is therefore spent on a task the model can solve without the representation the objective is supposed to build.

**Bottleneck H2 — the sensor-absent regime spends half the budget on an unused stream.** The 2-tube floor of ADR-005 contract 3 forces at least 5 tubes of the auxiliary stream on every Mode-1 step, because a single visual stream caps at 9 tubes (25 % of 36) and 14 tubes must be allocated somewhere. The floor's stated purpose — "leave the encoder without visual evidence" — is already satisfied by 350 tokens of TIR. The measured cost of the floor is facts 2 and 3 above: 125–175 tokens per step that the ablation shows are not used, taken from the one stream that is.

**Bottleneck H3 — the RGB2D question has no matched training arm.** Both measurements behind facts 2–3 hold one frozen checkpoint fixed, so they establish that the ADR-005 arm does not *use* RGB2D at inference. They cannot establish whether a stream the model was never required to use could have helped: the RGB2D adapter is randomly initialised (ADR-005 I2), and the source record states the limit itself — "a training-time benefit cannot be excluded without a 2-stream arm" — while no such arm exists. Closing this needs matched training arms (§2.4), not another probe on the same weights.

**H1 and H2 are hypotheses, and both are cheap to test before any epoch is spent** (§3.1). H3 is a measurement gap rather than a hypothesis: it is closed by the ablation of §2.4, which this ADR registers as part of the arm rather than as a follow-up someone may or may not run.

### 1.3 What this ADR does *not* change

The deployed sensing modality is TIR (ADR-005 §1.1, §2.1), so TIR is the primary visual stream here. The stream set, the budget, the caps, the floors, the alignment arithmetic, the tokenizer geometry, the initialisation contract and the curriculum stay exactly as ADR-005 states them. `signal_weight` stays 0.5 and RGB2D stays in the budget on steps where the respiration target is visible, so that the rerun isolates the two clauses of §2 and nothing else.

## 2. Decision & Parameter Exposure

**Two amendments to ADR-005, both default-off so that every existing configuration reproduces its previous behaviour exactly.**

### 2.1 D1 — Primary-stream saturation in the sensor-absent regime

On a Mode-1 step (the resolved mode of ADR-005 contract 4, $K_{\text{resp}} = 0$), the budget is allocated to **one designated primary visual stream** instead of being split:

$$S_{\text{primary}} = \frac{n_{\text{budget}}}{T} = 14 \text{ tubes} \quad (350 \text{ tokens}), \qquad S_{\text{aux}} = 0, \qquad K_{\text{resp}} = 0.$$

This is the only place the 2-tube floor is waived, and it is waived on one side only: the auxiliary stream keeps **zero** tokens, the primary keeps 350. The floor of ADR-005 contract 3 is retained in full on every step on which respiration is visible, where it still prevents the visual budget from rounding to zero.

Normative consequences:

- **No cap is required on this step**, because the allocation is fixed rather than drawn. The binding precondition is a geometry fact: the primary stream must have at least $n_{\text{budget}}/T = 14$ tubes, which holds at 36.
- The resulting kept fraction is recorded, not tuned: $14/36 = 38.9 \%$ kept, $61.1 \%$ masked. It is a weakening of ADR-005's 25 % cap, and the pretext bound it buys is still 22 masked tubes of 36 per step at the full 350-token budget. The measured MSE of the arm this reproduces is 1.0408 z², i.e. the full visual reconstruction target is intact on those steps.
- The auxiliary stream's **decoder remains a reconstruction target** on these steps and receives its gradient through the conditioning of §2.2; only its adapter and encoder tokens are absent. ADR-005 contract 9 (one loss key per declared stream per step) is unchanged.
- Steps on which respiration is visible keep the ADR-005 split verbatim: $S_{\text{total}} = (350 - K_{\text{resp}})/25 \in [9, 13]$, $S_{\text{primary}} \in [\max(2, S_{\text{total}}-M), \min(M, S_{\text{total}}-2)]$.

*Why TIR is the primary and RGB2D the auxiliary.* It is the deployed modality (§1.3), and it is the stream the measurements attribute the whole sensor-free signal to (§1.1 facts 2–3). The designation is a policy parameter, not an architectural constant; a run that swaps it is a different arm and must record the swap.

### 2.2 D2 — Visual-only conditioning for the signal decoders

The conditioning set of a stream's decoder becomes a function of the stream kind. With $\mathcal{V}$ the declared visual streams and $\mathcal{S}$ the declared signal streams:

$$C(m) = \begin{cases}
\mathcal{V} \cup \mathcal{S} & m \in \mathcal{V} \quad \text{(unchanged: all latents)} \\
\mathcal{V} \cup \mathcal{S} & m \in \mathcal{S},\ \texttt{signal\_cond\_scope} = \texttt{all} \quad \text{(ADR-005 behaviour)} \\
\mathcal{V} & m \in \mathcal{S},\ \texttt{signal\_cond\_scope} = \texttt{visual}
\end{cases}$$

So a signal stream's decoder still receives the full encoder latent sequence when the scope is `all`, and **only the visual-stream latents** when the scope is `visual`. Visual decoders are unchanged in both cases. Mechanism (a) of ADR-005 contract 7 (prefixing) is retained; only its *scope* is narrowed, which is why this is an amendment and not a new conditioning mechanism.

The upgraded, checkable gradient condition replaces ADR-005 contract 7's zero-token case:

> For every signal stream $s$ and **every** step — including steps with $K_s > 0$ — $\partial \mathcal{L}_s / \partial z_{\mathcal{V}} \neq 0$ and $\partial \mathcal{L}_s / \partial z_{\mathcal{S}} = 0$.

The second half is the new part: on a step where the encoder holds respiration tokens, they may influence the waveform target only through attention into the visual latents. Their influence on the visual decoders is untouched.

*Risk recorded, not mitigated.* Phase-I grounding (ADR-005 §2.7) assumed that surviving respiration tokens act as a biological phase anchor for the visual latents. Under D2 that anchor now reaches the target only *through the encoder*, which is precisely the pressure the decision wants to create — but it is also a real change to the early-phase learning dynamics. If Phase-I loss diverges or the visual reconstruction loss stops improving, the counter is a lower `p_warm` in a new decision, not a silent edit here.

### 2.3 D3 — Curriculum held constant

$p_{\text{warm}} = 0.10$ for epochs $e < E_{\text{warm}}$, $p_{\text{target}} = 0.50$ from $E_{\text{warm}}$ onwards, with $E_{\text{warm}} = 50$ and a **0-based** epoch index (`e < E_warm`). Unchanged from ADR-005 §2.7, deliberately: the rerun must attribute any movement to D1 and D2, not to a third change. The `p_warm <= p_target` validation, the fixed-seed validation mask sequence and the "never pool the validation curve across $e = E_{\text{warm}}$" rule all carry over.

### 2.4 D4 — The RGB2D contribution study is part of the arm, not a follow-up

The question *could RGB2D improve respiration reconstruction* is registered here with three matched arms, because H3 (§1.2) says it is otherwise unanswerable. All three run on the same fold, the same recipe, the same seed and the same budget, and all three carry D1 + D2.

| arm | `streams` | `mode1_primary_stream` | `signal_cond_scope` | `loss_weights` | what the arm isolates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`3s`** (main) | `tir,rgb2d,resp` | `tir` | `visual` | default | the ADR-005 stream set under D1+D2 — the thesis arm (Target A) |
| **`3s-ctx`** | `tir,rgb2d,resp` | `tir` | `visual` | `tir=1, rgb2d=0, resp=0.5` | RGB2D as **encoder context only**, budget unchanged, its own reconstruction objective removed |
| **`2s`** | `tir,resp` | `tir` | `visual` | default | RGB2D removed, and the tokens it used given to TIR |

**Why the ablation is clean under D1.** All three arms are *identical on every Mode-1 step* — `tir 350`, respiration absent, and RGB2D absent wherever it is declared. The sensor-free condition on which the headline is scored is therefore a common input in all three arms, and the arms can differ only in what the shared encoder learns during respiration-visible steps. The contrast answers the question asked and confounds nothing about the deployment condition.

**Interpretation rules, fixed before any run.**

- `3s` vs `2s` — the total value of the second visual stream, with its tokens, its context and its reconstruction term removed together.
- `3s` vs `3s-ctx` — the value of RGB2D **as encoder context at a constant budget**. This is the only arm pair that can show "RGB2D helps RESP" without a token-reallocation confound, and it is the primary answer to the question.
- `2s` vs `3s-ctx` — the value of the 125 tokens currently spent on RGB2D when they are reassigned to TIR.
- A gain that appears in `3s-ctx` but not in `2s` means RGB2D carries target-relevant information while its own reconstruction objective competes with the target for capacity. A gain that appears in `2s` but not in `3s-ctx` means the stream earns its keep only by freeing budget, and should be dropped.
- **No arm may be dropped from the report.** A null on `3s-ctx` is the honest answer to the question and must be recorded as one.

**Stated limits, so that no result is over-read.** (i) All three arms route the VideoMAE kernel to TIR (ADR-005 I2), so none of them tests RGB2D under a pre-trained tokenizer; a `patch_stream: rgb2d` swap is a fourth, deferred question that the existing config key already permits. (ii) `2s` and `3s-ctx` change the *set* of reconstruction terms the encoder is optimised against, so a difference between them is not purely an input-change effect; this is why `3s-ctx` — not `2s` — is the registered answer to the content question. (iii) One fold, one seed per arm: the contrast is reportable with a subject-level CI over the fold's held-out subjects, not as a claim about population variance.

### 2.5 D5 — RGB-only single-modality sensing is deferred, not adopted

The draft that preceded this ADR proposed a third regime in which **either** visual stream may be the sole carrier, including RGB-only. That half is **not adopted** and is not part of this arm:

- allocating *more* budget to RGB2D measurably *reduces* the sensor-free correlation (§1.1 fact 2);
- corrupting RGB2D's content changes nothing, while the TIR control collapses the signal (§1.1 fact 3);
- and the RGB2D adapter is randomly initialised (ADR-005 I2), so a claim about that stream's capability could not be separated from its unconverged tokenizer.

Target C (RGB-only) therefore remains a stated downstream target with no pre-training support, which is a recorded gap. Re-opening it requires an arm that first fixes the RGB2D tokenizer, and that is a separate decision.

### 2.6 Normative apportionment algorithm (delta to ADR-005 §2.4)

Only the Mode-1 branch changes; the Mode-2 branch is ADR-005 §2.4 verbatim.

```
# parameters: as ADR-005 §2.9, plus
#   primary = mode1_primary_stream (name of a declared visual stream, or None)
#   aux     = the other declared visual stream

function SAMPLE_STEP(e):
    if uniform_real(0, 1) < p(e):
        if primary is None:                    # ADR-005 behaviour
            S_total = N / T
            S_p, S_a = SPLIT_TUBES(S_total)     # clamped-uniform, floor m, cap M
        else:                                  # D1
            S_p, S_a = N / T, 0
    else:
        ...                                    # ADR-005 §2.4 Mode 2, unchanged
        S_p, S_a = SPLIT_TUBES(S_total)        # floor m and cap M still apply
    assert K_p + K_a + K_resp == N
    return mode, (K_p, K_a, K_resp), (S_p, S_a)
```

### 2.7 Feasibility preconditions (delta to ADR-005 §2.5)

ADR-005 conditions 1–3 are unchanged and still validated at construction. Three are added, all rejected at configuration-resolve time rather than at the first step:

4. `mode1_primary_stream`, when set, must name a declared **visual** stream, and that stream must satisfy $\text{units}_v \ge n_{\text{budget}}/T$ (36 ≥ 14 ✓). A named signal stream, an undeclared stream, or a tube grid too small to hold the whole budget is an infeasible configuration.
5. `signal_cond_scope = visual` requires at least one declared visual stream (implied by ADR-005 condition 2, asserted separately because the conditioning set would otherwise be empty).
6. A set with **one** visual stream is admissible only when `mode1_primary_stream` names it, and then `video_cap_ratio` must satisfy $M \ge n_{\text{budget}}/T$ (i.e. $\ge 14/36 \approx 0.3889$), because the single stream must absorb both the fixed Mode-1 allocation and the whole Mode-2 residual $S_{\text{total}} \in [9, 13]$. A single-visual set without a designated primary is the ADR-005 infeasible case and is still rejected.

### 2.8 Parameter surface (delta to ADR-005 §2.9)

| parameter | type | default | admissible | meaning |
| :--- | :--- | ---: | :--- | :--- |
| `--mode1_primary_stream` | str | `''` | `''` or a declared visual stream name | `''` = ADR-005 Mode-1 split; a name = D1 saturation of that stream |
| `--signal_cond_scope` | enum | `all` | `all`, `visual` | `all` = ADR-005 contract 7; `visual` = D2 |

Both default to the ADR-005 behaviour, so an existing configuration without either key is bit-identical to the arm that produced §1.1. Every other row of ADR-005 §2.9 — `n_budget`, `tokens_per_unit`, `video_cap_ratio`, `physio_cap_ratio`, `video_min_tubes`, `p_virtual_sensing`, the two warm-up keys, `physio_placement`, `physio_span_tokens`, `spatial_patches`, `physio_tokens` — keeps its meaning unchanged, and the geometry rows are unchanged (the caches stay valid). Two rows are *re-parameterised by the arm*, not by a new flag:

- `streams` — a single-visual set (`tir,resp`) becomes admissible under feasibility rule 6 above; the 25 % `video_cap_ratio` is preserved verbatim for every multi-visual set and must be raised to at least $14/36$ only in the single-visual arm.
- `loss_weights` — already exposed per stream and unchanged; the `3s-ctx` arm of §2.4 sets `rgb2d=0` to remove that stream's reconstruction objective while keeping its tokens in the encoder.

Every parameter that appears in §2.4's arm table is therefore an existing key of §2.9. The ablation adds no knob that could be tuned on its own.

### 2.9 Observability (delta to ADR-005 §2.8)

In addition to the ADR-005 record, every run must emit, per step: the resolved mode, the primary/auxiliary designation, and the conditioning scope actually used for each decoder. The masking record's `policy` block must carry both new keys, the declared stream set, the per-stream `loss_weights` and the realised Mode-1 allocation; a resumed run whose record disagrees on any of them is rejected, exactly as ADR-005 already requires for the rest of the tuple. Because §2.4 compares three arms, the arm identity must also be recoverable from the record alone — from the stream set, the weights and the two new keys — rather than from the output directory's name.

### 2.10 Rejected alternatives

- **Adopt the draft as written** (three regimes including RGB-only; `E_warm = 20` with a 1-based index and an `epoch <= warmup` test; cross-attention mandated; 38.9 % cap presented as a "saturation capacity" alongside an unchanged 25 % cap; acceptance at $r \ge +0.45/+0.40/+0.30$). Rejected: it restates ADR-005 rather than amending it; two of its clauses contradict each other; its thresholds are 3–4× the strongest measured arm and inverted in direction for RGB-only (§1.1); and its off-by-one and status conventions repeat defects ADR-005's Appendix already recorded. Only its TIR-only saturation idea survives, as D1.
- **Drop RGB2D without a matched arm.** Rejected: it is now the `2s` arm of §2.4, and dropping it on the strength of two inference-only probes is exactly the inference H3 forbids.
- **An RGB2D placebo arm** (keep the tokens, corrupt the input during training). Rejected: it spends a full arm's compute to demonstrate the absence of an effect, and `3s-ctx` already isolates whether RGB2D's *content* reaches the target at a constant budget.
- **Replace "could RGB2D help?" with a rate-only or HF-power proxy on the existing checkpoint.** Rejected: the evaluation protocol excludes amplitude and high-frequency proxies from comparisons, so no such number could be acted on.
- **Raise `signal_weight` instead of D2.** Rejected: it scales an already-available gradient rather than removing the intra-modal alternative that starves it, and it would confound the attribution of this rerun.
- **Mandate cross-attention (mechanism (b)) for the signal decoders.** Rejected as over-specification, for the same reason ADR-005 rejected it: narrowing the prefix scope of mechanism (a) is sufficient and cheaper, and the mechanism used must be *recorded*, not prescribed.
- **Keep the 7+7 average Mode-1 split and rely on the curriculum.** Rejected: the split is what facts 2–3 measure as harmful, and no schedule changes an allocation.

## 3. Falsifiable Criteria

### 3.1 Criterion 0 — pre-flight diagnostics (required before training; ~2 GPU-hours, no new epochs)

Both run on the **existing** ADR-005 fold-0 checkpoint with `runners/run_eval_recon.py`, pinned plans, 5 repeats, the same subject aggregation as §1.1. They exist so that the 150-epoch rerun is not spent on a hypothesis that a probe can refute.

- **P1 (tests D2 / H1).** Mode-2 plan with respiration visible (`--kept_resp 0.5 --resp_probe random`) and `--signal_cond_scope visual`. Pre-registered prediction: the subject-median $r_{\text{masked}}$ collapses from +0.996 toward the Mode-1 level of +0.11. **Gating:** if the drop is smaller than 0.5 in absolute value, the inpainting step already carries no intra-modal shortcut, H1 is false, and D2 must be withdrawn before training.
- **P2 (tests D1 / H2).** Saturated Mode-1 plan (`tir 350 / rgb2d 0 / resp 0`) against arm A (`tir 225 / rgb2d 125`, +0.1129). **Gating:** D1 is worth training only if P2 exceeds arm A by at least **+0.02** subject-median $r_{\text{masked}}$ — ADR-005's own materiality margin — because D1's only justification is that the tokens currently spent on the auxiliary stream are worth more on the primary.

Both are single-checkpoint, fold-0, zero-shot, per-clip z-scored and masked-position correlations. Neither is the headline metric, and neither may be reported as one.

### 3.2 Criterion 1 — policy conformance (behavioural; no learning required)

Over ≥ 10⁴ sampled steps: (i) $K_{\text{tir}} + K_{\text{rgb2d}} + K_{\text{resp}} = 350$ on 100 % of steps and the encoder input is `[B, 350, D]`; (ii) on every resolved Mode-1 step the allocation is exactly `tir 350` with every auxiliary visual at 0 and `resp 0`; (iii) every respiration-visible step keeps the ADR-005 count sets and the 2-tube floor on both declared visual streams (`3s`, `3s-ctx`), and gives the single declared visual stream the whole residual inside its cap (`2s`); (iv) counts are reproducible from `(seed, epoch, step)`, are identical across ranks, and placements differ per sample; (v) the Mode-1 fraction is $0.10 \pm 0.02$ before $E_{\text{warm}}$ and $0.50 \pm 0.02$ after; (vi) on a synthetic batch, $\partial \mathcal{L}_{\text{resp}}/\partial z_{\mathcal{V}} \neq 0$ on a step with $K_{\text{resp}} > 0$ under D2, and $\partial \mathcal{L}_{\text{resp}}/\partial z_{\mathcal{S}} = 0$; (vii) both new parameters are validated at configuration-resolve time, and an infeasible tuple of §2.7 is rejected there.

### 3.3 Criterion 2 — the scientific claim

- **Zero-shot gate (available now).** Mode-1 subject-median $r_{\text{masked}}$ on the fold-0 `test` role, same probe, same aggregation, must exceed the ADR-005 reference of **+0.1129** by at least **+0.02**. This is the only gate that can be evaluated without a Stage-3 run, and it is the one that decides whether the arm is worth fine-tuning.
- **Headline gate (protocol, after fine-tuning).** Band-limited Pearson (0.1–0.6 Hz), per session → median per subject → median with a bootstrap 95 % CI over subjects, subject-disjoint test roles, pooled across folds, against the **matched ADR-005 arm fine-tuned with the identical Stage-3 recipe**. Threshold: at least +0.02 above that arm, with the CI reported. That baseline number does not exist yet; the ADR fixes the comparison rule now so that it cannot be chosen after the fact.
- **Non-gating.** All masked-position reconstruction scores, including every number in §3.1 and §3.3, are pretext diagnostics. Amplitude metrics are excluded by the evaluation protocol and must not appear in a comparison.
- **RGB2D decision rule (D4), fixed in advance.** The three arms of §2.4 are scored with the identical Mode-1 probe, the identical fold and the identical aggregation, and are reported together with the paired subject-level CI of `3s-ctx` − `3s` and `2s` − `3s`. The selection rule, using ADR-005's +0.02 materiality margin: (a) if `3s-ctx` exceeds `3s` by ≥ +0.02, RGB2D's content reaches the target and its own reconstruction objective competes with it, so the next arm of record is `3s-ctx`; (b) else if `2s` exceeds `3s` by ≥ +0.02, the stream earns its place only through the budget it occupies, so the next arm of record is `2s`; (c) else `3s` remains the arm of record and RGB2D is recorded as inert for this target on this fold. No other comparison may be substituted for these two, and no arm is dropped from the report.

### 3.4 Criterion 3 — smoke and resource record

One epoch, warm-up disabled, `--max_clips 100`, batch 8, single device: completes with no assertion failure or `NaN`, prints the §1.1 geometry (36 tubes, 250 signal tokens, $T = 25$), emits every §2.9 record, and reproduces a saturated Mode-1 plan and a conforming Mode-2 plan **for each of the three §2.4 arms**. Record peak allocated and reserved memory at the production batch — the geometry is unchanged from ADR-005, so the ADR-005 figure is expected to transfer, but it must be re-measured rather than assumed.

### 3.5 Acceptance artefacts

Same artefact set as ADR-005 §3.5 (partition record, selection record, per-epoch metric log, pooled fold JSON with the subject bootstrap CI, per-subject metrics, masking record). The two §3.1 probe reports land under `code/notes/eval/` beside the ADR-005 baseline they are compared against.

## 4. Consequences & Trade-offs

- **Positive.**
  - Removes, rather than reweights, the intra-modal alternative that starves the cross-modal gradient on ~50 % of steps (D2), and makes that removal a checkable gradient condition rather than an architectural claim.
  - Stops spending 125–175 tokens per Mode-1 step on a stream the ablation shows is unused (D1). This is the only change that acts on measured facts 2–3 directly.
  - Both amendments are default-off, so the ADR-005 arm stays reproducible as a comparator and no existing configuration silently changes meaning.
  - The RGB2D question stops being an open-ended "we should check that" and becomes three registered arms with a selection rule fixed before the runs (§2.4, §3.3). Under D1 the ablation is also unusually clean: all arms share an identical sensor-free input, so the contrast cannot leak into the deployment condition.
  - The decision is falsifiable before it is expensive: two probes on an existing checkpoint can refute either half.
- **Negative / Technical debt.**
  - **D1 weakens the 25 % cap.** 61.1 % masking on Mode-1 steps is a lower pretext difficulty in the regime that carries the deployment task. If the visual reconstruction loss on those steps saturates, the cap is the first thing to revisit, in a new decision.
  - **D1 puts the whole sensor-free burden on one stream.** If TIR-only saturation does not beat the 225/125 arm, the auxiliary stream was not idle after all and D1 must be withdrawn.
  - **D2 changes Phase-I dynamics.** The respiration anchor now reaches the target only through the encoder; a warm-up that depended on direct latent access must be re-observed, not assumed.
  - **The RGB2D answer costs two extra arms.** `3s-ctx` and `2s` are full pre-training runs on the same fold; the ablation is cheap per run but not free, and it is the reason this ADR registers three arms rather than one.
  - **No arm can test RGB2D under a pre-trained tokenizer** (§2.4 limit (i)), so even a positive result is conditional on the random-init Conv3d. Target C has no pre-training support (D5); this is a recorded gap, not a plan.
  - **Attribution is partial.** D1 and D2 move together in every arm. Separating them needs the two one-clause arms, which the default-off parameters make cheap but do not make free.
- **Rollback / Rejection Condition.** Reject (i) D2 if §3.1 P1 fails its gate; (ii) D1 if §3.1 P2 fails its gate; (iii) the arm if Criterion 1 fails on the first full run, in which case the policy was not executed and no result is attributable; (iv) the arm if the zero-shot gate of §3.3 is not met, in which case ADR-005's conditioning and allocation are restored rather than partially patched; (v) an ablation arm may not be re-run to a different rule than §3.3's — a changed rule is a new decision.

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A` — not executed. The measurements of §1.1 come from `/work/scratch/ne95ocyg/pretrain/adr005_3stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth`, job record in `code/notes/eval/baselines/tir_rgb2d_resp/README.md` §0.
- **Evidence Artifacts**: `code/notes/eval/baselines/tir_rgb2d_resp/README.md`, `resp_recon_arms_ab.md`, `kept_ratio_distribution.md`, `artifacts/`. §3.1's two probe reports will be added there. All are informative; no clause of §§1–4 depends on them beyond the numbers cited in §1.1.
- **Tests / Smoke Checks**: §3.2 conformance (extend `code/tests/test_budget_masking.py` with the D1 Mode-1 allocation, its feasibility rejection, and the default-off equivalence; extend `code/tests/test_multimae_rgb2d.py` or the multimae suite with the D2 gradient condition), then §3.4.
- **Rollback / Rejection Condition**: §4.
- **Deferred, non-blocking decisions.** (1) Whether the RGB2D tokenizer should receive a pre-trained kernel before any claim about Target C, i.e. the `patch_stream: rgb2d` swap of §2.4 limit (i). (2) `signal_weight`, unchanged here on purpose. (3) Whether the primary designation should be revisited for a non-TIR deployment. None of these changes the policy defined in §2, and (1) is the only one that can change the §3.3 rule's conclusion.

--------------------------------------------------------------------------------

## Appendix A. Mapping to the present toolchain (informative, non-normative)

**D1.** `code/core/budget_masking.py::ConstantBudgetApportioner._mode1` currently splits the 14 tubes with `_split_tubes`, which clamps to `[m, M]`; `_plan` re-asserts that floor, and `_validate` rejects any tuple where the Mode-1 total cannot be split inside it. D1 therefore needs: a `primary` argument, a fixed-allocation Mode-1 branch, a floor check that excludes the zeroed auxiliary stream on that branch, and feasibility rule 4 of §2.7. `code/core/multimae.py` needs no change for D1 — a stream with `plan.kept_tokens == 0` already skips its adapter and keeps its loss key.

**D5 (single-visual arm).** `_validate` currently rejects `n_visual != 2` outright, and `_split_tubes` assumes two visual streams; `MultiModalMAE` already accepts one visual plus one signal. The `2s` arm therefore needs the `n_visual == 1` branch admitted under feasibility rule 6 of §2.7 (with the `M >= N/T` check), and `_split_tubes` bypassed when there is one visual stream. `--loss_weights` for the `3s-ctx` arm already exists (`code/core/multimae.py::build_model`), so that arm is config-only.

**D2.** `code/core/multimae.py::MultiModalMAE.forward` builds the decoder prefix as `dec = torch.cat([z, mask_tokens])`. D2 needs the per-stream conditioning mask over the encoder sequence, i.e. the slice offsets of the visual streams within `z` (available from `plan.kept_tokens` and the declared stream order, which is already the concatenation order). `code/runners/run_pretrain.py::write_masking_manifest` hard-codes `'conditioning': {'mechanism': 'prefix'}` and must record the scope; `run_eval_recon.py` needs the same flag so §3.1 P1 can be run on the ADR-005 checkpoint.

**Configs.** One sibling of `code/configs/pretrain/stage2_budget_3stream_curriculum.yaml` per §2.4 arm, each carrying `mode1_primary_stream: tir` and `signal_cond_scope: visual` and differing from it only by the delta the arm table states: `..._tir_primary.yaml` (`3s`, no further delta), `..._tir_primary_ctx.yaml` (`3s-ctx`, adds `loss_weights: 1.0,0.0,0.5`), `..._tir_only_2s.yaml` (`2s`, `streams: tir,resp` and `video_cap_ratio: 0.389`). Every other key identical — the same delta discipline that config already uses against its no-curriculum comparator.

**Probe path.** `code/runners/run_eval_recon.py::pin_plan_from_flags` clamps a visual count to `[floor, cap]` and then fills the residual to the constant budget, which is why arm A realises `tir 225 / rgb2d 125` even when `--kept_rgb2d 0.0` is requested. P2 of §3.1 needs the same fixed-allocation branch as D1, or it cannot express 350/0/0.

**Reference invocations.** Pre-training `code/runners/run_pretrain.py -c <new config>`; probes `code/runners/run_eval_recon.py`; pooled headline `code/runners/run_evaluate_kfold.py`; per-step masking records `code/engines/pretrain.py`.

**Terminology.** `code/notes/GLOSSARY.md` retires "AMMM" and fixes *Mode 1* / *Mode 2* as the canonical names for the sensor-absent and inpainting step classes. This ADR uses those names and introduces no "Regime" numbering.
