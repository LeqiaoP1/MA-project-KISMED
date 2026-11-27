# `tir_resp` baseline — ADR-008 two-stream composite arm

Stage-2 pre-training of **`tir` → `resp`** under the ADR-007 two-stream masking policy, trained with the
**ADR-008 composite respiration objective** (`alpha/beta/gamma = 1/2/1`, `lambda_resp 5.0`) plus the
Mode-2-gated cross-modal InfoNCE regularizer (`lambda_align 0.5`).

- Design of record: `code/notes/design/ADR-008-composite-objective-and-cross-modal-alignment.md`
- Companion policy: `code/notes/design/ADR-007-two-stream-anatomy-aware-constant-budget-masking.md`
- Implementation plan: `code/notes/plan/features/ADR-008-composite-objective-and-cross-modal-alignment-plan.md`
- Arm config: `code/configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr008.yaml`
- Sibling (pure-MSE comparator): `code/notes/eval/baselines/tir_rgb2d_resp/`

## Run of record — fold 0, 150 epochs

| | |
| :--- | :--- |
| Checkpoint | `/work/scratch/ne95ocyg/pretrain/adr008_2stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth` |
| Selection | `fixed_epoch`, epoch **149** = all 150, `joint_loss 6.6248` (`selection.json`) |
| Schedule | 13 `acc_short` links (8 + 5 chained with `--dependency=afterany`), 4× A100, 25 min each, `--save_ckpt_freq 10` |
| Partition | fold 0, subject-disjoint, seed 42, stratify `sex` → 97 train / 28 test subjects |
| Params | 103,906,080 (= pure-MSE arm + 1,378,048 alignment heads) |
| `masking.json` | `objective.adr = 'ADR-008'`, `spectral_n_fft 32768`, 16 band bins `[4..19]`, `signal_weight 5.0`, `run.grad_balance_freq 50` |

Final epoch means (epoch 149): `loss 3.529`, `mse_tir 0.769`, `mse_resp 0.220`, `resp_pearson 0.127`,
`resp_spectral 0.213`, `lr 1.6e-9`. Training moved `mse_resp` from `1.89` (init) to `0.18–0.22`.

Note the last top-up link reports `FAILED` in `sacct`; its stderr is the runner's benign
`nothing to do: … already at epoch 150`.

## Results

| artefact | content |
| :--- | :--- |
| [`mode1_pretext_fold0.md`](mode1_pretext_fold0.md) | ADR-008 §3.2 **Criterion 2**, Mode-1 zero-shot pretext (non-gating): variance retention, spectral peak error, masked `r`, gradient balance + the subject/task decomposition |
| [`artifacts/mode1_pretext_fold0_summary.json`](artifacts/mode1_pretext_fold0_summary.json) | the same numbers machine-readable (per-clip dump omitted; full report in `output/eval/adr008_pretext/mode1_fold0.json`) |
| [`artifacts/resp-reconstruction-mode1-fold0.png`](artifacts/resp-reconstruction-mode1-fold0.png) | slide figure: three held-out clips, mean PSD, rate scatter, per-subject `r` |
| [`figure_resp_reconstruction.py`](figure_resp_reconstruction.py) | regenerates that figure from the checkpoint |
| [`multimae_style_panel.md`](multimae_style_panel.md) + [`artifacts/multimae_style_kept_density.png`](artifacts/multimae_style_kept_density.png) | MultiMAE Figure-6-style panel (TIR / RESP × three 200-token budget allocations **plus a fully visible TIR remote-sensing column**, one held-out clip) and full test-set evaluation of each condition |
| [`artifacts/respmax_fold0_summary.json`](artifacts/respmax_fold0_summary.json), [`artifacts/mid_fold0_summary.json`](artifacts/mid_fold0_summary.json), [`artifacts/remote_sensing_fold0_summary.json`](artifacts/remote_sensing_fold0_summary.json), [`artifacts/remote_sensing_tir_rolled_fold0_summary.json`](artifacts/remote_sensing_tir_rolled_fold0_summary.json) | test-set summaries for the three non-Mode-1 panel columns **plus the rolled-TIR content control** (trimmed; full reports in `output/eval/adr008_recon/`) |

Headline (fold-0 test role, 1742 clips / 213 sessions / 27 subjects, 5 mask draws, `K_resp = 0`,
`K_tir = 200`):

| indicator | measured (subject median, 95 % CI) | predecessor |
| :--- | :--- | :--- |
| zero-shot masked `r` | **+0.326 [+0.244, +0.406]** | +0.0062 (four-stream), +0.113 (three-stream) |
| variance retention `Var(ŷ)` | 0.472 [0.457, 0.530] | ≈ 0 (collapse) |
| spectral peak error | 4.92 bpm [4.01, 5.61] | — |
| gradient balance (median, Phase II) | 66.5 | 4.47 (pure-MSE smoke) |

The condition-mean collapse is gone, but three of the ADR's four success thresholds are still missed —
see `mode1_pretext_fold0.md` §"Honest reading".

The two inpainting allocations (respiration partially visible) are reported separately in
[`multimae_style_panel.md`](multimae_style_panel.md) §4: masked `r` is **+0.999** at both 100/250
and 75/250 visible tokens — the local-interpolation ceiling — which is why the Mode-1 row above, not
those, is the sensing headline.

The same note also measures the **deployment / remote-sensing condition** (whole TIR clip visible,
budget-exempt): subject-median `r` reaches **+0.382 [+0.273, +0.535]** (per-clip paired Δ
**+0.055 ± 0.007** over the 200-token Mode-1 row), with `Var(ŷ)` rising from 0.47 to 0.65 and
spectral peak error falling from 4.92 to 3.66 bpm. A rolled-TIR control (same 900 tokens, temporal
alignment broken) drops to subject-median `r` **+0.082** (clip mean `+0.108`), confirming that the
gain comes from TIR content rather than token count.

## Reproduce

```bash
# the evaluation report (writes the full per-clip JSON)
cd code && python -u runners/run_eval_recon.py \
    --checkpoint <EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth \
    --split test --resp_probe hidden --kept_resp 0.0 --kept_tir 1.0 \
    --repeats 5 --batch_size 16 --device cuda --pretext_metrics \
    --label adr008_fold0_mode1 --out ../output/eval/adr008_pretext/mode1_fold0.json

# the figure (add --max_clips 96 for a ~20 s preview)
cd code && python notes/eval/baselines/tir_resp/figure_resp_reconstruction.py \
    --checkpoint <EXP_ROOT>/fold0/stage2/checkpoints/checkpoint-0149.pth \
    --out ../presentation/figures/resp-reconstruction-mode1-fold0.png
```

`--pretext_metrics` is the opt-in ADR-008 §3.2 extension of the recon evaluator; without it every
pre-existing report is byte-identical.

## Not done here

The gating **ADR-008 §3.3** criterion — five-fold Stage-3 band-limited Pearson, `>= +0.02` over the
matched pure-MSE ADR-007 encoder — needs the comparator arm on this fold first. Nothing in this folder
supports a Go/No-Go claim on its own.
