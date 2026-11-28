# ADR-017 — Fold-0 Stage-2 zero-shot respiration probes (Gates G2 / G3)

**Folder:** `code/notes/eval/baselines/tir_resp/` (see [`README.md`](README.md) for the run of record).
**Status:** measured 2026-11-28. **Non-gating** for adoption (ADR-017 §3): this file is the Stage-2
diagnostic evidence for Gates G2/G3 only. The gating five-fold Stage-3 comparison (G4,
`band_limited_pearson >= +0.02` vs. the matched ADR-008 arm) was **not** run — by decision this
experiment stops at Stage 2, so ADR-017 remains **Proposed**, not `Accepted`.

## What was evaluated

| | |
| :--- | :--- |
| Checkpoint | `/work/scratch/ne95ocyg/exp_20261127_adr017_tir_resp/fold0/stage2/checkpoints/checkpoint-0149.pth` (epoch 149 = all 150; sha256 `6c80aa4e4327…`) |
| Arm | `configs/pretrain/stage2_budget_tir_resp_2stream_anatomy_adr017.yaml` — the ADR-008 arm with `physio_placement: masked_span`, `physio_gap_min/max_tokens 63/125` = `2.016/4.000 s` (`masking.json` policy block) |
| Training | 150 epochs, 4× A100, 13-link `acc_short` chain + 3-link top-up; `selection.json`: `fixed_epoch` → 149, `partition_digest dc80d46c…` (= ADR-008 partition) |
| Role | fold 0 **test** (subject-disjoint), 27 held-out subjects, 213 sessions, **1742 clips** × 5 mask draws |
| Job | Slurm `57574325` (`acc`, `gaoc0002`, A100-SXM4-40GB), `COMPLETED` exit `0:0`, 16:10 wall clock; per-condition 2:45–3:53 |
| Report | `output/eval/exp_20261127_adr017_fold0/{respmax_masked_span,mid_masked_span,mode1,remote_sensing,remote_sensing_tir_rolled}_fold0.json` + per-condition `.log` |

Command shape (all conditions): `runners/run_eval_recon.py --checkpoint <ckpt> --split test
--repeats 5 --batch_size 16 --preload --seed 0 --device cuda --pretext_metrics`
plus `--resp_probe masked_span --interp_baseline --kept_tir 0.111 --kept_resp 0.40` (respmax),
`--kept_tir 0.139 --kept_resp 0.30` (mid), `--resp_probe hidden --kept_tir 1.0 --kept_resp 0.0`
(Mode 1), `--resp_probe hidden --kept_tir 1.0 --exempt_budget` (TIR-900 and its rolled control).
The `masked_span` probe reuses the run's recorded `(63, 125)` bounds.

## Results (subject median with 95 % bootstrap CI over 27 subjects, 2000 resamples)

| condition | subject-median `r_masked` | subject-median `mse` | `Var(ŷ)` | spectral peak error |
| :--- | ---: | ---: | ---: | ---: |
| **`respmax` 100/100 `masked_span`** | **+0.8595 [+0.8408, +0.8805]** | 0.2289 [+0.2105, +0.2771] | 0.6919 [0.6652, 0.7157] | 3.43 bpm [3.20, 3.78] |
| **`mid` 125/75 `masked_span`** | **+0.8063 [+0.7648, +0.8540]** | 0.3115 [+0.2810, +0.3877] | 0.6807 [0.6411, 0.7135] | 3.78 bpm [3.43, 4.58] |
| Mode-1 `TIR-200` | +0.3299 [+0.1964, +0.3850] | 1.0082 [+0.9314, +1.1532] | 0.4861 [0.4723, 0.5100] | 4.81 bpm [3.66, 5.72] |
| `TIR-900` remote sensing | +0.4601 [+0.3248, +0.5436] | 0.8903 [+0.7860, +1.0013] | 0.5217 [0.4464, 0.5476] | 3.66 bpm [1.83, 3.66] |
| `TIR-900-rolled` (control) | +0.1044 [+0.0459, +0.1285] | 1.3829 [+1.3032, +1.4384] | 0.5424 [0.4729, 0.5703] | 5.49 bpm |

Per-clip means (n = 1742, `r_masked`): respmax +0.8231 (sd 0.1536), mid +0.7657 (sd 0.1928),
Mode-1 +0.3151 (sd 0.3036), TIR-900 +0.3986 (sd 0.3600), rolled +0.0956 (sd 0.3701).

### No-video floor on the identical masks (`--interp_baseline`, per-clip means)

| condition | `np.interp` `r_masked` | `np.interp` MSE | constant-zero MSE |
| :--- | ---: | ---: | ---: |
| `respmax` 100/100 | +0.4408 | 1.1987 | 0.9927 |
| `mid` 125/75 | +0.3446 | 1.3280 | 0.9906 |

The interpolator is **worse than the constant predictor** (`1.20 > 0.99` and `1.33 > 0.99`) — the
G0 mechanism, reproduced on the test role.

### Gate verdicts

| Gate | Requirement | Measured | Verdict |
| :--- | :--- | :--- | :--- |
| **G2** (`respmax`) | paired subject-median `r_masked(model) − r_masked(np.interp)`, 95 % CI > 0 | **+0.3571 [+0.3010, +0.4223]** *(derived: per-clip difference → per-subject median → 2000-resample bootstrap, seed 0)* | **PASS** |
| **G2** (`mid`) | same | **+0.4027 [+0.3246, +0.5165]** *(same derivation)* | **PASS** |
| **G3** (Mode-1 `TIR-200`) | subject median `>= +0.3056` (`+0.3256 − 0.02`) | **+0.3299** — passes the gate as written, though the arm-vs-arm **paired** test is `−0.0283 [−0.0501, −0.0113]` (see below) | **PASS** |
| G4 (five-fold Stage-3 Δ `band_limited_pearson`) | adopt if `>= +0.02`, 95 % CI > 0 | **not run** — Stage 3 out of scope for this experiment | **not evaluated** |

## Comparison with the matched ADR-008 `scattered` arm (same partition, same test role)

| indicator | ADR-017 `masked_span` | ADR-008 `scattered` |
| :--- | ---: | ---: |
| Mode-2 `respmax` 100/100 `r_masked` (subject median) | +0.8595 | +0.9989 |
| Mode-2 `mid` 125/75 `r_masked` (subject median) | +0.8063 | +0.9986 |
| Mode-1 `TIR-200` `r_masked` (subject median) | +0.3299 | +0.3256 [+0.2444, +0.4061] |
| Mode-1 `Var(ŷ)` | 0.4861 | 0.4716 |
| Mode-1 spectral peak error | 4.81 bpm | 4.92 bpm |
| `TIR-900` remote sensing `r_masked` | +0.4601 | +0.3819 [+0.2731, +0.5349] |
| `TIR-900-rolled` control `r_masked` | +0.1044 | +0.0823 [+0.0379, +0.1527] |

The lower Mode-2 correlation of the ADR-017 arm is **not** a regression: under `scattered` the
+0.999 is reproduced by `np.interp` (the floor probe measured `r = +0.999` at `K_resp = 100`), so it
carries no cross-modal evidence. Under `masked_span` the model is **+0.36…+0.40 above** the
interpolator on the same masks (G2). The rolled-TIR control collapsing from +0.460 to +0.104 shows
that both arms' remote sensing is content-driven rather than a prior/length artefact.

### Paired, clip-aligned comparison against ADR-008 (same 1742 clips, same seed)

Both arms score the identical clips with `--seed 0`, and `K_resp = 0` consumes **no** placement RNG
draw (ADR-017 R4/I4), so the Mode-1 and remote-sensing masks are identical across arms. That allows a
per-clip paired test (1742/1742 clips matched on `subject + session + t_start`), which is the decisive
comparison — the table above compares *aggregates*, and the two arms' per-clip distributions differ in
shape.

| condition | metric | paired Δ (ADR-017 − ADR-008), subject median [95 % CI] | ADR-017 wins |
| :--- | :--- | ---: | ---: |
| `TIR-900` | `r_masked` | **+0.0078 [−0.0219, +0.0511]** | 49.3 % of clips |
| `TIR-900` | `mse` | −0.0496 [−0.1167, −0.0045] | — |
| `TIR-900` | `Var(ŷ)` | **−0.1053 [−0.1318, −0.0961]** | 32.5 % |
| `TIR-900` | peak error | 0.00 bpm [0.00, 0.00] | 38.6 % |
| Mode-1 `TIR-200` | `r_masked` | **−0.0283 [−0.0501, −0.0113]** | 43.7 % of clips |
| Mode-1 `TIR-200` | `mse` | +0.0409 [+0.0171, +0.0675] | — |
| Mode-1 `TIR-200` | `Var(ŷ)` | −0.0105 [−0.0171, +0.0095] | 49.4 % |
| `TIR-900-rolled` | `r_masked` | −0.0062 [−0.0485, +0.0368] | 47.7 % of clips |
| `TIR-900-rolled` | `Var(ŷ)` | −0.0988 [−0.1293, −0.0559] | 35.2 % |

- **Remote sensing is a tie, not a gain.** The `+0.078` gap between the subject-median *aggregates*
  (`+0.4601` vs `+0.3819`) does not survive pairing: per-clip mean Δ `+0.0034`, subject-median Δ
  `+0.0078` with a CI spanning 0, and ADR-017 wins 49.3 % of clips. The clip-level marginals are
  nearly identical (means `+0.3986` vs `+0.3952`, medians `+0.4615` vs `+0.4565`). The aggregate gap
  comes from per-subject medians of a few **small-`n` subjects** moving up (F030 `+0.044→+0.265`,
  M016 `+0.088→+0.382`, F028 `−0.216→−0.076`; `n = 23 / 8 / 4` clips) while the **largest-`n`
  subjects regress** (F041 `+0.306→+0.144` over 81 clips, M058 `+0.704→+0.544` over 78, F064
  `+0.702→+0.549` over 52). A median over 27 subject medians is not robust to that.
- **The lower `TIR-900` MSE is amplitude shrinkage, not better sensing.** `Var(ŷ)` falls by `0.105`
  with a tight CI, and the **same** shrinkage appears in the content-destroyed rolled control
  (`−0.099`). A shrunk prediction has lower MSE at low correlation without being better aligned, so
  the MSE difference cannot be credited to cross-modal sensing.
- **Mode-1 sensing is slightly *worse*, not preserved.** The G3 subject-median threshold still passes
  (`+0.3299 ≥ +0.3056`), but the paired test shows `−0.0283 [−0.0501, −0.0113]` with ADR-017 winning
  only 43.7 % of clips and a `+0.041` MSE penalty. The "no change" reading comes from overlapping
  marginal CIs, which is the weaker test.

Remaining weakness, unchanged in kind from ADR-008: **spectral peak error stays at 3.4–3.8 bpm**
(Mode 2) and **4.8 bpm** (Mode 1), i.e. above the ADR-008 `< 3.0 bpm` failure band. Amplitude and
correlation are healthy (`Var(ŷ)` 0.68–0.69 on Mode 2), so the deficit is in spectral/temporal
precision.

## Within-arm training diagnostics (see also `mode1_pretext_fold0.md` §"Honest reading")

`ADR-017 §4` forbids cross-placement comparison of `loss` / `mse_resp` / `grad_ratio`. Within this
arm: train-side masked-token correlation (`1 − resp_pearson`) reaches **0.858** at epoch 149
(≈ 0.001 at init), Phase-II `grad_ratio_resp_to_tir` median **45.5** (p25 33.9, p75 59.5), and the
validation `joint_loss` bottoms at epoch 75 (9.60) then drifts to 10.16 at epoch 149 — the shipped
checkpoint is still `fixed_epoch` 149 per the protocol.

## Reproducibility

- Slurm `57574325`, per-condition logs `output/eval/exp_20261127_adr017_fold0/<condition>.log`,
  job stdout `code/logs/adr017_probe_57574325.out` (timestamped `START`/`END` per condition).
- Submission script: `code/scripts/hpc/submit_adr017_probes.sbatch` (1× A100, `--preload`,
  `--mem-per-cpu=16G`; the 1742-clip preload is ~38 GB host RAM).
- `partition_digest dc80d46c…` matches the ADR-008 run, so all cross-arm pairs are subject-identical.
- Determinism: `--seed 0` fixes both the clip subset and the per-repeat mask placement, so the
  `--interp_baseline` comparison is on identical masks by construction.

**Not covered here (deliberately):** five-fold Stage 2, Stage-3 fine-tuning, pooled
`band_limited_pearson`, `psd_mae` / `dominant_freq_error_hz` / `hf_power_rel` Tier-2 aggregation, and
the ADR-017 adoption decision.
