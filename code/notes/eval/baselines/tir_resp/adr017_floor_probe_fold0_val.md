# ADR-017 — Gate G0: Criterion-3a no-video floor probe (fold-0 `val`)

**Folder:** `code/notes/eval/baselines/tir_resp/` (see [`README.md`](README.md) for the run of record).
**Status:** measured 2026-11-27. **Gating for Stage-2 submission**: ADR-017 aborts before spending
GPU-hours unless `mse_interp > mse_const` at the adopted gap bounds for every Mode-2 count.
Result: **PASS** at the shipped `(g_min, g_max) = (63, 125)`; no config change needed.

## What was evaluated

| | |
| :--- | :--- |
| Checkpoint (args only, **no weights, zero GPU**) | `/work/scratch/ne95ocyg/pretrain/adr008_2stream_fold0/fold0/stage2/checkpoints/checkpoint-0149.pth` (geometry + policy record of the matched ADR-008 arm) |
| Partition | `.../adr008_2stream_fold0/fold0/partition.json`, role **`val`** (14 subjects) |
| Geometry | `N_s = 250` signal tokens × `32` samples, `fs = 1000 Hz`, `K_resp ∈ {25, 50, 75, 100}` (apportioner-derived), `N(M) = {225: [2,3], 200: [2,3], 175: [2], 150: [2]}` |
| Clips | **1194** val clips, `--repeats 5`, `--seed 0`, `0` clips skipped as flat |
| Grid | `--gap_grid 63:125,63:150,75:150,87:150` plus the `scattered` sanity comparator and the legacy `span` (4 s kept block) |
| Command | `runners/run_probe_masking_baselines.py --checkpoint <adr008 ckpt> --partition <adr008 fold0 partition.json> --split val --repeats 5 --seed 0 --gap_grid 63:125,63:150,75:150,87:150 --out output/eval/adr017_floor/fold0_val_floor.json` |
| Report | `output/eval/adr017_floor/fold0_val_floor.json` |

Predictors: `np.interp` between visible samples, constant-clamped at the edges (`mse_interp`,
`r_masked_interp`) vs. the constant-zero predictor (`mse_const`). Both are scored on the **masked**
positions only, so `mse_interp > mse_const` means a no-video local interpolator is beaten by the
trivial baseline and cannot solve Mode 2 by intra-modal interpolation.

## Verdict

| Rule | `K_resp` | `mse_interp` | `mse_const` | `r_masked_interp` | Interp beats const? |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `scattered` (sanity) | 25 | 0.1161 | 0.9996 | +0.9484 | yes |
| `scattered` (sanity) | 50 | 0.0194 | 0.9985 | +0.9916 | yes |
| `scattered` (sanity) | 75 | 0.0050 | 0.9999 | +0.9979 | yes |
| `scattered` (sanity) | 100 | 0.0021 | 1.0012 | +0.9991 | yes |
| **`masked_span[63,125]`** | 25 | **1.2368** | 0.9898 | +0.3792 | **no** |
| **`masked_span[63,125]`** | 50 | **1.1368** | 0.9880 | +0.4438 | **no** |
| **`masked_span[63,125]`** | 75 | **1.1870** | 0.9825 | +0.4139 | **no** |
| **`masked_span[63,125]`** | 100 | **1.0926** | 0.9849 | +0.4811 | **no** |
| `masked_span[63,150]` | 25 / 50 / 75 / 100 | 1.2489 / 1.1155 / 1.2008 / 1.2606 | 0.9886 / 0.9824 / 0.9833 / 0.9771 | +.3812 / +.4494 / +.4124 / +.3491 | no (all K) |
| `masked_span[75,150]` | 25 / 50 / 75 / 100 | 1.2480 / 1.2934 / 1.2057 / 1.2767 | 0.9901 / 0.9843 / 0.9832 / 0.9758 | +.3816 / +.3502 / +.4050 / +.3509 | no (all K) |
| `masked_span[87,150]` | 25 / 50 / 75 / 100 | 1.3696 / 1.2904 / 1.2179 / 1.4228 | 0.9877 / 0.9832 / 0.9818 / 0.9741 | +.3095 / +.3486 / +.4038 / +.2208 | no (all K) |
| `span` (legacy, 4 s kept block) | 25 / 50 / 75 / 100 | 1.9439 / 1.9155 / 1.8670 / 1.8188 | 1.0030 / 1.0064 / 1.0139 / 1.0172 | +.1374 / +.0979 / +.1164 / +.1544 | no (all K) |

- **Criterion 3a: PASS** — every swept `masked_span` point has `mse_interp > mse_const` at all four
  `K_resp`, i.e. the masked gaps are not locally interpolatable.
- **Sanity (`scattered`): PASS** — `mse_interp` is 1–3 orders of magnitude **below** `mse_const`
  (`0.0021 – 0.1161` vs. `~1.0`, `r_masked_interp >= +0.948`), reproducing the ADR-008 baseline
  finding that `scattered` gaps are trivially interpolated.
- **Adopted point: `(63, 125)`** — the shipped config's Criterion-3a hypothesis; it is the tightest
  bound and passes without widening, so `stage2_budget_tir_resp_2stream_anatomy_adr017.yaml` is
  unchanged. `(63,150) / (75,150) / (87,150)` also pass and stay on the record as fallbacks.

## Reading

Under `scattered` the best no-video interpolator reaches `r = +0.999` at `K_resp = 100`, matching
the measured Mode-2 probe of the ADR-008 checkpoint (`+0.9989`). Under `masked_span[63,125]`, the
same interpolator falls to `r = +0.379…+0.481` and is **worse than predicting zero** (MSE
`1.09–1.24` vs. `≈0.98`): a 2.0–4.0 s gap spans 0.2–2.4 respiratory cycles at the 0.10–0.60 Hz
passband, so linear reconstruction aliases the waveform. Mode 2 can therefore only be solved by
reading respiratory phase from the `tir` stream — the mechanism ADR-017 hypothesises.

This gate says nothing about representational quality; it only certifies that the pretext task is
non-degenerate under the adopted bounds. Gates G2 (paired model-vs-`np.interp` inpainting), G3
(Mode-1 `TIR-200` non-degradation) and G4 (five-fold Stage-3 `band_limited_pearson`) are recorded
separately after the Stage-2 run.
