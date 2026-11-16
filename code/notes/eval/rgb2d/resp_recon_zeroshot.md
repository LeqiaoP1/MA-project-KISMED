# Zero-shot RESP reconstruction: does the extra RGB2D stream help?

Measured 2026-11-15 on fold 0 of the 96 px lineage. No fine-tuning, no Stage-3:
both checkpoints are used exactly as saved after 150 epochs of Stage-2.

- Tool: `runners/run_eval_recon.py` + `scripts/hpc/submit_recon_eval.sbatch`
  (array 0-7, a100:1; results `$WORK_SCRATCH/eval/resp_recon_rgb2d/*.json`).
- Evaluation set: the **28 held-out subjects** of fold 0 (identical list in both
  runs), **3099 clips / 204 sessions / 27 subjects** after the rgb2d gates
  (one test subject has no rgb2d-covered clip). `--data_checkpoint` pins the
  dataset to the tri-modal run, so *every* arm scores the same clips in the same
  order; 5 mask draws per clip, `--seed 0`.
- Weights: tri = `stage2_hpc_tir_roi_resp_rgb2d_fold0/checkpoint-0149.pth`,
  twin = `stage2_hpc_tir_roi_resp_96respgrid_fold0/checkpoint-0149.pth`.
- Metric: `r_masked` = Pearson over the MASKED samples only (z units, the
  dataset's per-clip normalisation), averaged over draws per clip first, then
  over clips; paired bootstrap CI (2000 resamples) on the per-clip difference.

## Why three resp probes
| probe | what it hides | r_masked (tri, rgb2d on) |
|---|---|---|
| `random50` | 50 % scattered (the training condition) | +0.9975 |
| `span50` | 50 % as ONE contiguous 4 s block | +0.5419 |
| `hidden` | 100 % (no resp token at all) | +0.0055 |

`random50` is **locally interpolatable** (neighbours on both sides) and hits the
ceiling in every arm, so it cannot discriminate. `span50` is the informative
probe. `hidden` exposes a structural limit, see below.

## Results (3099 clips x 5 draws)
| arm | r_masked | r_full | mse | mae |
|---|---|---|---|---|
| tri `random50` rgb2d **on** | +0.9975 | +0.9988 | 0.0051 | 0.037 |
| tri `random50` rgb2d off | +0.9467 | +0.9728 | 0.1120 | 0.232 |
| tri `span50` rgb2d **on** | **+0.5419** | +0.8277 | 0.6215 | 0.610 |
| tri `span50` rgb2d off | **+0.4558** | +0.8030 | 0.6980 | 0.672 |
| tri `hidden` rgb2d on | +0.0055 | +0.0055 | 1.1424 | 0.893 |
| tri `hidden` rgb2d off | +0.0055 | +0.0055 | 1.1424 | 0.893 |
| twin `random50` | +0.9978 | +0.9989 | 0.0044 | 0.034 |
| twin `span50` | +0.5744 | +0.8393 | 0.5826 | 0.587 |

Paired comparisons (per-clip, 95 % bootstrap CI):

| comparison | delta r_masked | CI | clips improving |
|---|---|---|---|
| **within-model**, span50: rgb2d on - off | **+0.0861** | [+0.0833, +0.0888] | 87.8 % |
| within-model, random50: on - off | +0.0508 | [+0.0502, +0.0514] | 100 % |
| within-model, hidden: on - off | +0.0000 | [0, 0] | 0 % |
| within-model, span50 MSE (lower better) | -0.0764 | [-0.0789, -0.0739] | 88.7 % |
| **across-model**, span50: tri - twin | **-0.0325** | [-0.0387, -0.0258] | 43.2 % |
| across-model, random50: tri - twin | -0.0003 | [-0.0005, -0.0001] | 42.5 % |

## Verdict
1. **The rgb2d STREAM carries information about respiration that the frozen
   model uses** (+0.086 Pearson on a contiguous 4 s mask, 88 % of clips, MSE
   -11 %): paired, same weights, same tir/resp masks. This is the answer to
   "could the extra modality improve reconstruction": yes, at inference.
2. **Training WITH rgb2d did not improve the frozen encoder for resp**: the
   TIR+RESP twin is slightly *better* (+0.033 r on span50, i.e. tri is -0.033).
   Confounds: the tri arm trains on 15 406 clips vs the twin's 16 378 (-6 %
   — the rgb2d gates drop 972 clips), an extra decoder/adapters, and a different
   mask RNG sequence. One fold, one pretraining seed: treat this as "no gain
   demonstrated", not "rgb2d hurts".
3. **A fully hidden 1-D stream cannot be reconstructed from the visuals at all**
   (delta exactly 0, mse 1.14 = the signal variance). Structural, not a training
   failure: in `MultiModalMAE.forward`, a stream with 0 visible tokens feeds the
   decoder ONLY ``mask_token + pos_embed`` -- `enc_s` is empty, so the per-stream
   self-attention decoder has no path to the other modalities' tokens. Any
   "respiration from video alone" claim needs a decoder change (cross-attention
   or encoder-conditioned queries), not a different mask ratio.

## Caveats
- `span50` is out of the training distribution: both runs used
  `physio_mask: random`. The comparison is still fair (both arms equally OOD and
  identically masked), but absolute `span50` numbers are not training-time loss.
- `r_full` (assembled waveform) is inflated by the visible 50 %; quote
  `r_masked` as the result.
- Session-level aggregation was not attempted; 27 test subjects is the whole
  block, so this is a cohort result, not a per-subject ranking.

## Where the gain comes from (per task / per subject)
Within-model span50 (rgb2d on - off), paired per clip (`tmp/recon_by_task.py`):

| task | clips | on | off | delta | clips improving |
|---|---|---|---|---|---|
| T8 | 367 | +0.5919 | +0.4858 | **+0.1061** | 92.1 % |
| T7 | 224 | +0.5888 | +0.4884 | +0.1003 | 91.5 % |
| T4 | 40 | +0.6068 | +0.5077 | +0.0991 | 95.0 % |
| T6 | 326 | +0.5891 | +0.4919 | +0.0972 | 92.0 % |
| T10 | 67 | +0.6056 | +0.5087 | +0.0969 | 92.5 % |
| T9 | 883 | +0.5303 | +0.4485 | +0.0819 | 87.2 % |
| T3 | 618 | +0.4944 | +0.4125 | +0.0818 | 84.8 % |
| T2 | 162 | +0.5265 | +0.4522 | +0.0743 | 87.0 % |
| T5 | 216 | +0.5304 | +0.4564 | +0.0741 | 84.3 % |
| T1 | 196 | +0.5087 | +0.4450 | +0.0637 | 82.7 % |

The gain is **global**: every task improves (+0.064 .. +0.106) and so does
**every one of the 25 subjects** in the pool (+0.063 .. +0.133, improving-clip
share 83-100 %). It is not driven by a task or a cohort. The across-model
comparison is negative for every task (-0.008 .. -0.046).

## Reproduce
```
sbatch scripts/hpc/submit_recon_eval.sbatch                  # 8 arms
python tmp/recon_compare.py <results dir>                    # table + paired CIs
```

## Downstream (Stage 3) — preflight PASSED, NOT launched
**No new config file is needed.** `configs/finetune/resp_tir_roi_hpc.yaml` is
already geometry-matched to the tri-modal Stage-2 run: every field of
`run_waveform.ROI_CONTRACT_KEYS` agrees —

| key | Stage-2 (tri-modal) | Stage-3 config |
|---|---|---|
| `roi_landmarks` | `''` -> nose_mouth (12 pts) | `''` |
| `roi_padding` | 0.10 | 0.10 |
| `input_size` | 96 | 96 |
| `crop_rule` | square | square |
| `max_sway_px` | 40 | 40 |

so the tri-modal encoder is a drop-in Stage-3 init; only the lineage NAME
changes (the launcher resolves `pretrain/<STAGE2_RUN_NAME>_fold<k>/`):

```
cd code
# 1. CPU preflight against the real loader (exit 0 = COMPATIBLE; RAN 2026-11-16)
PYTHONPATH=. python notes/eval/tir_resp/stage3_compat_probe.py \
    --finetune $WORK_SCRATCH/pretrain/stage2_hpc_tir_roi_resp_rgb2d_fold0/checkpoints/checkpoint-0149.pth
# 2. the job, ONE fold, paired with its OWN Stage-2 run (never STAGE2_DIR here)
STAGE2_RUN_NAME=stage2_hpc_tir_roi_resp_rgb2d RUN_NAME=resp_tir_roi_hpc_rgb2d \
FOLDS=0 GPU_TYPE=rtx6000 GPUS=4 PARTITION=acc TIME=10:00:00 \
    bash scripts/hpc/submit_stage3_kfold.sh --submit
```

Preflight verdict (2026-11-16): **COMPATIBLE, exit 0** — encoder 148 + head
loaded, ROI contract satisfied on all five keys, geometry mirror equal on
`model/tubelet/fps/temporal_stride/clip_duration/input_size/fs/seq_len/
sig_kernel/task_set/clip_stride/physio_norm`, MR-STFT windows
`(128,256,512)` fit the 800-sample target, and every Stage-3 visual stream
(`['tir']`) exists in the Stage-2 run (`['tir','rgb2d','resp']`).
Two ALLOWED notes: `min_signal_spread 0.0 -> 0.1` and `rail_touch_v 0.0 -> 9.9`
(the Stage-2 run used the resp cache, which forces both to 0). The twin has the
same asymmetry, so the downstream A/B stays like-for-like.

Notes: Stage 3 feeds `streams: tir` only, so the extra Stage-2 streams
(`rgb2d`, `resp`) are simply unused — the probe's expected, benign difference.
`data_set` stays `tir_roi_resp`, so BOTH arms (tri-modal init vs the
`96respgrid` twin init) are evaluated on the same 16 378-clip corpus with the
same target and the same loss — that is what makes the downstream A/B valid.
Cost, measured earlier: ~8h40 per fold on 4x rtx6000 / ~21 h on 4x a100.
