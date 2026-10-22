# Fine-Tuning (stage-3 in pipeline) to reconstruct the targeted waveform

> Stage 3 is the deployment condition: the contact sensor is assumed to have failed
> completely, so only RGB video enters the model. The Stage-2 encoder is fine-tuned
> end to end (nothing is frozen) and one lightweight regression head emits the whole
> waveform window. Two independent runs share that encoder, one per target waveform
> (`bp`, `resp`). No masking is applied anywhere in this stage.
>
> Companion documents: `code/notes/design/SolutionC_EarlyTokenMixing_stage2.md` holds the
> Stage-2 pre-training diagrams (input + encoder, decoder + loss);
> `code/notes/plan/local_downstream_plan.md` is the setup record for
> the two local 64 px runs. Numbers below are the **local 64 px** geometry with the
> 224 px variant called out in a note.

## Input + Encoder

```plantuml
@startuml
skinparam RoundCorner 10
skinparam ComponentStyle rectangle

title Part 1: Stage-3 Input + Encoder (RGB only, unmasked, end-to-end)

package "Inputs (visual modality only)" {
  [RGB CLIP INPUT (B, 3, T = num_frames = 100, H = 64, W = 64)] as V_IN
  [NO 1-D INPUT AT ALL (simulated total contact-sensor failure)] as A_ABSENT
}

node "STAGE-2 ENCODER CHECKPOINT (enc_blocks.*, enc_norm.*, adapters.rgb.*, positions.rgb.*)" as CKPT
node "Tubelet Embed 3D Conv (k = 2, 16, 16)" as V_EMB
node "Space-Time Pos Embed (3D sincos over t, h, w)" as V_POS
node "NO MASKING: all N_v tokens stay visible" as V_PASS

rectangle "STAGE-2 ViT ENCODER (joint space-time self-attention, 768-d, 12 layers, 12 heads)" as ENCODER #LightBlue
node "SPATIAL MEAN POOL per time step -> H (B, G_t = 50, D = 768)" as LATENT

V_IN --> V_EMB : Video tokens V (N_v = G_t x 4 x 4 = 800)
CKPT ..> V_EMB : weights loaded, then UPDATED (no freeze)
V_EMB --> V_POS
V_POS --> V_PASS
V_PASS --> ENCODER
ENCODER --> LATENT

note right of CKPT
  224 px variant of the same block:
  N_v = 50 x 14 x 14 = 9,800 tokens
end note

@enduml
```

## Head + Joint Loss

```plantuml
@startuml
skinparam RoundCorner 10
skinparam ComponentStyle rectangle

title Part 2: Stage-3 Waveform Head + Joint Loss

node "LATENT FEATURES H (B, G_t = 50, D = 768)" as LATENT
node "WAVEFORM HEAD default nn.Linear(D, samples_per_token = 8), one output row per tubelet step" as HEAD
node "HEAD VARIANTS head_hidden above 0 gives a 2-layer MLP; head_style transposed_conv gives Conv1d then ConvTranspose1d at stride 8" as HEADALT
node "PREDICTED WAVEFORM (B, output_len = 400), the flatten of B x G_t x 8" as PRED
node "TARGET WAVEFORM label column bp or resp, z-scored per clip (physio_norm zscore)" as TGT
node "PER-TOKEN ALIGNMENT segment k covers exactly the window of tubelet k; build raises unless samples_per_token equals sig_kernel" as ALIGN

LATENT --> HEAD
HEADALT ..> HEAD : alternative, not the default
HEAD --> PRED
ALIGN --> TGT

node "L_time = StdLoss, absolute difference of std(pred) and std(target), amplitude only; time_loss smoothl1 swaps in Huber on the point-wise error" as L1
node "L_Pearson = 1 minus Pearson r, batch mean, phase locking" as PEAR
node "L_MR-STFT per window spectral convergence plus log magnitude L1, averaged over the kept windows" as STFT

PRED --> L1
TGT --> L1
PRED --> PEAR
TGT --> PEAR
PRED --> STFT
TGT --> STFT

node "LOSS EVALUATED IN FP32 the autocast output is cast back before the criterion runs" as FP32
node "TOTAL LOSS L_joint = alpha x L_time + beta x L_Pearson + gamma x L_MR-STFT, all weights 1.0 by default" as TOTAL #Pink
L1 --> TOTAL
PEAR --> TOTAL
STFT --> TOTAL
FP32 ..> TOTAL

node "OPTIMIZER AdamW, full fine-tune (no frozen params), lr 1e-4, weight decay 0.05" as OPT
node "SCHEDULE 5 epoch warmup then cosine to min_lr 0.0; layer_decay 1.0 (off) by default" as SCHED
node "TWO RUNS share the encoder, one per target: bp and resp" as BRANCHES
node "CHECKPOINT SELECTION best validation Pearson" as SEL

TOTAL --> OPT
OPT --> SCHED
SCHED --> SEL
BRANCHES --> SEL

note right of HEAD
  Stage-2 heads.bp / heads.resp DO transfer into waveform_head, but only
  when head_init is transfer (the default) AND the head is a single
  Linear, i.e. samples_per_token equals sig_kernel and head_hidden is 0.
  head_init none keeps the pre-training decoder discarded and starts the
  head RANDOM, which is what both current A / C configs use. The legacy
  key heads.bvp is accepted for bp and named in the load log.
end note

note bottom of STFT
  Windows 64, 128, 256 at fs = 100 Hz for the local RGB runs (400 or 496
  samples); the resp_tir_roi configs use 128, 256, 512. A window longer
  than output_len is dropped before training, and samples whose target
  carries no spectral energy are excluded from the term.
end note

note right of SEL
  best.pth stores model and epoch only. Per-epoch history DOES reach
  disk: metrics.jsonl gets one line per evaluated epoch,
  training_curves.png is refreshed, and metrics_final.json carries the
  per-clip metrics plus the loss table and the selection record. Each
  per-epoch checkpoint also carries the training args. With k folds the
  selection stays on val Pearson while metrics_final.json reports the
  TEST fold.
end note

@enduml
```

## Evaluation (in-training per clip, offline per session)

```plantuml
@startuml
skinparam RoundCorner 10
skinparam ComponentStyle rectangle

title Part 3: Evaluation (not part of the loss graph)

node "PER-CLIP PREDICTION (B, output_len = 400)" as PRED

node "SAVE PREDICTIONS via the save_preds flag: preds.npy, targets.npy, entries.json" as SAVE
node "SESSION ASSEMBLY Hann overlap-add with weight-sum normalisation, evaluation / assemble.py" as ASSEMBLE
node "SESSION REFERENCE read from that session signals.csv" as REF
node "TIER-1 and TIER-2 recomputed on the assembled session waveform" as SESS #Pink
node "TIER-3 HRV via NeuroKit2, needs at least 30 s, offline only" as T3
node "SESSION METRICS JSON session_metrics.json written by run_evaluate_session.py" as JSON

PRED --> SAVE
SAVE --> ASSEMBLE
REF --> SESS
ASSEMBLE --> SESS
SESS --> T3
SESS --> JSON
T3 --> JSON

note right of SAVE
  The save_preds flag writes the three files from the final pass,
  and the eval_only flag re-runs just that pass on a finished run.
  Both must run SINGLE-PROCESS, otherwise each rank dumps only its
  own slice of the split.
end note

@enduml
```

## Notes on correctness

* **RGB only.** `streams: rgb` plus `use_tir: false`, so the dataset yields a
  `[B, 3, T, H, W]` clip and `MultiModalWaveformRegressor` tokenizes that one stream.
  The 1-D waveform exists solely as the regression target, which is how the
  "complete contact-sensor failure" is simulated (by omission, not by masking).
* **No masking.** `forward(self, x)` has no mask argument and raises unless the
  tokenized count equals `n_visual`, i.e. the full token grid is always fed, in both
  training and evaluation.
* **End-to-end.** `freeze_encoder()` exists but is never called; the optimizer covers
  every parameter, so `enc_blocks`, `enc_norm`, `adapters.rgb` and `positions.rgb` are
  all updated alongside the head.
* **What is persisted.** `checkpoints/checkpoint-NNNN.pth`, `latest_checkpoint.txt` and
  `best.pth` (model + epoch only). Each per-epoch checkpoint also carries the training
  `args`. Per epoch the run appends one line to `metrics.jsonl` and refreshes
  `training_curves.png`; at the end it writes `metrics_final.json` with the per-clip
  Tier-1/2 metrics plus the loss and selection provenance. `--save_preds` adds
  `preds.npy`, `targets.npy` and `entries.json`, which `runners/run_evaluate_session.py`
  turns into `session_metrics.json` offline, Tier-3 included.
* **Formerly pending, now implemented.** The Stage-2 head transfer
  (`load_stage2_encoder`, `head_init: transfer` by default), the warmup + cosine LR
  schedule (`cosine_scheduler` with `warmup_epochs`), the prediction dump
  (`--save_preds`), the session-level evaluation runner
  (`runners/run_evaluate_session.py`) and the Tier-3 30 s guard
  (`evaluation/clinical.py`, `min_samples = 30 * fs`) all exist in the code.
* **Rendering.** Labels avoid the creole traps known in
  `code/notes/design/SolutionC_EarlyTokenMixing_stage2.md`: no `--`
  pairs, no `>>`, no angle brackets. Validate with
  `java -jar /tmp/plantuml.jar -checkonly <file>.puml` (silent means OK).
