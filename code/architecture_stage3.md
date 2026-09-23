# Fine-Tuning (stage-3 in pipeline) to reconstruct the targeted waveform

> Stage 3 is the deployment condition: the contact sensor is assumed to have failed
> completely, so only RGB video enters the model. The Stage-2 encoder is fine-tuned
> end to end (nothing is frozen) and one lightweight regression head emits the whole
> waveform window. Two independent runs share that encoder, one per target waveform
> (`bp`, `resp`). No masking is applied anywhere in this stage.
>
> Companion documents: `architecture.md` holds the Stage-2 pre-training diagrams
> (input + encoder, decoder + loss); `local_downstream_plan.md` is the setup record for
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

node "LATENT FEATURES H (B, G_t = 50, D)" as LATENT
node "WAVEFORM HEAD Linear(D, samples_per_token = output_len / G_t = 8)" as HEAD
node "PREDICTED WAVEFORM (B, output_len = 400)" as PRED
node "TARGET WAVEFORM label column bp or resp, z-scored per clip" as TGT
node "PER-TOKEN ALIGNMENT segment k covers exactly the window of tubelet k" as ALIGN

LATENT --> HEAD
HEAD --> PRED
ALIGN --> TGT

node "L_time = L1 amplitude" as L1
node "L_Pearson = 1 - Pearson r (phase locking)" as PEAR
node "L_MR-STFT = multi-resolution STFT magnitude (windows 64, 128, 256 at fs = 100 Hz)" as STFT

PRED --> L1
TGT --> L1
PRED --> PEAR
TGT --> PEAR
PRED --> STFT
TGT --> STFT

node "TOTAL LOSS L_joint = alpha x L_time + beta x L_Pearson + gamma x L_MR-STFT" as TOTAL #Pink
L1 --> TOTAL
PEAR --> TOTAL
STFT --> TOTAL

node "OPTIMIZER AdamW, full fine-tune (no frozen params), lr 1e-4" as OPT
node "TWO RUNS share the encoder, one per target: bp and resp" as BRANCHES
node "CHECKPOINT SELECTION best validation Pearson" as SEL

TOTAL --> OPT
OPT --> SEL
BRANCHES --> SEL

note right of HEAD
  Stage-2 heads.bp / heads.resp transfer into waveform_head is shape-exact
  but NOT wired yet (pending Phase 1 of local_downstream_plan.md)
end note

note right of SEL
  best.pth stores model and epoch only;
  no metrics JSON and no per-epoch history are written
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

node "SAVE PREDICTIONS (planned) preds.npy, targets.npy, entries.json" as SAVE
node "SESSION ASSEMBLY (planned) Hann overlap-add with weight-sum normalisation" as ASSEMBLE
node "SESSION REFERENCE read from that session signals.csv" as REF
node "TIER-1 and TIER-2 recomputed on the assembled session waveform" as SESS #PINK
node "TIER-3 HRV via NeuroKit2, needs at least 30 s, offline only" as T3
node "SESSION METRICS JSON (planned) written by run_evaluate_session.py" as JSON

PRED --> SAVE
SAVE --> ASSEMBLE
REF --> SESS
ASSEMBLE --> SESS
SESS --> T3
SESS --> JSON
T3 --> JSON

note right of SAVE
  Nothing saves predictions today: the offline evaluator
  needs prediction and target npy files supplied by hand
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
* **What is persisted.** Only `checkpoints/checkpoint-NNNN.pth`, `latest_checkpoint.txt`
  and `best.pth` (model + epoch, plus the training `args` inside each checkpoint). The
  Tier-1/2 numbers are printed per epoch and never written to disk, there is no metrics
  JSON and no prediction dump; Tier-3 is offline and needs saved predictions, which no
  runner produces today.
* **Pending items shown as notes.** The Stage-2 head transfer, the warmup + cosine LR
  schedule, the prediction dump, the session-level evaluation runner, and the Tier-3
  guard (2 s today, needs 30 s) are not implemented yet.
* **Rendering.** Labels avoid the creole traps known in `architecture.md`: no `--`
  pairs, no `>>`, no angle brackets. Validate with
  `java -jar /tmp/plantuml.jar -checkonly <file>.puml` (silent means OK).
