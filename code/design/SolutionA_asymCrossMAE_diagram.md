# Solution A — Asymmetric Cross-MAE: architecture diagrams

PlantUML source for **Solution A (Modality-Isolated / Asymmetric Cross-MAE)**, the
asymmetric variant of the current shared-encoder design (**Solution C**).

Design of record: `code/design/SolutionA_asymCrossMAE.md`.
Both solutions share one dataset and one encoder interface; they differ **only** in the
pre-training stage (see diagram 3).

---

## 1. SSL pre-training — Solution A (resp tokens excluded from the encoder)

```plantuml
@startuml SolutionA_SSL
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam defaultFontName Helvetica
skinparam rectangle {
  BackgroundColor #F7F9FC
  BorderColor #4A6FA5
}
skinparam component {
  BackgroundColor #E8F0FB
  BorderColor #2E5A88
}
skinparam note {
  BackgroundColor #FFF8DC
  BorderColor #C8A951
}

title Solution A: Asymmetric Cross-MAE pre-training

actor "BP4D raw tree\nThermal + IRFeatures + Resp_Volts" as RAW

package "Dataset (identical for A and C)" as DS {
  rectangle "Static clip ROI\n12 nose+mouth landmarks, +20% pad, one box per clip" as ROI
  rectangle "Zero-variance guard\nraw spread < 0.01 V -> clip dropped" as GUARD
  rectangle "Subject-disjoint split\nsplit_by = subject" as SPLIT
}

rectangle "x_TIR   [B,3,T,H,W]\nT=100, H=W=64" as XTIR
rectangle "y_RESP   [B,L]\nL=800 raw volts, z-score per clip" as YRESP

package "Encoder (SHARED by Solution A and Solution C)" as ENC {
  component "TubeletEmbed\nConv3d 2x16x16 -> 800 visual tokens" as TOKVIS
  component "Tube mask 90%\n-> 10% visible = 80 tokens" as MASKV
  component "ViT Encoder\n12 blocks, D=768\nenc_blocks + enc_norm" as VIT
}

rectangle "Z_v   [B, 80, 768]\nvisual tokens ONLY, no resp tokens" as ZV

package "Decoder 1: TIR pixel decoder (self-attention)" as DECTIR {
  component "concat visible + learnable MASK tokens\nunshuffle then add pos" as DT1
  component "self-attention dec_blocks" as DT2
  component "head: Linear(768 to 3*2*16*16)" as DT3
}
rectangle "x_TIR_pred   [B, 800, patch]" as XTIRPRED
rectangle "L_TIR = MSE over masked patches" as LTIR

package "Decoder 2: RESP cross-attention decoder (NEW)" as DECRESP {
  component "Q_RESP: learnable positional queries\n[B, n_signal=50, 768]" as QRESP
  component "CrossAttention\nQ = Q_RESP,  K = V = Z_v" as XATT
  component "1-D projection\nLinear(768 to sig_kernel=16)" as P1D
}
rectangle "y_RESP_pred   [B, 50, 16] -> [B, 800]" as YPRED
rectangle "L_RESP = MSE over ALL time slots\n(vs z-scored target)" as LRESP

rectangle "L_SSL = 1.0 * L_TIR + lambda_RESP * L_RESP\nlambda_RESP about 0.5 to 1.0, NO spectral term in SSL" as LSSL

RAW --> DS
DS --> XTIR
DS --> YRESP

XTIR --> TOKVIS
TOKVIS --> MASKV
MASKV --> VIT
VIT --> ZV

ZV --> DT1
DT1 --> DT2
DT2 --> DT3
DT3 --> XTIRPRED
XTIRPRED --> LTIR

ZV --> XATT
QRESP --> XATT
XATT --> P1D
P1D --> YPRED
YPRED --> LRESP
YRESP --> LRESP

LTIR --> LSSL
LRESP --> LSSL

note right of ZV
  THE ASYMMETRY:
  RESP is a reconstruction TARGET only.
  No resp token ever enters the encoder,
  so the encoder cannot interpolate the
  1-D stream from its own visible samples.
end note

note bottom of DECRESP
  Solution C (current) instead keeps the visible
  resp tokens inside the encoder and reconstructs
  them with the shared self-attention decoder.
end note

@enduml
```

---

## 2. Supervised fine-tuning — discard decoders, keep the encoder

```plantuml
@startuml SolutionA_Finetune
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam defaultFontName Helvetica
skinparam rectangle {
  BackgroundColor #F7F9FC
  BorderColor #4A6FA5
}
skinparam component {
  BackgroundColor #E8F0FB
  BorderColor #2E5A88
}
skinparam note {
  BackgroundColor #FFF8DC
  BorderColor #C8A951
}

title Solution A: supervised fine-tuning (spec section 4)

rectangle "x_TIR   [B,3,100,64,64]\n100% unmasked, 0% masking" as XIN

package "Kept from pre-training" as KEEP {
  component "ViT Encoder\nenc_blocks + enc_norm" as ENC2
}
rectangle "DISCARD\nTIR pixel decoder + RESP cross-attn decoder" as DISCARD

component "1-D upsampling head\nConv1d then GELU then ConvTranspose1d\nstride = samples_per_token = 16" as UP
rectangle "y_pred   [B, 800]   z-scored" as YPRED2
rectangle "L_FT = a * (1 - r)  +  b * abs(std(y) - std(yhat))  +  c * MR-STFT" as LFT
rectangle "Layer-wise LR decay 0.65 to 0.75\nLayerDecayValueAssigner" as LWD

XIN --> ENC2
ENC2 --> UP
UP --> YPRED2
YPRED2 --> LFT
LWD --> ENC2

note right of ENC2
  Encoder keys and shapes are IDENTICAL
  for Solution A and Solution C
  -> the pre-trained checkpoint is swappable.
end note

note bottom of UP
  Encoder unfrozen, fine-tuned end to end.
  Layer-wise LR decay preserves lower layers.
end note

@enduml
```

---

## 3. Solution A vs Solution C — differ only in pre-training

```plantuml
@startuml SolutionA_vs_C
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam defaultFontName Helvetica
skinparam rectangle {
  BackgroundColor #F7F9FC
  BorderColor #4A6FA5
}
skinparam component {
  BackgroundColor #E8F0FB
  BorderColor #2E5A88
}
skinparam note {
  BackgroundColor #FFF8DC
  BorderColor #C8A951
}

title Solution A vs Solution C: shared contract, single switch

rectangle "SHARED dataset and preprocessing\nsame ROI, same task set, same input_size,\nsame subject-disjoint split" as SHARED

rectangle "Pre-training switch: resp_in_encoder" as SWITCH

package "Solution A  (resp_in_encoder = false)" as A {
  component "encoder input: TIR visible tokens only" as AI
  component "RESP decoded by cross-attention\nlearned queries Q_RESP" as AD
}

package "Solution C  (resp_in_encoder = true)" as C {
  component "encoder input: TIR + visible RESP tokens" as CI
  component "RESP decoded by shared\nself-attention decoder" as CD
}

rectangle "IDENTICAL ENCODER INTERFACE\nadapters.tir, positions.tir,\nenc_blocks, enc_norm" as IFACE

component "Fine-tuning: MultiModalWaveformRegressor\nstreams = tir, 0% mask" as FT
rectangle "1-D upsampling head -> y_pred" as HEAD

SHARED --> SWITCH
SWITCH --> A
SWITCH --> C
AI --> IFACE
CI --> IFACE
IFACE --> FT
FT --> HEAD

note right of IFACE
  Byte-for-byte equal keys and shapes.
  Either pre-trained checkpoint loads into
  the SAME fine-tuning model.
end note

note bottom of SHARED
  Only the pre-training stage differs;
  the data pipeline and the downstream
  fine-tuning recipe are identical.
end note

@enduml
```

---

## Notes

- `\n` inside quoted labels is a literal backslash-n (PlantUML line break), matching the
  convention used in `code/design/SolutionC_EarlyTokenMixing_stage2.md` and
  `code/design/SolutionC_EarlyTokenMixing_stage3.md`.
- Validate with:
  `curl -sSL -o /tmp/plantuml.jar https://github.com/plantuml/plantuml/releases/download/v1.2024.7/plantuml-1.2024.7.jar`
  then `java -jar /tmp/plantuml.jar -checkonly <block>.puml` and `-tpng` to render.
- Solution C is the current shipped design (`configs/pretrain/stage2_local_tir_roi_resp.yaml`);
  Solution A is the planned twin (`configs/pretrain/stage2_local_tir_roi_crossmae.yaml`).
