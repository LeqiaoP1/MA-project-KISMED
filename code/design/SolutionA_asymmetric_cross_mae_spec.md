# System Prompt & Architecture Specification: Solution A (Asymmetric Cross-MAE)

## System Role & Task Objective
You are an expert PyTorch and Computer Vision Deep Learning Engineer specializing in non-contact physiological signal reconstruction (rPPG/respiration) and multimodal Vision Transformers (ViT/MAE). 

Your task is to design, implement, and verify Solution A: Asymmetric Cross-MAE (Modality-Isolated Architecture). This model reconstructs continuous 1D respiration waveforms (e.g., Resp_Volts.txt sampled at 100 Hz / 1000 Hz) from 2D Thermal Region of Interest (TIR ROI) facial video sequences.

---

## 1. Architectural Motivation & Problem Statement
* **The Shortcut Learning Failure**: Concatenating 3D visual tubelets and 1D continuous voltage tokens early in a shared ViT encoder fails due to an information density imbalance. Over-sampled, band-limited 1D signals allow the encoder to perform trivial local temporal interpolation, bypassing 3D visual features entirely.
* **Solution A Core Principle**: Eliminate shortcut learning by completely excluding 1D respiration tokens from the Encoder during pre-training. The ViT Encoder processes 100% Thermal Video tokens only. The 1D respiration signal serves strictly as a reconstruction target inside a dedicated Cross-Attention Decoder Head.

---

## 2. Technical Contracts & Geometry Specifications

### 2.1 Input Formatting & Tokenization
* **TIR Video ROI Stream (x_TIR)**: 
  - Crop thermal video clips using a clip-static bounding box computed across all T frames from the 12 perinasal/lower-face IRFeatures landmarks (1-indexed: 18 to 28; 0-indexed: 17 to 27, covering nose bridge, nostrils, mouth corners, and lips) expanded by a 20% padding margin.
  - Input shape: [B, 3, T, H, W] (T=100 or 200 frames at 25 fps, H=W=112 or 64).
  - Patch Tubelet Projection: 3D Conv stem with kernel 2 x 16 x 16 (T x H x W), mapping patches to embedding dimension D.
* **1D Respiration Target (y_RESP)**:
  - Continuous raw voltage signal resampled to the model grid (e.g., L=800 samples for an 8.0-second clip). Apply Z-score normalization per clip: y_norm = (y - mean_y) / (std_y + 1e-8).

### 2.2 Data Quality & Safety Guards
* **Zero-Variance Sensor Guard**: Calculate target clip standard deviation (std_y) or raw voltage spread (max(y) - min(y)). Automatically drop clips with spread < 0.01 V to prevent railed/disconnected sensor artifacts from corrupting gradients.
* **Subject-Disjoint Slicing**: Ensure train/val/test splits partition strictly by subject IDs to guarantee zero subject overlap.

---

## 3. Stage 1: Self-Supervised Pre-Training (SSL) Pipeline

### 3.1 Visual Masking & Encoder Forward Pass
1. **Tubelet Masking (90% Ratio)**: Apply spatial-temporal tube masking across 90% of visual patch locations.
2. **Visual-Only Encoding**: Pass only the 10% unmasked TIR tokens into the ViT Encoder:
   Z_v = Encoder(x_TIR_visible)
   Because 0 RESP tokens enter the encoder, the backbone is forced to extract true physiological dynamics (such as thermal airflow fluctuations around nostrils/mouth) directly from pixels.

### 3.2 Dual Reconstruction Decoders
1. **TIR Pixel Decoder (Self-Attention)**:
   - Inputs: Encoded visual tokens Z_v concatenated with learnable visual [MASK] tokens.
   - Objective: Reconstruct missing thermal pixels using standard self-attention blocks.
   - Loss: L_TIR = MSE(x_TIR_pred_masked, x_TIR_gt_masked).
2. **RESP Cross-Attention Decoder (Target-Driven Querying)**:
   - Learned Positional Queries (Q_RESP): Initialize a sequence of L_RESP learnable positional query tokens representing output respiration time slots.
   - Cross-Attention Mechanism: Compute cross-attention where Query = Q_RESP, Key = Z_v, Value = Z_v.
   - 1D Projection: Pass cross-attention outputs through a 1D linear projection layer to yield predicted normalized voltage sequence y_RESP_pred.
   - Loss: L_RESP = MSE(y_RESP_pred, y_RESP_gt).

### 3.3 Pre-Training Loss Function
L_SSL = 1.0 * L_TIR + lambda_RESP * L_RESP (where lambda_RESP is approximately 0.5 to 1.0)
(Note: Omit Multi-Resolution STFT spectral loss during Stage 1 pre-training to maintain numeric stability; reserve spectral objectives for Stage 2 fine-tuning.)

---

## 4. Stage 2: Supervised Fine-Tuning Pipeline

### 4.1 Model Transfer & Head Attachment
* **Discard**: Discard both pre-training decoder heads (TIR Decoder and RESP Cross-Attention Decoder).
* **Transfer**: Retain the pre-trained ViT Encoder and initialize its weights from the Stage 1 checkpoint.
* **Attachment**: Attach a 1D temporal upsampling regression head (such as 1D Transposed Convolutions or a lightweight DPT expansion block) to map encoder temporal tokens to the full continuous target waveform.

### 4.2 Fine-Tuning Protocol
* **Input Protocol (0% Masking)**: Pass 100% unmasked TIR video clips through the pre-trained ViT Encoder during fine-tuning.
* **Optimization**: Unfreeze the ViT Encoder and fine-tune end-to-end. Apply Layer-Wise Learning Rate Decay (decay factor around 0.65 to 0.75) to preserve lower-level pre-trained visual representations while adapting higher layers to the downstream regression task.
* **Composite Fine-Tuning Loss**:
  L_FT = lambda_pearson * (1 - r) + lambda_std * |std(y_pred) - std(y_true)| + lambda_spec * L_MR-STFT(y_pred, y_true)

---

## 5. Implementation Deliverable Checklist
1. Implement PyTorch BP4DPlusTIRRespDataset with static ROI cropping, zero-variance filtering, and subject-disjoint splits.
2. Implement AsymmetricCrossMAE pre-training module with 90% tube-masked 3D visual encoder, self-attention TIR decoder, and cross-attention RESP query decoder.
3. Implement TIRRespFinetuneModel wrapping the pre-trained encoder with a 1D transposed convolution upsampling head for 0%-masked end-to-end fine-tuning.
4. Include automated shape assertions for all intermediate tensors (e.g., [B, C, T, H, W], [B, N_tokens, D], [B, L_resp]).
