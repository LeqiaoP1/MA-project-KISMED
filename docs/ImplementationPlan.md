Implementation Plan
-------------------

### **1. Core Content & Biophysical Grounding**

Your work establishes a contactless **Photometric-to-Physiological (2D-to-1D) generative recovery pipeline** under simulated sensor failure.

* **Surviving Inputs (Contactless Visuals):** Standard **2D RGB video** (capturing sub-visual facial skin color variations via rPPG) and **2D Thermal Infrared (TIR) video** (detecting respiratory thermal fluctuations around the nostrils).
* **Target Waveforms (1D Outputs):** Reconstructing continuous, morphologically complete **Blood Volume Pulse (BVP)**
  and **Respiration (RESP)** waveforms. This preserves rich clinical features, unlike simple scalar rate averages (BPM)
  and **Electrodermal Activity (EDA)**

---

### **2. The Three-Stage Progressive Pipeline**

To adapt a generic visual network to precise physiological wave generation, your architecture progresses through three distinct, increasingly specialized phases:

* **Stage 1: Spatial Initialization (Transferring Visual Priors)**
  * *Mechanism:* Initialize your heavy Vision Transformer (ViT-Base) encoder with weights pre-trained on **ImageNet-1K**.
  * *Objective:* Inherit highly robust, low-level spatial priors (edges, shapes, boundaries) to bypass the prohibitive computational cost of training a ViT from scratch.
* **Stage 2: Multimodal Pre-Training (Unsupervised Representation Learning)**
  * *Mechanism:* Train the ViT encoder on the **BP4D+ dataset** using **heavy asymmetric masking** (50%–75% masking on visual streams, but 90%+ on 1D BVP and RESP streams).
  * *Objective:* Optimize the network using simple, point-level **L1/MSE reconstruction losses** on masked patches. This forces the shared self-attention layers to map facial visual variations to underlying cardiac and respiratory rhythms.
* **Stage 3: Downstream Supervised Fine-Tuning (Task Adaptation)**
  * *Mechanism:* Simulate complete contact-sensor failure (100% masking of physical 1D streams). You duplicate your Stage 2 encoder and branch it into **two independent, task-specialized runs**—one for BVP regression and one for RESP regression.
    Each run caps the encoder with a modality-restricted, parallel lightweight decoder.
  * *Optimization:* Both runs are trained end-to-end using a unified **Spatio-Temporal-Spectral Joint Loss Function**:
    $$
    \mathcal{L}_{\text{joint}} = \alpha \mathcal{L}_{\text{time}} + \beta \mathcal{L}_{\text{Pearson}} + \gamma \mathcal{L}_{\text{MR-STFT}}
    $$
  * This mathematically constrains amplitude (L1 loss), temporal phase-locking (Negative Pearson), and multi-scale frequency dynamics (Multi-Resolution STFT over FFT window sizes of 64, 128, and 256).
    Branching the runs prevents gradient interference between high-frequency BVP (1.0–2.5 Hz) and slower RESP (0.16–0.4 Hz) waves.

---

### **3. Post-Processing Evaluation (Quantifying Information Loss)**

Your testing phase evaluates the generative quality of your waves using a systematic, multi-tiered framework:

* **Tier 1 (Temporal Alignment):** Global MAE, RMSE, and Pearson's \\(r\\). *Academic Nuance:* These are basic gatekeepers; point-level error can be deceptively minimized by a flat "average" line that lacks physiological peaks.
* **Tier 2 (Spectral Fidelity):** Welch PSD consistency and MR-STFT tracking to verify correct rhythmic frequencies.
* **Tier 3 (Clinical Fidelity & The Non-Differentiable Bottleneck):** HRV parameters depend on discrete peak-finding, which has zero or undefined gradients and cannot be used during training. You resolve this bottleneck by evaluating the predicted continuous waves completely **offline** through **NeuroKit2**, extracting clinically valid **RMSSD, pNN50, MedianNN, and Shannon Entropy (ShanEn)**.

---
