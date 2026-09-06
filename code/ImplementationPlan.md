Implementation Plan
-------------------

### **1. Core Content & Biophysical Grounding**

Your work establishes a contactless **Photometric-to-Physiological (2D-to-1D) generative recovery pipeline** under simulated sensor failure.

* **Scope:** Only **2D** modalities are considered. Surviving visual inputs are the **2D RGB video** (capturing sub-visual facial skin-color variations via rPPG) and the **2D Thermal Infrared (TIR) video** (detecting respiratory thermal fluctuations around the nostrils). **No 3D/depth** data is used, and the pipeline has **no face-ROI dependency** — full-frame input with a global resize/crop.
* **Target Waveforms (1D Outputs):** Reconstructing continuous, morphologically complete **Blood Volume Pulse (BVP)**, **Respiration (RESP)** and **Electrodermal Activity (EDA)** waveforms. This preserves rich clinical features — unlike simple scalar rate averages (BPM) — i.e. cardiac pulse morphology (BVP), respiratory rhythm (RESP), and tonic + phasic electrodermal dynamics (EDA).

---

### **2. The Three-Stage Progressive Pipeline**

To adapt a generic visual network to precise physiological wave generation, your architecture progresses through three distinct, increasingly specialized phases:

* **Stage 1: Spatial Initialization (Transferring Visual Priors)**

  * *Mechanism:* Initialize your heavy Vision Transformer (ViT-Base) encoder with weights pre-trained on **ImageNet-1K**.
  * *Objective:* Inherit highly robust, low-level spatial priors (edges, shapes, boundaries) to bypass the prohibitive computational cost of training a ViT from scratch.
* **Stage 2: Multimodal Pre-Training (Unsupervised Representation Learning)**

  * *Mechanism:* Train the ViT encoder on the **BP4D+ dataset** using **heavy asymmetric masking** (50%–75% masking on the RGB and TIR visual streams, but 90%+ on the three 1D streams — BVP, RESP and EDA).
  * *Objective:* Optimize the network using a point-level **masked MSE reconstruction loss**, computed **independently per modality over its masked positions only** and combined as a **weighted sum** across modalities.
  * *Weighting:* **λ_RGB = λ_TIR = 1.0**; every physio (1-D) stream uses **λ_physio ∈ [0.5, 1.0]** (default 0.5, `signal_weight`), tuned from the **normalized variance of the physio stream's masked target tokens** so the 1-D signals neither dominate the loss gradient (high variance ⇒ lean 0.5) nor get ignored (low variance ⇒ lean 1.0). This forces the shared self-attention layers to map facial visual variations to underlying cardiac (BVP), respiratory (RESP) and electrodermal/autonomic (EDA) dynamics.
* **Stage 3: Downstream Supervised Fine-Tuning (Task Adaptation)**

  * *Mechanism:* Simulate complete contact-sensor failure (100% masking of physical 1D streams). You duplicate your Stage 2 encoder and branch it into **three independent, task-specialized runs** — one for BVP regression, one for RESP regression, and one for EDA regression.
    Each run caps the encoder with a modality-restricted, parallel lightweight decoder.
  * *Optimization:* The restoration of BVP and RESP runs are trained end-to-end using a unified **Spatio-Temporal-Spectral Joint Loss Function**:

    $$
    \mathcal{L}_{\text{joint}} = \alpha \mathcal{L}_{\text{time}} + \beta \mathcal{L}_{\text{Pearson}} + \gamma \mathcal{L}_{\text{MR-STFT}}
    $$

    For the EDA task, the Loss function would be simplifed by removing the term because the EDA is event-driven aperiodic.

    ```math
    \mathcal{L}_{\text{MR-STFT}}
    ```
  * This mathematically constrains amplitude (L1 loss), temporal phase-locking (Negative Pearson), and multi-scale frequency dynamics (Multi-Resolution STFT over FFT window sizes of 64, 128, and 256).
    Branching the runs prevents gradient interference between high-frequency BVP (1.0–2.5 Hz), slower RESP (0.16–0.4 Hz) and slow tonic/phasic EDA (mostly < 0.5 Hz) waves. Because MR-STFT windows of 64/128/256 samples at fs = 100 Hz only resolve down to ≈1.6/0.78/0.39 Hz, the EDA branch uses longer STFT windows (e.g., 256/512/1024) to capture its slow tonic component.

---

### **3. Post-Processing Evaluation (Quantifying Information Loss)**

Your testing phase evaluates the generative quality of your waves using a systematic, multi-tiered framework:

* **Tier 1 (Temporal Alignment):** Global MAE, RMSE, and Pearson's \\(r\\). *Academic Nuance:* These are basic gatekeepers.
* **Tier 2 (Spectral Fidelity):** Welch PSD consistency and MR-STFT tracking to verify correct rhythmic frequencies.
* **Tier 3 (Clinical Fidelity & The Non-Differentiable Bottleneck):** HRV parameters depend on discrete peak-finding, which has zero or undefined gradients and cannot be used during training. You resolve this bottleneck by evaluating the predicted continuous waves completely **offline** through **NeuroKit2**, extracting clinically valid **RMSSD, pNN50, MedianNN, and Shannon Entropy (ShanEn)**.

---

### **4. Clarifications & Implementation Status (code/, 2026-09)**

#### 4.1 Clip reconstruction vs. target-signal reconstruction (important distinction)

* **Stage 2 reconstructs *inside a clip* (self-supervised masked autoencoding).** A clip is a self-contained multimodal sample (RGB + TIR + 1-D signals all *present*). A random subset of its tokens is masked per forward pass and the network reconstructs the masked tokens from the visible ones. The 1-D streams are masked hardest (90%+) so the visual streams must explain the physiology, but nothing is "missing" in the physical world — it is all already in the clip.
* **Stage 3 reconstructs the *target 1-D signal* of a clip (supervised regression / generative recovery).** Simulating full contact-sensor failure, only the visual streams are fed and the network must **generate the physiological waveform that is truly absent** (BVP / RESP / EDA). Training is per clip — `(video window -> waveform window [seq_len])` — optimised end-to-end with the joint time + Pearson + MR-STFT loss. The model never sees a whole session as a single training sample.

#### 4.2 Whole-session waveform = offline inference (not a training stage)

* A session (subject/task) is longer than one window, so the continuous whole-session waveform is produced **after training** by sliding the window over the session (`clip_duration` + `clip_stride`) and **overlap-adding (stitching)** the per-window predictions. This step trains **no** parameters — it is an inference/assembly step.
* STATUS: per-clip Stage-3 training + Tier-1/2/3 metric code exist (`runners/run_waveform.py`, `engines/waveform.py`, `runners/run_evaluate.py`, `evaluation/`). The **session-level reconstruction/stitching module is NOT yet implemented** (planned next).
* The restored whole-session waveform is then evaluated **offline** with all tiers: **Tier 1** (MAE/RMSE/Pearson), **Tier 2** (Welch / MR-STFT), and **Tier 3** (NeuroKit2 HRV — BVP/HRV-specific; RESP/EDA use their own offline spectral/clinical measures).

#### 4.3 Code status snapshot

* **Stage 1 — encoder weight inheritance (implemented).** `core/multimae.py::load_pretrained_encoder()` copies a MAE ViT-Base checkpoint (`blocks.*` → `enc_blocks.*`, `norm.*` → `enc_norm.*`) and inflates the 2-D `patch_embed.proj` into the RGB 3-D tubelet. Enabled by `--pretrained_encoder` / `configs/pretrain/stage2_local_pretrained.yaml`. An official ImageNet-1K (timm) classifier converter remains optional.
* **Stage 2 — multimodal MAE local milestone (implemented & run).** `core/multimae.py` (`MultiModalMAE`: rgb+tir tubelets, bvp signal windows, per-stream asymmetric masks, shared encoder+decoder, per-stream linear heads, normalized-patch **masked MSE + weighted sum** — see §4.4); `data/paired_dataset.py::PairedPretrainDataset` + `build_pretraining_dataset`; `runners/run_pretrain.py` multimodal branch; configs `configs/pretrain/stage2_local{,_pretrained}.yaml`. Remaining: RESP/EDA streams, larger 224 geometry, full-data HPC run.
* **Stage 3 — scaffold (implemented, needs a Stage-2 ckpt).** `core/waveform_losses.py::WaveformJointLoss` (α·L1 + β·(−Pearson) + γ·MR-STFT; per-branch FFT sizes), `runners/run_waveform.py`, `engines/waveform.py::evaluate_waveforms`, `runners/run_evaluate.py`, configs `configs/finetune/{bvp,resp,eda}.yaml`. Finer temporal decoder + session stitching are to-do.
* **Evaluation (implemented, offline).** `evaluation/metrics.py` (Tier 1 time, Tier 2 Welch PSD), `evaluation/clinical.py` (Tier 3 NeuroKit2: RMSSD/pNN50/MedianNN/ShanEn).

#### 4.4 Training knobs & fixes (reproducibility notes)

* **LR semantics (`runners/run_pretrain.py`, official-MAE).** An explicit `--lr` is the absolute peak LR. The `blr` linear-scaling rule (`lr = blr*batch*world/256`) applies **only** when `--lr` is omitted (large-batch HPC). For small-batch local runs always set `--lr`; otherwise LR collapses to ~1e-6 and training stalls (this was the cause of an earlier perfectly-flat Stage-2 loss; the per-epoch masked-recon loss — L1 at the time — was constant across 40 epochs with rel. weight change ~1e-6).
* **LR schedule.** A step-level warmup + cosine schedule is applied (`utils/lr_sched.py::cosine_scheduler`); `warmup_epochs` and `min_lr` are now honoured.
* **Stage-2 loss & per-modality weighting (`core/multimae.py`).** The Stage-2 objective is a **masked MSE**, evaluated **independently per modality over its masked positions only** (`mask == 1` ⇒ to reconstruct; `core/criterion.py::MaskedMSELoss`), then combined as a **weighted sum** `L = Σ_s λ_s·MSE_s`. Default policy: **λ_RGB = λ_TIR = 1.0**, and every physio (1-D) stream uses **λ = `--signal_weight`** (default **0.5**). Choose the physio weight in **~[0.5, 1.0]** from the **normalized variance of the physio stream's masked target tokens** so the 1-D signals do not dominate the gradient (high variance ⇒ lean 0.5) nor get ignored (low variance ⇒ lean 1.0). `--loss_weights` is a full per-stream override (one comma value per `--streams` modality, in order; empty ⇒ policy above). `MultiModalMAE.forward` returns `losses_mse` (raw per-modality masked MSE) and `losses` (weighted contributions); `loss = Σ losses`. NOTE: the older L1-sum baselines are not comparable to the current MSE scale.
* **Window sampling.** `clip_stride` (`data/paired_dataset.py`) enables overlapping windows to scale clip count per session (`0`/`None` = non-overlapping default). Keep `clip_stride >= clip_duration/2` to limit redundancy; real diversity comes from covering all sessions/subjects/tasks, not from overlap or from more epochs. Random per-forward masking remains the primary augmentation.
