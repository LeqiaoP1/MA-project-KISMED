Implementation Plan
-------------------

### **1. Core Content & Biophysical Grounding**

Your work establishes a contactless **Photometric-to-Physiological (2D-to-1D) generative recovery pipeline** under simulated sensor failure.

* **Scope:** Only **2D** modalities are considered. Surviving visual inputs are the **2D RGB video** (capturing sub-visual facial skin-color variations via rPPG) and the **2D Thermal Infrared (TIR) video** (detecting respiratory thermal fluctuations around the nostrils). **No 3D/depth** data is used, and the pipeline has **no face-ROI dependency** — full-frame input with a global resize/crop.
* **TIR encoding — verified 2026-09, do NOT assume "gray".** The BP4D thermal `.wmv` is a **false-colour (rainbow) thermal *rendering*** with the camera's °C legend burned into the right edge, not a gray thermal image. Two independent decoders agree (OpenCV 5.0 and PyAV 18.1): `wmv3`/`yuv420p`, **3 planes** decoded, mean `|U-128| ≈ 23-27`, `|V-128| ≈ 16-20` on **all 11** local sessions, with the chroma plane tracking the scene (not noise, not a flat cast). Raw `Thermal/<S>/<T>.wmv` vs canonical `tir.wmv` are byte-identical (md5 `ae2d0247c921662a4b322e62f040143f`) ⇒ source data, not a preprocessing artefact. The pipeline therefore treats TIR as **3-channel** (`tir_channels: 3`, default): the Stage-2 TIR adapter is `Conv3d(3→D)` (`core/multimae.STREAM_CHANNELS`, mirrored in `core/au_probe.py`) and the Stage-3 input is `3+3 = 6` channels (`in_chans: 6` in `configs/finetune/{bvp,resp,eda}.yaml`). `--tir_channels 1` restores the legacy luma-only surrogate and is needed only to match Stage-2 checkpoints trained before 2026-09 (it changes the TIR adapter geometry). **Thesis caveat:** this makes the "thermal" branch a *rendered* proxy whose palette depends on the camera's per-frame auto-range, not radiometric temperature — state it wherever TIR is claimed to carry temperature. The legend is cropped away by `resize_center_crop` at 64 px and 224 px.
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
* **Stage 2 — multimodal MAE local milestone (implemented & run).** `core/multimae.py` (`MultiModalMAE`: rgb+tir tubelets + bvp/resp/eda signal windows — any **≥1 video + ≥1 physio** subset is accepted, five-stream default; per-stream asymmetric masks, shared encoder+decoder, per-stream linear heads, normalized-patch **masked MSE + weighted sum** — see §4.4); `data/paired_dataset.py::PairedPretrainDataset` (serves *exactly* the configured streams) + `build_pretraining_dataset`; `runners/run_pretrain.py` multimodal branch; configs `configs/pretrain/stage2_local{,_pretrained}.yaml` (all five streams). The Stage-2 modality contract (≥2 streams: ≥1 video `rgb`/`tir` **and** ≥1 1-D physio `bvp`/`resp`/`eda`, the Stage-3 regression target) is enforced in `MultiModalMAE.__init__` + `build_pretraining_model` and in the dataset builders. Remaining: larger 224 geometry, full-data HPC run, separate deeper decoders.
* **Stage 3 — scaffold (implemented, needs a Stage-2 ckpt).** `core/waveform_losses.py::WaveformJointLoss` (α·L1 + β·(−Pearson) + γ·MR-STFT; per-branch FFT sizes), `runners/run_waveform.py`, `engines/waveform.py::evaluate_waveforms`, `runners/run_evaluate.py`, configs `configs/finetune/{bvp,resp,eda}.yaml`. Finer temporal decoder + session stitching are to-do.
* **Evaluation (implemented, offline).** `evaluation/metrics.py` (Tier 1 time, Tier 2 Welch PSD), `evaluation/clinical.py` (Tier 3 NeuroKit2: RMSSD/pNN50/MedianNN/ShanEn).

#### 4.4 Training knobs & fixes (reproducibility notes)

* **LR semantics (`runners/run_pretrain.py`, official-MAE).** An explicit `--lr` is the absolute peak LR. The `blr` linear-scaling rule (`lr = blr*batch*world/256`) applies **only** when `--lr` is omitted (large-batch HPC). For small-batch local runs always set `--lr`; otherwise LR collapses to ~1e-6 and training stalls (this was the cause of an earlier perfectly-flat Stage-2 loss; the per-epoch masked-recon loss — L1 at the time — was constant across 40 epochs with rel. weight change ~1e-6).
* **LR schedule.** A step-level warmup + cosine schedule is applied (`utils/lr_sched.py::cosine_scheduler`); `warmup_epochs` and `min_lr` are now honoured.
* **Stage-2 loss & per-modality weighting (`core/multimae.py`).** The Stage-2 objective is a **masked MSE**, evaluated **independently per modality over its masked positions only** (`mask == 1` ⇒ to reconstruct; `core/criterion.py::MaskedMSELoss`), then combined as a **weighted sum** `L = Σ_s λ_s·MSE_s`. Default policy: **λ_RGB = λ_TIR = 1.0**, and every physio (1-D) stream uses **λ = `--signal_weight`** (default **0.5**). Choose the physio weight in **~[0.5, 1.0]** from the **normalized variance of the physio stream's masked target tokens** so the 1-D signals do not dominate the gradient (high variance ⇒ lean 0.5) nor get ignored (low variance ⇒ lean 1.0). `--loss_weights` is a full per-stream override (one comma value per `--streams` modality, in order; empty ⇒ policy above). `MultiModalMAE.forward` returns `losses_mse` (raw per-modality masked MSE) and `losses` (weighted contributions); `loss = Σ losses`. NOTE: the older L1-sum baselines are not comparable to the current MSE scale.
* **Window sampling.** `clip_stride` (`data/paired_dataset.py`) enables overlapping windows to scale clip count per session (`0`/`None` = non-overlapping default). Keep `clip_stride >= clip_duration/2` to limit redundancy; real diversity comes from covering all sessions/subjects/tasks, not from overlap or from more epochs. Random per-forward masking remains the primary augmentation.
* **RGB decode cost, measured (`data/video_io.py`, 2026-09).** ~90 % of the per-frame loader cost is libjpeg **decode CPU**, not I/O: reading the jpg bytes costs 0.6 ms/frame, reading + full decode 11.6 ms/frame, and decoding bytes already in RAM with no filesystem at all still costs 9.2 ms/frame (a 1.4 MP source downsampled 97 % for a 224 px target). `read_image` therefore selects a DCT-scaled decode (`IMREAD_REDUCED_{COLOR,GRAYSCALE}_{8,4,2}`) from the source size parsed out of the JPEG SOF header (`_jpeg_size`, no decode), keeping the coarsest scale that still covers `target_size` and never upscaling (`target_size=None`, non-JPEG inputs and small sources keep the full decode). Measured: 224 px -> `_4`, 13.7 -> 9.9 ms/frame (1.38x, mean |diff| 0.69/255); 64 px -> `_8`, 1.34x; 512 px -> `_2`, 1.15x; a 10 s / 250-frame clip at 224 px 3450 -> 2630 ms. Shapes are unchanged for every `target_size` x `gray` combination. The stream that repeats is RGB: TIR is decoded **once per session** into `_tir_cache` at dataset construction, while RGB re-reads 250 jpgs per 10 s window, so one worker sustains only ~0.35 clips/s and `num_workers` sizes the loader throughput. A materialised uint8 `.npy` store (4.5 ms/clip measured, ~770x) would remove the remaining decode and is NOT implemented.

---

### **5. Downstream Probe — AU Occurrence Detection (Semantic Representation Quality)**

An **ADD-ON / diagnostic** module (separate from Stages 1-3; it re-uses the Stage-2 encoder but changes nothing in the Stage code paths). A regression-only read-out is a weak witness of representation quality, so this probe tests whether the frozen Stage-2 features **linearly separate FACS facial-action semantics**: a linear (or fine-tuned) head on top of the Stage-2 shared encoder predicts AU *occurrence*.

* **Probe = multi-label binary relevance, per-AU F1.** BCE loss; per-AU F1 @0.5, macro-averaged over the standard BP4D 12-AU subset `{1,2,4,6,7,10,12,14,15,17,23,24}`. AU rows/values `9` (unknown) are dropped (never treated as negative). Prediction is **clip-level but centre-frame anchored**: one label vector per `num_frames` clip, supervised by the centre frame only.
* **Data facts (verified).** AU coding exists for tasks **T1/T6/T7/T8 only** — 560 files = 140 subjects (82F/58M) × 4 tasks, each subject exactly `{T1,T6,T7,T8}`. Files are `AUCoding/AU_OCC/<Session>.csv` with a header row of AU ids and a 1-based global frame column; cell `0/1/9`. AU frame `f` ⇔ RGB jpg number `f-1` (canonical `rgb/` keeps the raw names). Locally only subjects `F001..F004` (tasks T1,T2) are downloaded ⇒ only `F00X_T1` sessions are AU-usable — local = smoke only.
* **Configurable AU set.** `--au_list` (explicit, string `"6,7,10,12,14"` or YAML list) **or** `--au_freq_topk N` (auto top-N AUs by presence rate over the full AU corpus, independent of which media is downloaded; `N=5` ⇒ the most frequent `AU7/10/14/12/6` = `{6,7,10,12,14}`). Default = the BP4D 12-AU subset. The head width, dataset label columns and per-AU table all follow this list, so the probe can be restricted to any AU subset (e.g. the frequent, near-constant ones — note those are also the *least* discriminative targets, so report alongside a trivial baseline).
* **Local configs = one probe variant per locally-trained encoder.** `configs/finetune/au_local.yaml` probes `stage2_local` (tiny 192-d/6-layer from-scratch; clip 4.0 s → num_frames 100, batch 4) with the default full 12-AU subset; `configs/finetune/au_local_pretrained.yaml` probes `stage2_local_pretrained` (MAE ViT-Base-inherited 768-d/12-layer; clip 4.0 s → num_frames 100, batch 2) and deliberately uses the 5 most-frequent near-constant AUs `{6,7,10,12,14}`. Each config's geometry MUST mirror its probed checkpoint (Code bullet). Both use the SAME window length; the smaller batch (2 vs 4) keeps the ~16× bigger 768-d encoder on a LOCAL GPU. Within one geometry, keep the protocol identical across C0/C1/C2 (C1 = the MAE 768-d checkpoint only fits the 768-d geometry); the two configs use different AU sets by design, so they are not a head-to-head encoder A/B — fix one AU set for that.
* **Controls (identical probe protocol, only `--finetune` changes).** C0 random init (lower bound) → C1 Stage-1 MAE/ImageNet (`--finetune base`, auto-downloaded into `models/initial/`) → **C2 Stage-2** (BP4D multimodal masked pre-training, the headline). Expect `C2 ≥ C1 ≫ C0` ⇒ masked pre-training causes linear AU decodability. `--probe linear` freezes the encoder; `--probe ft` fine-tunes.
* **Splits: subject-disjoint (mandatory).** AU sessions are split by subject prefix (`F00X`), never frame/session; full runs use leave-N-subjects-out over the 140 subjects; explicit `--train_subjects/--val_subjects` or `--train_ratio`.
* **Caveats for the thesis.** (i) Stage-2 protocol is pre-train-on-all, so the probe measures representational fidelity on the training distribution, not OOD generalisation — state this. (ii) Coding covers the *most expressive* ~15-28 s per task, so base rates of some AUs are very high (e.g. AU7/AU10/AU14 ≈ 60-66%) — report per-AU F1 **and a trivial baseline** (always-positive per AU), since macro-F1 alone overstates a degenerate all-positive head on near-constant AUs.
* **Code (all ADD-ON files).** `data/au_dataset.py` (`AuOccurrenceDataset`, `build_au_datasets` subject split), `core/au_probe.py` (`MultiModalMAEProbe` = Stage-2 encoder reuse + mean-pool + head; `load_au_probe_weights` loads Stage-2 or MAE checkpoints with a geometry guard; `weights_only=False`), `engines/au_probe.py` (`train_one_epoch_au`, `evaluate_au`), `runners/run_au_probe.py`, configs `configs/finetune/au_local{,_pretrained}.yaml` (geometry MUST reproduce the probed Stage-2 checkpoint: `tubelet/input_size/clip_duration→num_frames/enc_embed_dim/enc_depth/enc_num_heads`), `scripts/local/au_smoke.sh`. Visual-stream-only input (no physio leakage).
* **Status.** End-to-end local smoke passes on `F001..F004_T1` (subject split, Stage-2 ckpt load: 78 encoder tensors, forward `[B,12]`, BCE+per-AU F1, best-ckpt saved). Real numbers need full media on HPC; the loader cost is RGB *decode* CPU rather than the disk (see §4.4 and the README's *Data-loading throughput*), so `num_workers` is the effective knob.
