# Engineering Project Guide & Architecture Specification: Thermal ROI to Respiration Waveform Reconstruction (`TirROI-Resp`)

This document serves as an exhaustive engineering and architecture specification for automated code generation. It outlines data contracts, geometric constraints, preprocessing specifications, model architectures, loss formulations, and evaluation protocols for reconstructing continuous 1D respiration waveforms from BP4D+ thermal video ROI clips.

---

## Section 1: Environment Setup & Accelerator Diagnostics

### 1.1 Dependency Verification & Package Imports
* **Objective**: Audit and import core machine learning, computer vision, and signal processing libraries.
* **Required Libraries**: `torch`, `torchvision`, `cv2` (OpenCV), `scipy.signal`, `scipy.fft`, `numpy`, `pandas`, `matplotlib`.
* **Execution Contract**: Verify that all submodules (e.g., `torch.utils.data.DataLoader`, `scipy.fft.fft`) load without missing symbols or version mismatches.

### 1.2 Hardware & Accelerator Diagnostics
* **Objective**: Inspect CUDA runtime state and establish execution hardware properties.
* **Verification Checks**: Audit GPU device count, device name, compute capability, total VRAM, and driver compatibility.
* **Runtime Flags**: Initialize automatic mixed-precision (`torch.cuda.amp.autocast`) runtime context and verify Tensor Core availability.

### 1.3 Determinism & Seed Calibration
* **Objective**: Ensure exact experiment reproducibility across dataset slicing and model weight initialization.
* **Calibration Contract**: Set identical random seeds across Python `random`, NumPy (`np.random.seed`), and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`).
* **Backend State**: Enforce `torch.backends.cudnn.deterministic = True` and disable `torch.backends.cudnn.benchmark`.

---

## Section 2: Global Configuration & Geometry Contracts

### 2.1 File System & Path Specification
* **Dataset Root**: `/BP4D+/`
* **Thermal Video Location**: `/BP4D+/Thermal/{Subject}/{Subject}_{Task}.wmv` (25 fps).
* **Thermal Landmark File Location**: `/BP4D+/IRFeatures/{Subject}_{Task}.txt` (28 2D landmark points per frame).
* **Ground-Truth Respiration Location**: `/BP4D+/Physiology/{Subject}/{Task}/Resp_Volts.txt` (1000 Hz single-column voltage).

### 2.2 Thermal Landmark Geometry Contract (12 Mouth + Nose Points)
* **Target ROI Specification**: Isolate 12 perinasal and lower-facial feature points defining nostrils, upper/lower lips, and mouth corners (BP4D+ User Guide Figure 3).
* **1-Indexed Landmark Labels**: `[9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]`
* **0-Indexed Array Indices**: `[8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23]`

### 2.3 Temporal & Spatial Hyperparameter Specifications
* **Temporal Clip Duration**: $8.0\text{ seconds}$ sliding window.
* **Video Frame Count ($T$)**: $200\text{ frames}$ at $25\text{ fps}$.
* **Respiration Sample Count ($N$)**: $8000\text{ samples}$ at $1000\text{ Hz}$.
* **Bounding Box Expansion Margin**: $20\%$ expansion padding relative to bounding box height and width.
* **Spatial Crop Target Size**: $112 \times 112\text{ pixels}$.

### 2.4 Task Distortion Grouping & Filtering Contracts
* **Group A — Low Distortion (Passive Baseline)**: Task 2 (Graphic Show), Task 3 (Sadness Video Clip).
* **Group B — Moderate / Controlled Distortion (Stress/Pain/Threat)**: Task 4 (Startle Probe), Task 7 (Physical Threat), Task 8 (Cold Pressor Pain), Task 10 (Unpleasant Smell).
* **Group C — Excluded High Distortion (Speech/Vocalization)**: Task 1 (Interview/Jokes), Task 5 (Interview/Questions), Task 6 (Improvised Song), Task 9 (Interview/Feedback).

---

## Section 3: Data Ingestion & Thermal ROI Pre-Processing

### 3.1 Landmark File Parser (`parse_ir_features`)
* **Input**: Path to landmark text file (`/BP4D+/IRFeatures/{Subject}_{Task}.txt`).
* **Processing Specification**: Parse space/tab-delimited rows containing 28 $(X, Y)$ coordinate pairs per frame.
* **Exception Contract**: Gracefully catch and skip missing or untracked feature sequences (e.g., untracked subject sequences `M049`, `F016_T2..T4`).
* **Output Tensor Shape**: `(num_frames, 28, 2)` NumPy array.

### 3.2 Clip-Level Static Bounding Box Extractor (Strategy 2)
* **Input**: 200-frame video sequence and corresponding `(200, 28, 2)` landmark array.
* **Coordinate Extraction**: Slice coordinates for the 12 target mouth/nose indices across all 200 frames in the window.
* **Global Bounds Calculation**: Compute global temporal envelope bounds:
  $$X_{\min} = \min_{t, i} X_{t, i}, \quad X_{\max} = \max_{t, i} X_{t, i}, \quad Y_{\min} = \min_{t, i} Y_{t, i}, \quad Y_{\max} = \max_{t, i} Y_{t, i}$$
* **Expansion Padding**: Calculate width $W_b = X_{\max} - X_{\min}$ and height $H_b = Y_{\max} - Y_{\min}$. Expand bounds by $0.2 \times W_b$ and $0.2 \times H_b$.
* **Boundary Clamping**: Clamp final box coordinates to frame boundaries $[0, W_{\text{frame}}-1]$ and $[0, H_{\text{frame}}-1]$.
* **Spatial Crop & Resize**: Crop all 200 frames using this **identical clip-static box** and resize each patch to $112 \times 112$ pixels via bilinear interpolation (eliminates spatial warping noise).

### 3.3 Respiration Waveform Loader & Quality Guards
* **Input**: Path to target text file (`/BP4D+/Physiology/{Subject}/{Task}/Resp_Volts.txt`).
* **Zero-Variance Guard**: Calculate target clip standard deviation $\sigma_y$. Reject clip if $\sigma_y < 1e-6$ (detects sensor rail or disconnection failures).
* **Sample Alignment Guard**: Validate that physiological sample count matches video frame count ($1000\text{ Hz} / 25\text{ fps} = 40$ physiological points per frame).

### 3.4 Spatial & Temporal Normalization Pipelines
* **Spatial Normalization**: Rescale cropped thermal frame pixel values to $[0.0, 1.0]$.
* **Waveform Normalization**: Apply Z-score normalization to target respiration clip $y$:
  $$y_{\norm} = \frac{y - \mu_y}{\sigma_y + 1e-8}$$

---

## Section 4: PyTorch Dataset & DataLoader Architecture

### 4.1 Subject-Disjoint Train/Val/Test Split Generator
* **Partitioning Contract**: Split 140 BP4D+ subjects strictly into mutually exclusive sets: Training ($70\%$), Validation ($15\%$), Testing ($15\%$).
* **Data Leakage Guard**: Ensure zero subject overlap between sets to prevent temporal clip memorization.

### 4.2 Dataset Implementation (`BP4DPlusTIRRespDataset`)
* **Class Inheritance**: `torch.utils.data.Dataset`.
* **Index Building**: Construct a lookup table of valid 8-second clip tuples `(subject, task, frame_start_idx)` using a 2-second temporal stride during dataset initialization.
* **Output Item Contract**:
  * `'tir_video'`: `FloatTensor` of shape `(1, 200, 112, 112)` ($[C, T, H, W]$).
  * `'resp_signal'`: `FloatTensor` of shape `(8000,)` ($[N]$).
  * `'subject_task'`: String identifier (`'F001_T3'`).

### 4.3 DataLoader Construction & Data Integrity Verifier
* **Instantiation**: Construct PyTorch `DataLoader` instances for training and validation splits with configurable batch size and worker processes.
* **Batch Shape Assertions**: Execute an automated tensor assertion check verifying that batch tensors strictly match expected shapes and memory layouts.

---

## Section 5: Model Architecture Specification (`MultiModalMAE`)

### 5.1 Thermal Visual Adapter & Patch Embedder
* **Patch Stem**: Construct a 3D Convolutional Stem or Tubelet Projection layer with patch dimension $2 \times 16 \times 16$ ($T \times H \times W$).
* **Linear Projection**: Map projected tubelets to hidden embedding dimension $D$.
* **Positional Encoding**: Add learnable 1D spatial-temporal positional embeddings.

### 5.2 Joint Spatiotemporal Transformer Backbone
* **Encoder Stack**: Construct $L$ stacked Vision Transformer encoder blocks containing Multi-Head Self-Attention (MHSA) and MLP networks.
* **Normalization**: Apply Layer Normalization prior to attention blocks with residual skip connections.

### 5.3 1D Respiration Waveform Decoder Head
* **Temporal Upsampling Network**: Construct a 1D decoder using transposed 1D convolutions or linear projection layers.
* **Dimension Transformation**: Map latent temporal representations from $T=200$ tokens to $N=8000$ physiological output samples.

### 5.4 Parameter Freezing & Layer-Wise Learning Rate Allocation
* **Pre-trained Weight Ingestion**: Support loading pre-trained weights from VideoMAE or MultiMAE backbones.
* **Layer-Wise Decay**: Enforce layer-wise learning rate decay factor ($0.65$) to preserve low-level visual filters while adapting upper layers.

---

## Section 6: Stage-2 Multimodal Masked Pre-Training Pipeline

### 6.1 Spatial-Temporal Tubelet Masking Engine
* **Visual Masking**: Apply high-ratio tubelet masking ($80\% - 90\%$) randomly across spatial-temporal visual patches.
* **Physiological Masking**: Apply span masking across 1D respiration signal tokens.

### 6.2 Pre-Training Objective & Loss Calculation
* **Masked MSE Loss**: Compute Mean Squared Error restricted solely to masked visual patches and masked waveform spans.

### 6.3 Pre-Training Loop & Checkpoint State Management
* **Optimization Loop**: Execute self-supervised pre-training loop over designated pre-training epochs.
* **State Preservation**: Save encoder backbone weights and optimizer states to disk.

---

## Section 7: Stage-3 Supervised Fine-Tuning Pipeline

### 7.1 Multi-Objective Anti-Collapse Loss Function (`CompositeLoss`)
* **Objective**: Eliminate variance collapse and peak flattening by combining shape alignment, dynamic range matching, and log spectral power preservation:
  $$\mathcal{L}_{\text{total}} = \lambda_{\text{pearson}} \cdot (1 - r) + \lambda_{\text{std}} \cdot |\text{std}(\hat{y}) - \text{std}(y)| + \lambda_{\text{spec}} \cdot \left\| \log(|\text{FFT}(\hat{y})| + \epsilon) - \log(|\text{FFT}(y)| + \epsilon) \right\|_1$$
* **Hyperparameter Configuration**: Set default weights $\lambda_{\text{pearson}} = 1.0$, $\lambda_{\text{std}} = 1.0$, $\lambda_{\text{spec}} = 1.0$.

### 7.2 Fine-Tuning Optimization Strategy
* **Optimizer**: AdamW optimizer with weight decay $0.05$.
* **Learning Rate Schedule**: Cosine Annealing learning rate scheduler with initial warm-up.
* **Gradient Stability**: Apply gradient norm clipping ($\text{max\_norm} = 1.0$).

### 7.3 Training & Validation Loop Execution
* **Execution**: Execute supervised fine-tuning loop over specified epochs.
* **Checkpoint Selection**: Evaluate validation metrics after every epoch and preserve model weights achieving peak validation Pearson correlation.

---

## Section 8: Model Evaluation & Signal Diagnostics

### 8.1 Time-Domain Metric Computation
* **Mean Absolute Error (MAE)**: $\frac{1}{N}\sum |y_i - \hat{y}_i|$
* **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$
* **Pearson Correlation Coefficient ($r$)**: $\frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2 \sum (\hat{y}_i - \bar{\hat{y}})^2}}$

### 8.2 Frequency-Domain & Spectral Diagnostics
* **Power Spectral Density (PSD)**: Compute PSD using Welch's method across $0.1\text{ Hz} - 0.5\text{ Hz}$ respiration band.
* **Dominant Peak Extraction**: Locate spectral peak to extract predicted Respiration Rate in Breaths Per Minute (BPM).
* **Respiration Rate Error**: Calculate MAE between predicted BPM and ground-truth BPM.

### 8.3 Task-Wise Performance Stratification Analysis
* **Task Disaggregation**: Evaluate model metrics separately across Low Distortion (Tasks 2, 3) and Moderate Distortion (Tasks 4, 7, 8, 10) groups.
* **Performance Reporting**: Generate comparative tables detailing metric shifts across task categories.

---

## Section 9: Visualization & Qualitative Analysis

### 9.1 Time-Domain Overlay Waveform Renderer
* **Waveform Comparison**: Generate overlay time-series plots comparing target ground-truth signals (grey) vs. model predictions (blue) across 8-second clips.
* **Fidelity Inspection**: Inspect peak alignment, breath count, and amplitude height.

### 9.2 Spectral Power Distribution Comparison Plots
* **Frequency Overlay**: Render log-scale PSD comparison curves to confirm frequency peak alignment and energy preservation.

### 9.3 Task Distortion Group Performance Comparison Chart
* **Summary Visualization**: Generate bar plots illustrating Pearson correlation and BPM MAE metrics across individual BP4D+ tasks.
