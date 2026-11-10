# Software Design Document: BP4D+ 2D Texture Facial ROI Ingestion and Preprocessing Pipeline

This document defines the software architecture, mathematical specifications, parametric configurations, locked architectural invariants, and implementation interfaces for the 2D facial texture video and landmark ingestion pipeline in the BP4D+ multimodal research framework. The pipeline extracts spatiotemporally aligned, standardized facial Regions of Interest (ROIs) from high-resolution 2D texture frames to serve as visual token inputs for multi-modal Masked Autoencoders (MultiMAE) alongside thermal video (TIR-ROI), respiration waveforms, and continuous arterial blood pressure signals.

---

## 1. System Context & Architectural Objectives

In the BP4D+ Multimodal Spontaneous Emotion Corpus, visual recordings consist of synchronized 3D geometric models, 2D facial texture videos, long-wave infrared (TIR) thermal sequences, and contact physiological records. The 2D texture stream contains high-definition color information capturing superficial capillary blood volume pulse (rPPG) dynamics, micro-expressions, and fine facial movement patterns.

The primary software objective of this pipeline is to ingest raw, uncompressed 2D texture video sequences and companion 2D facial landmark tracking files, isolate the face geometry, suppress severe head motion artifacts, and project the cropped face into a fixed-dimensional spatiotemporal tensor.

### Key Architectural Invariants & Requirements
1. **Geometric Precision & Invariant Square Cropping**: Derive facial bounding boxes from tracked 49-point facial landmarks. Square bounding box enforcement is a **non-negotiable structural invariant** (`square_crop = true` hardcoded), preventing non-affine facial stretching during spatial downscaling.
2. **Standardized Color Representation**: Output channel layout is locked strictly to **RGB format**, matching standard Vision Transformer pre-training backbones.
3. **Zero-Tolerance Landmark Tracking Gating**: The validity ratio is locked to **`min_valid_ratio = 1.0`** (100% valid frames required per clip). Clips containing even a single missing or dropped tracking frame (`[]`) are rejected.
4. **Target Resolution & Patch Divisibility**: Output spatial resolution is locked to **$96 \times 96$ pixels**, providing an exact $6 \times 6 = 36$ patch grid per frame under standard $16 \times 16$ Vision Transformer tokenization.
5. **Fixed Anti-Aliasing Resampling**: Downsampling from high-resolution raw crops to target resolution is locked to OpenCV **`INTER_AREA`** resampling, guaranteeing consistent anti-aliasing without user configuration overhead.
6. **Multimodal Temporal Synchronization**: Guarantee frame-level temporal alignment across the 25 frames-per-second (fps) 2D video, 25 fps thermal camera stream, and 100 Hz resampled physiological channels (respiration voltage and blood pressure mmHg).
7. **I/O Throughput Optimization**: Alleviate filesystem bottlenecks caused by decoding individual high-resolution image frames ($1040 \times 1392$ pixels) on-the-fly during distributed model training through an offline caching subsystem.

---

## 2. Parametric Configuration Schema & Architectural Invariants

The ingestion engine cleanly separates **tunable experimental parameters** from **locked architectural invariants**.

### 1. Locked Architectural Invariants
The following pipeline attributes are permanently fixed in the system architecture and are no longer user-configurable options:

* **`square_crop = true`**: Bounding boxes are always expanded to a symmetric 1:1 aspect ratio based on $\max(w_t, h_t)$ before isotropic padding. Non-square cropping is unsupported.
* **`min_valid_ratio = 1.0`**: Candidate clips must possess 100% valid, non-empty 49-point landmark annotations across every frame in the window. Partial or interpolated tracking is strictly forbidden.
* **`color_space = "RGB"`**: All decoded frames are converted from raw OpenCV BGR order to standard RGB tensor representations.
* **`target_size = [96, 96]`**: Standardized output resolution of $96 \times 96$ pixels, yielding 36 spatial patches of $16 \times 16$ per frame (saving ~43.75% token memory relative to $128 \times 128$).
* **`interpolation = "area"`**: Hardcoded to `cv2.INTER_AREA`, providing optimal area relation pixel decimation without Moiré artifacts.

### 2. Configurable Parameter Taxonomy

| Parameter Key | Type | Default Value | Valid Range | Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| `roi_padding` | float | `0.10` | `[0.0, 0.50]` | Fractional expansion applied to all four sides of the minimal facial landmark bounding box (10% isotropic margin). |
| `clip_duration` | float | `8.0` | `[1.0, 30.0]` | Temporal window duration in seconds per training/evaluation sample ($8.0\text{ s} = 200\text{ frames}$). |
| `stride_duration` | float | `2.0` | `[0.5, 10.0]` | Temporal shift between consecutive overlapping clips during session sliding-window assembly ($2.0\text{ s} = 50\text{ frames}$). |
| `max_sway_px` | float | `15.0` | `[5.0, 100.0]` | Maximum allowable translation (Euclidean drift) of the face center point within a single clip window. |
| `cache_format` | string | `npy_memmap` | `npy_memmap`, `h5`, `tar` | Binary serialization format utilized by the offline ROI cache builder. |
| `tubelet_t_size` | int | `2` | `1, 2, 4` | Number of consecutive temporal frames merged into a single 3D tubelet token. |



## 3. Mathematical Formulation & Geometric Extraction

The raw 2D texture images in BP4D+ are stored as discrete color images of dimensions $W_{\text{raw}} = 1392$ and $H_{\text{raw}} = 1040$ pixels. Companion landmark metadata files contain 49 two-dimensional coordinates per frame.

### 1. Landmark Parsing & Coordinate Extraction
For each frame $t \in [1, T]$, the tracking record `fit(t)` provides the landmark coordinate matrix:

$$P_t = \{ (x_i, y_i) \}_{i=1}^{49}, \quad x_i \in [0, W_{\text{raw}}), \; y_i \in [0, H_{\text{raw}})$$

The tracked 49 points delineate the structural landmarks of the face:
* Eyebrows: Points 1 to 10
* Eye Contours: Points 20 to 31
* Nose Bridge and Base: Points 11 to 19
* Outer and Inner Lips: Points 32 to 49

### 2. Bounding Box Derivation
The tight bounding box enclosing all 49 landmark points at frame $t$ is defined by its extremal coordinates:

$$x_{\min, t} = \min_{i \in [1, 49]} x_{i, t}, \quad x_{\max, t} = \max_{i \in [1, 49]} x_{i, t}$$

$$y_{\min, t} = \min_{i \in [1, 49]} y_{i, t}, \quad y_{\max, t} = \max_{i \in [1, 49]} y_{i, t}$$

The instantaneous width and height of the raw facial enclosure are:

$$w_t = x_{\max, t} - x_{\min, t}, \quad h_t = y_{\max, t} - y_{\min, t}$$

The facial center of mass $C_t = (c_{x, t}, c_{y, t})$ is computed as:

$$c_{x, t} = \frac{x_{\min, t} + x_{\max, t}}{2}, \quad c_{y, t} = \frac{y_{\min, t} + y_{\max, t}}{2}$$

### 3. Invariant Square Enforcement & 10% Padding Expansion
To ensure consistent geometric proportions across subjects and eliminate non-affine facial distortion, the crop is strictly forced to a square aspect ratio. The base side length $S_t$ is determined by the maximum dimension:

$$S_t = \max(w_t, h_t)$$

Applying the configured isotropic expansion factor $\alpha = \text{roi\_padding} = 0.10$, the final padded bounding box side length $L_t$ is:

$$L_t = S_t \cdot (1 + 2\alpha) = S_t \cdot 1.20$$

The continuous bounding box coordinates before boundary clamping are:

$$X_{1, t} = c_{x, t} - \frac{L_t}{2}, \quad X_{2, t} = c_{x, t} + \frac{L_t}{2}$$

$$Y_{1, t} = c_{y, t} - \frac{L_t}{2}, \quad Y_{2, t} = c_{y, t} + \frac{L_t}{2}$$

### 4. Boundary Clamping & Spatial Resizing to $96 \times 96$
When a subject sits near the camera image boundary, $X_{1, t}$ or $Y_{1, t}$ may evaluate to negative values, or $X_{2, t}, Y_{2, t}$ may exceed the image margins. The coordinates are clamped to image boundaries:

$$\hat{X}_{1, t} = \max(0, \lfloor X_{1, t} \rfloor), \quad \hat{X}_{2, t} = \min(W_{\text{raw}}, \lceil X_{2, t} \rceil)$$

$$\hat{Y}_{1, t} = \max(0, \lfloor Y_{1, t} \rfloor), \quad \hat{Y}_{2, t} = \min(H_{\text{raw}}, \lceil Y_{2, t} \rceil)$$

The extracted sub-image $I_{\text{crop}, t} \in \mathbb{R}^{H_{\text{sub}} \times W_{\text{sub}} \times 3}$ is converted from raw BGR to RGB and resampled using area interpolation ($\text{INTER\_AREA}$) to yield the target tensor:

$$I_{\text{target}, t} = \mathcal{R}_{[96, 96]}(I_{\text{crop}, t}) \in \mathbb{R}^{96 \times 96 \times 3}$$

---

## 4. Temporal Alignment, Windowing & Invariant Quality Gating

Because the video frames feed a multi-stream masked autoencoder requiring strict temporal synchrony, sequences are parsed into overlapping sliding windows with mathematical quality validation.

### 1. Clip Frame Count Contract
Given a sampling frequency of $f_{\text{video}} = 25.0\text{ fps}$, an $8.0\text{-second}$ clip requires an exact frame count:

$$N_{\text{frames}} = \lfloor \text{clip\_duration} \cdot f_{\text{video}} \rfloor = \lfloor 8.0 \cdot 25.0 \rfloor = 200\text{ frames}$$

With a temporal stride of $\Delta \tau = 2.0\text{ seconds}$, the frame offset between consecutive training windows is:

$$N_{\text{stride}} = \lfloor \text{stride\_duration} \cdot f_{\text{video}} \rfloor = \lfloor 2.0 \cdot 25.0 \rfloor = 50\text{ frames}$$

### 2. Multi-Stream Token Alignment Contract ($96 \times 96$ Grid)
For an $8.0\text{-second}$ clip with $200$ video frames, the temporal token structure across all four modalities aligns on a uniform grid:

* **2D-RGB Texture Stream ($96 \times 96$)**: 
  * Spatial patches per frame ($16 \times 16$ kernel):
    $$N_{\text{spatial}} = \left(\frac{96}{16}\right) \times \left(\frac{96}{16}\right) = 6 \times 6 = 36\text{ patches}$$
  * Temporal tubelets ($t_{\text{kernel}} = 2$, $t_{\text{stride}} = 2$):
    $$T_{\text{tokens, 2D}} = \frac{200}{2} = 100\text{ temporal steps}$$
  * Total 2D video tokens per clip: $100 \times 36 = 3,600\text{ tokens}$.
* **3D TIR-ROI Thermal Stream**: Standardized at $96 \times 96$, tokenized with identical $2 \times 16 \times 16$ kernels, yielding $3,600\text{ tokens}$ per clip.
* **1D Physiological Streams (RESP & BP)**: 1000 Hz Biopac signals resampled to 100 Hz yield $N_{\text{signal}} = 8.0 \cdot 100 = 800\text{ samples}$. Tokenized with kernel size 16 and stride 16 ($160\text{ ms}$ per step):
  $$T_{\text{tokens, Physio}} = \frac{800}{16} = 50\text{ tokens}$$

### 3. Invariant Zero-Tolerance Tracking Gating (`min_valid_ratio = 1.0`)
BP4D+ landmark tracking files represent missing tracking data as an empty array `[]`. A frame $t$ is marked valid if and only if:

$$V_t = \begin{cases} 1, & \text{if } \text{fit}(t).\text{pts\_2d} \neq \emptyset \text{ and } \text{shape}(\text{fit}(t).\text{pts\_2d}) = (49, 2) \\ 0, & \text{otherwise} \end{cases}$$

Under the locked invariant $\text{min\_valid\_ratio} = 1.0$, a candidate clip spanning frames $[t_{\text{start}}, t_{\text{start}} + N_{\text{frames}} - 1]$ is valid if and only if every single frame is tracked:

$$\sum_{k=0}^{N_{\text{frames}}-1} V_{t_{\text{start}} + k} = N_{\text{frames}} = 200$$

Any window containing even one un-tracked or dropped frame is rejected.

### 4. Dynamic Sway Gating
Large participant head lunges create motion blur and boundary truncation. For all frames within the window, the maximum Euclidean displacement of the facial center relative to the window's mean center is bounded:

$$\bar{C} = \frac{1}{N_{\text{frames}}} \sum_{k=0}^{N_{\text{frames}}-1} C_{t_{\text{start}} + k}$$

$$\Delta_{\text{sway}} = \max_{k \in [0, N_{\text{frames}}-1]} \| C_{t_{\text{start}} + k} - \bar{C} \|_2$$

If $\Delta_{\text{sway}} > \text{max\_sway\_px} = 15.0\text{ px}$, the clip is flagged as motion-corrupted and rejected.

---

## 5. High-Performance Caching & Data Loader Architecture

Reading individual $1040 \times 1392$ `.jpg` files on-the-fly during distributed model training creates severe I/O bottlenecks. An uncompressed $8.0\text{-second}$ clip of 200 raw frames occupies approximately $290\text{ MB}$ on disk.

### 1. Offline Precomputation Subsystem
An offline build utility (`runners/run_build_2d_roi_cache.py`) processes all participant tasks in parallel ahead of model training:

1. **Dataset Traversal**: Inspect `/BP4D+/2D+3D/<Subject>/<Task>/` directories and corresponding `2DFeatures/<Subject>_<Task>.mat` files.
2. **Quality & Motion Verification**: Enforce 100% landmark validity and sway displacement $\le 15.0\text{ px}$ across contiguous $8.0\text{-second}$ windows.
3. **Optimized Cropping & Scaling**: Compute the invariant square padded bounding box with $\alpha = 0.10$, convert to RGB, and apply `cv2.resize` to $96 \times 96$ using invariant `INTER_AREA` interpolation.
4. **Binary Serialization**: Save processed clip tensors into a unified, memory-mapped binary array per subject-task session.

### 2. Binary Storage Layout & Memory-Mapping ($96 \times 96$)
The cached arrays are stored as raw binary arrays accompanied by JSON metadata descriptors:
* **Shape**: $(N_{\text{clips}}, N_{\text{frames}}, 96, 96, 3)$
* **Data Type**: Unsigned 8-bit integer (`uint8`, values $0 \dots 255$)
* **Storage Footprint**: An $8.0\text{-second}$ clip resized to $96 \times 96 \times 3$ occupies exactly:
  $$200 \times 96 \times 96 \times 3 \text{ bytes} = 5,529,600\text{ bytes} \approx 5.53\text{ MB}$$
  This represents a **52.4-fold reduction in data volume** compared to raw images while drastically cutting DataLoader memory pressure.
* **Runtime Access**: PyTorch `DataLoader` workers access the `.dat` files using `numpy.memmap` in `r` (read-only) mode. This avoids heap allocations, allowing the operating system page cache to service data reads with zero serialization latency.

---

## 3. Only for illustration, not as reference of implementation interfaces

### 1. Core Geometric Extractor Module (`data/roi_extractor.py`)

```python
"""
data/roi_extractor.py
Robust bounding box computation and spatial ROI extraction for BP4D+ 2D texture images.
Enforces invariant square cropping, RGB output conversion, fixed 96x96 resolution, 
and invariant cv2.INTER_AREA interpolation.
"""
from typing import Tuple, Dict, Any
import numpy as np
import cv2

class FaceROIExtractor2D:
    # Locked Architectural Invariants
    TARGET_SIZE: Tuple[int, int] = (96, 96)
    INTERPOLATION = cv2.INTER_AREA
    SQUARE_CROP: bool = True
    COLOR_SPACE: str = "RGB"

    def __init__(self, config: Dict[str, Any]):
        # Configurable geometry padding (default 0.10)
        self.roi_padding = float(config.get("roi_padding", 0.10))

    def compute_bounding_box(
        self, 
        landmarks: np.ndarray, 
        img_h: int, 
        img_w: int
    ) -> Tuple[int, int, int, int]:
        """
        Computes the expanded, squared, clamped bounding box from 49 facial landmark coordinates.
        Args:
            landmarks: Array of shape (49, 2) in (x, y) pixel coordinates.
            img_h: Image height in pixels (1040).
            img_w: Image width in pixels (1392).
        Returns:
            Tuple (x1, y1, x2, y2) in integer image coordinates.
        """
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        w = x_max - x_min
        h = y_max - y_min
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        
        # Invariant: Square cropping is strictly enforced
        side = max(w, h)
        side_padded = side * (1.0 + 2.0 * self.roi_padding)
        
        x1 = cx - side_padded / 2.0
        x2 = cx + side_padded / 2.0
        y1 = cy - side_padded / 2.0
        y2 = cy + side_padded / 2.0
            
        # Boundary clamping to image limits
        x1_clamp = max(0, int(np.floor(x1)))
        y1_clamp = max(0, int(np.floor(y1)))
        x2_clamp = min(img_w, int(np.ceil(x2)))
        y2_clamp = min(img_h, int(np.ceil(y2)))
        
        return x1_clamp, y1_clamp, x2_clamp, y2_clamp

    def extract_and_resize(
        self, 
        frame_bgr: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crops the bounding box from the frame, applies invariant RGB conversion, and downscales to 96x96.
        """
        x1, y1, x2, y2 = bbox
        cropped = frame_bgr[y1:y2, x1:x2]
        
        if cropped.size == 0:
            raise ValueError(f"Extracted crop has zero area from bbox: {bbox}")
            
        # Invariant: Output color space is strictly RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            
        # Invariant: Resized to (96, 96) using INTER_AREA
        resized = cv2.resize(
            cropped_rgb, 
            self.TARGET_SIZE, 
            interpolation=self.INTERPOLATION
        )
        return resized
```

### 2. PyTorch Dataset Integration (`data/bp4d_2d_dataset.py`)

```python
"""
data/bp4d_2d_dataset.py
PyTorch Dataset implementing memory-mapped 2D texture facial ROI retrieval.
"""
from typing import Dict, Any, List
import json
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class BP4DPlus2DTextureDataset(Dataset):
    TARGET_SIZE: tuple = (96, 96)

    def __init__(self, manifest_path: str, config: Dict[str, Any], is_train: bool = True):
        self.config = config
        self.is_train = is_train
        
        with open(manifest_path, "r") as f:
            self.index_records: List[Dict[str, Any]] = json.load(f)
            
        self.clip_frames = int(config.get("clip_duration", 8.0) * config.get("source_fps", 25.0))

    def __len__(self) -> int:
        return len(self.index_records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.index_records[idx]
        dat_path = record["cache_file"]
        clip_idx = record["clip_idx"]
        
        # Access memory-mapped binary array without loading whole file
        mmap_arr = np.memmap(
            dat_path, 
            dtype=np.uint8, 
            mode="r", 
            shape=tuple(record["mmap_shape"])
        )
        
        # Sliced array shape: (clip_frames, 96, 96, 3)
        clip_raw = np.array(mmap_arr[clip_idx], dtype=np.float32)
        
        # Transform to PyTorch standard tensor layout: (C, T, H, W) -> (3, 200, 96, 96)
        clip_tensor = torch.from_numpy(clip_raw).permute(3, 0, 1, 2)
        
        # Per-clip normalization (Scale to [0, 1] then Standardize mean=0, std=1)
        clip_tensor = clip_tensor / 255.0
        mean = clip_tensor.mean(dim=(1, 2, 3), keepdim=True)
        std = clip_tensor.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        clip_tensor = (clip_tensor - mean) / std

        return {
            "video_2d": clip_tensor,  # Shape: (3, 200, 96, 96)
            "subject_id": record["subject_id"],
            "task_id": record["task_id"],
            "start_frame": record["start_frame"]
        }
```

---

## 4. Edge Cases, Failure Modes & Validation Rules

The software pipeline enforces explicit boundary and numerical guards against data anomalies documented in the BP4D+ corpus.

### Edge Case Matrix

| Fault Condition | Dataset Source Indicator | Pipeline Detection Mechanism | Software Mitigation / Resolution |
| :--- | :--- | :--- | :--- |
| **Empty Landmark Array** | `fit(t).pts_2d == []` | Check `len(pts_2d) == 0` during frame ingestion. | **Strict Invariant**: Drop candidate clip immediately (`min_valid_ratio = 1.0`). |
| **Out-of-Frame Camera Disconnect** | Subjects F027_T4, F041_T7, M035_T6 | Visual coordinates evaluate outside $[0, W_{\text{raw}}]$ or $[0, H_{\text{raw}}]$. | Boundary clamping preserves in-bounds regions. If $\text{area}_{\text{clamp}} < 0.5 \cdot \text{area}_{\text{bbox}}$, discard clip. |
| **Asymmetric Multi-Stream Start** | Subject F037 (T1, T2) | Initial frame index discrepancy between 2D and thermal metadata. | Enforce synchronization by trimming lead frames to match the physiological Biopac trigger index. |
| **Extreme Participant Sway** | Elicitation Tasks T7, T8, T9 | $\Delta_{\text{sway}} > \text{max\_sway\_px} = 15.0\text{ px}$ across the 200-frame window. | Filter clip from pre-training dataset to prevent ViT positional embedding corruption. |
| **Non-Zero Amplifier Clamping** | Physiological grounding | Sensor flatline or saturation $(-10\text{ V})$. | Discard physiological clip pairing if respiration or BP waveforms rail. |

### Automated Unit Test Specification
Before initiating multi-fold training, the data engineering suite executes the following programmatic sanity checks:

1. **Aspect Ratio Preservation Test**: Feed synthetic landmark coordinates with extreme aspect ratios ($w/h = 2.5$). Verify that `compute_bounding_box()` produces a strict $1:1$ bounding box using the invariant square crop logic.
2. **Deterministic Resizing Test**: Ingest test frames from Subject `F001_T1`. Confirm that `extract_and_resize()` outputs an RGB array of exact shape $(96, 96, 3)$ with non-zero variance.
3. **Memory-Map Concurrency Test**: Spawn 16 parallel processes accessing distinct offsets in the same `.dat` file. Assert zero segmentation faults and verify read throughput exceeding $1.5\text{ GB/sec}$.
4. **Token Divisibility Test**: Validate that `TARGET_SIZE = (96, 96)` is strictly divisible by the ViT patch size ($16\text{ px}$), yielding an integer number of spatial patches $N_p = (96/16) \times (96/16) = 6 \times 6 = 36$ patches per frame.
