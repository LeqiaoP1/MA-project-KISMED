# Software Design Document: BP4D+ 2D Texture Facial ROI Ingestion and Preprocessing Pipeline

**Version: v2 (2026-11-10).** v1 described prerendered clip stacks; v2 matches
the artefact the implementation plan builds. Build order of record:
`code/notes/plan/2DTexture_ROI_cache_implementation_plan.md`. Geometry doctrine
shared with the thermal stream: `code/notes/design/ROI_crop_and_rescaling.md`.

| v1 → v2 change | why |
|---|---|
| Cache stores **native-resolution hull pixels** per `<S>_<T>`, not `(N_clips, 200, 96, 96, 3)` clip stacks | the TIR doctrine: *store native, rescale at load*. Window selection and every quality gate stay in the DATASET, so one cache serves any hop, any `input_size`, any fold (decision 1/2/11 of the plan) |
| Output resolution is a **load-time** property (`input_size`, pinned to 96 for every visual stream), not a cache property | `input_size` cannot change a stored byte, so it must not be in the cache key; the single resample happens at load |
| Quality gates (validity, sway, out-of-frame) are **dataset-level** with counters, not build-time filters | a build-time filter would bake selection knobs into the artefact and force a rebuild per experiment |
| Storage is a raw `.dat` memmap + a `.json` commit marker per session (~0.19 MB/frame), not a `.npz` | a materialising reader re-allocates hundreds of MB per sample under DataLoader shuffling; `.npz` cannot be memory-mapped |

A v1 reader may not be used with a v2 artefact (different shape, different key).

This document defines the software architecture, mathematical specifications, parametric configurations, locked architectural invariants, and implementation interfaces for the 2D facial texture video and landmark ingestion pipeline in the BP4D+ multimodal research framework. The pipeline extracts spatiotemporally aligned, standardized facial Regions of Interest (ROIs) from high-resolution 2D texture frames to serve as visual token inputs for multi-modal Masked Autoencoders (MultiMAE) alongside thermal video (TIR-ROI), respiration waveforms, and continuous arterial blood pressure signals.

---

## 1. System Context & Architectural Objectives

In the BP4D+ Multimodal Spontaneous Emotion Corpus, visual recordings consist of synchronized 3D geometric models, 2D facial texture videos, long-wave infrared (TIR) thermal sequences, and contact physiological records. The 2D texture stream contains high-definition color information capturing superficial capillary blood volume pulse (rPPG) dynamics, micro-expressions, and fine facial movement patterns.

The primary software objective of this pipeline is to ingest raw, uncompressed 2D texture video sequences and companion 2D facial landmark tracking files, isolate the face geometry, suppress severe head motion artifacts, and project the cropped face into a fixed-dimensional spatiotemporal tensor.

### Key Architectural Invariants & Requirements
1. **Geometric Precision & Invariant Square Cropping**: Derive facial bounding boxes from tracked 49-point facial landmarks. Square bounding box enforcement is a **non-negotiable structural invariant** (`square_crop = true` hardcoded), preventing non-affine facial stretching during spatial downscaling.
2. **Standardized Color Representation**: Output channel layout is locked strictly to **RGB format**, matching standard Vision Transformer pre-training backbones.
3. **Zero-Tolerance Landmark Tracking Gating**: The validity ratio is locked to **`min_valid_ratio = 1.0`** (100% valid frames required per clip). Clips containing even a single missing or dropped tracking frame (`[]`) are rejected.
4. **Output Resolution & Patch Divisibility**: the model-side spatial resolution is **$96 \times 96$ pixels** for *every* video-like stream (`tir` and `rgb2d` share one grid), giving an exact $6 \times 6 = 36$ patch grid per frame under $16 \times 16$ tokenization. This is a **load-time** property (`input_size`), not a stored one.
5. **One Resample, at Load**: the cache stores native source pixels; the only resampling in the pipeline is the per-clip `cv2.resize` from the clip box to `input_size`, locked to **`INTER_AREA`** when downscaling (`INTER_LINEAR` otherwise). No second resample is ever applied to already-resampled pixels.
6. **Multimodal Temporal Synchronization**: Guarantee frame-level temporal alignment across the 25 frames-per-second (fps) 2D video, 25 fps thermal camera stream, and 100 Hz resampled physiological channels (respiration voltage and blood pressure mmHg). v2 assumes jpg index == thermal frame index == `IRFeatures` line index (the assumption both existing arms already make); the F037 lead-frame case is a documented open item.
7. **I/O Throughput Optimization**: Alleviate filesystem bottlenecks caused by decoding individual high-resolution image frames ($1040 \times 1392$ pixels) on-the-fly during distributed model training through an offline caching subsystem.
8. **Store Native, Rescale At Load**: the cached artefact holds *source-resolution* hull pixels (`store_resolution = "native"`). `input_size`, `clip_stride`, `temporal_stride`, `max_sway_px` and `min_valid_ratio` therefore never appear in the cache key.
9. **Two-Level Out-of-Frame Gate**: a task whose hull is mostly outside the frame is skipped at build time, and a **clip** whose own box is mostly outside the frame is dropped at load time (50 % in-frame share). Black padding must never hand a mostly-empty clip to the model.

---

## 2. Parametric Configuration Schema & Architectural Invariants

The ingestion engine cleanly separates **tunable experimental parameters** from **locked architectural invariants**.

### 1. Locked Architectural Invariants
The following pipeline attributes are permanently fixed in the system architecture and are no longer user-configurable options:

* **`square_crop = true`**: Bounding boxes are always expanded to a symmetric 1:1 aspect ratio based on $\max(w_t, h_t)$ before isotropic padding. Non-square cropping is unsupported.
* **`min_valid_ratio = 1.0`**: Candidate clips must possess 100% valid, non-empty 49-point landmark annotations across every frame in the window. Partial or interpolated tracking is strictly forbidden. Applied at **load** (dataset filter), not at build.
* **`color_space = "RGB"`**: All decoded frames are converted from raw OpenCV BGR order to standard RGB tensor representations.
* **`input_size = 96`**: model-side output resolution for every visual stream, yielding 36 spatial patches of $16 \times 16$ per frame. A load-time knob (valid alternatives: 64, 112 — the same cache serves them), pinned to 96 in the shipped config because `tir` and `rgb2d` must share a grid.
* **`store_resolution = "native"`**: the cached artefact keeps source pixels. Locked: there is no integer `store_size`, because resampling in the builder would bake `input_size` into the artefact.
* **`interpolation = "area"`**: at load, `cv2.INTER_AREA` when downscaling and `cv2.INTER_LINEAR` otherwise (TIR parity).

### 2. Configurable Parameter Taxonomy

| Parameter Key | Type | Default Value | Valid Range | Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| `roi_padding` | float | `0.10` | `[0.0, 0.50]` | Fractional expansion applied to all four sides of the minimal facial landmark bounding box (10% isotropic margin). **In the cache key** (it changes stored pixels). |
| `clip_duration` | float | `8.0` | `[1.0, 30.0]` | Temporal window duration in seconds per training/evaluation sample ($8.0\text{ s} = 200\text{ frames}$). Load-time; only its value at build time (via `clip_seconds`) affects the hull. |
| `stride_duration` | float | `2.0` | `[0.5, 10.0]` | Temporal shift between consecutive clips. **Load-time only** — never in the key. |
| `max_sway_px` | float | measured (Phase 0) | `[0.0, 100.0]` | Maximum centroid excursion inside one window. Load-time selection rule. v1 proposed 15 px; at 1392x1040 that is ~0.06 face widths and is expected to be very aggressive, so the value is measured before being pinned. **Measured 2026-11-10** (`code/notes/eval/rgb2d/sway_retention.md`): median 8 s union sway is 65 px, 15 px keeps 0.9 % of windows and 40 px keeps 1 % of sessions, so the library default is `110` (`DEFAULT_RGB2D_MAX_SWAY_PX`, mirroring the TIR module's measured `DEFAULT_MAX_SWAY_PX = 40`; `0` = gate off) and the Stage-2 config pins the same `110` (the same statistic as the TIR gate, `landmark_sway`, converted on the same sessions so both gates reject the same fraction of clips: RGB reports 3.2× the TIR excursion, 40 px on TIR ↔ 110 px here, 128 px by pure scale matching; `code/notes/eval/rgb2d/tir_rgb2d_sway_ratio.py`). |
| `min_valid_ratio` | float | `1.0` | `(0.0, 1.0]` | Fraction of the window's frames that must carry a finite 49-point track. Load-time. |
| `area_min_frac` | float | `0.5` | `[0.0, 1.0]` | Minimum in-frame share of a clip's own box; below it the clip is dropped (`clips_dropped_2d_envelope`). Load-time. |
| `input_size` | int | `96` | multiple of the patch size | Model-side resolution (load-time rescale target). **Not in the key.** |
| `store_resolution` | string | `native` | `native` | Artefact resolution rule. **In the key.** |
| `source_policy` | string | `dir-then-zip` | `dir-then-zip` | Frame source resolution: extracted directory when present, else zip members decoded in memory. `2D+3D` is never extracted to disk. In the key. |
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

### 4. Boundary Padding and the Two Resolutions
When a subject sits near the camera image boundary, $X_{1, t}$ or $Y_{1, t}$ may evaluate to negative values, or $X_{2, t}, Y_{2, t}$ may exceed the image margins. The coordinates are clamped to image boundaries for extraction, and the missing regions are zero-padded to preserve the strict square aspect ratio:

$$\hat{X}_{1, t} = \max(0, \lfloor X_{1, t} \rfloor), \quad \hat{X}_{2, t} = \min(W_{\text{raw}}, \lceil X_{2, t} \rceil)$$

$$\hat{Y}_{1, t} = \max(0, \lfloor Y_{1, t} \rfloor), \quad \hat{Y}_{2, t} = \min(H_{\text{raw}}, \lceil Y_{2, t} \rceil)$$

A v2 artefact stores **source-resolution** pixels, so this section has two
distinct products:

1. **Stored (build time).** The square **hull** $B$ over every stride-1 clip box
   is cut from the frame and black-padded back to its own square extent, which
   may extend past the frame. It is written **without any resampling**:
   $$I_{\text{store}} \in \mathbb{R}^{T \times L \times L \times 3},\quad L = \text{hull side (native px)}$$
2. **Model input (load time).** Each clip's own box $B_{\text{clip}} \subseteq B$
   is cut out of the stored hull and resized **once**:
   $$I_{\text{target}} = \mathcal{R}_{[s,s]}(I_{\text{store}}[B_{\text{clip}}]) \in \mathbb{R}^{s \times s \times 3},\quad s = \texttt{input\_size} = 96$$
   with $\mathcal{R}$ = `INTER_AREA` when $s \le \min(w, h)$ and `INTER_LINEAR`
   otherwise. Because the hull contains every clip box (invariant I2), the
   translation $B_{\text{clip}} - B_{0}$ is always inside the stored array; a
   violation is an `IndexError`, never a silent clamp or wrap.

Consequence: the per-clip crop is **tight** (the face nearly fills the frame),
v1's fixed hull-relative framing no longer applies, and `input_size` is free.

**Coordinate frame (binding).** The stored array's origin **is** the hull origin
$B_0$, padding included: stored index `c` holds source column $x_0 + c$, and
the first `pad_left` columns are the black bar. So the load mapping is
`stored_index = source_coordinate - hull_origin`, with **no** pad term — `pad`
records where the black bars sit, it is never an offset. (Measured on a hull at
$x_0 = -3, y_0 = -2$: source `(1,1)` sits at `stored[3, 4]`; adding
`pad_top/pad_left` reads the wrong pixel and then runs off the array.)

---

## 4. Temporal Alignment, Windowing & Invariant Quality Gating

Because the video frames feed a multi-stream masked autoencoder requiring strict temporal synchrony, sequences are parsed into overlapping sliding windows with mathematical quality validation.

**Every gate in this section runs at LOAD time, inside the dataset, and is
counted per rule** (`clips_dropped_2d_invalid`, `_sway`, `_envelope`,
`_short`). None of them is a cache-key field: the cache stores pixels, the
dataset decides which windows are usable, so changing a threshold costs no
rebuild.

### 1. Clip Frame Count Contract
Given a sampling frequency of $f_{\text{video}} = 25.0\text{ fps}$, an $8.0\text{-second}$ clip requires an exact frame count:

$$N_{\text{frames}} = \lfloor \text{clip\_duration} \cdot f_{\text{video}} \rfloor = \lfloor 8.0 \cdot 25.0 \rfloor = 200\text{ frames}$$

With a temporal stride of $\Delta \tau = 2.0\text{ seconds}$, the frame offset between consecutive training windows is:

$$N_{\text{stride}} = \lfloor \text{stride\_duration} \cdot f_{\text{video}} \rfloor = \lfloor 2.0 \cdot 25.0 \rfloor = 50\text{ frames}$$

### 2. Multi-Stream Token Alignment Contract ($s \times s$ Grid, $s = 96$)
For an $8.0\text{-second}$ clip with $200$ video frames, the temporal token structure across the visual and physiological streams aligns on a uniform grid:

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

If $\Delta_{\text{sway}} > \texttt{max\_sway\_px}$, the clip is flagged as motion-corrupted and rejected (`clips_dropped_2d_sway`). The threshold is a **load-time** value measured in Phase 0 of the plan; v1's fixed 15 px is not adopted without that measurement, because at 1392x1040 it corresponds to ~0.06 face widths (TIR keeps 96.8 % of clips at 40 px on 726x480, i.e. ~0.4 face widths).

---

## 5. High-Performance Caching & Data Loader Architecture

Reading individual $1040 \times 1392$ `.jpg` files on-the-fly during distributed model training creates severe I/O bottlenecks. An uncompressed $8.0\text{-second}$ clip of 200 raw frames occupies approximately $290\text{ MB}$ on disk.

### 1. Offline Precomputation Subsystem
An offline build utility (`runners/run_build_2d_roi_cache.py`) processes all participant tasks in parallel ahead of model training. It stores **pixels only** — it never selects windows and never applies a selection rule:

1. **Dataset Traversal**: `2DFeatures/<S>_<T>.mat` and the frame source for the same session (extracted `2D+3D/<S>/<T>/*.jpg`, else the members of `2D+3D/<S>.zip` decoded in memory). **Nothing is ever extracted to disk.**
2. **Track Construction**: `parse_2d_features` → a `[n_jpg, 49, 2]` track with NaN where the 1-based `frame` field leaves a gap. An image/mat count mismatch skips the session (`image_count_mismatch`) — the v1 (`rgb_roi_dataset`) rule, kept so the cached track can never mis-align.
3. **Storage Box**: the square **hull** over the stride-1 clip windows of valid rows, black-padded to its own square extent when it crosses the frame; a task whose in-frame share is $< 0.5$ is skipped (`box_out_of_frame`).
4. **Crop & Serialize**: per frame, clamp → pad → `cv2.cvtColor(BGR2RGB)` → write straight into a memory-mapped `.dat`. **No resize.** A sibling `.lm.dat` holds the track and a `.json` holds the scalars; the `.json` is written last and is the commit marker.

Selection rules — `min_valid_ratio`, `max_sway_px`, the 50 % envelope gate, `clip_stride`, `input_size` — live in the **dataset**, not here. Skipping a session on a label/selection rule would make the cache depend on a knob it is supposed to be independent of.

### 2. Binary Storage Layout & Memory-Mapping
Per session `<S>_<T>`, under `<out_root>/<CACHE_KEY>/tasks/`:

| file | content | dtype / shape |
| :--- | :--- | :--- |
| `<S>_<T>.dat` | hull pixels, RGB | `uint8`, $(T, L_y, L_x, 3)$, $L_x = L_y = L$ (square) |
| `<S>_<T>.lm.dat` | the landmark track the shard was built from | `float32`, $(T, 49, 2)$ |
| `<S>_<T>.json` | `session, source, n_frames, box, pad, src_hw, hull_hw, store_resolution, area_in_frac`, plus both shapes/dtypes and the byte total | scalars |

* **Storage Footprint**: **MEASURED 2026-11-10** (`code/notes/eval/rgb2d/build_preflight.md`): 1.21 MB/frame, **1.37 GB per session**, **~1.9 TB of frame data (~2.25 TB on disk) for 1390 sessions** — i.e. about **2× smaller than the raw** 1040 × 1392 frames. The earlier ~0.19 MB/frame / ~0.4 TB estimate assumed a ~250 px hull; the real hull is the union over every stride-1 8 s window of the WHOLE task (design I2), so it is a task-wide motion envelope of 621–1036 px and ~3× wider than a clip box. That cost is disk only: the loader crops the *clip's* hull out of the stored task box before the rescale, so `input_size = 96` is not diluted. Measured quota on the Lichtenberg scratch: 40 TB with 623 GB used, so this is ~6 % of headroom. (v1's 5.53 MB-per-clip / 52.4x figures assumed prerendered 96 px pixels and no longer apply.)
* **No header, offset 0**: `.npy` was rejected because `np.lib.format.open_memmap` writes a 128-byte header and maps from offset 128, and because `shape`/`dtype` would then live in two places. Here they live only in the `.json`.
* **Runtime Access**: `np.memmap(path, dtype=..., mode="r", shape=...)`, sliced per clip. Nothing is materialised per shard; the OS page cache serves the reads and its pages are evictable. A `ShardHandle` (context manager) owns the two mappings and releases their file descriptors on `close()` — which drops the references, because calling `mmap.close()` behind a live numpy view raises `BufferError`.
* **Integrity**: a `.dat` without its `.json` is `incomplete` (never read); a `.dat` whose size disagrees with the `.json` shape is `truncated`; both are reported by `--status`, which also reports total bytes.

---

## 6. Illustrative interfaces (NOT the implementation contract)

The two sketches below show the *shape* of the geometry helpers. The build
order and exact signatures live in
`code/notes/plan/2DTexture_ROI_cache_implementation_plan.md`; when they
disagree, the plan wins.

### 1. Core Geometric Extractor Module (build side: box only, no resize)

```python
"""
Illustrative: square-hull geometry for BP4D+ 2D texture crops.
The BUILDER uses compute_bounding_box() only -- it stores native pixels.
The DATASET uses extract_and_resize() with the clip's own box (see 6.2).
"""
from typing import Tuple, Dict, Any
import numpy as np
import cv2

class FaceROIExtractor2D:
    # Locked Architectural Invariants
    SQUARE_CROP: bool = True
    COLOR_SPACE: str = "RGB"

    def __init__(self, config: Dict[str, Any]):
        # Configurable geometry padding (default 0.10)
        self.roi_padding = float(config.get("roi_padding", 0.10))
        # Load-time only: the builder never resizes
        self.input_size = int(config.get("input_size", 96))

    def compute_bounding_box(
        self, 
        landmarks: np.ndarray, 
        img_h: int, 
        img_w: int
    ) -> Tuple[int, int, int, int]:
        """
        SQUARE box from 49 facial landmarks -- NOT clamped to the frame.

        Build side: the unclamped span is what the hull unions over, and the
        builder black-pads wherever it leaves the image (invariant 9).
        Args:
            landmarks: Array of shape (N, 49, 2) in (x, y) source pixels.
            img_h: Image height in pixels (1040).
            img_w: Image width in pixels (1392).
        Returns:
            Half-open (x0, x1, y0, y1) in source pixels; may exceed the frame.
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
            
        # Return exact square coordinates (may be out of bounds)
        x1_sq = int(np.floor(x1))
        y1_sq = int(np.floor(y1))
        x2_sq = int(np.ceil(x2))
        y2_sq = int(np.ceil(y2))
        
        return x1_sq, y1_sq, x2_sq, y2_sq

    def extract_and_resize(
        self, 
        frame_bgr: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        LOAD TIME: crop one box, black-pad it square, RGB, resize once.

        The only resampling in the 2D pipeline. ``input_size`` is the model's
        spatial size, so the same cache serves 64 / 96 / 112.
        """
        x1, y1, x2, y2 = bbox
        img_h, img_w = frame_bgr.shape[:2]
        
        # Calculate clamping for extraction
        x1_clamp = max(0, x1)
        y1_clamp = max(0, y1)
        x2_clamp = min(img_w, x2)
        y2_clamp = min(img_h, y2)
        
        # Check area threshold (Completeness fix: Out-of-Frame Camera Disconnect)
        area_bbox = (x2 - x1) * (y2 - y1)
        area_clamp = (x2_clamp - x1_clamp) * (y2_clamp - y1_clamp)
        if area_clamp < 0.5 * area_bbox:
            raise ValueError(f"Crop area below 50% threshold from bbox: {bbox}")
            
        cropped = frame_bgr[y1_clamp:y2_clamp, x1_clamp:x2_clamp]
        
        # Pad to preserve square aspect ratio (Correctness fix)
        pad_top = y1_clamp - y1
        pad_bottom = y2 - y2_clamp
        pad_left = x1_clamp - x1
        pad_right = x2 - x2_clamp
        
        cropped_padded = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Invariant: Output color space is strictly RGB
        cropped_rgb = cv2.cvtColor(cropped_padded, cv2.COLOR_BGR2RGB)
            
        # The ONLY resample: to the model's input_size, kernel by TIR parity
        s = int(self.input_size)
        h, w = cropped_rgb.shape[:2]
        interp = cv2.INTER_AREA if s <= min(w, h) else cv2.INTER_LINEAR
        return cv2.resize(cropped_rgb, (s, s), interpolation=interp)
```

### 2. PyTorch Dataset Integration (illustrative)

The window (`frame_start`/`frame_end`) is **dataset-owned** and shared with the
TIR and RESP streams; the cache stores no clip index. The dataset crops the
clip's own square box out of the native hull and resizes it once (Step 3.2 of
the plan for the exact containment checks).

```python
"""
data/bp4d_2d_dataset.py
PyTorch Dataset implementing memory-mapped 2D texture facial ROI retrieval.
"""
from typing import Dict, Any, List
import json
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class BP4DPlus2DTextureDataset(Dataset):
    INPUT_SIZE: int = 96   # load-time; the artefact itself is native

    def __init__(self, manifest_path: str, config: Dict[str, Any], is_train: bool = True):
        self.config = config
        self.is_train = is_train
        
        with open(manifest_path, "r") as f:
            self.index_records: List[Dict[str, Any]] = json.load(f)
            
        self.clip_frames = int(config.get("clip_duration", 8.0) * config.get("source_fps", 25.0))
        self._mmap_cache: Dict[str, np.memmap] = {}

    def _get_mmap(self, dat_path: str, shape: tuple) -> np.memmap:
        if dat_path not in self._mmap_cache:
            self._mmap_cache[dat_path] = np.memmap(
                dat_path, 
                dtype=np.uint8, 
                mode="r", 
                shape=shape
            )
        return self._mmap_cache[dat_path]

    def __len__(self) -> int:
        return len(self.index_records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.index_records[idx]
        dat_path = record["cache_file"]
        f0, f1 = record["frame_start"], record["frame_end"]
        y0, y1, x0, x1 = record["hull_local_box"]   # square_span_2d - hull origin
        
        # Access memory-mapped binary array using cached handle
        mmap_arr = self._get_mmap(dat_path, tuple(record["mmap_shape"]))
        
        # Crop each frame's own square box out of the native hull, resize once.
        # (f0, f1) come from the dataset entry; (y0:y1, x0:x1) is hull-local.
        s = self.INPUT_SIZE
        clip_u8 = np.stack([
            cv2.resize(np.ascontiguousarray(mmap_arr[f0 + i, y0:y1, x0:x1]),
                       (s, s), interpolation=cv2.INTER_AREA)
            for i in range(f1 - f0)])
        
        # (T, s, s, C) uint8 -> PyTorch layout (C, T, s, s)
        clip_tensor = torch.from_numpy(clip_u8.transpose(3, 0, 1, 2))
        
        # Dataset contract: [0, 1]. The MODEL z-scores the clip
        # (core/input_adapters.zscore_clip); a dataset-level standardisation
        # would be a second, undocumented normalisation.
        clip_tensor = clip_tensor / 255.0

        return {
            "video_2d": clip_tensor,  # Shape: (3, T, 96, 96)
            "subject_id": record["subject_id"],
            "task_id": record["task_id"],
            "start_frame": record["start_frame"]
        }
```

---

## 7. Edge Cases, Failure Modes & Validation Rules

The software pipeline enforces explicit boundary and numerical guards against data anomalies documented in the BP4D+ corpus.

### Edge Case Matrix

| Fault Condition | Dataset Source Indicator | Pipeline Detection Mechanism | Software Mitigation / Resolution |
| :--- | :--- | :--- | :--- |
| **Empty Landmark Field** | `fit(t).pts_2d` is `[]` | `~np.isfinite(pts)` on the target subset (`valid_rows_2d`). | **Load-time gate**: a window whose invalid-frame fraction exceeds `1 - min_valid_ratio` (default 1.0, i.e. zero tolerance) is dropped as `clips_dropped_2d_invalid`. |
| **Image / Track Count Mismatch** | `n_jpg != n_mat`, or a `frame` field pointing outside the jpg range | Build-time count check; out-of-range rows discarded and their jpg indices left NaN. | Session skipped at build (`image_count_mismatch`), so the cached track can never mis-align with the images. |
| **Out-of-Frame Camera Disconnect** | Subjects F027_T4, F041_T7, M035_T6 | Two levels: the hull's in-frame share at build, then each **clip's** own box at load. | Black padding keeps the crop square; share $< 0.5$ skips the task (`box_out_of_frame`) or drops the clip (`clips_dropped_2d_envelope`). |
| **Asymmetric Multi-Stream Start** | Subject F037 (T1, T2) | Initial frame index discrepancy between 2D and thermal metadata. | **Open item**: v2 assumes index alignment (as both existing arms do). A landmark-only F037 check must be documented before the corpus build. |
| **Extreme Participant Sway** | Elicitation Tasks T7, T8, T9 | $\Delta_{\text{sway}} > \texttt{max\_sway\_px}$ across the window. | **Load-time gate** (`clips_dropped_2d_sway`). The threshold is MEASURED (Phase 0), not fixed at v1's 15 px; a selection rule never enters the cache key. |
| **Non-Zero Amplifier Clamping** | Physiological grounding | Sensor flatline or saturation ($-10\text{ V}$). | Discard physiological clip pairing if respiration or BP waveforms rail (existing RESP gate; unchanged by this stream). |

### Automated Unit Test Specification
Before initiating multi-fold training, the data engineering suite executes the following programmatic sanity checks:

1. **Aspect Ratio Preservation Test**: Feed synthetic landmark coordinates with extreme aspect ratios ($w/h = 2.5$). Verify that `compute_bounding_box()` produces a strict $1:1$ bounding box using the invariant square crop logic.
2. **One-Resample Test**: `extract_and_resize()` on a synthetic hull produces exactly `cv2.resize(crop, (s, s), INTER_AREA)` for $s \in \{64, 96, 112\}$ — the same artefact serving three `input_size` values.
3. **Memory-Map Concurrency Test**: Spawn 16 parallel processes accessing distinct offsets in the same `.dat` file. Assert zero segmentation faults and that a clip slice is a **view** (`s.base is not None`, `s.flags.owndata is False`).
4. **Token Divisibility Test**: Validate that $s = 96$ is strictly divisible by the ViT patch size ($16\text{ px}$), yielding $N_p = (96/16) \times (96/16) = 6 \times 6 = 36$ patches per frame — and that the same holds for `tir` so both visual streams report one grid.
5. **Hull Containment Test (I2)**: every stride-1 clip box is inside the stored hull, and a box one pixel outside raises `IndexError` rather than wrapping or clamping.
