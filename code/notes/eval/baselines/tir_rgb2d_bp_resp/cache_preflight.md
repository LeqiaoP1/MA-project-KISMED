# Four-stream cache preflight — 2026-11-21

- **Commit**: `b193414`
- **Raw root**: `/work/projects/l0003511/test`
- **Verdict**: **PASS**

This record covers the four artefacts required by
`configs/pretrain/stage2_fourstream_tir_rgb2d_bp_resp_small_random.yaml`:
TIR-ROI, 2D-texture ROI (`px112`), 1000 Hz RESP, and 1000 Hz BP.

## 1. Cache roots and keys

| Stream | Root | Key fields |
|---|---|---|
| `tir` | `/work/scratch/ne95ocyg/tir_roi_cache/tirroi_nose_mouth_pad0.10_square_window-hull_clip8s_src726x480_native_v3_fa9f91bdd5` | `clip_seconds=8`, `fps=25`, `roi_padding=0.10`, `crop_rule=square`, landmarks `nose_mouth` |
| `rgb2d` | `/work/scratch/ne95ocyg/rgb2d_roi_cache/rgb2droi_face_pad0.10_square_window-hull_clip8s_src1040x1392_px112_v1_9ecebe2386` | `clip_seconds=8`, `fps=25`, `roi_padding=0.10`, `crop_rule=square`, `face`, `store_resolution=px112` |
| `resp` | `/work/scratch/ne95ocyg/resp_cache/respgrid_src1000_fs1000_sat9.95-1s_flat0.3-5s_run8s_v1_9cdddaff98` | `fs_in=1000`, `fs_out=1000`, `min_run_s=8` |
| `bp` | `/work/scratch/ne95ocyg/bp_artefact/bpsession_src1000_range30-250_flat200ms_rail30ms_seg4s_v1_a89590f953` | `fs_in=fs_out=1000`, `min_segment_s=4`, frozen store |

The native RGB2D cache also exists, but this run uses the `px112` store because
the config sets `rgb2d_store: 112` and `input_size: 96`.

## 2. Coverage (`--status`)

| Cache | Cached | Skipped | Missing | Size |
|---|---:|---:|---:|---:|
| TIR-ROI | 1385 | 15 (`missing_ir_features`) | 0 | 78.4 GB |
| RGB2D `px112` | 1360 | 40 (38 `image_count_mismatch`, 2 `undecodable_images`) | 0 | 58.824 GB |
| RESP 1000 Hz | 1399 | 1 (`missing_resp_volts`) | 0 | 378.6 MB |
| BP 1000 Hz | 1400 | 0 | 0 | 380.2 MB |

## 3. Bit-parity checks

| Cache | Command scope | Result |
|---|---|---|
| BP | `--check 2`, seed 0 | **PASS** — both sessions identical; `F042_T7` is 0 % valid / 0 segments, `M018_T7` is 100 % valid / 1 segment |
| RESP | `--check 2`, seed 0 | **PASS** — both sessions identical; `F042_T7` is 0 % valid, `M026_T6` is 100 % valid |
| RGB2D | `--check 2` | **PASS** — 2/2 sessions identical |
| TIR-ROI | `--check`, `--hops 4.0`, `--input_size 96`, 2 sessions × 2 clips | **PASS** — 4 clips / 2 sessions |

## 4. BP 8 s window yield at a 4 s hop

Measured over all 1400 BP shards:

- 8 s windows: **13,530**
- usable windows: **13,278 (98.14 %)**
- sessions with at least one usable window: **1386 / 1400 (99.00 %)**
- sessions with zero usable windows: **14 (1.00 %)**
- sessions with zero valid segments: **6**
- usable windows per session: min **0**, median **7**, max **50**

The 14 zero-yield sessions are legitimate exclusions (no 8 s window lies
entirely inside one valid BP segment), not cache corruption.

## 5. Real-cache dataset construction

Built `TirRoiResp2DBPPretrainDataset` with:
`subjects='F001'`, `max_entries=2`, `input_size=96`, `clip_duration=8.0`,
`temporal_stride=2`, `tubelet=4,16,16`, `fs=1000`, `sig_kernel=32`.

Result:

- `len(ds) = 2`
- `clips_before_bp_filter = 2`
- `clips_after_bp_filter = 2`
- `clips_dropped_bp_skipped = 0`
- `clips_dropped_bp_unusable = 0`
- sample keys: `bp`, `resp`, `rgb2d`, `tir`
- `bp`: `(1, 8000)` float32
- `resp`: `(1, 8000)` float32
- `rgb2d`: `(3, 100, 96, 96)` float32
- `tir`: `(3, 100, 96, 96)` float32

All four cache roots were found through the dataset’s own
`find_cache`/`verify_identity` paths, so the keys match the run geometry.

## 6. Conclusion

The four artefacts are complete, key-compatible, parity-checked, and consumable
by the four-stream dataset at the requested 96 px / 8 s / 1000 Hz geometry.
Task 8 is green; the remaining gate before a full HPC run is the local and
fold-0 Stage-2 smoke (task 9).
