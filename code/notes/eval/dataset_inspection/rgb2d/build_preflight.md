# rgb2d cache — build pre-flight (measured 2026-11-10)

Everything below was run against the production corpus
(`RAW_DATA_PATH=/work/projects/l0003511/test`) on the Lichtenberg **login node**,
writing to `/work/scratch/ne95ocyg/rgb2d_preflight` (since deleted). It replaces
the *estimated* storage/time budget in design §5 and plan Phase 0 with measured
numbers.

## 1. `--init` (key creation)

```
../.env/bin/python runners/run_build_2d_roi_cache.py --init \
    --raw_root "$RAW_DATA_PATH" --out_root <scratch>/rgb2d_preflight
```

```
[rgb2d] WARNING unreadable archive skipped: F069.zip -- 0 bytes (interrupted upload)
[rgb2d] source frame 1040x1392 (h x w = 1392 x 1040), probed 3 session(s)
[rgb2d] key = rgb2droi_face_pad0.10_square_window-hull_clip8s_src1040x1392_native_v1_3f2f30da2d
real 1m7.9s
```

68 s, dominated by `zipfile.namelist()` over 140 zip archives. Each of the 32
array elements pays it once — irrelevant next to the build.

## 2. `--build` (one real subject, 10 tasks)

```
--subjects F001 --array_index 0 --array_count 1
```

| task | box (h×w) | frames | MB |
|---|---|---|---|
| T1 | 795×769 | 1612 | 2956 |
| T2 | 967×871 | 552 | 1395 |
| T3 | 635×648 | 1876 | 2316 |
| T4 | 807×807 | 428 | 836 |
| T5 | 625×621 | 571 | 665 |
| T6 | 670×633 | 1606 | 2043 |
| T7 | 813×797 | 1008 | 1959 |
| T8 | 1036×1036 | 227 | 731 |
| T9 | 821×791 | 2949 | 5745 |
| T10 | 767×758 | 465 | 811 |

* **10/10 built, 0 skipped, 0 errors** in **267.6 s** → 26.8 s/session.
* **13.7 GB of frame data**, 17 GB on disk (`du`, incl. `.lm.dat` + block
  rounding) → **1.37 GB/session**, 1.21 MB/frame average.
* Boxes are **621–1036 px** on a 1392×1040 frame.

### Why the box is that large

`hull_box_2d` is the union over **every stride-1 8 s window of the whole task**
(design decision I2: one box per session so any `clip_stride`/`task_set` works
without re-caching). For T9 that is 2949 frames ≈ 118 s of head motion, so the
stored square must contain the face wherever it went. It is therefore a
*task-wide motion envelope*, not a clip box.

This costs **disk only**: `TirRoiResp2DPretrainDataset._rgb2d_clip` crops the
**clip's own** hull box out of the stored task box (translating by the hull
origin) before the `INTER_AREA` rescale to `input_size`, so the 96 px crop is not
diluted by the task-wide envelope.

## 3. `--check 3` (integrity)

```
VERDICT: PASS  (3/3 sessions identical, 0 failure(s))     # 234 s
```

Byte-identical against a fresh re-derivation (decode + crop + landmark parse).

## 4. Real join: production TIR-ROI cache + this rgb2d cache

With `roi_cache=/work/scratch/ne95ocyg/tir_roi_cache` (the existing
`tirroi_nose_mouth_pad0.10_square_window-hull_clip8s_src726x480_native_v3`
artefact) and `resp_cache=None`:

```
entries: 2
  resp   (1, 800)          torch.float32  min=-1.934 max=1.517
  rgb2d  (3, 200, 96, 96)  torch.float32  min=0.043 max=0.988
  tir    (3, 200, 96, 96)  torch.float32  min=0.000 max=1.000
```

The three-stream contract holds on real data with the two caches that exist on
this cluster. (The TIR cache's `corpus` key is unchanged by the 0-byte skip
below — it counts `IRFeatures` sessions, `n1400`.)

## 5. Full-corpus forecast

| | per session | × 1390 sessions |
|---|---|---|
| frame data | 1.37 GB | **1.91 TB** |
| on disk | ~1.6 GB | **~2.25 TB** |
| wall time (1 core) | 26.8 s | 10.3 core-hours |
| per element of `--array=0-31` | — | **~19 min** (cap is 4 h) |

Recomputed on the FINAL 1400-session corpus (1362 usable, 1 514 433 frames,
stored box side p50 722 px = 36 % of the 1392x1040 frame):

| term | value |
|---|---|
| frames | 1 514 433 (mean 1112/session, max 5150) |
| bytes per frame | 722^2 x 3 = **1.56 MB** (RGB8, uncompressed) |
| **total** | **~2.4 TB** of frames |
| what-if: 96 px store | 42 GB (but `input_size` becomes a key field) |
| what-if: 128 px store | 74 GB (same caveat) |
| what-if: JPEG q90 | ~240 GB (a decode per frame per epoch) |
| what-if: per-clip box | ~1.8 TB (-25 %), and 4-8x WORSE for overlapping strides |

For scale, the raw JPEG corpus is ~4.3 TB, so the cache is ~55 % of it: it keeps
only 36 % of each frame's area, but uncompressed.

Scratch quota is 40 TB with 623 GB used → ~2.25 TB is **6 %**. The design's
~0.4 TB estimate assumed a ~250 px box; the measured task-wide envelope is ~3×
wider, hence ~5.6× the bytes. The `--max_task_gb 12` guard is safe (worst task
seen: 5.7 GB).

## 6. F069 was never missing -- it was being copied in (RETRACTED)

Earlier revisions of this note said F069 had no visible-camera frames and that
its 0-byte archive was "a corpus fact". **That was wrong.** F069 was simply
still being delivered while the build ran:

| time | `2D+3D/F069.zip` |
|---|---|
| 18:0x (first `--init`) | 0 bytes |
| 18:2x (first array running) | ~15 GB, no end-of-central-directory |
| 18:4x | 33.4 -> 34.6 GB growing at **58 MB/s** |
| 18:5x | **41.78 GB, 36 906 members, complete** |

What misled the earlier reading: `2DFeatures/F069_*.mat` (10 fits),
`IRFeatures/F069_*` and `Thermal/F069/` all already existed, so "the tracks are
there and the frames are not" looked like a deliberate corpus property. It was
an ordering artefact of the delivery, nothing more.

Consequences now in the code:

* `unreadable_archives()` reports 0-byte and truncated archives as
  **"not readable yet"** — one transient class, not two kinds of fact. It
  printed `[]` as soon as the copy finished.
* the builder only **reports** it (in `--init`); it records **no** skip marker.
  A permanent marker would outlive the copy and keep excluding a subject whose
  pixels are on disk. The exclusion is run-local: the subject is built on the
  next run, when its archive is complete.
* the dataset's `sessions_2d_missing` counter covers the same situation for any
  session the cache does not yet hold.

Lessons for the next corpus delivery: check
`notes/eval/dataset_inspection/rgb2d/zip_audit.py` (~30 s) and wait for `readable: 140, bad: 0`
before submitting a build. A corpus that changes under a build also changes the
cache KEY, so the work is wasted, not just incomplete.

## 7. Verbatim output (for comparison against the array's logs)

```
$ time python runners/run_build_2d_roi_cache.py --build \
      --raw_root "$RAW_DATA_PATH" --out_root <scratch>/rgb2d_preflight \
      --subjects F001 --array_index 0 --array_count 1
[rgb2d] element 0/1: 10 of 10 session(s) -> <root>/rgb2droi_face_pad0.10_square_window-hull_clip8s_src1040x1392_native_v1_3f2f30da2d
  [1/10] ok   F001_T1 795x769 x1612f = 2956.5 MB
  [2/10] ok   F001_T10 767x758 x465f = 811.0 MB
  [3/10] ok   F001_T2 967x871 x552f = 1394.8 MB
  [4/10] ok   F001_T3 635x648 x1876f = 2315.8 MB
  [5/10] ok   F001_T4 807x807 x428f = 836.2 MB
  [6/10] ok   F001_T5 625x621 x571f = 664.9 MB
  [7/10] ok   F001_T6 670x633 x1606f = 2043.4 MB
  [8/10] ok   F001_T7 813x797 x1008f = 1959.4 MB
  [9/10] ok   F001_T8 1036x1036 x227f = 730.9 MB
  [10/10] ok   F001_T9 821x791 x2949f = 5745.3 MB
[rgb2d] element 0/1 done: 10 built, 0 already there, 0 skipped, 0 errors, 267.6 s
real    5m31.131s

$ time python runners/run_build_2d_roi_cache.py --check 3 ... --subjects F001
VERDICT: PASS  (3/3 sessions identical, 0 failure(s))
real    3m54.409s
```

Expected shape of the real-data join (`--max_entries 2`, TIR-ROI cache + this
cache, `resp_cache=None`):

```
entries: 2
  resp   (1, 800)          torch.float32  min=-1.934 max=1.517
  rgb2d  (3, 200, 96, 96)  torch.float32  min=0.043 max=0.988
  tir    (3, 200, 96, 96)  torch.float32  min=0.000 max=1.000
```

One job submission builds the whole corpus:
`cd code && mkdir -p logs && sbatch scripts/hpc/submit_2d_roi_cache.sbatch`
(32 elements, ~19 min each, `--time=02:00:00`, no GPU).


## 8. Is the task-wide box worth revisiting? **No — measured.**

The stored box is the union over every stride-1 8 s window of the WHOLE task
(design I2), which sounds wasteful: a per-clip box should be much tighter. It is
not, and the size argument for re-cutting the cache does not survive
measurement (1340 sessions, 1.5 M frames, landmarks only — no decode):

| | p50 | mean |
|---|---|---|
| task-wide box side | 720 px | 751 px |
| per-clip (non-overlapping 8 s) box side | 647 px | 660 px |
| **area ratio** | **1.21×** | 1.30× |
| **bytes ratio (area × frames)** | **1.32×** | |

→ a per-clip cache would be **~1.45 TB instead of ~1.9 TB (≈25 % less)**, not the
3–4× a "one box per task" intuition suggests. And that 25 % evaporates as soon
as clips overlap: at the Stage-2 `clip_stride = 2.0 s` an 8 s clip grid
duplicates every frame **4×** (8× at Stage-3's 1.0 s), i.e. 5.8–11.6 TB. Storing
the union once is what makes overlapping strides free.

Why the boxes are so similar: the box side is dominated by **face size, not
motion**. Measured box side vs the window the hull must cover:

| window | box side p50 | relative corpus bytes |
|---|---|---|
| 1 s | 604 px | 1.00 |
| 2 s | 611 px | 1.02 |
| 4 s | 621 px | 1.05 |
| 8 s | 634 px | 1.07 |
| 16 s | 655 px | 1.02 (fewer windows fit) |

A 1 s window already needs a 604 px box on a 1392×1040 source: the 49-point face
spans ~550 px, and 8 s of head motion adds only ~30 px — consistent with the
measured centroid excursion (p50 24 px, `sway_retention.md`). So the ~1.9 TB is
the cost of storing native ~640 px crops of 1.5 M frames, and **no box
granularity changes it**.

Real disk levers, if it ever matters: store a compressed pixel format (≈4–8×,
at the price of a decode on load — defeats the point of the cache), store at a
reduced `store_resolution` (makes `input_size` a key field, losing "one artefact
serves every input size"), or prune the corpus (`--subjects` / `task_groups`).
Box granularity is not one of them.

## 9. Scratch quota check (2026-11-10, before the array)

`df -h /work/scratch/ne95ocyg` reports the **account's project quota**, not the
site filesystem:

| | value |
|---|---|
| quota (`/work/scratch/ne95ocyg`) | **40 TB** |
| used | 623 GB (2 %) |
| free | **~39.4 TB** |
| this cache needs | ~1.9 TB of frames, **~2.25 TB on disk** |
| after the build | **~2.9 TB used ≈ 7 % of quota** |
| inodes | 11 300 / 26.2 M used; the build adds ~4 200 files (3 per session) |

Where the 623 GB sits: `pretrain` 328 GB, `finetune` 202 GB, `tir_roi_cache`
73 GB, `batch_sweep` 7 GB, two smoke dirs 12 GB. Sanity check on the forecast:
the 2D cache is **~31× the TIR ROI cache** (73 GB → 2.25 TB), which is what
~640 px crops at 1392x1040 cost against the thermal hull's ~200 px at 726x480.

Site-level headroom, for context: `df -h /work` = 2.5 PB with 838 TB available
(67 % used). `/work/projects/l0003511` (the read-only source) is 12 TB with
7.5 TB free and is only read.

**Verdict: enough space with ~17× margin.** Monitor after the first element with
`--status` (it prints total bytes) rather than waiting for the whole array.

## 10. `--max_task_gb` is an ARTEFACT budget, not RAM

`CACHE_MAX_TASK_GB` refuses a session whose stored shard would exceed the given
size (`too_large_for_memory: <n> GB > <budget> GB`). It does not size the job:
the builder writes 32-frame chunks into the memmap, so peak RAM is ~50 MB
regardless and `--mem-per-cpu=8G x 2` is unchanged.

The first array (`56160572`, 12 GB budget) refused 3 sessions, e.g.
`F019_T9`: 3366 frames at 1172x1146 = **13.56 GB**. A second array
(`56161371`, `CACHE_MAX_TASK_GB=24`) is chained with
`--dependency=afterok:56160572`; it re-scans (68 s), skips every committed shard
(`0 already there`) and builds only the refused ones. A refused session is
*re-attemptable*: `--build` ignores skip markers when deciding what to work on.

**Bug found while wiring the rebuild:** `commit_shard` left any previous
`skipped/<session>.json` in place, and the loader consults the marker BEFORE the
shard — so a rebuilt session would have stayed excluded. `commit_shard` now
unlinks the stale marker; regression test
`test_commit_clears_a_stale_skip_marker`.

## 11. First production build: outcome, two bugs, cleanup

**Stopped and wiped 2026-11-10 18:2x.** Job `56160572` (32 elements, 12 GB budget)
ended with **29 COMPLETED / 3 FAILED**; the 24 GB follow-up `56161371` was
CANCELLED by `--dependency=afterok`. Peak cache 1.6 TB, deleted.

Failure of elements 16/18/19 (all on one node): a **race in
`roi_cache.write_params`**, not in the build itself --

```
FileNotFoundError: .../params.json.tmp -> .../params.json
```

Three `--init` calls shared one staging name, so one element's `os.replace`
consumed another's temp file. Fixed with `tempfile.mkstemp` in the target
directory (verified: 8 processes x 60 writes, 0 failures, no leftover temps).
`os.replace` makes the publication atomic; only a UNIQUE name makes the write
race-free.

**The corpus is being written while it is read**: during the run `F059.zip`
grew (15.81 -> 15.82 GB, no end-of-central-directory) and `F069.zip` stopped
being a 0-byte placeholder. Both are mid-copy, so `ZipFile` raised
`BadZipFile`. Consequences:

* `discover_rgb_sessions` now skips a subject whose archive cannot be read yet
  and prints `[data] 2D+3D/<S>.zip not readable yet (...)` instead of aborting
  the enumeration of all 138 other subjects.
* `unreadable_archives()` distinguishes `placeholder: 0 bytes` (no source in
  this corpus) from `not a readable archive yet` (still being copied).
* the cache KEY moves with the corpus (`...native_v1_3f2f30da2d` ->
  `..._1f0820c403` for 1380 sessions instead of 1390), so the 1.6 TB built
  during the delivery is orphaned by definition. Wiping it was correct.

**Before the next submit**, confirm the delivery settled (must print `[]`):

```
cd code && ../.env/bin/python -c "import sys; sys.path.insert(0,'.'); \
from data import rgb_roi_dataset as r; \
print(r.unreadable_archives('/work/projects/l0003511/test'))"
```

then `CACHE_MAX_TASK_GB=24 sbatch scripts/hpc/submit_2d_roi_cache.sbatch`
(no `--dependency`: nothing is queued).

## 12. Archive audit of the live corpus (`zip_audit.py`)

`notes/eval/dataset_inspection/rgb2d/zip_audit.py` classifies every `2D+3D/*.zip` (≈30 s: one
`is_zipfile` per file, which reads only the end record). It exists because the
corpus was **delivered while being read**, and `zipfile` reports all three
states as `BadZipFile`.

First scan (18:2x, mid-delivery):

| | |
|---|---|
| archives | 140 |
| readable | **139** |
| bad | **1 — `F069.zip`**, `TRUNCATED / still being copied` (+320 KB during its own 0.2 s check) |

So there was **exactly one** unreadable archive and it was not corrupt: F069 was
being copied in (0 bytes → 15 → 32 → 34.6 → **41.78 GB**, 36 906 members,
58 MB/s, complete ~2 min later). `F059.zip`, which killed the earlier run, had
finished by then and was readable.

Second scan after the copy completed:

```
readable: 140   bad: 0   readable-but-no-jpgs: 0
discover_rgb_sessions -> 1400 sessions, 1362 usable
unreadable_archives() -> []      # corpus is stable, safe to build
```

The session count moved 1390 → 1380 (mid-copy, F059+F069 hidden) → **1400**
(delivery complete), i.e. the cache KEY moves with every scan of a corpus in
flight — another reason to audit first and build once.

## 13. Would a compressed store slow training? Measured: no.

Benchmark on a real shard (`F001_T1`, task box 795x769, clip crop 795x769,
200-frame clip, `input_size=96`). "cold" drops the page cache with
`posix_fadvise(DONTNEED)` before each clip, so it reads from Lustre.

| path | ms / clip | bytes read / clip |
|---|---|---|
| uncompressed mmap + resize (warm page cache) | **240** | 367 MB |
| uncompressed mmap + resize (cold) | **317** | 367 MB |
| JPEG q90 decode + resize (in memory) | **299** | **3.8 MB** |

Read amplification: **367 MB read to produce 5.5 MB of input = 66x**. Compressed
is **97x smaller** and **no slower**.

What the numbers say:

1. **The bottleneck is the resize, not the format.** 200 frames x `INTER_AREA`
   795x769 -> 96x96 costs ~250-270 ms by itself; JPEG decode is ~30 ms of the
   299. Both variants pay the resize, so the format is nearly irrelevant to
   throughput.
2. **I/O is a 25 % penalty, not a wall**: 367 MB in +77 ms = ~4.8 GB/s from
   Lustre with readahead, and only because the clip is 200 *consecutive* frames.
   A random clip would seek instead and lose that.
3. Per-sample CPU is therefore **~300 ms for rgb2d alone** (+~100 ms for the TIR
   crop), i.e. ~400 ms/sample. At `batch=16` that is ~6.4 core-seconds per step:
   **the loader, not the GPU, is the likely limiter** -- see the `num_workers`
   note below.

So the compressed-store decision should be made on **disk** (2.4 TB -> ~25 GB),
never on speed. Its real costs are structural, not performance: it forfeits the
`memmap` invariant (every clip becomes sequential decode rather than a page-fault
window), it needs a decode path plus its own parity tests, and it loses the
"artefact is the pixels" property that makes `--check` a byte comparison.

**The actual speed lever is the 96 px resize.** Storing at `input_size` (or at
the largest `input_size` you will use, 112) makes load-time work a slice instead
of a resample: 112x112x3 = 37 kB/frame -> **56 GB** for the corpus, ~8x faster
per clip -- at the cost of making `input_size` a key field. If a Stage-2 run is
loader-bound, that trade is worth revisiting *before* compressing anything.

## 14. Decision of record: keep the UNCOMPRESSED NATIVE store

Agreed 2026-11-10 after §13. The cache stays as designed:

* `store_resolution = native`, `pixel_format = rgb8`, raw memory-mapped `.dat`;
* one task-wide square hull per session, one artefact for every `input_size`,
  hop and fold;
* no compression: 2.4 TB is 6 % of the quota and compression buys ~0 ms
  (§13: 299 vs 317 ms/clip) while costing the memmap invariant and byte-exact
  `--check`.

Escalation order IF a Stage-2 run turns out loader-bound (~300 ms/clip for
rgb2d + ~100 ms for TIR at `input_size=96`, i.e. ~400 ms/sample, ~6.4 core-s per
step at batch 16) -- cheapest first, and only one at a time:

1. `--num_workers` / `--prefetch_factor` (no cache change, no key change);
2. store at 112 px, the largest `input_size` in use: 37 kB/frame -> ~56 GB,
   ~8x less resize work, at the price of `input_size` becoming a key field;
3. compression (last resort: it changes what the artefact IS).

## 15. FINAL: the build completed (2026-11-10)

Job `56163046` (32 elements, `CACHE_MAX_TASK_GB=24`) on the completed 1400-session
corpus:

```
cached 1360   skipped 40   missing 0   incomplete 0   truncated 0
bytes  2683.089 GB  (2.68 TB)
skip reasons:  38 image_count_mismatch | 2 undecodable_images
all 32 elements logged "done"   |   0 Traceback in any .err
--index   -> manifest: 1360 ok, 40 skipped, 0 missing, 2683.089 GB, 1400 entries
--check 3 -> VERDICT: PASS (3/3 sessions identical, 0 failure(s))  [99 s]
```

`du` 2.5 TiB (same bytes, TiB vs TB), 4080 files, quota now **3.7 / 40 TB (10 %)**.
Wall time ~1 h; no `write_params` race and no archive crash — both fixes held.

**Artefact of record:**

```
RGB2D_CACHE=/work/scratch/ne95ocyg/rgb2d_roi_cache
key        = rgb2droi_face_pad0.10_square_window-hull_clip8s_src1040x1392_native_v1_c22b3c2ef9
```

2.68 TB against the 2.4 TB forecast: the forecast used the median box (722 px)
weighted per session, the real data is a touch heavier per frame. The 40 skipped
are data facts, not failures (`n_jpg != n_mat`; 2 archives containing an
undecodable JPEG).

Remaining known gap: **42 sessions** have no shard (40 skipped + the 2 whose
subject had an undecodable image?). The trimodal dataset counts these as
`sessions_2d_excluded`/`_missing` and excludes them, so Stage 2 runs on 1360 of
1400 sessions -- check `[data] rgb2d:` lines at start-up.
