# Offline TIR-ROI precomputation: why it helps training, and reuse in Stage 3

**Date:** 2026-10-08
**Scope:** the thermal-ROI branch (`data_set: tir_roi` / `tir_roi_resp`,
`configs/pretrain/stage2_hpc_tir_roi_resp.yaml` -> `configs/finetune/resp_tir_roi_hpc.yaml`)
on Lichtenberg.
**Questions answered:**
1. Why would precomputing the 112x112 ROIs offline help the training process?
2. Would those offline results be reusable for Stage 3?
**Status:** BUILT AND VERIFIED. The full-corpus cache exists at
`$WORK_SCRATCH/tir_roi_cache/<KEY>` -- **1385 shards, 91.2 GB, 0 missing** -- and
`--check` on the artifact passes against the live pipeline (section 10.7). Writer,
loader and the fail-loudly gate are implemented (`data/roi_cache.py`,
`runners/run_build_roi_cache.py`, `scripts/hpc/submit_roi_cache.sbatch`, plus a
`--roi_cache` flag on the dataset, `run_pretrain.py` and `run_waveform.py`). What
remains is to ENABLE it in a training run (`ROI_CACHE=<out_root>`). Every number
below is MEASURED unless explicitly flagged `estimate`. Revised 2026-10-08:
sections 0, 1.2, 1.4, 2.3-2.6, 3.2, 3.3, 4.2, 5.1, 6, 7, 8, 9, 10.

---

## 0. Answers in brief

**Why it helps.** The training loop is CPU-work-bound, and the dominant term is
WMV3 decoding. Over one epoch the loader decodes 4.25 M frames = **4.5
worker-hours**; the GPU work it is feeding costs **0.07 core-hours**. The epoch
duration is therefore, to first order, `frames_needed / (workers x decode_rate)`,
and an offline cache deletes that term instead of parallelising it (~450x less
decode CPU across 150 epochs). Secondary but comparable benefits: it removes the
failure mode that actually cost time (a stalled decoder killed a 4-GPU job after
7 epochs), it removes the loader-vs-compute contention that makes tuning
zero-sum, and it makes GPU/scaling choices matter again.

**Reuse in Stage 3: yes -- but only for one of the two cache designs.** Stage 3
matches Stage 2 on every ROI contract key, but its `clip_stride` is **1.0 s
against Stage 2's 2.0 s**, and `clip_stride` is deliberately OUTSIDE the
enforced contract. A per-clip cache hard-codes Stage 2's window lattice and
would be missing half of Stage 3's windows; a **per-task, native-resolution**
cache stores pixels and is hop-independent, so one **~92 GB** cache (measured;
section 3.2) serves Stage 2,
Stage 3, both hops, and both the 112 px and 64 px lineages. That single fact
decides the cache design.

---

## 1. Why the offline cache helps the training process

### 1.1 Decode is the clock

Arithmetic for the shipped Stage-2 config (job `55368106`):

| | |
|---|---|
| clips | 21,231 |
| frames decoded per epoch | 21,231 x 200 source frames = **4.25 M** (the 2 s hop over an 8 s window decodes each source frame ~4x) |
| decode rate | **265 fps per core** (section 4.1) |
| decode demand per epoch | 4.25 M / 265 = 16,038 s = **4.5 worker-hours** |
| GPU work per epoch | 0.20 s/step x 331 steps x 4 ranks = **0.07 core-hours** |
| job CPU budget per epoch | 32 cores x 841 s = **7.5 core-hours** |
| epoch, predicted by decoding alone | 4.25 M / (16 workers x 265 fps) = **1002 s** |
| epoch, observed | **841 s** (14.0 min, 2.54 s/it) |

The prediction from decode throughput alone lands within 20 % of the observed
epoch time, with the gap covered by loading/compute overlap. That agreement is
the argument: **the epoch is essentially the time 16 cores need to decode the
frames the epoch requires.** Decoding is ~60 % of the job's entire CPU budget;
its only competitor is the 0.07 core-hours of GPU-driving work.

The remaining loader cost (ROI crop + resize + collate) is ~10 s/epoch/rank and
is **not separately measured** -- it cannot be large enough to change the
conclusion, but its exact share is unknown.

### 1.2 What changes with a cache

| | now | with an offline cache |
|---|---|---|
| dominant per-iteration cost | decode 4.25 M frames/epoch | read ~92 GB/epoch (one cache pass) |
| epoch floor | ~841 s (decode-rate-limited) | `max(compute, read)` = 66-132 s compute vs reads of the same order |
| decode CPU over 150 epochs | ~670 core-hours | ~1.6 core-hours (one-off) |

The loop stops being a decoder and becomes a training loop whose next constraint
is the read path. Total decode CPU falls by ~450x.

`estimate`: the ~92 GB/epoch figure assumes each task shard is read once and its
~17 overlapping windows are then served from page cache. The 4x window
redundancy therefore costs no extra I/O ONLY while the shard stays resident --
which is exactly the read-path tuning that open question 2 is about.

### 1.3 Secondary benefits

1. **It removes the failure mode that actually cost time.** Job `55368106` ran 7
   clean epochs and then died: a `/work/projects` read stalled ~266 s
   (`Stream timeout triggered after 265852 ms`, 5 workers), OpenCV has no default
   timeout, the decord fallback was missing from `.env`, ranks 1/2 exited, ranks
   0/3 hung in `ALLREDUCE`, and the 600 s NCCL watchdog aborted the job -- 1h40m
   and all but `checkpoint-0000.pth` lost (`save_ckpt_freq: 20`). With no `.wmv`
   in the training loop there is no decoder to stall, no mid-run decord
   dependency, and no way for one slow container to take down four ranks.
2. **It removes an interference term.** Capping compute threads to 1 moved time
   from the loader to the body (data 1.274 -> 0.650 s, body 1.152 -> 1.718 s)
   and left the total at 2.45 s: loader and training process compete for the same
   cores, so reallocating threads is zero-sum. Removing decode removes the
   competition, making iteration time predictable and tunable.
3. **It makes hardware and scaling choices matter.** While decode dominates,
   GPU choice is nearly irrelevant (a single Blackwell step is 0.20 s of a 2.45 s
   iteration). With decode gone, GPU type and rank count start to pay.
4. **It makes geometry changes cheap.** Regeneration costs tens of minutes of
   idle CPU capacity, which matters for a project carrying two ROI lineages
   (64/112 px) and seven tir_roi configs.

### 1.4 What it does NOT fix

- **No reduction in optimizer work.** The `estimate` of 5-15 h for 150 epochs
  assumes the compute floor (66-132 s/epoch) and comparable reads; it is an
  estimate, not a measurement.
- **The read path becomes the next constraint** and needs its own tuning (shard
  size, prefetch, worker count). Reads and compute are of the same order, so this
  is a real limit, not a formality.
- **The 1 -> 4 rank scaling cost stays unexplained.** 1 rank = 1.895 s/it,
  4 ranks = 2.452 s/it: 4 GPUs deliver ~3.1x, not 4x. So the post-cache floor is
  realistically 0.3-0.6 s/iteration, not a clean 0.20 s.
- **It does not help the inspection tooling** (section 2.6).

---

## 2. Reuse in Stage 3

### 2.1 The two configs, key by key

`run_waveform.py` enforces `ROI_CONTRACT_KEYS = ('roi_landmarks', 'roi_padding',
'input_size')` against the Stage-2 checkpoint. Everything else is
free to differ -- and one thing does:

| key | Stage 2 `stage2_hpc_tir_roi_resp` | Stage 3 `resp_tir_roi_hpc` | match |
|---|---|---|---|
| `input_size` | 112 | 112 | yes (enforced) |
| `roi_padding` | 0.2 | 0.2 | yes (enforced) |
| `roi_landmarks` | `''` (12-point nose+mouth) | `''` | yes (enforced) |
| `clip_duration` | 8.0 | 8.0 | yes |
| `temporal_stride` | 2 | 2 | yes |
| `fps` | 25.0 | 25.0 | yes |
| `fs` | 100.0 | 100 | yes |
| `tubelet` | 2,16,16 | 2,16,16 | yes |
| `task_set` | `''` (all tasks) | `''` (all tasks) | yes |
| **`clip_stride`** | **2.0 s** | **1.0 s** | **NO** |

`clip_stride` is intentionally outside the enforced contract; both configs
document that the hop may legitimately differ between stages. It is nonetheless
the key that decides cache reuse, because it selects *which windows exist*.

### 2.2 What that means for each design

- **(a) per-clip cache** (store exactly the 112x112 windows Stage 2 trains on):
  hard-codes the 2.0 s lattice. Stage 3 needs a window every 1.0 s, so **half of
  its windows are not in the cache** -- not reusable. Matching Stage 3 would mean
  a second cache (~150 GiB at the denser hop) or a full regeneration.
- **(b) per-task, native-resolution cache** (store each subject-task's frames
  cropped to a native-resolution box, resize at load): **reusable**. The pixels
  are hop-independent; only the window selection happens at load time. One cache
  serves the 2.0 s and 1.0 s hops, and -- because it is at native resolution --
  also the 64 px and 112 px lineages, which today are separate decode paths.

### 2.3 Requirement that makes (b) work: a whole-task union box

The cached box must be the **whole-task union** -- min/max of the task's landmark
track plus padding -- not a per-clip box. A clip's box is a
min/max over *that clip's* frames, which depends on the clip and therefore on the
hop; the whole-task box is a superset for any hop. That inflates the cache
(see 3.2) but is what makes it hop-independent.

Stated as the rule the implementation must satisfy: the cached rectangle is the
**smallest fixed rectangle that provably contains every clip box any future
config could ask for**. "Any future config" is what forces a union -- `clip_stride`
(2.0 s vs 1.0 s), `task_set`/split, and the cleaning knobs
(`min_signal_spread`, `rail_touch_v`) all change *which windows exist*, and the
cache has to outlive all of them.

**Valid-frame rule (recommended).** Take the union over the task's **valid**
landmark rows only: drop non-finite rows AND `(0,0)` tracker-sentinel rows. This
is both safe and cheap. Safe, because a clip is dropped entirely if it overlaps
ANY missing or `(0,0)` target landmark (`missing_frame_mask`), so every KEPT
clip's frames are a subset of the valid rows -- containment still holds. Cheap,
because including sentinels inflates the p90 task box from **174 px to 385 px**
(section 8). The rule changes pixels, so it belongs in the cache key (section 9).

**Measured over the full corpus** (1380/1380 scannable subject-tasks; per-task
CSV at `$WORK_SCRATCH/tir_roi_cache_plan/box_per_task.csv`):

| box | median | p10 | p90 |
|---|---|---|---|
| whole-task union (what is cached) | **127.5 px** | 102.5 | **174.0 px** |
| clip box, 2 s hop (Stage 2 input) | 105.8 px | 90.0 | 136.6 |
| clip box, 1 s hop (Stage 3 input) | 105.7 px | 90.1 | 136.7 |

The two clip-box rows being equal to within 0.1 px is the whole design in one
line: the hop changes which windows exist but not the box statistics, so a
hop-independent cache is possible at all.

### 2.4 Two levels of box: storage container vs model input

The union box is **not** a modelling choice and never reaches the encoder. There
are two distinct levels, and keeping them apart is the whole point of (b):

| level | unit | decided when | role |
|---|---|---|---|
| **subject-task** (`F001_T1`, the dataset's `session`) | whole-task union, native px | **precompute** | the *storage rectangle* -- fixes which pixels are written |
| **clip** | that clip's own landmark frames | **load** | the *read cursor* -- the box the encoder sees |

The load path is otherwise unchanged from the live pipeline: `clip_roi_box()`
recomputes the clip box from that clip's landmark frames, and `_roi_patches`
crops it out of the cached task crop and resizes.

Why the two levels cannot be collapsed:

- **A per-clip box cannot be the storage rectangle.** It is defined by the clip,
  and the clip set depends on the hop, the `task_set`/split and the cleaning
  knobs -- everything the cache must be independent of (section 2.3).
- **A per-task box must not be the model input.** Feeding it to the encoder would
  cost real resolution on the input side: median **x1.166** linear, p90
  **x1.481**, with **9.1 %** of subject-tasks losing **>= x1.5** detail. Nothing
  in the shipped design pays that.

This makes `roi_box_impact.py`'s "linear inflation" table a *what-if*
measurement: it prices promoting the storage box to the input box. Design (b)
pays only the storage side (section 3.2), never the resolution side.

### 2.5 Resolution policy: store native, never resize

The cache stores the task's union crop at **native resolution**: raw decoded
pixels, integer-sliced. No scaling, no interpolation, and no colour conversion
beyond the one the decoder already performs. Crop + resize happen **only** at
load, in `_roi_patches`.

This is mandatory, not an optimisation. Three reasons, in order of severity:

1. **Resize does not commute with sub-cropping.** The union box and a clip box
   have different extents, so "resize to 112" means different scale factors --
   median union 127.5 px -> **x0.87** (downscale), median clip 105.8 px ->
   **x1.06** (upscale). Once the union has been resampled the clip's native
   pixels no longer exist, and sub-cropping from it is a *second* resample of a
   different operation.
2. **The interpolation filter would be the wrong one.** `_roi_patches` picks
   `INTER_AREA if input_size <= min(box side) else INTER_LINEAR`. A clip box of
   ~106 px with `input_size` 112 selects `INTER_LINEAR`; the union box at
   ~128 px selects `INTER_AREA`. Baking the resize in would apply the wrong
   filter to every clip below 112 px -- and because the switch sits right at
   112, the error would flip **per task**, not be a constant bias.
3. **It would break the 64 px lineage**, which is half of the reuse argument: a
   second cache, or a lossy 112 -> 64 double resample.

**Cost of going native (honest).** Native is ~1.5x LARGER than baking 112 in:
**~92 GB** against **~60 GB** (mean union area 19,061 px^2 against
112^2 = 12,544). That is the price of exactness plus serving both `input_size`
lineages from one cache, and it is worth paying.

**What the cache therefore contains** is exactly what `_frames()` returns: uint8
**RGB** `[T, H_union, W_union, 3]`, the colour conversion having already happened
inside the video reader. The load path is then a pure *slice + resize* -- no
decode, no conversion -- which is the minimum possible distance from
byte-identical. `H_union`/`W_union` differ per task, so the manifest must record
each task's box and array shape (section 9.3).

### 2.6 Stage 3 needs this at least as much as Stage 2

At the shipped 1.0 s hop the corpus is 41,926 clips (survey job `55348993`), so
Stage 3 decodes 41,926 x 200 = **8.4 M frames per epoch -- twice Stage 2's
4.25 M** -> ~1980 s = **~33 min/epoch** at 16 workers, sustained for its 50
epochs. Stage-3 evaluation reads clips too, so it benefits as well.

Two limits to keep expectations right:

- **Not usable by the inspection tooling.**
  `runners/run_inspect_tir_resp.py` and `run_survey_tir_resp.py` draw the ROI box
  on the **full source frame**, so they need the raw `.wmv`; the ROI cache does
  not serve them.
- **The cache key must fail LOUDLY on mismatch.** A Stage-3 run whose `roi_*` keys
  differ must refuse the cache rather than silently fall back to slightly wrong
  pixels. Silent data drift is the one failure mode that would invalidate every
  downstream comparison while looking healthy.

---

## 3. Design consequences

### 3.1 The (a)-vs-(b) choice is settled by reuse, not size

An earlier draft argued (a) was preferable because it looked much smaller; a
later draft called it a toss-up on size. With Stage-3 reuse as a requirement it
is no longer a toss-up: **(b) is the only design that yields one cache for both
stages**, and the decisive evidence is the config difference in section 2.1, not
a size estimate. If Stage 3 were frozen to the 2.0 s hop, (a) would become viable
again; as the configs stand it is not.

### 3.2 Sizes and cost (measured, not estimated)

Measured over the full corpus (1380/1380 scannable subject-tasks). The cache is
hop-independent by construction, so the Stage-3 hop needs no extra measurement.

| | design (a), per-clip | design (b), per-task native |
|---|---|---|
| content | 21,231 x 100 frames x 112x112x3 | **1,540,664 frames** (sum over the corpus) at the union box |
| size | **74.4 GiB** (per hop) | **~92 GB = 86 GiB** |
| cached box | 112x112 fixed | **median 127.5 px** side (p10 102.5, p90 174.0) |
| serves 1.0 s hop (Stage 3) | no | yes |
| serves 64 px lineage | no | yes |
| bit-exactness | by construction | needs a strict coordinate/parity test (3.3) |
| shard | ~60 MB per subject-task | **~66 MB mean / ~41 MB median**, ~1400 shards |

Design (b) costs only **x1.15** the bytes of (a) despite carrying the superset box,
because (a) re-stores each source frame ~4x for its overlapping windows. Per
FRAME, the union box uses median x1.166 linear = **x1.36** the pixels of that
task's average clip box -- the superset overhead, paid once in disk and read
bandwidth, invisible to the model (section 2.4).

Precompute cost: ~1.54 M frames to decode at 265 fps/core = **~1.6 core-hours**,
plus a ~92 GB write -- roughly 3-5 min wall on `deflt` at 32 workers, against
4.5 worker-hours of decoding *per epoch* if left inline.

Optional variant: store (b) as JPEG rather than raw uint8 -- **~17.5 GB**
(measured, ~19 % of raw) instead of ~92 GB, at the cost of JPEG decode CPU
(~1.5 M small crops/epoch, tens of seconds per epoch). A storage/CPU trade, not a
solution to the bottleneck.

### 3.3 Non-negotiable correctness gate

The cache must reproduce the live pipeline **bit-exactly**. Sharp edges:
`roi_box_from_landmarks` padding/rounding, the interpolation switch in
`_roi_patches` (`INTER_AREA` if `input_size <= box side` else `INTER_LINEAR`),
BGR->RGB ordering, and frame/label truncation
(`F011_T1: thermal video has 519 frames but IRFeatures has 507`). Note also that
(b) only stays exact at **native** resolution (section 2.5).

**The risk that must be measured, not assumed.** The live path **seeks**
(`CV2ClipReader.read_range` sets `CAP_PROP_POS_FRAMES`), while a cache writer
would naturally decode **sequentially**. For an inter-frame codec, seek-based and
sequential decode can disagree at non-keyframe positions, so "same frame index"
does not imply "same uint8". If they do disagree, the finding is that the live
pipeline is not deterministic across window alignments -- it decodes the same
source frame ~4x under different `frame_start` values -- and the cache is then
not reproducing a well-defined target but *defining* one (a single canonical
decode per frame). That would be an improvement, but it must be reported, not
smoothed over with a tolerance.

**MEASURED 2026-10-08: they agree.** `analysis/tir_resp/cache_parity_probe.py`
compared `read_all()` (sequential) against `read_range(start, 200)` (seeking) on
8 subject-tasks spanning 372-1905 frames: **8000/8000 frames byte-identical,
max |d| 0** (section 10). So the live pipeline IS deterministic across window
alignments and the cache reproduces a well-defined target. No design change
needed. The probe did surface one adjacent hazard -- see section 10.2.

**Containment.** The storage box must contain every clip box at every hop. This
follows from monotonicity of the box construction EXCEPT in `_clamp_box`'s
degenerate `min_size` branch, which re-centres a sub-2-px box by 1 px. That
corner must be covered explicitly.

**Cache identity.** Everything frozen must be in the key: corpus fingerprint,
`roi_landmarks`, `roi_padding`, box rule (min/max; no quantile), valid-frame
rule (sentinel-excluded), source frame size, **decoder identity** (OpenCV/FFmpeg
version and backend -- section 2.5), pixel format, and tool version. `clip_stride`,
`input_size`, `fs`, `sig_kernel`, the cleaning knobs and `task_set` must NOT be
in the key: they select windows or resize, both of which happen at load. The
detailed list is section 9.1.

Mitigation must be a **test, not a one-off script**: sample N clips across
several tasks and BOTH hops, and assert (i) containment and (ii) byte-equality of
the cached against the live `_roi_patches` output. Precedent exists --
`check_view_parity` in `data/tir_resp_dataset.py`, and the arithmetic parity
checks in `runners/run_inspect_thermal.py`.

---

## 4. Measured evidence base

### 4.1 Decode rate (single core, login node)

| file | frames | fps | ms/frame |
|---|---|---|---|
| `F040/T9.wmv` (warm) | 2107 | 282.8 | 3.54 |
| `F053/T1.wmv` | 546 | 265.9 | 3.76 |
| `F013/T1.wmv` | 848 | 271.9 | 3.68 |
| `F068/T1.wmv` | 1270 | 256.5 | 3.90 |

- **~265 fps = ~3.8 ms/frame per core** is the figure to plan with.
- **Decode is CPU-bound, not I/O-bound**: warm and cold files decode at
  283 vs 257-272 fps (<10 % apart, so the cost does not depend on cache state),
  while reading the same bytes raw takes 0.06 s against 7.45 s to decode
  (~130x).
- **`decord` is NOT faster** than OpenCV (266.4 vs 284.7 fps on the same file),
  so "just switch the decoder" is not an option. decord remains as the
  robustness fallback for containers OpenCV cannot open.

### 4.2 Corpus arithmetic, and a 10x trap

| | |
|---|---|
| "sessions" | 1260 used / 1400 found / 140 skipped **by SUBJECT-TASK** -- the dataset's `session` key is `F001_T1`, one task video |
| clips | 21,231 at `clip_stride: 2.0` (16.9 per subject-task) |
| corpus thermal video | **1,540,664 frames** measured (sum over all 1380 scannable subject-tasks; the old "~1.5 M from 20 sampled subjects" estimate was within 10 %) |
| how much the clips use | **98.7 % of each video** (measured on 29 subject-tasks; 93-100 % per task) |
| precompute volume | **1.54 M frames = ~0.36x one epoch** |

The trap: summing every `.wmv` in a `Thermal/F0xx/` directory gives a frame
count for a **subject**, and dividing that over the subject-**task** count (1260)
inflates the corpus ~10x. An earlier draft of this file did exactly that and
reported 13.9 M frames / 14.6 core-hours / a 3.3x precompute; the corrected
values are 1.54 M / 1.6 core-hours / 0.36x. The 98.7 % figure also kills the
assumption that a precompute could skip most of each video -- there is no slack.

---

## 5. Storage and execution venue

### 5.1 Where the cache goes

| Candidate | Size / quota | Verdict |
|---|---|---|
| **`$WORK_SCRATCH` = `/work/scratch/ne95ocyg`** | `/work` (lbfs22) **2.5 PB total, 858 TB free** | only sensible home |
| `$HOME` / repo `data/` | **60 GB quota, 47 GB free** -- and the SAME file system as /work, quota only | neither 74 GiB nor 126 GB fits |
| node-local `/tmp` | -- | **proven unusable**: a 1.2 GB checkpoint write died at ~509 MiB |
| `/shared` | 26 TB, 17 TB free | site software area |

Layout, mirroring `$WORK_SCRATCH/pretrain/…` and `$WORK_SCRATCH/tir_resp_survey`:

```
$WORK_SCRATCH/tir_roi_cache/<CACHE_KEY>/
  manifest.json        # geometry + corpus fingerprint + tool version + per-shard status
  tasks/F001_T1.npz    # ~66 MB mean / ~41 MB median per subject-task, ~1400 shards
```

One shard per subject-task holds that task's own union crop, so shard size is
proportional to `frames x H_union x W_union x 3` and varies with both head-motion
range and video length (section 9.3).

Per-subject-task shards rather than ~21 k per-clip files: fewer, larger files
keep metadata traffic down on the parallel file system. Scratch is normally not
backed up and may be purged, which the manifest plus resumable shards mitigate.

### 5.2 Where it is computed

The task is decode + crop + resize + write: **no GPU is involved.**

| Venue | Verdict |
|---|---|
| login node | **no** -- shared; heavy compute belongs in Slurm |
| `acc` / `acc_short` (GPU) | possible but wasteful: it occupies a scarce GPU to do CPU work. Our own 4-GPU job waited **17 h** for a slot |
| **`deflt` (CPU)** | **yes** -- 1-day limit, ~165 nodes x 96 CPUs x 356 GB RAM, **~140 idle**, starts immediately |

Shape: `-p deflt -c 32 --mem-per-cpu=2G`, ideally an **array job sharded by
subject-task** (`--array=0-31`) so it is resumable. It can run while the training
job is still queued, so the cache is ready before the GPUs are.

---

## 6. Corrections to earlier drafts

Kept on the record because the wrong numbers were repeated before being checked,
and because these are the ones most likely to be repeated again.

| Claim | Corrected |
|---|---|
| corpus = 41,926 clips | that is the **1.0 s hop** survey figure; the shipped 2.0 s hop gives **21,231** |
| 8.4 M frames per epoch (Stage 2) | **4.25 M**; 8.4 M is Stage 3's figure at the 1 s hop |
| ~13.9 M corpus frames; precompute = 3.3x an epoch | 10x error: per-SUBJECT frames over per-SUBJECT-TASK sessions. Corrected to **~1.4 M / 0.33x**, then MEASURED at **1.54 M / 0.36x** (row below) |
| corpus ~1.4 M frames (itself a refinement, still a sampled estimate) | **1,540,664** measured (sum over 1380 scannable tasks) |
| cache = 147 GiB; design (b) = 100-520 GB | **(a) 74.4 GiB; (b) ~92 GB** measured (section 3.2) |
| whole-task union box ~150x200 px | **median 127.5 px** side, p10 102.5 / p90 174.0 (measured, sentinel-excluded) |
| precompute = 14.6 core-hours | **~1.6 core-hours** (first corrected to ~1.5 from the sampled frame count) |
| JPEG variant of (b) ~25 GB | **~17.5 GB** (measured) |
| (b) is much larger than (a) because the box is a superset | only **x1.15** in bytes -- (a) re-stores each frame ~4x for its windows |
| the "linear inflation" metric measures the cost of the cache | it prices promoting the STORAGE box to the INPUT box; design (b) pays only the storage side |
| clips use only a fraction of each video | **98.7 %** -- there is no decode slack to skip |
| `gpu:rtx6000` = Turing Quadro RTX 6000 | **RTX PRO 6000 Blackwell** (verify with `nvidia-smi`, never the GRES label) |
| A100 would be faster than rtx6000 | A100 is **2.6x slower per step** (0.52 vs 0.20 s pure step), measured |
| thread oversubscription costs time | refuted -- `OMP_NUM_THREADS=8` is already set by Slurm from `--cpus-per-task`; capping is **zero-sum** |
| switching to `decord` speeds decoding | decord is **slower** (266 vs 285 fps) |
| `time - data` attributes the body cost | not causal when loader and consumer share cores; use total s/it |

Process lessons from the harness work that produced these numbers:
`SBATCH_GRES` does **not** survive `scripts/hpc/submit.sh` (that wrapper passes
`--gres` as a sbatch command-line option, which wins) -- use `GPU_TYPE=... GPUS=N`;
`df -P` and `--output` are mutually exclusive and, under `set -euo pipefail` with
stderr suppressed, kill a job silently; `squeue --wait` returns immediately on
this cluster.

---

## 7. Open questions

1. Does the applied cores/workers change (`--cpus-per-task` 8 -> 16,
   `NUM_WORKERS` 4 -> 8) reach 1.72 s/it, the rate 150 epochs in 24 h requires?
   One bench arm on `acc_short` answers it.
2. What is the warm-cache loader throughput floor? Production's `data` = 0.16 s
   only proves the loader beat a 2.54 s consumer, not by how much.
3. The 1.29x 1 -> 4-rank scaling cost is attributed to CPU contention by
   elimination, but is not decomposed (decode / collate / H2D / DDP).
4. ~~The whole-task union box size (~150x200 px) is an estimate.~~ **CLOSED**
   2026-10-08: measured at **median 127.5 px** side (p10 102.5, p90 174.0) over
   all 1380 scannable subject-tasks, giving a **~92 GB** cache (section 3.2).
5. `/work/scratch` retention policy for a **~92 GB** artifact that must stay
   valid across a multi-week campaign.
6. ~~**Seek vs sequential decode** (section 3.3): do `read_range(frame_start, ...)`
   and a sequential read produce identical uint8 for the same frame index?~~
   **CLOSED 2026-10-08: they agree** -- 8000/8000 frames byte-identical over 8
   subject-tasks (section 10.1). The live pipeline is deterministic across window
   alignments, so the cache reproduces a well-defined target.
7. **Post-cache read floor** (extends question 2): with ~41-66 MB shards, does a
   worker keep its current task shard in page cache across the ~17 windows that
   use it, or does it re-read? This decides whether the ~92 GB/epoch estimate in
   section 1.2 holds or the 4x window redundancy is paid in I/O. Partly answered
   by construction -- the loader holds an LRU over shards -- but the LRU size is
   a guess, not a measurement.
8. **Should the shard be a mmap-able ``.npy`` instead of an ``npz``?** ``npz``
   members are decompressed on access, so the LRU exists to avoid re-inflating
   66 MB per window. A single ``frames.npy`` (plus a separate metadata file)
   could be opened with ``np.load(mmap_mode='r')``, letting the OS page cache do
   the job and reading only the pages a clip actually needs. That would remove
   the read-amplification question entirely -- at the cost of a format bump and a
   two-file shard.

---

## 8. Record: `roi_quantile` REMOVED (2026-10-08)

`roi_quantile` no longer exists. It selected an outlier-robust PERCENTILE box
instead of the min/max of the clip's landmark cloud, was `0.0` (i.e. min/max, the
historical box) in every one of the 10 configs, and was therefore doing nothing
numerically. Removed from `data/tir_resp_dataset.py`, `data/rgb_roi_dataset.py`,
the four runners, the 10 configs, the tests and the docs -- 19 files, and
`ROI_CONTRACT_KEYS` is now `('roi_landmarks', 'roi_padding', 'input_size')`.

**Safety of the removal.** Bit-exact: the box was already computed by the
min/max branch whenever the value was `0.0`, so no pixel of any shipped run
changes. The repo's unknown-config-key guard is a HARD ERROR, so the removal
cannot be half-done -- any config or namespace still carrying the key now fails
loudly (verified).

**What was given up** (measured over all 1380 scannable IRFeatures tracks,
1,175,070 frames, by `analysis/tir_resp/roi_box_impact.py`; tracker `(0,0)`
sentinel frames excluded, since a single failed frame inflates a min/max box):

| q | clip box, linear reduction | area |
|---|---|---|
| 0.05 | x0.835 | -30 % |
| 0.10 | x0.713 | -49 % |
| 0.20 | x0.536 | -71 % |

The leverage is essentially IDENTICAL with tracker failures removed, so the
extent the knob trimmed is driven by genuine head motion, not corruption. And
the population it targeted exists: `corr(box side, jitter) = +0.78`, with the
small-box decile at 86 px / jitter 2.6 against a large-box decile at 153 px /
jitter 11.3. This reproduces the reasoning recorded (until this removal) in the
`DEFAULT_ROI_QUANTILE` comment, dated 2026-09-26: *"the min/max union box grows
with head motion; sessions with a SMALL box + low landmark jitter reconstruct at
Pearson ~0.7 while large/jerky ones sit at ~0.0 (possible because the inflated
crop is mostly STATIC BACKGROUND, so the mean-pooled tokens encode pose rather
than nostril temperature)."*

So the deletion removes the cheapest available mitigation for a defect the
project had already measured. That was a deliberate call, on the grounds that the
knob was unused, unvalidated (no test ever exercised `q > 0`) and part of the
enforced contract. The alternative mitigation, `roi_landmarks: nostrils`
(also unused, and a coarser instrument -- a 2-point box), remains.

**To re-add it:** restore the `quantile` argument and percentile branch in
`roi_box_from_landmarks`, the `roi_quantile` parameter on both dataset classes,
the flag on the four runners, the key in `ROI_CONTRACT_KEYS`, the 10 configs and
the test assertions -- and ADD A TEST for `q > 0` (containment in the min/max
box, monotone shrinking with `q`, bit-exactness at `0.0`), which the previous
implementation never had. The expected magnitudes are in the table above.

**A separate finding from the same measurement, still open:** a min/max box is
fragile to ONE failed frame -- including sentinel rows inflated the p90 task box
from 174 px to 385 px, and created a >= x2 resolution-loss class covering 9.9 %
of subject-tasks against 0.1 % clean. Clip-level boxes are protected because a
clip overlapping ANY `(0,0)`/missing target landmark is DROPPED
(`missing_frame_mask`), but a ONE-BOX-PER-TASK rule would be maximally exposed to
it and must mask sentinel/NaN frames explicitly. That is why the valid-frame rule
in section 2.3 excludes sentinel rows from the cached union box, and why it is
part of the cache key rather than an implementation detail.

**Accuracy note (2026-10-08).** The 1,175,070 frames quoted above are
`median(frames) x n_tasks`, an approximation; the measured SUM over the same 1380
tasks is **1,540,664**, which is the figure used for the cache size in section
3.2 (the earlier value under-counted by 24 %). Separately,
`analysis/tir_resp/roi_box_impact.py` can no longer call the production
percentile path, so it now carries a LOCAL re-implementation of the removed box,
labelled analysis-only in the source. It was validated by reproducing the table
above exactly (q=0.05 -> x0.835, q=0.10 -> x0.713, q=0.20 -> x0.536).

---

## 9. Settled design for implementation (2026-10-08)

Everything needed to write the cache without re-deciding anything. Sections
2.3-2.5 are the rationale; this is the specification.

**Governing principle: the cache is ONE STEP of the pipeline, so it must encode
no downstream decision.** It is a derived artifact whose only job is to answer
"which pixels" faster than a decoder can -- nothing more. Every rule below
follows from that:

- Anything that only **selects windows** (`clip_stride`, `task_set`/split,
  `min_signal_spread`, `rail_touch_v`) or **resizes** (`input_size`) stays out of
  the key and out of the writer (9.2), and the per-clip box is recomputed at load
  exactly as today (9.5).
- The gate is **geometry-only** (9.4): a session is dropped only when there is no
  pixel to crop, never because its label is unusable.
- Because it encodes no decision, it must also be **verifiable and disposable**:
  byte-parity against the live path (9.7, 10), a reason per shard in the manifest
  (9.3), and resumable shards so a purged or partial cache is repairable.
- And it must **fail loudly** rather than degrade (9.2): a cache that can change
  what the model is trained on without saying so is worse than no cache at all.

### 9.1 Frozen into `CACHE_KEY`

| item | value |
|---|---|
| corpus fingerprint | raw-tree root + sorted subject-task list + per-task frame counts |
| `roi_landmarks` | `''` -> the 12-point nose+mouth set (`TARGET_LANDMARKS`) |
| `roi_padding` | `0.2` |
| box rule | min/max of the window's landmark cloud (no quantile) |
| valid-frame rule | non-finite rows dropped AND `(0,0)` sentinels excluded |
| source frame | `726 x 480` |
| pixel format | uint8 RGB, exactly as `_frames()` returns it |
| store resolution | **native** -- no resize, no interpolation |
| decoder | OpenCV/FFmpeg version + backend (`cv2` / `decord`) |
| writer version | cache-writer version string |

### 9.2 Deliberately NOT in the key

`clip_stride` (Stage 2 `2.0` / Stage 3 `1.0`), `input_size` (`112` / `64`), `fs`,
`sig_kernel`, `tubelet`, `fps`, `temporal_stride`, `min_signal_spread`,
`rail_touch_v`, `task_set`, and the train/val splits. Each of these either selects
windows or resizes, and both happen at load -- which is precisely what lets one
cache serve both stages and both lineages.

The key must be checked **fail-loudly** on load: a mismatch refuses the cache
rather than falling back to a slightly wrong pixel set (section 2.6).

### 9.3 Shard and manifest schema

```
$WORK_SCRATCH/tir_roi_cache/<CACHE_KEY>/
  manifest.json
  tasks/F001_T1.npz
```

Per shard:

- `frames` uint8 `[T, H_union, W_union, 3]` -- the union crop, native, RGB
- `box` int32 `[x0, x1, y0, y1]` in source pixels (half-open)
- `session` str, `video` str, `n_frames` int
- optional: the target-landmark track, to skip `IRFeatures` I/O at load. It must
  stay **per-task**; a per-clip landmark view would reintroduce hop dependence.
  The writer SHIPS this (it makes a shard self-verifying: the box can be
  re-derived without `IRFeatures`, which is what `--check` relies on).

`manifest.json`: the 9.1 table, per-shard status (`ok` / `skipped:<reason>`),
per-shard frame count and box, writer version, timestamp.

Shards are written atomically (temp + rename) and the manifest updated after each,
so the run is resumable and a purged scratch can be repaired incrementally.

### 9.4 Write path

**Step 0 -- the GEOMETRY gate, applied before any decode.** The writer excludes a
session only when there is nothing to crop, which makes its exclusion set a
STRICT SUBSET of the dataset's. Measured over the corpus (section 10.3):

| class | sessions | writer |
|---|---|---|
| `missing_ir_features` | 15 | **SKIP** -- no landmark track, so no union box exists |
| `missing_resp_volts` (1) + `resp_too_short` (6) | 7 | **CACHE** -- label-side only; the pixels are fine |
| `all_clips_dropped` (18 with cleaning off, 118 with it ON) | 18 / 118 | **CACHE** -- the box comes from valid landmark ROWS, so a task with no sentinel-free window still caches; it simply yields no clips at load. The rail/spread rules are NOT in the cache key (9.2), so the writer must not apply them |
| sentinel frames inside a task | -- | **CACHE** -- excluded from the BOX only (2.3); the per-clip drop stays at load |
| frame-count mismatch (`min(n_vid, n_ir)`) | -- | **CACHE** -- truncated, not excluded |

Would also be skipped if they occurred, but do NOT in this corpus:
`invalid_ir_features`, `undecodable_video`, `too_short` (all zero).

Expected output: **~1385 shards** -- 1400 discovered minus the 15 with no
IRFeatures -- NOT the 1260 the dataset reports as usable. Caching the label-side
skips costs ~1 % of the disk and is what keeps the key independent of the
cleaning knobs.

Then, per task:

1. `discover_sessions(raw_root)` -> the subject-task list.
2. Parse `IRFeatures`, take the target landmark rows, drop non-finite and `(0,0)`
   rows, min/max + padding + clamp -> the union box. **The landmark set is
   1-INDEXED**: `resolve_roi_landmarks` returns LABELS and the dataset converts
   them with `l - 1` (`BP4DPlusTIRRespDataset.target_idx`). Passing labels
   straight to numpy is off by one and silently selects a different set -- see
   section 10.4, where that produced boxes 1.8x too large.
3. Compute the storage box from the landmarks plus a ONE-FRAME probe (for the
   frame size), allocate the output crop, then decode in **256-frame chunks**
   (`CHUNK_FRAMES`). Chunking is what keeps the peak bounded -- see 10.6. Use the
   frame count `min(container num_frames, n_ir)`, never `CAP_PROP_FRAME_COUNT`
   alone: the container over-reported by 12 frames on `M005_T3` (1905 claimed
   against 1893 decodable, matching its 1893 IRFeatures rows). A container that
   over-reports makes a chunked read RAISE on the short tail, so `read_all`
   remains as a FALLBACK for that case, guarded by `--max_task_gb`.
4. Integer-slice every frame to the union box; write the shard.

### 9.5 Load path

Behaviourally unchanged: `clip_roi_box()` recomputes the clip box from its own
landmark frames, and `_roi_patches` crops that box out of the cached crop and
resizes. Only the source of the frames changes.

Implemented in `data/tir_resp_dataset.py` as an optional
``BP4DPlusTIRRespDataset(roi_cache=<out_root>)`` (threaded through both Stage
views, both builders, the dataset CLI, and `--roi_cache` / `$ROI_CACHE` on
`run_pretrain.py` and `run_waveform.py`):

* `_attach_cache` resolves the directory by KEY and prints it at startup;
* `_clip_frames(entry)` returns `(frames, origin)` -- the stored native crop for
the clip's frame range plus the union-box corner;
* `_roi_patches(..., origin)` subtracts that origin, so the box is still
  computed in SOURCE pixels exactly as today. Reading a clip box out of a native
  union crop is an exact integer translation, and the interpolation choice
  depends only on the box EXTENT, which the translation preserves;
* `_source_hw(entry)` reads the source frame size from `params.json`, so
  `clip_roi_box` needs **no decode at all** -- which is what makes a per-clip box
  affordable at load time;
* an LRU over shards is NOT optional: without it every one of a task's ~17
  overlapping windows would re-decompress the whole 66 MB crop against a 3.2 s
  decode, which is the difference between the cache clearly paying and merely
  breaking even.

Two behaviours are deliberate and load-bearing:

* **A missing shard is an ERROR, never a fallback.** A partially built cache must
  fail loudly rather than silently decode (defeating the purpose) or skip the
  session (silently shrinking the corpus).
* **The cache keeps label-side skips.** A session whose every window is dropped
  by the rail/spread rules still gets a shard; the DATASET drops it at load,
  exactly as today. Observed working on `F001_T8`: present in the cache
  (33 MB) and skipped by the dataset with `all_clips_dropped`. That is the
  9.4 gate doing its job.

NOTE the shard is a compressed ``npz``, so members are DECOMPRESSED on access
rather than memory-mapped (see open question 8).

### 9.6 Two choices to confirm before the run

| choice | recommendation | rationale |
|---|---|---|
| valid-frame rule | **sentinel-excluded** | safe (a kept clip contains no sentinel row) and 174 px against 385 px at p90 (section 2.3) |
| coverage | **every task that passes the geometry gate (~1385)**, NOT the 1260 the dataset reports usable | ~91 GB instead of ~83 GB, and it keeps the cache key independent of the cleaning knobs (section 9.4) |

### 9.7 Test gate

Items 1-3 were written and RUN on 2026-10-08 (`analysis/tir_resp/cache_parity_probe.py`,
full-corpus sweep, section 10); item 4 still has to be written with the loader.

1. **Containment** -- every clip box inside the stored task box, for both hops,
   including the `_clamp_box` degenerate branch. **PASS**: 73,809 clips over
   1361 subject-tasks, 0 outside, plus 2050 synthetic degenerate cases with 0
   failures. Tightest margin 0 px -- the union is TIGHT, so there is no slack
   for an off-by-one in the clamp ordering.
2. **Pixel parity** -- cached against live `_roi_patches` output byte-equal.
   **PASS**: 10/10 clips over 10 sessions, max |d| 0.
3. **Seek parity** -- sequential (cache) against `read_range` (live) decode for
   the same frame index. **PASS**: 8000/8000 frames, max |d| 0.
4. **Key mismatch** -- a deliberately altered `roi_padding` must be REFUSED, not
   warned about. **PASS**: `find_cache` matches on the key fields and raises with
   a field-by-field diff; verified accidentally but exactly -- a run with the
   dataset CLI's default `roi_padding 0.1` against the 0.2 cache aborted with
   `roi_padding: cache=0.2 run=0.1` (section 10.5).

### 9.8 Job shape

`-p deflt -c 32 --mem-per-cpu=2G`, an array job sharded by subject-task, output to
`$WORK_SCRATCH`. ~1.6 core-hours of decode plus ~92 GB of writes, so it can run
while the training job is still queued.

---

## 10. Probe record (2026-10-08)

`analysis/tir_resp/cache_parity_probe.py`. No cache files are needed -- the union
crop is simulated in memory -- so this ran BEFORE the writer exists.

```bash
python analysis/tir_resp/cache_parity_probe.py --raw_root $RAW_DATA_PATH \
    --tasks 0 --pixel_clips 10 --t3_tasks 8 --t3_probes 5 --max_video_frames 2600 \
    --json $WORK_SCRATCH/tir_roi_cache_plan/probe_full.json
```

**VERDICT: PASS** (exit 0). Log and JSON under `$WORK_SCRATCH/tir_roi_cache_plan/`.

| check | result |
|---|---|
| T0 frame size | decoded `480x726x3`; the `726x480` constant MATCHES; production `clip_roi_box` agrees with the landmarks-only computation on 4 clips |
| T0 cleaning cross-check | cleaning OFF builds 24,810 clips (2 s) / 48,999 (1 s) against the shipped 21,231 / 41,926 -- strict supersets, confirming the rail/spread rules only REMOVE clips, so the sweep is conservative |
| T1 containment | **73,809 clips over 1361 subject-tasks, 0 outside**, both hops |
| T1 tightness | tightest margin **0 px** -- the union is exact, not generous |
| T1 synthetic clamp corner | **2050 degenerate cases** of 4000 trials, 0 failures |
| T2 pixel parity | **10/10 clips byte-identical**, max abs d = 0 |
| T3 seek parity | **8 subject-tasks, 8000/8000 frames identical**, max abs d = 0 |

### 10.1 What the probe settled

- **Containment holds corpus-wide.** Because the margin is 0 px this is not a
  slack result -- it is a genuine check of the `floor`/`ceil`/clamp ordering.
- **The `min_size` branch is genuinely covered.** The first version measured the
  CLAMPED extent, which `_clamp_box` always grows to >= 2 px, so it reported
  "0 degenerate cases" and claimed coverage it did not have. Measuring the
  PRE-clamp padded extent shows 51 % of trials reach the branch.
- **Seek and sequential decode agree**, on both a short (372-frame) and a long
  (1905-frame) task. The live pipeline HAS one canonical decode per frame, so the
  cache reproduces the training target rather than defining it (section 3.3).
- **The failure mode that killed job `55368106` has no path in**: no seeking in
  the writer, no decoder in the training loop.

### 10.2 The one hazard the probe surfaced

`M005_T3`: the container reports **1905** frames, `read_all()` decodes **1893**,
and its IRFeatures has **1893** rows. `CAP_PROP_FRAME_COUNT` over-reports, so
anything that caches `num_frames` and later asks for the tail gets a short read
(`read_range` RAISES on that rather than silently returning fewer frames, which
is the right behaviour).

The dataset is already safe because it uses `n_common = min(n_vid, n_ir)`. The
writer must use the **decoded** count and record it in the manifest; the loader
must trust the manifest, not the container. This is the same class as the known
`F011_T1` truncation (519 video frames against 507 IRFeatures rows), and it is
why section 9.4 step 3 is worded as it is.

### 10.3 Exclusion-reason census

`discover_sessions` finds 1400 subject-tasks. The dataset's `skipped` list gives a
per-session reason, and the cleaning knobs move exactly ONE category:

| reason | cleaning off | cleaning ON (0.1 V / 9.9 V) |
|---|---|---|
| `all_clips_dropped` | 18 | **118** |
| `missing_ir_features` | 15 | 15 |
| `resp_too_short` | 6 | 6 |
| `missing_resp_volts` | 1 | 1 |
| `invalid_ir_features` | 0 | 0 |
| `undecodable_video` | 0 | 0 |
| `too_short` | 0 | 0 |
| **usable sessions** | **1360** | **1260** |

(The 20 in section 4.2's "1380/1400 scannable" is the box scan's own looser
filter -- it needed only a parseable track with at least one window, not the
respiration gate.)

Two things this settles:

- Only **15 sessions are geometry-fatal**, so the writer's output is **~1385
  shards**, not the 1260 the dataset reports as usable (section 9.4).
- The cleaning knobs are worth **100 sessions** of coverage (18 -> 118). Those
  tasks have perfectly good pixels and their drop is purely label-side, which is
  exactly why the writer must not apply the knobs: doing so would drag them
  inside the cache key and break the key-independence of section 9.2.

### 10.4 Writer and artifact verification (2026-10-08)

Implemented:

| piece | role |
|---|---|
| `data/roi_cache.py` | cache key, geometry gate, union box, shard/manifest IO |
| `runners/run_build_roi_cache.py` | `--init` / `--build` / `--status` / `--index` / `--check` |
| `scripts/hpc/submit_roi_cache.sbatch` | `deflt` array job, CPU only |

Smoke run over 12 subject-tasks -- built, DELETED after the bug below, and
rebuilt with the fix:

| step | result |
|---|---|
| `--init` | probed `726x480x3` on `F001_T1/T2/T3`; key `...native_v2_eaf18ce7e5` |
| `--build`, 2 array elements | 12/12 cached, 0 skipped, 0.6 min total |
| `--status` | cached 12, skipped 0, missing 0, **0.8 GB** |
| `--index` | 12 ok, 0 skipped, 0 missing |
| `--check` | **24 clips over 4 sessions x 2 hops: PASS** -- cached == live, byte for byte |

**Size confirmed.** 0.8 GB over 12 tasks is 65 MB/task, extrapolating to **~90 GB**
for ~1385 tasks -- inside the ~92 GB of section 3.2, and 2.1x smaller than the
first, buggy build.

**The bug this caught, and why `--check` exists.** The first build produced boxes
**1.8x too large**: `resolve_roi_landmarks` returns **1-indexed LABELS** and the
writer passed them straight to numpy, selecting landmarks 9..13/20..26 instead of
8..12/19..25 (`BP4DPlusTIRRespDataset.target_idx` converts with `l - 1`). A box
from the WRONG set is not guaranteed to contain the right set's clip boxes, so
this was not merely wasteful -- it was a design violation. It was first noticed as
a size disagreement with section 3.2, and the fix (a single `landmark_indices()`
chokepoint that all call sites must pass through) is verified by `--check`.

**A property of the key worth remembering.** `CACHE_KEY` hashes PARAMETERS, not
code. A fix that changes pixels therefore does NOT change the key, and a stale
cache would be silently reused -- which is exactly what would have happened here.
That is why the fix also bumped `CACHE_FORMAT_VERSION` to 2, and why `--check` has
to be PERMANENT rather than a one-off acceptance test: it is the only mechanism
that detects a stale-vs-fixed mismatch.

### 10.5 Loader verification (2026-10-08)

The dataset now takes `roi_cache=<out_root>`; the point is that the EXISTING clip
checks compare against a LIVE decode, so attaching the cache turns them into the
cached-vs-live parity test with no new test code:

```bash
python -u data/tir_resp_dataset.py --raw_root $RAW_DATA_PATH --subject F001 \
    --roi_padding 0.2 --roi_cache $WORK_SCRATCH/tir_roi_cache_smoke --check_views
```

```
[data] roi_cache: .../tirroi_nose_mouth_pad0.20_minmax_src726x480_native_v2_9d0214aefa
[data]   src 726x480, landmarks nose_mouth (12 pts), padding 0.2, format v2
sessions : 9 used / 10 found / 1 skipped
clips    : 52
roi_cache: .../tirroi_nose_mouth_pad0.20_minmax_src726x480_native_v2_9d0214aefa
[ok] #0  F001_T1  frames 0..200:  tir [3,200,64,64], resp [8000]
[ok] #51 F001_T10 frames 200..400: tir [3,200,64,64], resp [8000]
[ok] Stage-2 and Stage-3 views return the SAME ROI crop, the SAME box and the
     matching target
PASS: 2/2 clip check(s) passed, 52 clip(s) in the dataset
```

`F001_T8` appears in BOTH states, which is the 9.4 gate working in practice: it is
in the cache (33 MB) and skipped by the dataset (`all_clips_dropped`) -- pixels
kept, label decision left where it belongs.

**Two integration traps found and fixed:**

1. **The internal drop-rule probes in the dataset's `main()` inherited the
   cache.** They run on a synthetic 1-session temp tree and on partial subject
   selections, so their corpus can never match a real cache -- and they test
   RULES, never a pixel. They now pass `roi_cache=None`.
2. **The CLI default padding is 0.1, every config ships 0.2.** The gate caught
   this immediately rather than reading the wrong pixels, which is the whole
   argument for making the key the verification (and a reminder that the
   `DEFAULT_ROI_PADDING = 0.1` module default is not what any shipped run uses).

### 10.6 First full build: 8 elements OOM-killed, and what that exposed

Job `55385777` (32-way, `--mem-per-cpu=4G`) processed 1211 of 1400 tasks and then
**8 of the 32 elements were killed with `OUT_OF_MEMORY`** (exit 0:125). Recovery
was cheap because the build is RESUMABLE: job `55385927` re-ran the same array and
only redid what was missing, keeping the shards already on disk.

**Root cause 1 -- the memory model was wrong by ~12x.** `read_all()` builds a
Python LIST of every frame and then `np.stack`s it, so its peak is ~2x the task's
pixels, not 1x. The corpus' longest task is **5149 frames** (not the ~851 the
sizing assumed) = 5.4 GB of pixels -> **MEASURED 11.06 GB RSS**. Against an 8 GB
request that is an OOM every time, and the killed elements were exactly the long
ones.

| | before | after |
|---|---|---|
| decode | `read_all()` (list + stack) | 256-frame chunks into a preallocated crop |
| peak RSS, F074 (5149 frames) | **11.06 GB** | **1.32 GB** |
| peak vs task length | linear | ~constant (one chunk + the crop) |
| pixels | -- | **byte-identical, 10/10 shards** |
| wall clock, F074 (10 tasks) | 76 s | 78 s |

The fix is deliberately NOT "request more memory" -- that only moves the cliff.
The box is a pure function of the landmark track and the frame size, so it can be
computed BEFORE any pixel is read; the output crop is then allocated once and
filled chunk by chunk. The chunks use `read_range(start, k)`, i.e. one seek per
chunk: the same access pattern as the live dataset's per-clip reads, which
section 10.1 measured byte-identical to a sequential decode. `read_all` survives
only as the FALLBACK for a container that over-reports its frame count (which
makes `read_range` raise on the short tail, 10.2).

**Root cause 2 -- a shared staging filename.** `write_shard` staged to
`<session>.npz` in `.tmp/`, so two processes writing the SAME task would truncate
each other's partial file and one `os.replace` would then fail. No two array
elements share a task, so it did not bite here -- but the staging name now
carries the PID.

**A filesystem lesson worth keeping -- and a self-correction.** After the OOM
kills, `ls | wc -l` and `glob` reported **1200** shard names while only **361**
could be `stat`ed, and `du` printed "No such file or directory" for the rest. I
first read that as a permanent inconsistency; it is NOT. It is a **propagation
lag**: on `/work` the new directory entries become visible before their inodes
resolve, and the two views agree once the metadata settles. After the second
array had finished, the SAME directory reported **1385 dirents / 1385
stat-able / 91.2 GB** -- so the resubmit had worked all along, and my "unchanged
at 361" reading was simply taken seconds too early.

The operational rule that survives: `--status` is an EVENTUALLY CONSISTENT view,
so a build must not be judged from a single reading taken immediately after the
array exits -- and `os.stat` (what `rc.scan` uses) is the measurement, not `ls`.

(Also corrected here: an earlier reading of `sinfo` piped through `head` made
`deflt` look ~94 % drained. It actually had **159 idle nodes / 28,719 idle
CPUs**. Do not truncate `sinfo`.)

### 10.7 Final artifact (2026-10-08)

```
$WORK_SCRATCH/tir_roi_cache/tirroi_nose_mouth_pad0.20_minmax_src726x480_native_v2_9d0214aefa/
  params.json      manifest.json      tasks/ (1385 shards)      skipped/ (15)
```

| | |
|---|---|
| shards | **1385** |
| skipped | **15**, all `missing_ir_features` (geometry-fatal) |
| missing | **0** |
| size | **91.2 GB (84.9 GiB)** |
| predicted in section 3.2 | ~1385 shards, ~92 GB |

`--check` ran against the ARTIFACT rather than an in-memory simulation: cached vs
live `_roi_patches` for **35 clips over 6 sessions x both hops -> PASS**, and the
shards were byte-identical to the `read_all` path in the 10-shard A/B of section
10.6.

Two harness lessons, both fixed:

* the filesystem needs a moment after a job exits before `os.stat` resolves new
  entries, so a build must not be judged from one `--status` taken immediately
  after the array returns (10.6);
* `mode_check` kept its pre-refactor two-argument signature while the other modes
  moved to `(args, corpus, sessions)`. It only surfaced when the FINAL
  verification ran -- a reminder that the verification step also tests the
  harness around it.
