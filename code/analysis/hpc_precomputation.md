# Offline TIR-ROI precomputation: why it helps training, and reuse in Stage 3

**Date:** 2026-10-08
**Scope:** the thermal-ROI branch (`data_set: tir_roi` / `tir_roi_resp`,
`configs/pretrain/stage2_hpc_tir_roi_resp.yaml` -> `configs/finetune/resp_tir_roi_hpc.yaml`)
on Lichtenberg.
**Questions answered:**
1. Why would precomputing the 112x112 ROIs offline help the training process?
2. Would those offline results be reusable for Stage 3?
**Status:** the precomputation is **NOT implemented**. Every number below is
MEASURED unless explicitly flagged `estimate`.

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
cache stores pixels and is hop-independent, so one ~126 GB cache serves Stage 2,
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
| dominant per-iteration cost | decode 4.25 M frames/epoch | read 60-126 GB/epoch |
| epoch floor | ~841 s (decode-rate-limited) | `max(compute, read)` = 66-132 s compute vs reads of the same order |
| decode CPU over 150 epochs | ~670 core-hours | ~1.5 core-hours (one-off) |

The loop stops being a decoder and becomes a training loop whose next constraint
is the read path. Total decode CPU falls by ~450x.

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
- **It does not help the inspection tooling** (section 2.4).

---

## 2. Reuse in Stage 3

### 2.1 The two configs, key by key

`run_waveform.py` enforces `ROI_CONTRACT_KEYS = ('roi_landmarks', 'roi_padding',
'roi_quantile', 'input_size')` against the Stage-2 checkpoint. Everything else is
free to differ -- and one thing does:

| key | Stage 2 `stage2_hpc_tir_roi_resp` | Stage 3 `resp_tir_roi_hpc` | match |
|---|---|---|---|
| `input_size` | 112 | 112 | yes (enforced) |
| `roi_padding` | 0.2 | 0.2 | yes (enforced) |
| `roi_landmarks` | `''` (12-point nose+mouth) | `''` | yes (enforced) |
| `roi_quantile` | 0.0 | 0.0 | yes (enforced) |
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

### 2.3 Requirement that makes (b) work

The cached box must be the **whole-task union** -- min/max of the task's landmark
track plus padding -- not a per-clip box. A clip's box is a `roi_quantile: 0.0`
min/max over *that clip's* frames, which depends on the clip and therefore on the
hop; the whole-task box is a superset for any hop. That inflates the cache
(see 3.2) but is what makes it hop-independent.

### 2.4 Stage 3 needs this at least as much as Stage 2

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

### 3.2 Sizes and cost

| | design (a), per-clip | design (b), per-task native |
|---|---|---|
| content | 21,231 x 100 frames x 112x112x3 | ~1.4 M frames at a whole-task box |
| size | **74.4 GiB** (per hop) | **~126 GB** (`estimate`, box ~150x200 px) |
| serves 1.0 s hop (Stage 3) | no | yes |
| serves 64 px lineage | no | yes |
| bit-exactness | by construction | needs a strict coordinate/parity test |
| shard | ~60 MB per subject-task, ~1260 shards | ~99 MB per subject-task, ~1260 shards |

Precompute cost is the same either way: ~1.4 M frames to decode = **~1.5
core-hours** (~5-15 min wall on `deflt` at 32 workers), versus 4.5 worker-hours
of decoding *per epoch* if left inline.

Optional variant: store (b) as JPEG rather than raw uint8 -- ~25 GB instead of
~126 GB, at the cost of JPEG decode CPU (~1.4 M small crops/epoch, tens of
seconds per epoch). A storage/CPU trade, not a solution to the bottleneck.

### 3.3 Non-negotiable correctness gate

The cache must reproduce the live pipeline **bit-exactly**. Sharp edges:
`roi_box_from_landmarks` quantile/padding/rounding, the interpolation switch in
`_roi_patches` (`INTER_AREA` if `input_size <= box side` else `INTER_LINEAR`),
BGR->RGB ordering, and frame/label truncation
(`F011_T1: thermal video has 519 frames but IRFeatures has 507`). Also note that
(b) only stays exact at **native** resolution: caching the union box already
resized to 112x112 and then cropping a smaller sub-box from it does NOT equal
resizing the original crop.

Mitigation must be a **test, not a one-off script**: sample N clips and assert
byte-equality between the cached and live paths. Precedent exists --
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
| corpus thermal video | ~1.5 M frames (20 subjects sampled, ~11,002 frames per subject over ~10 task videos, x 1400 subject-tasks) |
| how much the clips use | **98.7 % of each video** (measured on 29 subject-tasks; 93-100 % per task) |
| precompute volume | ~1.4 M frames = **0.33x one epoch** |

The trap: summing every `.wmv` in a `Thermal/F0xx/` directory gives a frame
count for a **subject**, and dividing that over the subject-**task** count (1260)
inflates the corpus ~10x. An earlier draft of this file did exactly that and
reported 13.9 M frames / 14.6 core-hours / a 3.3x precompute; the corrected
values are 1.4 M / 1.5 core-hours / 0.33x. The 98.7 % figure also kills the
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
  tasks/F001_T1.npz    # ~99 MB (design b) / ~60 MB (design a), ~1260 shards
```

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
| ~13.9 M corpus frames; precompute = 3.3x an epoch | **~1.4 M; 0.33x** (10x extrapolation error: per-SUBJECT frames over per-SUBJECT-TASK sessions) |
| precompute = 14.6 core-hours | **~1.5 core-hours** |
| cache = 147 GiB; design (b) = 100-520 GB | **(a) 74.4 GiB; (b) ~126 GB** |
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
4. The whole-task union box size (~150x200 px) is an **estimate**; measuring it
   over a sample of subject-tasks would pin down the (b) cache size.
5. `/work/scratch` retention policy for a ~126 GB artifact that must stay valid
   across a multi-week campaign.
