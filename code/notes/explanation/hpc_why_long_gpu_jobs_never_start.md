# Why does a long GPU job never start here, and how does a 30-minute chain get around it?

*Plain-English explanation, written 2026-11-22, of the Lichtenberg (Slurm 25.05)
behaviour that cost this project 13.5 hours of a pending job and produced the
`scripts/hpc/submit_pretrain_chain.sh` wrapper. The operational scripts are the
normative artefact (`scripts/hpc/*.sbatch`, `scripts/hpc/*.sh`); measured run
outcomes belong in `notes/eval/`, not here. Nothing in this document changes what
the code does — it explains **why** the code is shaped the way it is.*

---

## 1. The short version

A multi-hour GPU job on this cluster waits behind 30-minute jobs **forever**, for
two independent reasons that both come from the partition, not from the job:

1. `acc_short` carries `PriorityJobFactor=100`; `acc` and `acc_long` carry `10`.
   Same `PriorityTier`, so the only ranking difference is a 10× multiplier.
2. A job that asks for four GPUs needs a node to be **completely free of GPU
   work** for its whole walltime, and the four-GPU nodes are continuously packed
   with 2-GPU jobs that backfill has already committed.

The fix is not to wait harder. It is to submit a **chain of 30-minute
allocations** in the high-factor partition, each resuming from the previous
link's checkpoint, with `--dependency=afterany` doing the hand-off.

---

## 2. Priority is a property of the partition, not of the walltime

```
acc        MaxTime=1-00:00:00  PriorityTier=100  PriorityJobFactor=10
acc_long   MaxTime=7-00:00:00  PriorityTier=100  PriorityJobFactor=10
acc_short  MaxTime=00:30:00    PriorityTier=100  PriorityJobFactor=100
special_computetugether  MaxTime=7-00:00:00  PriorityTier=100  PriorityJobFactor=100
special_turin            MaxTime=7-00:00:00  PriorityTier=100  PriorityJobFactor=100
long / deflt             MaxTime=7-00:00:00  PriorityTier=100  PriorityJobFactor=10
```

`PriorityTier` is equal everywhere, so it separates nothing. `PriorityJobFactor`
multiplies the job's computed multifactor priority: an identical request weighs
**10×** more in `acc_short` than in `acc`. Observed on our own jobs the same day:

| Job | Partition | Reported `Priority` |
|---|---|---|
| 56951161 (2 h, 2×a100) | `acc` | 166,975 |
| 56957082 (30 min, 4×a100) | `acc_short` | 1,066,975 |

Not exactly 10× — age and the other multifactor terms still contribute and the
two jobs were submitted at different times — but the direction and the order of
magnitude are the factor.

**The 30 minutes is not the privilege.** What makes short jobs privileged here is
the LUA submit plugin, which maps the time slot onto the partition
(`TIME_SLOT is 'short'` → `acc_short`). Priority comes from the partition;
*placeability* comes from the short walltime. On this cluster the two arrive
together, which makes "use the 30-minute queue" the practical rule.

---

## 3. The submit plugin changes more than the partition

`[I] LUA job_submit Plugin V1.1.0004` intercepts every `sbatch` and rewrites or
rejects the job before `slurmctld` sees it. Four behaviours have bitten us:

| Behaviour | Symptom | Consequence |
|---|---|---|
| `--mem-per-cpu` is mandatory | `[E] Parameter --mem-per-cpu missing`, then the useless `sbatch: error: Batch job submission failed: Unspecified error` | every job script must set it; `--mem` alone is rejected |
| time slot → partition | `[I] Partition changed from acc_long to acc` | a 5-minute request cannot stay in a 7-day partition |
| special partitions are account-gated | `'special_computetugether' is a special Partition` … `Invalid account or account/partition combination` | the rtx6000/7-day partition is simply closed to account `l0003511` |
| the decision is recorded | `scontrol show job` → `AdminComment=[I] TIME_SLOT is long …` | you can always read back *what* it decided and why |

The plugin's rewrites are visible after the fact, so a confusing state
("why is my `acc_long` job in `acc`?") is answerable from the job record rather
than by guessing.

---

## 4. A 4-GPU request is a whole-node request

The accelerator nodes are not uniformly sized:

| Node family | GPUs | Notes |
|---|---|---|
| `gaqc*` | `a100:4` | 4-GPU nodes, 96 CPUs, 356 GB |
| `gaoc*` | `a100:8` | mostly DRAINED (`SlurmdSpoolDir is full`) |
| `ghqd*` | `h100:4` | 4-GPU nodes, 1.5 TB |
| `gaod*` | `mi300x:8` | AMD/ROCm — unusable by the CUDA-only build |
| `gboe*` | `rtx6000:8` | 5 of 8 DOWN (`Reboot ASAP … reboot timed out`) |
| `gvqc*` | `v100:4` | sm_70 — not in this torch build's arch list |

So on a 4-GPU node, a 4-GPU request is satisfied only when *all four* cards are
free. Meanwhile the scheduler packs 2-GPU jobs onto those same nodes — including
our own — and backfill has already promised future slots to them
(`State=MIXED+PLANNED`). That is the mechanism behind:

- the 2-GPU request that started in **10–20 s**, three separate times;
- the identical 4-GPU request that showed `PENDING (None)` with a
  `squeue --start` estimate of **09:46** — and then started at **09:23** when a
  node emptied. Backfill estimates are a lower bound on optimism, not a promise.

`Reason=None` is not a diagnosable reason. On this cluster it appears for jobs
that simply have no placeable resources; the informative step is to read the
candidate nodes' `AllocTRES` and compare GPU totals against GPU allocations.

---

## 5. How a dependency chain actually works

`--dependency=afterany:<jobid>` is stored by `slurmctld`, not polled by the
scripts:

```
JobState=PENDING  Reason=Dependency
Dependency=afterany:56957081(unfulfilled)
```

- The dependency list lives in the controller's job record (persisted under
  `StateSaveLocation=/ha/slurm/ctld`).
- A job with `Reason=Dependency` is **excluded from the scheduling candidate set
  before resources are considered** (`SchedulerTimeSlice=30 s` cycles), so
  backfill cannot start it early even when GPUs are idle.
- When the predecessor reaches *any* terminal state — `COMPLETED`, `FAILED`,
  `TIMEOUT`, `CANCELLED` — the controller clears the satisfied entry and the
  successor becomes an ordinary pending job.
- `scancel` does **not** propagate; it merely satisfies the successors'
  dependencies, so they run (and exit immediately if nothing new was
  checkpointed).

The choice of `afterany` over `afterok` is the whole trick:

| Type | Predecessor must reach | A walltime kill (TIMEOUT) |
|---|---|---|
| `afterok` | exit code 0 | **blocks the chain forever** |
| `afterany` | any terminal state | hands off immediately |

A 30-minute link is *expected* to die of its walltime, so `afterok` would
dead-end the chain at link 1 every single time.

---

## 6. What a chain requires from the training code

Three properties, all already true in `runners/run_pretrain.py` plus one wrapper
guarantee:

1. **Resume.** `auto_resume_model` reads
   `<output_dir>/checkpoints/latest_checkpoint.txt` and sets
   `start_epoch = checkpoint['epoch'] + 1`, restoring model, optimizer and
   scaler. A walltime kill therefore costs at most the epochs since the last
   checkpoint.
2. **A stable schedule.** The LR schedule is rebuilt from `--epochs`, so every
   link must pass the *same* `--epochs`. Changing it mid-chain silently moves the
   schedule position.
3. **A checkpoint per link.** A link that writes no checkpoint advances nothing,
   so `submit_pretrain_chain.sh` appends `--save_ckpt_freq` (default 5) unless the
   caller supplies one. `save_model` has no retention policy — files accumulate
   (`checkpoint-NNNN.pth`, ~366 MB each for ViT-Small four-stream).

The measured budget on this corpus (fold 0, 6648 clips, global batch 32):

| Width | Per-GPU batch | Steps/epoch | Epoch time | Epochs per 30-min link | 150 epochs |
|---|---|---|---|---|---|
| 2×a100 | 16 | 208 | 108 s (0.523 s/it) | ~15 | 4.5 h compute, ~10 links |
| 4×a100 | 8 | 208 | 66 s (0.322 s/it) | ~24 | 2.75 h compute, ~7 links |

Both rows keep the **global** batch at 32, which is what keeps the derived LR
(`blr · batch · world / 256 = 1.875e-05`) identical. Halving the world size
without doubling the per-GPU batch would halve the LR and double the optimizer
steps — a different optimisation regime, not just a slower one.

---

## 7. Diagnostics worth reaching for

| Question | Command |
|---|---|
| Is the dependency what's blocking it? | `squeue -o "%.10i %.12T %.26E"` |
| What did the job actually request? | `scontrol show job <id> \| grep -E 'ReqTRES\|Dependency\|AdminComment'` |
| How much of each node is free? | `scontrol show node <n> \| grep -E 'State\|CfgTRES\|AllocTRES'` |
| When does backfill think it will start? | `squeue --start -j <id>` (treat as optimistic) |
| Is the partition high- or low-factor? | `scontrol show partition <p> \| grep -E 'Priority\|MaxTime'` |
| Did it die of the walltime? | `sacct -j <id> -o JobID,State,Elapsed,ExitCode` |

---

## 8. What this document does not claim

- The priority factors and node inventories are a **snapshot** (2026-11-22).
  Clusters get reconfigured; re-read them with `scontrol` before relying on them.
- The throughput table is one corpus, one geometry (8 s clip, 96 px, four
  streams, ViT-Small). It is not a general benchmark, and per-run outcomes belong
  in `notes/eval/`.
- Nothing here justifies changing the *science* (epoch budget, global batch,
  masking policy). The chain changes **where and how long** a job runs, not what
  it computes.
