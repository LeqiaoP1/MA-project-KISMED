# ADR-016: Automatic Mixed Precision (`--amp_dtype`) and DDP Gradient Accumulation

- **Status**: Implemented (Pending Eval)
- **Date**: 2026-10-13 (codified as ADR-016 on 2026-11-26)
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/plan/features/ADR-016-mixed-precision-and-ddp-accumulation-plan.md`, `code/notes/spec/03_engine_and_loss.md` (§7)
- **Supersedes**: `hpc_AMP.md` (pre-implementation audit & proposal promoted into ADR contract)

> **Scope.** This ADR governs one research/engineering aspect: **mixed-precision execution
> (`--amp_dtype`), conditional gradient scaling, FP32 protection for numerically sensitive losses,
> checkpoint precision provenance, and DDP `no_sync` micro-step gradient accumulation across all
> training engines**. Model architectures (`ADR-014`, `ADR-015`) and specific loss formulas
> (`ADR-008`, `ADR-015`) are governed separately.

---

## 1. Context & Problem Statement

### 1.1 Objective and Scope
Training ViT-Base multimodal models across 5 subject-disjoint folds on Lichtenberg HPC (A100 / H100 / RTX 6000 Blackwell GPUs) is compute- and activation-memory-intensive. Enabling Tensor Core mixed precision (`bf16` / `fp16`) and eliminating redundant DDP gradient all-reduces during micro-batch accumulation can substantially improve throughput and batch headroom, provided numerical stability of waveform/spectral losses and bit-identical backward compatibility with FP32 baselines are strictly preserved.

### 1.2 Design Pressures (Pre-Implementation Audit, 2026-10-13)
1. **Loss scaling without autocast.** Before this change, runners unconditionally instantiated `NativeScalerWithGradNormCount` (wrapping `GradScaler`), but zero `torch.autocast` calls existed anywhere in `code/`. Consequently, forward passes ran in FP32 with zero Tensor Core acceleration or memory savings.
2. **Numerical fragility of waveform/spectral losses.** Correlation, variance normalization, rFFT/MR-STFT, and InfoNCE reductions suffer precision loss or overflow in FP16 and must remain in FP32.
3. **Silent checkpoint resume mismatches.** Resuming an FP32 checkpoint under `fp16` or `bf16` (or vice versa) changes optimizer/scaler behavior silently unless `amp_dtype` metadata is persisted and checked.
4. **Redundant DDP all-reduces under gradient accumulation.** When `update_freq > 1`, running non-final micro-batches without `model.no_sync()` triggers unnecessary cross-rank gradient synchronization on every micro-step.

### 1.3 Requirements
- **R1 — Opt-in FP32-identical default.** `--amp_dtype` defaults to `'none'`, reproducing FP32 execution with autocast disabled and `GradScaler` bypassed so all pre-existing runs remain valid FP32 baselines.
- **R2 — Single shared AMP policy across all training engines.** One central resolver (`resolve_amp`, `autocast_context`) governs `pretrain`, `finetune`/`waveform`, and `au_probe`; models are never cast with `.half()`.
- **R3 — Mode-appropriate scaling.** Enable `GradScaler` exclusively for `fp16`; disable scaling for `bf16` and `none` while preserving gradient-norm logging and the `clip_grad > 0` guard (`0.0` = no clipping).
- **R4 — FP32 protection for sensitive losses.** Stage-3 waveform training evaluates `WaveformJointLoss` outside autocast in FP32 (`loss_in_fp32=True`), and Stage-2 composite/InfoNCE terms (`ADR-008`) compute reductions and rFFTs in FP32.
- **R5 — CUDA-only fail-soft & checkpoint provenance.** Requesting `fp16`/`bf16` on CPU degrades gracefully to `none` with a printed log; checkpoints persist `'amp_dtype'` and emit a visible warning on resume mismatch.
- **R6 — DDP `no_sync` micro-step accumulation.** Non-final micro-steps under `update_freq > 1` execute inside `model.no_sync()`.

---

## 2. Decision & Parameter Exposure

**Introduce one central opt-in `--amp_dtype {none, fp16, bf16}` policy (default `none`) shared by all training runners and engines, enabling CUDA `torch.autocast` with FP16-only `GradScaler`, FP32 loss casting for waveform criteria, checkpoint `amp_dtype` metadata verification, and DDP `model.no_sync()` during gradient accumulation.**

### 2.1 Mode Matrix

| `amp_dtype` | CUDA `autocast` | `GradScaler` (`scaler_enabled`) | Primary Role |
|---|---|---|---|
| `none` *(default)* | OFF | OFF (bypassed) | Bit-identical FP32 baseline & CPU fallback |
| `fp16` | `torch.float16` | **ON** | Narrow exponent range; requires dynamic loss scaling |
| `bf16` | `torch.bfloat16` | **OFF** | Preferred HPC mixed precision on Ampere+ GPUs (A100/H100/Blackwell) for waveform/STFT stability |

### 2.2 Engine & Checkpoint Invariants
- **I1 (No `.half()` weight casting):** Model parameters remain FP32 master weights; autocast lowers only supported kernels.
- **I2 (Unified optimizer/clipping path):** `NativeScalerWithGradNormCount.__call__(..., scaler_enabled=...)` handles both scaled (`fp16`) and unscaled (`none`, `bf16`) backward/clipping/step paths, strictly honouring `clip_grad > 0` (`0.0` disables clipping).
- **I3 (FP32 waveform criterion):** `run_waveform.py` passes `loss_in_fp32=True` to `engines/finetune.py`, casting `outputs.float()` and evaluating the criterion outside `autocast_context`. Offline evaluation runners (`run_evaluate*.py`) always run FP32 inference and do not expose `--amp_dtype`.
- **I4 (DDP `no_sync`):** When `update_freq > 1` and `(step + 1) % update_freq != 0`, forward and backward execute inside `model.no_sync()` if available (strict no-op when `update_freq == 1`).
- **I5 (Checkpoint metadata):** `save_model` writes `'amp_dtype'`; `load_model` warns visibly if the checkpoint's recorded `amp_dtype` differs from the active run's mode while allowing training to proceed; legacy checkpoints lacking `'amp_dtype'` resume silently.
- **I6 (CPU fail-soft):** When `not torch.cuda.is_available()` or device is non-CUDA, `resolve_amp` logs a notice and returns the `none` policy.

### 2.3 Logical Parameter Surface

| Parameter | Type | Default | Allowed Values | Role |
|---|---|---|---|---|
| `amp_dtype` | str | `'none'` | `'none' \| 'fp16' \| 'bf16'` | Selects training mixed-precision mode (CLI `--amp_dtype` or YAML `amp_dtype`) |
| `update_freq` | int | `1` | `>= 1` | Gradient accumulation micro-steps; triggers DDP `no_sync` when `> 1` |
| `clip_grad` | float | `0.0` | `>= 0.0` (`0.0` = off) | Max gradient L2 norm across all AMP modes |

### 2.4 Rejected Alternatives
- **Global `model.half()`**: Rejected because BatchNorm/LayerNorm, reductions, and STFT/Pearson losses destabilize in pure FP16.
- **Per-engine bespoke AMP conditionals**: Rejected in favor of one central resolver in `code/utils/native_scaler.py` to prevent engine drift.
- **Defaulting to `bf16` without HPC comparison**: Rejected so existing FP32 baselines remain strictly reproducible until measured on cluster GPUs.

---

## 3. Falsifiable Criteria

- **Criterion 1 (Policy & BDD conformance — CPU suite)**: Automated tests verify all five BDD scenarios: (1) default/`none` disables autocast and bypasses scaler; (2) `fp16` enables `float16` autocast + loss scaling; (3) `bf16` enables `bfloat16` autocast without scaling; (4) checkpoint resume logs a warning on `amp_dtype` mismatch and handles legacy checkpoints cleanly; (5) DDP `update_freq > 1` enters `no_sync()` on non-final micro-steps; plus CPU fail-soft and `loss_in_fp32=True` on `run_waveform.py`.
- **Criterion 2 (Smoke execution)**: All training engines (`pretrain`, `finetune`/`waveform`, `au_probe`) complete bounded smoke steps with finite loss and logged `[amp]` startup banner.
- **Criterion 3 (HPC GPU adoption gate — pending)**: Before claiming AMP speedup or using `bf16`/`fp16` for headline thesis sweeps, execute a matched 1-GPU HPC comparison (`AMP_DTYPE=none` vs. `bf16` vs. `fp16` on `smoke_pretrain.sbatch` / `submit_sweep.sbatch`) and record peak VRAM, step throughput, loss finiteness, and validation metric parity under `code/notes/eval/`.

- **Acceptance Artifacts**: `code/tests/test_amp.py` (18 BDD tests), HPC AMP benchmark note under `code/notes/eval/` (pending GPU run).

---

## 4. Consequences & Trade-offs

- **Positive**: Safe, unified, opt-in mixed precision across every training runner; eliminates wasted DDP communication during gradient accumulation; protects waveform/spectral losses in FP32; preserves exact backward compatibility with FP32 runs.
- **Negative / Technical Debt**:
  - Throughput (`1.2–2×`) and memory (`25–50%`) benefits remain **unmeasured planning estimates** until Criterion 3 is executed on HPC GPUs (A100/H100/Blackwell); `Status` therefore remains `Implemented (Pending Eval)`.
  - `BF16` acceleration requires Ampere-or-newer hardware (satisfied by Lichtenberg A100/H100/RTX 6000 Blackwell nodes).
  - Legacy `NativeScalerWithGradNormCount.state_dict_key = 'amp_scaler'` remains unused (`checkpoint.py` stores `'scaler'`) to avoid breaking existing checkpoints.

---

## 5. Reproducibility Record

- **Run / Job IDs**: Implemented & CPU-verified 2026-10-13; GPU benchmark pending (`AMP_DTYPE={none,bf16,fp16}`).
- **Evidence Artifacts**: As-built record in `code/notes/plan/features/ADR-016-mixed-precision-and-ddp-accumulation-plan.md`.
- **Tests / Smoke Checks**: `code/tests/test_amp.py` (18 tests passing).
- **Rollback / Rejection Condition**: Omit `--amp_dtype` or pass `--amp_dtype none` to execute the pure FP32 path with zero code changes.
