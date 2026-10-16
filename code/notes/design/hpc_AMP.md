# AMP Action Proposal and HPC Adoption Plan

**Repository review:** 2026-10-13  
**Scope:** `code/` training runners, engines, checkpointing, and HPC launch scripts

**Implementation status (2026-10-13): IMPLEMENTED, opt-in.** The
`--amp_dtype {none,fp16,bf16}` switch, the shared autocast context, the optional
loss scaler, DDP `no_sync` accumulation, checkpoint `amp_dtype` metadata and the
BDD tests (`code/tests/test_amp.py`) are all in place. It is CUDA-only and the
default is `none` (autocast off, scaler bypassed), so every pre-existing FP32 run
is unchanged. The diagnosis below is the pre-implementation audit that motivated
the change. **As-built details: [`hpc_AMP_implementation.md`](hpc_AMP_implementation.md).**

## Executive summary

Automatic mixed precision is **not currently active as a complete AMP pipeline**.
The repository has a gradient-scaling helper, but no `autocast` context and no
runtime precision flag. Consequently, model forward passes, loss computation,
and most backward computations execute in the model's default (normally FP32)
dtype. The existing scaler only scales the loss before backward and unscales
gradients before the optimizer step.

This is best described as **loss scaling without mixed-precision execution**,
not as usable FP16/BF16 AMP.

## Evidence from the current code

| Area | Current behavior |
|---|---|
| [`utils/native_scaler.py`](../utils/native_scaler.py) | `NativeScalerWithGradNormCount` wraps `torch.amp.GradScaler('cuda')`, with a legacy fallback for older PyTorch. It scales the loss, performs backward, unscales gradients, clips or measures the gradient norm, and steps the optimizer. |
| [`engines/pretrain.py`](../engines/pretrain.py) | Uses the scaler when supplied, otherwise ordinary `loss.backward()` / `optimizer.step()`. There is no autocast around `model(x)`. |
| [`engines/finetune.py`](../engines/finetune.py) | Same optional scaler branch; `outputs = model(samples)` and criterion evaluation are full precision unless a caller changes global/model behavior. |
| [`engines/au_probe.py`](../engines/au_probe.py) | Same optional scaler interface. |
| [`runners/run_pretrain.py`](../runners/run_pretrain.py), [`runners/run_finetune.py`](../runners/run_finetune.py), [`runners/run_waveform.py`](../runners/run_waveform.py), [`runners/run_au_probe.py`](../runners/run_au_probe.py) | Instantiate `NativeScalerWithGradNormCount` unconditionally; there is no `--enable_amp`, `--amp_dtype`, or equivalent switch. |
| [`utils/checkpoint.py`](../utils/checkpoint.py) | Saves and restores scaler state under the `scaler` checkpoint key, so the current loss-scaling state is resumable. |
| [`scripts/hpc/smoke_pretrain.sbatch`](../scripts/hpc/smoke_pretrain.sbatch) | The smoke-test comment calls out the scaler path, but this validates loss scaling and checkpointing, not autocast-based mixed precision. |

There are no `torch.autocast`, `torch.cuda.amp.autocast`, or
`torch.amp.autocast` calls in `code/`. The only AMP-related implementation is
the scaler helper.

## Current HPC implications

1. **Expected memory and throughput benefits are absent.** The A100/H100 jobs
   still hold activations and run kernels primarily in FP32. Instantiating a
   `GradScaler` alone does not reduce activation memory or select Tensor Core
   mixed-precision kernels.
2. **The code is safer than naïvely enabling FP16 globally.** Inputs, model
   parameters, waveform losses, STFT-related operations, and evaluation remain
   in their existing precision. No current AMP-specific numerical failures are
   implied by the scaler path because autocast is not active.
3. **The scaler is still operational.** It can detect inf/nan gradients and
   adjust its scale, but with FP32 forward computation this is primarily
   overflow protection, not mixed-precision acceleration.
4. **Gradient accumulation is already wired through the scaler.** The engines
   call the scaler on every micro-batch and pass `update_grad` only on the
   optimizer-step boundary. The existing positive `clip_grad` guard must be
   retained: the configured value `0.0` means “do not clip”.
5. **Checkpoint compatibility needs care when AMP is introduced.** Existing
   checkpoints contain a `scaler` state, but no autocast dtype or AMP mode
   metadata. A future run must explicitly preserve the current FP32 behavior
   when resuming old checkpoints, or record the selected AMP mode in the
   checkpoint/config.

## Recommended implementation path before enabling AMP on HPC

1. Add an explicit configuration/CLI choice, for example `--amp_dtype
   {none,fp16,bf16}`, defaulting to `none` for backward compatibility. Select
   `torch.autocast(device_type="cuda", dtype=...)` only when CUDA and the
   requested mode are available.
2. Put the **forward pass and loss calculation** inside one shared autocast
   helper/context used by pretraining, classification/AU training, and
   waveform training. Do not duplicate slightly different AMP logic in each
   engine.
3. Keep `GradScaler` enabled for FP16; use no loss scaling for BF16 unless a
   measured experiment demonstrates a need. The scaler API should therefore
   be optional rather than unconditionally constructed.
4. Keep numerically sensitive operations in FP32 where required, especially
   waveform correlation/statistics, MR-STFT losses, reductions, and metric
   calculations. Validate loss finiteness and gradient norms.
5. **DDP Gradient Accumulation:** When accumulating gradients across micro-batches,
   wrap the forward and backward passes in `model.no_sync()` for all steps except
   the final one before `optimizer.step()`. This prevents unnecessary cross-node
   communication overhead and maximizes throughput.
6. Add a one-GPU HPC smoke comparison for `none`, `bf16`, and `fp16` covering:
   successful forward/backward, checkpoint save/resume, peak memory, step
   throughput, loss finiteness, and representative validation metrics.
7. **Explicit Checkpoint Metadata:** Record the AMP mode by adding `"amp_dtype": args.amp_dtype`
   to the dictionary saved in `utils/checkpoint.py:save_model`. Add a check in `load_model`
   to explicitly `logger.warning()` if a user resumes an `fp32` checkpoint with
   `--amp_dtype fp16` (or vice versa), as optimizer momentum states might behave differently.
   Compare against an FP32 baseline before using AMP for thesis results.

## Proposed Implementation Details

The smallest useful change is to add an explicit `--amp_dtype` option and wrap
each training engine's forward pass and loss construction with
`torch.autocast(device_type="cuda", dtype=...)`. The existing
`NativeScalerWithGradNormCount` should be modified to accept an `enabled` flag
so that gradient norm logging and clipping are preserved even when AMP is off
or BF16 is used without scaling.

```python
# Setup (e.g., in runner)
amp_enabled = args.amp_dtype in ["fp16", "bf16"]
amp_dtype = torch.float16 if args.amp_dtype == "fp16" else (torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32)
scaler_enabled = args.amp_dtype == "fp16"

# Engine loop
with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
    outputs = model(samples)
    loss = criterion(outputs, targets) / update_freq

grad_norm = loss_scaler(
    loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
    update_grad=should_update, scaler_enabled=scaler_enabled
)
```

**Note on `NativeScalerWithGradNormCount`:** Modify the `__call__` method to bypass `self._scaler` when `scaler_enabled=False`. It should still perform standard `loss.backward()`, `clip_grad_norm_`, and `optimizer.step()` to keep engine code clean without needing `if/else` branches for the optimizer step.

Do not convert the whole model with `.half()`. Autocast selects lower
precision for supported kernels while retaining FP32 for operations that need
it. The default should remain `none` so existing runs and checkpoints preserve
their behavior.

For A100/H100 jobs, a practical first comparison is:

| Mode | Expected role | Loss scaler |
|---|---|---|
| `none` | FP32 baseline and fallback | No |
| `fp16` | Smallest change using the current scaler | Yes |
| `bf16` | Often more stable for waveform/STFT workloads | Usually no |

The exact benefit must be measured, but ViT/video workloads commonly expose
approximately **1.2x–2x throughput**, **25%–50% lower activation memory**, and
potentially a **1.3x–2x larger batch size**. The end-to-end gain can be lower
because video decoding, ROI preparation, and waveform processing may be
CPU-bound.

For waveform fine-tuning, a conservative implementation should autocast the
model forward but calculate correlation, standard-deviation, MR-STFT, and other
numerically sensitive loss reductions in FP32:

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(samples)
with torch.autocast("cuda", enabled=False):
    loss = criterion(outputs.float(), targets.float())
```

The existing HPC smoke test should be run once in each mode and compare
forward/backward success, checkpoint save/resume, peak GPU memory, step
throughput, finite losses, and representative validation metrics. Until that
comparison is complete, the acceleration figures above are planning estimates,
not measured repository results.

## BDD Acceptance Criteria

To ensure the AMP implementation behaves correctly across different configurations, the following Behavior-Driven Development (BDD) scenarios should be verified (e.g., via `pytest` in `code/tests/` once the test suite is populated):

**Scenario 1: Default behavior preserves FP32 execution**
* **Given** the training script is executed without the `--amp_dtype` flag (or with `--amp_dtype none`)
* **When** the model performs a forward and backward pass
* **Then** the `torch.autocast` context should be disabled
* **And** the loss scaler should be bypassed (`scaler_enabled=False`)
* **And** the checkpoint metadata should record `"amp_dtype": "none"`

**Scenario 2: FP16 enables autocast and loss scaling**
* **Given** the training script is executed with `--amp_dtype fp16`
* **When** the model performs a forward and backward pass
* **Then** the `torch.autocast` context should be enabled with `dtype=torch.float16`
* **And** the loss scaler should scale the loss and unscale the gradients (`scaler_enabled=True`)
* **And** the gradient norm should be correctly logged and clipped if configured

**Scenario 3: BF16 enables autocast without loss scaling**
* **Given** the training script is executed with `--amp_dtype bf16`
* **When** the model performs a forward and backward pass
* **Then** the `torch.autocast` context should be enabled with `dtype=torch.bfloat16`
* **And** the loss scaler should be bypassed (`scaler_enabled=False`)
* **And** the gradient norm should still be correctly logged and clipped if configured

**Scenario 4: Checkpoint resumption warns on dtype mismatch**
* **Given** a checkpoint saved with `"amp_dtype": "none"`
* **When** the user attempts to resume training from this checkpoint with `--amp_dtype fp16`
* **Then** the system should log a visible warning about the dtype mismatch
* **And** the training should proceed using the newly requested `fp16` dtype

**Scenario 5: DDP Gradient Accumulation avoids unnecessary syncs**
* **Given** a DDP training run with gradient accumulation steps > 1
* **When** processing micro-batches before the final update step
* **Then** the forward and backward passes should execute within the `model.no_sync()` context
* **And** cross-node gradient synchronization should only occur on the final micro-batch

## Decision

The AMP switch is **implemented and opt-in**. Every pre-existing run keeps the
FP32 behaviour because the default is `--amp_dtype none` (autocast off, scaler
bypassed), so historical results are still valid FP32 runs — do NOT relabel them
as “AMP runs” (they were FP32 with a now-bypassed loss scaler).

To adopt AMP for thesis results, run the one-GPU comparison described above:

```bash
AMP_DTYPE=none sbatch scripts/hpc/smoke_pretrain.sbatch
AMP_DTYPE=bf16 sbatch scripts/hpc/smoke_pretrain.sbatch
AMP_DTYPE=fp16 sbatch scripts/hpc/smoke_pretrain.sbatch
```

and record the peak-memory / throughput / metric deltas against the FP32
baseline. Until that comparison is done, treat every speed or memory figure in
this note as a planning estimate, not a measured repository result.
