# 03 — Engine and Loss

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/engines/`, `code/utils/`, `code/runners/` on 2026-10-04.

---

## 1. Engine Overview

| File | Function(s) | Stage | Notes |
|------|------------|-------|-------|
| `engines/pretrain.py` | `train_one_epoch` | Stage 2 | Batch is `dict {stream: Tensor}` |
| `engines/finetune.py` | `train_one_epoch`, `evaluate` | Stage 3 / classification | Supervised `(samples, targets)` tuples |
| `engines/waveform.py` | `evaluate_waveforms`, `predict_waveforms` | Stage 3 eval | `@torch.no_grad()` decorated |
| `engines/au_probe.py` | `train_one_epoch_au`, `evaluate_au` | AU probe | BCE + per-AU F1 |
| `engines/visualize.py` | `visualize` | Debug | Prediction visualisation |

**No `trainer.py` or `evaluator.py`** exist. The Stage-3 training loop lives directly in `runners/run_waveform.py`.

---

## 2. Stage-2 Pre-training Loop — `engines/pretrain.py::train_one_epoch`

### Batch contract

```python
# dict batch (multimodal MAE):
x = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
out = model(x);  loss = out['loss'] / update_freq

# plain tensor (single-stream fallback):
samples = batch[0].to(device, non_blocking=True)
out = model(samples)
loss = (out['loss'] if isinstance(out, dict) else out) / update_freq
```

### Per-step LR / WD update

```python
sched_idx = step if start_steps is None else start_steps + step
for pg in optimizer.param_groups:
    pg['lr'] = lr_schedule_values[sched_idx] * pg.get('lr_scale', 1.0)
    if pg['weight_decay'] > 0:
        pg['weight_decay'] = wd_schedule_values[sched_idx]
```

`lr_scale` enables layer-wise LR decay without overwriting the per-group scale. Groups without `lr_scale` default to 1.0.

### Gradient step — AMP path

```python
grad_norm = loss_scaler(
    loss, optimizer, clip_grad=max_norm,
    parameters=model.parameters(),
    update_grad=(data_iter_step + 1) % update_freq == 0)
if (data_iter_step + 1) % update_freq == 0:
    optimizer.zero_grad()
```

### Gradient step — non-AMP path

```python
loss.backward()
if max_norm is not None and max_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
if (data_iter_step + 1) % update_freq == 0:
    optimizer.step(); optimizer.zero_grad()
```

### Metric logging (all `.item()` before storing)

```python
metric_logger.update(loss=loss.item() * update_freq)
metric_logger.update(lr=optimizer.param_groups[0]['lr'])
if grad_norm: metric_logger.update(grad_norm=grad_norm.item())
# Per-stream diagnostics (value already a Python float from model forward):
for name, val in out.get('losses_mse', {}).items():
    metric_logger.update(**{f'mse_{name}': val})
for name, val in out.get('losses_spectral', {}).items():
    metric_logger.update(**{f'spec_{name}': val})
```

---

## 3. Fine-tuning Loop — `engines/finetune.py::train_one_epoch`

```python
samples, targets = batch[:2]
samples = samples.to(device, non_blocking=True)
targets = targets.to(device, non_blocking=True)
if mixup_fn: samples, targets = mixup_fn(samples, targets)
outputs = model(samples)
loss = criterion(outputs, targets) / update_freq
```

### `evaluate` (classification only)

```python
@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()   # hardcoded — classification only
    model.eval()
    ...
    acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    metric_logger.update(loss=loss.item())
```

> **Potential Risk**: `CrossEntropyLoss` is hardcoded. This function is **not** suitable for waveform regression. Stage-3 uses `engines/waveform.py::evaluate_waveforms` instead.

---

## 4. Stage-3 Waveform Evaluation — `engines/waveform.py`

### `predict_waveforms`

```python
@torch.no_grad()
def predict_waveforms(data_loader, model, device):
    model.eval()
    for batch in data_loader:
        samples, target = batch[:2]
        samples = samples.to(device, non_blocking=True)
        out = model(samples)      # [B, T]
        preds.append(to_numpy(out)); targets.append(to_numpy(target))
    return np.concatenate(preds, 0), np.concatenate(targets, 0)  # [N, T]
```

### `evaluate_waveforms`

```python
pred, target = predict_waveforms(data_loader, model, device)
results = time_domain_metrics(pred, target)   # Tier 1: MAE, RMSE, Pearson
if with_spectral:
    results.update(spectral_metrics(pred, target, fs=fs, band=band))  # Tier 2
```

### Metric tiers

| Tier | Metrics | Library |
|------|---------|---------|
| 1 | MAE, RMSE, Pearson | numpy |
| 2 | Welch PSD, dominant freq error, HF power | scipy (optional) |
| 3 | RMSSD, pNN50, MedianNN, ShanEn | neurokit2 (optional, offline only) |

Tier 3 requires >= 30 s of signal; run offline via `runners/run_evaluate.py`.

---

## 5. AU Probe Engine — `engines/au_probe.py`

- Default `criterion = nn.BCEWithLogitsLoss()` (multi-label binary).
- `model.set_train(True)` instead of `model.train(True)` — respects linear-probe frozen-layer semantics.
- Threshold 0.5; macro-averaged F1 over per-AU scores.
- `loss.item()` called before `metric_logger.update` — safe.

---

## 6. Multi-task Loss Composition

### Stage-2 weighted sum

```python
# In MultiModalMAE.forward:
L_total = sum(loss_weights.get(s, signal_weight if s in signals else 1.0)
              * MaskedMSELoss(pred_s, target_s, mask_s)
              for s in streams)
       + sum(spectral_weights.get(s, 0.0) * STFT_s for s in signal_streams)
```

Default: visual streams = 1.0, physiological streams = 0.5 (`signal_weight`).
`spectral_weight=0.0` in all current configs (decision 2026-09-26).

### Stage-3 joint loss

```python
# WaveformJointLoss.forward:
L = alpha * StdLoss(pred, target)
  + beta  * PearsonLoss(pred, target)
  + gamma * MultiResolutionSTFTLoss(pred, target)
```

Gradient norm balance (measured 2026-09-28, N=496 @ 100 Hz):
- `StdLoss`: `1/sqrt(N) ≈ 0.0449` (~1.81 % of composite)
- `MultiResolutionSTFTLoss`: ~63–84 % of composite norm at alpha=beta=gamma=1

---

## 7. AMP — `utils/native_scaler.py::NativeScalerWithGradNormCount`

```python
self._scaler = torch.cuda.amp.GradScaler()  # deprecated PyTorch >= 2.4

def __call__(self, loss, optimizer, clip_grad=None, ...):
    self._scaler.scale(loss).backward()
    if update_grad:
        if clip_grad is not None and clip_grad > 0:  # GUARD: 0.0 = OFF
            self._scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            self._scaler.unscale_(optimizer)
            norm = get_grad_norm_(parameters)
        self._scaler.step(optimizer); self._scaler.update()
    return norm
```

> **Critical fix (2026-09-23)**: Before this fix, `clip_grad=0.0` (the default in many configs) entered the `clip_grad_norm_` branch with `max_norm=0.0`, which scales every gradient to zero (`clip_coef = 0 / (norm + eps)`). Learning was silently disabled. The guard `clip_grad > 0` now routes 0.0 to the no-clip path.

> **No `torch.amp.autocast`** anywhere in the codebase. Forward passes run in full precision; only the backward is scaled.

---

## 8. LR Schedule — `utils/lr_sched.py::cosine_scheduler`

Returns a numpy array of length `epochs * niter_per_ep`:
- Linear warmup from `start_warmup_value` to `base_value` over `warmup_epochs`.
- Cosine decay from `base_value` to `final_value`.

Applied **per step** in engine loops:

```python
pg['lr'] = lr_schedule_values[sched_idx] * pg.get('lr_scale', 1.0)
```

---

## 9. Optimizer — `utils/optim_factory.py::create_optimizer`

Supports: `adamw` (default), `adam`, `sgd`.

Parameter grouping:
- `no_decay`: 1-D params, biases, `no_weight_decay()` set -> `weight_decay=0`.
- `decay`: all others -> `weight_decay=args.weight_decay`.

Layer-wise LR decay (`build_layer_decay_assigner`):

```python
def get_num_layer_for_multimae(var_name, num_max_layer):
    if var_name.startswith(('adapters.', 'positions.')): return 0   # most decayed
    if var_name.startswith('enc_blocks.'): return int(block_idx) + 1
    return num_max_layer - 1                                        # top (LR scale 1.0)
```

`lr_scale[i] = layer_decay ^ (num_layers + 1 - i)`.

---

## 10. DDP — `utils/dist.py::init_distributed_mode`

| Launch mode | Detection | Backend |
|-------------|-----------|---------|
| `torchrun` / `torch.distributed.launch` | `RANK`, `WORLD_SIZE`, `LOCAL_RANK` | NCCL |
| SLURM `srun` | `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID` | NCCL |

SLURM `MASTER_ADDR` auto-derived from `SLURM_NODELIST`. `MASTER_PORT` defaults to `29500`.

```python
torch.distributed.init_process_group(backend='nccl', init_method='env://', ...)
torch.distributed.barrier()
setup_for_distributed(args.rank == 0)  # suppress non-master prints
```

`args.distributed=False` when no env vars found (local single-GPU mode).

DDP helpers: `get_rank`, `get_world_size`, `is_main_process`, `save_on_master`, `get_model`, `all_gather`.

`MetricLogger.synchronize_between_processes()` calls `dist.all_reduce` on each meter's `(count, total)`.

---

## 11. Checkpointing — `utils/checkpoint.py`

Checkpoint dict: `{'model', 'optimizer', 'epoch', 'scaler', 'args', ['model_ema']}`.

```python
torch.load(path, map_location='cpu', weights_only=False)
# weights_only=False required: 'args' is argparse.Namespace with pickled numpy scalars.
# torch >= 2.6 defaults to weights_only=True and refuses unpickling.
```

`auto_resume_model` reads `{output_dir}/checkpoints/latest_checkpoint.txt` if no explicit `--resume`.

EMA: `utils/model_ema.py`. Updated after each optimizer step: `model_ema.update(model)`.

---

## 12. Metric Logging — `utils/logger.py`

`SmoothedValue`: running window deque + global (count, total). Properties: `median`, `avg`, `global_avg`, `max`, `value`.

`MetricLogger.update(**kwargs)`: calls `.item()` on any Tensor before storing -> no computational graph retained (defensive).

`log_every(iterable, print_freq, header)`: wraps the data iterator, prints ETA and per-step timing.
