# 05 — Risks and Compliance

> **READ-ONLY SPECIFICATION** — reverse-engineered from `code/` on 2026-10-04.
> This file documents hardcoded parameters, missing dimension assertions, potential OOM hazards,
> and other compliance risks found in the live codebase.
> **No source files were modified during this audit.**

---

## 1. Hardcoded Parameters

### `core/criterion.py`

| Location | Value | Risk |
|----------|-------|------|
| Module-level `_EPS = 1e-6` | Denominator in all masked losses | Safe in practice; could surface if `mask.sum() == 0` (all tokens visible). |
| `MaskedMSELoss.__init__`: `patch_size=16`, `stride=1` | Default constructor args | Unused in current `forward()` — misleading interface; callers may assume they affect computation. |

### `core/waveform_losses.py`

| Location | Value | Risk |
|----------|-------|------|
| Module-level `_EPS = 1e-8` | Denominator in waveform losses | Slightly tighter than criterion.py; consistent within file. |
| `MultiResolutionSTFTLoss.__init__`: `fft_sizes=(64,128,256)` | Default FFT windows | **Wrong for RESP** (0.1–0.6 Hz): 64/128-sample windows (0.64/1.28 s) are shorter than one breath period. `resp_tir_roi_local.yaml` correctly overrides to `128,256,512`. |
| `_warned_degenerate = False` | Instance-level warning flag | Warning fires once per model instance. All subsequent degenerate batches are silently skipped — no per-epoch count in logs. |

### `core/multimae.py`

| Location | Value | Risk |
|----------|-------|------|
| `STREAM_CHANNELS = {'rgb':3,'tir':3,'bp':1,'resp':1,'eda':1}` | Default channel counts | A new modality requires updating this dict. |
| `dec_depth=2` (constructor default) | Decoder depth | Not exposed as a YAML key; changing requires a code edit or explicit CLI override. |
| `_VISUAL_STREAMS = ('rgb','tir')` | Valid visual stream names | Hardcoded tuple; adding a new visual modality requires editing two constants. |
| `_SIGNAL_STREAMS = ('bp','resp','eda')` | Valid signal stream names | Same risk. |
| `spectral_weight=0.0` constructor default | MR-STFT disabled | Intentional (decision 2026-09-26). Documented in all YAML comments. |

### `utils/dist.py`

| Location | Value | Risk |
|----------|-------|------|
| `MASTER_PORT = '29500'` | Default SLURM port | Port collision if multiple jobs on the same node. Override via `MASTER_PORT` env var per job. |
| `args.dist_backend = 'nccl'` | Fixed backend | NCCL requires CUDA. CPU-only or Gloo-based runs need a code change. |

### `runners/_common.py`

| Location | Value | Risk |
|----------|-------|------|
| `--device` default `'cuda'` | Silent CPU fallback | No warning printed when CUDA unavailable; a slow CPU run may appear correct. |
| `--wandb_project` default `'thesis-project'` | Hardcoded W&B project | All runs log to the same project by default; must override for separate experiments. |
| `torch.backends.cudnn.benchmark = True` | Set in `init_env` | Disables deterministic CUDA algorithms. Disable for strict reproducibility. |

### `utils/native_scaler.py`

| Location | Value | Risk |
|----------|-------|------|
| `torch.cuda.amp.GradScaler()` | Deprecated PyTorch >= 2.4 | Currently functional; migrate to `torch.amp.GradScaler('cuda')` before removal. |

---

## 2. Missing Tensor Dimension Assertions

### Missing (risks)

| File | Class / Method | Missing check |
|------|---------------|---------------|
| `core/input_adapters.py` | `PatchedInputAdapter.forward` | No `assert x.ndim == 4`; a 3-D input raises inside Conv2d with an unhelpful error. |
| `core/input_adapters.py` | `SemSegInputAdapter.forward` | No channel count assertion. |
| `core/input_adapters.py` | `SignalInputAdapter.forward` | No `ndim` check. |
| `core/output_adapters.py` | `SpatialOutputAdapter.forward` | No `assert self.head is not None`; if `init()` was skipped, raises NoneType with no context. |
| `core/blocks.py` | `CrossAttention.forward` | No check that `x` and `context` share the same `C`; mismatch surfaces inside Linear. |

### Present — positive examples

| Location | Assertion |
|----------|-----------|
| `MultiModalMAE.__init__` | `assert input_size % ph == 0`, `assert num_frames % t == 0` |
| `MultiModalMAE._check_space_time_alignment` | `ValueError` on time-axis misalignment |
| `MultiModalWaveformRegressor.__init__` | `raise ValueError` if `output_len % grid_t != 0` |
| `MultiModalWaveformRegressor.forward` | `raise ValueError` if `tok.shape[1] != n_visual` |
| `MultiModalMAEProbe.__init__` | `assert input_size % ph == 0`, `assert num_frames % t == 0` |

---

## 3. Missing `.item()` Calls — GPU OOM Risk

A tensor loss stored without `.item()` retains the computational graph in GPU memory.

### Safe paths (`.item()` IS called)

| File | Location | Status |
|------|----------|--------|
| `engines/pretrain.py` | `metric_logger.update(loss=loss.item() * update_freq)` | OK |
| `engines/pretrain.py` | `metric_logger.update(grad_norm=grad_norm.item())` | OK |
| `engines/finetune.py` | `metric_logger.update(loss=loss.item() * update_freq)` | OK |
| `engines/finetune.py::evaluate` | `metric_logger.update(loss=loss.item())` | OK |
| `engines/au_probe.py` | `metric_logger.update(loss=loss.item() * ...)` | OK |
| `engines/au_probe.py::evaluate_au` | `total_loss += loss.item() * b` | OK |
| `utils/logger.py::MetricLogger.update` | Calls `.item()` on any Tensor before storing | OK (defensive) |

### Potential Risk — `MultiResolutionSTFTLoss`

```python
total = torch.tensor(0.0, device=p.device, dtype=p.dtype)
total = total + sc + lm   # accumulates as GPU tensor
```

`total` is returned as a tensor, passed to `.backward()`, then extracted with `.item()` by the engine's `metric_logger.update`. The graph is held only for one batch. **No persistent OOM risk; low severity.**

### Action Required — `runners/run_waveform.py`

The Stage-3 training loop is inline in this large file. Verify all metric scalars use `.item()` before accumulation:

```bash
grep -n "\.item()" code/runners/run_waveform.py
```

---

## 4. Unhandled Edge Cases

### 4.1 `engines/finetune.py::evaluate` — hardcoded `CrossEntropyLoss`

```python
criterion = torch.nn.CrossEntropyLoss()   # classification only
```

If called for waveform regression, silently produces incorrect metrics. Stage-3 uses `engines/waveform.py::evaluate_waveforms`. No runtime guard. **Risk: Medium.**

### 4.2 `data/paired_dataset.py::_pad_time` — edge padding without warning

```python
np.pad(arr, pad, mode='edge')
```

Short clips (session end) are padded by repeating the last frame/sample. No warning emitted. **Risk: Low (training), Medium (evaluation).**

### 4.3 `data/masking_generator.py::TubeMaskingGenerator` — dimension naming confusion

```python
self.height, self.width, self.depth = input_size[0], input_size[1], input_size[2]
# input_size = (T, H, W) but:  height=T  width=H  depth=W
```

The maths is correct (`total = T*H*W`) but `__repr__` outputs `[depth, height, width]` = `[W, T, H]` — incorrect display order. **Risk: Low functional; High readability. Implicit Assumption.**

### 4.4 `core/au_probe.py::load_au_probe_weights` — `print()` not logger

Load summary uses `print()` rather than `MetricLogger`. Suppressed on non-master DDP ranks by `setup_for_distributed`. Not a bug; deviates from logging convention.

### 4.5 `utils/dist.py::all_gather` — deprecated `torch.ByteStorage`

```python
storage = torch.ByteStorage.from_buffer(buffer)
tensor = torch.ByteTensor(storage).to('cuda')
```

Deprecated in PyTorch >= 2.0. **Risk: Low now; Medium in future PyTorch versions. Implicit Assumption (written for PyTorch 1.x patterns).**

### 4.6 `data/paired_dataset.py` — `split_by='session'` default allows subject leakage

Alphabetical session split at `train_ratio` may put `F001_T1` in train and `F001_T2` in val. Always use `split_by='subject'` for evaluation configs. **Risk: Medium (evaluation validity).**

### 4.7 `core/waveform_losses.py::MultiResolutionSTFTLoss` — `_warned_degenerate` fires once per instance

After the first degenerate batch, all subsequent ones are silently skipped in every subsequent epoch. No per-epoch degenerate count appears in logs. **Risk: Low (training integrity); High (data quality monitoring).**

### 4.8 `core/waveform_model.py` — no `@torch.inference_mode()` on `forward`

Callers must explicitly use `torch.no_grad()` during evaluation. `engines/waveform.py::predict_waveforms` does this correctly. Ad-hoc usage outside the engine may retain unnecessary gradients.

---

## 5. Summary Risk Table

| Risk | Severity | File(s) | Recommended action |
|------|----------|---------|-------------------|
| `split_by='session'` default allows subject leakage | Medium | `paired_dataset.py`, YAML configs | Always set `split_by: subject` in eval configs |
| `finetune.py::evaluate` hardcodes `CrossEntropyLoss` | Medium | `engines/finetune.py` | Verify not called for waveform regression |
| MR-STFT default `fft_sizes=(64,128,256)` unsuitable for RESP | Medium | `waveform_losses.py` | Override to `128,256,512` in all RESP configs |
| `torch.cuda.amp.GradScaler` deprecated | Low-Medium | `utils/native_scaler.py` | Migrate to `torch.amp.GradScaler('cuda')` |
| `torch.ByteStorage` deprecated | Low | `utils/dist.py` | Migrate `all_gather` to `torch.frombuffer` |
| `TubeMaskingGenerator` dimension naming confusion | Low | `data/masking_generator.py` | Rename `height/width/depth` -> `temporal/spatial_h/spatial_w` |
| `_warned_degenerate` fires once only | Low | `core/waveform_losses.py` | Add per-epoch degenerate batch counter |
| `SpatialOutputAdapter.forward` without `init()` guard | Low | `core/output_adapters.py` | Add `assert self.head is not None` |
| Unused `patch_size`/`stride` in `MaskedMSELoss` | Low | `core/criterion.py` | Remove misleading constructor args or implement norm_pix |
| `_pad_time` edge-padding without warning | Low | `data/paired_dataset.py` | Add warning log when padding occurs |
| `MASTER_PORT=29500` fixed | Low | `utils/dist.py` | Set `MASTER_PORT` env var per job in `.sbatch` |
| Silent CPU fallback on `--device cuda` | Low | `runners/_common.py` | Add explicit CUDA availability warning |
