# AMP Implementation — the `--amp_dtype` switch

**Status:** implemented and verified (CPU test suite) on 2026-10-13.
**Companion doc:** [`hpc_AMP.md`](hpc_AMP.md) — the pre-implementation audit and
rationale. Speed/memory figures in that note remain **estimates** until the
one-GPU comparison below is run.
**Scope:** `code/` training runners, engines, the AMP scaler module, checkpointing.

---

## 1. TL;DR

`--amp_dtype {none,fp16,bf16}` (default `none`) turns real mixed precision on.
`none` reproduces the previous FP32 behaviour exactly, so nothing changes unless
the flag is passed. Autocast is CUDA-only; on a CPU device the flag degrades to
`none` with a printed note instead of failing mid-run.

```bash
# FP32 (default, unchanged) / bf16 / fp16
python runners/run_pretrain.py -c configs/pretrain/stage2_hpc_tir_roi_resp.yaml --amp_dtype bf16
```

---

## 2. The contract

| Mode | Autocast forward/loss | Loss scaling (`GradScaler`) | Typical use |
|---|---|---|---|
| `none` | OFF | **OFF** (scaler bypassed) | FP32 baseline / fallback; **default** |
| `fp16` | `torch.float16` | **ON** | Smallest change; narrow FP16 exponent range needs scaling |
| `bf16` | `torch.bfloat16` | **OFF** | Usually most stable for waveform / MR-STFT workloads |

The model is **never** converted with `.half()`. Autocast lowers precision only
for the kernels that support it; loss reductions, MR-STFT, normalisation and
metrics stay FP32.

---

## 3. Where the code lives

| File | Change |
|---|---|
| `code/utils/native_scaler.py` | **Single source of truth.** `AMP_DTYPES`, `AmpConfig`, `resolve_amp()`, `autocast_context()`, `scaler_enabled()`, `add_amp_args()`; `NativeScalerWithGradNormCount.__call__` gained `scaler_enabled=True`. |
| `code/utils/checkpoint.py` | `save_model` stores `'amp_dtype'`; `load_model` warns on a resume in a different mode. |
| `code/engines/pretrain.py` | `amp_config=None` param; forward/loss wrapped in `autocast_context`; scaler bypass; DDP `no_sync` on accumulation micro-steps. |
| `code/engines/finetune.py` | Same, plus `loss_in_fp32=False` (waveform uses `True` → criterion in FP32). Shared by classification **and** Stage-3 waveform. |
| `code/engines/au_probe.py` | Same `amp_config` + `no_sync` treatment for the AU probe loop. |
| `code/runners/_common.py` | Calls `add_amp_args(parser)` inside `add_common_args`. |
| `code/runners/run_pretrain.py`, `run_finetune.py`, `run_waveform.py`, `run_au_probe.py` | Resolve the policy after `init_env`, print `[amp] …`, pass `amp_config`. `run_waveform` also passes `loss_in_fp32=True`. |
| `code/tests/test_amp.py` | 18 BDD tests (CPU-only). |
| `code/scripts/hpc/smoke_pretrain.sbatch` | New `AMP_DTYPE=` env override for the one-GPU comparison. |
| `code/spec/00_overview.md`, `code/spec/03_engine_and_loss.md §7` | Specs updated. |

`utils/__init__.py` needs no change: it already does `from .native_scaler import *`,
so the new names are re-exported automatically (call `from utils import resolve_amp`).

---

## 4. How to use

### Command line

```bash
# local smoke
python runners/run_pretrain.py -c configs/pretrain/example.yaml \
    --amp_dtype bf16 --max_sessions 2 --max_clips 1 --max_entries 2 \
    --batch_size 1 --epochs 1 --warmup_epochs 0 --num_workers 0

# HPC one-GPU comparison (the doc's acceptance step)
AMP_DTYPE=none sbatch scripts/hpc/smoke_pretrain.sbatch
AMP_DTYPE=bf16 sbatch scripts/hpc/smoke_pretrain.sbatch
AMP_DTYPE=fp16 sbatch scripts/hpc/smoke_pretrain.sbatch
```

### YAML config

Because configs supply argparse **defaults**, you may set it in a config:

```yaml
amp_dtype: bf16
```

The unknown-key guard treats a typo (`amp_dtypee`) as a hard error, so a config
cannot silently miss the knob.

### Log line

Every training runner prints the **effective** policy once at startup:

```
[amp] amp_dtype=bf16 (autocast torch.bfloat16 on cuda, loss scaler OFF)
[amp] amp_dtype=none (FP32; autocast OFF, scaler OFF)
```

On a CPU device a request for `fp16`/`bf16` prints:

```
[amp] --amp_dtype fp16 requested but the active device is 'cpu' (AMP requires
CUDA) -> disabling AMP and running FP32.
```

---

## 5. Design decisions & invariants

1. **`none` is bit-identical to the old behaviour.** Autocast is off and the
   `GradScaler` is bypassed. The scaler **object is still constructed** and used
   — cheap, and it keeps gradient-norm logging on one code path.
2. **One shared context, not per-engine AMP logic.** Every engine wraps its
   forward/loss in `autocast_context(amp_config)`.
3. **FP16 scales, BF16 does not.** `scaling = (mode == "fp16")` in one place.
4. **`scaler_enabled=False` keeps the same code path.** Plain `loss.backward()`
   → `clip_grad_norm_`/`get_grad_norm_` → `optimizer.step()`, still returning the
   grad norm and still honouring the `clip_grad > 0` guard (0.0 = "do not clip").
5. **Numerically sensitive losses stay FP32.** `run_waveform.py` passes
   `loss_in_fp32=True`; the engine casts `outputs.float()` and evaluates the
   criterion **outside** the autocast context.
6. **DDP gradient accumulation skips redundant all-reduces.** Non-final
   micro-steps run under `model.no_sync()`. This is a strict no-op for the
   default `--update_freq 1` (every step is the final step).
7. **CUDA-only, fail-soft.** `resolve_amp` needs `torch.cuda.is_available()` and
   a `cuda*` device; otherwise it returns the `none` policy and prints why.

---

## 6. Backward compatibility & edge cases

| Situation | Behaviour |
|---|---|
| No `--amp_dtype` flag | `none` → FP32, scaler bypassed. Existing runs unchanged. |
| Old checkpoint (no `amp_dtype` key) | Resumes silently; no warning. |
| Resume with a **different** mode | Prints `[amp] WARNING: … written with amp_dtype=… but this run uses …`; training proceeds in the requested mode. |
| CPU device + `fp16`/`bf16` | Degrades to `none` with a note. |
| Notebook / external caller | The engines default `amp_config=None` → treated as `none` (FP32), so `train_one_epoch_waveform(...)` in `notebooks/TirROI_Resp_Pipeline.ipynb` is unaffected. |
| `update_freq == 1` | `no_sync` never enters → unchanged gradient sync. |
| YAML sets an invalid mode | `resolve_amp` raises `ValueError` (CLI is additionally guarded by argparse `choices`). |

**Who has the flag:** every runner that calls `add_common_args` —
`run_pretrain.py`, `run_finetune.py`, `run_waveform.py`, `run_au_probe.py`,
`run_visualize.py`. The `run_evaluate*.py` runners do not use `add_common_args`
and therefore do not expose it (evaluation is FP32 inference; AMP is a training
concern).

## 7. What each engine does

| Engine | Autocast scope | Loss precision |
|---|---|---|
| `engines/pretrain.py` | `model(x)` **and** the (`out['loss']`) term | via autocast |
| `engines/finetune.py` (classification) | `model(samples)` **and** `criterion(...)` | via autocast |
| `engines/finetune.py` via `run_waveform.py` | `model(samples)` only | **FP32** (`loss_in_fp32=True`) |
| `engines/au_probe.py` | `model(samples)` **and** `criterion(...)` | via autocast |

Illustrative shape of the loop (finetune):

```python
final_step = (data_iter_step + 1) % update_freq == 0
with contextlib.ExitStack() as stack:
    if not final_step and hasattr(model, 'no_sync'):   # DDP accumulation
        stack.enter_context(model.no_sync())
    with autocast_context(amp_config):                 # no-op unless fp16/bf16
        outputs = model(samples)
    if loss_in_fp32:                                   # waveform criteria
        loss = criterion(outputs.float(), targets.float())
    else:
        with autocast_context(amp_config):
            loss = criterion(outputs, targets)
    loss = loss / update_freq
    grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), update_grad=final_step,
                            scaler_enabled=use_scaler)
```

---

## 8. Testing

`code/tests/test_amp.py` encodes the BDD scenarios from `hpc_AMP.md`:

| # | Scenario | Test |
|---|---|---|
| 1 | Default preserves FP32 (autocast off, scaler bypassed, metadata `none`) | `test_default_is_fp32_…`, `test_explicit_none_…`, `test_legacy_none_amp_config_…` |
| 2 | FP16 → autocast + loss scaling | `test_fp16_enables_autocast_and_loss_scaling` |
| 3 | BF16 → autocast, no scaling | `test_bf16_enables_autocast_without_loss_scaling` |
| 4 | Checkpoint dtype-mismatch warning | `test_checkpoint_records_amp_dtype_and_warns_on_mismatch`, `test_old_checkpoint_without_amp_metadata_still_resumes` |
| 5 | DDP accumulation + plumbing | `test_scaler_bypass_…`, `test_every_training_runner_resolves_and_passes_amp`, `test_waveform_runner_uses_the_fp32_loss`, … |

CPU-only: cases needing a CUDA *decision* monkeypatch `torch.cuda.is_available`
and pass a `cuda` device, so only the **policy** is asserted (never a real
kernel). Plus `test_amp_degrades_to_none_on_cpu`, `test_unknown_mode_raises`.

```bash
cd code
../.env/bin/python -m pytest tests/test_amp.py -q     # 18 passed
../.env/bin/python -m pytest -q                       # 206 passed (whole suite)
```

---

## 9. Verification performed

- `python -m compileall code` — clean.
- `python -m pytest` — **206 passed**, including 18 new AMP tests.
- `--help` on all four training runners shows `[--amp_dtype {none,fp16,bf16}]`.
- All three engine loops (`pretrain`, `finetune`/waveform with `loss_in_fp32`,
  `au_probe`) executed end-to-end on CPU with `--amp_dtype none`.

## 10. Known gaps / next steps

1. **No GPU validation here.** The autocast kernels themselves were not exercised
   (no CUDA in the development session). Run the `none`/`bf16`/`fp16` smoke
   comparison on the cluster and record **peak memory, step throughput, loss
   finiteness and validation metrics** before quoting any speed-up.
2. **Perf numbers are unmeasured.** `hpc_AMP.md`'s 1.2–2× throughput /
   25–50 % memory figures are planning estimates.
3. **BF16 needs Ampere-or-newer for speed.** The usable HPC pool (a100, h100,
   rtx6000→Blackwell) qualifies; older (e.g. V100) devices would not benefit.
4. Minor legacy wart: `NativeScalerWithGradNormCount.state_dict_key = 'amp_scaler'`
   is unused — checkpoints store the scaler under the key `'scaler'`
   (`utils/checkpoint.py`). Left as-is to avoid changing the checkpoint format.

## 11. Disabling AMP

Pass `--amp_dtype none` (or omit the flag). That restores the FP32 path with the
scaler bypassed. To also restore the *previous* loss-scaling behaviour on FP32
(scaling active without autocast) — which the audit argues was never useful — call
`NativeScalerWithGradNormCount.__call__(..., scaler_enabled=True)` explicitly, or
update the engine to honour a scaler without an `amp_config`.
