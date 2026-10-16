"""AMP helpers: the autocast policy + a loss scaler that returns the grad norm.

Port of MultiMAE ``tmp/MultiMAE/utils/native_scaler.py`` (used by MAE and
VideoMAE), extended with the repository's shared mixed-precision policy.

This module is the single source of truth for ``--amp_dtype {none,fp16,bf16}``:
:func:`resolve_amp` turns the flag into an :class:`AmpConfig` (autocast on/off +
loss-scaling on/off) and :func:`autocast_context` is the context manager every
training engine wraps its forward/loss in. See ``notes/design/hpc_AMP.md``.

Invariants
----------
* ``none`` (default) = FP32 with the scaler BYPASSED -- bit-identical to the
  pre-AMP behaviour. The scaler OBJECT is still built and used so the engines
  keep logging the gradient norm through one code path.
* ``fp16`` = autocast + loss scaling: FP16 has a narrow exponent range, so loss
  scaling is required to avoid gradient underflow.
* ``bf16`` = autocast WITHOUT loss scaling: BF16 shares FP32's exponent range.
* CUDA-only: on a CPU device the flag degrades to ``none`` with a loud note.

The model is NEVER converted with ``.half()`` -- autocast only lowers the
precision of the kernels that support it, keeping FP32 for the operations that
need it (loss reductions, MR-STFT, normalisation, metrics).
"""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ['AMP_DTYPES', 'AmpConfig', 'add_amp_args', 'resolve_amp',
           'autocast_context', 'scaler_enabled',
           'NativeScalerWithGradNormCount']

#: the ``--amp_dtype`` choices mapped to the autocast dtype (None = no autocast)
AMP_DTYPES = {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}


@dataclass(frozen=True)
class AmpConfig:
    """Resolved AMP policy for one run (see :func:`resolve_amp`)."""
    mode: str                      # 'none' | 'fp16' | 'bf16' (as requested)
    dtype: Optional[torch.dtype]   # autocast dtype, None when disabled
    enabled: bool                  # run the forward/loss under torch.autocast
    scaler: bool                   # use the GradScaler (loss scaling)
    device_type: str               # 'cuda' | 'cpu'

    def describe(self) -> str:
        """One log line naming the EFFECTIVE policy (not just the request)."""
        if not self.enabled:
            return f'amp_dtype={self.mode} (FP32; autocast OFF, scaler OFF)'
        return (f'amp_dtype={self.mode} (autocast {self.dtype} on '
                f'{self.device_type}, loss scaler '
                f'{"ON" if self.scaler else "OFF"})')


def add_amp_args(parser) -> None:
    """Add the shared ``--amp_dtype`` switch to a runner's argument parser.

    Lives next to the code that acts on it so the choice list and the help text
    have ONE definition.
    """
    parser.add_argument(
        '--amp_dtype', default='none', type=str, choices=list(AMP_DTYPES),
        help="mixed-precision mode for the training forward/backward. "
             "'none' (default) = FP32 with the GradScaler BYPASSED -- "
             "bit-identical to the pre-AMP behaviour; 'fp16' = autocast + loss "
             "scaling; 'bf16' = autocast WITHOUT loss scaling (usually the most "
             "stable choice for waveform/STFT workloads). CUDA-only: on a CPU "
             "device the flag degrades to 'none' with a note. The model is "
             "never converted with .half(); autocast only lowers the precision "
             "of the kernels that support it.")


def resolve_amp(amp_dtype: Optional[str], device) -> AmpConfig:
    """Turn a ``--amp_dtype`` string into an :class:`AmpConfig` for ``device``.

    Raises ``ValueError`` on an unknown mode: argparse ``choices`` guards the
    CLI, but a YAML value reaches the variable through ``parser.set_defaults``
    (which skips ``type``/``choices``), so the string must be validated here.
    """
    mode = str(amp_dtype if amp_dtype is not None else 'none').strip().lower()
    if mode not in AMP_DTYPES:
        raise ValueError(
            f'unknown --amp_dtype {amp_dtype!r}; expected one of '
            f'{sorted(AMP_DTYPES)}')

    # ``torch.autocast('cuda', ...)`` needs a real CUDA device: on CPU it would
    # either error (fp16) or change nothing meaningful for this repo's loops.
    is_cuda = torch.cuda.is_available() and str(device).startswith('cuda')
    device_type = 'cuda' if is_cuda else 'cpu'

    if mode == 'none' or not is_cuda:
        if mode != 'none':
            print(f'[amp] --amp_dtype {mode} requested but the active device is '
                  f'{device_type!r} (AMP requires CUDA) -> disabling AMP and '
                  f'running FP32.')
        return AmpConfig(mode='none', dtype=None, enabled=False, scaler=False,
                         device_type=device_type)

    return AmpConfig(mode=mode, dtype=AMP_DTYPES[mode], enabled=True,
                     scaler=(mode == 'fp16'), device_type=device_type)


def autocast_context(amp_config: Optional[AmpConfig]):
    """Context manager: autocast when enabled, else a no-op ``nullcontext``.

    Use as ``with autocast_context(amp_config): out = model(x); loss = ...``.
    ``None`` (a caller that predates AMP, e.g. the notebook) is treated as
    ``none`` so no caller has to construct an :class:`AmpConfig` to stay FP32.
    """
    if amp_config is None or not amp_config.enabled:
        return nullcontext()
    return torch.autocast(device_type=amp_config.device_type,
                          dtype=amp_config.dtype)


def scaler_enabled(amp_config: Optional[AmpConfig]) -> bool:
    """Whether ``NativeScalerWithGradNormCount`` should actually scale.

    ``None`` means a caller supplied a loss scaler WITHOUT an AMP policy (the
    legacy call sites): honour it, i.e. keep loss scaling. An explicit
    :class:`AmpConfig` wins, so ``none``/``bf16`` bypass the scaler while still
    logging the gradient norm through the same helper object.
    """
    return True if amp_config is None else bool(amp_config.scaler)


class NativeScalerWithGradNormCount:
    state_dict_key = 'amp_scaler'

    def __init__(self):
        # torch.cuda.amp.GradScaler() is deprecated and prints a FutureWarning
        # on EVERY rank of EVERY job (it polluted the Stage-2 logs). The
        # torch.amp form needs torch>=2.3 while requirements.txt only asks for
        # >=2.1, so keep the legacy call as a fallback instead of raising the
        # floor. Both are the same object; only the constructor moved.
        try:
            self._scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):   # torch < 2.3
            self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None,
                 create_graph=False, update_grad=True,
                 scaler_enabled: bool = True):
        # ``scaler_enabled=False`` (--amp_dtype none/bf16) bypasses the
        # GradScaler but keeps the SAME code path, so the gradient norm is still
        # measured and clipping still applies.
        if scaler_enabled:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        else:
            loss.backward(create_graph=create_graph)
        if update_grad:
            # ``clip_grad`` must be POSITIVE to clip. The guard mirrors the
            # non-AMP branch in engines/*: ``clip_grad_norm_(..., max_norm=0.0)
            # scales EVERY gradient to exactly zero (clip_coef = 0/(norm+eps)),
            # which silently disabled learning for every run whose config left
            # clip_grad at its 0.0 default ("no clipping"), while still
            # RETURNING a healthy-looking grad norm (measured before the
            # scaling). Stage-3 bp/resp and Stage-2 were both affected.
            if scaler_enabled:
                self._scaler.unscale_(optimizer)   # unscale the gradients
            if clip_grad is not None and clip_grad > 0:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = get_grad_norm_(parameters)
            if scaler_enabled:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
                         for p in parameters]), norm_type)
    return total_norm
