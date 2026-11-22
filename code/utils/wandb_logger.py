"""Optional Weights & Biases logging for the training runners.

``--log_wandb`` / ``--wandb_project`` have been declared in
``runners/_common.py::add_common_args`` for a long time, but **nothing consumed
them**: a config with ``log_wandb: true`` trained exactly as before and logged
nowhere. This module is that consumer (Stage 2 today; add it to a runner with two
lines).

Rules that keep it safe in a multi-hour, multi-GPU job
------------------------------------------------------
* **Lazy import** — ``wandb`` is imported only when logging is switched ON, so a
  run (and the whole test suite) without the package behaves as before.
* **Rank 0 only** — a DDP job must not open one run per rank.
* **Never fatal** — every import/init failure degrades to "logging disabled"
  with one warning line; a metrics backend must not kill a training run.
* **Offline by default on Slurm** — compute nodes have no outbound internet, so
  ``WANDB_MODE`` defaults to ``offline`` inside a batch job (``SLURM_JOB_ID``
  set) and to ``online`` elsewhere. Override explicitly with
  ``WANDB_MODE=online|offline|disabled``; sync an offline run later from a
  LOGIN node with ``wandb sync <output_dir>/wandb/offline-run-*``.
* **Artefacts stay with the run** — ``WANDB_DIR`` defaults to
  ``<output_dir>/wandb`` (``$WORK_SCRATCH`` here), never the repo or ``$HOME``.

Resuming after a Slurm requeue: export ``WANDB_RUN_ID=<id>`` (the id printed in
the run URL / ``WANDB_RUN_ID`` file) and the same run is continued instead of a
new one being created.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, Mapping, Optional

__all__ = ['WandbLogger']

#: keys of ``vars(args)`` that are safe/meaningful to store as run config
_SCALAR_TYPES = (str, int, float, bool, list, tuple, dict, type(None))


class WandbLogger:
    """Thin, failure-tolerant wrapper around one W&B run.

    ``WandbLogger(None)`` is a VALID object: every method is a no-op, so call
    sites never need to branch on whether logging is enabled.
    """

    def __init__(self, run=None, *, project: Optional[str] = None,
                 run_name: Optional[str] = None, mode: Optional[str] = None):
        self._run = run
        self.project = project
        self.run_name = run_name
        self.mode = mode

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_args(cls, args, *, job_type: str = 'train',
                  tags: Iterable[str] = (), extra: Optional[Mapping] = None
                  ) -> 'WandbLogger':
        """Build a logger from the runner's ``args`` namespace.

        :param args: must carry ``log_wandb``, ``wandb_project`` and
            ``output_dir`` (all supplied by ``add_common_args``).
        :param job_type: W&B ``job_type`` label (``pretrain`` / ``finetune``).
        :param tags: extra run tags (e.g. the data set and masking engine).
        :param extra: additional run-config entries (e.g. counts computed after
            the dataset was built).
        """
        if not bool(getattr(args, 'log_wandb', False)):
            return cls(None)

        from .dist import is_main_process          # lazy: no torch import cost
        if not is_main_process():
            return cls(None)                       # DDP: rank 0 owns the run

        project = str(getattr(args, 'wandb_project', '') or 'thesis-project')
        out_dir = str(getattr(args, 'output_dir', '') or '.')
        run_name = os.path.basename(os.path.normpath(out_dir)) or None

        # offline by default inside a batch job: compute nodes have no egress
        if 'WANDB_MODE' not in os.environ:
            os.environ['WANDB_MODE'] = ('offline' if os.environ.get('SLURM_JOB_ID')
                                        else 'online')
        mode = os.environ['WANDB_MODE']
        # wandb appends its own ``wandb/`` level, so the base must be the RUN
        # DIR: files land in <out_dir>/wandb/offline-run-* (setting it to
        # <out_dir>/wandb yields the silly <out_dir>/wandb/wandb/).
        os.environ.setdefault('WANDB_DIR', out_dir)

        try:
            import wandb
        except ImportError:
            print('[wandb] --log_wandb is on but the `wandb` package is not '
                  'installed; logging disabled (pip install wandb).')
            return cls(None, mode='disabled')

        config: Dict[str, Any] = {
            k: v for k, v in vars(args).items() if isinstance(v, _SCALAR_TYPES)}
        config.update(dict(extra or {}))

        kwargs: Dict[str, Any] = dict(
            project=project, name=run_name, config=config,
            dir=os.environ['WANDB_DIR'], job_type=job_type,
            tags=[str(t) for t in tags if str(t)], reinit=True)
        run_id = os.environ.get('WANDB_RUN_ID', '').strip()
        if run_id:
            kwargs.update(id=run_id, resume='allow')

        try:
            run = wandb.init(**kwargs)
        except Exception as exc:                   # noqa: BLE001 - never fatal
            print(f'[wandb] init failed ({type(exc).__name__}: {exc}); '
                  f'logging disabled for this run.')
            return cls(None, mode='disabled')

        print(f'[wandb] project={project} run={run_name} mode={mode} '
              f'dir={os.environ["WANDB_DIR"]}/wandb')
        if mode != 'online':
            print('[wandb] OFFLINE run: upload it later from a LOGIN node with\n'
                  f'        wandb sync {os.environ["WANDB_DIR"]}/wandb/offline-run-*')
        return cls(run, project=project, run_name=run_name, mode=mode)

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Log scalars; tensors are detached, non-finite values are dropped.

        Only numbers, bools, strings and tensors are forwarded -- an arbitrary
        object would make W&B serialise a pickle the dashboard cannot plot.
        """
        if self._run is None:
            return
        clean: Dict[str, Any] = {}
        for key, value in metrics.items():
            try:
                if hasattr(value, 'detach'):        # torch.Tensor
                    value = float(value.detach().cpu())
                elif isinstance(value, bool):
                    pass
                elif isinstance(value, (int, float)):
                    value = float(value)
                    if not math.isfinite(value):
                        continue
                elif not isinstance(value, str):
                    continue
            except (TypeError, ValueError):
                continue
            clean[str(key)] = value
        if clean:
            self._run.log(clean, step=step)

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        """TensorBoard-compatible sink: ``add_scalar(tag, value, global_step)``.

        Exists so an engine can write a per-step series through its generic
        ``log_writer`` parameter without importing W&B. A disabled logger makes
        it a no-op, exactly like :meth:`log`.
        """
        self.log({tag: value}, step=step)

    def finish(self) -> None:
        """Close the run (safe to call twice / on a disabled logger)."""
        if self._run is None:
            return
        try:
            self._run.finish()
        finally:
            self._run = None
