"""Masked pre-training loop.

This mirrors the *shape* of VideoMAE's
``tmp/videomae/engine_for_pretraining.py`` and MultiMAE's
``run_pretraining_multimae.py`` training loop.

The batch-level reconstruction step is thesis-specific. Reference recipe
(VideoMAE): the dataset yields ``(samples, labels, index, bool_masked_pos)``;
the model implements ``forward(samples, bool_masked_pos)`` returning
``(pred, target)`` token tensors, and the masked loss is::

    loss = criterion(pred, target, bool_masked_pos)   # see core/criterion.py

Unpack and adapt inside the loop below where marked.
"""
import contextlib
from typing import Iterable, Optional

import torch

from utils import (MetricLogger, SmoothedValue, autocast_context,
                   scaler_enabled)


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler=None, max_norm: float = 0.0,
                    model_ema=None, log_writer=None,
                    lr_schedule_values=None, wd_schedule_values=None,
                    update_freq: int = 1,
                    start_steps: Optional[int] = None,
                    amp_config=None):
    model.train(True)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    use_scaler = scaler_enabled(amp_config)   # False for --amp_dtype none/bf16

    optimizer.zero_grad()
    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, 10, header)):
        # Multimodal MAE: the batch is a dict {stream: tensor}. A plain tensor
        # batch (single-stream placeholder model) is still accepted.
        step = data_iter_step // update_freq
        # index into the GLOBAL step schedule (start_steps = epoch * n_steps)
        sched_idx = step if start_steps is None else start_steps + step

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    # multiply by the group's layer-wise LR scale (1.0 when
                    # layer decay is off): the schedule is ONE curve, so without
                    # this a per-group lr_scale would be overwritten every step.
                    param_group['lr'] = (lr_schedule_values[sched_idx]
                                         * param_group.get('lr_scale', 1.0))
                if (wd_schedule_values is not None
                        and param_group['weight_decay'] > 0):
                    param_group['weight_decay'] = wd_schedule_values[sched_idx]

        final_step = (data_iter_step + 1) % update_freq == 0
        with contextlib.ExitStack() as stack:
            # autocast the forward/loss: a no-op unless --amp_dtype fp16/bf16
            stack.enter_context(autocast_context(amp_config))
            # DDP: skip the gradient all-reduce on accumulation micro-steps
            # (a no-op for update_freq == 1, i.e. every existing config)
            if not final_step and hasattr(model, 'no_sync'):
                stack.enter_context(model.no_sync())

            if isinstance(batch, dict):
                x = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                out = model(x)
                loss = out['loss'] / update_freq
            else:
                samples = batch[0].to(device, non_blocking=True)
                out = model(samples)
                loss = (out['loss'] if isinstance(out, dict) else out) / update_freq

            if loss_scaler is None:
                loss.backward()
                if max_norm is not None and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if final_step:
                    optimizer.step()
                    optimizer.zero_grad()
                grad_norm = None
            else:
                grad_norm = loss_scaler(
                    loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(),
                    update_grad=final_step,
                    scaler_enabled=use_scaler)
                if final_step:
                    optimizer.zero_grad()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss.item() * update_freq)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm.item())
        # Per-stream diagnostics for the multimodal MAE: the RAW masked MSE of
        # every stream (the weighted contributions would hide which stream
        # actually moved). A placeholder/single-stream model simply logs
        # nothing extra.
        if isinstance(out, dict):
            for name, value in (out.get('losses_mse') or {}).items():
                metric_logger.update(**{f'mse_{name}': value})
            # AMMM plan diagnostics (masking='ammm' only, so the key set of the
            # legacy path is untouched): the number of KEPT units per stream and
            # the resulting encoder sequence length. Logged every step because
            # they are a property of the step's Dirichlet draw, not of the run.
            kept = out.get('kept') or {}
            if kept:
                metric_logger.update(**{f'kept_{name}': value
                                        for name, value in kept.items()})
            if 'n_batch' in out:
                metric_logger.update(n_batch=out['n_batch'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate_pretrain(model: torch.nn.Module, data_loader: Iterable,
                      device: torch.device, *, mask_base: int,
                      kept_override=None) -> dict:
    """One deterministic validation pass (ADR-001 I6, Criterion 2).

    The recipe is fixed here rather than left to the caller, because a
    validation curve is only comparable across epochs if every stochastic part
    of the forward path is pinned:

    * ``model.eval()`` and ``torch.inference_mode()`` -- no dropout, no graph;
    * **no autocast**: the pass runs in FP32 so two evaluations of one
      checkpoint compare equal bit-for-bit;
    * the mask GENERATOR is re-created per batch as
      ``protocol.val_mask_seed(mask_base, batch_idx)`` -- the same masks every
      epoch, different masks across batches;
    * ``kept_override`` (``val_kept_<stream>``) pins the masking FRACTION, so
      the loss is not a draw from the AMMM distribution;
    * the caller supplies an ORDERED, single-worker loader (``shuffle=False``,
      ``drop_last=False``, ``num_workers=0``).

    :returns: ``{'val_loss', 'components': {'mse_<stream>': …}, 'kept':
        {'kept_<stream>': …}, 'n_samples', 'n_batches', 'mask_base',
        'pinned'}`` -- the per-component record ADR-001 section 2.3 asks for.
    """
    from protocol import val_mask_seed

    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_samples = 0
    n_batches = 0
    components: dict = {}
    kept_totals: dict = {}

    for batch_idx, batch in enumerate(data_loader):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(val_mask_seed(mask_base, batch_idx)))
        if isinstance(batch, dict):
            x = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(x, mask_generator=generator,
                        kept_override=kept_override)
        else:
            samples = batch[0].to(device, non_blocking=True)
            out = model(samples)
        if not isinstance(out, dict) or 'loss' not in out:
            raise SystemExit(
                '[val] the model returned no masked loss; evaluate_pretrain '
                'needs the multimodal MAE forward, not a placeholder model.')
        batch_n = int(next(iter(out['masks'].values())).shape[0])
        total_loss += float(out['loss']) * batch_n
        n_samples += batch_n
        n_batches += 1
        for name, value in (out.get('losses_mse') or {}).items():
            components.setdefault(f'mse_{name}', 0.0)
            components[f'mse_{name}'] += float(value) * batch_n
        for name, value in (out.get('kept') or {}).items():
            kept_totals.setdefault(f'kept_{name}', 0.0)
            kept_totals[f'kept_{name}'] += float(value) * batch_n

    model.train(was_training)
    if n_samples == 0:
        raise SystemExit(
            '[val] the validation loader produced no sample; an empty '
            'validation pass cannot produce a criterion value (I7).')

    return {
        'val_loss': total_loss / n_samples,
        'components': {k: v / n_samples for k, v in components.items()},
        'kept': {k: v / n_samples for k, v in kept_totals.items()},
        'n_samples': n_samples,
        'n_batches': n_batches,
        'mask_base': int(mask_base),
        'pinned': dict(kept_override or {}),
    }
