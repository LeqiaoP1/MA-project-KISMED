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
                    amp_config=None,
                    mask_base: int = 42, steps_per_epoch: int = 0,
                    rank: int = 0):
    """One pre-training epoch.

    ``mask_base`` / ``steps_per_epoch`` / ``rank`` drive the ADR-002 mask RNG:
    every step gets a COUNTS generator from a rank-independent key (so all
    ranks derive the same plan without a collective) and a PLACEMENT generator
    that includes the rank (so the samples differ across ranks). The global RNG
    is left to the model's own dropout and is never used for masking, which is
    what keeps the counts identical across DDP ranks.
    """
    from core.mask_placement import counts_generator, placement_generator
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
                # global step index -> reproducible, rank-independent plan
                global_step = (epoch * max(1, int(steps_per_epoch))
                               + data_iter_step)
                out = model(
                    x,
                    counts_generator=counts_generator(
                        mask_base, epoch, global_step),
                    placement_generator=placement_generator(
                        mask_base, epoch, global_step, rank=rank))
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
            # ADR-002 plan diagnostics: the sampled MODE and the kept TOKEN
            # count per stream. Logged every step because they are a property
            # of the step's draw, not of the run; the key set is a function of
            # the config (every declared stream appears on every step).
            kept = out.get('kept') or {}
            if kept:
                metric_logger.update(**{f'kept_{name}': value
                                        for name, value in kept.items()})
            if 'mode' in out:
                metric_logger.update(mode=out['mode'])
            if 'n_batch' in out:
                metric_logger.update(n_batch=out['n_batch'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate_pretrain(model: torch.nn.Module, data_loader: Iterable,
                      device: torch.device, *, mask_base: int) -> dict:
    """One deterministic validation pass (ADR-001 I6, Criterion 2).

    The recipe is fixed here rather than left to the caller, because a
    validation curve is only comparable across epochs if every stochastic part
    of the forward path is pinned:

    * ``model.eval()`` and ``torch.inference_mode()`` -- no dropout, no graph;
    * **no autocast**: the pass runs in FP32 so two evaluations of one
      checkpoint compare equal bit-for-bit;
    * the mask generators are re-created per batch as
      ``protocol.val_mask_seed(mask_base, batch_idx)`` -- the same sequence of
      modes, counts and placements every epoch, different across batches. No
      per-stream fraction is hand-pinned (ADR-002 §6): the policy replays a
      deterministic DRAW sequence instead;
    * the caller supplies an ORDERED, single-worker loader (``shuffle=False``,
      ``drop_last=False``, ``num_workers=0``).

    :returns: ``{'val_loss', 'components': {'mse_<stream>': …}, 'kept':
        {'kept_<stream>': …}, 'mode': {…}, 'n_samples', 'n_batches',
        'mask_base', 'pinned'}``.
    """
    from core.mask_placement import counts_generator, placement_generator
    from protocol import val_mask_seed

    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_samples = 0
    n_batches = 0
    components: dict = {}
    kept_totals: dict = {}
    mode_totals: dict = {}

    for batch_idx, batch in enumerate(data_loader):
        seed = int(val_mask_seed(mask_base, batch_idx))
        if isinstance(batch, dict):
            x = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(x,
                        counts_generator=counts_generator(seed, 0, batch_idx),
                        placement_generator=placement_generator(
                            seed, 0, batch_idx, rank=0))
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
        if 'mode' in out:
            mode_totals.setdefault(f'mode{int(out["mode"])}', 0.0)
            mode_totals[f'mode{int(out["mode"])}'] += batch_n

    model.train(was_training)
    if n_samples == 0:
        raise SystemExit(
            '[val] the validation loader produced no sample; an empty '
            'validation pass cannot produce a criterion value (I7).')

    return {
        'val_loss': total_loss / n_samples,
        'components': {k: v / n_samples for k, v in components.items()},
        'kept': {k: v / n_samples for k, v in kept_totals.items()},
        'mode': {k: v / n_samples for k, v in mode_totals.items()},
        'n_samples': n_samples,
        'n_batches': n_batches,
        'mask_base': int(mask_base),
        'pinned': {},
    }
