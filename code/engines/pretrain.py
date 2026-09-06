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
from typing import Iterable, Optional

import torch

from utils import MetricLogger, SmoothedValue


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler=None, max_norm: float = 0.0,
                    model_ema=None, log_writer=None,
                    lr_schedule_values=None, wd_schedule_values=None,
                    update_freq: int = 1,
                    start_steps: Optional[int] = None):
    model.train(True)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

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
                    param_group['lr'] = lr_schedule_values[sched_idx]
                if (wd_schedule_values is not None
                        and param_group['weight_decay'] > 0):
                    param_group['weight_decay'] = wd_schedule_values[sched_idx]

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
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
            grad_norm = None
        else:
            grad_norm = loss_scaler(
                loss, optimizer, clip_grad=max_norm,
                parameters=model.parameters(),
                update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss.item() * update_freq)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm.item())

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
