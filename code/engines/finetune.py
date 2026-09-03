"""Fine-tuning / evaluation loop.

Generic supervised loop used by ``runners/run_finetune.py``. Follows the
structure of VideoMAE ``tmp/videomae/engine_for_finetuning.py``.

Batch contract (override in your dataset):
    ``samples, targets``    -> classification (this loop)
For masked pre-training see ``engines/pretrain.py``.
"""
from typing import Iterable, Optional

import torch

from utils import MetricLogger, SmoothedValue
from utils.metrics import accuracy


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    loss_scaler=None, max_norm: float = 0.0,
                    model_ema=None, mixup_fn=None, log_writer=None,
                    lr_schedule_values=None, wd_schedule_values=None,
                    update_freq: int = 1):
    model.train(True)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    optimizer.zero_grad()
    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, 10, header)):
        # NOTE: adjust unpacking to your dataset yield, e.g. VideoMAE yields
        # (samples, targets, index, bool_masked_pos).
        samples, targets = batch[:2]
        step = data_iter_step // update_freq

        # per-step LR / weight-decay update (cosine schedules)
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group['lr'] = lr_schedule_values[step]
                if (wd_schedule_values is not None
                        and param_group['weight_decay'] > 0):
                    param_group['weight_decay'] = wd_schedule_values[step]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / update_freq

        if loss_scaler is None:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
            grad_norm = None
        else:
            # AMP path (loss_scaler = NativeScalerWithGradNormCount)
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
        metric_logger.update(min_lr=optimizer.param_groups[0]['lr'])

    # gather the stats across all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """Classification evaluation. Returns {top1, top5, loss}."""
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter='  ')
    header = 'Test:'

    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, targets = batch[:2]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats across all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5,
                  losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
