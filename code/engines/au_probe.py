"""AU-occurrence probe training/eval -- ADD-ON (does not touch Stage loops).

Multi-label binary BCE training + per-AU F1 evaluation (BP4D protocol:
threshold 0.5, macro-average F1 over the target AUs). AU labels contain no
"9" here (the dataset already drops them), so the loss needs no masking.
"""
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from utils import MetricLogger, SmoothedValue

__all__ = ['train_one_epoch_au', 'evaluate_au']


def train_one_epoch_au(model: nn.Module, data_loader: Iterable,
                       optimizer: torch.optim.Optimizer, device: torch.device,
                       epoch: int, criterion=None, loss_scaler=None,
                       max_norm: float = 0.0, update_freq: int = 1,
                       lr_schedule_values=None, start_steps: int = 0):
    model.set_train(True)
    criterion = criterion or nn.BCEWithLogitsLoss()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'AU Epoch: [{}]'.format(epoch)

    optimizer.zero_grad()
    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, 10, header)):
        samples, targets = batch[:2]
        step = start_steps + data_iter_step // max(1, update_freq)

        if lr_schedule_values is not None:
            for i, pg in enumerate(optimizer.param_groups):
                if step < len(lr_schedule_values):
                    pg['lr'] = lr_schedule_values[step]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(samples)
        loss = criterion(logits, targets)
        loss = loss / max(1, update_freq)

        if loss_scaler is None:
            loss.backward()
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if (data_iter_step + 1) % max(1, update_freq) == 0:
                optimizer.step()
                optimizer.zero_grad()
            grad_norm = None
        else:
            grad_norm = loss_scaler(
                loss, optimizer, clip_grad=max_norm,
                parameters=model.parameters(),
                update_grad=(data_iter_step + 1) % max(1, update_freq) == 0)
            if (data_iter_step + 1) % max(1, update_freq) == 0:
                optimizer.zero_grad()

        metric_logger.update(loss=loss.item() * max(1, update_freq))
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_au(data_loader, model: nn.Module, device: torch.device):
    """Multi-label evaluation -> per-AU F1 @ 0.5 and macro-averaged F1."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    metric_logger = MetricLogger(delimiter='  ')
    header = 'AU Test:'

    n_aus = None
    tp = fp = fn = torch.zeros(0)
    total_loss, total = 0.0, 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, targets = batch[:2]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(samples)
        loss = criterion(logits, targets)

        if n_aus is None:
            n_aus = targets.shape[1]
            tp = torch.zeros(n_aus, device=device)
            fp = torch.zeros(n_aus, device=device)
            fn = torch.zeros(n_aus, device=device)

        preds = (torch.sigmoid(logits) >= 0.5)
        tgt = (targets >= 0.5)
        tp += (preds & tgt).sum(dim=0).float()
        fp += (preds & ~tgt).sum(dim=0).float()
        fn += (~preds & tgt).sum(dim=0).float()
        b = samples.shape[0]
        total_loss += loss.item() * b
        total += b
        metric_logger.meters['loss'].update(loss.item(), n=b)

    denom = (2 * tp + fp + fn).clamp(min=1e-6)
    # an AU with no positives AND no predictions is undefined; leave its F1 ~0
    f1_per_au = (2 * tp / denom).cpu().tolist()
    macro_f1 = float(torch.tensor(f1_per_au).mean())
    metric_logger.synchronize_between_processes()
    mean_loss = metric_logger.meters['loss'].global_avg
    print(f'* AU macro-F1 {macro_f1 * 100:.2f}  loss {mean_loss:.4f}')
    return {'f1': macro_f1 * 100.0, 'loss': mean_loss,
            'f1_per_au': f1_per_au}
