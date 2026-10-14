"""AU-occurrence probe training/eval -- ADD-ON (does not touch Stage loops).

Multi-label binary BCE training + per-AU F1 evaluation (BP4D protocol:
threshold 0.5, macro-average F1 over the target AUs). AU labels contain no
"9" here (the dataset already drops them), so the loss needs no masking.

Protocol of record: ``analysis/tir_resp/AU_OccurenceProbe.md``. This module adds
the three protocol-mandated pieces the original engine lacked:

* a **trivial always-positive baseline** per AU (``F1 = 2p/(1+p)``) -- a
  degenerate all-positive head already scores a high macro-F1 on the
  near-constant AUs, so the headline is reported next to this floor (protocol §5);
* **DDP-correct** metrics: the raw probabilities are all-gathered before the F1
  is computed, so a multi-GPU run reports the GLOBAL val metric instead of one
  rank's shard (protocol §5 DDP caveat);
* an optional **per-AU threshold tuning** upper bound (protocol §10.1), kept
  OUT of the @0.5 headline.
"""
import contextlib
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from utils import (MetricLogger, SmoothedValue, autocast_context,
                   scaler_enabled)

__all__ = ['train_one_epoch_au', 'evaluate_au', 'trivial_f1',
           'tune_per_au_thresholds', 'DEFAULT_THRESHOLD_GRID']


#: threshold grid for the optional per-AU tuning upper bound (protocol §10.1).
DEFAULT_THRESHOLD_GRID = tuple(round(x, 3) for x in np.arange(0.1, 0.91, 0.05))


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module (the probe) whether or not ``model`` is a
    ``DistributedDataParallel`` wrapper. Custom methods such as ``set_train``
    live on the probe, not on the wrapper, which does not proxy attributes."""
    return model.module if hasattr(model, 'module') else model


def _dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def trivial_f1(presence: Sequence[float]) -> np.ndarray:
    """Always-positive-per-AU baseline F1 from the AU presence rate ``p``:
    ``2p / (1 + p)`` (protocol §5.3)."""
    p = np.asarray(presence, dtype=np.float64)
    denom = 1.0 + p
    return np.where(denom > 0.0, 2.0 * p / denom, 0.0)


def _f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2.0 * tp + fp + fn
    return np.where(denom > 0.0, 2.0 * tp / np.maximum(denom, 1e-12), 0.0)


def tune_per_au_thresholds(probs: np.ndarray, targets: np.ndarray,
                           grid: Sequence[float] = DEFAULT_THRESHOLD_GRID
                           ) -> tuple:
    """Per-AU decision threshold maximising F1 (protocol §10.1).

    ``probs``/``targets`` are ``[N, A]`` (probabilities, and 0/1 targets).
    Returns ``(thresholds[A], f1[A])`` where ``f1`` is the best F1 per AU over
    ``grid``. Ties keep the smallest threshold (highest recall).
    """
    probs = np.asarray(probs, dtype=np.float64)
    tgt = np.asarray(targets) >= 0.5
    n_au = probs.shape[1] if probs.ndim == 2 else 0
    thresholds = np.full(n_au, 0.5, dtype=np.float64)
    best = np.zeros(n_au, dtype=np.float64)
    for j in range(n_au):
        pj, tj = probs[:, j], tgt[:, j]
        for thr in sorted(grid):
            pred = pj >= thr
            tp = float(np.sum(pred & tj))
            fp = float(np.sum(pred & ~tj))
            fn = float(np.sum(~pred & tj))
            f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
            if f1 > best[j]:
                best[j], thresholds[j] = f1, float(thr)
    return thresholds, best


def _gather_predictions(probs: torch.Tensor, tgt: torch.Tensor
                        ) -> tuple:
    """All-gather ``(probs, targets)`` across ranks so the metric is global.

    Uses ``all_gather_object`` (rank shards may differ in length under a
    ``DistributedSampler``). No-op when DDP is not initialised.
    """
    if not _dist_initialized() or dist.get_world_size() == 1:
        return probs, tgt
    payload = [probs.numpy(), tgt.numpy()]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    probs = torch.from_numpy(np.concatenate([g[0] for g in gathered], axis=0))
    tgt = torch.from_numpy(np.concatenate([g[1] for g in gathered], axis=0))
    return probs, tgt


def train_one_epoch_au(model: nn.Module, data_loader: Iterable,
                       optimizer: torch.optim.Optimizer, device: torch.device,
                       epoch: int, criterion=None, loss_scaler=None,
                       max_norm: float = 0.0, update_freq: int = 1,
                       lr_schedule_values=None, start_steps: int = 0,
                       amp_config=None):
    _unwrap(model).set_train(True)
    criterion = criterion or nn.BCEWithLogitsLoss()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'AU Epoch: [{}]'.format(epoch)

    use_scaler = scaler_enabled(amp_config)   # False for --amp_dtype none/bf16

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

        final_step = (data_iter_step + 1) % max(1, update_freq) == 0
        with contextlib.ExitStack() as stack:
            stack.enter_context(autocast_context(amp_config))
            # DDP: skip the gradient all-reduce on accumulation micro-steps
            # (a no-op for update_freq == 1, i.e. every existing config)
            if not final_step and hasattr(model, 'no_sync'):
                stack.enter_context(model.no_sync())

            logits = model(samples)
            loss = criterion(logits, targets)
            loss = loss / max(1, update_freq)

            if loss_scaler is None:
                loss.backward()
                if max_norm and max_norm > 0:
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

        metric_logger.update(loss=loss.item() * max(1, update_freq))
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_au(data_loader, model: nn.Module, device: torch.device,
                thresholds: Optional[Sequence[float]] = None,
                tune_thresholds: bool = False,
                threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID) -> dict:
    """Multi-label evaluation -> per-AU F1, macro-F1, and the trivial baseline.

    Returns a dict with the headline ``f1`` (macro-F1 @ ``thresholds``, default
    0.5) and, so the number can be read honestly (protocol §5):

    * ``trivial_f1`` / ``trivial_f1_per_au`` -- the always-positive baseline;
    * ``f1_minus_trivial`` -- the headline minus that baseline;
    * ``presence_rate`` -- per-AU positive rate on the evaluated split;
    * ``oracle_f1`` / ``oracle_thresholds`` (only when ``tune_thresholds``) --
      the per-AU-tuned UPPER BOUND; never a headline (protocol §10.1).

    Metrics are computed from all-gathered predictions, so a multi-GPU run
    reports the global split rather than one rank's shard (protocol §5).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    metric_logger = MetricLogger(delimiter='  ')
    header = 'AU Test:'

    probs_parts: List[torch.Tensor] = []
    tgt_parts: List[torch.Tensor] = []
    total_loss, total = 0.0, 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, targets = batch[:2]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(samples)
        loss = criterion(logits, targets)

        probs_parts.append(torch.sigmoid(logits).float().cpu())
        tgt_parts.append((targets >= 0.5).float().cpu())
        b = samples.shape[0]
        total_loss += loss.item() * b
        total += b
        metric_logger.meters['loss'].update(loss.item(), n=b)

    metric_logger.synchronize_between_processes()
    mean_loss = metric_logger.meters['loss'].global_avg

    if probs_parts:
        probs = torch.cat(probs_parts, dim=0)
        tgt = torch.cat(tgt_parts, dim=0)
    else:                                   # empty split (should not happen)
        probs = torch.zeros(0, 0)
        tgt = torch.zeros(0, 0)
    probs, tgt = _gather_predictions(probs, tgt)          # DDP: make it global

    n_au = int(tgt.shape[1]) if tgt.ndim == 2 else 0
    if n_au == 0:
        return {'f1': 0.0, 'f1_per_au': [], 'loss': mean_loss,
                'trivial_f1': 0.0, 'trivial_f1_per_au': [],
                'f1_minus_trivial': 0.0, 'presence_rate': [], 'n': 0}

    tgt_np = tgt.numpy()
    probs_np = probs.numpy()
    presence = tgt_np.mean(axis=0)
    trivial_per_au = trivial_f1(presence)

    if thresholds is None:
        thr = np.full(n_au, 0.5, dtype=np.float64)
    else:
        thr = np.asarray(thresholds, dtype=np.float64)
        if thr.shape[0] != n_au:
            raise ValueError(
                f'thresholds must have {n_au} entries, got {thr.shape[0]}')
    pred = probs_np >= thr[None, :]
    tb = tgt_np >= 0.5
    tp = np.sum(pred & tb, axis=0).astype(np.float64)
    fp = np.sum(pred & ~tb, axis=0).astype(np.float64)
    fn = np.sum(~pred & tb, axis=0).astype(np.float64)
    f1_per_au = _f1_from_counts(tp, fp, fn)
    macro_f1 = float(f1_per_au.mean())
    trivial_macro = float(trivial_per_au.mean())

    out = {
        'f1': macro_f1 * 100.0,
        'f1_per_au': f1_per_au.tolist(),
        'loss': mean_loss,
        'trivial_f1': trivial_macro * 100.0,
        'trivial_f1_per_au': trivial_per_au.tolist(),
        'f1_minus_trivial': (macro_f1 - trivial_macro) * 100.0,
        'presence_rate': presence.tolist(),
        'thresholds': thr.tolist(),
        'n': int(tgt.shape[0]),
    }

    if tune_thresholds:
        best_thr, best_f1 = tune_per_au_thresholds(
            probs_np, tgt_np, threshold_grid)
        out['oracle_f1'] = float(best_f1.mean()) * 100.0
        out['oracle_f1_per_au'] = best_f1.tolist()
        out['oracle_thresholds'] = best_thr.tolist()
        out['oracle_f1_minus_trivial'] = (
            float(best_f1.mean()) - trivial_macro) * 100.0

    print(f'* AU macro-F1 {out["f1"]:.2f}  (trivial {out["trivial_f1"]:.2f}, '
          f'delta {out["f1_minus_trivial"]:+.2f})  loss {mean_loss:.4f}')
    if tune_thresholds:
        print(f'* AU macro-F1 (per-AU tuned, UPPER BOUND) '
              f'{out["oracle_f1"]:.2f}')
    return out
