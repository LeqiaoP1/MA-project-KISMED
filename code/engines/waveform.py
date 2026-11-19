"""Stage-3 waveform evaluation (offline post-processing of predictions).

Computes Tier-1 (MAE / RMSE / Pearson) and Tier-2 (Welch PSD) metrics over a
validation loader whose batches yield ``(samples, target_waveform)``.
Tier-3 clinical metrics are intentionally NOT computed here during training;
use ``runners/run_evaluate.py`` offline on saved predictions.
"""
import numpy as np
import torch

from evaluation.metrics import spectral_metrics, time_domain_metrics, to_numpy

__all__ = ['evaluate_waveform_loss', 'predict_waveforms']


@torch.no_grad()
def predict_waveforms(data_loader, model, device):
    """Run ``data_loader`` and return the raw ``(pred, target)`` arrays.

    Shapes are ``[N, output_len]``, with the rows in loader order -- which
    equals ``dataset.entries`` order for a non-shuffled single-process loader
    (that is what the session-level assembly relies on).
    """
    model.eval()
    preds, targets = [], []
    for batch in data_loader:
        samples, target = batch[:2]
        samples = samples.to(device, non_blocking=True)
        out = model(samples)                # [B, T]
        preds.append(to_numpy(out))
        targets.append(to_numpy(target))
    if not preds:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    if pred.ndim > 2:                       # defensive: [N, 1, T] -> [N, T]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    return pred, target


def _spectral_unavailable():
    """The Tier-2 keys at NaN when scipy is missing (same key set as a run)."""
    return {'psd_mae': float('nan'), 'dominant_freq_error_hz': float('nan'),
            'psd_band_bins': None, 'psd_skipped': 0,
            'hf_power_rel': float('nan'),
            'target_hf_power_rel': float('nan'), 'hf_lo_hz': float('nan')}


@torch.inference_mode()
def evaluate_waveform_loss(data_loader, model, criterion, device,
                           with_spectral=True) -> dict:
    """Deterministic validation pass of the TRAINING objective (I6, I7).

    The validation record must be comparable across epochs, so nothing here may
    depend on the optimiser state or on randomness:

    * ``model.eval()`` + ``torch.inference_mode()``, **no autocast** (FP32), so
      two evaluations of one checkpoint agree bit-for-bit;
    * the caller supplies an ORDERED, single-worker loader
      (``shuffle=False``, ``drop_last=False``, ``num_workers=0``);
    * the same ``WaveformJointLoss`` instance as training is evaluated, and its
      per-component terms are reported next to the total (ADR-001 section 2.3).

    :returns: ``{'val_loss', 'components': {'time', 'pearson', 'stft',
        'metric_pearson', 'metric_mae', ...}, 'n_clips', 'n_batches'}``.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    sums = {'time': 0.0, 'pearson': 0.0, 'stft': 0.0}
    n_clips = 0
    n_batches = 0
    preds, targets = [], []

    for batch in data_loader:
        samples, target = batch[:2]
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        out = model(samples)
        p = out.reshape(out.shape[0], -1) if out.ndim > 2 else out
        t = target.reshape(target.shape[0], -1) if target.ndim > 2 else target
        batch_n = int(p.shape[0])
        # Components FIRST, then the weighted sum: calling ``criterion(p, t)``
        # as well would run the multi-resolution STFT a second time per batch
        # and roughly double the cost of the pass (measured: 1.8x a training
        # step on a tiny model, i.e. over the 5% budget by itself).
        term_time = criterion.time(p, t)
        term_pearson = criterion.pearson(p, t)
        term_stft = criterion.stft(p, t)
        total += float(criterion._time_w * term_time
                       + criterion._pearson_w * term_pearson
                       + criterion._stft_w * term_stft) * batch_n
        sums['time'] += float(term_time) * batch_n
        sums['pearson'] += float(term_pearson) * batch_n
        sums['stft'] += float(term_stft) * batch_n
        n_clips += batch_n
        n_batches += 1
        preds.append(to_numpy(p))
        targets.append(to_numpy(t))

    model.train(was_training)
    if n_clips == 0:
        raise SystemExit(
            '[val] the validation loader produced no clip; an empty pass '
            'cannot produce a criterion value (I7).')

    components = {name: value / n_clips for name, value in sums.items()}
    arrays_pred = np.concatenate(preds, axis=0)
    arrays_target = np.concatenate(targets, axis=0)
    components.update(time_domain_metrics(arrays_pred, arrays_target))
    if with_spectral:
        try:
            components.update(spectral_metrics(arrays_pred, arrays_target))
        except ImportError:
            components.update(_spectral_unavailable())

    return {'val_loss': total / n_clips,
            'components': components,
            'n_clips': n_clips,
            'n_batches': n_batches}
