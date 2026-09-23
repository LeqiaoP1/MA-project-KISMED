"""Stage-3 waveform evaluation (offline post-processing of predictions).

Computes Tier-1 (MAE / RMSE / Pearson) and Tier-2 (Welch PSD) metrics over a
validation loader whose batches yield ``(samples, target_waveform)``.
Tier-3 clinical metrics are intentionally NOT computed here during training;
use ``runners/run_evaluate.py`` offline on saved predictions.
"""
import numpy as np
import torch

from evaluation.metrics import spectral_metrics, time_domain_metrics, to_numpy

__all__ = ['evaluate_waveforms', 'predict_waveforms']


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


@torch.no_grad()
def evaluate_waveforms(data_loader, model, device, fs=100.0, band=None,
                       with_spectral=True):
    """Evaluate predicted vs ground-truth waveforms.

    :param data_loader: batches of ``(samples, target_waveform)``
    :param fs: sampling rate of the waveforms (Hz)
    :param band: optional (f_low, f_high) for the spectral comparison
    :param with_spectral: run Tier-2 Welch PSD metrics (requires scipy)
    :returns: dict of metrics (e.g. mae, rmse, pearson, psd_mae)

    Metrics only; use :func:`predict_waveforms` when the raw arrays are needed
    (saved predictions, figures, session assembly).
    """
    pred, target = predict_waveforms(data_loader, model, device)
    if pred.size == 0:
        return {}

    results = time_domain_metrics(pred, target)
    if with_spectral:
        try:
            results.update(spectral_metrics(pred, target, fs=fs, band=band))
        except ImportError:
            results['psd_mae'] = float('nan')
    return results
