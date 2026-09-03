"""Stage-3 waveform evaluation (offline post-processing of predictions).

Computes Tier-1 (MAE / RMSE / Pearson) and Tier-2 (Welch PSD) metrics over a
validation loader whose batches yield ``(samples, target_waveform)``.
Tier-3 clinical metrics are intentionally NOT computed here during training;
use ``runners/run_evaluate.py`` offline on saved predictions.
"""
import numpy as np
import torch

from evaluation.metrics import spectral_metrics, time_domain_metrics, to_numpy

__all__ = ['evaluate_waveforms']


@torch.no_grad()
def evaluate_waveforms(data_loader, model, device, fs=100.0, band=None,
                       with_spectral=True):
    """Evaluate predicted vs ground-truth waveforms.

    :param data_loader: batches of ``(samples, target_waveform)``
    :param fs: sampling rate of the waveforms (Hz)
    :param band: optional (f_low, f_high) for the spectral comparison
    :param with_spectral: run Tier-2 Welch PSD metrics (requires scipy)
    :returns: dict of metrics (e.g. mae, rmse, pearson, psd_mae)
    """
    model.eval()
    preds, targets = [], []

    for batch in data_loader:
        samples, target = batch[:2]
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(samples)                # [B, T]
        preds.append(to_numpy(out))
        targets.append(to_numpy(target))

    if not preds:
        return {}

    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)

    results = time_domain_metrics(pred, target)
    if with_spectral:
        try:
            results.update(spectral_metrics(pred, target, fs=fs, band=band))
        except ImportError:
            results['psd_mae'] = float('nan')
    return results
