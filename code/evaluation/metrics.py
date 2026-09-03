"""Tier-1 / Tier-2 waveform metrics (offline evaluation).

Operate on 1D waveforms given as numpy arrays of shape ``[T]`` or ``[N, T]``
(predicted / ground truth). SciPy is only required for the spectral metrics.
"""
import numpy as np

__all__ = ['time_domain_metrics', 'spectral_metrics', 'to_numpy']

try:
    from scipy import signal as _sp_signal
except ImportError:      # pragma: no cover - scipy optional
    _sp_signal = None


def to_numpy(x):
    """Accept torch.Tensor or numpy arrays and return a numpy array."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float32)


def _flatten(x):
    x = to_numpy(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    assert x.ndim == 2, f'Expected [T] or [N, T], got {x.shape}'
    return x


def time_domain_metrics(pred, target):
    """Tier 1: per-sample MAE, RMSE and Pearson r; returns macro averages.

    NOTE (plan nuance): point-level error can look good for a flat 'average'
    line lacking physiological peaks -- always read alongside Tier 2/3.
    """
    p = _flatten(pred)
    t = _flatten(target)
    if p.shape != t.shape:
        raise ValueError(f'Shape mismatch pred {p.shape} vs target {t.shape}')

    mae = np.mean(np.abs(p - t), axis=1)
    rmse = np.sqrt(np.mean((p - t) ** 2, axis=1))

    def _pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    pearson = np.mean([_pearson(pi, ti) for pi, ti in zip(p, t)])

    return {
        'mae': float(np.mean(mae)),
        'rmse': float(np.mean(rmse)),
        'pearson': float(pearson),
    }


def spectral_metrics(pred, target, fs=100.0, band=None):
    """Tier 2: Welch PSD consistency + dominant-frequency error.

    :param fs: sampling rate (Hz) of the waveforms
    :param band: optional (f_low, f_high) band-pass region to compare, e.g.
        (1.0, 2.5) for BVP or (0.16, 0.4) for RESP.
    """
    if _sp_signal is None:
        raise ImportError('spectral_metrics requires scipy (`pip install scipy`)')

    p = _flatten(pred)
    t = _flatten(target)
    if p.shape != t.shape:
        raise ValueError(f'Shape mismatch pred {p.shape} vs target {t.shape}')

    psd_mae, dom_freq_err = [], []
    for pi, ti in zip(p, t):
        f_p, pxx_p = _sp_signal.welch(pi, fs=fs, nperseg=min(256, len(pi)))
        f_t, pxx_t = _sp_signal.welch(ti, fs=fs, nperseg=min(256, len(ti)))

        if band is not None:
            keep = (f_p >= band[0]) & (f_p <= band[1])
            f_p, f_t = f_p[keep], f_t[keep]
            pxx_p, pxx_t = pxx_p[keep], pxx_t[keep]

        # normalise each PSD to unit sum for a scale-free comparison
        pxx_p = pxx_p / (pxx_p.sum() + 1e-12)
        pxx_t = pxx_t / (pxx_t.sum() + 1e-12)
        psd_mae.append(float(np.mean(np.abs(pxx_p - pxx_t))))

        def _dominant(f, pxx):
            return float(f[np.argmax(pxx)])

        dom_freq_err.append(abs(_dominant(f_p, pxx_p) - _dominant(f_t, pxx_t)))

    return {
        'psd_mae': float(np.mean(psd_mae)),
        'dominant_freq_error_hz': float(np.mean(dom_freq_err)),
    }
