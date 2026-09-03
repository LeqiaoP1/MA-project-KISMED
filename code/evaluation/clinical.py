"""Tier-3 clinical HRV metrics via NeuroKit2 (offline, non-differentiable).

Maps the plan's clinical set -- RMSSD, pNN50, MedianNN, Shannon entropy
(ShanEn) -- to NeuroKit2. Install with ``pip install neurokit2``.

``extract_hrv_metrics`` first detects PPG peaks, derives NN intervals and uses
``nk.hrv_time`` for RMSSD/pNN50/MedianNN plus an entropy estimator for ShanEn.
"""
import numpy as np

__all__ = ['extract_hrv_metrics']


def extract_hrv_metrics(signal, fs=100.0, method='elgendi'):
    """Compute clinical HRV metrics from a single predicted BVP waveform.

    :param signal: 1D numpy array (the continuous predicted BVP).
    :param fs: sampling rate in Hz.
    :param method: NeuroKit2 peak-detection method for ``ppg`` (e.g. 'elgendi').
    :returns: dict with keys rmssd_ms, pnn50, median_nn_ms, shannon_entropy
        (NaN-filled when NeuroKit2 is unavailable or peak finding fails).
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError(
            'Tier-3 clinical metrics require NeuroKit2 '
            '(`pip install neurokit2`).')

    result = {
        'rmssd_ms': np.nan,
        'pnn50': np.nan,
        'median_nn_ms': np.nan,
        'shannon_entropy': np.nan,
    }

    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.ndim != 1 or len(signal) < 2 * fs:
        return result   # too short / malformed

    try:
        ppg = nk.ppg_clean(signal, sampling_rate=fs, method='elgendi')
        _, info = nk.ppg_process(ppg, sampling_rate=fs, method=method)
        peaks = info['PPG_Peaks']
    except Exception:
        return result

    try:
        hrv = nk.hrv_time(peaks, sampling_rate=fs)
        col_map = {
            'HRV_RMSSD': 'rmssd_ms',
            'HRV_pNN50': 'pnn50',
            'HRV_MedianNN': 'median_nn_ms',
        }
        for col, key in col_map.items():
            if col in hrv.columns:
                result[key] = float(hrv[col].iloc[0])
    except Exception:
        pass

    # Shannon entropy of the RR-interval distribution (density-based estimate)
    try:
        rr = np.diff(np.asarray(peaks).astype(np.float64)) / fs * 1000.0
        rr = rr[~np.isnan(rr)]
        if len(rr) > 1:
            hist, _ = np.histogram(rr, bins='auto', density=True)
            hist = hist[hist > 0]
            result['shannon_entropy'] = float(-np.sum(hist * np.log(hist)))
    except Exception:
        pass

    return result
