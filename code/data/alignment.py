"""Temporal registration / resampling between the asynchronous sensors.

Handles the fact that RGB, TIR and the 1D physiological streams each live on
their own clock / grid (all nominally 25 fps for the videos; the signals are
sampled at ``fs``). These helpers convert between seconds, frame indices and
sample indices, and resample 1D signals to a common grid.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'frame_indices', 'sample_indices', 'slice_1d', 'resample_1d',
    'available_duration', 'plan_clip', 'frame_indices_at_target_rate',
]

_EPS = 1e-9


def frame_indices(t_start_s: float, duration_s: float, fps: float) -> Tuple[int, int]:
    """First frame index and frame count covering a [t_start, t_start+dur] window."""
    start = int(round(t_start_s * fps))
    n = max(1, int(round(duration_s * fps)))
    return start, n


def sample_indices(t_start_s: float, duration_s: float, fs: float) -> Tuple[int, int]:
    """First sample index and sample count for a 1D stream sampled at ``fs``."""
    start = int(round(t_start_s * fs))
    n = max(1, int(round(duration_s * fs)))
    return start, n


def slice_1d(x: np.ndarray, fs: float, t_start_s: Optional[float] = None,
             start: Optional[int] = None, n: Optional[int] = None) -> np.ndarray:
    """Slice a 1D signal by time or by (start, count); clamps to length."""
    x = np.asarray(x).reshape(-1)
    if t_start_s is not None:
        s, nn = sample_indices(t_start_s, 1.0, fs)
        s = max(0, s)
        start = s
        n = min(x.size - s, n or int(round(fs)))
    else:
        start = start or 0
        n = n if n is not None else x.size - start
    start = max(0, min(start, x.size))
    end = min(x.size, start + n)
    return x[start:end]


def resample_1d(x: np.ndarray, fs_in: float, fs_out: float,
                length: Optional[int] = None) -> np.ndarray:
    """Resample a 1D signal to ``fs_out`` (or to an exact ``length``)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x
    n_out = length if length else max(1, int(round(len(x) * fs_out / fs_in)))
    if n_out == len(x):
        return x.copy()
    old_axis = np.linspace(0, len(x) - 1, num=len(x))
    new_axis = np.linspace(0, len(x) - 1, num=n_out)
    return np.interp(new_axis, old_axis, x).astype(np.float32)


def available_duration(durations_s: Sequence[float]) -> float:
    """Overlap window common to all streams (seconds)."""
    return float(min(d for d in durations_s if d and d > 0))


def frame_indices_at_target_rate(t_start_s: float, duration_s: float,
                                 fps_src: float, fps_target: float,
                                 temporal_stride: int = 1) -> np.ndarray:
    """Source-frame indices covering a window, decimated to a target rate.

    Needed when RGB and TIR run at different fps: both modalities are read on
    the SAME time grid of ``fps_target`` frames, so the outputs have equal T.

    ``temporal_stride`` > 1 keeps one common-grid frame every that many frames
    (e.g. 2 -> 12.5 fps at a nominal 25 fps) while STILL mapping to ``fps_src``
    source indices, so the decimation happens before any image is decoded.
    """
    stride = max(1, int(temporal_stride))
    n_target = max(1, int(round(duration_s * fps_target / stride)))
    t_axis = t_start_s + np.arange(n_target) * stride / fps_target
    idx = np.floor(t_axis * fps_src + _EPS).astype(np.int64)
    return idx


def plan_clip(t_start_s: float, duration_s: float, fps_rgb: float,
              fps_tir: float, fs: float,
              temporal_stride: int = 1) -> Dict[str, dict]:
    """Return per-stream read plans for a clip on a common time axis.

    Uses the lower of the two video rates as the common grid, so RGB and TIR
    always yield the same number ``T`` of frames for a clip.

    ``temporal_stride`` > 1 decimates that common grid *inside* the window by
    keeping one frame every ``temporal_stride`` frames (e.g. 2 -> 12.5 fps at
    the nominal 25 fps), so both videos still return the same reduced ``T``.
    The 1-D signal window is NOT decimated (it stays at ``fs``).
    """
    fps_target = min(fps_rgb, fps_tir)
    stride = max(1, int(temporal_stride))
    n_target = max(1, int(round(duration_s * fps_target / stride)))
    rgb_idx = frame_indices_at_target_rate(t_start_s, duration_s, fps_rgb,
                                           fps_target, stride)
    tir_idx = frame_indices_at_target_rate(t_start_s, duration_s, fps_tir,
                                           fps_target, stride)
    sig_start, sig_n = sample_indices(t_start_s, duration_s, fs)
    return {
        'rgb': {'indices': rgb_idx, 'n': n_target,
                'fps': fps_target / stride},
        'tir': {'indices': tir_idx, 'n': n_target,
                'fps': fps_target / stride},
        'signal': {'start': sig_start, 'n': sig_n},
    }
