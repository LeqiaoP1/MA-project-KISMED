"""The consumer's window contract for the blood-pressure artefact (ADR-004).

Every consumer — the training dataset, the evaluation reference reader, any
analysis script — addresses windows through THIS module, so the ADR's window
rule is written once and cannot drift between two implementations:

    s = round(t_start * fs_out)        n = round(clip_seconds * fs_out)
    usable(s, n)  <=>  n > 0 and 0 <= s and s + n <= N
                      and mask[s : s + n].all()

Four deliberate properties:

* ``window_indices`` does NOT assert that the window fits. A window-generating
  loop must be able to probe the end of a session and reject cleanly instead of
  crashing on an ``AssertionError``; the fit is ``usable``'s business.
* ``usable`` checks the bounds BEFORE slicing. ``mask[s:s+n]`` truncates
  silently, so a window running past the end whose remaining samples happen to be
  ``True`` would otherwise be reported usable.
* ``read_window`` never returns a hole and never returns zeros: an invalid
  sample is ``NaN`` in the artefact and an unmasked read raises.
* there is NO ``clip_seconds <= min_segment_s`` refusal. The clip is a consumer
  choice (ADR §2.7 C7); a clip longer than the 4 s minimum is a legal request
  that simply finds no usable window where no segment is long enough.
"""
from typing import Any, Dict, Tuple

import numpy as np

__all__ = ['window_indices', 'usable', 'read_window', 'segment_of',
           'verify_identity']


def window_indices(t_start: float, clip_seconds: float, fs_out: float
                   ) -> Tuple[int, int]:
    """``(s, n)`` for a window starting at ``t_start`` seconds.

    :raises ValueError: if ``clip_seconds`` is not positive (``n == 0`` would
        make ``mask[].all()`` vacuously true), if ``t_start`` is negative, or if
        ``fs_out`` is not positive. Out-of-range windows do NOT raise here.
    """
    if not np.isfinite(t_start):
        raise ValueError(f'bp_windows: t_start must be finite, got {t_start!r}')
    if t_start < 0:
        raise ValueError(f'bp_windows: t_start must be >= 0, got {t_start!r}')
    if not np.isfinite(clip_seconds) or clip_seconds <= 0:
        raise ValueError(
            f'bp_windows: clip_seconds must be > 0, got {clip_seconds!r}; an '
            f'empty window is never usable.')
    fs_out = float(fs_out)
    if not np.isfinite(fs_out) or fs_out <= 0:
        raise ValueError(f'bp_windows: fs_out must be > 0, got {fs_out!r}')
    s = int(round(float(t_start) * fs_out))
    n = int(round(float(clip_seconds) * fs_out))
    if n <= 0:
        raise ValueError(
            f'bp_windows: clip_seconds={clip_seconds!r} at fs_out={fs_out:g} '
            f'rounds to 0 samples; the clip is shorter than one output period.')
    return s, n


def usable(mask, s: int, n: int) -> bool:
    """Is the window at ``s`` of length ``n`` entirely valid and in range?

    Bounds first, slice second: a truncated all-``True`` slice must not be
    mistaken for a usable window.
    """
    m = np.asarray(mask)
    if m.ndim != 1:
        raise ValueError(f'bp_windows: mask must be 1-D, got {m.shape}')
    if n <= 0:
        return False
    if s < 0 or s + n > m.size:
        return False
    return bool(m[s:s + n].all())


def read_window(shard, s: int, n: int) -> np.ndarray:
    """Return ``signal[s : s+n]`` — raising rather than returning a hole.

    :param shard: a shard mapping from :func:`data.bp_artefact.read_shard`, or a
        bare signal array.
    :raises ValueError: if the window is not usable (out of bounds, ``n <= 0``,
        or overlapping a hole), or if any value in it is ``NaN``.
    """
    if isinstance(shard, dict):
        signal = np.asarray(shard['signal'])
        mask = shard.get('mask')
    else:
        signal = np.asarray(shard)
        mask = None
    if mask is not None and not usable(mask, s, n):
        raise ValueError(
            f'bp_windows: window [{s}, {s + n}) is not usable (mask has a hole '
            f'or the window runs past {np.asarray(mask).size}); test readiness '
            f'with usable() before reading.')
    if s < 0 or n <= 0 or s + n > signal.size:
        raise ValueError(
            f'bp_windows: window [{s}, {s + n}) is outside [0, {signal.size})')
    seg = signal[s:s + n]
    if np.isnan(seg).any():
        raise ValueError(
            f'bp_windows: window [{s}, {s + n}) contains NaN; an unmasked read '
            f'must fail loudly instead of training on a hole.')
    return seg


def segment_of(shard, s: int, n: int) -> int:
    """Index of the ONE valid segment a usable window lies in (ADR §2.7 C3).

    Contract, in this order:

    * ``usable(shard['mask'], s, n)`` is ``False`` → **``ValueError``**: a window
      that is out of bounds, empty, or touches any masked sample (a short
      segment, a flatline, a rail dwell) has no segment, and that is a caller
      error the caller can fix by testing ``usable`` first;
    * otherwise the index of the one row whose ``[seg_start, seg_stop)`` contains
      ``[s, s + n)``;
    * **``AssertionError``** if no row contains it, or if two do — a usable
      window outside every valid segment, or spanning two, means the mask and the
      segment table disagree, which is a shard defect rather than a caller error.
    """
    mask = np.asarray(shard['mask'])
    if not usable(mask, s, n):
        raise ValueError(
            f'bp_windows: window [{s}, {s + n}) is not usable, so it lies in no '
            f'valid segment; test usable() first.')
    start = np.asarray(shard['seg_start'], dtype=np.int64)
    stop = np.asarray(shard['seg_stop'], dtype=np.int64)
    hits = np.flatnonzero((start <= s) & (s + n <= stop))
    if hits.size != 1:
        raise AssertionError(
            f'bp_windows: a usable window [{s}, {s + n}) is contained in '
            f'{hits.size} valid segments (expected exactly 1) — the mask and the '
            f'segment table disagree, so the shard is corrupt.')
    return int(hits[0])


def verify_identity(cache_root, expected_params: Dict) -> Any:
    """The only entry point a consumer may use to obtain an artefact root.

    Accepts either an ``out_root`` holding keyed candidate directories or a
    single keyed directory, and always verifies ``params.json`` against
    ``expected_params`` before returning (ADR §2.7 C6).
    """
    from pathlib import Path

    try:                                  # inside the ``data`` package
        from . import bp_artefact as bpa
    except ImportError:                   # loaded standalone (``code/data`` on path)
        import bp_artefact as bpa         # noqa: F401

    root = Path(cache_root)
    if (root / bpa.PARAMS_NAME).is_file():
        bpa.check_params_compatible(root, expected_params)
        return root
    return bpa.find_cache(root, expected_params)
