"""The consumer's window contract for the respiration artefact (ADR-003).

Every consumer — the training dataset, the evaluation reference reader, any
analysis script — addresses windows through THIS module, so the ADR's window
rule is written once and cannot drift between two implementations:

    s = round(t_start * fs_out)        n = round(clip_seconds * fs_out)
    usable(s, n)  <=>  n > 0 and 0 <= s and s + n <= N_out
                      and mask[s : s + n].all()

Three deliberate properties:

* ``window_indices`` does NOT assert that the window fits. A window-generating
  loop must be able to probe the end of a session and reject cleanly instead of
  crashing on an ``AssertionError``; the fit is ``usable``'s business.
* ``usable`` checks the bounds BEFORE slicing. ``mask[s:s+n]`` truncates
  silently, so a window running past the end whose remaining samples happen to
  be ``True`` would otherwise be reported usable.
* ``read_window`` never returns a hole and never returns zeros: an invalid
  sample is ``NaN`` in the artefact and an unmasked read raises (ADR I4).

Clip length is bounded by the minimum usable run (ADR §2.4), so a consumer
requesting more than ``min_run_s`` is refused with the rebuild command rather
than served a shorter window.
"""
from typing import Any, Dict, Tuple

import numpy as np

__all__ = ['window_indices', 'usable', 'read_window', 'window_cause',
           'assert_clip_fits', 'verify_identity', 'slots_per_session']

#: Tolerance for comparing a requested clip length against ``min_run_s``. The
#: bound is a policy value, not a measured one, so the comparison is exact up to
#: float noise in a YAML round-trip (``8.0`` must pass, ``8.0001`` must not).
_CLIP_EPS = 1e-9


def window_indices(t_start: float, clip_seconds: float, fs_out: float
                   ) -> Tuple[int, int]:
    """``(s, n)`` for a window starting at ``t_start`` seconds.

    :raises ValueError: if ``clip_seconds`` is not positive (``n == 0`` would
        otherwise make ``mask[].all()`` vacuously true) or ``t_start`` is
        negative. Out-of-range windows do NOT raise here — see module docstring.
    """
    if not np.isfinite(t_start):
        raise ValueError(f'resp_window: t_start must be finite, got {t_start!r}')
    if t_start < 0:
        raise ValueError(f'resp_window: t_start must be >= 0, got {t_start!r}')
    if not np.isfinite(clip_seconds) or clip_seconds <= 0:
        raise ValueError(
            f'resp_window: clip_seconds must be > 0, got {clip_seconds!r}; an '
            f'empty window is never usable.')
    fs_out = float(fs_out)
    if not np.isfinite(fs_out) or fs_out <= 0:
        raise ValueError(f'resp_window: fs_out must be > 0, got {fs_out!r}')
    s = int(round(float(t_start) * fs_out))
    n = int(round(float(clip_seconds) * fs_out))
    if n <= 0:
        raise ValueError(
            f'resp_window: clip_seconds={clip_seconds!r} at fs_out={fs_out:g} '
            f'rounds to 0 samples; the clip is shorter than one output period.')
    return s, n


def usable(mask, s: int, n: int) -> bool:
    """Is the window at ``s`` of length ``n`` entirely valid and in range?

    Bounds first, slice second: a truncated all-``True`` slice must not be
    mistaken for a usable window.
    """
    m = np.asarray(mask)
    if m.ndim != 1:
        raise ValueError(f'resp_window: mask must be 1-D, got {m.shape}')
    if n <= 0:
        return False
    if s < 0 or s + n > m.size:
        return False
    return bool(m[s:s + n].all())


def read_window(source, s: int, n: int) -> np.ndarray:
    """Return ``signal[s : s+n]`` — raising rather than returning a hole.

    :param source: a shard mapping from :func:`data.resp_cache.read_shard`, or a
        bare signal array.
    :raises ValueError: if the window is out of bounds, if ``n <= 0``, or if any
        value in it is ``NaN`` (which means the window overlaps a hole and the
        caller forgot to test it with :func:`usable`).
    """
    if isinstance(source, dict):
        signal = np.asarray(source['signal'])
        mask = source.get('mask')
    else:
        signal = np.asarray(source)
        mask = None
    if mask is not None and not usable(mask, s, n):
        raise ValueError(
            f'resp_window: window [{s}, {s + n}) is not usable '
            f'(mask has a hole or the window runs past {np.asarray(mask).size}); '
            f'test readiness with usable() before reading.')
    if s < 0 or n <= 0 or s + n > signal.size:
        raise ValueError(
            f'resp_window: window [{s}, {s + n}) is outside [0, {signal.size})')
    seg = signal[s:s + n]
    if np.isnan(seg).any():
        raise ValueError(
            f'resp_window: window [{s}, {s + n}) contains NaN; an unmasked read '
            f'must fail loudly instead of training on a hole (ADR I4).')
    return seg


def window_cause(cause, s: int, n: int) -> int:
    """The most severe NON-ZERO cause code inside the window (0 = no defect).

    Used only for the per-rule drop accounting the runs have always logged; it
    never decides usability. Zeros are ignored, so a window that is dropped for
    one hole still reports that hole's rule rather than ``valid``; ties are
    broken by the ADR precedence (non-finite > rail > flatline > short run).
    """
    from . import resp_preprocess as rpp

    c = np.asarray(cause)
    if c.size == 0:
        return 0
    if n <= 0 or s < 0:
        return 0
    codes = np.unique(c[s:s + n])
    codes = codes[codes != rpp.CAUSE_VALID]
    if codes.size == 0:
        return int(rpp.CAUSE_VALID)
    return int(min(codes.tolist(),
                   key=lambda code: int(rpp.CAUSE_RANK[code])))


def assert_clip_fits(clip_seconds: float, params: Dict) -> None:
    """Refuse a clip longer than the artefact's minimum usable run.

    ADR §2.4: a window must fit inside one valid run, so clip-length flexibility
    is bounded by ``min_run_s``. The bound is in the artefact identity, so a
    consumer asking for more does not get "fewer windows" silently — it gets a
    rebuild instruction.
    """
    min_run_s = float(params['min_run_s'])
    if float(clip_seconds) > min_run_s + _CLIP_EPS:
        raise ValueError(
            f'resp_window: clip_seconds={float(clip_seconds):g} s exceeds the '
            f'artefact\'s min_run_s={min_run_s:g} s, so no window could fit '
            f'inside one valid run. Rebuild the artefact with '
            f'`min_run_s >= {float(clip_seconds):g}` '
            f'(`runners/run_build_resp_cache.py --init --min_run_s '
            f'{float(clip_seconds):g}`) or shorten the clip.')


def verify_identity(cache_root, expected_params: Dict) -> Any:
    """The only entry point a consumer may use to obtain an artefact root.

    Accepts either an ``out_root`` holding keyed candidate directories or a
    single keyed directory, and always verifies ``params.json`` against
    ``expected_params`` before returning (ADR I6).
    """
    from pathlib import Path

    from . import resp_cache as respc

    root = Path(cache_root)
    if (root / respc.PARAMS_NAME).is_file():
        respc.check_params_compatible(root, expected_params)
        return root
    return respc.find_cache(root, expected_params)


def slots_per_session(n_out: int, s_limit: int) -> int:
    """Convenience for callers that enumerate windows: count of starts ``0..s_limit``."""
    return max(int(s_limit) + 1, 0)
