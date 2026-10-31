"""Respiration session preprocessing: validity mask, per-run DC removal, resample.

Contract of record: ``code/notes/design/RESP_session_preprocess.md``.
Build order of record: ``code/notes/plan/RESP_session_preprocess_implementation_plan.md``.

This module is **pure** numpy/scipy: it does no IO and knows no paths. The writer
that turns its output into a shard lives in :mod:`data.resp_cache`.

WHAT THIS STAGE DECIDES
-----------------------
Given one continuous ``Resp_Volts.txt`` session at ``fs_in`` it answers exactly
one question -- *which volts are trustworthy at which instant* -- and emits the
DC-removed, anti-alias-resampled signal on a fixed output grid plus a per-sample
validity mask. It chooses **no** clip: window length, hop, split, task set and
normalisation stay downstream (D2). That is what lets ONE artefact serve Stage 2
(a 2.0 s hop) and Stage 3 (a 1.0 s hop).

FROZEN ARTEFACT CONTRACT (a copy of design §3; the design stays the authority)
-----------------------------------------------------------------------------
``params.json`` fields hashed into the cache directory name:

==========================  =======  ==========================================
field                       default  why it changes the stored bytes
==========================  =======  ==========================================
``format_version``          ``1``    shard schema
``corpus``                  digest   session identity set
``fs_in``                   ``1000``  source grid (nominal)
``fs_out``                  ``100``   output grid -- must equal the run's ``--fs``
``sat_v``                   ``9.95``  rail threshold (volts, magnitude)
``sat_min_s``               ``1.0``   a saturation event shorter than this is KEPT
``flat_window_s``           ``5.0``   flatline probe length
``flat_ptp_v``              ``0.3``   flatline peak-to-peak threshold
``min_run_s``               ``8.0``   valid runs shorter than this are holes
``resample``                ``poly``  filter shape -> the stored values
``mask_rule``               §3.4      conservative-support
``dtype`` / ``store``       ``f4``    representation
==========================  =======  ==========================================

Deliberately NOT in the key (they select or rescale, never change a stored
sample): ``clip_seconds``, ``clip_stride``, ``task_set``/``tasks``, the subject
split, ``physio_norm``, ``input_size``, ``max_entries`` and every ROI-side knob.

``min_run_s`` IS in the key, so the design's "any clip length" holds for
``clip_seconds <= min_run_s``; a consumer with a longer clip must rebuild.

GRID MAPPING (the consumer's contract)
--------------------------------------
::

    N_out = floor((N_src - 1) * fs_out / fs_in) + 1   # never reads past the file
    s     = round(t_start * fs_out) , n = round(clip_seconds * fs_out)
    usable(t_start) <=> mask[s : s + n].all()
    thermal frame f -> respiration sample round((f / fps) * fs_out)

MASK DERIVATION (all on the RAW grid; design §3.4)
--------------------------------------------------
1. ``nonfinite = ~isfinite(x)`` -> cause ``4``.
2. **Saturation is EVENT-based**: ``|x| > sat_v`` grouped into maximal
   contiguous runs; a run lasting ``>= sat_min_s`` invalidates ALL its samples
   (cause ``1``). A shorter contact is KEPT -- the behaviour change of design
   C1 (the shipped whole-window ``rail_touch_v`` test is retired on this path).
   A 1 s *total* rail fraction spread over many short grazes is NOT an event.
3. **Flatline**: a sample is invalid iff it lies in ANY ``flat_window_s`` window
   whose peak-to-peak spread is ``< flat_ptp_v`` (cause ``2``). Implemented as
   ``bad_start`` over every window start, dilated by the window length -- which
   is exactly "any window containing the sample".
4. **Runs**: maximal ``True`` stretches of the result; a run shorter than
   ``min_run_s`` becomes a hole (cause ``3``). Runs are an internal device, they
   are never stored.
5. **Per run**: ``y = x[run] - mean(x[run])`` (float64), then
   ``resample_poly`` onto the output grid after replicating the run's own edge
   values by ``RESAMPLE_HALF_TAPS`` -- a constant subtraction introduces no
   transient, so there is no edge guard and amplitude/shape survive.
6. **Conservative grid mask**: a grid sample is valid only if every source
   sample in its resampling support is valid AND in the same run, implemented as
   ``valid[k] = ~dilate(invalid)[round(k * fs_in / fs_out)]`` with the dilation
   radius ``fs_in / (2 * fs_out) + RESAMPLE_HALF_TAPS``. Without it a break
   shorter than one output period would vanish and a window could straddle it.
7. Invalid samples are ``NaN`` in the signal, never ``0``, so an unmasked read
   fails loudly. ``np.isnan(signal) == ~mask`` is asserted on the way out.

No high-pass filter, no detrend, no scaling: amplitude is preserved.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import resample_poly

#: Nominal source rate of ``Resp_Volts.txt`` (the corpus reads ~1001.8 Hz on
#: F001_T1; the frame<->sample mapping is a corpus-wide convention).
FS_IN = 1000.0
#: Output grid. Must equal the consuming run's ``--fs``.
FS_OUT = 100.0

#: A sample is invalid when it sits in a saturation EVENT at ``|V| > SAT_V``
#: lasting at least ``SAT_MIN_S`` seconds. Shorter contacts stay valid.
#: Design §2 (updated 2026-10-31): the alarm window is **1 s**.
SAT_V = 9.95
SAT_MIN_S = 1.0

#: A sample is invalid when ANY ``FLAT_WINDOW_S`` window containing it has a
#: peak-to-peak spread below ``FLAT_PTP_V``. Design §2 (updated 2026-10-31):
#: the threshold is **0.3 V** -- 6x the original 0.05 V, so a frozen channel AND
#: a channel that only rings a little are rejected, while real shallow breathing
#: (M058's reference std is ~0.12 V, i.e. ~0.3-0.5 V p-p) survives the probe.
#: A p-p probe cannot separate the two by construction; 0.3 V is the calibrated
#: compromise, not a decision the data supports on its own.
FLAT_WINDOW_S = 5.0
FLAT_PTP_V = 0.3

#: Valid runs shorter than this become holes (the design's clip length).
MIN_RUN_S = 8.0

#: Source samples replicated on each run edge before ``resample_poly``. The
#: default scipy filter has a ~10-input-sample radius at a 10x decimation, so 20
#: clears the transient without touching the run's interior.
RESAMPLE_HALF_TAPS = 20

#: Per-sample cause codes (stored in the shard's ``cause`` member).
CAUSE_VALID = 0
CAUSE_SAT = 1
CAUSE_FLAT = 2
CAUSE_SHORT = 3
CAUSE_NONFINITE = 4

CAUSE_NAMES = {
    CAUSE_VALID: 'valid',
    CAUSE_SAT: 'saturation',
    CAUSE_FLAT: 'flatline',
    CAUSE_SHORT: 'short_run',
    CAUSE_NONFINITE: 'nonfinite',
}

#: The resampler / mask rule names, part of the frozen parameter set.
RESAMPLE_RULE = 'poly'
MASK_RULE = 'conservative-support'


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def grid_length(n_src: int, fs_in: float = FS_IN,
                fs_out: float = FS_OUT) -> int:
    """Number of output samples that never read past the source file.

    ``N_out = floor((N_src - 1) * fs_out / fs_in) + 1`` (design §3.3).
    """
    n_src = int(n_src)
    if n_src <= 0:
        return 0
    return int(math.floor((n_src - 1) * float(fs_out) / float(fs_in))) + 1


def contiguous_runs(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    """Maximal ``True`` stretches of ``mask`` as half-open ``(start, end)``.

    Runs are an internal device of the design: they are never stored, only used
    to remove DC per run and to keep a window from straddling a break.
    """
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == 0:
        return []
    padded = np.concatenate(([False], m, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _window_ptp(x: np.ndarray, n: int) -> np.ndarray:
    """Peak-to-peak of the CONTIGUOUS window ``x[j : j + n]`` for every start.

    ``minimum_filter1d``/``maximum_filter1d`` with ``size=n`` centre the window
    at ``j + n // 2`` (verified against a brute-force ``np.ptp`` loop in
    ``tests/test_resp_preprocess.py``), so the per-start statistics are the
    central ``len(x) - n + 1`` entries. That is the one place an off-by-one
    would silently change the mask, hence the explicit pin.
    """
    n = int(n)
    if n <= 1 or x.size < n:
        return np.zeros(0, dtype=np.float64)
    off = n // 2
    lo = minimum_filter1d(x, size=n, mode='nearest')
    hi = maximum_filter1d(x, size=n, mode='nearest')
    return (hi - lo)[off: off + (x.size - n + 1)]


def _dilation_radius(fs_in: float, fs_out: float,
                     half_taps: int = RESAMPLE_HALF_TAPS) -> int:
    """Source samples whose invalidity taints one output sample (design §3.4.6)."""
    return int(math.ceil(float(fs_in) / (2.0 * float(fs_out)))) + int(half_taps)


def _dilate_bool(flags: np.ndarray, radius: int) -> np.ndarray:
    """``True`` where ANY flag lies within ``radius`` samples (edge-replicating).

    ``mode='nearest'`` deliberately treats the file boundary as a copy of the
    edge sample: a run that starts at sample 0 is not contaminated by the
    non-existent samples before it, so a session's first window is still usable.
    """
    r = int(radius)
    if r <= 0 or flags.size == 0:
        return np.asarray(flags, dtype=bool)
    grown = maximum_filter1d(np.asarray(flags, dtype=np.uint8),
                             size=2 * r + 1, mode='nearest')
    return grown.astype(bool)


# --------------------------------------------------------------------------- #
# 1-4: validity
# --------------------------------------------------------------------------- #
def validity_mask(x: Sequence[float], fs_in: float = FS_IN,
                  sat_v: float = SAT_V, sat_min_s: float = SAT_MIN_S,
                  flat_window_s: float = FLAT_WINDOW_S,
                  flat_ptp_v: float = FLAT_PTP_V,
                  min_run_s: float = MIN_RUN_S
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample validity of the RAW signal, plus the cause of each rejection.

    :param x: one-value-per-line respiration trace (any real dtype).
    :param fs_in: source sample rate (Hz).
    :returns: ``(mask [N] bool, cause [N] uint8)`` with ``mask = cause == 0``.
        ``True`` means usable.

    Rules, in this order (the ORDER fixes which rule gets credit when a sample
    fails several): non-finite -> saturation event -> flatline -> short run.
    """
    if float(fs_in) <= 0.0:
        raise ValueError(f'fs_in must be positive; got {fs_in!r}')
    if float(sat_v) < 0.0 or float(flat_ptp_v) < 0.0 or float(min_run_s) < 0.0:
        raise ValueError('sat_v, flat_ptp_v and min_run_s must be >= 0')
    xf = np.asarray(x, dtype=np.float64).ravel()
    n = int(xf.size)
    cause = np.zeros(n, dtype=np.uint8)
    if n == 0:
        return np.zeros(0, dtype=bool), cause

    # --- 1. non-finite ---------------------------------------------------
    finite = np.isfinite(xf)
    nonfinite = ~finite
    cause[nonfinite] = CAUSE_NONFINITE

    # --- 2. SATURATION EVENTS (maximal contiguous runs, NOT a rolling sum)
    # A NaN compares False, so it never counts as a rail contact.
    sat = np.abs(xf) > float(sat_v)
    min_sat = float(sat_min_s) * float(fs_in)
    for s, e in contiguous_runs(sat):
        if (e - s) >= min_sat:
            cause[s:e] = CAUSE_SAT

    # --- 3. FLATLINE: any flat window containing the sample invalidates it.
    # Non-finite samples are already invalid; feed 0.0 so a NaN cannot hide a
    # flat stretch in the surrounding windows.
    n_w = max(1, int(round(float(flat_window_s) * float(fs_in))))
    valid = cause == CAUSE_VALID
    if n > 0 and n_w <= n:
        probe = np.where(finite, xf, 0.0)
        bad_start = _window_ptp(probe, n_w) < float(flat_ptp_v)
        if bad_start.any():
            # A sample k lies in the window starting at j iff j in
            # [k - n_w + 1, k]; count bad starts in that range with a prefix
            # sum. The dilation is one-sided ON PURPOSE -- a symmetric window
            # would also taint samples AFTER a flat stretch, i.e. up to twice
            # its length, and would silently change the mask.
            cum = np.concatenate(([0], np.cumsum(bad_start.astype(np.int64))))
            k = np.arange(n)
            hi = np.minimum(k, bad_start.size - 1)
            lo = np.maximum(k - n_w + 1, 0)
            flat = (cum[hi + 1] - cum[lo]) > 0
            cause[valid & flat] = CAUSE_FLAT

    # --- 4. short runs become holes --------------------------------------
    valid = cause == CAUSE_VALID
    min_run = float(min_run_s) * float(fs_in)
    for s, e in contiguous_runs(valid):
        if (e - s) < min_run:
            cause[s:e] = CAUSE_SHORT

    return cause == CAUSE_VALID, cause


# --------------------------------------------------------------------------- #
# 5-7: per-run DC removal, resample, conservative grid mask
# --------------------------------------------------------------------------- #
def _rates_integral(fs_in: float, fs_out: float) -> Tuple[int, int]:
    """``fs_in``/``fs_out`` as integers, for ``resample_poly``.

    The nominal pair (1000, 100) is exact. A non-integral value is refused
    rather than silently rounded, because the grid mapping in §3.3 assumes an
    exact ratio.
    """
    fi, fo = float(fs_in), float(fs_out)
    if fi <= 0.0 or fo <= 0.0:
        raise ValueError(f'fs_in/fs_out must be positive; got {fs_in!r}/{fs_out!r}')
    ii, io = int(round(fi)), int(round(fo))
    if abs(fi - ii) > 1e-9 or abs(fo - io) > 1e-9:
        raise ValueError(
            f'fs_in and fs_out must be INTEGRAL for resample_poly; got '
            f'{fs_in!r}/{fs_out!r}')
    return ii, io


def resample_run(y: Sequence[float], fs_in: float, fs_out: float,
                 half_taps: int = RESAMPLE_HALF_TAPS,
                 n_out: Optional[int] = None) -> np.ndarray:
    """Anti-alias resample ONE run, with its own edge values replicated.

    :param y: the run's samples (already DC-removed), float.
    :param n_out: exact number of output samples to return. ``None`` ->
        ``round(len(y) * fs_out / fs_in)`` (the run's own length). A consumer
        filling a FIXED grid passes the count instead: rounding each run
        independently of the session grid formula
        ``floor((n_src - 1) * fs_out / fs_in) + 1`` is what used to leave the
        session's LAST grid sample unfilled whenever a run reached EOF (the
        leftover became ``NaN`` with cause 0 and could drop the final window).
    :returns: ``n_out`` output samples, float64.

    The padding is the design's "no transient, no edge guard" applied to the
    anti-alias filter: ``resample_poly`` pads with ZERO by default, which would
    create a step at both run edges. Replicating ``half_taps`` samples and then
    discarding the corresponding output samples keeps the run's interior exact
    and its edges clean.
    """
    yi, fo = _rates_integral(fs_in, fs_out)
    g = math.gcd(yi, fo)
    up, down = fo // g, yi // g
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return np.zeros(0, dtype=np.float64)
    count = (int(round(y.size * fo / yi)) if n_out is None else int(n_out))
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    if up == down:
        return np.resize(y, count)
    pad = max(0, int(half_taps))
    padded = np.pad(y, (pad, pad), mode='edge')
    out = resample_poly(padded, up, down)
    n_pad = int(round(pad * fo / yi))
    out = out[n_pad: n_pad + count]
    if out.size < count:
        # the replicated edges support `pad * fo/yi` extra samples on each
        # side, so this only trips for a count beyond that; extend with the
        # run's own last value rather than repeating the buffer
        out = np.concatenate([out, np.full(count - out.size, out[-1])])
    return out


def preprocess_session(x: Sequence[float], fs_in: float = FS_IN,
                       fs_out: float = FS_OUT,
                       sat_v: float = SAT_V, sat_min_s: float = SAT_MIN_S,
                       flat_window_s: float = FLAT_WINDOW_S,
                       flat_ptp_v: float = FLAT_PTP_V,
                       min_run_s: float = MIN_RUN_S,
                       half_taps: int = RESAMPLE_HALF_TAPS,
                       dtype: str = 'float32') -> Dict[str, object]:
    """Whole-session artefact payload (the shard members minus provenance).

    :param x: raw ``Resp_Volts.txt`` values at ``fs_in``.
    :returns: dict with ``signal [N_out]`` (float, ``NaN`` where invalid),
        ``mask [N_out] bool``, ``cause [N_out] uint8``, and the scalars
        ``n_src``, ``n_out``, ``fs_in``, ``fs_out``, ``n_runs``,
        ``valid_frac``, ``n_nonfinite``.

    Validity is decided ONCE on the raw signal (:func:`validity_mask`); nothing
    here re-tests a window. DC is removed PER RUN, never across a break.
    """
    fi, fo = _rates_integral(fs_in, fs_out)
    xf = np.asarray(x, dtype=np.float64).ravel()
    n_src = int(xf.size)
    n_out = grid_length(n_src, fi, fo)

    mask_src, cause_src = validity_mask(
        xf, fs_in=fi, sat_v=sat_v, sat_min_s=sat_min_s,
        flat_window_s=flat_window_s, flat_ptp_v=flat_ptp_v,
        min_run_s=min_run_s)

    # --- 6. conservative grid mask (before any per-run work) -------------
    radius = _dilation_radius(fi, fo, half_taps)
    tainted = _dilate_bool(~mask_src, radius)
    if n_out > 0:
        src_of = np.rint(np.arange(n_out, dtype=np.float64) * (fi / fo))
        src_of = np.clip(src_of.astype(np.int64), 0, max(0, n_src - 1))
        grid_mask = ~tainted[src_of]
        # attribute the cause of a dilated hole to the nearest invalid sample
        cause_dil = maximum_filter1d(cause_src, size=2 * radius + 1,
                                     mode='nearest')
        grid_cause = np.where(cause_src[src_of] != 0,
                              cause_src[src_of], cause_dil[src_of])
    else:
        src_of = np.zeros(0, dtype=np.int64)
        grid_mask = np.zeros(0, dtype=bool)
        grid_cause = np.zeros(0, dtype=np.uint8)

    # --- 5. per-run DC removal + resample onto the grid -------------------
    signal = np.full(n_out, np.nan, dtype=np.float64)
    runs = contiguous_runs(mask_src)
    for s, e in runs:
        seg = xf[s:e]
        # mean in float64 over the run's own samples (all finite by the mask)
        seg = seg - float(seg.mean())
        # Fill EVERY grid sample whose instant lies inside this run. The session
        # grid length is floor((n_src-1)*fo/fi)+1, i.e. NOT round(n_src*fo/fi),
        # so the last run must reach n_out explicitly -- the +1 is the safe
        # margin that keeps k0's rounding from leaving an interior sample empty.
        k0 = int(round(s * fo / fi))
        want = min(n_out - k0,
                   int(math.ceil((e - s) * fo / fi)) + 1)
        out = resample_run(seg, fi, fo, half_taps=half_taps, n_out=want)
        k1 = min(n_out, k0 + out.size)
        if k1 > k0:
            signal[k0:k1] = out[: k1 - k0]

    # --- 7. invalid == NaN, asserted on the way out -----------------------
    grid_mask = grid_mask & np.isfinite(signal)
    signal[~grid_mask] = np.nan
    if not np.array_equal(np.isnan(signal), ~grid_mask):
        raise AssertionError(
            'resp_preprocess: signal/mask mismatch -- invalid samples must be '
            'NaN and every NaN must be invalid')

    out_dtype = np.dtype(dtype)
    signal_out = signal.astype(out_dtype)
    n_valid = int(grid_mask.sum())
    return {
        'signal': signal_out,
        'mask': grid_mask,
        'cause': grid_cause.astype(np.uint8),
        'n_src': n_src,
        'n_out': int(n_out),
        'fs_in': int(fi),
        'fs_out': int(fo),
        'n_runs': len(runs),
        'valid_frac': (float(n_valid) / float(n_out)) if n_out else 0.0,
        'n_valid': n_valid,
        'n_nonfinite': int((~np.isfinite(xf)).sum()),
        'sat_v': float(sat_v),
        'sat_min_s': float(sat_min_s),
        'flat_window_s': float(flat_window_s),
        'flat_ptp_v': float(flat_ptp_v),
        'min_run_s': float(min_run_s),
        'half_taps': int(half_taps),
    }


__all__ = [
    'FS_IN', 'FS_OUT', 'SAT_V', 'SAT_MIN_S', 'FLAT_WINDOW_S', 'FLAT_PTP_V',
    'MIN_RUN_S', 'RESAMPLE_HALF_TAPS', 'RESAMPLE_RULE', 'MASK_RULE',
    'CAUSE_VALID', 'CAUSE_SAT', 'CAUSE_FLAT', 'CAUSE_SHORT', 'CAUSE_NONFINITE',
    'CAUSE_NAMES', 'grid_length', 'contiguous_runs', 'validity_mask',
    'resample_run', 'preprocess_session',
]
