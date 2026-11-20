"""Blood-pressure session preprocessing — the whole-session artefact (ADR-004).

One pure function turns one raw ``BP_mmHg.txt`` trace into one artefact payload.
It decides *which mmHg samples are trustworthy at which instant* and nothing
else:

* validity is decided **once**, per sample, on the raw trace, before any offset
  removal (ADR §2.2 step 1), and the emitted mask is the only statement of
  validity;
* the three gates are applied in the fixed order **1 → 2 → 3**, each to the
  previous gate's survivors only, so a rejected sample is never re-evaluated and
  carries exactly one cause (ADR §2.4);
* maximal continuous stretches of admitted samples form **segments**; a segment
  shorter than 4.0 s is invalid (ADR §2.5);
* each valid segment's own arithmetic mean is removed — one constant per
  segment, no filter, no detrend, no rescale (ADR §2.2 step 4);
* the result is stored on the **unchanged 1000 Hz source grid**, with every
  invalid sample ``NaN``, never ``0`` (ADR §2.1).

No clip length, hop, stride, split, task subset or normalisation may enter this
module (ADR §2.8, Criterion 10). If a function here takes one of those, it is a
bug.

Build order of record: ``code/notes/plan/ADR-004-bp-session-cache-plan.md``.

--------------------------------------------------------------------------------
FROZEN CONTRACT (code-side copy of ADR §2.6/§2.1/§2.3–§2.5; the ADR is authority)
--------------------------------------------------------------------------------
Identity fields (hashing these is what makes a stale artefact impossible to
reuse silently)::

    format_version  corpus  fs_in  fs_out  gate1_min_mmhg  gate1_max_mmhg
    flatline_max_ms  rail_dwell_min_ms  min_segment_s  store

Never in the identity, never in a shard (ADR §2.8): ``clip_seconds``,
``clip_stride``, the subject split or fold, ``task_set`` / ``task_groups``,
every normalisation statistic, ``input_size``, ``max_entries`` and per-clip
offsets.

Shard members (ADR §2.1)::

    signal           float32 [N]  segment-mean-removed mmHg; NaN iff mask is False
    mask             bool    [N]  True = usable; the sole authority
    cause            uint8   [N]  0 valid, 1 non-finite, 2 range, 3 flatline,
                                  4 rail, 5 short segment
    seg_start        int64   [S]  valid segment start, source index
    seg_stop         int64   [S]  valid segment stop, exclusive
    seg_mean_mmhg    float64 [S]  mean removed from that segment
    seg_seconds      float64 [S]  segment length in seconds
    n_src, fs_in, fs_out          grid identity, self-verifying without the raw file
    n_segments, valid_frac, n_nonfinite   build statistics
    session, source_file          provenance

Grid arithmetic (ADR §2.7 C4) — ``fs_out == fs_in``, so grid index == source
index and no resampling, phase alignment or edge handling exists at all::

    N = N_src
    s = round(t_start * fs_out)      n = round(clip_seconds * fs_out)
    usable(t_start)  <=>  n > 0 and 0 <= s and s + n <= N
                         and mask[s : s + n].all()
"""
import numpy as np

__all__ = [
    'FS_IN', 'FS_OUT', 'GATE1_MIN_MMHG', 'GATE1_MAX_MMHG', 'FLATLINE_MAX_MS',
    'RAIL_DWELL_MIN_MS', 'MIN_SEGMENT_S', 'STORE', 'STORE_DTYPE',
    'CAUSE_VALID', 'CAUSE_NONFINITE', 'CAUSE_RANGE', 'CAUSE_FLATLINE',
    'CAUSE_RAIL', 'CAUSE_SHORT', 'CAUSE_NAMES',
    'contiguous_runs', 'equal_runs', 'gate1_range', 'gate2_flatline',
    'gate3_rail', 'admitted_mask', 'preprocess_session',
]

# --------------------------------------------------------------------------- #
# frozen quantities — one constant per identity field, so the key cannot drift
# --------------------------------------------------------------------------- #
#: Nominal source and output rate, in Hz. Both are **integers**: a float spelling
#: would serialise as ``1000.0`` and produce a different ``params.json`` digest
#: (and therefore a different cache key) for the same artefact.
FS_IN = 1000
FS_OUT = 1000

GATE1_MIN_MMHG = 30.0     #: Gate 1 lower bound, mmHg
GATE1_MAX_MMHG = 250.0    #: Gate 1 upper bound, mmHg
FLATLINE_MAX_MS = 200     #: Gate 2: a run strictly LONGER than this is rejected
RAIL_DWELL_MIN_MS = 30    #: Gate 3: a dwell AT LEAST this long is rejected
MIN_SEGMENT_S = 4.0       #: a shorter segment is invalid

#: The ``store`` identity field, as ONE frozen string. A sub-dict would let two
#: spellings of the same schema hash differently.
STORE = 'signal+mask+cause+segments:float32,bool,uint8,int64,float64'
STORE_DTYPE = 'float32'

# --------------------------------------------------------------------------- #
# cause codes — the gate order IS the precedence (ADR §2.4)
# --------------------------------------------------------------------------- #
CAUSE_VALID = 0
CAUSE_NONFINITE = 1
CAUSE_RANGE = 2
CAUSE_FLATLINE = 3
CAUSE_RAIL = 4
CAUSE_SHORT = 5

#: ``gate1_nonfinite`` / ``gate1_range`` / ``gate2_flatline`` / ``gate3_rail`` /
#: ``short_segment`` are the ADR's LABELS for these codes, not a second scale.
CAUSE_NAMES = {
    CAUSE_VALID: 'valid',
    CAUSE_NONFINITE: 'gate1_nonfinite',
    CAUSE_RANGE: 'gate1_range',
    CAUSE_FLATLINE: 'gate2_flatline',
    CAUSE_RAIL: 'gate3_rail',
    CAUSE_SHORT: 'short_segment',
}


# --------------------------------------------------------------------------- #
# run bookkeeping
# --------------------------------------------------------------------------- #
def contiguous_runs(mask) -> list:
    """Maximal ``True`` stretches of a 1-D boolean array as ``[(start, stop)]``.

    ``stop`` is exclusive. Empty / all-``False`` input yields ``[]``.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise ValueError(f'bp_preprocess: contiguous_runs expects 1-D, got {m.shape}')
    if m.size == 0 or not m.any():
        return []
    padded = np.concatenate(([False], m, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def _run_index(xs, alive):
    """Run index of every sample: equal, trace-adjacent, and survivor-only.

    A run is broken at index ``i`` when any of these holds:

    * ``i == 0``;
    * ``xs[i] != xs[i-1]`` (the value changed);
    * ``i`` or ``i-1`` is not alive (a rejected sample both breaks the run and
      is never compared to its neighbour).

    :param xs: the trace **sanitised** by the caller (``np.where(alive, x, 0)``),
        so no comparison ever sees a ``NaN`` — an unmasked comparison would emit
        ``RuntimeWarning: invalid value encountered in not_equal``.
    :returns: ``(run_id, starts, lengths, values)`` where ``run_id`` assigns every
        sample to a run, ``starts``/``lengths``/``values`` describe the runs that
        are wholly alive, in index order.
    """
    n = xs.size
    eq = np.zeros(n, dtype=bool)
    if n > 1:
        eq[1:] = (xs[1:] == xs[:-1]) & alive[1:] & alive[:-1]
    break_at = np.zeros(n, dtype=bool)
    break_at[0] = True
    if n > 1:
        break_at[1:] = ~eq[1:] | ~alive[1:] | ~alive[:-1]
    run_id = np.cumsum(break_at) - 1
    starts = np.flatnonzero(break_at)
    lengths = np.bincount(run_id)
    keep = alive[starts]                     # runs wholly alive by construction
    return run_id, starts[keep], lengths[keep], xs[starts[keep]]


def equal_runs(x, alive) -> list:
    """Maximal runs of exactly-equal consecutive samples inside ``alive`` blocks.

    Public form of :func:`_run_index`, used by Gate 2 and Gate 3 and pinned
    against a brute-force run-length loop in the tests: this is the one place an
    off-by-one silently changes the mask.
    """
    xs = np.where(np.asarray(alive, dtype=bool), np.asarray(x), 0.0)
    _, starts, lengths, _ = _run_index(xs, np.asarray(alive, dtype=bool))
    return [(int(a), int(a + ln)) for a, ln in zip(starts, lengths)]


def _span_mask(starts, lengths, n) -> np.ndarray:
    """Boolean array marking every sample covered by any ``[start, start+len)``."""
    if starts.size == 0:
        return np.zeros(n, dtype=bool)
    delta = (np.bincount(starts, minlength=n + 1)[:n + 1]
             - np.bincount(starts + lengths, minlength=n + 1)[:n + 1])
    return np.cumsum(delta[:n]) > 0


# --------------------------------------------------------------------------- #
# the three gates, in order
# --------------------------------------------------------------------------- #
def _as_trace(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(
            f'bp_preprocess: expected a 1-D trace, got shape {arr.shape}')
    if arr.size == 0:
        raise ValueError('bp_preprocess: empty trace')
    return arr.astype(np.float64, copy=False)


def gate1_range(x, min_mmhg=GATE1_MIN_MMHG, max_mmhg=GATE1_MAX_MMHG):
    """Gate 1 — dynamic range, **per sample** → ``(alive, cause)``.

    Cause ``1`` for a non-finite sample, cause ``2`` for an out-of-range one.
    The non-finite test runs first because comparisons with ``NaN`` are false and
    would otherwise pass the range test. Neighbours are never affected:
    ``29.99`` is rejected and ``30.00`` admitted; ``250.00`` is admitted and
    ``250.01`` rejected.
    """
    x = _as_trace(x)
    finite = np.isfinite(x)
    xf = np.where(finite, x, 0.0)            # no comparison ever sees a NaN
    cause = np.zeros(x.size, dtype=np.uint8)
    cause[~finite] = CAUSE_NONFINITE
    cause[finite & ((xf < float(min_mmhg)) | (xf > float(max_mmhg)))] = CAUSE_RANGE
    return cause == CAUSE_VALID, cause


def gate2_flatline(x, alive, flatline_max_ms=FLATLINE_MAX_MS, fs_in=FS_IN):
    """Gate 2 — flatline, on the Gate-1 survivors → ``(alive2, rejected)``.

    Any invariant run **strictly longer** than ``flatline_max_ms`` is rejected in
    full. At 1 kHz that is ``> 200`` samples: 201 ms is rejected, 200 ms kept.
    The strict comparison is the ADR's wording and is deliberately *not* the same
    operator as Gate 3's ``>=``.
    """
    alive = np.asarray(alive, dtype=bool)
    xs = np.where(alive, np.asarray(x, dtype=np.float64), 0.0)
    _, starts, lengths, _ = _run_index(xs, alive)
    threshold = float(flatline_max_ms) * float(fs_in) / 1000.0
    bad = lengths > threshold
    rejected = _span_mask(starts[bad], lengths[bad], alive.size)
    return alive & ~rejected, rejected


def gate3_rail(x, alive, rail_dwell_min_ms=RAIL_DWELL_MIN_MS, fs_in=FS_IN):
    """Gate 3 — amplifier rail, on the Gate-2 survivors → ``(alive3, rejected)``.

    The extrema are recomputed **on the survivors**, never on the raw trace: a
    ``300 mmHg`` spike removed by Gate 1 must not define ``y_max``. A dwell is a
    maximal equal run whose value equals ``y_max`` or ``y_min``; it is rejected
    when **at least** ``rail_dwell_min_ms`` long (30 ms rejected, 29 ms kept).
    An interior plateau is never a dwell site, and ``y_max == y_min`` is one
    dwell, not two.

    A cascade that reaches this gate with no survivor returns without any
    reduction: ``x[alive].max()`` on an empty selection raises ``ValueError:
    zero-size array to reduction operation maximum which has no identity``, which
    would turn a fully-dropped session — an expected corpus case — into a build
    failure.
    """
    alive = np.asarray(alive, dtype=bool)
    n = alive.size
    if not alive.any():
        return alive.copy(), np.zeros(n, dtype=bool)
    x = np.asarray(x, dtype=np.float64)
    xs = np.where(alive, x, 0.0)
    y_max = float(xs[alive].max())
    y_min = float(xs[alive].min())
    _, starts, lengths, values = _run_index(xs, alive)
    threshold = float(rail_dwell_min_ms) * float(fs_in) / 1000.0
    bad = (lengths >= threshold) & ((values == y_max) | (values == y_min))
    rejected = _span_mask(starts[bad], lengths[bad], n)
    return alive & ~rejected, rejected


def admitted_mask(x, fs_in=FS_IN, gate1_min_mmhg=GATE1_MIN_MMHG,
                  gate1_max_mmhg=GATE1_MAX_MMHG,
                  flatline_max_ms=FLATLINE_MAX_MS,
                  rail_dwell_min_ms=RAIL_DWELL_MIN_MS):
    """Apply Gates 1 → 2 → 3 in order → ``(admitted, cause)``.

    The order is the precedence: each gate sees only the previous gate's
    survivors, a rejected sample is never re-evaluated, and it carries the cause
    of the gate that rejected it.
    """
    alive1, cause = gate1_range(x, gate1_min_mmhg, gate1_max_mmhg)
    alive2, flat = gate2_flatline(x, alive1, flatline_max_ms, fs_in)
    cause[flat] = CAUSE_FLATLINE
    alive3, rail = gate3_rail(x, alive2, rail_dwell_min_ms, fs_in)
    cause[rail] = CAUSE_RAIL
    return alive3, cause


# --------------------------------------------------------------------------- #
# the artefact payload
# --------------------------------------------------------------------------- #
def _column(values, dtype) -> np.ndarray:
    """Fixed-dtype column, so an EMPTY segment table keeps its schema.

    ``np.array([])`` would silently make ``seg_start``/``seg_stop`` ``float64``
    and break the shard contract for exactly the sessions that have no segment.
    """
    if len(values):
        return np.asarray(values, dtype=dtype)
    return np.empty(0, dtype=dtype)


def preprocess_session(x, fs_in=FS_IN, fs_out=FS_OUT,
                       gate1_min_mmhg=GATE1_MIN_MMHG,
                       gate1_max_mmhg=GATE1_MAX_MMHG,
                       flatline_max_ms=FLATLINE_MAX_MS,
                       rail_dwell_min_ms=RAIL_DWELL_MIN_MS,
                       min_segment_s=MIN_SEGMENT_S,
                       dtype=STORE_DTYPE):
    """Clean one session into its whole-session payload.

    Pure: no IO, no config object, no globals. The result is everything a shard
    stores except provenance, and the mask is the ONLY validity statement it
    makes.

    :param x: the raw trace, one mmHg value per sample at the nominal ``fs_in``.
    :param fs_out: must equal ``fs_in`` — this artefact applies no resampling
        (ADR §2.2 step 5), so a full-rate grid is the only supported geometry.
    :returns: dict with ``signal`` (``float32 [N]``, ``NaN`` where invalid),
        ``mask`` (``bool [N]``), ``cause`` (``uint8 [N]``), the four segment
        columns, and the scalars ``n_src``, ``fs_in``, ``fs_out``,
        ``n_segments``, ``valid_frac``, ``n_nonfinite``.
    """
    x = _as_trace(x)
    fs_in = float(fs_in)
    fs_out = float(fs_out)
    if fs_out != fs_in:
        raise ValueError(
            f'bp_preprocess: fs_out={fs_out:g} Hz != fs_in={fs_in:g} Hz. This '
            f'artefact applies no resampling (ADR §2.2 step 5), so the grid must '
            f'BE the source grid.')
    if str(dtype) != STORE_DTYPE:
        raise ValueError(
            f'bp_preprocess: store dtype {dtype!r} is not the frozen '
            f'{STORE_DTYPE!r}; a different representation is a different artefact.')

    n = int(x.size)
    mask, cause = admitted_mask(
        x, fs_in=fs_in, gate1_min_mmhg=gate1_min_mmhg,
        gate1_max_mmhg=gate1_max_mmhg, flatline_max_ms=flatline_max_ms,
        rail_dwell_min_ms=rail_dwell_min_ms)

    # --- segments: maximal admitted stretches; the 4 s rule is a segment rule
    min_len = int(round(float(min_segment_s) * fs_in))
    segments = contiguous_runs(mask)
    for start, stop in segments:
        if (stop - start) < min_len:
            mask[start:stop] = False
            cause[start:stop] = CAUSE_SHORT
    valid = [(a, b) for a, b in segments if (b - a) >= min_len]

    # --- per-segment DC removal, at absolute source positions
    signal = np.full(n, np.nan, dtype=np.float64)
    seg_start, seg_stop, seg_mean, seg_seconds = [], [], [], []
    for start, stop in valid:
        y = x[start:stop]
        mean = float(y.mean())
        signal[start:stop] = y - mean
        seg_start.append(start)
        seg_stop.append(stop)
        seg_mean.append(mean)
        seg_seconds.append((stop - start) / fs_in)

    signal = signal.astype(dtype, copy=False)

    # --- invariants: the writer asserts, it never repairs
    if not np.array_equal(np.isnan(signal), ~mask):
        raise AssertionError(
            'bp_preprocess: emitted signal/mask disagree — an invalid sample '
            'must be NaN and every valid one must be filled.')
    if not np.array_equal(mask, cause == CAUSE_VALID):
        raise AssertionError(
            'bp_preprocess: mask and cause disagree — `mask[k] is False` must '
            'imply `cause[k] != 0`.')

    return {
        'signal': signal,
        'mask': mask,
        'cause': cause,
        'seg_start': _column(seg_start, np.int64),
        'seg_stop': _column(seg_stop, np.int64),
        'seg_mean_mmhg': _column(seg_mean, np.float64),
        'seg_seconds': _column(seg_seconds, np.float64),
        'n_src': n,
        'fs_in': int(round(fs_in)),
        'fs_out': int(round(fs_out)),
        'n_segments': len(valid),
        'valid_frac': float(mask.mean()) if n else 0.0,
        'n_nonfinite': int((~np.isfinite(x)).sum()),
    }
