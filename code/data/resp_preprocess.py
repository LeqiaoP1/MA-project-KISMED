"""Respiration session preprocessing — the whole-session grid artefact (ADR-003).

One pure function turns one raw trace into one artefact payload. It decides
*which volts are trustworthy at which instant* and nothing else:

* validity is decided ONCE, on raw volts, before any offset removal (ADR §2.2
  step 1), and the emitted mask is the only statement of validity;
* the offset of each contiguous valid run is removed separately, so no estimate
  is ever taken across a break (ADR §2.2 step 2, invariant I2);
* each run is anti-alias resampled onto the output grid **at its absolute
  session position**, with the run's own edge values replicated over the
  filter's full support — no high-pass filter, no detrend, no edge guard
  (ADR Criterion 4);
* the emitted mask is the validity verdict **sampled** at the grid instants, and
  an invalid sample is ``NaN``, never ``0`` (ADR §2.1).

No clip length, hop, stride, split, task subset or normalisation may enter this
module (ADR Criterion 5). If a function here takes one of those, it is a bug.

Build order of record: ``code/notes/plan/ADR-003-resp-session-cache-plan.md``.

--------------------------------------------------------------------------------
FROZEN CONTRACT (code-side copy of ADR §4; the ADR is the authority)
--------------------------------------------------------------------------------
Identity fields (hashing these into the cache key is what makes a stale
artefact impossible to reuse silently, invariant I9)::

    format_version  corpus  fs_in  fs_out  sat_v  sat_min_s
    flat_window_s   flat_ptp_v  min_run_s  resample{k, taps_per_phase, up↓,
    kernel_radius_src}  store

Never in the identity, never in a shard (§4.2): ``clip_seconds``,
``clip_stride``, ``task_set`` / ``tasks`` / ``task_groups``, the subject split
or fold, ``physio_norm`` and every normalisation statistic, ``input_size``,
``max_entries`` and every raster-side knob.

Shard members (§4.3)::

    signal  float32 [N_out]   run-mean-removed volts; NaN exactly where mask is False
    mask    bool    [N_out]   True = usable; the sole authority
    cause   uint8   [N_out]   0 valid, 1 rail, 2 flatline, 3 short run, 4 non-finite
    n_src, fs_in, fs_out      grid identity, self-verifying without the raw file
    n_runs, valid_frac, n_nonfinite, session, resp_file   statistics/provenance

Grid arithmetic (§4.4)::

    N_out = floor((N_src - 1) * fs_out / fs_in) + 1
    s     = round(t_start * fs_out)      n = round(clip_seconds * fs_out)
    usable(t_start)  <=>  mask[s : s + n].all()
    thermal frame f  ->  grid sample round((f / fps) * fs_out)
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import resample_poly

__all__ = [
    'FS_IN', 'FS_OUT_DEFAULT', 'SAT_V', 'SAT_MIN_S', 'FLAT_WINDOW_S',
    'FLAT_PTP_V', 'MIN_RUN_S', 'RESAMPLE_TAPS_PER_PHASE', 'MIN_TAPS_PER_PHASE',
    'RESAMPLE_KIND', 'STORE_DTYPE', 'STORE_KIND',
    'CAUSE_VALID', 'CAUSE_SAT', 'CAUSE_FLAT', 'CAUSE_SHORT', 'CAUSE_NONFINITE',
    'CAUSE_NAMES', 'CAUSE_RANK', 'PRECEDENCE_ORDER',
    'resolve_geometry', 'contiguous_runs', 'flatline_window_samples',
    'validity_mask', 'preprocess_session',
]

# --------------------------------------------------------------------------- #
# frozen quantities (one constant per identity field, so the key cannot drift)
# --------------------------------------------------------------------------- #
#: Nominal source rate of ``Resp_Volts.txt``. DELIBERATELY a corpus constant and
#: never a measured value (ADR §2.3): the frame-to-sample mapping between
#: respiration and thermal time is a corpus-wide convention defined against it.
FS_IN = 1000.0

#: Default output grid rate. Not an ADR quantity — it is fixed at build time and
#: recorded in every shard, and it must divide ``FS_IN`` (see
#: :func:`resolve_geometry`).
FS_OUT_DEFAULT = 100.0

SAT_V = 9.95          #: rail threshold, magnitude in volts
SAT_MIN_S = 1.0       #: a rail contact SHORTER than this is KEPT (event-based)
FLAT_WINDOW_S = 5.0   #: flatline probe length — long enough that 0.05 Hz survives
#: NOTE the exact boundary the 5 s probe implies: at the extremum of a sine of
#: frequency f and amplitude A, the peak-to-peak spread over a window w is
#: A * (1 - cos(pi * f * w)). For f = 0.05 Hz and w = 5 s that is 0.293 * A, so a
#: 0.05 Hz breath is spared exactly when A > 0.3 / 0.293 = 1.02 V. Real belts
#: swing several volts (the rail is at 10 V), so this is not a practical limit —
#: but it IS the boundary, and it is recorded rather than assumed away.
FLAT_PTP_V = 0.3      #: flatline peak-to-peak threshold, volts
MIN_RUN_S = 8.0       #: valid runs shorter than this become holes

#: Taps per polyphase phase. ``scipy.signal.resample_poly`` designs
#: ``half_len = 10 * max(up, down)`` taps, i.e. this many taps per phase, so a
#: smaller value would silently under-pad the filter. Refused below 10.
RESAMPLE_TAPS_PER_PHASE = 10
MIN_TAPS_PER_PHASE = 10
RESAMPLE_KIND = 'polyphase-kaiser'

STORE_DTYPE = 'float32'
STORE_KIND = 'signal+mask+cause'

# --------------------------------------------------------------------------- #
# cause codes (precedence: non-finite > rail > flatline > short run)
# --------------------------------------------------------------------------- #
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

#: Cause codes from highest to lowest precedence. A sample inside two rules
#: carries exactly one code: the first match in this order (ADR §2.1 lists the
#: rules without an order; this module fixes one and the tests pin it).
PRECEDENCE_ORDER = (CAUSE_NONFINITE, CAUSE_SAT, CAUSE_FLAT, CAUSE_SHORT,
                    CAUSE_VALID)

#: ``CAUSE_RANK[code]`` = position in :data:`PRECEDENCE_ORDER` (0 = strongest).
#: Used as ``minimum_filter1d`` input wherever a reduction has to keep the most
#: severe code, and inverted with :data:`PRECEDENCE_ORDER`.
CAUSE_RANK = np.zeros(5, dtype=np.int8)
for _rank, _code in enumerate(PRECEDENCE_ORDER):
    CAUSE_RANK[_code] = _rank


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def resolve_geometry(fs_in=FS_IN, fs_out=FS_OUT_DEFAULT,
                     taps_per_phase=RESAMPLE_TAPS_PER_PHASE):
    """Validate the rate pair and return ``(up, down, kernel_radius_src)``.

    ``fs_out`` must **divide** ``fs_in``. That is a build constraint this
    implementation adds (plan A8), not an ADR quantity: with an integral
    decimation factor ``down`` the master grid index ``k`` sits at source index
    ``k * down`` exactly, so a run can be placed on the grid without a
    sub-sample phase error. A non-divisor rate is refused rather than silently
    shifted.

    ``kernel_radius_src`` is the polyphase filter's support in SOURCE samples,
    derived from the filter actually used rather than guessed::

        half_len_taps = taps_per_phase * max(up, down)      # at the upsampled rate
        radius_src    = half_len_taps / up = taps_per_phase * max(1, down/up)

    For 1000 Hz -> 100 Hz that is 100 source samples, not 20: replicating fewer
    would hand ``resample_poly`` its own zero padding and reintroduce exactly the
    run-edge transient ADR Criterion 4 forbids.
    """
    fs_in = float(fs_in)
    fs_out = float(fs_out)
    if not np.isfinite(fs_in) or fs_in <= 0:
        raise ValueError(f'resp_preprocess: fs_in must be > 0, got {fs_in!r}')
    if not np.isfinite(fs_out) or fs_out <= 0:
        raise ValueError(f'resp_preprocess: fs_out must be > 0, got {fs_out!r}')
    if fs_out > fs_in:
        raise ValueError(
            f'resp_preprocess: fs_out={fs_out:g} Hz is above fs_in={fs_in:g} Hz; '
            f'the artefact is a decimation of the source grid.')
    ratio = fs_in / fs_out
    down = int(round(ratio))
    if abs(ratio - down) > 1e-9 or down < 1:
        raise ValueError(
            f'resp_preprocess: fs_out={fs_out:g} Hz must divide fs_in={fs_in:g} '
            f'Hz (got ratio {ratio!r}). Pick a divisor of {fs_in:g} '
            f'(e.g. 100, 125, 200, 250, 500) — an integral decimation factor is '
            f'what lets each run land on exact master-grid instants.')
    taps = int(taps_per_phase)
    if taps < MIN_TAPS_PER_PHASE:
        raise ValueError(
            f'resp_preprocess: taps_per_phase={taps} < {MIN_TAPS_PER_PHASE}; '
            f'resample_poly designs 10 taps per phase and a shorter filter is '
            f'not the filter being padded for.')
    up = 1
    radius = taps * max(1, down // up)
    return up, down, int(radius)


# --------------------------------------------------------------------------- #
# run bookkeeping
# --------------------------------------------------------------------------- #
def contiguous_runs(mask) -> list:
    """Maximal ``True`` stretches as ``[(start, stop_exclusive), ...]``."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise ValueError(f'contiguous_runs: expected 1-D, got shape {m.shape}')
    if m.size == 0 or not m.any():
        return []
    edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def flatline_window_samples(fs_in: float, flat_window_s: float) -> int:
    """Effective flatline probe length in SOURCE samples (always odd).

    ``round(flat_window_s * fs_in)`` rounded UP to the next odd number, so the
    rolling min/max filter can be aligned to the window START exactly
    (``origin = size // 2``; SciPy rejects a larger origin for even sizes). At
    1000 Hz and 5 s this is 5001 samples instead of 5000 — 0.02 % of a probe that
    the ADR itself calls an operating point, and far from the 10 s period of a
    0.05 Hz breath the probe exists to spare.
    """
    n_w = int(round(float(flat_window_s) * float(fs_in)))
    if n_w < 3:
        raise ValueError(
            f'flatline_window_samples: flat_window_s={flat_window_s!r} at '
            f'fs_in={fs_in!r} gives {n_w} samples; a probe needs >= 3.')
    return n_w if n_w % 2 else n_w + 1


# --------------------------------------------------------------------------- #
# phase 1 — validity on the RAW grid
# --------------------------------------------------------------------------- #
def validity_mask(x, fs_in=FS_IN, sat_v=SAT_V, sat_min_s=SAT_MIN_S,
                  flat_window_s=FLAT_WINDOW_S, flat_ptp_v=FLAT_PTP_V,
                  min_run_s=MIN_RUN_S):
    """Per-sample validity of the RAW trace → ``(mask, cause)``.

    Decided once, on raw volts, before any offset removal or normalisation, so
    validity is a property of the SESSION and not of whichever window a consumer
    happens to cut (ADR §2.2 step 1, invariant I1).

    Order, and the reason for each position:

    1. **non-finite** first, because no filter may see a NaN (cause 4).
    2. **rail is event-based**: maximal contiguous stretches of ``|x| > sat_v``
       that last at least ``sat_min_s`` invalidate all of their samples; a
       shorter contact is KEPT (cause 1). A rolling duration or a fraction of the
       file is wrong by construction — corpus files carry 40-80 % rail fraction
       made of many sub-second grazes.
    3. **flatline is a window probe dilated back onto samples**: every sample
       lying in ANY window of ``flat_window_s`` whose peak-to-peak spread is
       below ``flat_ptp_v`` is invalid (cause 2). The probe is 5 s so that a
       0.05 Hz breath (10 s period) cannot be mistaken for a dead channel.
    4. **runs shorter than ``min_run_s`` become holes** (cause 3). Runs are an
       internal device and are never stored.

    :returns: ``(mask, cause)``, both length ``x.size``; ``cause == 0`` exactly
        where ``mask`` is True.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f'validity_mask: expected 1-D, got shape {x.shape}')
    fs_in = float(fs_in)
    if fs_in <= 0:
        raise ValueError(f'validity_mask: fs_in must be > 0, got {fs_in!r}')

    n = x.size
    xf = np.asarray(x, dtype=np.float64)
    cause = np.zeros(n, dtype=np.uint8)

    # 1 — non-finite
    finite = np.isfinite(xf)
    nonfinite = ~finite

    # 2 — rail, event-based on maximal contiguous runs
    rail = np.abs(np.where(finite, xf, 0.0)) > float(sat_v)
    min_event = max(int(round(float(sat_min_s) * fs_in)), 1)
    # Group `rail OR non-finite`, so a NaN gap INSIDE a saturation block cannot
    # split one event into two sub-second contacts that would both be kept. The
    # event qualifies on its number of RAIL samples, not on the group length, so
    # a short contact followed by a long gap is still just a contact.
    for a, b in contiguous_runs(rail | nonfinite):
        seg = slice(a, b)
        if int(rail[seg].sum()) >= min_event:
            cause[seg][rail[seg]] = CAUSE_SAT

    # 3 — flatline probe, dilated back onto every sample it covers
    n_w = flatline_window_samples(fs_in, flat_window_s)
    if n >= n_w:
        half = n_w // 2
        # scipy shifts the filter LEFT for a POSITIVE origin, so -half aligns the
        # rolling window to [j, j + n_w - 1]: the probe evaluates every start j,
        # exactly like the brute-force sliding-window loop it is pinned against.
        hi = maximum_filter1d(xf, size=n_w, origin=-half, mode='nearest')
        lo = minimum_filter1d(xf, size=n_w, origin=-half, mode='nearest')
        ptp = (hi - lo)[: n - n_w + 1]
        # A window holding a non-finite sample is NOT a flatline: the min/max
        # filters skip NaN instead of propagating it, so a window straddling a
        # gap would otherwise report a bogus peak-to-peak of the few finite
        # values it can see. Non-finite samples are already invalid (cause 4).
        csum = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
        full = (csum[n_w:] - csum[:-n_w]) == n_w
        bad_start = full & (ptp < float(flat_ptp_v))
        if bad_start.any():
            cover = np.zeros(n + 1, dtype=np.int32)
            idx = np.flatnonzero(bad_start)
            np.add.at(cover, idx, 1)
            np.add.at(cover, idx + n_w, -1)
            covered = np.cumsum(cover[:n]) > 0
            cause[covered & (cause == CAUSE_VALID)] = CAUSE_FLAT

    # 4 — runs shorter than the minimum become holes
    cause[nonfinite] = CAUSE_NONFINITE
    mask = cause == CAUSE_VALID
    min_run = int(round(float(min_run_s) * fs_in))
    for a, b in contiguous_runs(mask):
        if (b - a) < max(min_run, 1):
            mask[a:b] = False
            cause[a:b] = CAUSE_SHORT

    # invariant: the mask and the cause array say the same thing
    assert np.array_equal(mask, cause == CAUSE_VALID)
    return mask, cause


# --------------------------------------------------------------------------- #
# phase 2 — holes that no grid instant can see
# --------------------------------------------------------------------------- #
def _hole_cause(cause, a, b):
    """Most severe cause code inside ``[a, b)``, by :data:`PRECEDENCE_ORDER`."""
    ranks = CAUSE_RANK[cause[a:b]]
    if ranks.size == 0:
        return CAUSE_FLAT
    return int(PRECEDENCE_ORDER[int(ranks.min())])


def _widen_subperiod_holes(mask, cause, down):
    """Widen every hole that contains no grid instant, in place.

    A hole is a maximal ``False`` stretch. If it is narrower than one output
    period it can fall **between** grid instants; the sampled mask would then
    call both neighbours valid and a window could span two runs whose means
    differ — a step of arbitrary size inside a window the mask calls valid.

    Widening costs at most ONE grid sample per hole and is extended on the side
    whose extension is shorter (left on a tie), so a run bounded by holes wider
    than one period keeps every sample it has. That is what keeps the ADR's
    ``clip_seconds <= min_run_s`` bound satisfiable (an exactly 8 s run still
    yields 800 usable samples at 100 Hz).

    :returns: ``True`` if anything changed.
    """
    changed = False
    n = mask.size
    for a, b in contiguous_runs(~mask):
        first_grid = -(-a // down) * down          # smallest multiple of down >= a
        if first_grid <= b - 1:
            continue                               # a grid instant is inside
        left_pos = ((a - 1) // down) * down if a > 0 else None
        right_pos = first_grid if first_grid <= n - 1 else None
        if left_pos is None and right_pos is None:
            continue
        cost_left = (a - left_pos) if left_pos is not None else None
        cost_right = (right_pos - (b - 1)) if right_pos is not None else None
        code = _hole_cause(cause, a, b)
        if cost_left is not None and (cost_right is None or cost_left <= cost_right):
            mask[left_pos:b] = False
            cause[left_pos:b] = code
        else:
            mask[a:right_pos + 1] = False
            cause[a:right_pos + 1] = code
        changed = True
    return changed


def _resolve_runs(mask, cause, fs_in, min_run_s, down, max_iter=8):
    """Apply hole widening and the minimum-run rule until both are stable.

    Terminates because widening is monotone: each step either registers a hole
    on a grid instant or turns an adjacent run into a hole, and converted runs
    only ever merge into an existing hole.
    """
    mask = np.array(mask, dtype=bool, copy=True)
    cause = np.array(cause, dtype=np.uint8, copy=True)
    min_run = max(int(round(float(min_run_s) * fs_in)), 1)
    for _ in range(max_iter):
        changed = _widen_subperiod_holes(mask, cause, down)
        short = [r for r in contiguous_runs(mask) if (r[1] - r[0]) < min_run]
        if short:
            for a, b in short:
                mask[a:b] = False
                cause[a:b] = CAUSE_SHORT
            changed = True
        if not changed:
            break
    runs = [r for r in contiguous_runs(mask) if (r[1] - r[0]) >= min_run]
    return mask, cause, runs


# --------------------------------------------------------------------------- #
# the artefact payload
# --------------------------------------------------------------------------- #
def preprocess_session(x, fs_in=FS_IN, fs_out=FS_OUT_DEFAULT,
                       sat_v=SAT_V, sat_min_s=SAT_MIN_S,
                       flat_window_s=FLAT_WINDOW_S, flat_ptp_v=FLAT_PTP_V,
                       min_run_s=MIN_RUN_S,
                       taps_per_phase=RESAMPLE_TAPS_PER_PHASE,
                       dtype=STORE_DTYPE):
    """Clean one session into its whole-session grid payload.

    Pure: no IO, no config object, no globals. The result is everything a shard
    stores except provenance, and the mask is the ONLY validity statement it
    makes.

    Per run (ADR §2.2 step 3): subtract the run's own ``float64`` mean, then
    anti-alias resample onto the master grid at the run's ABSOLUTE position.
    The pad is the run's own edge values, ``radius + ((down - t) % down)`` on the
    left and ``radius`` on the right, so (a) the filter's full support is covered
    — no transient, no guard band — and (b) the retained outputs land on exact
    grid instants. The pad is built in memory from the run and never read from a
    file, so a run beginning at sample 0 is treated identically to any other; the
    outputs produced by a virtual negative start are precisely the discarded
    ones.

    :returns: dict with ``signal`` (``float32 [N_out]``, ``NaN`` where invalid),
        ``mask`` (``bool [N_out]``), ``cause`` (``uint8 [N_out]``), ``n_src``,
        ``fs_in``, ``fs_out``, ``n_runs``, ``valid_frac``, ``n_nonfinite``.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(
            f'preprocess_session: expected a 1-D trace, got shape {x.shape}')
    if x.size == 0:
        raise ValueError('preprocess_session: empty trace')
    up, down, radius = resolve_geometry(fs_in, fs_out, taps_per_phase)
    fs_in = float(fs_in)
    fs_out = float(fs_out)
    n_src = int(x.size)

    mask_src, cause_src = validity_mask(
        x, fs_in=fs_in, sat_v=sat_v, sat_min_s=sat_min_s,
        flat_window_s=flat_window_s, flat_ptp_v=flat_ptp_v,
        min_run_s=min_run_s)
    mask_src, cause_src, runs = _resolve_runs(
        mask_src, cause_src, fs_in, min_run_s, down)

    # N_out = floor((N_src - 1) * fs_out / fs_in) + 1 == the number of multiples
    # of `down` inside [0, N_src - 1]: the grid never addresses a sample the
    # file does not have (ADR §4.4).
    n_out = -(-n_src // down)
    signal = np.full(n_out, np.nan, dtype=np.float64)

    for a, b in runs:
        k0 = -(-a // down)               # first grid instant inside the run
        k1 = (b - 1) // down             # last grid instant inside the run
        if k1 < k0:
            continue
        n_keep = k1 - k0 + 1

        y = np.asarray(x[a:b], dtype=np.float64)
        y -= y.mean()                    # per-run offset, float64, no filtering

        t = (k0 * down - a) % down       # sub-sample offset of the first instant
        pad_left = radius + (down - t) % down
        padded = np.pad(y, (pad_left, radius), mode='edge')
        out = resample_poly(padded, up, down)

        j_first = radius // down + (1 if t > 0 else 0)
        seg = out[j_first:j_first + n_keep]
        if seg.size != n_keep:
            raise AssertionError(
                f'resp_preprocess: run [{a}, {b}) resampled to {seg.size} '
                f'samples, expected {n_keep} (grid {k0}..{k1})')
        signal[k0:k1 + 1] = seg

    # the mask is the raw verdict SAMPLED at the grid instants — never dilated:
    # resampling is per run, so no invalid volt can enter an output (plan A6).
    mask = np.asarray(mask_src[::down], dtype=bool)
    cause = np.asarray(cause_src[::down], dtype=np.uint8)

    if not np.array_equal(np.isnan(signal), ~mask):
        raise AssertionError(
            'resp_preprocess: emitted signal/mask disagree — an invalid sample '
            'must be NaN and every valid one must be filled. Refusing to write '
            'a shard whose mask is not the sole authority (ADR Criterion 1).')
    if not np.array_equal(mask, cause == CAUSE_VALID):
        raise AssertionError(
            'resp_preprocess: mask and cause disagree — `mask[k] == False` must '
            'imply `cause[k] != 0`.')

    signal = signal.astype(dtype, copy=False)
    return {
        'signal': signal,
        'mask': mask,
        'cause': cause,
        'n_src': n_src,
        'fs_in': int(round(fs_in)),
        'fs_out': int(round(fs_out)),
        'n_runs': int(len(runs)),
        'valid_frac': float(mask.mean()) if mask.size else 0.0,
        'n_nonfinite': int((~np.isfinite(np.asarray(x, dtype=np.float64))).sum()),
    }
