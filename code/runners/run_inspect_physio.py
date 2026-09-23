"""Inspect the RAW BP4D ``Physiology/`` channels (statistics + plots).

Reads the single-column physiology files straight out of the raw tree::

    <raw_root>/Physiology/<subject>/<task>/<channelfile>.txt

for one or more ``(subject, task)`` sessions and reports, per channel, the
basic statistics -- LENGTH (samples + duration), MIN/MAX (and mean/median/std/
percentiles), and the ESTIMATED FREQUENCY (the dominant in-band spectral peak,
cross-checked against an autocorrelation period estimate) -- then saves the
figure(s) and a JSON summary into::

    <output_dir>/<subject>_<task>/          # default <repo>/output/inspect_data/F001_T1/
        <CHANNEL>.png                       # one figure per channel
        <CHANNEL>.json                      # the same stats, machine-readable
        <FAMILY>_overview.png               # per family: BP_overview.png, Resp_overview.png
        physio_summary.json                 # all requested channels of the session
        overview.png                        # stacked native-rate traces (>=2 channels)

Channel selection (``--channel``, case-insensitive, comma-separated or ``all``)
is by canonical name. The three canonical channels map to raw files through
``data.prepare_bp4d.CHANNEL_FILES`` -- the SAME table the canonical converter
uses, so the inspector and the converter can never disagree. HR is an
inspector-only extra (see ``raw_file`` below)::

    BP  (bp)  <- BP_mmHg.txt            [mmHg]  pulse-pressure surrogate
    Resp (resp) <- Resp_Volts.txt         [V]     respiration belt
    EDA  (eda)  <- EDA_microsiemens.txt   [uS]    skin conductance
    HR   (hr)   <- Pulse Rate_BPM.txt     [BPM]   vendor heart rate

HR IS NOT A WAVEFORM, so it is analysed differently. ``Pulse Rate_BPM.txt`` is a
vendor per-beat series held constant between beats; its spectrum is a staircase,
so it gets ``--analysis 'step'``: no spectral estimate at all, just
``step_update_rate`` (how often the value changed = the beat rate, plus how long
each value was held) and the plain min/max/mean BPM in ``stats``. That is also
why HR lives in ``CHANNEL_META['hr']['raw_file']`` instead of in
``prepare_bp4d.CHANNEL_FILES``: that shared table drives the columns of the
canonical ``signals.csv`` and treats a missing channel as a session failure, so
adding HR there would silently add an 'hr' column and break every session
without a ``Pulse Rate_BPM.txt``.

A CHANNEL CAN HAVE A WHOLE FAMILY OF RAW FILES. Inspecting a channel also loads
the extra files that belong with it (``CHANNEL_FAMILIES``), summarises them into
the channel's JSON under ``family_components`` and draws them together in
``<FAMILY>_overview.png``:

    BP  -> BP_overview.png    BP_mmHg.txt + LA Systolic + LA Mean + BP Dia
    Resp -> Resp_overview.png  Resp_Volts.txt + Respiration Rate_BPM.txt

They are NOT all the same kind of signal, so the two kinds are treated
differently and each component says which it is via ``derived``/``kind``:

* ``BP_mmHg.txt`` and ``Resp_Volts.txt`` are continuous WAVEFORMS (they change
  every sample) and get the usual ``frequency`` block.
* ``LA Systolic``, ``LA Mean``, ``BP Dia`` and ``Respiration Rate_BPM`` are
  vendor-derived, STEP-HELD between beats (LA Systolic holds 114.433 for >300
  samples -- about a whole beat -- then jumps to 113.901). A spectral estimate
  is meaningless for a staircase: its flat plateaus dominate, so the in-band
  Welch peak pins to the LOWER band edge (measured 0.6 Hz = 36.0/min for all
  three BP series against a ~90/min pulse). They therefore get
  ``step_update_rate`` (value-change count = beat rate, plus how long each
  value was held) instead.

A family may also MIX UNITS -- the Resp family pairs volts with breaths/min --
in which case the figure puts the second unit on a right-hand axis instead of
drawing two scales on one. The step-held files contain exact 0.0 dropouts
(F001_T1 has a 26-sample run in both LA Systolic and LA Mean at line 29705),
reported by ``stats.frac_exactly_zero``; the canonical converter does not read
those vendor-derived files, so the zeros never reach ``signals.csv``.

ORIGINAL SAMPLE RATE IS KEPT. The raw files carry no time column, so the
sample rate cannot be recovered from the data itself; it is taken from the BP4D
nominal rate (``--phys_fs``, default 1000 Hz) or, with ``--phys_fs 0``,
ESTIMATED per session as ``n_samples / (n_frames / --fps_rgb)`` -- the same
duration assumption ``prepare_bp4d.py`` makes. Either way the ORIGINAL samples
are analysed and plotted 1:1 -- nothing is low-pass filtered, decimated or
resampled to a canonical rate (contrast: ``prepare_bp4d.py`` resamples to
``--fs``, default 100 Hz). ``fs`` is used only for the time axis and the
Hz <-> per-minute conversions.

ESTIMATED FREQUENCY is reported as ``welch_peak_hz`` (the in-band PSD maximum,
parabolically refined) plus two references: ``autocorr_hz`` and
``cycle_count_per_min`` (peak counting). The runner DIAGNOSES its own estimates
instead of hiding failures -- a peak on the search-band edge, >25% disagreement
between estimators, or a channel railed at a hard limit (respiration pinned at
exactly -10.0000 V for 10-45% of several sessions) becomes a warning, and
``preferred_estimate`` names which number to trust. ``--band`` moves the
spectral search band if a session gets flagged.

Per-session artifacts are overwritten on every run but never pruned: inspecting
a session with a NARROWER ``--channel`` selection leaves the figures of the
channels inspected previously in place (``physio_summary.json`` lists what the
last run actually covered).

Usage (from ``code/``)::

    # one session, every channel
    python runners/run_inspect_physio.py --subject F001 --task T1 --channel all

    # just the vendor heart rate (writes HR.png)
    python runners/run_inspect_physio.py --subject F001 --task T1 --channel HR

    # just respiration + EDA, several sessions, estimated sample rate
    python runners/run_inspect_physio.py --subject F001,F002 --task T1,T2 \
        --channel Resp,EDA --phys_fs 0

    # what is on disk?
    python runners/run_inspect_physio.py --list

Paths default to ``$RAW_DATA_PATH`` (raw BP4D root) and ``$OUTPUT_DIR`` (see
``scripts/env_local.sh``); without the env profile they fall back to the
in-repo ``data/raw/BP4D`` and ``<repo>/output``. ``--output_dir`` is the
``inspect_data`` ROOT -- the ``<subject>_<task>`` subdirectory is appended, as
in the layout above.
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import prepare_bp4d as prep

# --------------------------------------------------------------------------- #
# channel table: canonical name -> raw file + how to interpret the signal
# --------------------------------------------------------------------------- #
# ``band`` is the physiological band the dominant-frequency search is limited
# to; searching the full spectrum would just find the DC/drift peak.
# ``periodic`` says whether a "dominant frequency" is a meaningful quantity for
# the channel at all (EDA is an event/slow-drift signal, not an oscillation).
# ``analysis`` picks the rate analysis that actually applies to the channel:
#   'spectral' -- continuous waveform: Welch peak + autocorrelation + counting
#   'step'     -- vendor per-beat series held between beats: value-change rate,
#                 and NO spectral estimate at all (a staircase pins the Welch
#                 peak to the band edge -- see step_update_rate)
#   'slow'     -- non-oscillatory drift (EDA): tonic/phasic split, no rate
# ``raw_file`` is an inspector-LOCAL override of the raw-file needle, used by HR.
# HR is deliberately NOT added to ``prepare_bp4d.CHANNEL_FILES``: that table also
# drives the canonical signals.csv -- both its columns (`['time'] +
# list(CHANNEL_FILES)`) and its fail-the-whole-session behaviour on a missing
# channel -- so putting HR there would silently add an 'hr' column and break
# every session that lacks Pulse Rate_BPM.txt.
CHANNEL_META = {
    'bp': {
        'name': 'BP',
        'unit': 'mmHg',
        'label': 'blood pulse (pulse-pressure surrogate)',
        'band': (0.60, 4.00),
        'band_note': 'pulse band, 36-240 bpm',
        'periodic': True,
        'analysis': 'spectral',
    },
    'resp': {
        'name': 'Resp',
        'unit': 'V',
        'label': 'respiration belt (raw volts)',
        'band': (0.10, 1.00),
        'band_note': 'breathing band, 6-60 br/min',
        'periodic': True,
        'analysis': 'spectral',
    },
    'eda': {
        'name': 'EDA',
        'unit': 'uS',
        'label': 'skin conductance',
        'band': (0.01, 0.50),
        'band_note': 'slow/phasic band (NOT a periodic rate)',
        'periodic': False,
        'analysis': 'slow',
    },
    'hr': {
        'name': 'HR',
        'unit': 'BPM',
        'label': 'heart rate (vendor-derived from the BP waveform)',
        'raw_file': 'Pulse Rate_BPM.txt',
        # 'band' is unused under the 'step' analysis (no spectral search is
        # attempted); kept so every row of the table has the same shape
        'band': (0.50, 3.00),
        'band_note': 'vendor per-beat rate, step-held -- reported as a '
                     'value-change rate, not a spectral estimate',
        'periodic': False,
        'analysis': 'step',
    },
}
CHANNEL_ORDER = ['bp', 'resp', 'eda', 'hr']

# CLI aliases -> canonical channel key. NOTE 'pulse' stays mapped to bp (the
# blood-pulse waveform, as documented); the RATE channel answers to
# 'hr'/'heart_rate'/'pulse_rate'/'bpm' instead.
CHANNEL_ALIASES = {
    'bp': 'bp', 'bp_mmhg': 'bp', 'pulse': 'bp',
    'blood_pulse': 'bp', 'pressure': 'bp',
    'resp': 'resp', 'respiration': 'resp', 'resp_volts': 'resp',
    'breathing': 'resp',
    'eda': 'eda', 'gsr': 'eda', 'eda_microsiemens': 'eda',
    'skin_conductance': 'eda',
    'hr': 'hr', 'heart_rate': 'hr', 'heartrate': 'hr', 'pulse_rate': 'hr',
    'bpm': 'hr',
}

# Raw files that belong TOGETHER with a canonical channel. Inspecting a channel
# with a family entry also loads those extra files, summarises them into the
# channel's JSON and draws them in ``<FAMILY>_overview.png``.
#
# Row layout: (key, raw file needle, label, colour, unit, kind) where ``kind``
# is 'waveform' (continuous -- a spectral estimate is meaningful) or 'step'
# (vendor-derived, held between beats -- it gets a value-change rate instead,
# because a staircase pins the Welch peak to the band edge; see
# :func:`step_update_rate`).
#
# ``unit`` is PER ROW on purpose: a family may mix units. The BP family is all
# mmHg and is overlaid on one axis, while the Resp family pairs a volts
# waveform with a breaths/min rate, which the figure puts on two y axes rather
# than pretending they share a scale.
#
# The needles are EXACT raw basenames, so prep.find_channel_file matches them
# directly (its Dia/Mean/Systolic exclusion only affects substring fallbacks).
CHANNEL_FAMILIES = {
    'bp': {
        'name': 'BP',
        'title': 'BP family',
        'components': (
            ('bp_pulse',     'BP_mmHg.txt',              'BP (waveform)', 'tab:blue',   'mmHg', 'waveform'),
            ('bp_systolic',  'LA Systolic BP_mmHg.txt',  'LA Systolic',   'tab:red',    'mmHg', 'step'),
            ('bp_mean',      'LA Mean BP_mmHg.txt',      'LA Mean',       'tab:orange', 'mmHg', 'step'),
            ('bp_diastolic', 'BP Dia_mmHg.txt',          'BP Dia',        'tab:green',  'mmHg', 'step'),
        ),
    },
    'resp': {
        'name': 'Resp',
        'title': 'Resp family',
        'components': (
            ('resp_volts', 'Resp_Volts.txt',           'Resp (waveform)', 'tab:blue', 'V',   'waveform'),
            ('resp_rate',  'Respiration Rate_BPM.txt', 'Resp rate (vendor)', 'tab:red', 'BPM', 'step'),
        ),
    },
}

SCHEMA = 'bp4d-physio-inspect/1'

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_CODE_DIR)


# --------------------------------------------------------------------------- #
# path defaults
# --------------------------------------------------------------------------- #
def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def default_raw_root() -> str:
    """``$RAW_DATA_PATH`` when set, else the in-repo raw BP4D root."""
    return _env('RAW_DATA_PATH', '') or prep.default_raw_root()


def default_output_dir() -> str:
    """``$OUTPUT_DIR/inspect_data``, else ``<repo>/output/inspect_data``.

    Absolute on purpose (same reasoning as ``run_inspect_data.py``): a bare
    ``./output/...`` would land inside ``code/`` when the env profile was not
    sourced.
    """
    root = _env('OUTPUT_DIR', '') or os.path.join(_REPO_DIR, 'output')
    return os.path.join(root, 'inspect_data')


# --------------------------------------------------------------------------- #
# raw data access
# --------------------------------------------------------------------------- #
def channel_raw_file(key: str) -> str:
    """Raw-file needle for a channel (a basename inside ``phys_dir``).

    Prefers the inspector-local ``raw_file`` override in ``CHANNEL_META`` and
    otherwise falls back to ``prepare_bp4d.CHANNEL_FILES``. See the ``raw_file``
    note there for why HR is kept out of that shared table.
    """
    return CHANNEL_META.get(key, {}).get('raw_file') or prep.CHANNEL_FILES[key]


def discover_sessions(raw_root: str):
    """Sorted ``[(subject, task), ...]`` present under ``<raw_root>/Physiology``."""
    phys = os.path.join(raw_root, 'Physiology')
    out = []
    if not os.path.isdir(phys):
        return out
    for subj in sorted(os.listdir(phys)):
        pdir = os.path.join(phys, subj)
        if not os.path.isdir(pdir):
            continue
        for task in sorted(os.listdir(pdir)):
            if os.path.isdir(os.path.join(pdir, task)):
                out.append((subj, task))
    return out


def load_channel_tolerant(path: str):
    """Load a single-column .txt as float64, skipping unparsable lines.

    The raw BP4D channel files are one value per line with no header and no
    time column. A plain ``np.loadtxt`` aborts on a single bad line, which is
    the wrong behaviour for an INSPECTION tool: here blank/truncated lines are
    skipped (counted, then reported as a warning) so a partly-written file
    still yields its statistics.

    Returns ``(values, n_lines, n_bad)``.
    """
    vals, n_lines, n_bad = [], 0, 0
    with open(path, 'r', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n_lines += 1
            try:
                vals.append(float(s.split()[0]))
            except (ValueError, IndexError):
                n_bad += 1
    return np.asarray(vals, dtype=np.float64), n_lines, n_bad


def load_family_components(phys_dir: str, family_key: str, fs: float,
                           band) -> list:
    """Load the raw files that belong with ``family_key``'s channel.

    Missing files are skipped rather than fatal -- a session without e.g.
    ``LA Systolic BP_mmHg.txt`` still gets the components it does have.

    Each record carries the same ``stats`` block as a real channel so it
    serialises straight into the JSON, plus ``derived``/``kind``/``unit``. The
    two kinds get DIFFERENT rate analyses, because a spectral estimate is only
    meaningful for one of them:

    * ``kind='waveform'`` gets the usual ``frequency`` block;
    * ``kind='step'`` gets ``step_update_rate`` -- its value-change count. That
      is the beat/refresh rate, and it also tells you how long the vendor held
      each value (``update_interval_s_median``), which is how coarse the
      vendor's own estimate is.
    """
    comps = []
    for key, needle, label, color, unit, kind in CHANNEL_FAMILIES[family_key]['components']:
        path = prep.find_channel_file(phys_dir, needle)
        if path is None:
            continue
        x, n_lines, n_bad = load_channel_tolerant(path)
        if x.size == 0:
            continue
        waveform = kind == 'waveform'
        rec = {
            'key': key, 'label': label, 'color': color, 'unit': unit,
            'raw_file': path, 'file': os.path.basename(path),
            'derived': not waveform,
            'kind': ('continuous waveform' if waveform
                     else 'per-beat value, step-held'),
            'stats': basic_stats(x),
            'duration_s': float(x.size / fs),
            '_x': x,
        }
        if waveform:
            rec['frequency'] = estimate_frequency(x, fs, band)
        else:
            rec['step_update_rate'] = step_update_rate(x, fs)
        if n_bad:
            rec['n_bad_lines'] = n_bad
        rec['warnings'] = quality_warnings(rec['stats'], unit)
        comps.append(rec)
    return comps


def rgb_duration_estimate(raw_root: str, subject: str, task: str, fps_rgb: float):
    """Estimate the SESSION DURATION from the RGB frame count.

    The physiology files have no time column, so the only in-dataset handle on
    a session's wall-clock duration is its RGB frame count. Returns
    ``(duration_s, n_frames, note)`` with ``duration_s`` 0.0 when it cannot be
    derived (missing frame dir); the caller then reports that ``--phys_fs 0``
    was not possible.

    The sample RATE is deliberately NOT computed here: it needs the number of
    physiology samples, which only the caller knows (fs = n_samples / duration).
    Dividing frames by fps and calling that the physiology rate is a trap -- it
    just returns the RGB fps.

    Only the directory listing is touched -- no JPEG is decoded (see the
    decode-cost notes in ``code/README.md``), so this stays cheap.
    """
    rgb_dir = os.path.join(raw_root, '2D+3D', subject, task)
    if not os.path.isdir(rgb_dir):
        return 0.0, 0, f'no RGB frame dir ({rgb_dir})'
    n = len([f for f in os.listdir(rgb_dir)
             if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')])
    if n == 0:
        return 0.0, 0, f'{n} RGB frames'
    if fps_rgb <= 0:
        return 0.0, n, f'--fps_rgb must be > 0 (got {fps_rgb:g})'
    return n / fps_rgb, n, f'{n} RGB frames / {fps_rgb:g} fps = {n / fps_rgb:.1f} s'


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def basic_stats(x: np.ndarray) -> dict:
    """Length / min / max / central tendency / spread of the finite samples."""
    finite = x[np.isfinite(x)]
    out = {
        'n_samples': int(x.size),
        'n_samples_finite': int(finite.size),
        'n_samples_non_finite': int(x.size - finite.size),
    }
    if finite.size == 0:
        return out
    p = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    out.update({
        'min': float(finite.min()),
        'max': float(finite.max()),
        'peak_to_peak': float(finite.max() - finite.min()),
        'mean': float(finite.mean()),
        'std': float(finite.std(ddof=0)),
        'median': float(p[3]),
        'p01': float(p[0]), 'p05': float(p[1]), 'p25': float(p[2]),
        'p75': float(p[4]), 'p95': float(p[5]), 'p99': float(p[6]),
        'iqr': float(p[4] - p[2]),
        # constant-ness is a real failure mode of these exports (sensor dropouts
        # are written as a flat line), so surface it instead of hiding it in std
        'n_unique': int(np.unique(finite).size),
        'constant_frac': float(np.mean(finite == finite[0])),
        # a big pile of samples exactly at an extreme means the channel railed
        # (e.g. the F001 respiration belt pins at -10.0000 V for ~10-13% of
        # several sessions), which silently distorts rates and spectra
        'frac_at_min': float(np.mean(finite == finite.min())),
        'frac_at_max': float(np.mean(finite == finite.max())),
        # exactly 0.0 is outside the physiological range of EVERY channel here
        # (BP in mmHg, a respiration belt in volts, skin conductance in uS), so
        # an exact zero is a dropout in the vendor export. Found on F001_T1:
        # LA Systolic and LA Mean each contain a 26-sample run of 0s (line 29705
        # ff), which is 0.04% of the session and so slipped past the railing
        # check above.
        'n_zero_samples': int(np.count_nonzero(finite == 0.0)),
        'frac_exactly_zero': float(np.mean(finite == 0.0)),
    })
    # smallest non-zero gap between distinct values = the ADC/LSB step, which
    # tells you how much of a small-amplitude trace is quantisation
    uniq = np.unique(finite)
    if uniq.size > 1:
        d = np.diff(uniq)
        d = d[d > 0]
        if d.size:
            out['quantization_step'] = float(d.min())
    return out


def quality_warnings(stats: dict, unit: str) -> list:
    """Data-quality complaints for one signal's ``stats`` block.

    Shared by the canonical channels AND the ``CHANNEL_FAMILIES`` components, so
    a dropout in e.g. ``LA Mean BP_mmHg.txt`` is reported just like one in a
    channel file. (It was previously inline in the channel loop only, which is
    why the 26 zero samples in LA Systolic/LA Mean on F001_T1 were present in
    the JSON but never warned about.)
    """
    out = []
    if 'min' not in stats:
        return out
    if stats['std'] == 0:
        out.append('signal is constant (std == 0) - likely a sensor dropout '
                   'in the raw export')
    if stats.get('frac_exactly_zero', 0.0) > 0:
        out.append(f"{stats['n_zero_samples']} samples are exactly 0.0 "
                   f"({stats['frac_exactly_zero'] * 100:.2f}%), which is "
                   f"outside the range of this channel - a dropout in the "
                   f"vendor export, not a real measurement")
    for side, frac_key in (('min', 'frac_at_min'), ('max', 'frac_at_max')):
        frac = stats.get(frac_key, 0.0)
        if frac > 0.01:
            out.append(f'{frac * 100:.1f}% of samples sit exactly at the {side} '
                       f'({stats[side]:.6g} {unit}) - the signal looks '
                       f'railed/clipped there')
    return out


def _bandpass(x: np.ndarray, fs: float, band) -> np.ndarray:
    """Zero-phase Butterworth band-pass; returns the input on any failure."""
    try:
        from scipy.signal import butter, sosfiltfilt
        nyq = fs / 2.0
        lo, hi = float(band[0]), float(band[1])
        hi = min(hi, nyq * 0.95)
        if lo <= 0 or hi <= lo:
            return x
        sos = butter(4, [lo / nyq, hi / nyq], btype='bandpass', output='sos')
        padlen = int(min(max(0, x.size - 1), max(1, round(3 * fs))))
        if padlen < 1:
            return x
        return sosfiltfilt(sos, x, padlen=padlen)
    except Exception:
        return x


def _autocorr_freq(x: np.ndarray, fs: float, band):
    """Period estimate from the strongest autocorrelation peak inside ``band``.

    Complements the Welch peak: for a NOTCHED, asymmetric waveform (the BP
    pulse has a dicrotic notch, and the raw export carries high-frequency
    noise) the number of zero crossings overcounts the fundamental period,
    while the autocorrelation peak does not. The search is restricted to the
    lags the band allows (``1/band[1] <= lag <= 1/band[0]``), so it cannot
    latch onto the DC/drift timescale.

    Returns ``(freq_hz, r_peak, at_band_edge)`` or None.
    """
    y = _bandpass(x, fs, band)
    y = y - np.nanmean(y)
    n = y.size
    if n < max(32, int(4 * fs)):
        return None
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo_lag = max(1, int(np.floor(fs / float(band[1]))))
    hi_lag = min(n - 2, int(np.ceil(fs / float(band[0]))))
    if hi_lag - lo_lag < 2:
        return None
    k = int(np.argmax(ac[lo_lag:hi_lag + 1]))
    lag = float(lo_lag + k)
    at_edge = k in (0, hi_lag - lo_lag)
    if 0 < lag < n - 1:                            # parabolic refinement
        y0, y1, y2 = ac[int(lag) - 1], ac[int(lag)], ac[int(lag) + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) < 1.0:
                lag += float(delta)
    if lag <= 0:
        return None
    return fs / lag, float(ac[int(round(lo_lag + k))]), bool(at_edge)


def estimate_frequency(x: np.ndarray, fs: float, band, window_s: float = 20.0):
    """Estimate the dominant frequency inside ``band`` (Hz), three ways.

    1. Welch PSD peak: the maximum of the in-band power spectral density,
       refined by parabolic interpolation between the three bins around the
       peak (the raw bin spacing ``fs/nperseg`` would otherwise quantise e.g. a
       75 bpm pulse in ~3 bpm steps at a 20 s window). ``welch_freq_resolution_hz``
       reports the unresampled bin spacing so the sharpening is not mistaken
       for accuracy. This is the headline ``welch_peak_hz``.
    2. Autocorrelation peak: a robust secondary estimate (see
       :func:`_autocorr_freq`). ``method_agreement_pct`` compares the two.
    3. Zero crossings of the band-passed signal (one per cycle for a smooth
       oscillation). Reported for reference only -- on notched/noisy waveforms
       such as the BP pulse it systematically OVERCOUNTS, so it is deliberately
       NOT the basis of the agreement metric.

    Returns an empty dict when there is too little data to analyse.
    """
    finite = x[np.isfinite(x)]
    n = finite.size
    if n < 16 or fs <= 0:
        return {}

    out = {
        'band_hz': [float(band[0]), float(band[1])],
        'method': 'welch-peak (primary) + autocorrelation (secondary) + '
                  'zero-crossing (reference)',
    }

    # ---- reference: zero crossings of the band-passed signal -------------- #
    bp = _bandpass(finite, fs, band)
    bp = bp - np.nanmean(bp)
    if bp.size > 1:
        up = int(np.count_nonzero((bp[:-1] <= 0) & (bp[1:] > 0)))
        out['zero_crossing_hz'] = float(up / (n / fs))
        out['zero_crossing_per_min'] = float(up * 60.0 / (n / fs))

    # ---- secondary: autocorrelation peak ---------------------------------- #
    ac = _autocorr_freq(finite, fs, band)
    if ac is not None:
        out['autocorr_hz'] = float(ac[0])
        out['autocorr_per_min'] = float(ac[0] * 60.0)
        out['autocorr_peak_r'] = float(ac[1])
        if ac[2]:
            out['autocorr_at_band_edge'] = True

    # ---- primary: Welch PSD peak ------------------------------------------ #
    try:
        from scipy.signal import welch
        nperseg = int(min(n, max(64, round(window_s * fs))))
        nfft = int(max(4096, 4 * nperseg))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f, pxx = welch(finite, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2, nfft=nfft, detrend='linear')
        out['psd_window_s'] = float(nperseg / fs)
        out['welch_freq_resolution_hz'] = float(fs / nperseg)
        m = (f >= band[0]) & (f <= min(band[1], f[-1]))
        if np.count_nonzero(m) >= 3:
            fb, pb = f[m], pxx[m]
            k = int(np.argmax(pb))
            fpk = float(fb[k])
            # parabolic refinement around the peak (interpolation, not added
            # resolution -- see the docstring)
            if 0 < k < fb.size - 1:
                y0, y1, y2 = np.log(pb[k - 1]), np.log(pb[k]), np.log(pb[k + 1])
                denom = (y0 - 2 * y1 + y2)
                if denom != 0:
                    delta = 0.5 * (y0 - y2) / denom
                    if abs(delta) < 1.0:
                        fpk += float(delta * (fb[1] - fb[0]))
            out['welch_peak_hz'] = fpk
            out['welch_peak_per_min'] = float(fpk * 60.0)
            # a peak ON the search-band edge means the true peak probably lies
            # outside it (drift below, noise above) -- surface it, do not hide it
            if k in (0, fb.size - 1):
                out['welch_peak_at_band_edge'] = True
            med = float(np.median(pb))
            out['welch_peak_over_median_power'] = (
                float(pb[k] / med) if med > 0 else None)
            tot = float(np.sum(pxx[f > 0])) if np.any(f > 0) else 0.0
            out['band_power_frac'] = (
                float(np.sum(pb) / tot) if tot > 0 else None)
    except Exception as exc:                      # scipy missing / degenerate
        out['welch_error'] = str(exc)

    # agreement is measured against the two period-preserving methods only
    if 'welch_peak_hz' in out and 'autocorr_hz' in out:
        a, b = out['welch_peak_hz'], out['autocorr_hz']
        if max(a, b) > 0:
            out['method_agreement_pct'] = float(abs(a - b) / max(a, b) * 100.0)
    return out


def cycle_count_rate(x: np.ndarray, fs: float, band) -> dict:
    """Count breathing/pulse CYCLES by peak picking (reference estimate).

    The most interpretable of the reference estimators: the band-passed trace is
    smoothed, then its peaks are counted. Unlike the spectral estimator it makes
    no stationarity assumption over the whole recording, so it is a useful
    tie-breaker when the Welch peak locks onto a band edge (F003_T1 respiration
    is a real example: 6.0/min at the band edge vs 13.7/min by counting).

    Parameters are derived from the search band's GEOMETRIC MEAN period
    ``p = 1/sqrt(lo*hi)``: smoothing ``0.25*p`` and minimum peak separation
    ``0.6*p`` with a prominence of ``0.35*std``. Taking the band's UPPER edge as
    the expected rate (the obvious-looking choice) is wrong for these signals --
    it lets the dicrotic notch and the high-frequency noise inside the BP band
    register as extra "cycles", which inflated the pulse rate by ~40% in testing.
    """
    try:
        from scipy.signal import find_peaks
        y = _bandpass(x, fs, band)
        y = y - np.nanmean(y)
        n = y.size
        if n < 16 or fs <= 0 or band[0] <= 0:
            return {}
        period_s = 1.0 / float(np.sqrt(band[0] * band[1]))
        smooth_s, dist_s, prom = 0.25 * period_s, 0.6 * period_s, 0.35
        k = int(max(1, round(smooth_s * fs)))
        if 1 < k < n:
            y = np.convolve(y, np.ones(k) / k, mode='same')
        sd = float(np.nanstd(y))
        if not np.isfinite(sd) or sd == 0:
            return {}
        pk, _ = find_peaks(y, distance=int(max(1, round(dist_s * fs))),
                           prominence=prom * sd)
        dur_min = n / fs / 60.0
        return {
            'cycles': int(pk.size),
            'cycle_count_per_min': float(pk.size / dur_min) if dur_min > 0 else None,
            'cycle_count_hz': float(pk.size / (n / fs)),
            'cycle_count_params': {'smooth_s': float(smooth_s),
                                   'min_separation_s': float(dist_s),
                                   'prominence_std': prom},
        }
    except Exception as exc:
        return {'error': str(exc)}


def step_update_rate(x: np.ndarray, fs: float) -> dict:
    """Update rate of a STEP-HELD series -- the meaningful rate for one.

    ``LA Systolic``, ``LA Mean`` and ``BP Dia`` hold a value between beats, so
    their spectrum is a STAIRCASE. A spectral estimate is the wrong tool for
    that and fails loudly-in-the-wrong-direction: because the long flat plateaus
    dominate, the in-band Welch peak pins to the LOWER edge of the search band.
    Measured on F001_T1: all three report 0.6 Hz = 36.0/min against a ~90/min
    pulse, i.e. an artefact of the band edge rather than a beat rate.

    The meaningful quantity is instead how often the value CHANGES, which is the
    beat rate and cross-checks the waveform's ``welch_peak_hz``.
    """
    finite = x[np.isfinite(x)]
    if finite.size < 2 or fs <= 0:
        return {}
    idx = np.flatnonzero(np.diff(finite) != 0) + 1
    dur_min = finite.size / fs / 60.0
    out = {
        'n_updates': int(idx.size),
        'update_rate_per_min': float(idx.size / dur_min) if dur_min > 0 else None,
        'update_rate_hz': float(idx.size / (finite.size / fs)),
        'method': 'value-change count (step-held series; a spectral peak is '
                  'not meaningful for a staircase)',
    }
    if idx.size > 1:
        gaps = np.diff(idx) / fs
        out['update_interval_s_median'] = float(np.median(gaps))
        out['update_interval_s_iqr'] = float(np.subtract(
            *np.percentile(gaps, [75, 25])))
    return out


def eda_tonic_phasic(x: np.ndarray, fs: float) -> dict:
    """Split EDA into its slow (tonic) and fast (phasic) components.

    Deliberately NOT an event detector. Counting "SCRs" here was tried and
    removed: with NeuroKit2 absent the only option was a peak-picking heuristic,
    and on these recordings it produced meaningless rates (2-20 "events"/min,
    including 14.9/min on F001_T1 whose whole range is 0.13 uS of ADC
    quantisation). A labelled-but-bogus number is worse than no number, so the
    report keeps only descriptors that are robust at any amplitude: where the
    slow level drifted to, and how much fast activity rides on top of it.
    """
    finite = x[np.isfinite(x)]
    if finite.size < 16 or fs <= 0:
        return {}
    window_s = 20.0                                # ~ the 0.05 Hz split
    k = int(max(1, round(window_s * fs)))
    if 1 < k < finite.size:
        tonic = np.convolve(finite, np.ones(k) / k, mode='same')
    else:
        tonic = np.full_like(finite, np.nanmean(finite))
    phasic = finite - tonic
    out = {
        'tonic_phasic_split_s': window_s,
        'tonic_mean': float(np.mean(tonic)),
        'tonic_ptp': float(np.ptp(tonic)),
        'tonic_std': float(np.std(tonic)),
        'phasic_std': float(np.std(phasic)),
        'phasic_ptp': float(np.ptp(phasic)),
        'note': f'simple split at a {window_s:g} s moving average; NOT a '
                f'validated EDA decomposition and NOT an SCR event count',
    }
    out['phasic_to_tonic_std'] = (float(out['phasic_std'] / out['tonic_std'])
                                  if out['tonic_std'] > 0 else None)
    return out


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _import_pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f'[plot] matplotlib unavailable, skipping figures ({exc})')
        return None


def _envelope(t: np.ndarray, x: np.ndarray, max_points: int):
    """Min/max envelope decimation FOR DISPLAY ONLY (statistics are unaffected).

    Only used when ``--plot_max_points`` is set; with the default 0 the trace is
    drawn sample-for-sample at its original rate.
    """
    n = x.size
    if max_points <= 0 or n <= max_points:
        return t, x
    step = int(np.ceil(n / max(1.0, max_points / 2.0)))
    n_bins = int(np.ceil(n / step))
    pad = n_bins * step - n
    xp = np.concatenate([x, np.full(pad, np.nan)])
    tp = np.concatenate([t, np.full(pad, np.nan)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')           # all-NaN bins
        xb = xp.reshape(n_bins, step)
        tb = tp.reshape(n_bins, step)
        lo, hi = np.nanmin(xb, axis=1), np.nanmax(xb, axis=1)
        tmid = np.nanmean(tb, axis=1)
    return (np.concatenate([tmid, tmid]), np.concatenate([lo, hi]))


def plot_channel(report: dict, out_dir: str, zoom_s: float,
                 plot_max_points: int = 0) -> str:
    """Save the per-channel figure; returns the path ('' when skipped).

    Three panels: the full session at the ORIGINAL sample rate (spanning the
    top), a zoom on the first ``zoom_s`` seconds, and the amplitude
    distribution.

    Deliberately shows the raw signal only -- no band-passed overlay and no
    Welch PSD panel. Those are diagnostic views of how the frequency estimate
    was obtained; they are reported through ``frequency`` in the JSON (with the
    band-edge / estimator-agreement warnings) and are not duplicated here. The
    figure is numbers-free for the same reason: the statistics live in the
    per-channel JSON and on stdout, not printed over the plot.
    """
    plt = _import_pyplot()
    if plt is None:
        return ''
    x, fs = report['_x'], report['fs_hz']
    st = report['stats']
    m = report['channel_meta']
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return ''
    t = np.arange(x.size) / fs
    t_d, x_d = _envelope(t, np.where(np.isfinite(x), x, np.nan),
                         plot_max_points)

    # constrained layout (NOT tight_layout): tight_layout warns "This figure
    # includes Axes that are not compatible with tight_layout" for axes built
    # from an explicit fig.add_gridspec, which is what this used to do
    fig = plt.figure(figsize=(12.5, 8.0), layout='constrained')
    mosaic = fig.subplot_mosaic([['full', 'full'], ['zoom', 'hist']],
                                height_ratios=[1.25, 1.0])
    ax_full, ax_zoom, ax_hist = mosaic['full'], mosaic['zoom'], mosaic['hist']
    fig.suptitle(f"{report['session']} - {m['name']} "
                 f"({report['raw_file_name']}) - raw {fs:g} Hz, "
                 f"no resampling", fontsize=12)

    # ---- full trace (native sample rate) --------------------------------- #
    ax = ax_full
    ax.plot(t_d, x_d, lw=0.5, color='tab:blue')
    if fs * zoom_s < x.size:                      # mark the zoom region
        ax.axvspan(0, zoom_s, facecolor='tab:orange', alpha=0.18, lw=0)
    if 'min' in st:
        for y in (st['min'], st['max']):
            ax.axhline(y, color='0.5', lw=0.6, ls='--')
        ax.set_ylim(st['min'] - 0.05 * st['peak_to_peak'] - 1e-12,
                    st['max'] + 0.05 * st['peak_to_peak'] + 1e-12)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f"{m['name']} ({m['unit']})")
    ax.set_title(f"full session, {x.size} samples / {report['duration_s']:.1f} s"
                 + (" (min/max envelope)" if plot_max_points else ''))
    ax.grid(alpha=0.3)

    # ---- zoom on the first zoom_s seconds -------------------------------- #
    ax = ax_zoom
    n_z = int(min(x.size, max(1, round(zoom_s * fs))))
    seg = np.where(np.isfinite(x[:n_z]), x[:n_z], np.nan)
    ax.plot(np.arange(n_z) / fs, seg, lw=0.9, color='tab:blue')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f"{m['name']} ({m['unit']})")
    ax.set_title(f'first {n_z / fs:.1f} s zoom')
    ax.grid(alpha=0.3)

    # ---- amplitude distribution ------------------------------------------ #
    ax = ax_hist
    ax.hist(finite, bins=60, color='0.62', edgecolor='0.35', lw=0.3)
    for val, col, lab in ((st.get('mean'), 'tab:red', 'mean'),
                          (st.get('median'), 'tab:green', 'median')):
        if val is not None:
            ax.axvline(val, color=col, lw=1.0, ls='--', label=lab)
    ax.set_xlabel(f"{m['name']} ({m['unit']})")
    ax.set_ylabel('samples')
    ax.set_title('amplitude distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, f"{m['name']}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_overview(reports, out_dir: str, plot_max_points: int = 0) -> str:
    """Stack the requested channels' full native-rate traces on one time axis."""
    if len(reports) < 2:
        return ''
    plt = _import_pyplot()
    if plt is None:
        return ''
    fig, axes = plt.subplots(len(reports), 1, figsize=(12.5, 2.2 * len(reports)),
                             sharex=True, squeeze=False, layout='constrained')
    for ax, rep in zip(axes[:, 0], reports):
        x, fs = rep['_x'], rep['fs_hz']
        t, xd = _envelope(np.arange(x.size) / fs,
                          np.where(np.isfinite(x), x, np.nan), plot_max_points)
        m = rep['channel_meta']
        ax.plot(t, xd, lw=0.5, color='tab:blue')
        ax.set_ylabel(f"{m['name']}\n({m['unit']})")
        st, fq = rep['stats'], rep['frequency']
        sub = (f"min {st['min']:.4g} / max {st['max']:.4g}"
               if 'min' in st else 'no finite samples')
        if 'welch_peak_hz' in fq:
            sub += (f"   est. {fq['welch_peak_hz']:.3f} Hz = "
                    f"{fq['welch_peak_per_min']:.1f} /min")
        ax.set_title(sub, fontsize=9, loc='left')
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel('time (s)')
    fig.suptitle(f"{reports[0]['session']} - raw physiology overview "
                 f"({reports[0]['fs_hz']:g} Hz, no resampling)", fontsize=12)
    path = os.path.join(out_dir, 'overview.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_family_overview(components, family: dict, session: str, fs: float,
                         out_dir: str, zoom_s: float,
                         plot_max_points: int = 0) -> str:
    """Every raw file of one channel's family, over the whole session.

    Triggered by the channel (``bp`` -> BP family, ``resp`` -> Resp family).
    Two layouts, chosen by whether the family shares a unit:

    * SHARED unit (BP: all mmHg) -- all series overlay on one axis, and the
      third panel is a box comparison of every series (box = IQR, whiskers =
      min/max). Overlaying is what makes it readable: the pulse waveform
      oscillates inside the systolic/diastolic envelope.
    * MIXED units (Resp: volts vs breaths/min) -- the extra unit goes on a
      right-hand axis instead of being drawn on a scale it does not share, and
      the third panel is the amplitude distribution of the waveform.

    Written to ``<output_dir>/<FAMILY>_overview.png``. Deliberately
    numbers-free -- the per-file statistics live in the channel's JSON
    (``family_components``) and on stdout.
    """
    have = [c for c in components if c['_x'].size]
    if not have:
        return ''
    plt = _import_pyplot()
    if plt is None:
        return ''

    units = []
    for c in have:
        if c['unit'] not in units:
            units.append(c['unit'])
    mixed = len(units) > 1

    fig = plt.figure(figsize=(12.5, 9.0), layout='constrained')
    mosaic = fig.subplot_mosaic([['full', 'full'], ['zoom', 'side']],
                                height_ratios=[1.4, 1.0])
    ax_full, ax_zoom, ax_side = mosaic['full'], mosaic['zoom'], mosaic['side']
    twin_full = ax_full.twinx() if mixed else None
    twin_zoom = ax_zoom.twinx() if mixed else None

    def _axis(comp, primary, twin):
        """The right-hand axis when this component's unit is not the primary one.

        Returned explicitly (never via ``or``, which would depend on the
        truthiness of an Axes object).
        """
        if mixed and twin is not None and comp['unit'] != units[0]:
            return twin
        return primary

    # ---- full session, native rate --------------------------------------- #
    for c in have:
        t, xd = _envelope(np.arange(c['_x'].size) / fs,
                          np.where(np.isfinite(c['_x']), c['_x'], np.nan),
                          plot_max_points)
        ax = _axis(c, ax_full, twin_full)
        ax.plot(t, xd, lw=0.6, color=c['color'],
                label=f"{c['label']}  ({c['file']})")
    ax_full.set_xlabel('time (s)')
    ax_full.set_ylabel(f"{family['name']} ({units[0]})")
    ax_full.set_title(f"full session, {fs:g} Hz original rate "
                      f"(waveform = continuous, the others = per-beat step-held)")
    if mixed:
        twin_full.set_ylabel(units[1])
    _family_legend(ax_full, twin_full, fontsize=8, ncol=2)

    # ---- zoom: the first zoom_s seconds ---------------------------------- #
    n_z = int(min(have[0]['_x'].size, max(1, round(zoom_s * fs))))
    for c in have:
        seg = np.where(np.isfinite(c['_x'][:n_z]), c['_x'][:n_z], np.nan)
        ax = _axis(c, ax_zoom, twin_zoom)
        ax.plot(np.arange(n_z) / fs, seg, lw=1.0, color=c['color'],
                label=c['label'])
    ax_zoom.set_xlabel('time (s)')
    ax_zoom.set_ylabel(f"{family['name']} ({units[0]})")
    ax_zoom.set_title(f'first {n_z / fs:.1f} s zoom')
    if mixed:
        twin_zoom.set_ylabel(units[1])
    _family_legend(ax_zoom, twin_zoom, fontsize=8)

    # ---- third panel ------------------------------------------------------ #
    if mixed:
        # units differ, so a cross-series box plot would compare unlike things;
        # show the waveform's own distribution instead
        prim = next((c for c in have if c['kind'] == 'continuous waveform'),
                    have[0])
        data = prim['_x'][np.isfinite(prim['_x'])]
        ax_side.hist(data, bins=60, color='0.62', edgecolor='0.35', lw=0.3)
        for val, col, lab in ((prim['stats'].get('mean'), 'tab:red', 'mean'),
                              (prim['stats'].get('median'), 'tab:green',
                               'median')):
            if val is not None:
                ax_side.axvline(val, color=col, lw=1.0, ls='--', label=lab)
        ax_side.set_xlabel(f"{prim['label']} ({prim['unit']})")
        ax_side.set_ylabel('samples')
        ax_side.set_title('amplitude distribution', fontsize=10)
        ax_side.legend(fontsize=8)
    else:
        # whis=(0, 100) makes the whisker caps the data min/max, which is the
        # point here (systolic peak / diastolic trough); fliers are suppressed
        # because a 65k-sample outlier cloud would bury the boxes.
        ax_side.boxplot([c['_x'][np.isfinite(c['_x'])] for c in have],
                        orientation='horizontal', whis=(0, 100),
                        showfliers=False, widths=0.6)
        ax_side.set_yticks(range(1, len(have) + 1))
        ax_side.set_yticklabels([c['label'] for c in have], fontsize=8)
        ax_side.invert_yaxis()                    # first component on top
        ax_side.set_xlabel(f"{family['name']} ({units[0]})")
        ax_side.set_title('amplitude range (box = IQR, whiskers = min/max)',
                          fontsize=10)
    ax_side.grid(alpha=0.3, axis='x')

    fig.suptitle(f"{session} - {family['title']} overview "
                 f"(original rate, no resampling)", fontsize=12)
    path = os.path.join(out_dir, f"{family['name']}_overview.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _family_legend(ax, twin, **kwargs):
    """One legend covering both y axes of a family panel (twin may be None)."""
    handles, labels = ax.get_legend_handles_labels()
    if twin is not None:
        h2, l2 = twin.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    if handles:
        ax.legend(handles, labels, loc='upper right', framealpha=0.85, **kwargs)


# --------------------------------------------------------------------------- #
# one session
# --------------------------------------------------------------------------- #
def inspect_session(raw_root: str, subject: str, task: str, channels,
                    output_dir: str, phys_fs_cfg: float, fps_rgb: float,
                    band_cfg, zoom_s: float, plot: bool,
                    plot_max_points: int) -> dict:
    """Inspect the requested ``channels`` of one ``(subject, task)`` session."""
    session = f'{subject}_{task}'
    phys_dir = os.path.join(raw_root, 'Physiology', subject, task)
    out_dir = os.path.join(output_dir, session)
    summary = {
        'schema': SCHEMA, 'session': session, 'subject': subject, 'task': task,
        'raw_root': raw_root, 'physiology_dir': phys_dir,
        'output_dir': out_dir, 'channels': {}, 'warnings': [],
    }

    if not os.path.isdir(phys_dir):
        summary['ok'] = False
        summary['error'] = (f'no physiology directory {phys_dir} '
                            f'(available: see --list)')
        return summary
    os.makedirs(out_dir, exist_ok=True)

    # ---- load the requested channels once, up front ------------------------ #
    # (loaded before the sample rate is settled, because estimating fs needs the
    # number of physiology samples: fs = n_samples / RGB_duration)
    # resolve every requested channel through channel_raw_file -- NOT
    # prep.resolve_channels, which only knows the canonical three and would
    # silently skip HR
    resolved = {c: prep.find_channel_file(phys_dir, channel_raw_file(c))
                for c in channels}
    loaded, load_info = {}, {}
    for key in channels:
        path = resolved.get(key)
        if path is None:
            continue
        x, n_lines, n_bad = load_channel_tolerant(path)
        loaded[key] = x
        load_info[key] = {'path': path, 'n_lines': n_lines, 'n_bad': n_bad}
    if not loaded:
        summary['ok'] = False
        summary['error'] = (
            f'none of the requested channels has a raw file in {phys_dir} '
            f'(looking for {[channel_raw_file(c) for c in channels]})')
        return summary

    # ---- sample rate: nominal, or estimated from the RGB duration --------- #
    fs, fs_source, n_frames, fs_note = float(phys_fs_cfg), 'nominal --phys_fs', 0, ''
    if phys_fs_cfg <= 0:
        dur_rgb, n_frames, fs_note = rgb_duration_estimate(
            raw_root, subject, task, fps_rgb)
        n_med = int(np.median([x.size for x in loaded.values()]))
        if dur_rgb > 0 and n_med > 0:
            fs = n_med / dur_rgb
            fs_source = 'estimated from RGB duration'
            summary['fs_estimate_note'] = fs_note
            summary['warnings'].append(
                f'fs estimated as {fs:.2f} Hz = {n_med} samples / '
                f'{dur_rgb:.1f} s of RGB ({fs_note}); the raw files carry no '
                f'time column, so this assumes the RGB covers the whole session')
        else:
            summary['ok'] = False
            summary['error'] = (f'--phys_fs 0 asked for an estimate but the '
                                f'duration could not be derived: {fs_note}')
            return summary
    summary.update({'fs_hz': fs, 'fs_source': fs_source,
                    'n_rgb_frames': n_frames,
                    'raw_file_names': {c: os.path.basename(channel_raw_file(c))
                                       for c in CHANNEL_ORDER}})

    # ---- per channel ------------------------------------------------------ #
    lengths, reports = {}, []
    for key in channels:
        meta = dict(CHANNEL_META[key])
        meta['raw_file_needle'] = channel_raw_file(key)
        info = load_info.get(key)
        path = info['path'] if info else None
        rep = {'channel': key, 'channel_meta': meta, 'session': session,
               'raw_file': path or '',
               'raw_file_name': os.path.basename(path) if path else '',
               'fs_hz': fs, 'fs_source': fs_source, 'warnings': [], 'extra': {}}
        if info is None:
            rep['error'] = (f'no raw file matching {channel_raw_file(key)} '
                            f'in {phys_dir}')
            summary['warnings'].append(f'{meta["name"]}: {rep["error"]}')
            summary['channels'][key] = _public(rep)
            continue

        x, n_lines, n_bad = loaded[key], info['n_lines'], info['n_bad']
        rep['_x'] = x                              # kept for the plots only
        rep['raw_file_bytes'] = os.path.getsize(path)
        rep['n_file_lines'] = n_lines
        if n_bad:
            rep['n_bad_lines'] = n_bad
            rep['warnings'].append(
                f'{n_bad} of {n_lines} lines in {os.path.basename(path)} were '
                f'not parsable as a single float and were skipped')
        if x.size == 0:
            rep['error'] = 'file contains no parsable values'
            summary['warnings'].append(f'{meta["name"]}: {rep["error"]}')
            summary['channels'][key] = _public(rep)
            continue

        rep['stats'] = basic_stats(x)
        rep['duration_s'] = float(x.size / fs)
        lengths[key] = x.size
        band = band_cfg.get(key) or meta['band']
        # ``analysis`` decides which rate is meaningful here. A step-held series
        # (HR) deliberately gets NO spectral estimate: the staircase would just
        # pin the Welch peak to the lower band edge -- the artefact that made the
        # BP family report 36.0/min for a ~90 bpm pulse.
        if meta['analysis'] == 'step':
            rep['frequency'] = {}
            rep['step_update_rate'] = step_update_rate(x, fs)
        else:
            rep['frequency'] = estimate_frequency(x, fs, band)
        if meta['periodic']:
            # the cycle counter keeps the CHANNEL's own physiological band even
            # when --band moved the spectral search band, so the reference
            # estimate stays stable while the user tunes the spectrum
            rep['frequency'].update(cycle_count_rate(x, fs, meta['band']))
        elif meta['analysis'] == 'slow':
            # EDA is an event/slow-drift signal: neither a "dominant frequency"
            # nor an autocorrelation period is a meaningful study variable for
            # it, so those fields (and the cross-method agreement metric built
            # on them) are dropped rather than reported and misread. Only the
            # descriptive in-band spectral peak and the SCR rate are kept.
            rep['frequency']['periodic'] = False
            rep['frequency']['band_note'] = meta['band_note']
            for drop in ('autocorr_hz', 'autocorr_per_min', 'autocorr_peak_r',
                         'autocorr_at_band_edge', 'method_agreement_pct'):
                rep['frequency'].pop(drop, None)
            rep['extra'] = eda_tonic_phasic(x, fs)
        fq = rep['frequency']
        if meta['periodic'] and (fq.get('welch_peak_at_band_edge')
                                 or fq.get('autocorr_at_band_edge')):
            rep['warnings'].append(
                f"the frequency peak sits ON the edge of the "
                f"{band[0]:g}-{band[1]:g} Hz search band, so the true peak may "
                f"lie outside it -- re-run with --band to widen/move the band")
        ref = fq.get('cycle_count_per_min')
        if meta['periodic'] and ref and fq.get('welch_peak_per_min'):
            diff_pct = abs(ref - fq['welch_peak_per_min']) \
                / max(ref, fq['welch_peak_per_min']) * 100.0
            fq['welch_vs_cycle_count_pct'] = float(diff_pct)
            if diff_pct > 25.0:
                rep['warnings'].append(
                    f'rate estimators disagree: Welch {fq["welch_peak_per_min"]:.1f}'
                    f'/min vs peak counting {ref:.1f}/min ({diff_pct:.0f}% apart) '
                    f'-- treat the estimate as approximate (check the zoom panel '
                    f'and the search band)')
        if meta['periodic'] and fq.get('cycle_count_per_min'):
            # Record WHICH number to use instead of leaving the choice to the
            # reader of the JSON: the spectral peak is the headline estimate
            # only when nothing flagged it; otherwise peak counting (which makes
            # no stationarity assumption) is the one to trust.
            flagged = bool(fq.get('welch_peak_at_band_edge')
                           or fq.get('autocorr_at_band_edge')) \
                or not fq.get('welch_peak_per_min') \
                or fq.get('welch_vs_cycle_count_pct', 0.0) > 25.0
            fq['preferred_estimate'] = 'cycle_count' if flagged else 'welch_peak'
            fq['preferred_estimate_per_min'] = float(fq['cycle_count_per_min']
                                                     if flagged else
                                                     fq['welch_peak_per_min'])
            if flagged:
                rep['warnings'].append(
                    f"the Welch peak looks unreliable here, so the preferred "
                    f"estimate is the peak-counting rate "
                    f"({fq['preferred_estimate_per_min']:.1f} /min; see "
                    f"preferred_estimate in the JSON)")
        # dropout / constant / railing checks (shared with the family
        # components -- see quality_warnings)
        rep['warnings'].extend(quality_warnings(rep['stats'], meta['unit']))
        if rep['duration_s'] < 4.0:
            rep['warnings'].append(
                f'only {rep["duration_s"]:.1f} s of data; the frequency '
                f'estimate is unreliable')
        if meta['periodic'] and 'welch_peak_hz' not in rep['frequency']:
            rep['warnings'].append('dominant frequency could not be estimated')

        rep['figure'] = plot_channel(rep, out_dir, zoom_s, plot_max_points) \
            if plot else ''
        reports.append(rep)
        summary['channels'][key] = _public(rep)

    # ---- raw files that belong with a channel (see CHANNEL_FAMILIES) ------- #
    # "when the target is BP, draw an overview of BP with BP_mmHg / BP Dia /
    # LA Mean / LA Systolic"; likewise Resp_Volts / Respiration Rate_BPM for
    # Resp. Driven by the table, so a third family only needs a table entry.
    for fam_key, family in CHANNEL_FAMILIES.items():
        fam_rep = summary['channels'].get(fam_key)
        if fam_key not in channels or fam_rep is None or 'error' in fam_rep:
            continue
        comps = load_family_components(
            phys_dir, fam_key, fs,
            band_cfg.get(fam_key) or CHANNEL_META[fam_key]['band'])
        if not comps:
            continue
        if plot:
            fpath = plot_family_overview(comps, family, session, fs, out_dir,
                                         zoom_s, plot_max_points)
            if fpath:
                fam_rep['family_overview_figure'] = fpath
        # _public drops the arrays, so the components serialise as plain
        # statistics; family_files lists them even when unplotted
        fam_rep['family'] = family['name']
        fam_rep['family_components'] = [_public(c) for c in comps]
        fam_rep['family_files'] = [c['file'] for c in comps]
        missing = [needle for _, needle, *_ in family['components']
                   if needle not in fam_rep['family_files']]
        if missing:
            fam_rep['warnings'].append(
                f"{family['title']} incomplete: {missing} not present in "
                f'{phys_dir}')
        # a component's own data-quality problems belong in the channel's
        # warning list too, labelled so it is clear which file is at fault
        for comp in comps:
            for w in comp.get('warnings', ()):
                fam_rep['warnings'].append(f"{comp['label']}: {w}")

    # cross-channel length check: prepare_bp4d uses the MEDIAN length, so a
    # channel that is markedly shorter/longer silently shifts against the rest
    if len(lengths) > 1 and min(lengths.values()) > 0:
        spread = (max(lengths.values()) - min(lengths.values())) \
            / float(np.median(list(lengths.values())))
        summary['length_spread_frac'] = float(spread)
        if spread > 0.01:
            summary['warnings'].append(
                'channel lengths differ by '
                f'{spread * 100:.2f}% ({lengths}); prepare_bp4d.py aligns the '
                'channels on the MEDIAN length, so the shorter channels are '
                'time-stretched against the others')
            for rep in reports:
                rep['warnings'].append('channel lengths differ within session')

    if plot and len(reports) >= 2:
        summary['overview_figure'] = plot_overview(reports, out_dir,
                                                   plot_max_points)
    summary['ok'] = all('error' not in rep for rep in reports) and bool(reports)
    return summary


def _public(rep: dict) -> dict:
    """Report dict without the (large) in-memory array."""
    return {k: v for k, v in rep.items() if not k.startswith('_')}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_channels(spec: str):
    spec = (spec or '').strip()
    if not spec or spec.lower() in ('all', '*'):
        return list(CHANNEL_ORDER)
    out = []
    for tok in spec.replace(';', ',').split(','):
        t = tok.strip().lower().replace('-', '_').replace(' ', '_')
        if not t:
            continue
        if t not in CHANNEL_ALIASES:
            raise SystemExit(
                f'unknown channel {tok!r}; use BP, Resp, EDA or all '
                f'(aliases: {", ".join(sorted(CHANNEL_ALIASES))})')
        k = CHANNEL_ALIASES[t]
        if k not in out:
            out.append(k)
    return [k for k in CHANNEL_ORDER if k in out] or list(CHANNEL_ORDER)


def _parse_list(spec: str):
    return [s.strip() for s in (spec or '').replace(';', ',').split(',')
            if s.strip()]


def _parse_band(spec: str):
    """``--band 0.5,4`` -> {'bp': (0.5, 4.0)} / ('all' key when no channel)."""
    spec = (spec or '').strip()
    if not spec:
        return {}
    if ':' in spec:                               # CHANNEL:lo,hi
        ch, _, rng = spec.partition(':')
        keys = _parse_channels(ch)
    else:
        keys, rng = list(CHANNEL_ORDER), spec
    nums = [float(v) for v in rng.replace(';', ',').split(',') if v.strip()]
    if len(nums) != 2 or nums[0] >= nums[1]:
        raise SystemExit(f'--band expects lo,hi in Hz (lo < hi), got {spec!r}')
    return {k: (nums[0], nums[1]) for k in keys}


def get_args(argv=None):
    p = argparse.ArgumentParser(
        'BP4D raw physiology inspection', add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--subject', default='',
                   help='subject id(s), comma-separated, e.g. F001 or F001,F002')
    p.add_argument('--task', default='',
                   help='task id(s), comma-separated (T1) or "all"')
    p.add_argument('--channel', '--channels', '--target', dest='channel',
                   default='all',
                   help='channel(s): BP | Resp | EDA | all (comma-separated; '
                        'case-insensitive) - default all')
    p.add_argument('--raw_root', default=default_raw_root(),
                   help='raw BP4D root (default: $RAW_DATA_PATH, else the '
                        'in-repo data/raw/BP4D)')
    p.add_argument('--output_dir', default=default_output_dir(),
                   help='inspect_data ROOT; <output_dir>/<subject>_<task>/ is '
                        'created (default: $OUTPUT_DIR/inspect_data, else '
                        '<repo>/output/inspect_data)')
    p.add_argument('--phys_fs', default=1000.0, type=float,
                   help='raw physiology sample rate in Hz (BP4D nominal, '
                        'default 1000); 0 = estimate from the RGB frame count '
                        'and --fps_rgb. The ORIGINAL samples are kept either '
                        'way - nothing is resampled')
    p.add_argument('--fps_rgb', default=25.0, type=float,
                   help='RGB frame rate used by --phys_fs 0 (default 25)')
    p.add_argument('--band', default='',
                   help='override the frequency search band: "lo,hi" in Hz for '
                        'all channels, or "CHANNEL:lo,hi" for one')
    p.add_argument('--zoom_s', default=10.0, type=float,
                   help='seconds shown in the zoom panel (default 10)')
    p.add_argument('--plot_max_points', default=0, type=int,
                   help='DISPLAY-ONLY min/max envelope decimation per trace; '
                        '0 = off (plot every original sample)')
    p.add_argument('--no-plot', dest='plot', action='store_false', default=True,
                   help='write the JSON summaries only, skip the figures')
    p.add_argument('--list', action='store_true',
                   help='list the (subject, task) sessions found in the raw '
                        'tree and exit')
    return p.parse_args(argv)


def main() -> int:
    args = get_args()
    raw_root = args.raw_root
    available = discover_sessions(raw_root)
    if not os.path.isdir(raw_root):
        raise SystemExit(f'raw root not found: {raw_root}\n'
                         f'set $RAW_DATA_PATH or pass --raw_root')
    if args.list:
        print(f'raw root: {raw_root}')
        print(f'{len(available)} session(s) under Physiology/:')
        subj = sorted({s for s, _ in available})
        for s in subj:
            tasks = [t for ss, t in available if ss == s]
            print(f'  {s}: {" ".join(tasks)}')
        if not available:
            print('  (none - is this the right --raw_root?)')
        return 0

    if not args.subject:
        raise SystemExit('--subject is required (or use --list); '
                         f'available: {", ".join(sorted({s for s, _ in available}))}')
    if not args.task:
        raise SystemExit('--task is required (or "all", or use --list)')

    channels = _parse_channels(args.channel)
    band_cfg = _parse_band(args.band)
    subjects = _parse_list(args.subject)
    tasks = _parse_list(args.task)
    known_tasks = {}
    for s, t in available:
        known_tasks.setdefault(s, []).append(t)

    wanted = []
    for s in subjects:
        if s not in known_tasks:
            raise SystemExit(f'subject {s!r} not found under '
                             f'{os.path.join(raw_root, "Physiology")}; '
                             f'available: {", ".join(sorted(known_tasks))}')
        ts = known_tasks[s] if any(t.lower() == 'all' for t in tasks) else tasks
        for t in ts:
            if t not in known_tasks[s]:
                raise SystemExit(f'task {t!r} not found for {s!r}; available: '
                                 f'{", ".join(known_tasks[s])}')
            wanted.append((s, t))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'raw root   : {raw_root}')
    print(f'output root: {args.output_dir}')
    print(f'channels   : {", ".join(CHANNEL_META[c]["name"] for c in channels)}')
    print(f'sessions   : {", ".join(f"{s}_{t}" for s, t in wanted)}')
    print()

    summaries, n_ok = [], 0
    for subject, task in wanted:
        print(f'=== {subject}_{task} ===')
        try:
            summary = inspect_session(
                raw_root=raw_root, subject=subject, task=task,
                channels=channels, output_dir=args.output_dir,
                phys_fs_cfg=args.phys_fs, fps_rgb=args.fps_rgb,
                band_cfg=band_cfg, zoom_s=args.zoom_s, plot=args.plot,
                plot_max_points=args.plot_max_points)
        except Exception as exc:
            print(f'  [FAIL] {subject}/{task}: {type(exc).__name__}: {exc}')
            summaries.append({'session': f'{subject}_{task}', 'ok': False,
                              'error': f'{type(exc).__name__}: {exc}'})
            continue

        os.makedirs(summary['output_dir'], exist_ok=True)
        if not summary.get('ok', False):
            print(f"  [FAIL] {summary.get('error', 'no channel inspected')}")
        for key in channels:
            rep = summary['channels'].get(key)
            if not rep or 'error' in rep:
                print(f"  [skip] {CHANNEL_META[key]['name']}: "
                      f"{rep.get('error') if rep else 'not inspected'}")
                continue
            st, fq = rep['stats'], rep['frequency']
            ur = rep.get('step_update_rate') or {}
            # build the numeric tail from whatever this channel's 'analysis'
            # actually produced, so a step-held channel shows its update rate
            # instead of a meaningless 'est.freq=n/a'
            parts = [f"n={st['n_samples']:<7}" if 'n_samples' in st else 'n=?',
                     f"{rep['duration_s']:7.2f} s"]
            if 'min' in st:
                parts += [f"min={st['min']:<11.5g}",
                          f"max={st['max']:<11.5g}",
                          f"mean={st['mean']:<11.5g}"]
            if 'welch_peak_hz' in fq:
                parts.append(f"est.freq={fq['welch_peak_hz']:.4f} Hz "
                             f"({fq['welch_peak_per_min']:.2f}/min)")
            if 'cycle_count_per_min' in fq:
                parts.append(f"cycles={fq['cycle_count_per_min']:.2f}/min")
            if 'update_rate_per_min' in ur:
                parts.append(f"updates={ur['update_rate_per_min']:.1f}/min "
                             f"({ur['n_updates']} steps)")
            print(f"  {CHANNEL_META[key]['name']:<4} " + '  '.join(parts)
                  + ('  <- ' + '; '.join(rep['warnings'])
                     if rep['warnings'] else ''))
            # the extra raw files that belong with this channel (see
            # CHANNEL_FAMILIES); units are printed because a family can mix them
            for comp in rep.get('family_components', ()):
                cst = comp['stats']
                if 'min' not in cst:
                    continue
                if comp['derived']:
                    # step-held series: a spectral peak would just report the
                    # band edge, so the value-change rate is the rate shown
                    ur = comp.get('step_update_rate', {})
                    rate = (f"updates={ur['update_rate_per_min']:.1f}/min "
                            f"({ur['n_updates']} steps)"
                            if 'update_rate_per_min' in ur else 'updates=n/a')
                else:
                    cfq = comp.get('frequency', {})
                    rate = (f"est.freq={cfq['welch_peak_per_min']:.1f}/min"
                            if 'welch_peak_per_min' in cfq else 'est.freq=n/a')
                print(f"      + {comp['label']:<19} {comp['file']:<26}"
                      f"min={cst['min']:<10.5g} max={cst['max']:<10.5g}"
                      f"mean={cst['mean']:<10.5g} [{comp['unit']}] {rate}"
                      f"{'  (step-held)' if comp['derived'] else ''}"
                      f"{'  <- ' + '; '.join(comp['warnings']) if comp.get('warnings') else ''}")
        summary_path = os.path.join(summary['output_dir'], 'physio_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        for key in channels:
            rep = summary['channels'].get(key)
            if rep and 'error' not in rep:
                with open(os.path.join(summary['output_dir'],
                                       f"{CHANNEL_META[key]['name']}.json"),
                          'w') as f:
                    json.dump(rep, f, indent=2)
        print(f"  -> {summary['output_dir']}/ "
              f"(figures: {len([k for k in channels if summary['channels'].get(k, {}).get('figure')])}"
              f", summary: {os.path.basename(summary_path)})")
        print()
        summaries.append(summary)
        n_ok += 1 if summary.get('ok') else 0

    out_index = os.path.join(args.output_dir, 'physio_index.json')
    with open(out_index, 'w') as f:
        json.dump({'schema': SCHEMA, 'raw_root': raw_root,
                   'channels': channels, 'sessions': summaries}, f, indent=2)
    print(f'{n_ok}/{len(wanted)} session(s) inspected; index -> {out_index}')
    return 0 if n_ok == len(wanted) else 1


if __name__ == '__main__':
    sys.exit(main())
