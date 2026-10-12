"""Tier-1 / Tier-2 waveform metrics (offline evaluation).

Operate on 1D waveforms given as numpy arrays of shape ``[T]`` or ``[N, T]``
(predicted / ground truth). SciPy is only required for the spectral metrics.
"""
import numpy as np

__all__ = ['time_domain_metrics', 'spectral_metrics', 'to_numpy',
           'aggregate_metrics', 'breakdown_by_label', 'bandpass_fft',
           'band_limited_pearson', 'single_tone_template', 'bootstrap_ci',
           'estimate_rate_bpm', 'rate_metrics', 'rate_summary']

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


def _pearson_1d(a, b):
    """Pearson r of two 1-D arrays; 0.0 when either is (near-)constant."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_pearson(p, t):
    """Macro Pearson over the rows of two ``[N, T]`` arrays."""
    return float(np.mean([_pearson_1d(pi, ti) for pi, ti in zip(p, t)]))


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
    pearson = _mean_pearson(p, t)

    return {
        'mae': float(np.mean(mae)),
        'rmse': float(np.mean(rmse)),
        'pearson': float(pearson),
    }


#: Welch window used when NO band is requested. Kept at the historical value so
#: unbanded ``psd_mae`` stays comparable with previously recorded runs.
DEFAULT_NPERSEG = 256
#: How many Welch bins a requested ``band`` should contain for the dominant-
#: frequency metric to be informative (``argmax`` over 1 bin is constant 0).
DEFAULT_MIN_BAND_BINS = 3
#: Relative floor on a clip's de-meaned TARGET: below it the band/window carries
#: no target energy and the comparison is SKIPPED (a constant window -- e.g. a
#: railed -10 V respiration rail -- has no spectrum to match).
_PSD_REL_TOL = 1e-6

#: Lowest frequency counted as OUT OF BAND when ``hf_lo`` is not given: it is
#: raised to the eval band's upper edge when that is higher (so for a 1.0-2.5 Hz
#: BP band the window starts at 2.5 Hz). 2 Hz is above any plausible respiration
#: harmonic (resp is 0.1-1 Hz) and above the BP band, i.e. everything in it is
#: measurement/modelling garbage rather than physiology.
DEFAULT_HF_LO = 2.0

#: A target is ALSO degenerate for a spectral comparison when its Welch PSD
#: carries less than this fraction of its own variance. ``welch`` de-means PER
#: SEGMENT and only segments the first ``nperseg`` samples, so a target that is a
#: dead-sensor PLATEAU across the whole segment (with a brief excursion elsewhere
#: in the clip) has a PSD of ~0 while its max deviation is huge -- the older
#: max-deviation guard misses it, and it then scores a misleadingly GOOD
#: ``psd_mae`` (measured 2026-10-01: 4/75 clips of the TIR-ROI val split, all
#: F004_T3/F004_T7, PSD sum exactly 0.0 with std 1.0).
_DEGENERATE_PSD_REL = 0.01


def _resolve_nperseg(n, fs, band, min_band_bins=DEFAULT_MIN_BAND_BINS):
    """Welch window length that puts >= ``min_band_bins`` bins inside ``band``.

    The window is only ENLARGED when the historical default
    (:data:`DEFAULT_NPERSEG`) is too coarse for the requested band, so a band
    that already resolves fine is bit-identical to the old behaviour::

        nperseg = min(n, max(DEFAULT_NPERSEG, ceil(min_bins * fs / band_width)))

    WHY THIS EXISTS (measured 2026-10-01): at ``fs=100`` and
    ``nperseg=min(256, n)`` the bin spacing is ``100/256 = 0.3906 Hz``, so the
    whole RESP band ``(0.16, 0.4)`` -- 0.24 Hz wide -- contained EXACTLY ONE
    bin. ``dominant_freq_error_hz`` was therefore identically ``0.0`` (argmax
    over one bin) and ``psd_mae`` degenerated into "the fraction of clips whose
    banded PSD is zero". A band that is narrower than the achievable resolution
    cannot be scored this way; the largest window a clip of ``n`` samples allows
    is the best we can do (``1/n`` Hz bins), which is reported as
    ``psd_band_bins`` so a still-under-resolved band is visible, not silent.
    """
    n = int(n)
    if band is None or n <= 0:
        return int(min(DEFAULT_NPERSEG, n)) if n > 0 else int(DEFAULT_NPERSEG)
    width = float(band[1]) - float(band[0])
    if width <= 0:
        return int(min(DEFAULT_NPERSEG, n))
    want = int(np.ceil(int(min_band_bins) * float(fs) / width))
    return int(max(16, min(n, max(DEFAULT_NPERSEG, want))))


def spectral_metrics(pred, target, fs=100.0, band=None,
                     min_band_bins=DEFAULT_MIN_BAND_BINS, warn=True,
                     hf_lo=None):
    """Tier 2: Welch PSD consistency, dominant-frequency error, out-of-band energy.

    :param fs: sampling rate (Hz) of the waveforms
    :param band: optional (f_low, f_high) band-pass region to compare, e.g.
        (1.0, 2.5) for BP or (0.16, 0.4) for RESP.
    :param min_band_bins: minimum number of Welch bins the band should hold;
        the window is enlarged up to the clip length to try to reach it (see
        :func:`_resolve_nperseg`).
    :param warn: print a one-shot note when the band cannot be resolved well
        enough, or when clips were skipped for having no in-band target energy.
    :param hf_lo: lowest frequency counted as OUT OF BAND (default
        :data:`DEFAULT_HF_LO`, raised to ``band[1]`` when that is higher).

    Clips whose de-meaned TARGET is (near-)zero are SKIPPED rather than scored
    ``1.0``: with a unit-sum normalisation a constant target produces a
    zero spectrum and the difference against any prediction saturates, so
    averaging it in makes ``psd_mae`` a count of degenerate clips. The same goes
    for a target whose Welch PSD carries no energy relative to its own variance
    -- a dead-sensor plateau filling the whole Welch segment (see
    :data:`_DEGENERATE_PSD_REL`), which the max-deviation test cannot see. The
    skipped count is returned as ``psd_skipped`` and the in-band bin count as
    ``psd_band_bins`` (``None`` when no band was given).

    ``hf_power_rel`` / ``target_hf_power_rel`` are the OUT-OF-BAND energy of the
    prediction / of the target, in units of the target's TOTAL variance (the
    target is z-scored, so a perfect prediction measures ~0.003). WHY THEY ARE
    NEEDED (measured 2026-10-01): ``psd_mae`` normalises the PSD INSIDE the band
    and the Tier-1 metrics are dominated by the low-frequency amplitude, so a
    prediction that adds a broadband high-frequency floor -- the per-token
    waveform head emitting 16 phase-independent samples per visual token -- was
    invisible: the gamma=0.01 TIR-ROI run had x143 (4-8 Hz) / x553 (8-20 Hz) the
    target's out-of-band power while ``psd_mae``, Pearson AND the
    dominant-frequency error all read ~0. ``hf_lo_hz`` records the window used.
    """
    if _sp_signal is None:
        raise ImportError('spectral_metrics requires scipy (`pip install scipy`)')

    if hf_lo is None:
        hf_lo = (DEFAULT_HF_LO if band is None
                 else max(DEFAULT_HF_LO, float(band[1])))
    hf_lo = float(hf_lo)

    p = _flatten(pred)
    t = _flatten(target)
    if p.shape != t.shape:
        raise ValueError(f'Shape mismatch pred {p.shape} vs target {t.shape}')

    n = int(t.shape[1])
    nperseg = _resolve_nperseg(n, fs, band, min_band_bins)
    f_probe = np.fft.rfftfreq(nperseg, d=1.0 / float(fs))
    band_bins = (int(((f_probe >= band[0]) & (f_probe <= band[1])).sum())
                 if band is not None else None)

    # a clip whose de-meaned target is (near-)constant has nothing to compare
    t_energy = np.abs(t - t.mean(axis=1, keepdims=True)).max(axis=1)
    tol = _PSD_REL_TOL * float(t_energy.max()) if t_energy.size else 0.0

    psd_mae, dom_freq_err = [], []
    hf_rel, hf_ref = [], []          # out-of-band energy, target-relative
    n_skipped = 0
    for pi, ti, e_t in zip(p, t, t_energy):
        if e_t <= tol:
            n_skipped += 1
            continue
        f_p, pxx_p = _sp_signal.welch(pi, fs=fs, nperseg=min(nperseg, len(pi)))
        f_t, pxx_t = _sp_signal.welch(ti, fs=fs, nperseg=min(nperseg, len(ti)))

        # --- out-of-band energy ----------------------------------------- #
        # Computed on the FULL spectrum, BEFORE the in-band slice + unit-sum
        # normalisation below, which is exactly what makes the jitter invisible
        # to psd_mae. `pxx * df` is power, so the ratio is a power ratio.
        df = float(f_p[1] - f_p[0]) if f_p.size > 1 else 1.0
        tot = float(pxx_t.sum() * df)
        # A staircase target can pass the max-deviation guard above and still
        # have NO spectrum (a plateau that fills the whole Welch segment). Every
        # spectral comparison against it is meaningless AND the ratio below
        # would explode, so it is skipped on the spectrum instead.
        t_var = float(np.var(ti))
        if tot <= 0.0 or tot < _DEGENERATE_PSD_REL * t_var:
            n_skipped += 1
            continue
        hi = f_p >= hf_lo
        hf_rel.append(float(pxx_p[hi].sum() * df) / (tot + 1e-30))
        hf_ref.append(float(pxx_t[hi].sum() * df) / (tot + 1e-30))

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

    if warn:
        if band is not None and band_bins is not None and band_bins < min_band_bins:
            print(f'[metrics] NOTE: the eval band {tuple(band)} holds only '
                  f'{band_bins} Welch bin(s) at fs={fs:g} with nperseg={nperseg} '
                  f'(>= {min_band_bins} wanted; a clip of {n} samples cannot '
                  f'resolve better than {fs / max(n, 1):.3f} Hz). Widen --eval_band '
                  f'(resp: >= 0.5 Hz, e.g. 0.1,0.6) or evaluate longer '
                  f'(session-level) signals for a meaningful Tier-2.')
        if n_skipped:
            print(f'[metrics] NOTE: {n_skipped}/{len(t)} clip(s) skipped in the '
                  f'spectral metrics -- a target with no spectrum has nothing to '
                  f'compare (a constant or dead-sensor-plateau window).')
    if not psd_mae:
        return {'psd_mae': float('nan'), 'dominant_freq_error_hz': float('nan'),
                'psd_band_bins': band_bins, 'psd_skipped': n_skipped,
                'hf_power_rel': float('nan'),
                'target_hf_power_rel': float('nan'), 'hf_lo_hz': hf_lo}

    return {
        'psd_mae': float(np.mean(psd_mae)),
        'dominant_freq_error_hz': float(np.mean(dom_freq_err)),
        'psd_band_bins': band_bins,
        'psd_skipped': n_skipped,
        'hf_power_rel': float(np.mean(hf_rel)),
        'target_hf_power_rel': float(np.mean(hf_ref)),
        'hf_lo_hz': hf_lo,
    }


# --------------------------------------------------------------------------- #
# grouped aggregation (per task / per distortion group)
# --------------------------------------------------------------------------- #
def aggregate_metrics(records, keys, with_mean=True):
    """Median / IQR (and mean) of each metric over a list of metric dicts.

    The per-SESSION counterpart of the per-clip macro average: sessions differ
    in length and quality, so a median + IQR is reported alongside the mean.
    NaNs are dropped PER KEY (a session that could not produce a metric -- e.g.
    Tier-2 on a degenerate target -- must not erase that key for the group).

    :param records: iterable of ``{metric_key: float}`` dicts.
    :param keys: metric keys to aggregate.
    :param with_mean: also emit ``<key>_mean`` (default True).
    :returns: ``{'n': n, '<key>_median': .., '<key>_iqr': .., '<key>_mean': ..}``
        with NaN where a key had no finite value across ``records``.
    """
    recs = list(records)
    out = {'n': len(recs)}
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in recs], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f'{k}_median'] = float(np.median(vals))
            out[f'{k}_iqr'] = float(np.percentile(vals, 75)
                                    - np.percentile(vals, 25))
            if with_mean:
                out[f'{k}_mean'] = float(np.mean(vals))
        else:
            out[f'{k}_median'] = float('nan')
            out[f'{k}_iqr'] = float('nan')
            if with_mean:
                out[f'{k}_mean'] = float('nan')
    return out


def breakdown_by_label(records_by_name, label_of, keys, with_mean=True):
    """Group per-session metric dicts by a label and aggregate each group.

    :param records_by_name: ``{session_name: {metric_key: float}}``. Sessions
        with no metrics (e.g. a missing reference) should simply be omitted.
    :param label_of: ``{session_name: label}`` or
        ``{session_name: [label, ...]}`` -- a session may belong to SEVERAL
        groups (an overlapping group definition).
    :param keys: metric keys to aggregate within each group.
    :returns: ``{label: aggregate_metrics(...)}`` with the labels sorted;
        sessions whose label is missing or empty are skipped.
    """
    buckets = {}
    for name, rec in records_by_name.items():
        labels = label_of.get(name)
        if not labels:
            continue
        if isinstance(labels, str):
            labels = [labels]
        for lab in labels:
            if lab:
                buckets.setdefault(str(lab), []).append(rec)
    return {lab: aggregate_metrics(recs, keys, with_mean=with_mean)
            for lab, recs in sorted(buckets.items())}


# --------------------------------------------------------------------------- #
# band-limited correlation, trivial baselines, and uncertainty
# --------------------------------------------------------------------------- #
def bandpass_fft(x, fs=100.0, band=(0.1, 0.6)):
    """Zero-phase brick-wall FFT band-pass along the LAST axis.

    The band-limited counterpart of ``run_evaluate_session.smooth_band`` (a
    Butterworth IIR, ill-conditioned for a narrow low band). A brick-wall FFT
    filter is exactly zero outside ``band`` and exactly linear-phase, so a
    band-limited correlation is reproducible with no filter-transient
    ambiguity -- the reason the TIR-ROI investigation used it for the "does the
    rhythm survive drift removal" test (median session Pearson 0.437 -> 0.593).

    :param x: ``[T]`` or ``[N, T]`` array.
    :param fs: sampling rate (Hz).
    :param band: ``(lo, hi)`` in Hz; ``None`` = no-op.
    :returns: the band-passed array, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if band is None or n < 4:
        return x.copy()
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    keep = (freqs >= float(band[0])) & (freqs <= float(band[1]))
    if not keep.any():
        return np.zeros_like(x)
    shape = (1,) * (x.ndim - 1) + keep.shape
    spec = np.fft.rfft(x, axis=-1) * keep.reshape(shape)
    return np.fft.irfft(spec, n=n, axis=-1)


def band_limited_pearson(pred, target, fs=100.0, band=(0.1, 0.6)):
    """Pearson after band-passing BOTH series -- the DRIFT-ROBUST correlation.

    The headline metric for a respiration reconstruction: band-limiting removes
    the slow baseline drift that dominates a long window and inflates a raw
    Pearson, leaving the correlation of the breathing rhythm itself. Returns
    the macro (mean) Pearson over rows.
    """
    p = _flatten(pred)
    t = _flatten(target)
    if p.shape != t.shape:
        raise ValueError(f'Shape mismatch pred {p.shape} vs target {t.shape}')
    return _mean_pearson(bandpass_fft(p, fs, band), bandpass_fft(t, fs, band))


def single_tone_template(target, fs=100.0, band=(0.1, 0.6)):
    """Least-squares single-tone fit -> the ``beat-the-template`` baseline.

    A model that knows ONLY the breathing rate can do no better than a pure
    tone at the true dominant frequency. The fit is handed the TARGET's own
    peak frequency for free (three numbers read off the reference), so its
    band-limited Pearson is the bar a reconstruction must clear to be more than
    "rate tracked" (measured 2026-10-10: the Stage-3 model's 0.593 median tied
    this baseline's 0.600, winning only 49.1 % of sessions).

    :returns: ``(template[N, T], median_peak_freq_hz)``.
    """
    t = _flatten(target)
    tb = bandpass_fft(t, fs, band)
    rows, freqs = [], []
    for row in tb:
        n = row.size
        if n < 4:
            rows.append(np.zeros(n))
            freqs.append(np.nan)
            continue
        f = np.fft.rfftfreq(n, d=1.0 / float(fs))
        spec = np.abs(np.fft.rfft(row))
        if band is not None:
            m = (f >= float(band[0])) & (f <= float(band[1]))
        else:
            m = np.ones(f.size, dtype=bool)
        fpeak = (float(f[m][np.argmax(spec[m])]) if m.any()
                 else float(f[np.argmax(spec)]))
        tt = np.arange(n) / float(fs)
        w = 2.0 * np.pi * fpeak
        design = np.stack([np.ones(n), np.cos(w * tt), np.sin(w * tt)], axis=1)
        coef, *_ = np.linalg.lstsq(design, row, rcond=None)
        rows.append(design @ coef)
        freqs.append(fpeak)
    return np.stack(rows, 0), float(np.nanmedian(freqs))


def bootstrap_ci(values, statistic=np.median, n_boot=2000, ci=95.0, seed=0):
    """Percentile bootstrap CI of ``statistic`` over ``values`` (NaNs dropped).

    The across-subject uncertainty: each TESTED SUBJECT is one observation, so
    the CI is over subjects, NOT over clips (clips within a subject are not
    independent). ``statistic`` defaults to the median (robust to the
    heavy-tailed per-subject distribution).

    :returns: ``{'point': .., 'lo': .., 'hi': .., 'n': n_subjects}``; all-NaN
        (``n`` 0) when no finite value is available.
    """
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {'point': float('nan'), 'lo': float('nan'),
                'hi': float('nan'), 'n': 0}
    point = float(statistic(vals))
    if vals.size == 1:
        return {'point': point, 'lo': point, 'hi': point, 'n': 1}
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(rng.choice(vals, size=vals.size, replace=True))
                      for _ in range(int(n_boot))], dtype=float)
    lo = float(np.percentile(boots, (100.0 - float(ci)) / 2.0))
    hi = float(np.percentile(boots, 100.0 - (100.0 - float(ci)) / 2.0))
    return {'point': point, 'lo': lo, 'hi': hi, 'n': int(vals.size)}


# --------------------------------------------------------------------------- #
# derived vital sign: respiration RATE (breaths/min, BPM)
# --------------------------------------------------------------------------- #
def estimate_rate_bpm(x, fs=100.0, band=(0.1, 0.6)):
    """Dominant respiratory frequency per row -> breaths/min (BPM).

    The band-limited FFT peak with parabolic (sub-bin) interpolation, so a rate
    that falls between FFT bins is still recovered. Rows with no spectrum in the
    band return NaN. This is the READOUT for the derived-vital-sign target (a
    single averaged rate per window), independent of any waveform head.

    :param x: ``[T]`` or ``[N, T]`` waveform(s).
    :param fs: sampling rate (Hz).
    :param band: ``(lo, hi)`` respiration band in Hz (default 0.1-0.6 Hz =
        6-36 BPM).
    :returns: ``[N]`` array of rates in breaths/min (NaN where undefined).
    """
    x = _flatten(x)
    out = np.full(x.shape[0], np.nan, dtype=float)
    for i, row in enumerate(x):
        n = row.size
        if n < 4:
            continue
        f = np.fft.rfftfreq(n, d=1.0 / float(fs))
        spec = np.abs(np.fft.rfft(row - row.mean()))
        if band is not None:
            m = (f >= float(band[0])) & (f <= float(band[1]))
            if not m.any():
                continue
            k = int(np.where(m)[0][np.argmax(spec[m])])
        else:
            k = int(np.argmax(spec))
        fpeak = float(f[k])
        if 0 < k < spec.size - 1:                       # parabolic sub-bin
            a, b, c = spec[k - 1], spec[k], spec[k + 1]
            denom = a - 2.0 * b + c
            if abs(denom) > 1e-18:
                delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
                fpeak += delta * float(f[1] - f[0])
        out[i] = fpeak * 60.0
    return out


def rate_summary(pred_bpm, target_bpm, tol=0.10, min_bpm=1.0):
    """MAE (BPM) + percentage-within-``tol`` from paired per-window rates.

    The two literature-standard headline numbers for a summarized rate:

    * ``mae_bpm`` -- mean absolute error in breaths/min (physical units, the
      universal reporting standard);
    * ``within_10pct`` -- fraction of windows whose error is within ``tol`` of
      the TRUE (contact-sensor) rate, the ICMI-2023 style acceptance margin.

    Also returns ``rmse_bpm`` / ``bias_bpm`` (signed) for completeness. Windows
    with a non-finite or near-zero reference rate are dropped (a rate error is
    meaningless when the reference has no rate).
    """
    p = np.asarray(pred_bpm, dtype=float).reshape(-1)
    t = np.asarray(target_bpm, dtype=float).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f'Shape mismatch pred {p.shape} vs target {t.shape}')
    key = f'within_{int(round(float(tol) * 100))}pct'
    ok = np.isfinite(p) & np.isfinite(t) & (np.abs(t) >= float(min_bpm))
    p, t = p[ok], t[ok]
    if t.size == 0:
        return {'n': 0, 'mae_bpm': float('nan'), 'rmse_bpm': float('nan'),
                'bias_bpm': float('nan'), key: float('nan')}
    err = p - t
    return {
        'n': int(t.size),
        'mae_bpm': float(np.mean(np.abs(err))),
        'rmse_bpm': float(np.sqrt(np.mean(err ** 2))),
        'bias_bpm': float(np.mean(err)),
        key: float(np.mean(np.abs(err) <= float(tol) * np.abs(t))),
    }


def rate_metrics(pred, target, fs=100.0, band=(0.1, 0.6), tol=0.10):
    """Estimate + compare per-window rates: ``rate_summary`` on ``pred``/``target``."""
    return rate_summary(estimate_rate_bpm(pred, fs, band),
                        estimate_rate_bpm(target, fs, band), tol=tol)


# --------------------------------------------------------------------------- #
# self-test: ``python -m evaluation.metrics``
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    """Regression test for the Tier-2 resolution + degenerate-target fixes."""
    fails = []

    def check(label, got, want):
        if got != want:
            fails.append(f'{label}: got {got!r}, want {want!r}')

    # 1. nperseg is only ENLARGED, never shrunk -> well-resolved bands unchanged
    check('nperseg band=None', _resolve_nperseg(800, 100.0, None), 256)
    check('nperseg bp 1.0-2.5 (unchanged)', _resolve_nperseg(496, 100.0, (1.0, 2.5)), 256)
    check('nperseg resp 0.16-0.4 -> clip length',
          _resolve_nperseg(800, 100.0, (0.16, 0.4)), 800)
    check('nperseg short clip clamps to n', _resolve_nperseg(120, 100.0, (0.16, 0.4)), 120)

    rng = np.random.default_rng(0)
    fs, n = 100.0, 800
    time = np.arange(n) / fs
    n_clips = 4
    t = np.stack([np.sin(2 * np.pi * 0.3 * time + 0.3 * i) for i in range(n_clips)])
    t[-1] = 5.0                                    # a (railed) constant window

    # 2. pred == target -> zero PSD error, and the degenerate clip is SKIPPED
    r = spectral_metrics(t.copy(), t, fs=fs, band=(0.16, 0.4), warn=False)
    check('identical -> psd_mae 0', r['psd_mae'], 0.0)
    check('identical -> dom err 0', r['dominant_freq_error_hz'], 0.0)
    check('degenerate clip skipped', r['psd_skipped'], 1)

    # 3. the band now holds >= 2 bins (it held EXACTLY 1 at nperseg 256)
    check('band bins at the fix', r['psd_band_bins'], 2)
    old_bins = int(((np.fft.rfftfreq(min(256, n), 1.0 / fs) >= 0.16)
                    & (np.fft.rfftfreq(min(256, n), 1.0 / fs) <= 0.4)).sum())
    check('band bins before the fix (the bug)', old_bins, 1)

    # 4. the metric still DISCRIMINATES (a flat prediction is scored worse)
    flat = np.zeros_like(t)
    r_flat = spectral_metrics(flat, t, fs=fs, band=(0.16, 0.4), warn=False)
    if not r_flat['psd_mae'] > r['psd_mae']:
        fails.append(f"flat pred not scored worse: {r_flat['psd_mae']} vs "
                     f"{r['psd_mae']}")

    # 5. a fully degenerate target set returns NaN + the skip count (no 1.0s)
    r_deg = spectral_metrics(t.copy(), np.full_like(t, 3.0), fs=fs,
                             band=(0.16, 0.4), warn=False)
    if not (np.isnan(r_deg['psd_mae']) and r_deg['psd_skipped'] == n_clips):
        fails.append(f'fully degenerate set: {r_deg}')

    # 6. band=None is untouched by the fix (historical nperseg, no skip field
    #    surprise) and matches the hand-rolled Welch comparison
    r_nb = spectral_metrics(t.copy(), t, fs=fs, band=None, warn=False)
    check('unbanded band_bins is None', r_nb['psd_band_bins'], None)
    check('unbanded identical -> 0', r_nb['psd_mae'], 0.0)

    # 7. OUT-OF-BAND energy: invisible to every other Tier-2 number by design.
    #    A prediction with a high-frequency floor must be caught; a perfect
    #    prediction and a low-amplitude (collapsed) one must measure ~0.
    rng2 = np.random.default_rng(1)
    jitter = t.copy() + rng2.normal(0.0, 0.12, size=t.shape)
    r_j = spectral_metrics(jitter, t, fs=fs, band=(0.16, 0.4), warn=False)
    if not r_j['hf_power_rel'] > 10 * max(r['hf_power_rel'], 1e-6):
        fails.append(f"jitter not caught: hf_power_rel {r_j['hf_power_rel']} vs "
                     f"clean {r['hf_power_rel']}")
    if not r_j['target_hf_power_rel'] < 0.01:
        fails.append(f"target's own hf_power_rel not small: "
                     f"{r_j['target_hf_power_rel']}")
    if r['hf_power_rel'] > 1e-3:
        fails.append(f'identical pred -> hf_power_rel should be ~0, got '
                     f"{r['hf_power_rel']}")
    r_low = spectral_metrics(t.copy() * 0.01, t, fs=fs, band=(0.16, 0.4),
                             warn=False)
    if r_low['hf_power_rel'] > 1e-4:
        fails.append(f'collapsed (tiny) pred must not be blamed for HF energy: '
                     f"{r_low['hf_power_rel']}")
    check('hf_lo defaults to the band edge', r['hf_lo_hz'], 2.0)
    check('hf_lo raised for a higher band',
          spectral_metrics(t.copy(), t, fs=fs, band=(1.0, 2.5),
                           warn=False)['hf_lo_hz'], 2.5)
    r_hf = spectral_metrics(t.copy(), t, fs=fs, band=(0.16, 0.4), warn=False,
                            hf_lo=4.0)
    check('explicit hf_lo is honoured', r_hf['hf_lo_hz'], 4.0)

    # 8. a DEAD-SENSOR PLATEAU target passes the max-deviation guard but has no
    #    spectrum; it must be SKIPPED (and must not blow the ratio up). The
    #    geometry matters: `welch` de-means PER SEGMENT and only segments the
    #    first ~nperseg samples, so the plateau has to fill the segments it
    #    actually uses -- here nperseg 256 for the (1.0, 2.5) band, i.e. the
    #    first 768 samples, which is exactly the TIR-ROI failure mode.
    plateau = np.concatenate([np.full(768, 1.5), np.full(32, 9.0)])
    tp = np.concatenate([t[:3], plateau[None, :]])
    r_p = spectral_metrics(tp.copy(), tp, fs=fs, band=(1.0, 2.5), warn=False)
    check('plateau target skipped', r_p['psd_skipped'], 1)
    r_pj = spectral_metrics(tp.copy() + rng2.normal(0, 0.1, tp.shape), tp,
                            fs=fs, band=(1.0, 2.5), warn=False)
    if np.isfinite(r_pj['hf_power_rel']) and r_pj['hf_power_rel'] > 1e3:
        fails.append(f'plateau target ratio exploded: {r_pj["hf_power_rel"]}')

    print('=' * 72)
    print('evaluation.metrics self-test (Tier-2 resolution + degenerate guard)')
    print('=' * 72)
    print('band (0.16,0.4) @ fs 100, 800-sample clip -> nperseg',
          _resolve_nperseg(n, fs, (0.16, 0.4)), f'({r["psd_band_bins"]} bins; was',
          f'{old_bins} at nperseg 256)')
    print(f'out-of-band energy (> {r["hf_lo_hz"]:g} Hz, in units of the target '
          f'variance): perfect {r["hf_power_rel"]:.2e}, jittered '
          f'{r_j["hf_power_rel"]:.2e}, collapsed {r_low["hf_power_rel"]:.2e}, '
          f'target reference {r["target_hf_power_rel"]:.2e}')
    if fails:
        for f in fails:
            print(f'[FAIL] {f}')
        print(f'FAIL: {len(fails)} check(s) failed')
        return 1
    print('PASS: all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(_self_test())
