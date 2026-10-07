"""Offline SESSION-level evaluation of a Stage-3 run (writes PNG + JSON).

Consumes the prediction dump written by ``runners/run_waveform.py --save_preds``
(``preds.npy``, ``targets.npy``, ``entries.json`` in ``--output_dir``), rebuilds
each session's full waveform by Hann overlap-add of its clips, and reports:

* **per-clip** Tier-1 / Tier-2, macro-averaged (the training/validation signal);
* **session-level** Tier-1 / Tier-2 on the ASSEMBLED waveform -- this is the
  number that supports a "reconstruct the whole waveform" claim;
* **Tier-3** clinical HRV (bp branch only), aggregated over 30-60 s windows,
  and only when NeuroKit2 is installed and the window is long enough.

Outputs (defaults under the same directory as the dump):

    session_metrics.json        the machine-readable document (all sessions)
    figures/session_<name>.png  assembled vs reference + a zoom (per session)

Usage (from ``code/``)::

    python runners/run_evaluate_session.py --pred_dir ../output/finetune/bp_local \\
        --waveform bp --fs 100 --tier 1,2,3

Note on scale: with ``physio_norm: zscore`` every clip prediction is defined up
to a per-clip affine map, so session MAE/RMSE are reported twice -- raw, and
after one least-squares per-session calibration (``mae_affine``). Pearson and
the PSD shape are affine-invariant and need no calibration.

Note on the reference: ``entries.json`` normally points at a canonical
``signals.csv``. The TIR-ROI respiration dataset instead records the raw
``Resp_Volts.txt`` (a bare column at 1000 Hz), which is read and resampled to
``--fs`` by :func:`read_raw_series`; pass ``--raw_fs`` if its native rate is
not 1000 Hz. ``hf_power_rel`` is a ratio, so it needs no affine calibration.

Note on ``--smooth``: an optional zero-phase band applied to BOTH the prediction
and the reference before anything is scored (Tier-1, Tier-2 and the figures),
for presenting a waveform free of the token-seam jitter. It is a post-processing
step, it is recorded in the JSON under ``smooth``, and it must be reported
together with Pearson and the constant-0 baseline -- filtering lowers MAE by
shrinking the prediction, which is not the same as predicting better.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import clinical as _clinical
from evaluation import metrics as _metrics
from evaluation.assemble import (affine_calibrate, assemble_session,
                                 group_entries)
from evaluation.report import plot_session_waveform, save_json

__all__ = ['main', 'read_reference', 'read_raw_series', 'smooth_band']


def read_raw_series(path, raw_fs, fs):
    """Read a one-value-per-line physiological file and resample it to ``fs``.

    ``Resp_Volts.txt`` (the raw BP4D+ respiration trace) is a bare column of
    floats at 1000 Hz with NO header, so it cannot go through the canonical
    ``signals.csv`` reader. Resampling is ``np.interp`` on the time axis, the
    same convention ``data/tir_resp_dataset.py`` uses for its clip windows.
    """
    x = np.asarray(np.loadtxt(path, ndmin=1), dtype=float).ravel()
    if x.size == 0:
        raise ValueError(f'{path}: empty raw series')
    if fs is None or not raw_fs or float(raw_fs) == float(fs):
        return x
    t = np.arange(x.size) / float(raw_fs)
    n = int(np.floor(t[-1] * float(fs))) + 1
    return np.interp(np.arange(n) / float(fs), t, x)


def read_reference(path, name, fs=None, raw_fs=None):
    """Read the target series named ``name`` (bp/resp/eda) at ``fs``.

    Two layouts are supported:

    * a canonical ``signals.csv`` (comma-separated with a header) -- the
      pre-2026-09-23 spelling ``bvp`` for ``bp`` is still accepted;
    * a raw one-value-per-line ``.txt`` (``Resp_Volts.txt``), which is read via
      :func:`read_raw_series` and resampled from ``raw_fs`` to ``fs``.
    """
    if str(path).lower().endswith('.txt'):
        return read_raw_series(path, raw_fs, fs)
    with open(path) as f:
        header = [h.strip().lower() for h in next(f).strip().split(',')]
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    for cand in (name, 'bvp' if name == 'bp' else name):
        if cand in header:
            return data[:, header.index(cand)].astype(float)
    raise ValueError(f'{path}: no {name!r} column (header: {header})')


def smooth_band(x, fs, band, order=4):
    """Zero-phase Butterworth filter applied to ``x`` along the last axis.

    ``band`` is ``(lo, hi)`` for a band-pass, ``(fc,)`` / ``(fc, None)`` for a
    low-pass, or ``None`` for a no-op. ``filtfilt`` is used so the filtered
    waveform stays time-aligned with the reference -- a causal filter would
    delay it by ~``order`` samples and that delay would show up as a phase
    error in Tier-1, which is exactly the quantity under test here.

    This is POST-PROCESSING for presentation. It removes the token-seam jitter
    that the 16-samples-per-token head emits (``TirROI_Resp_plan.md`` §12), but
    it cannot add phase information, so the correlation does not move. The
    runner applies it to the prediction AND the reference, records the band in
    the JSON, and keeps Pearson beside the constant-0 baseline so a smoothed
    figure cannot be mistaken for a better one.
    """
    if band is None:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.shape[-1] < 3:
        return x
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:                       # pragma: no cover
        raise SystemExit(f'--smooth needs scipy: {exc}')
    nyq = float(fs) / 2.0
    band = tuple(band)
    if len(band) < 2 or band[1] is None:
        b, a = butter(order, float(band[0]) / nyq, btype='low')
    else:
        b, a = butter(order, [float(band[0]) / nyq, float(band[1]) / nyq],
                      btype='band')
    padlen = min(3 * (max(len(a), len(b)) - 1), x.shape[-1] - 1)
    return filtfilt(b, a, x, axis=-1, padlen=padlen)


def _agg(dicts, keys):
    """median + IQR over a list of metric dicts (NaNs ignored)."""
    out = {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d in dicts], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f'{k}_median'] = float(np.median(vals))
            out[f'{k}_iqr'] = float(np.percentile(vals, 75)
                                    - np.percentile(vals, 25))
        else:
            out[f'{k}_median'] = float('nan')
            out[f'{k}_iqr'] = float('nan')
    return out


def get_args():
    p = argparse.ArgumentParser('Offline session-level waveform evaluation',
                                add_help=True)
    p.add_argument('--pred_dir', default='', type=str,
                   help='run output dir holding preds.npy/targets.npy/entries.json')
    p.add_argument('--pred_path', default='', type=str)
    p.add_argument('--target_path', default='', type=str)
    p.add_argument('--entries_path', default='', type=str)
    p.add_argument('--waveform', default='bp', choices=['bp', 'resp'])
    p.add_argument('--fs', default=100.0, type=float)
    p.add_argument('--tier', default='1,2,3', type=str)
    p.add_argument('--band', default=None, type=str,
                   help='spectral band "f_low,f_high" (default: bp 1.0,2.5 / '
                        'resp 0.16,0.4)')
    p.add_argument('--out', default='', type=str,
                   help='session-metrics JSON path (default <pred_dir>/session_metrics.json)')
    p.add_argument('--fig_dir', default='', type=str,
                   help='figure dir (default <pred_dir>/figures)')
    p.add_argument('--zoom_s', default=20.0, type=float)
    p.add_argument('--max_sessions', default=0, type=int)
    p.add_argument('--window_s', default=30.0, type=float,
                   help='Tier-3 sub-window length in seconds')
    p.add_argument('--raw_fs', default=1000.0, type=float,
                   help='native rate of a raw one-value-per-line reference '
                        '(Resp_Volts.txt); ignored for signals.csv')
    p.add_argument('--smooth', default='', type=str,
                   help="zero-phase Butterworth band applied to BOTH the "
                        "prediction and the reference before scoring, e.g. "
                        "'0.1,0.6' (band-pass) or '0.7' (low-pass); empty = "
                        "raw. The band is recorded in the output JSON.")
    return p.parse_args()


def main(args):
    d = args.pred_dir
    pred_path = args.pred_path or os.path.join(d, 'preds.npy')
    target_path = args.target_path or os.path.join(d, 'targets.npy')
    entries_path = args.entries_path or os.path.join(d, 'entries.json')
    for p in (pred_path, entries_path):
        if not os.path.isfile(p):
            raise SystemExit(
                f'missing {p}. Run runners/run_waveform.py with --save_preds '
                f'(or save_preds: true in the config) first.')
    out_path = args.out or os.path.join(d or '.', 'session_metrics.json')
    fig_dir = args.fig_dir or os.path.join(d or '.', 'figures')

    preds = np.atleast_2d(np.load(pred_path)).astype(float)
    with open(entries_path) as f:
        entries = json.load(f)
    tiers = {int(t) for t in str(args.tier).split(',') if t.strip()}
    band = (tuple(float(x) for x in args.band.split(',')) if args.band
            else ((1.0, 2.5) if args.waveform == 'bp' else (0.16, 0.4)))
    smooth = (tuple(float(x) for x in str(args.smooth).split(','))
              if str(args.smooth).strip() else None)
    if smooth is not None:
        preds = smooth_band(preds, args.fs, smooth)
    print(f'[session-eval] {preds.shape[0]} clips, {len(entries)} entries, '
          f'waveform={args.waveform}, fs={args.fs:g}, band={band}, '
          f'smooth={smooth}, tiers={tiers}')

    # per-clip (macro) needs the per-clip targets, when they were dumped
    per_clip = {}
    if os.path.isfile(target_path) and 1 in tiers:
        tgts = np.atleast_2d(np.load(target_path)).astype(float)
        if smooth is not None:
            tgts = smooth_band(tgts, args.fs, smooth)
        if tgts.shape[0] == preds.shape[0]:
            per_clip.update(_metrics.time_domain_metrics(preds, tgts))
            if 2 in tiers:
                try:
                    per_clip.update(_metrics.spectral_metrics(
                        preds, tgts, fs=args.fs, band=band))
                except ImportError:
                    per_clip.update({'psd_mae': float('nan'),
                                     'hf_power_rel': float('nan')})

    groups = group_entries(entries, args.fs)
    names = sorted(groups)[:args.max_sessions or None]
    sessions, tier3_all = {}, []
    for name in names:
        pairs = groups[name]
        rows = [i for i, _ in pairs]
        offs = [o for _, o in pairs]
        t, y = assemble_session(preds[rows], offs, args.fs)
        sig_file = entries[rows[0]].get('signals_file') or ''
        rec = {'n_clips': len(rows), 'duration_s': float(t[-1]) if t.size else 0.0,
               'signals_file': sig_file}
        if not sig_file or not os.path.isfile(sig_file):
            rec['error'] = f'reference signals.csv not found: {sig_file!r}'
            sessions[name] = rec
            continue
        ref_full = read_reference(sig_file, args.waveform,
                                  fs=args.fs, raw_fs=args.raw_fs)
        n = min(y.size, ref_full.size)
        ref = ref_full[:n]
        yh = y[:n]
        if smooth is not None:
            ref = smooth_band(ref, args.fs, smooth)
            yh = smooth_band(yh, args.fs, smooth)
        rec['reference_samples'] = int(ref_full.size)
        if 1 in tiers:
            rec['session_tier1'] = dict(_metrics.time_domain_metrics(
                np.atleast_2d(yh), np.atleast_2d(ref)))
            a, b, fitted = affine_calibrate(yh, ref)
            rec['affine_a'], rec['affine_b'] = a, b
            if np.isfinite(fitted).all():
                fit = _metrics.time_domain_metrics(
                    np.atleast_2d(fitted), np.atleast_2d(ref))
                rec['session_tier1_affine'] = {
                    'mae': fit['mae'], 'rmse': fit['rmse'],
                    'pearson': fit['pearson']}
        if 2 in tiers:
            try:
                rec['session_tier2'] = dict(_metrics.spectral_metrics(
                    np.atleast_2d(yh), np.atleast_2d(ref), fs=args.fs, band=band))
            except ImportError:
                rec['session_tier2'] = {'psd_mae': float('nan'),
                                        'hf_power_rel': float('nan')}
        if 3 in tiers and args.waveform == 'bp':
            win = int(round(args.window_s * args.fs))
            wins = [yh[i:i + win] for i in range(0, max(1, yh.size - win + 1), win)]
            wins = [w for w in wins if w.size >= win]
            recs = []
            for w in wins:
                try:
                    recs.append(_clinical.extract_hrv_metrics(w, fs=args.fs))
                except ImportError as e:
                    rec['tier3_error'] = str(e).split('\n')[0]
                    break
            if recs:
                rec['tier3'] = _agg(recs, ['rmssd_ms', 'pnn50', 'median_nn_ms',
                                           'shannon_entropy', 'n_rr'])
                rec['tier3_windows'] = len(recs)
                tier3_all.extend(recs)
        if 1 in tiers or 2 in tiers:
            rec['figure'] = plot_session_waveform(
                t[:yh.size], ref, yh,
                out_png=os.path.join(fig_dir, f'session_{name}.png'),
                title=(f'{name} | {args.waveform} reconstruction | '
                       f'{len(rows)} clips'),
                fs=args.fs, zoom_s=args.zoom_s,
                metrics=rec.get('session_tier1', {}))
        sessions[name] = rec
        print(f'[session-eval] {name}: {rec.get("session_tier1", {})}')

    doc = {'waveform': args.waveform, 'fs': args.fs, 'band': list(band),
           'smooth': (list(smooth) if smooth is not None else None),
           'tiers': sorted(tiers), 'n_clips': int(preds.shape[0]),
           'n_sessions': len(sessions), 'per_clip_macro': per_clip,
           'sessions': sessions}
    if tier3_all:
        doc['tier3_aggregate'] = _agg(tier3_all, [
            'rmssd_ms', 'pnn50', 'median_nn_ms', 'shannon_entropy', 'n_rr'])
    save_json(out_path, doc)
    print(f'[session-eval] wrote {out_path} and {len(sessions)} figure(s) to '
          f'{fig_dir}')


if __name__ == '__main__':
    main(get_args())
