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

Note on scale: with ``signal_norm: zscore`` every clip prediction is defined up
to a per-clip affine map, so session MAE/RMSE are reported twice -- raw, and
after one least-squares per-session calibration (``mae_affine``). Pearson and
the PSD shape are affine-invariant and need no calibration.
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

__all__ = ['main', 'read_reference']


def read_reference(path, name):
    """Read the target column (bp/resp/eda) from a canonical ``signals.csv``.

    Accepts the pre-2026-09-23 header spelling (``bvp`` for ``bp``) so an older
    canonical layout still evaluates.
    """
    with open(path) as f:
        header = [h.strip().lower() for h in next(f).strip().split(',')]
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    for cand in (name, 'bvp' if name == 'bp' else name):
        if cand in header:
            return data[:, header.index(cand)].astype(float)
    raise ValueError(f'{path}: no {name!r} column (header: {header})')


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
                                add_help=False)
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
    print(f'[session-eval] {preds.shape[0]} clips, {len(entries)} entries, '
          f'waveform={args.waveform}, fs={args.fs:g}, band={band}, tiers={tiers}')

    # per-clip (macro) needs the per-clip targets, when they were dumped
    per_clip = {}
    if os.path.isfile(target_path) and 1 in tiers:
        tgts = np.atleast_2d(np.load(target_path)).astype(float)
        if tgts.shape[0] == preds.shape[0]:
            per_clip.update(_metrics.time_domain_metrics(preds, tgts))
            if 2 in tiers:
                try:
                    per_clip.update(_metrics.spectral_metrics(
                        preds, tgts, fs=args.fs, band=band))
                except ImportError:
                    per_clip['psd_mae'] = float('nan')

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
        ref_full = read_reference(sig_file, args.waveform)
        n = min(y.size, ref_full.size)
        ref = ref_full[:n]
        yh = y[:n]
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
                rec['session_tier2'] = {'psd_mae': float('nan')}
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
                t, ref, y, out_png=os.path.join(fig_dir, f'session_{name}.png'),
                title=(f'{name} | {args.waveform} reconstruction | '
                       f'{len(rows)} clips'),
                fs=args.fs, zoom_s=args.zoom_s,
                metrics=rec.get('session_tier1', {}))
        sessions[name] = rec
        print(f'[session-eval] {name}: {rec.get("session_tier1", {})}')

    doc = {'waveform': args.waveform, 'fs': args.fs, 'band': list(band),
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
