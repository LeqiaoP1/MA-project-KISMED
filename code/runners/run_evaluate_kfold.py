"""Pooled k-fold (subject-disjoint) evaluation over ALL BP4D+ subjects.

The Stage-3 pipeline is run once per fold, each fold holding out a different
subject block (``utils/splits.py::subject_folds``, ``n_folds`` = 5). Every fold
therefore predicts EVERY subject exactly once -- namely the subjects it never
trained on -- so pooling the folds gives a cross-validated number for the whole
corpus, not just for one test block.

This runner consumes the per-fold dumps (``preds.npy`` + ``entries.json``) and,
per session:

1. assembles the fold's clip predictions into one continuous waveform
   (Hann overlap-add, ``evaluation/assemble.py``);
2. scores it against the raw ``Resp_Volts`` reference in the three metric
   families asked for by the evaluation protocol
   (``code/notes/design/Evaluation_RespWaveform_protocol.md`` sec. 4):

   * **pearson** -- ``pearson_band`` (band-limited 0.1-0.6 Hz, the drift-robust
     headline) plus the plain ``pearson`` for contrast and the single-tone
     TEMPLATE baseline it must beat;
   * **amplitude** -- ``amp_ratio`` / ``amp_rel_err`` = in-band RMS of the
     prediction vs the reference AFTER the per-session affine calibration (so
     ORACLE-calibrated: the calibration uses the session's true scale);
   * **rate** -- per-window ``estimate_rate_bpm`` -> ``mae_bpm`` and
     ``within_10pct`` (``--rate_window_s``, default 30 s), pooled over every
     window of every fold;

3. aggregates per SUBJECT (median over that subject's sessions), then across
   subjects with a bootstrap 95 % CI, plus ``per_task`` / ``per_group`` and a
   per-fold spread table;
4. verifies the folds really form a subject-disjoint partition of the corpus
   (raises on any overlap or duplicated subject).

Usage (from ``code/``)::

    python runners/run_evaluate_kfold.py --root ../output/finetune \\
        --run_prefix resp_tir_roi_hpc_fold --folds 0,1,2,3,4

A smoke run: add ``--max_sessions 20``.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import metrics as _metrics
from evaluation.assemble import affine_calibrate, assemble_session, group_entries
from evaluation.report import save_json
from runners.run_evaluate_session import _task_of_name, read_reference
from runners.run_evaluate_subject import _window_rates, subject_of

__all__ = ['main', 'get_args', 'score_fold', 'aggregate_kfold']

#: Per-session metrics carried through the per-subject / per-task aggregation.
#: Raw cross-space ``mae``/``rmse`` are DELIBERATELY absent (the prediction is
#: session-z-scored while the reference is raw volts, so they only track the
#: reference DC); the volts-calibrated ``*_affine`` terms carry the point error.
_SESSION_KEYS = ('pearson', 'pearson_band', 'template_pearson_band',
                 'amp_ratio', 'amp_rel_err', 'amp_ref', 'amp_pred',
                 'mae_affine', 'rmse_affine', 'psd_mae',
                 'dominant_freq_error_hz', 'hf_power_rel',
                 'mae_bpm', 'within_10pct')

#: The three headline aggregates (one bootstrap CI each, over SUBJECTS).
_HEADLINE = ('pearson_band', 'amp_rel_err', 'mae_bpm')


def _fold_dir(root, prefix, k):
    """Prefer ``<prefix><k>``; accept ``<prefix>_fold<k>`` too."""
    for cand in (f'{prefix}{k}', f'{prefix}_fold{k}'):
        d = os.path.join(root, cand)
        if os.path.isdir(d):
            return d
    return os.path.join(root, f'{prefix}{k}')


def score_fold(pred_dir, fs, raw_fs, band, waveform, rate_window_s, rate_tol,
               fold, max_sessions=0):
    """Score every session of ONE fold.

    :returns: ``(records, windows)`` -- a list of per-session dicts and the
        concatenated ``(pred_bpm, target_bpm)`` window arrays for the pooled
        rate summary.
    """
    pred_path = os.path.join(pred_dir, 'preds.npy')
    entries_path = os.path.join(pred_dir, 'entries.json')
    for p in (pred_path, entries_path):
        if not os.path.isfile(p):
            raise SystemExit(f'[kfold] fold {fold}: missing {p}')
    preds = np.atleast_2d(np.load(pred_path)).astype(float)
    with open(entries_path) as f:
        entries = json.load(f)
    if preds.shape[0] != len(entries):
        raise SystemExit(
            f'[kfold] fold {fold}: {preds.shape[0]} predictions vs '
            f'{len(entries)} dataset entries -- this dump is a DDP SHARD, not a '
            f'single-process re-eval (see submissions with --eval_only).')

    groups = group_entries(entries, fs)
    names = sorted(groups)
    if max_sessions:
        names = names[:int(max_sessions)]

    records, pw_all, tw_all = [], [], []
    for name in names:
        pairs = groups[name]
        rows = [i for i, _ in pairs]
        offs = [o for _, o in pairs]
        sig = entries[rows[0]].get('signals_file') or ''
        if not sig or not os.path.isfile(sig):
            print(f'[kfold] fold {fold} {name}: skip (no reference)')
            continue
        t, y, cov = assemble_session(preds[rows], offs, fs)
        ref_full = read_reference(sig, waveform, fs=fs, raw_fs=raw_fs)
        n = int(min(y.size, ref_full.size))
        cov = cov[:n]
        if not cov.any():
            print(f'[kfold] fold {fold} {name}: skip (no covered sample)')
            continue
        ref, yh = ref_full[:n][cov].copy(), y[:n][cov].copy()

        rec = {
            'session': name,
            'subject': subject_of(name),
            'task': entries[rows[0]].get('task') or _task_of_name(name),
            'task_level': list(entries[rows[0]].get('task_level') or ()),
            'fold': int(fold),
            'n_clips': len(rows),
            'duration_s': float(t[-1]) if t.size else 0.0,
            'coverage': float(cov.mean()),
        }
        # --- pearson -------------------------------------------------------
        yh2 = np.atleast_2d(yh)
        ref2 = np.atleast_2d(ref)
        rec['pearson'] = float(_metrics.time_domain_metrics(yh2, ref2)['pearson'])
        rec['pearson_band'] = float(
            _metrics.band_limited_pearson(yh2, ref2, fs, band))
        tmpl, freq = _metrics.single_tone_template(ref2, fs, band)
        rec['template_pearson_band'] = float(
            _metrics.band_limited_pearson(tmpl, ref2, fs, band))
        rec['template_freq_hz'] = float(freq)

        # --- amplitude (ORACLE-calibrated: needs the session's true scale) --
        a, b, fitted = affine_calibrate(yh, ref)
        if np.isfinite(fitted).all():
            amp = _metrics.amplitude_metrics(np.atleast_2d(fitted), ref2, fs, band)
            rec.update({k: float(amp[k]) for k in
                        ('amp_ratio', 'amp_rel_err', 'amp_ref', 'amp_pred')})
            fit = _metrics.time_domain_metrics(np.atleast_2d(fitted), ref2)
            rec['mae_affine'] = float(fit['mae'])
            rec['rmse_affine'] = float(fit['rmse'])
            rec['affine_a'] = float(a)
            rec['affine_b'] = float(b)
        else:
            print(f'[kfold] fold {fold} {name}: degenerate affine fit')

        # --- spectral context ---------------------------------------------
        try:
            t2 = _metrics.spectral_metrics(yh2, ref2, fs=fs, band=band)
        except ImportError:                       # pragma: no cover
            t2 = {}
        for k in ('psd_mae', 'dominant_freq_error_hz', 'hf_power_rel'):
            if np.isfinite(t2.get(k, np.nan)):
                rec[k] = float(t2[k])

        # --- rate (derived vital sign) ------------------------------------
        if rate_window_s and float(rate_window_s) > 0:
            pw, tw = _window_rates(yh, ref, fs, band, rate_window_s)
            rs = _metrics.rate_summary(pw, tw, tol=rate_tol)
            if np.isfinite(rs['mae_bpm']):
                rec['mae_bpm'] = float(rs['mae_bpm'])
                rec[f'within_{int(round(rate_tol * 100))}pct'] = float(
                    rs[f'within_{int(round(rate_tol * 100))}pct'])
                rec['within_10pct'] = rec[f'within_{int(round(rate_tol * 100))}pct']
            rec['rate_n'] = int(rs['n'])
            pw_all.append(pw)
            tw_all.append(tw)

        records.append(rec)
    print(f'[kfold] fold {fold}: scored {len(records)} session(s) from '
          f'{pred_dir}')
    if pw_all:
        return records, (np.concatenate(pw_all), np.concatenate(tw_all))
    return records, (np.zeros(0), np.zeros(0))


def _flat(rec, keys):
    """Keep the finite metric values of one session record."""
    return {k: float(rec[k]) for k in keys
            if k in rec and np.isfinite(rec.get(k, np.nan))}


def _ci_over_subjects(per_subject, key, n_boot):
    """Bootstrap CI of the per-subject MEDIAN of ``key``."""
    vals = [v.get(f'{key}_median', np.nan) for v in per_subject.values()]
    return _metrics.bootstrap_ci(vals, n_boot=n_boot)


def aggregate_kfold(records, n_boot=2000):
    """Per-subject / per-task / per-group aggregation + across-subject CIs."""
    quality, subj_of, task_of, levels_of = {}, {}, {}, {}
    for rec in records:
        name = rec['session']
        quality[name] = _flat(rec, _SESSION_KEYS)
        subj_of[name] = rec['subject']
        task_of[name] = rec['task']
        levels_of[name] = list(rec['task_level'])

    per_subject = _metrics.breakdown_by_label(quality, subj_of, _SESSION_KEYS)
    per_task = _metrics.breakdown_by_label(quality, task_of, _SESSION_KEYS)
    per_group = _metrics.breakdown_by_label(quality, levels_of, _SESSION_KEYS)

    overall = {}
    for key in _HEADLINE:
        overall[key] = _ci_over_subjects(per_subject, key, n_boot)
    # template baseline, for the "does it beat a rate-only model" comparison
    overall['template_pearson_band'] = _ci_over_subjects(
        per_subject, 'template_pearson_band', n_boot)
    head = [v.get('pearson_band_median', np.nan) for v in per_subject.values()]
    tmpl = [v.get('template_pearson_band_median', np.nan)
            for v in per_subject.values()]
    beats = [1.0 if (np.isfinite(m) and np.isfinite(t) and m > t) else 0.0
             for m, t in zip(head, tmpl)]
    overall['beat_template_frac_subjects'] = (float(np.mean(beats))
                                              if beats else float('nan'))
    overall['n_subjects'] = len(per_subject)
    overall['n_sessions'] = len(quality)
    overall['n_clips'] = int(sum(r['n_clips'] for r in records))
    return {'per_subject': per_subject, 'per_task': per_task,
            'per_group': per_group, 'overall': overall}


def _fold_summary(fold, records, n_entries):
    """Per-fold headline medians -- the cross-validation spread."""
    def med(key):
        vals = [r[key] for r in records if np.isfinite(r.get(key, np.nan))]
        return float(np.median(vals)) if vals else float('nan')

    return {
        'fold': int(fold),
        'n_entries': int(n_entries),
        'n_sessions': len(records),
        'n_subjects': len({r['subject'] for r in records}),
        'subjects': sorted({r['subject'] for r in records}),
        'pearson_band_median': med('pearson_band'),
        'pearson_median': med('pearson'),
        'amp_rel_err_median': med('amp_rel_err'),
        'amp_ratio_median': med('amp_ratio'),
        'mae_bpm_median': med('mae_bpm'),
        'within_10pct_median': med('within_10pct'),
        'mae_affine_median': med('mae_affine'),
        'psd_mae_median': med('psd_mae'),
    }


def _check_partition(fold_summaries):
    """Every subject must be held out by EXACTLY one fold."""
    seen = {}
    for fs in fold_summaries:
        for s in fs['subjects']:
            if s in seen:
                raise SystemExit(
                    f'[kfold] SUBJECT OVERLAP: {s} is a test subject in folds '
                    f'{seen[s]} and {fs["fold"]} -- the pooled number would be '
                    f'leaked.')
            seen[s] = fs['fold']
    return len(seen)


def get_args():
    p = argparse.ArgumentParser(
        'Pooled k-fold (subject-disjoint) evaluation over all subjects',
        add_help=True)
    p.add_argument('--root', default='../output/finetune', type=str,
                   help='parent dir holding the per-fold run dirs')
    p.add_argument('--run_prefix', default='resp_tir_roi_hpc_fold', type=str,
                   help='fold dir name prefix (fold k dir = <prefix>k)')
    p.add_argument('--folds', default='0,1,2,3,4', type=str,
                   help='CSV of fold indices to pool')
    p.add_argument('--waveform', default='resp', choices=['bp', 'resp'])
    p.add_argument('--fs', default=100.0, type=float)
    p.add_argument('--raw_fs', default=1000.0, type=float,
                   help='native rate of a raw one-value-per-line reference')
    p.add_argument('--band', default=None, type=str,
                   help='band-limited band "lo,hi" (default resp 0.1,0.6 / '
                        'bp 1.0,2.5)')
    p.add_argument('--rate_window_s', default=30.0, type=float,
                   help='window length (s) for the RATE report; 0 disables')
    p.add_argument('--rate_tol', default=0.10, type=float,
                   help='relative acceptance margin of the within-tol%% metric')
    p.add_argument('--out', default='', type=str,
                   help='output JSON (default <root>/kfold_<run_prefix>.json)')
    p.add_argument('--n_boot', default=2000, type=int)
    p.add_argument('--max_sessions', default=0, type=int,
                   help='per fold; for smoke runs only')
    return p.parse_args()


def main(args):
    band = (tuple(float(x) for x in args.band.split(',')) if args.band
            else ((1.0, 2.5) if args.waveform == 'bp' else (0.1, 0.6)))
    folds = [int(k) for k in str(args.folds).split(',') if k.strip()]
    out_path = args.out or os.path.join(
        args.root, f'kfold_{args.run_prefix.rstrip("_")}.json')
    wkey = f'within_{int(round(args.rate_tol * 100))}pct'

    records, summaries, pw_all, tw_all = [], [], [], []
    for k in folds:
        d = _fold_dir(args.root, args.run_prefix, k)
        with open(os.path.join(d, 'entries.json')) as f:
            n_entries = len(json.load(f))
        recs, (pw, tw) = score_fold(
            d, args.fs, args.raw_fs, band, args.waveform,
            args.rate_window_s, args.rate_tol, k, args.max_sessions)
        records.extend(recs)
        summaries.append(_fold_summary(k, recs, n_entries))
        pw_all.append(pw)
        tw_all.append(tw)

    n_subjects = _check_partition(summaries)
    agg = aggregate_kfold(records, n_boot=args.n_boot)

    rate_overall = {}
    if pw_all:
        pw = np.concatenate(pw_all)
        tw = np.concatenate(tw_all)
        if pw.size:
            rate_overall = _metrics.rate_summary(pw, tw, tol=args.rate_tol)
            rate_overall['window_s'] = float(args.rate_window_s)

    doc = {
        'protocol': {
            'kind': 'pooled subject-disjoint k-fold cross-validation',
            'folds': folds,
            'root': os.path.abspath(args.root),
            'run_prefix': args.run_prefix,
            'waveform': args.waveform,
            'fs': args.fs,
            'raw_fs': args.raw_fs,
            'band': list(band),
            'pearson': 'band-limited Pearson (headline), plain Pearson for '
                       'contrast, single-tone template baseline',
            'amplitude': 'in-band RMS ratio after the per-session affine '
                         'calibration -> ORACLE-calibrated (not deployable)',
            'rate': f'dominant in-band FFT peak per {args.rate_window_s:g} s '
                    f'window; MAE (BPM) + {wkey}',
            'n_subjects': n_subjects,
        },
        'overall': agg['overall'],
        'rate_overall': rate_overall,
        'folds': summaries,
        'per_subject': agg['per_subject'],
        'per_task': agg['per_task'],
        'per_group': agg['per_group'],
        'sessions': {r['session']: _flat(r, _SESSION_KEYS) for r in records},
    }
    save_json(out_path, doc)

    # ---------------------------- console summary -------------------------
    o = agg['overall']
    print('\n[kfold] ===== pooled 5-fold subject-disjoint evaluation =====')
    print(f'[kfold] {n_subjects} subjects / {o["n_sessions"]} sessions / '
          f'{o["n_clips"]} clips, band {band} Hz')
    print(f'[kfold] PEARSON   band-limited (per-subject median): '
          f'{o["pearson_band"]["point"]:+.3f} '
          f'[95% CI {o["pearson_band"]["lo"]:+.3f}, '
          f'{o["pearson_band"]["hi"]:+.3f}]  |  template '
          f'{o["template_pearson_band"]["point"]:+.3f}, beaten by '
          f'{o["beat_template_frac_subjects"]:.0%} of subjects')
    print(f'[kfold] AMPLITUDE rel. error   : '
          f'{o["amp_rel_err"]["point"]:.3f} '
          f'[95% CI {o["amp_rel_err"]["lo"]:.3f}, '
          f'{o["amp_rel_err"]["hi"]:.3f}]  (oracle-calibrated)')
    print(f'[kfold] RATE      MAE          : '
          f'{o["mae_bpm"]["point"]:.2f} BPM '
          f'[95% CI {o["mae_bpm"]["lo"]:.2f}, {o["mae_bpm"]["hi"]:.2f}]'
          f'  |  pooled windows: MAE {rate_overall.get("mae_bpm", np.nan):.2f} '
          f'BPM, {wkey} {rate_overall.get(wkey, np.nan):.0%} '
          f'(n={rate_overall.get("n", 0)})')
    print('\n[kfold] per-fold (session medians):')
    print(f'  {"fold":>4} {"subj":>4} {"sess":>5} {"pearson_band":>13} '
          f'{"amp_rel_err":>12} {"mae_bpm":>8} {"within10%":>10}')
    for s in summaries:
        print(f'  {s["fold"]:>4} {s["n_subjects"]:>4} {s["n_sessions"]:>5} '
              f'{s["pearson_band_median"]:>13.3f} '
              f'{s["amp_rel_err_median"]:>12.3f} '
              f'{s["mae_bpm_median"]:>8.2f} '
              f'{s["within_10pct_median"]:>10.2f}')
    print(f'\n[kfold] wrote {out_path}')


if __name__ == '__main__':
    main(get_args())
