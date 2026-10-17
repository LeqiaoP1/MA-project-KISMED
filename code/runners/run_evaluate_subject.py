"""Subject-level (TESTED-SUBJECT) evaluation of a Stage-3 respiration run.

The final objective of the whole model is the reconstruction of the respiration
waveform for a HELD-OUT subject, so the unit of evaluation is the SUBJECT, not
the clip. This runner:

1. assembles each of a test subject's sessions into a continuous waveform
   (Hann overlap-add, ``evaluation/assemble.py``) and scores it against the raw
   ``Resp_Volts`` reference;
2. reports one number per subject -- the HEADLINE is band-limited Pearson
   (0.1-0.6 Hz), which is scale/offset-invariant AND drift-robust;
3. aggregates across subjects with a bootstrap 95% CI (each subject = one
   observation), and compares each subject against the single-tone TEMPLATE
   baseline it must beat;
4. writes ``subject_metrics.json`` and one figure per subject.

PROTOCOL: a single fixed subject-disjoint TEST set now; ``--test_subjects`` is
the hook for k-fold subject-disjoint cross-validation later -- a fold is just a
different subject list, no code change (see ``code/notes/design/Evaluation_RespWaveform_protocol.md``).

Usage (from ``code/``)::

    python runners/run_evaluate_subject.py --pred_dir OUT_DIR --waveform resp \\
        --fs 100 --band 0.1,0.6 --test_subjects F001,F002,F003
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import metrics as _metrics
from evaluation.assemble import affine_calibrate, assemble_session, group_entries
from evaluation.report import plot_session_waveform, save_json
from runners.run_evaluate_session import (_task_of_name, read_reference,
                                          smooth_band)

__all__ = ['main', 'subject_of', 'aggregate_subjects']

#: session metrics aggregated per subject / task / group. Cross-space raw
#: mae/rmse are deliberately EXCLUDED (the prediction is session-z-scored while
#: the reference is raw volts, so they track the reference DC, not error);
#: the volts-calibrated ``*_affine`` terms are used instead.
_AGG_KEYS = ('pearson', 'pearson_band', 'template_pearson_band', 'mae_affine',
             'rmse_affine', 'psd_mae', 'dominant_freq_error_hz', 'hf_power_rel',
             'mae_bpm', 'within_10pct')


def subject_of(session_name):
    """``'F001_T7'`` -> ``'F001'`` (the tested-subject key)."""
    s = str(session_name)
    return s.rsplit('_', 1)[0] if '_' in s else s


def _session_metrics(yh, ref, fs, band):
    """All reported metrics for ONE assembled session (1-D ``yh`` / ``ref``)."""
    yh2 = np.atleast_2d(np.asarray(yh, float))
    ref2 = np.atleast_2d(np.asarray(ref, float))
    rec = {'tier1': dict(_metrics.time_domain_metrics(yh2, ref2))}
    rec['pearson_band'] = float(_metrics.band_limited_pearson(yh2, ref2, fs, band))
    tmpl, freq = _metrics.single_tone_template(ref2, fs, band)
    rec['template_pearson_band'] = float(
        _metrics.band_limited_pearson(tmpl, ref2, fs, band))
    rec['template_freq_hz'] = float(freq)
    try:
        rec['tier2'] = dict(_metrics.spectral_metrics(yh2, ref2, fs=fs, band=band))
    except ImportError:
        rec['tier2'] = {}
    a, b, fitted = affine_calibrate(yh2[0], ref2[0])
    if np.isfinite(fitted).all():
        fit = _metrics.time_domain_metrics(np.atleast_2d(fitted), ref2)
        rec['affine'] = {'a': float(a), 'b': float(b),
                         'mae': fit['mae'], 'rmse': fit['rmse']}
    return rec


def _window_rates(yh, ref, fs, band, window_s):
    """Per-window ``(pred, target)`` respiration rates (BPM) for one session.

    The session is sliced into non-overlapping ``window_s`` windows and the
    dominant in-band rate is estimated for the prediction and the reference in
    each. A session shorter than one window is scored as a single window (a
    short clip cannot resolve a rate -- the caller should prefer sessions long
    enough to hold several respiration cycles).
    """
    yh = np.asarray(yh, float).reshape(-1)
    ref = np.asarray(ref, float).reshape(-1)
    n = int(min(yh.size, ref.size))
    win = int(round(float(window_s) * float(fs)))
    if win < 4 or n < win:
        spans = [(0, n)] if n >= 4 else []
    else:
        spans = [(i, i + win) for i in range(0, n - win + 1, win)]
    pw, tw = [], []
    for lo, hi in spans:
        pw.append(_metrics.estimate_rate_bpm(np.atleast_2d(yh[lo:hi]), fs, band)[0])
        tw.append(_metrics.estimate_rate_bpm(np.atleast_2d(ref[lo:hi]), fs, band)[0])
    return np.asarray(pw, float), np.asarray(tw, float)


def _flat(rec):
    """Flatten a session record to the space-consistent metrics for grouping."""
    out = {}
    t1 = rec.get('tier1') or {}
    if np.isfinite(t1.get('pearson', np.nan)):
        out['pearson'] = float(t1['pearson'])
    for k in ('pearson_band', 'template_pearson_band'):
        if np.isfinite(rec.get(k, np.nan)):
            out[k] = float(rec[k])
    t2 = rec.get('tier2') or {}
    for k in ('psd_mae', 'dominant_freq_error_hz', 'hf_power_rel'):
        if np.isfinite(t2.get(k, np.nan)):
            out[k] = float(t2[k])
    aff = rec.get('affine') or {}
    for k in ('mae', 'rmse'):
        if np.isfinite(aff.get(k, np.nan)):
            out[f'{k}_affine'] = float(aff[k])
    rate = rec.get('rate') or {}
    for k in ('mae_bpm', 'within_10pct'):
        if np.isfinite(rate.get(k, np.nan)):
            out[k] = float(rate[k])
    return out


def _figure_metrics(rec):
    """Space-consistent numbers for a subject figure's title (see the protocol)."""
    out = {}
    if np.isfinite(rec.get('pearson_band', np.nan)):
        out['pearson_band'] = rec['pearson_band']
    t1 = rec.get('tier1') or {}
    if np.isfinite(t1.get('pearson', np.nan)):
        out['pearson'] = t1['pearson']
    aff = rec.get('affine') or {}
    for k, label in (('mae', 'mae_aff_V'), ('rmse', 'rmse_aff_V')):
        if np.isfinite(aff.get(k, np.nan)):
            out[label] = aff[k]
    t2 = rec.get('tier2') or {}
    if np.isfinite(t2.get('psd_mae', np.nan)):
        out['psd_mae'] = t2['psd_mae']
    return out


def _evaluate_session(name, groups, entries, preds, fs, band, smooth, raw_fs,
                      waveform, rate_window_s=0.0, rate_tol=0.10):
    """Assemble + score ONE session. Returns ``(rec, t, ref, yh)`` or ``None``."""
    pairs = groups[name]
    rows = [i for i, _ in pairs]
    offs = [o for _, o in pairs]
    t, y, cov = assemble_session(preds[rows], offs, fs)
    sig = entries[rows[0]].get('signals_file') or ''
    if not sig or not os.path.isfile(sig):
        return None
    ref_full = read_reference(sig, waveform, fs=fs, raw_fs=raw_fs)
    # Score ONLY the samples a clip actually covers (see evaluation.assemble).
    n = min(y.size, ref_full.size)
    cov = cov[:n]
    duration_s = float(t[-1]) if t.size else 0.0
    if not cov.any():
        return None
    ref, yh = ref_full[:n][cov].copy(), y[:n][cov].copy()
    t = t[:n][cov]
    if smooth is not None:
        ref = smooth_band(ref, fs, smooth)
        yh = smooth_band(yh, fs, smooth)
    rec = _session_metrics(yh, ref, fs, band)
    rec['subject'] = subject_of(name)
    rec['task'] = entries[rows[0]].get('task') or _task_of_name(name)
    rec['n_clips'] = len(rows)
    rec['coverage'] = float(cov.mean())
    rec['duration_s'] = duration_s
    if rate_window_s and float(rate_window_s) > 0:
        pw, tw = _window_rates(yh, ref, fs, band, rate_window_s)
        rec['rate'] = _metrics.rate_summary(pw, tw, tol=rate_tol)
        rec['_rate_windows'] = (pw, tw)     # popped by main(); not persisted
    return rec, t, ref, yh


def aggregate_subjects(quality, subject_of_name, task_of, levels_of, keys):
    """Per-subject / per-task / per-group aggregation + the across-subject CI.

    :param quality: ``{session_name: {metric: float}}`` (see :func:`_flat`).
    :param subject_of_name: ``{session_name: subject}``.
    :param task_of: ``{session_name: task_label}``.
    :param levels_of: ``{session_name: [group, ...]}``.
    :param keys: metric keys to aggregate.
    :returns: ``{'per_subject', 'per_task', 'per_group', 'overall'}`` where
        ``overall`` carries the bootstrap CI of the per-subject headline (and of
        the template baseline) plus the fraction of subjects beating it.
    """
    per_subject = _metrics.breakdown_by_label(quality, subject_of_name, keys)
    per_task = _metrics.breakdown_by_label(quality, task_of, keys)
    per_group = _metrics.breakdown_by_label(quality, levels_of, keys)

    subj_head = [v.get('pearson_band_median', np.nan)
                 for v in per_subject.values()]
    subj_tmpl = [v.get('template_pearson_band_median', np.nan)
                 for v in per_subject.values()]
    beats = [1.0 if (np.isfinite(m) and np.isfinite(t) and m > t) else 0.0
             for m, t in zip(subj_head, subj_tmpl)]
    overall = {
        'pearson_band': _metrics.bootstrap_ci(subj_head),
        'template_pearson_band': _metrics.bootstrap_ci(subj_tmpl),
        'beat_template_frac_subjects': (float(np.mean(beats)) if beats
                                        else float('nan')),
    }
    return {'per_subject': per_subject, 'per_task': per_task,
            'per_group': per_group, 'overall': overall}


def _median_session(session_band_pairs):
    """The ``(name, pearson_band)`` whose band-limited Pearson is the median."""
    ordered = sorted(session_band_pairs,
                     key=lambda p: (np.nan_to_num(p[1]), p[0]))
    return ordered[len(ordered) // 2][0]


def get_args():
    p = argparse.ArgumentParser('Subject-level (tested-subject) evaluation',
                                add_help=True)
    p.add_argument('--pred_dir', default='', type=str,
                   help='run dir holding preds.npy / entries.json')
    p.add_argument('--pred_path', default='', type=str)
    p.add_argument('--entries_path', default='', type=str)
    p.add_argument('--waveform', default='resp', choices=['bp', 'resp'])
    p.add_argument('--fs', default=100.0, type=float)
    p.add_argument('--band', default=None, type=str,
                   help='band-limited metric band "lo,hi" (default resp '
                        '0.1,0.6 / bp 1.0,2.5)')
    p.add_argument('--raw_fs', default=1000.0, type=float,
                   help='native rate of a raw one-value-per-line reference')
    p.add_argument('--test_subjects', default='', type=str,
                   help='CSV of HELD-OUT test subjects to report. Empty = every '
                        'subject present. This is the k-fold hook.')
    p.add_argument('--out', default='', type=str,
                   help='subject-metrics JSON (default <pred_dir>/subject_metrics.json)')
    p.add_argument('--fig_dir', default='', type=str,
                   help='figure dir (default <pred_dir>/subject_figures)')
    p.add_argument('--zoom_s', default=20.0, type=float)
    p.add_argument('--max_sessions', default=0, type=int)
    p.add_argument('--rate_window_s', default=30.0, type=float,
                   help='window length (s) for the derived-vital-sign RATE '
                        'report (MAE BPM + %% within tolerance); 0 disables')
    p.add_argument('--rate_tol', default=0.10, type=float,
                   help='relative tolerance for the within-tol%% rate metric '
                        '(0.10 = the 10%% acceptance margin)')
    p.add_argument('--smooth', default='', type=str,
                   help="zero-phase SOS band applied to BOTH series, e.g. "
                        "'0.1,0.6'; empty = raw")
    p.add_argument('--n_boot', default=2000, type=int)
    p.add_argument('--no_figures', action='store_true', default=False)
    return p.parse_args()


def main(args):
    d = args.pred_dir
    pred_path = args.pred_path or os.path.join(d, 'preds.npy')
    entries_path = args.entries_path or os.path.join(d, 'entries.json')
    for p in (pred_path, entries_path):
        if not os.path.isfile(p):
            raise SystemExit(
                f'missing {p}. Run runners/run_waveform.py with --save_preds '
                f'(or save_preds: true) first.')
    out_path = args.out or os.path.join(d or '.', 'subject_metrics.json')
    fig_dir = args.fig_dir or os.path.join(d or '.', 'subject_figures')

    preds = np.atleast_2d(np.load(pred_path)).astype(float)
    with open(entries_path) as f:
        entries = json.load(f)
    band = (tuple(float(x) for x in args.band.split(',')) if args.band
            else ((1.0, 2.5) if args.waveform == 'bp' else (0.1, 0.6)))
    smooth = (tuple(float(x) for x in str(args.smooth).split(','))
              if str(args.smooth).strip() else None)
    if smooth is not None:
        preds = smooth_band(preds, args.fs, smooth)

    test = {s.strip() for s in str(args.test_subjects).split(',') if s.strip()}
    groups = group_entries(entries, args.fs)
    names = [n for n in sorted(groups) if not test or subject_of(n) in test]
    names = names[:args.max_sessions or None]
    if not names:
        raise SystemExit(f'no sessions selected '
                         f'(test_subjects={sorted(test) or "ALL"}).')
    print(f'[subject-eval] {len(names)} session(s), '
          f'test_subjects={sorted(test) or "ALL"}, waveform={args.waveform}, '
          f'band={band}, headline=pearson_band')

    quality, subj_of, task_of, levels_of = {}, {}, {}, {}
    session_names, per_subject_sessions = [], {}
    rate_pred_all, rate_tgt_all = [], []
    for name in names:
        out = _evaluate_session(name, groups, entries, preds, args.fs, band,
                                smooth, args.raw_fs, args.waveform,
                                args.rate_window_s, args.rate_tol)
        if out is None:
            print(f'[subject-eval] {name}: skip (no reference found)')
            continue
        rec, _, _, _ = out
        rw = rec.pop('_rate_windows', None)
        if rw is not None:
            rate_pred_all.append(rw[0])
            rate_tgt_all.append(rw[1])
        quality[name] = _flat(rec)
        subj_of[name] = rec['subject']
        task_of[name] = rec['task']
        levels_of[name] = list(entries[groups[name][0][0]].get('task_level') or ())
        session_names.append(name)
        per_subject_sessions.setdefault(rec['subject'], []).append(
            (name, rec['pearson_band']))

    agg = aggregate_subjects(quality, subj_of, task_of, levels_of, _AGG_KEYS)
    agg['overall']['pearson_band'] = _metrics.bootstrap_ci(
        [v.get('pearson_band_median', np.nan)
         for v in agg['per_subject'].values()], n_boot=args.n_boot)
    agg['overall']['template_pearson_band'] = _metrics.bootstrap_ci(
        [v.get('template_pearson_band_median', np.nan)
         for v in agg['per_subject'].values()], n_boot=args.n_boot)

    if not args.no_figures and per_subject_sessions:
        for subject, pairs in sorted(per_subject_sessions.items()):
            pick = _median_session(pairs)
            out = _evaluate_session(pick, groups, entries, preds, args.fs, band,
                                    smooth, args.raw_fs, args.waveform,
                                    args.rate_window_s, args.rate_tol)
            if out is None:
                continue
            rec, t, ref, yh = out
            plot_session_waveform(
                t[:yh.size], ref, yh,
                out_png=os.path.join(fig_dir, f'subject_{subject}.png'),
                title=(f'{subject} | respiration reconstruction | '
                       f'median session {pick} | {rec["n_clips"]} clips'),
                fs=args.fs, zoom_s=args.zoom_s, metrics=_figure_metrics(rec))

    doc = {'waveform': args.waveform, 'fs': args.fs, 'band': list(band),
           'smooth': (list(smooth) if smooth is not None else None),
           'headline': 'pearson_band',
           'test_subjects': sorted(test) or 'ALL',
           'n_sessions': len(session_names),
           'n_subjects': len(agg['per_subject']),
           'per_subject': agg['per_subject'],
           'per_task': agg['per_task'],
           'per_group': agg['per_group'],
           'overall': agg['overall'],
           'sessions': {n: quality[n] for n in session_names}}
    if rate_pred_all:
        doc['rate_overall'] = _metrics.rate_summary(
            np.concatenate(rate_pred_all), np.concatenate(rate_tgt_all),
            tol=args.rate_tol)
    save_json(out_path, doc)

    ci = agg['overall']['pearson_band']
    tmpl = agg['overall']['template_pearson_band']
    print(f'[subject-eval] HEADLINE band-limited Pearson (per-subject median '
          f'across {ci["n"]} subject(s)): {ci["point"]:+.3f} '
          f'[95% CI {ci["lo"]:+.3f}, {ci["hi"]:+.3f}]')
    print(f'[subject-eval] template baseline '
          f'{tmpl["point"]:+.3f} [95% CI {tmpl["lo"]:+.3f}, {tmpl["hi"]:+.3f}]; '
          f'beat-template subjects: '
          f'{agg["overall"]["beat_template_frac_subjects"]:.0%}')
    print(f'[subject-eval] wrote {out_path} and {len(agg["per_subject"])} '
          f'figure(s) to {fig_dir}')
    if 'rate_overall' in doc:
        r = doc['rate_overall']
        wkey = f'within_{int(round(args.rate_tol * 100))}pct'
        print(f'[subject-eval] RATE (derived vital sign, {args.rate_window_s:g} s '
              f'window): MAE {r["mae_bpm"]:.2f} BPM, '
              f'{wkey} {r.get(wkey, float("nan")):.0%} (n={r["n"]} windows)')


if __name__ == '__main__':
    main(get_args())
