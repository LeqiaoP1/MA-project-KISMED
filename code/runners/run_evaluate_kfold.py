"""Pooled k-fold (subject-disjoint) evaluation over ALL BP4D+ subjects.

The Stage-3 pipeline is run once per fold, each fold holding out a different
subject block (``protocol.FoldPlan``, K = 5). Every fold therefore predicts
EVERY subject exactly once -- namely the subjects it never trained on -- so
pooling the folds gives a cross-validated number for the whole corpus.

With ``--exp_root`` the ADR-001 gate runs first: five compliant folds of ONE
partition, all scored once, under one selection rule. ``--root/--run_prefix``
is the legacy mode that cannot check any of that.

This runner consumes the per-fold dumps (``preds.npy`` + ``entries.json``) and,
per session:

1. assembles the fold's clip predictions into one continuous waveform
   (Hann overlap-add, ``evaluation/assemble.py``);
2. scores it against the respiration reference in the metric families asked for
   by the evaluation protocol
   (``code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md`` sec. 4):

   * **pearson** -- ``pearson_band`` (band-limited 0.1-0.6 Hz, the drift-robust
     headline) plus the plain ``pearson`` for contrast and the single-tone
     TEMPLATE baseline it must beat;
   * **rate** -- per-window ``estimate_rate_bpm`` -> ``mae_bpm`` and
     ``within_10pct`` (``--rate_window_s``, default 30 s), pooled over every
     window of every fold;
   * **amplitude is NOT a metric** (protocol sec. 4.1, retired 2026-11-01: the
     claim is shape and rhythm). ``amp_ratio`` / ``amp_rel_err`` /
     ``mae_affine`` / ``rmse_affine`` are still COMPUTED and written per
     session as oracle-calibrated diagnostics, but they are excluded from
     ``overall``, from the console summary and from the per-fold table;

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

from data import task_groups as _tgrp
from evaluation import metrics as _metrics
from evaluation.assemble import affine_calibrate, assemble_session, group_entries
from evaluation.report import save_json
from runners.run_evaluate_session import (_task_of_name, read_reference,
                                          read_reference_from_cache)
from runners.run_evaluate_subject import _window_rates, subject_of

__all__ = ['main', 'get_args', 'score_fold', 'aggregate_kfold']

#: Per-session metrics carried through the per-subject / per-task aggregation.
#: Raw cross-space ``mae``/``rmse`` are DELIBERATELY absent (the prediction is
#: z-scored per clip while the reference is raw volts, so they only track the
#: reference DC); the volts-calibrated ``*_affine`` terms carry the point error.
_SESSION_KEYS = ('pearson', 'pearson_band', 'template_pearson_band',
                 'amp_ratio', 'amp_rel_err', 'amp_ref', 'amp_pred',
                 'mae_affine', 'rmse_affine', 'psd_mae',
                 'dominant_freq_error_hz', 'hf_power_rel',
                 'mae_bpm', 'within_10pct')

#: The headline aggregates (one bootstrap CI each, over SUBJECTS). Amplitude is
#: DELIBERATELY absent: the evaluation claim is morphological shape and rhythm,
#: and any amplitude number needs the session's true scale (protocol sec. 4.1).
_HEADLINE = ('pearson_band', 'mae_bpm')


def _fold_dir(root, prefix, k):
    """Prefer ``<prefix><k>``; accept ``<prefix>_fold<k>`` too."""
    for cand in (f'{prefix}{k}', f'{prefix}_fold{k}'):
        d = os.path.join(root, cand)
        if os.path.isdir(d):
            return d
    return os.path.join(root, f'{prefix}{k}')


def score_fold(pred_dir, fs, raw_fs, band, waveform, rate_window_s, rate_tol,
               fold, max_sessions=0, group_override=None,
               resp_cache='', resp_cache_key=''):
    """Score every session of ONE fold.

    :param group_override: optional ``{task: (group, ...)}``
        (:func:`data.task_groups.levels_by_task`) that REPLACES the
        ``task_level`` recorded in ``entries.json``. ``None`` keeps the recorded
        labels. The grouping is aggregation-only, so this re-labels an existing
        dump without re-training.
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
        ref_mask = None
        t, y, cov = assemble_session(preds[rows], offs, fs)
        if resp_cache:
            # Design C8: reference and label share ONE artefact.
            try:
                _, ref_full, ref_mask, _root = read_reference_from_cache(
                    resp_cache, name, fs=fs, cache_key=resp_cache_key)
            except (FileNotFoundError, ValueError) as exc:
                print(f'[kfold] fold {fold} {name}: skip (resp_cache: {exc})')
                continue
        else:
            sig = entries[rows[0]].get('signals_file') or ''
            if not sig or not os.path.isfile(sig):
                print(f'[kfold] fold {fold} {name}: skip (no reference)')
                continue
            ref_full = read_reference(sig, waveform, fs=fs, raw_fs=raw_fs)
        n = int(min(y.size, ref_full.size))
        scored = np.asarray(cov[:n], dtype=bool).copy()
        if ref_mask is not None:
            scored &= np.asarray(ref_mask[:n], dtype=bool)
        if not scored.any():
            print(f'[kfold] fold {fold} {name}: skip (no covered sample)')
            continue
        ref, yh = ref_full[:n][scored].copy(), y[:n][scored].copy()

        task_label = entries[rows[0]].get('task') or _task_of_name(name)
        rec = {
            'session': name,
            'subject': subject_of(name),
            'task': task_label,
            'task_level': (list(_tgrp.levels_for_task(group_override, task_label))
                           if group_override is not None
                           else list(entries[rows[0]].get('task_level') or ())),
            'fold': int(fold),
            'n_clips': len(rows),
            'duration_s': float(t[-1]) if t.size else 0.0,
            'coverage': float(scored.mean()),
        }
        if ref_mask is not None:
            rec['reference_coverage'] = float(np.asarray(ref_mask[:n]).mean())
            rec['reference_source'] = 'resp_cache'
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


def gate_protocol_partitions(exp_root, folds, allow_partial=False):
    """ADR-001 gate: only compliant folds of ONE partition may be pooled.

    Five independently-correct folds can still be summed into a meaningless
    number if they were built on different partitions, if one of them is a dev
    plan, or if two folds hold out the same subject. Every one of those is a
    silent error at aggregation time, so they are refused here rather than
    noticed later in the table.

    :returns: ``{fold: FoldPlan}``.
    """
    from protocol import K, ProtocolError, load_plan

    plans, missing = {}, []
    for k in folds:
        path = os.path.join(exp_root, f'fold{k}', 'partition.json')
        if not os.path.isfile(path):
            missing.append(path)
            continue
        try:
            plans[k] = load_plan(path)
        except ProtocolError as exc:
            raise SystemExit(f'[kfold] fold {k}: {exc}') from None
    if missing:
        raise SystemExit(
            '[kfold] missing partition record(s): ' + ', '.join(missing)
            + '. Create them with runners/run_partition.py.')

    dev = sorted(k for k, plan in plans.items() if plan.dev_smoke)
    if dev:
        raise SystemExit(
            f'[kfold] fold(s) {dev} used a --dev_smoke plan; those exist for '
            f'plumbing tests only and can never be pooled.')
    if len(folds) != K and not allow_partial:
        raise SystemExit(
            f'[kfold] ADR-001 pools exactly K={K} folds; got {len(folds)} '
            f'({sorted(folds)}). Finish the sweep, or pass --allow_partial for '
            f'a smoke pool (the artefact is then marked partial).')

    first = plans[folds[0]]
    for k, plan in sorted(plans.items()):
        if (plan.split_seed, plan.split_stratify, plan.n_folds) != \
                (first.split_seed, first.split_stratify, first.n_folds):
            raise SystemExit(
                f'[kfold] fold {k} used partition (seed='
                f'{plan.split_seed}, stratify={plan.split_stratify!r}, K='
                f'{plan.n_folds}) but fold {folds[0]} used (seed='
                f'{first.split_seed}, stratify={first.split_stratify!r}, K='
                f'{first.n_folds}). Folds of one report must come from ONE '
                f'partition rule.')
        if plan.universe_digest != first.universe_digest:
            raise SystemExit(
                f'[kfold] fold {k} partitions a different subject universe '
                f'({plan.universe_digest[:12]}) than fold {folds[0]} '
                f'({first.universe_digest[:12]}); the corpus changed between '
                f'runs, so the folds are not comparable.')

    seen = {}
    for k, plan in sorted(plans.items()):
        for subject in plan.roles['test']:
            if subject in seen:
                raise SystemExit(
                    f'[kfold] {subject} is the test subject of folds '
                    f'{seen[subject]} and {k}; the folds are not disjoint.')
            seen[subject] = k
    if not allow_partial and set(seen) != set(first.universe):
        uncovered = sorted(set(first.universe) - set(seen))
        raise SystemExit(
            f'[kfold] the pooled folds cover {len(seen)} of '
            f'{len(first.universe)} subject(s); {len(uncovered)} never appear '
            f'as a test subject (e.g. {uncovered[:5]}). That is not a '
            f'cross-validation of the corpus.')
    return plans


def gate_test_artefacts(exp_root, folds):
    """Every fold must have been scored once, under one selection rule."""
    from protocol import ProtocolError, read_json, read_selection

    rules = {}
    for k in folds:
        metrics_path = os.path.join(exp_root, f'fold{k}', 'stage3', 'test',
                                    'metrics.json')
        if not os.path.isfile(metrics_path):
            raise SystemExit(
                f'[kfold] fold {k} has no test artefact ({metrics_path}); run '
                f'`runners/run_waveform.py --eval_role test` for that fold.')
        try:
            doc = read_json(metrics_path)
        except ProtocolError as exc:
            raise SystemExit(f'[kfold] fold {k}: {exc}') from None
        if doc.get('skipped'):
            raise SystemExit(
                f'[kfold] fold {k}: the test role was not scorable '
                f'({doc.get("skip_reason", "unknown")}). A fold with nothing to '
                f'score cannot be part of a pooled number.')
        selection_path = os.path.join(exp_root, f'fold{k}', 'stage3',
                                      'selection.json')
        if os.path.isfile(selection_path):
            rules[k] = str(read_selection(selection_path).get('rule',
                                                             'unrecorded'))
        else:
            rules[k] = 'unrecorded'
    if len(set(rules.values())) > 1:
        raise SystemExit(
            f'[kfold] the folds used different selection rules {rules}; the '
            f'delivered checkpoint of each fold means something else, so their '
            f'scores are not comparable.')
    return rules


def get_args():
    p = argparse.ArgumentParser(
        'Pooled k-fold (subject-disjoint) evaluation over all subjects',
        add_help=True)
    p.add_argument('--exp_root', default='', type=str,
                   help='ADR-001 experiment root: reads '
                        '<exp_root>/fold<k>/{partition.json,stage3/test/} and '
                        'ENFORCES the protocol gate (five compliant folds of one '
                        'partition, one selection rule). Preferred over '
                        '--root/--run_prefix, which cannot check any of that.')
    p.add_argument('--allow_partial', action='store_true', default=False,
                   help='SMOKE ONLY: pool fewer than K folds; the artefact is '
                        'marked partial and must not be quoted as a result')
    p.add_argument('--root', default='../output/finetune', type=str,
                   help='parent dir holding the per-fold run dirs')
    p.add_argument('--run_prefix', default='resp_tir_roi_hpc_fold', type=str,
                   help='fold dir name prefix (fold k dir = <prefix>k)')
    p.add_argument('--folds', default='0,1,2,3,4', type=str,
                   help='CSV of fold indices to pool')
    p.add_argument('--waveform', default='resp', choices=['bp', 'resp'])
    p.add_argument('--fs', default=100.0, type=float)
    p.add_argument('--resp_cache', default='', type=str,
                   help='OUT_ROOT of the offline respiration cache. When set, '
                        'the REFERENCE is read from the same artefact as the '
                        'training labels (design C8) and its per-sample mask '
                        'restricts scoring. Empty = legacy raw Resp_Volts.txt.')
    p.add_argument('--resp_cache_key', default='', type=str,
                   help='pin the cache DIRECTORY name instead of resolving it.')
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
    p.add_argument('--task_groups', default='', type=str,
                   help="override the task -> distortion-group mapping recorded "
                        "in each fold's entries.json, e.g. "
                        "'low=T2|T3|T5;moderate=T1|T4;high=T6|T7|T8|T9|T10' "
                        "(see data/task_groups.py). Empty = use the recorded "
                        "task_level.")
    return p.parse_args()


def main(args):
    band = (tuple(float(x) for x in args.band.split(',')) if args.band
            else ((1.0, 2.5) if args.waveform == 'bp' else (0.1, 0.6)))
    folds = [int(k) for k in str(args.folds).split(',') if k.strip()]

    # ADR-001 mode: the evidence lives under <exp_root>/fold<k>. Everything the
    # gate checks (one partition, one selection rule, all folds scored) is
    # invisible to --root/--run_prefix, which is why that mode is legacy.
    protocol_block = {}
    fold_dirs = {}
    if getattr(args, 'exp_root', ''):
        plans = gate_protocol_partitions(args.exp_root, folds,
                                        allow_partial=args.allow_partial)
        rules = gate_test_artefacts(args.exp_root, folds)
        for k in folds:
            fold_dirs[k] = os.path.join(args.exp_root, f'fold{k}', 'stage3',
                                        'test')
        protocol_block = {
            'adr': 'ADR-001',
            'compliant': not args.allow_partial,
            'partial': bool(args.allow_partial),
            'exp_root': os.path.abspath(args.exp_root),
            'n_folds': len(folds),
            'universe_digest': plans[folds[0]].universe_digest,
            'split_seed': plans[folds[0]].split_seed,
            'split_stratify': plans[folds[0]].split_stratify,
            'partition_digests': {k: plans[k].digest for k in folds},
            'counts': {k: plans[k].counts for k in folds},
            'selection_rule': rules[folds[0]],
            'selection_rules': rules,
        }
        print(f'[kfold] protocol gate OK: {len(folds)} fold(s), universe '
              f'{plans[folds[0]].universe_digest[:12]}, '
              f'rule={rules[folds[0]]}'
              + ('  [PARTIAL: smoke only]' if args.allow_partial else ''))

    out_path = args.out or (
        os.path.join(args.exp_root, 'report', 'kfold.json')
        if getattr(args, 'exp_root', '') else
        os.path.join(args.root, f'kfold_{args.run_prefix.rstrip("_")}.json'))
    wkey = f'within_{int(round(args.rate_tol * 100))}pct'

    records, summaries, pw_all, tw_all = [], [], [], []
    # Optional RE-LABEL of existing dumps (aggregation-only; no re-training).
    group_levels = _tgrp.parse_task_groups(getattr(args, 'task_groups', ''))
    group_override = (_tgrp.levels_by_task(group_levels) if group_levels
                      else None)
    for k in folds:
        d = (fold_dirs[k] if fold_dirs
             else _fold_dir(args.root, args.run_prefix, k))
        with open(os.path.join(d, 'entries.json')) as f:
            n_entries = len(json.load(f))
        recs, (pw, tw) = score_fold(
            d, args.fs, args.raw_fs, band, args.waveform,
            args.rate_window_s, args.rate_tol, k, args.max_sessions,
            group_override=group_override,
            resp_cache=args.resp_cache, resp_cache_key=args.resp_cache_key)
        records.extend(recs)
        summaries.append(_fold_summary(k, recs, n_entries))
        pw_all.append(pw)
        tw_all.append(tw)

    if group_override is not None:
        missing = _tgrp.unmapped_tasks(
            [r['task'] for r in records], group_override)
        if missing:
            print(f'[kfold] WARNING: task(s) {", ".join(missing)} are in no '
                  f'--task_groups group; their sessions are EXCLUDED from '
                  f'per_group (breakdown_by_label skips an empty label list).')

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
            'waveform': args.waveform,
            'fs': args.fs,
            'raw_fs': args.raw_fs,
            'band': list(band),
            'pearson': 'band-limited Pearson (headline), plain Pearson for '
                       'contrast, single-tone template baseline',
            'amplitude': 'NOT a metric (protocol sec. 4.1) -- amp_ratio / '
                         'amp_rel_err / mae_affine / rmse_affine are recorded '
                         'per session as oracle-calibrated diagnostics only',
            'rate': f'dominant in-band FFT peak per {args.rate_window_s:g} s '
                    f'window; MAE (BPM) + {wkey}',
            'task_group_source': ('cli' if group_override is not None
                                  else 'entries'),
            'task_groups': {k: list(v) for k, v in group_levels.items()},
            'n_subjects': n_subjects,
            **(protocol_block or {'root': os.path.abspath(args.root),
                                  'run_prefix': args.run_prefix}),
        },
        'task_levels': {t: list(v) for t, v in (group_override or {}).items()},
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
    # "5-fold" was hardcoded here, so a partial sweep (--folds 0,1,2,3 while the
    # last fold is still queued) printed a header claiming five folds. Report the
    # number actually pooled.
    print(f'\n[kfold] ===== pooled {len(summaries)}-fold subject-disjoint '
          f'evaluation =====')
    print(f'[kfold] {n_subjects} subjects / {o["n_sessions"]} sessions / '
          f'{o["n_clips"]} clips, band {band} Hz')
    print(f'[kfold] PEARSON   band-limited (per-subject median): '
          f'{o["pearson_band"]["point"]:+.3f} '
          f'[95% CI {o["pearson_band"]["lo"]:+.3f}, '
          f'{o["pearson_band"]["hi"]:+.3f}]  |  template '
          f'{o["template_pearson_band"]["point"]:+.3f}, beaten by '
          f'{o["beat_template_frac_subjects"]:.0%} of subjects')
    print(f'[kfold] RATE      MAE          : '
          f'{o["mae_bpm"]["point"]:.2f} BPM '
          f'[95% CI {o["mae_bpm"]["lo"]:.2f}, {o["mae_bpm"]["hi"]:.2f}]'
          f'  |  pooled windows: MAE {rate_overall.get("mae_bpm", np.nan):.2f} '
          f'BPM, {wkey} {rate_overall.get(wkey, np.nan):.0%} '
          f'(n={rate_overall.get("n", 0)})')
    print('\n[kfold] per-fold (session medians):')
    print(f'  {"fold":>4} {"subj":>4} {"sess":>5} {"pearson_band":>13} '
          f'{"mae_bpm":>8} {"within10%":>10}')
    for s in summaries:
        print(f'  {s["fold"]:>4} {s["n_subjects"]:>4} {s["n_sessions"]:>5} '
              f'{s["pearson_band_median"]:>13.3f} '
              f'{s["mae_bpm_median"]:>8.2f} '
              f'{s["within_10pct_median"]:>10.2f}')
    print(f'\n[kfold] wrote {out_path}')


if __name__ == '__main__':
    main(get_args())
