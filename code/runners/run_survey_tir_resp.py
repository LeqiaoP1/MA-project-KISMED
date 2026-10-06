"""Whole-corpus quality survey for the TIR-ROI + RESP pipeline (CPU-only).

Sweeps EVERY ``<raw_root>/Thermal/<subject>/<task>`` session (1400 in the full
BP4D+ tree) and reports, per session and in aggregate:

* the usability gates the dataset itself applies
  (``missing_ir_features`` / ``invalid_ir_features`` / ``missing_resp_volts`` /
  ``resp_too_short`` / ``undecodable_video`` / ``too_short`` /
  ``all_clips_dropped``), plus ``n_vid`` vs ``n_ir`` mismatches and measured-fps
  deviations;
* how many 8 s clips each session yields, and how many are dropped for a
  ``(0,0)`` IR sentinel line or a (near-)constant respiration window;
* respiration-signal QUALITY: sample count, invalid samples, min/max/spread,
  and above all the fraction of samples pinned at the ``-10.0000 V`` rail (the
  corpus' dead-channel floor) -- both for the whole file and for the clips that
  would actually be trained on.

The heavy per-subject work is parallelised with ``multiprocessing`` (fork), so
this is meant to run as a Slurm CPU job, not on the login node::

    sbatch scripts/hpc/submit_survey_tir_resp.sbatch

or directly::

    python runners/run_survey_tir_resp.py --workers 32 \
        --output_dir $WORK_SCRATCH/tir_resp_survey

Outputs (under ``--output_dir``): ``tir_resp_survey.json`` (full records),
``tir_resp_survey_sessions.csv`` (one row per session) and a printed report.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import traceback
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import tir_resp_dataset as trd

#: the low/moderate/high split used by configs/pretrain/stage2_local_tir_roi_resp.yaml
DEFAULT_TASK_GROUPS = 'low=T2|T3;moderate=T4|T7|T8|T10;high=T1|T5|T6|T9'

#: the corpus' rail floor -- kept equal to the dataset's ``RESP_RAIL_V`` so the
#: survey and the clip filter never disagree (float32-exact, see the dataset).
DEFAULT_RAIL_V = -10.0

#: the ADOPTED clip filter threshold (volts, magnitude): a clip whose window
#: contains any sample at/past this is dropped by the touch rule.
DEFAULT_RAIL_TOUCH_V = 9.90

#: the dataset's own flat-window threshold (volts) -- the SHIPPED value, so the
#: ``clips flat`` line counts exactly what the guard is configured to drop.
DEFAULT_FLAT_SPREAD = 0.10

#: the shipped configs' clip geometry: an 8 s window with a 1 s hop
DEFAULT_CLIP_SECONDS = trd.DEFAULT_CLIP_SECONDS
DEFAULT_CLIP_STRIDE = 1.0

#: candidate thresholds for the trade-off curve (see ``--sweep``). The rail
#: values are magnitudes in volts; the spread values are the
#: ``min_signal_spread`` guard. Both are CLASSIFICATION thresholds -- the survey
#: always builds the corpus unfiltered and counts afterwards.
DEFAULT_RAIL_TOUCH_SWEEP = '9.99,9.97,9.95,9.9,9.8,9.7,9.5,9.0,8.0'
DEFAULT_SPREAD_SWEEP = '0.01,0.1,0.25,0.5,1.0,1.5,2.0,3.0'


def _sweep_keys(sweep: Sequence[float]) -> List[str]:
    """Stable column names for a sweep (``9.90 -> '9.9'``)."""
    return [f'{float(v):g}' for v in sweep]


def _subjects_from_arg(value: str) -> Optional[List[str]]:
    if not value:
        return None
    subs = [v.strip() for v in value.replace('|', ',').split(',') if v.strip()]
    return subs or None


def _float_list(value: str) -> List[float]:
    """``'9.9,9.5'`` -> ``[9.9, 9.5]`` (empty string -> no sweep)."""
    return [float(v) for v in str(value or '').replace('|', ',').split(',')
            if v.strip()]


def _list_subjects(raw_root: str) -> List[str]:
    th = os.path.join(raw_root, trd.RAW_TREE)
    return sorted(d for d in os.listdir(th)
                  if os.path.isdir(os.path.join(th, d)))


def _survey_subject(args: dict) -> dict:
    """Build the dataset for ONE subject and measure every used session."""
    subj = args['subject']
    try:
        ds = trd.BP4DPlusTIRRespDataset(
            raw_root=args['raw_root'], subjects=[subj], tasks=None,
            task_groups=args['task_groups'], task_set=None,
            input_size=args['input_size'], roi_padding=args['roi_padding'],
            clip_seconds=args['clip_seconds'], clip_stride=args['clip_stride'],
            min_signal_spread=0.0,          # keep every window; we classify below
            norm='none', allow_empty=True, verbose=False)
    except Exception as exc:                                 # pragma: no cover
        return {'subject': subj, 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(), 'sessions': []}

    # group the kept clips by session
    entries_by_session: Dict[str, List[dict]] = defaultdict(list)
    for e in ds.entries:
        entries_by_session[e['session']].append(e)

    rail_v = args['rail_v']
    # quantise exactly as the dataset does, so the survey and the filter agree
    # on a sample valued exactly at the threshold (see BP4DPlusTIRRespDataset)
    touch_v = float(np.float32(args['rail_touch_v']))
    flat_spread = args['flat_spread']
    rail_sweep = [float(v) for v in args['rail_touch_sweep']]
    spread_sweep = [float(v) for v in args['spread_sweep']]
    rows: List[dict] = []

    def _session_row(sess: str, meta: Optional[dict], status: str,
                     reason: str = '') -> dict:
        row = {'subject': subj, 'session': sess,
               'task': sess.split('_', 1)[1] if '_' in sess else '',
               'status': status, 'reason': reason,
               'n_vid': None, 'n_ir': None, 'n_resp': None, 'fps': None,
               'clips': 0, 'clips_dropped_sentinel': 0, 'clips_dropped_flat': 0,
               'resp_n': None, 'resp_nan': None, 'resp_min': None,
               'resp_max': None, 'resp_mean': None, 'resp_std': None,
               'resp_spread': None, 'resp_rail_pct': None,
               'clip_rail_gt0': None, 'clip_rail_gt10': None,
               'clip_rail_gt50': None, 'clip_fully_railed': None,
               'clip_flat': None,
               'clip_touched': None, 'clip_touched_lo': None,
               'clip_touched_hi': None}
        # trade-off curve: how many of THIS session's windows are dropped by
        # each candidate rail threshold, and -- among the windows that survive
        # the shipped rail rule -- by each candidate min_signal_spread
        row.update({f'clips_ge_{k}': None for k in _sweep_keys(rail_sweep)})
        row.update({f'kept_flat_lt_{k}': None
                    for k in _sweep_keys(spread_sweep)})
        if meta is not None:
            row.update(n_vid=meta.get('n_vid'), n_ir=meta.get('n_ir'),
                       n_resp=meta.get('n_resp'))
            row['fps'] = ds.session_fps.get(sess)
            row['clips_dropped_sentinel'] = meta.get('clips_dropped_sentinel')
            row['clips_dropped_flat'] = meta.get('clips_dropped_flat')
        return row

    # skipped sessions (keep the reason)
    for s in ds.skipped:
        rows.append(_session_row(s['session'], None, 'skipped', s['reason']))

    for sess, meta in ds.sessions.items():
        row = _session_row(sess, meta, 'used')
        try:
            y = trd._load_1d(meta['resp_file']).astype(np.float64)
        except Exception as exc:
            row['status'] = 'resp_read_error'
            row['reason'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
            continue
        n = int(y.size)
        row['resp_n'] = n
        row['resp_nan'] = int(np.count_nonzero(~np.isfinite(y)))
        if n:
            row['resp_min'] = float(np.nanmin(y))
            row['resp_max'] = float(np.nanmax(y))
            row['resp_mean'] = float(np.nanmean(y))
            row['resp_std'] = float(np.nanstd(y))
            row['resp_spread'] = float(np.nanmax(y) - np.nanmin(y))
            row['resp_rail_pct'] = float(
                100.0 * np.count_nonzero(y <= rail_v) / n)

        clips = entries_by_session.get(sess, [])
        row['clips'] = len(clips)
        if clips:
            rails, flats = [], 0
            n_lo = n_hi = n_touch = 0
            n_ge = {k: 0 for k in _sweep_keys(rail_sweep)}
            n_kept_flat = {k: 0 for k in _sweep_keys(spread_sweep)}
            for e in clips:
                w = ds._resp_clip(e).astype(np.float64)
                frac = float(np.count_nonzero(w <= rail_v) / w.size)
                rails.append(frac)
                lo = bool(np.any(w <= -touch_v))
                hi = bool(np.any(w >= touch_v))
                n_lo += int(lo)
                n_hi += int(hi)
                n_touch += int(lo or hi)
                sprd = float(w.max() - w.min())
                if sprd < flat_spread:
                    flats += 1
                peak = float(np.abs(w).max())
                for v in rail_sweep:
                    if peak >= v:
                        n_ge[f'{v:g}'] += 1
                if not (lo or hi):            # survives the shipped rail rule
                    for v in spread_sweep:
                        if sprd < v:
                            n_kept_flat[f'{v:g}'] += 1
            rails_arr = np.asarray(rails)
            row['clip_rail_gt0'] = int(np.count_nonzero(rails_arr > 0.0))
            row['clip_rail_gt10'] = int(np.count_nonzero(rails_arr > 0.10))
            row['clip_rail_gt50'] = int(np.count_nonzero(rails_arr > 0.50))
            row['clip_fully_railed'] = int(np.count_nonzero(rails_arr >= 1.0))
            row['clip_flat'] = int(flats)
            # the rule, measured exactly as the dataset implements it
            row['clip_touched_lo'] = n_lo
            row['clip_touched_hi'] = n_hi
            row['clip_touched'] = n_touch
            row.update({f'clips_ge_{k}': n for k, n in n_ge.items()})
            row.update({f'kept_flat_lt_{k}': n
                        for k, n in n_kept_flat.items()})
        rows.append(row)

    # a task that has a video but no usable clips is recorded as skipped only;
    # also surface tasks with a video but NO IRFeatures/Resp as discovered.
    discovered = trd.discover_sessions(args['raw_root'], subjects=[subj])
    seen = {r['session'] for r in rows}
    for spec in discovered:
        if spec['session'] not in seen:
            rows.append(_session_row(
                spec['session'], None, 'skipped',
                'missing_ir_features' if spec['ir_file'] is None
                else 'missing_resp_volts' if spec['resp_file'] is None
                else 'not_built'))
    return {'subject': subj, 'error': None, 'sessions': rows,
            'stats': ds.stats, 'warnings': list(ds.warnings)}


# --------------------------------------------------------------------------- #
# aggregation / report
# --------------------------------------------------------------------------- #
def _pct(a: int, b: int) -> str:
    return f'{100.0 * a / b:.1f}%' if b else 'n/a'


def _aggregate(rows: List[dict], task_groups: dict,
               rail_touch_v: float = DEFAULT_RAIL_TOUCH_V,
               rail_sweep: Sequence[float] = (),
               spread_sweep: Sequence[float] = (),
               flat_spread: float = DEFAULT_FLAT_SPREAD) -> dict:
    n_sess = len(rows)
    used = [r for r in rows if r['status'] == 'used']
    skipped = [r for r in rows if r['status'] != 'used']
    reasons = Counter(r['reason'].split(':')[0].strip() for r in skipped)

    n_clips = sum(r['clips'] for r in used)
    clips_rail0 = sum(r['clip_rail_gt0'] or 0 for r in used)
    clips_rail10 = sum(r['clip_rail_gt10'] or 0 for r in used)
    clips_rail50 = sum(r['clip_rail_gt50'] or 0 for r in used)
    clips_full = sum(r['clip_fully_railed'] or 0 for r in used)
    clips_flat = sum(r['clip_flat'] or 0 for r in used)
    clips_touch = sum(r['clip_touched'] or 0 for r in used)
    clips_touch_lo = sum(r['clip_touched_lo'] or 0 for r in used)
    clips_touch_hi = sum(r['clip_touched_hi'] or 0 for r in used)
    sessions_emptied_touch = sum(
        1 for r in used
        if r['clips'] and (r['clip_touched'] or 0) == r['clips'])

    # per level
    per_level: Dict[str, dict] = {}
    flat_col = f'kept_flat_lt_{flat_spread:g}'
    for name, tasks in task_groups.items():
        tset = set(tasks)
        lv_used = [r for r in used if r['task'] in tset]
        lv_all = [r for r in rows if r['task'] in tset]
        lv_clips = sum(r['clips'] for r in lv_used)
        lv_rail = sum(r['clip_rail_gt0'] or 0 for r in lv_used)
        lv_touch = sum(r['clip_touched'] or 0 for r in lv_used)
        lv_spread = sum(r.get(flat_col) or 0 for r in lv_used)
        rails = [r['resp_rail_pct'] for r in lv_used
                 if r['resp_rail_pct'] is not None]
        per_level[name] = {
            'sessions_discovered': len(lv_all),
            'sessions_used': len(lv_used),
            'clips': lv_clips,
            'clips_with_rail': lv_rail,
            'clips_touching_rail': lv_touch,
            'clips_kept_by_touch_rule': lv_clips - lv_touch,
            'clips_dropped_by_spread_rule': lv_spread,
            'clips_kept_after_both_rules': lv_clips - lv_touch - lv_spread,
            'sessions_emptied_by_touch_rule': sum(
                1 for r in lv_used
                if r['clips'] and (r['clip_touched'] or 0) == r['clips']),
            'sessions_emptied_by_spread_rule': sum(
                1 for r in lv_used
                if (r['clips'] - (r['clip_touched'] or 0))
                and (r.get(flat_col) or 0) == r['clips'] - (r['clip_touched'] or 0)),
            'sessions_with_any_rail': sum(1 for r in lv_used
                                          if (r['resp_rail_pct'] or 0) > 0),
            'mean_file_rail_pct': (float(np.mean(rails)) if rails else None),
        }

    # per task
    per_task: Dict[str, dict] = {}
    for r in rows:
        t = r['task']
        d = per_task.setdefault(t, {'discovered': 0, 'used': 0, 'clips': 0,
                                    'clips_with_rail': 0})
        d['discovered'] += 1
        if r['status'] == 'used':
            d['used'] += 1
            d['clips'] += r['clips']
            d['clips_with_rail'] += (r['clip_rail_gt0'] or 0)

    rails = [r['resp_rail_pct'] for r in used
             if r['resp_rail_pct'] is not None]

    # ---- trade-off curve over the two per-window knobs ----------------------
    rail_curve: Dict[str, dict] = {}
    for k in _sweep_keys(rail_sweep):
        col = f'clips_ge_{k}'
        n = sum(r.get(col) or 0 for r in used)
        emptied = sum(1 for r in used
                      if r['clips'] and (r.get(col) or 0) == r['clips'])
        rail_curve[k] = {'dropped': n, 'kept': n_clips - n,
                         'sessions_emptied': emptied,
                         'sessions_remaining': len(used) - emptied}
    spread_curve: Dict[str, dict] = {}
    kept_by_rail = n_clips - clips_touch
    for k in _sweep_keys(spread_sweep):
        n = sum(r.get(f'kept_flat_lt_{k}') or 0 for r in used)
        spread_curve[k] = {'dropped_from_kept': n,
                           'kept_after': kept_by_rail - n}
    return {
        'sessions_discovered': n_sess,
        'sessions_used': len(used),
        'sessions_skipped': len(skipped),
        'skip_reasons': dict(reasons.most_common()),
        'clips': n_clips,
        'clips_with_rail_gt0': clips_rail0,
        'clips_with_rail_gt10': clips_rail10,
        'clips_with_rail_gt50': clips_rail50,
        'clips_fully_railed': clips_full,
        'clips_flat_spread_lt_threshold': clips_flat,
        'flat_spread': flat_spread,
        'clips_dropped_by_spread_rule': sum(
            r.get(f'kept_flat_lt_{flat_spread:g}') or 0 for r in used),
        'clips_kept_after_both_rules': n_clips - clips_touch - sum(
            r.get(f'kept_flat_lt_{flat_spread:g}') or 0 for r in used),
        'sessions_emptied_by_spread_rule': sum(
            1 for r in used
            if (r['clips'] - (r['clip_touched'] or 0))
            and (r.get(f'kept_flat_lt_{flat_spread:g}') or 0)
            == r['clips'] - (r['clip_touched'] or 0)),
        'clips_touching_rail': clips_touch,
        'clips_touching_rail_neg': clips_touch_lo,
        'clips_touching_rail_pos': clips_touch_hi,
        'clips_dropped_by_touch_rule': clips_touch,
        'clips_kept_by_touch_rule': n_clips - clips_touch,
        'sessions_emptied_by_touch_rule': sessions_emptied_touch,
        'rail_touch_v': rail_touch_v,
        'rail_touch_curve': rail_curve,
        'spread_curve': spread_curve,
        'sessions_with_rail_file_gt0': sum(1 for r in used
                                           if (r['resp_rail_pct'] or 0) > 0),
        'sessions_file_rail_gt10': sum(1 for r in used
                                       if (r['resp_rail_pct'] or 0) > 10),
        'sessions_file_rail_gt50': sum(1 for r in used
                                       if (r['resp_rail_pct'] or 0) > 50),
        'mean_file_rail_pct': (float(np.mean(rails)) if rails else None),
        'median_file_rail_pct': (float(np.median(rails)) if rails else None),
        'per_level': per_level,
        'per_task': dict(sorted(per_task.items())),
    }


def _print_report(agg: dict, worst: List[dict], task_groups: dict) -> None:
    L = '=' * 76
    print(L)
    print('TIR-ROI + RESP -- WHOLE-CORPUS QUALITY SURVEY')
    print(L)
    print(f"sessions discovered : {agg['sessions_discovered']}")
    print(f"sessions used       : {agg['sessions_used']}")
    print(f"sessions skipped    : {agg['sessions_skipped']}")
    print(f"clips (8 s windows) : {agg['clips']}")
    print()
    print('--- skip reasons ---')
    for reason, n in agg['skip_reasons'].items():
        print(f'  {reason:<32} {n:5d}  ({_pct(n, agg["sessions_discovered"])})')
    print()
    print('--- respiration rail (-10.0000 V) ---')
    print(f"  mean file rail %    : {agg['mean_file_rail_pct']}")
    print(f"  median file rail %  : {agg['median_file_rail_pct']}")
    print(f"  sessions rail > 0%  : {agg['sessions_with_rail_file_gt0']} "
          f"({_pct(agg['sessions_with_rail_file_gt0'], agg['sessions_used'])})")
    print(f"  sessions rail >10%  : {agg['sessions_file_rail_gt10']} "
          f"({_pct(agg['sessions_file_rail_gt10'], agg['sessions_used'])})")
    print(f"  sessions rail >50%  : {agg['sessions_file_rail_gt50']} "
          f"({_pct(agg['sessions_file_rail_gt50'], agg['sessions_used'])})")
    print()
    print('--- clip-level impact (of the clips that would be trained on) ---')
    c = agg['clips']
    print(f"  clips with any rail smp : {agg['clips_with_rail_gt0']:6d} "
          f"({_pct(agg['clips_with_rail_gt0'], c)})")
    print(f"  clips >10% railed       : {agg['clips_with_rail_gt10']:6d} "
          f"({_pct(agg['clips_with_rail_gt10'], c)})")
    print(f"  clips >50% railed       : {agg['clips_with_rail_gt50']:6d} "
          f"({_pct(agg['clips_with_rail_gt50'], c)})")
    print(f"  clips 100% railed       : {agg['clips_fully_railed']:6d} "
          f"({_pct(agg['clips_fully_railed'], c)})")
    flat_lbl = (f"clips flat (<{agg.get('flat_spread', DEFAULT_FLAT_SPREAD):g} "
                f"V)")
    print(f"  {flat_lbl:<24}: {agg['clips_flat_spread_lt_threshold']:6d} "
          f"({_pct(agg['clips_flat_spread_lt_threshold'], c)})  "
          f"[dropped by min_signal_spread]")
    print()
    print(f"--- the rail-touch rule: drop a clip whose window EVER TOUCHES the "
          f"rail, |x| >= {agg['rail_touch_v']:g} V ---")
    print(f"  clips                   : {c:6d}")
    print(f"  clips touching the rail : {agg['clips_touching_rail']:6d} "
          f"({_pct(agg['clips_touching_rail'], c)})"
          f"   [neg {agg['clips_touching_rail_neg']}, "
          f"pos {agg['clips_touching_rail_pos']}]")
    print(f"  clips dropped           : {agg['clips_dropped_by_touch_rule']:6d} "
          f"({_pct(agg['clips_dropped_by_touch_rule'], c)})")
    print(f"  clips remaining         : {agg['clips_kept_by_touch_rule']:6d}")
    print(f"  sessions emptied        : "
          f"{agg['sessions_emptied_by_touch_rule']:6d}  -> "
          f"{agg['sessions_used'] - agg['sessions_emptied_by_touch_rule']} "
          f"sessions remain")
    print()
    print(f"--- the spread guard: drop a clip whose window spread (max-min) < "
          f"{agg.get('flat_spread', DEFAULT_FLAT_SPREAD):g} V ---")
    print(f"  clips                   : {c:6d}")
    print(f"  dropped by the guard    : "
          f"{agg['clips_dropped_by_spread_rule']:6d} "
          f"({_pct(agg['clips_dropped_by_spread_rule'], c)} of ALL clips; "
          f"{_pct(agg['clips_dropped_by_spread_rule'], agg['clips_kept_by_touch_rule'])} "
          f"of those surviving the rail rule)")
    print(f"  clips remaining         : "
          f"{agg['clips_kept_after_both_rules']:6d}  (both rules)")
    print(f"  sessions emptied        : "
          f"{agg['sessions_emptied_by_spread_rule']:6d}  -> "
          f"{agg['sessions_used'] - agg['sessions_emptied_by_touch_rule'] - agg['sessions_emptied_by_spread_rule']} "
          f"sessions remain (both rules)")
    print()
    print('--- by distortion level ---')
    for name, d in agg['per_level'].items():
        mr = d['mean_file_rail_pct']
        mr_s = f'{mr:5.2f}%' if mr is not None else '  n/a'
        print(f"  {name:<9} sessions {d['sessions_used']:4d}/{d['sessions_discovered']:<4d}"
              f" clips {d['clips']:6d}  touched {d['clips_touching_rail']:6d}"
              f"  kept {d['clips_kept_by_touch_rule']:6d}"
              f"  then spread-dropped {d['clips_dropped_by_spread_rule']:5d}"
              f"  -> {d['clips_kept_after_both_rules']:6d}"
              f"  emptied {d['sessions_emptied_by_touch_rule']:3d}+"
              f"{d['sessions_emptied_by_spread_rule']:3d}"
              f"  mean_file_rail {mr_s}")
    print()
    print('--- per task ---')
    for t, d in agg['per_task'].items():
        print(f"  {t:<4} used {d['used']:4d}/{d['discovered']:<4d}"
              f" clips {d['clips']:6d}  clips_with_rail {d['clips_with_rail']:6d}")
    print()
    print('--- worst 15 sessions by file rail % ---')
    for r in worst[:15]:
        print(f"  {r['session']:<10} rail {r['resp_rail_pct']:5.1f}%  "
              f"clips {r['clips']:3d}  (with rail {r['clip_rail_gt0']}/"
              f">50% {r['clip_rail_gt50']})  resp_n {r['resp_n']}")
    print(L)
    _print_sweep(agg)


def _print_sweep(agg: dict) -> None:
    """The trade-off curve over the two per-window knobs (see ``--sweep``)."""
    L = '=' * 76
    if not agg.get('rail_touch_curve') and not agg.get('spread_curve'):
        return
    c = agg['clips']
    print(L)
    print('TRADE-OFF CURVE (classification thresholds; the corpus is built '
          'unfiltered)')
    print(L)
    if agg.get('rail_touch_curve'):
        print('--- rail_touch_v: drop a clip whose window holds any sample '
              'with abs(x) >= V ---')
        print(f"  {'V':>6} {'dropped':>8} {'share':>7} {'kept':>8} "
              f"{'sessions emptied':>17}")
        for k, d in agg['rail_touch_curve'].items():
            print(f"  {k:>6} {d['dropped']:8d} "
                  f"{_pct(d['dropped'], c):>7} {d['kept']:8d} "
                  f"{d['sessions_emptied']:17d}")
    if agg.get('spread_curve'):
        print()
        print(f"--- min_signal_spread: drop a clip whose window spread "
              f"(max-min) is BELOW V, ON TOP of rail_touch_v="
              f"{agg['rail_touch_v']:g} V ---")
        kept = agg['clips_kept_by_touch_rule']
        print(f"  {'< V':>6} {'dropped':>8} {'of kept':>8} {'kept':>8}")
        for k, d in agg['spread_curve'].items():
            print(f"  {k:>6} {d['dropped_from_kept']:8d} "
                  f"{_pct(d['dropped_from_kept'], kept):>8} "
                  f"{d['kept_after']:8d}")
    print(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Whole-corpus quality survey for the TIR-ROI + RESP pipeline.')
    p.add_argument('--raw_root', default=trd.default_raw_root())
    p.add_argument('--output_dir', default=None)
    p.add_argument('--subjects', default='', help='comma list (empty = all)')
    p.add_argument('--task_groups', default=DEFAULT_TASK_GROUPS)
    p.add_argument('--input_size', type=int, default=112)
    p.add_argument('--roi_padding', type=float, default=0.2)
    p.add_argument('--clip_seconds', type=float, default=DEFAULT_CLIP_SECONDS)
    p.add_argument('--clip_stride', type=float, default=DEFAULT_CLIP_STRIDE,
                   help='window hop in SECONDS. The survey MUST use the same '
                        'hop as the configs it is describing, else the clip '
                        'counts do not refer to the corpus the runs build: the '
                        'shipped TIR-ROI/RESP configs use 1.0 (an 8 s window '
                        'with 7/8 overlap), 0 = non-overlapping')
    p.add_argument('--rail_v', type=float, default=DEFAULT_RAIL_V)
    p.add_argument('--flat_spread', type=float, default=DEFAULT_FLAT_SPREAD,
                   help='VOLTS: count (and report) a clip whose window spread '
                        'is below this -- the min_signal_spread guard the '
                        'configs ship')
    p.add_argument('--rail_touch_v', type=float, default=DEFAULT_RAIL_TOUCH_V,
                   help='VOLTS (magnitude): the clip filter -- a clip whose '
                        'window contains any sample at/past this is counted as '
                        'dropped by the rule (the survey always builds '
                        'unfiltered) and the retained corpus is clips - that '
                        'count')
    p.add_argument('--rail_touch_sweep', default=DEFAULT_RAIL_TOUCH_SWEEP,
                   help='comma list of VOLTS: candidate rail thresholds for '
                        'the trade-off curve (empty = skip the sweep)')
    p.add_argument('--spread_sweep', default=DEFAULT_SPREAD_SWEEP,
                   help='comma list of VOLTS: candidate min_signal_spread '
                        'values for the trade-off curve, applied ON TOP of '
                        '--rail_touch_v')
    p.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    p.add_argument('--limit_subjects', type=int, default=0, help='0 = no cap')
    args = p.parse_args(argv)

    if not os.path.isdir(args.raw_root):
        raise SystemExit(f'--raw_root does not exist: {args.raw_root}')

    subjects = _subjects_from_arg(args.subjects) or _list_subjects(args.raw_root)
    if args.limit_subjects:
        subjects = subjects[:args.limit_subjects]

    task_groups = trd.tgrp.parse_task_groups(args.task_groups)
    payload = [{'subject': s, 'raw_root': args.raw_root,
                'task_groups': args.task_groups, 'input_size': args.input_size,
                'roi_padding': args.roi_padding, 'rail_v': args.rail_v,
                'clip_seconds': args.clip_seconds,
                'clip_stride': args.clip_stride,
                'flat_spread': args.flat_spread,
                'rail_touch_v': args.rail_touch_v,
                'rail_touch_sweep': _float_list(args.rail_touch_sweep),
                'spread_sweep': _float_list(args.spread_sweep)} for s in subjects]

    print(f'[survey] raw_root  : {args.raw_root}')
    print(f'[survey] subjects  : {len(subjects)}')
    print(f'[survey] workers   : {args.workers}')
    print(f'[survey] rail_v    : {args.rail_v}   flat_spread: {args.flat_spread}',
          flush=True)
    print(f'[survey] touch rule: |x| >= {args.rail_touch_v:g} V (the cleaning '
          f'rule; a clip is dropped when a window sample touches the rail)',
          flush=True)
    print(f'[survey] clip        : {args.clip_seconds:g} s window, '
          f'{args.clip_stride:g} s hop '
          f"({args.clip_seconds / args.clip_stride:g}x overlap)", flush=True)

    rows: List[dict] = []
    errors: List[dict] = []
    warnings: List[str] = []
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=max(1, args.workers)) as pool:
        for i, res in enumerate(pool.imap_unordered(_survey_subject, payload), 1):
            if res.get('error'):
                errors.append({'subject': res['subject'], 'error': res['error']})
                print(f'[survey] ERROR {res["subject"]}: {res["error"]}',
                      flush=True)
            rows.extend(res.get('sessions', []))
            warnings.extend(res.get('warnings', []))
            if i % 10 == 0 or i == len(payload):
                print(f'[survey] {i}/{len(payload)} subjects done', flush=True)

    rows.sort(key=lambda r: (r['subject'], r['task']))
    used = [r for r in rows if r['status'] == 'used']
    worst = sorted(used, key=lambda r: (r['resp_rail_pct'] or 0), reverse=True)
    agg = _aggregate(rows, task_groups, args.rail_touch_v,
                     _float_list(args.rail_touch_sweep),
                     _float_list(args.spread_sweep), args.flat_spread)

    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'tir_resp_survey.json'), 'w') as fh:
            json.dump({'args': vars(args), 'aggregate': agg, 'errors': errors,
                       'warnings': warnings, 'sessions': rows}, fh, indent=2)
        keys = list(rows[0].keys()) if rows else []
        with open(os.path.join(out_dir, 'tir_resp_survey_sessions.csv'),
                  'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'[survey] wrote {out_dir}/tir_resp_survey.json + '
              f'tir_resp_survey_sessions.csv', flush=True)

    _print_report(agg, worst, task_groups)
    if errors:
        print(f'[survey] {len(errors)} subject(s) errored: '
              f'{[e["subject"] for e in errors]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
