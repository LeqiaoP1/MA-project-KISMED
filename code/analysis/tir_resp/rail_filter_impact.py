"""Cost of the rail-touch clip filter, split by the CAUSE of the rail.

Joins two artifacts that measure different things:

* ``run_survey_tir_resp.py --output_dir D`` -> ``D/tir_resp_survey.json``: the
  per-session clip yield and, per clip, whether it TOUCHES the rail (the one
  cleaning rule);
* ``rail_forensics.py --json F``: the per-session rail structure and the
  ``dead_channel`` CLASSIFICATION of the file (a pinned/dead channel vs a
  clipped-but-breathing one).

The point of the join is that "how many clips does the rule drop?" is not the
question that matters -- "how many of them were DEAD-CHANNEL clips and how many
were GENUINE clipped troughs?" is. This is a report, not a filter: nothing here
is applied to the data (the only data filter is the dataset's ``rail_touch_v``).

Usage (from ``code/``)::

    python analysis/tir_resp/rail_filter_impact.py \
        --survey    $WORK_SCRATCH/tir_resp_survey_1s/tir_resp_survey.json \
        --forensics $WORK_SCRATCH/rail_forensics_all.json
    ... --json /tmp/rail_filter_impact.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rail_forensics as rf                             # noqa: E402


def _used(row: dict) -> bool:
    return row.get('status') == 'used'


def impact(survey: dict, forensics: List[dict]) -> Dict[str, object]:
    """Score the rule's drop against the dead-channel / clipped split."""
    rows = [r for r in survey['sessions'] if _used(r)]
    n_clips = sum(r['clips'] for r in rows)
    dropped = sum(r['clip_touched'] or 0 for r in rows)
    emptied = sum(1 for r in rows
                  if r['clips'] and (r['clip_touched'] or 0) == r['clips'])

    by_class: Dict[str, Dict[str, int]] = {
        'dead': {'sessions': 0, 'sessions_in_survey': 0, 'clips': 0,
                 'dropped': 0, 'sessions_emptied': 0},
        'clipped': {'sessions': 0, 'sessions_in_survey': 0, 'clips': 0,
                    'dropped': 0, 'sessions_emptied': 0},
    }
    seen = set()
    for f in forensics:
        cls = 'dead' if rf.dead_channel(f) else 'clipped'
        d = by_class[cls]
        d['sessions'] += 1
        row = next((r for r in rows if r['session'] == f['session']), None)
        if row is None:                    # a session the dataset skipped
            continue
        seen.add(f['session'])
        d['sessions_in_survey'] += 1
        d['clips'] += row['clips']
        d['dropped'] += row['clip_touched'] or 0
        if row['clips'] and (row['clip_touched'] or 0) == row['clips']:
            d['sessions_emptied'] += 1

    return {
        'clips': n_clips,
        'sessions': len(rows),
        'forensics_sessions': len(forensics),
        'forensics_in_survey': len(seen),
        'dropped': dropped,
        'kept': n_clips - dropped,
        'sessions_emptied': emptied,
        'sessions_remaining': len(rows) - emptied,
        'dropped_from_dead': by_class['dead']['dropped'],
        'dropped_from_clipped': by_class['clipped']['dropped'],
        'by_class': by_class,
    }


def _print(imp: Dict[str, object]) -> None:
    L = '=' * 76
    print(L)
    print('RAIL FILTER IMPACT -- the drop, split by the cause of the rail')
    print(L)
    print(f"corpus          : {imp['sessions']} usable session(s), "
          f"{imp['clips']} clip(s)")
    print(f"forensics       : {imp['forensics_sessions']} session(s) with a "
          f"rail, {imp['forensics_in_survey']} of them usable")
    for cls, label in (('dead', 'dead channel'), ('clipped', 'clipped trough')):
        d = imp['by_class'][cls]
        print(f"  {label:<14}: {d['sessions']:3d} session(s), "
              f"{d['clips']:6d} clip(s) in the usable corpus")
    print()
    print(f"clips touching the rail : {imp['dropped']:6d} "
          f"({100.0 * imp['dropped'] / imp['clips']:.1f} %)")
    print(f"  from dead channels    : {imp['dropped_from_dead']:6d} "
          f"({100.0 * imp['dropped_from_dead'] / imp['dropped']:.1f} % of the "
          f"drop)")
    print(f"  from clipped troughs  : {imp['dropped_from_clipped']:6d} "
          f"({100.0 * imp['dropped_from_clipped'] / imp['dropped']:.1f} %)")
    print()
    print(f"clips kept              : {imp['kept']:6d}")
    print(f"sessions emptied        : {imp['sessions_emptied']:6d}  -> "
          f"{imp['sessions_remaining']} session(s) remain")
    print()
    print('per class:')
    for cls, label in (('dead', 'dead channel'), ('clipped', 'clipped trough')):
        d = imp['by_class'][cls]
        print(f"  {label:<14} sessions {d['sessions_in_survey']:4d}  clips "
              f"{d['clips']:6d}  dropped {d['dropped']:6d}  "
              f"kept {d['clips'] - d['dropped']:6d}  "
              f"sessions fully excluded {d['sessions_emptied']:3d}")
    print(L)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--survey', required=True,
                   help='tir_resp_survey.json from run_survey_tir_resp.py')
    p.add_argument('--forensics', required=True,
                   help='JSON from analysis/tir_resp/rail_forensics.py --json')
    p.add_argument('--json', default=None, help='write the table as JSON')
    args = p.parse_args(argv)

    with open(args.survey) as fh:
        survey = json.load(fh)
    with open(args.forensics) as fh:
        forensics = json.load(fh)
    imp = impact(survey, forensics)
    _print(imp)
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(imp, fh, indent=2)
        print(f'wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
