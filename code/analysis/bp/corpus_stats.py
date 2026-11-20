"""Re-measure the ADR-004 §1 corpus statistics from the RAW traces.

Provenance probe, not part of the build. It reads ``BP_mmHg.txt`` directly,
writes nothing into the artefact, and exists so that the numbers the ADR quotes
about the corpus are re-measured on demand instead of inherited from an earlier
draft.

Reported, per corpus and per session:

* sessions, total samples, hours at the nominal 1000 Hz;
* sessions with any sample below 30 mmHg, with any negative sample, with any
  sample above 250 mmHg, and the global min / max;
* the longest contiguous stretch below 30 mmHg (a sensor-off dropout, in
  seconds) and the longest run of exactly equal consecutive samples (ms);
* how many sessions have NO sample at or above 30 mmHg (those must be the
  artefact's zero-segment shards).

Usage (CPU only, no GPU, no torch)::

    python analysis/bp/corpus_stats.py --raw_root /work/projects/l0003511/test
    python analysis/bp/corpus_stats.py --raw_root ... --json out.json --limit 20
"""
import argparse
import json
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(CODE_DIR, 'data')
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

import numpy as np                                              # noqa: E402

import bp_artefact as bpa                                       # noqa: E402

FS = 1000.0
LOW_MMHG = 30.0
HIGH_MMHG = 250.0


def _runs_where(cond) -> tuple:
    """Longest run of ``True`` in a boolean array → ``(length, start)``."""
    if not cond.any():
        return 0, -1
    padded = np.concatenate(([False], cond, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    lengths = stops - starts
    best = int(np.argmax(lengths))
    return int(lengths[best]), int(starts[best])


def _longest_equal_run(x: np.ndarray) -> int:
    """Longest run of exactly-equal consecutive samples (in samples)."""
    n = x.size
    if n < 2:
        return n
    run = np.ones(n, dtype=np.int32)
    # vectorised: reset the counter at every value change
    change = np.flatnonzero(x[1:] != x[:-1]) + 1
    if change.size == 0:
        return n
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    return int((stops - starts).max())


def measure(spec: dict) -> dict:
    src = spec.get('source_file')
    row = {'session': spec['session'], 'readable': bool(src), 'n_src': 0}
    if not src:
        return row
    x = bpa._load_bp_mmhg(src)
    finite = np.isfinite(x)
    xf = np.where(finite, x, 0.0)
    below = finite & (xf < LOW_MMHG)
    longest_below, _ = _runs_where(below)
    lo = float(xf[finite].min()) if finite.any() else float('nan')
    hi = float(xf[finite].max()) if finite.any() else float('nan')
    # Gate 3's input, measured on the RAW extrema: how long does the trace hold
    # its own maximum or minimum? A real pulse repeats an exact extremum for a
    # handful of samples, so this bounds how often a 30 ms rail rule can fire.
    at_extremum = finite & ((xf == hi) | (xf == lo))
    longest_dwell, _ = _runs_where(at_extremum)
    row.update({
        'n_src': int(x.size),
        'n_nonfinite': int((~finite).sum()),
        'min': lo,
        'max': hi,
        'n_below_low': int(below.sum()),
        'n_negative': int((finite & (xf < 0.0)).sum()),
        'n_above_high': int((finite & (xf > HIGH_MMHG)).sum()),
        'longest_below_s': longest_below / FS,
        'longest_equal_run_ms': 1000.0 * _longest_equal_run(x) / FS,
        'longest_dwell_at_extremum_ms': 1000.0 * longest_dwell / FS,
    })
    return row


def summarise(rows) -> dict:
    ok = [r for r in rows if r['readable'] and r['n_src']]
    n = len(ok)
    total = int(sum(r['n_src'] for r in ok))
    out = {
        'sessions_discovered': len(rows),
        'sessions_readable': n,
        'sessions_unreadable': len(rows) - n,
        'total_samples': total,
        'hours_at_nominal_fs': total / FS / 3600.0,
        'sessions_with_sample_below_30': sum(1 for r in ok if r['n_below_low'] > 0),
        'sessions_with_negative_sample': sum(1 for r in ok if r['n_negative'] > 0),
        'sessions_with_sample_above_250': sum(1 for r in ok if r['n_above_high'] > 0),
        'sessions_with_no_sample_at_or_above_30': sum(
            1 for r in ok if r['n_below_low'] == r['n_src']),
        'global_min_mmhg': min(r['min'] for r in ok),
        'global_max_mmhg': max(r['max'] for r in ok),
        'longest_below_30_s': max(r['longest_below_s'] for r in ok),
        'longest_equal_run_ms': max(r['longest_equal_run_ms'] for r in ok),
        'longest_dwell_at_extremum_ms': max(r['longest_dwell_at_extremum_ms']
                                            for r in ok),
        'nonfinite_samples': int(sum(r['n_nonfinite'] for r in ok)),
    }
    longest = max(ok, key=lambda r: r['longest_below_s'])
    out['longest_below_30_session'] = longest['session']
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root', required=True)
    p.add_argument('--subjects', default='')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--json', default='', help='write the per-session rows + summary')
    args = p.parse_args(argv)

    subjects = [s for s in args.subjects.replace(',', ' ').split() if s]
    sessions = bpa.discover_sessions(args.raw_root, subjects or None)
    if args.limit:
        sessions = sessions[: args.limit]

    rows = []
    for i, spec in enumerate(sessions, 1):
        rows.append(measure(spec))
        if i % 200 == 0:
            print(f'  ... {i}/{len(sessions)}', flush=True)

    s = summarise(rows)
    print(f'corpus: {args.raw_root}')
    for k, v in s.items():
        print(f'  {k:38s} {v}')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'corpus': args.raw_root, 'summary': s, 'sessions': rows},
                      fh, indent=2)
        print(f'  -> {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
