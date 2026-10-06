"""Rail forensics: is a `-10 V` respiration rail a DEAD CHANNEL or a genuine
clipped extreme?

The corpus' ``Resp_Volts.txt`` dead channel reads exactly ``-10.0 V``. Two very
different causes look identical at the sample level, so this script measures the
STRUCTURE of the rail to separate them:

* **dead / pinned channel** (disconnected belt / saturated amplifier / stuck
  DAQ) -- long contiguous rail runs (often the whole file, or tens of seconds),
  often railed from ``t = 0``, NOT phase-locked to the breathing cycle;
* **genuine clipping** (the task provoked an extreme the recorder cannot
  represent, so the trough is flat-topped)
  -- SHORT runs (~a fraction of a breath), once per breathing cycle, bounded by
  ramps that approach the rail, and phase-locked to the respiratory trough.

This is a DESCRIPTIVE script: it is what showed that the adopted ``rail_touch_v``
clip filter removes DEAD-CHANNEL clips and GENUINE clipped troughs alike (the
cost is reported by ``rail_filter_impact.py``), and nothing here filters data.

Metrics per session (respiration only, no video decode):

``rail_pct``          share of samples at/below ``RESP_RAIL_V``
``runs``              number of contiguous rail runs
``runs_per_min``      runs / minute
``median_run_s``      median run duration (s)
``longest_run_s``     longest run (s)
``rail_at_start``     the file begins inside a rail run
``median_entry_gap``  |last sample before a run + 10| (V) -- small => ramp-in
                      (clipping), large => abrupt jump (disconnect)
``nonrail_gap_v``     mean of the NON-railed samples minus the floor (V) -- a
                      PINNED baseline sits ~0 V above the floor even between
                      runs (failure), a breathing one sits several V above
``period_s``          dominant breathing period from the NON-railed part (s)
``phase_R``           circular concentration of the run midpoints at that
                      period (1 = perfectly phase-locked => clipping,
                      ~0 = uniformly distributed => failure)
``verdict``           heuristic label: railed / long-run / clip-like / unclear

Usage (from ``code/``)::

    python analysis/tir_resp/rail_forensics.py
    python analysis/tir_resp/rail_forensics.py --json /tmp/rail_forensics.json
    python analysis/tir_resp/rail_forensics.py --sessions F001_T10,F004_T3
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data import tir_resp_dataset as trd            # noqa: E402

FS = 1000.0                     # Resp_Volts nominal sample rate (Hz)
BREATH_BAND = (0.10, 0.60)      # plausible breathing band (Hz)

#: heuristic thresholds (documented so they can be argued with)
LONG_RUN_S = 5.0                # a run this long is not a single clipped trough
RAMP_MAX_V = 0.75               # |gap| below this counts as a ramp into the rail
PHASE_LOCK_R = 0.60             # circular concentration above this = locked

#: heuristic thresholds for the two-cause CLASSIFICATION below
#: (``dead_channel``): any ONE of them means the FILE is a dead/pinned channel
#: rather than a clipped-but-breathing one.
DEAD_RAIL_PCT = 90.0            # the whole file is pinned at the floor
DEAD_LONG_RUN_S = 5.0           # one rail run this long = a dropout, not a trough
DEAD_NONRAIL_GAP_V = 1.5        # the "breathing" part never leaves the floor


def _runs(mask: np.ndarray) -> List[tuple]:
    """Contiguous True runs -> [(start, end_exclusive), ...]."""
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _circular_R(phases: np.ndarray) -> float:
    if phases.size == 0:
        return float('nan')
    return float(abs(np.mean(np.exp(1j * phases))))


def _dominant_period(y: np.ndarray) -> float:
    """Breathing period (s) from the gap-filled signal; NaN when not estimable."""
    good = np.isfinite(y) & (y > trd.RESP_RAIL_V)
    if good.sum() < FS * 8:                     # need a few seconds of signal
        return float('nan')
    idx = np.flatnonzero(good)
    filled = np.interp(np.arange(y.size), idx, y[idx])
    from scipy import signal as sg
    f, pxx = sg.periodogram(filled - filled.mean(), fs=FS, window='hann')
    band = (f >= BREATH_BAND[0]) & (f <= BREATH_BAND[1])
    if not band.any() or not (pxx[band] > 0).any():
        return float('nan')
    return float(1.0 / f[band][int(np.argmax(pxx[band]))])


def analyse(path: str) -> Dict[str, object]:
    y = trd._load_1d(path).astype(np.float64)
    n = y.size
    rail = y <= trd.RESP_RAIL_V
    runs = _runs(rail)
    durations = np.array([(b - a) / FS for a, b in runs], dtype=float)
    rail_pct = 100.0 * float(rail.sum()) / n if n else 0.0
    # how far ABOVE the floor the non-railed part of the file actually sits: a
    # pinned baseline stays ~0 V above it (failure), a breathing one several V
    nonrail = y[np.isfinite(y) & ~rail]

    # gap to the rail at each run boundary (small => the signal ramped in)
    gaps = []
    for a, b in runs:
        if a > 0:
            gaps.append(abs(y[a - 1] - trd.RESP_RAIL_V))
        if b < n:
            gaps.append(abs(y[b] - trd.RESP_RAIL_V))
    med_gap = float(np.median(gaps)) if gaps else float('nan')

    # A CLIPPED trough is entered while DESCENDING and left while ASCENDING (the
    # flat top brackets the signal minimum). A stuck/drifted sensor enters and
    # leaves at an arbitrary phase, so the slopes are ~50/50.
    k = int(round(0.15 * FS))
    desc = asc = 0
    for a, b in runs:
        if a - k >= 0 and y[a - 1] < y[a - k]:
            desc += 1
        if b + k <= n and y[b + k - 1] > y[b]:
            asc += 1
    n_runs = len(runs)
    descend_in_frac = (desc / n_runs) if n_runs else float('nan')
    ascend_out_frac = (asc / n_runs) if n_runs else float('nan')

    # regular spacing of run STARTS => driven by the breathing cycle
    if n_runs >= 3:
        iv = np.diff(np.array([a / FS for a, _ in runs]))
        run_gap_cv = float(iv.std() / (iv.mean() + 1e-9))
    else:
        run_gap_cv = float('nan')

    period = _dominant_period(y)
    phase_R = float('nan')
    if runs and np.isfinite(period) and period > 0:
        centres = np.array([(a + b) / 2.0 / FS for a, b in runs])
        phases = 2.0 * np.pi * np.mod(centres, period) / period
        phase_R = _circular_R(phases)

    # verdict
    if not runs:
        verdict = 'clean'
    elif rail_pct > 90.0:
        verdict = 'railed'                       # whole file pinned => failure
    elif durations.max() >= LONG_RUN_S:
        verdict = 'long-run'                     # >= 5 s in one run => failure
    elif (np.isfinite(descend_in_frac) and descend_in_frac >= 0.7
          and np.isfinite(ascend_out_frac) and ascend_out_frac >= 0.7
          and np.isfinite(med_gap) and med_gap <= RAMP_MAX_V):
        verdict = 'clip-like'                    # brackets a trough => genuine
    else:
        verdict = 'unclear'

    return {
        'session': os.path.basename(os.path.dirname(os.path.dirname(path))),
        'samples': n,
        'rail_pct': rail_pct,
        'runs': len(runs),
        'runs_per_min': (len(runs) / (n / FS / 60.0)) if n else 0.0,
        'median_run_s': float(np.median(durations)) if durations.size else 0.0,
        'longest_run_s': float(durations.max()) if durations.size else 0.0,
        'rail_at_start': bool(rail[0]) if n else False,
        'median_entry_gap': med_gap,
        'nonrail_gap_v': (float(nonrail.mean() - trd.RESP_RAIL_V)
                          if nonrail.size else float('nan')),
        'descend_in_frac': descend_in_frac,
        'ascend_out_frac': ascend_out_frac,
        'run_gap_cv': run_gap_cv,
        'period_s': period,
        'phase_R': phase_R,
        'verdict': verdict,
    }


def dead_channel(row: Dict[str, object]) -> bool:
    """CLASSIFICATION (not a filter): does the FILE look like a dead/pinned
    channel rather than a clipped-but-breathing one?

    Any ONE of (a) ``rail_pct > 90 %`` -- the whole file is pinned, (b) a single
    rail run ``>= 5 s`` -- a dropout, not a clipped trough, (c) the non-railed
    samples still sit ``< 1.5 V`` above the floor -- a pinned baseline with
    occasional spikes. ``rail_filter_impact.py`` uses it to say how much of the
    clip filter's drop lands on dead channels vs on genuine clipped troughs.
    It is a DESCRIPTIVE label for a session: nothing in the pipeline applies it
    to the data (the only data filter is the per-window ``rail_touch_v``).
    """
    pct = float(row.get('rail_pct') or 0.0)
    longest = float(row.get('longest_run_s') or 0.0)
    gap = row.get('nonrail_gap_v')
    if pct > DEAD_RAIL_PCT or longest >= DEAD_LONG_RUN_S:
        return True
    return (gap is not None and np.isfinite(gap)
            and float(gap) < DEAD_NONRAIL_GAP_V)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--raw_root', default=trd.default_raw_root())
    p.add_argument('--sessions', default='',
                   help='comma list of sessions to analyse (empty = all)')
    p.add_argument('--min_rail_pct', type=float, default=0.0,
                   help='only report sessions at/above this rail %% (default 0 '
                        '= every session with at least one railed sample)')
    p.add_argument('--limit', type=int, default=0, help='0 = no cap')
    p.add_argument('--json', default=None, help='write the full table as JSON')
    args = p.parse_args(argv)

    wanted = {s.strip() for s in args.sessions.split(',') if s.strip()}
    sessions = trd.discover_sessions(args.raw_root)
    rows: List[dict] = []
    for spec in sessions:
        if wanted and spec['session'] not in wanted:
            continue
        if spec['resp_file'] is None:
            continue
        row = analyse(spec['resp_file'])
        row['session'] = spec['session']
        row['subject'] = spec['subject']
        row['task'] = spec['task']
        if row['rail_pct'] <= 0.0 or row['rail_pct'] < args.min_rail_pct:
            continue
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    rows.sort(key=lambda r: -r['rail_pct'])
    by_verdict: Dict[str, int] = {}
    for r in rows:
        by_verdict[r['verdict']] = by_verdict.get(r['verdict'], 0) + 1
    n_dead = sum(1 for r in rows if dead_channel(r))

    print('=' * 88)
    print('RESPIRATION RAIL FORENSICS -- dead channel vs genuine clipping')
    print('=' * 88)
    print(f'raw_root       : {args.raw_root}')
    print(f'sessions with rail : {len(rows)}')
    print(f'verdicts       : {by_verdict}')
    print(f'dead-channel   : {n_dead} (rail_pct > {DEAD_RAIL_PCT:g} % OR a '
          f'run >= {DEAD_LONG_RUN_S:g} s OR a non-rail baseline gap '
          f'< {DEAD_NONRAIL_GAP_V:g} V)  [classification of the FILE, not a '
          f'filter]')
    print(f'clipped        : {len(rows) - n_dead}')
    print()
    hdr = (f"{'session':<10} {'rail%':>6} {'runs':>5} {'/min':>6} "
           f"{'med s':>6} {'max s':>7} {'start':>5} {'gap V':>6} "
           f"{'desc':>5} {'asc':>5} {'cv':>5}  verdict")
    print(hdr)
    print('-' * len(hdr))
    for r in rows[:60]:
        print(f"{r['session']:<10} {r['rail_pct']:6.1f} {r['runs']:5d} "
              f"{r['runs_per_min']:6.1f} {r['median_run_s']:6.2f} "
              f"{r['longest_run_s']:7.2f} {str(r['rail_at_start']):>5} "
              f"{r['median_entry_gap']:6.2f} "
              f"{r['descend_in_frac']:5.2f} {r['ascend_out_frac']:5.2f} "
              f"{r['run_gap_cv']:5.2f}  {r['verdict']}")
    if len(rows) > 60:
        print(f'... {len(rows) - 60} more')
    print()
    # compact per-verdict listing of the failure-like ones
    for tag in ('railed', 'long-run'):
        sel = [r for r in rows if r['verdict'] == tag]
        if sel:
            print(f'{tag} ({len(sel)}): ' + ', '.join(r['session'] for r in sel))
    clip = [r for r in rows if r['verdict'] == 'clip-like']
    if clip:
        print(f"clip-like ({len(clip)}): " + ', '.join(r['session'] for r in clip))
    print('=' * 88)

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rows, fh, indent=2)
        print(f'wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
