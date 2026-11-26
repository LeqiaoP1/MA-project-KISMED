"""Measure the respiration source rate per session — PROVENANCE ONLY.

Why this exists. ADR-003 §2.3 keeps ``fs_in = 1000 Hz`` **nominal** on purpose: the
frame-to-sample mapping between respiration and thermal time is a corpus-wide
convention defined against that rate, so correcting it would redefine the mapping
for every session. The ADR therefore excludes the *measured* rate from the
artefact identity and from every shard. But it also asks for the corpus spread to
be published so the decision can be reopened if it ever matters — and that is
exactly what this script produces, in an eval note, never in the artefact.

What it does NOT do: it never writes a shard, is never called by
``runners/run_build_resp_cache.py``, and records nothing a consumer could act on.
Its output is a CSV plus a spread summary.

Measurement. ``f_meas = N_src / duration_s`` per session, where

* ``N_src`` is the number of finite-or-not values in ``Resp_Volts.txt`` (the same
  count ``data.resp_cache`` reads), and
* ``duration_s`` is the session's paired thermal timeline, ``n_frames / fps``,
  read from the video's metadata (no pixel decode).

**Result on the shipped corpus (2026-11-20): this duration source does NOT
work.** The ratio is bimodal — ~75 % of the 1384 measurable sessions imply
~390 Hz and ~23 % imply ~1000 Hz, because the thermal video covers a window
~2.57x longer than the physiology for most sessions. So the script reports the
number and its source, and the ADR's reopening tolerance stays undefined. What
arbitrates the convention is physiology, not this ratio: the dominant spectral
peak of the trace sits at a median 14.6 bpm when read at the nominal 1000 Hz
(100 % of sampled sessions inside 8-30 bpm) and at 5.7 bpm when read at 390 Hz
(10.5 %). See ``notes/eval/dataset_inspection/resp/resp_artefact_build.md``. If a real
duration of record is ever found, pass it with ``--duration_csv``.

The duration source is a real choice, so it is explicit: pass ``--duration_csv``
(columns ``session,duration_s``) when the timeline is already known, otherwise the
thermal frame count is probed. A session whose duration cannot be established is
reported with ``nan`` and counted separately — never silently dropped.

    python analysis/tir_resp/measure_resp_rate.py \\
        --raw_root "$RAW_DATA_PATH" \\
        --out code/notes/eval/dataset_inspection/resp/resp_rate_spread.csv

CPU only. No torch, no GPU, no YAML.
"""
import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

FS_IN = 1000.0          # nominal, the ADR's corpus constant (§2.3)
DEFAULT_FPS = 25.0      # thermal nominal; override with --fps


def count_values(path: str) -> int:
    """Count the values in a one-value-per-line trace (same rule the cache uses)."""
    n = 0
    with open(path, 'r') as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def probe_duration(video: str) -> Optional[float]:
    """Session duration in seconds from the video's METADATA, or ``None``.

    Prefers the reader's ``duration`` (``num_frames / fps`` with the FILE's own
    fps, which is what the paired timeline actually is); falls back to
    ``num_frames`` so the caller can divide by the nominal rate. Deliberately
    never decodes pixels: this script must stay cheap enough to run over the whole
    corpus on a login node.
    """
    try:
        from data import video_io as vio
    except ImportError:                     # pragma: no cover - direct import
        return None
    reader = None
    try:
        reader = vio.open_video(video)
        dur = getattr(reader, 'duration', None)
        if dur:
            return float(dur)
        n = getattr(reader, 'num_frames', None) or getattr(reader, 'n_frames', None)
        fps = getattr(reader, 'fps', None)
        if n and fps:
            return float(n) / float(fps)
        return None
    except Exception:
        return None
    finally:
        close = getattr(reader, 'close', None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def load_durations(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[str(row['session']).strip()] = float(row['duration_s'])
    return out


def measure(raw_root: str, fps: float, durations: Dict[str, float],
            subjects: Optional[Sequence[str]] = None,
            limit: int = 0) -> List[Dict]:
    from data import tir_resp_dataset as trd

    sessions = trd.discover_sessions(raw_root, subjects=subjects)
    if limit:
        sessions = sessions[:limit]
    rows: List[Dict] = []
    for spec in sessions:
        name = str(spec['session'])
        resp_file = spec.get('resp_file')
        row = {'session': name, 'n_src': 0, 'duration_s': float('nan'),
               'f_meas_hz': float('nan'), 'duration_source': 'unavailable'}
        if not resp_file or not os.path.isfile(str(resp_file)):
            row['duration_source'] = 'missing_resp_volts'
            rows.append(row)
            continue
        row['n_src'] = count_values(str(resp_file))
        dur = durations.get(name)
        if dur:
            row['duration_s'] = float(dur)
            row['duration_source'] = 'csv'
        elif spec.get('ir_file'):
            dur = probe_duration(str(spec['ir_file']))
            if dur and dur > 0:
                row['duration_s'] = float(dur)
                row['duration_source'] = 'thermal_metadata'
        if np.isfinite(row['duration_s']) and row['duration_s'] > 0:
            row['f_meas_hz'] = row['n_src'] / row['duration_s']
        rows.append(row)
    return rows


def summarise(rows: Sequence[Dict]) -> Dict[str, float]:
    vals = np.asarray([r['f_meas_hz'] for r in rows
                       if np.isfinite(r['f_meas_hz'])], dtype=float)
    if vals.size == 0:
        return {'n': 0}
    q1, q3 = np.percentile(vals, [25, 75])
    return {
        'n': int(vals.size),
        'min': float(vals.min()),
        'median': float(np.median(vals)),
        'max': float(vals.max()),
        'iqr': float(q3 - q1),
        'rel_spread_vs_nominal': float((vals.max() - vals.min()) / FS_IN),
        'mean': float(vals.mean()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root', default=os.environ.get('RAW_DATA_PATH', ''))
    p.add_argument('--out', default='code/notes/eval/dataset_inspection/resp/'
                                   'resp_rate_spread.csv',
                   help='CSV destination (repo-root relative by default); the '
                        'artefact is modality-independent, so the evidence is '
                        'filed under eval/dataset_inspection/resp, not under a visual lineage')
    p.add_argument('--fps', type=float, default=DEFAULT_FPS,
                   help='nominal thermal frame rate for the timeline')
    p.add_argument('--duration_csv', default='',
                   help='optional session,duration_s table (skips video probing)')
    p.add_argument('--subjects', default='')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args(argv)

    if not args.raw_root or not os.path.isdir(args.raw_root):
        print(f'[resp_rate] raw_root not usable: {args.raw_root!r}',
              file=sys.stderr)
        return 2

    subjects = [s.strip() for s in args.subjects.split(',') if s.strip()] or None
    durations = load_durations(args.duration_csv) if args.duration_csv else {}
    rows = measure(args.raw_root, args.fps, durations, subjects, args.limit)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['session', 'n_src', 'duration_s',
                                           'f_meas_hz', 'duration_source'])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v)
                        for k, v in r.items()})

    s = summarise(rows)
    print(f'[resp_rate] sessions       : {len(rows)}')
    print(f'[resp_rate] measured       : {s["n"]}')
    if s['n']:
        print(f'[resp_rate] f_meas Hz      : min {s["min"]:.3f}  median '
              f'{s["median"]:.3f}  max {s["max"]:.3f}  IQR {s["iqr"]:.3f}')
        print(f'[resp_rate] spread vs 1000 : '
              f'{100.0 * s["rel_spread_vs_nominal"]:.3f} % (max-min)')
    unres = [r for r in rows if not np.isfinite(r['f_meas_hz'])]
    if unres:
        print(f'[resp_rate] unmeasured     : {len(unres)} (e.g. '
              f'{", ".join(r["session"] for r in unres[:5])})')
    print(f'[resp_rate] CSV            : {args.out}')
    print('[resp_rate] NOTE: this number is PROVENANCE ONLY. It never enters the '
          'artefact, its key or a shard (ADR §2.3); publish it in the eval note '
          'so the reopening tolerance can be fixed later.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
