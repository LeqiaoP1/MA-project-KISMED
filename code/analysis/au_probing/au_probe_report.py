"""Parse an AU-probe run's logs into the report tables (CPU-only, no model).

The AU-occurrence probe writes everything it reports to STDOUT only -- there is
no report JSON in the run directory -- so the per-epoch validation curve and the
per-AU F1 table have to be recovered from ``train.log`` / the sbatch ``.out``.
This script does exactly that, and derives the one quantity the run does NOT
print: the per-AU PRESENCE RATE.

The presence rate is recoverable because the engine's trivial baseline is
``F1 = 2p / (1 + p)`` (:func:`engines.au_probe.trivial_f1`), which inverts
exactly::

    p = t / (2 - t)          with t = trivial_F1

That inversion is what turns "4 of 7 AUs scored 0.00" into a diagnosis: those
four are the four RAREST AUs, and probe-F1 tracks presence rate at r = +0.984 --
the signature of a head that has learned the per-AU marginal and little else.

Usage (from ``code/``)::

    python analysis/au_probing/au_probe_report.py \
        --log /work/scratch/$USER/finetune/au_tir_roi_hpc/train.log \
        --out analysis/au_probing/au_tir_roi_hpc.json
"""
import argparse
import json
import re
import sys

import numpy as np

#: ``[epoch 11] val macro-F1 33.73 (trivial 44.58, delta -10.85)  loss 0.4387``
_EPOCH = re.compile(
    r'\[epoch\s+(\d+)\]\s+val macro-F1\s+([\d.]+)\s+'
    r'\(trivial\s+([\d.]+),\s+delta\s+([-+\d.]+)\)\s+loss\s+([\d.]+)')
#: ``  AU10:  81.08   (trivial  78.41)``
_PER_AU = re.compile(r'^\s*AU(\d+):\s+([\d.]+)\s+\(trivial\s+([\d.]+)\)')
#: ``Best val macro-F1: 34.33``
_BEST = re.compile(r'Best val macro-F1:\s+([\d.]+)')


def presence_from_trivial(trivial_pct):
    """Exact inverse of the always-positive baseline ``F1 = 2p / (1 + p)``."""
    t = np.asarray(trivial_pct, dtype=float) / 100.0
    return t / (2.0 - t)


def parse(path):
    """Read a probe log -> ``{'epochs': [...], 'per_au': [...], 'best': ..}``."""
    epochs, per_au, best = [], [], None
    with open(path) as fh:
        for line in fh:
            m = _EPOCH.search(line)
            if m:
                epochs.append({'epoch': int(m.group(1)),
                               'f1': float(m.group(2)),
                               'trivial': float(m.group(3)),
                               'delta': float(m.group(4)),
                               'loss': float(m.group(5))})
                continue
            m = _PER_AU.match(line)
            if m:
                per_au.append({'au': int(m.group(1)),
                               'probe_f1': float(m.group(2)),
                               'trivial_f1': float(m.group(3))})
                continue
            m = _BEST.search(line)
            if m:
                best = float(m.group(1))
    return {'epochs': epochs, 'per_au': per_au, 'best_val_macro_f1': best}


def summarise(doc):
    """Add the derived per-AU columns and the run-level verdict."""
    for row in doc['per_au']:
        row['presence_rate'] = float(
            presence_from_trivial([row['trivial_f1']])[0])
        row['delta'] = row['probe_f1'] - row['trivial_f1']
        row['beats_trivial'] = bool(row['delta'] > 1e-9)
        # an F1 of exactly 0 with a non-zero presence rate means the head never
        # predicted positive for that AU -- an imbalance collapse, not a miss
        row['collapsed_all_negative'] = bool(
            row['probe_f1'] == 0.0 and row['presence_rate'] > 0.0)

    probe = np.array([r['probe_f1'] for r in doc['per_au']], dtype=float)
    triv = np.array([r['trivial_f1'] for r in doc['per_au']], dtype=float)
    pres = np.array([r['presence_rate'] for r in doc['per_au']], dtype=float)

    n_beat = int((probe > triv).sum())
    doc['summary'] = {
        'n_au': len(probe),
        'macro_f1_probe': float(probe.mean()),
        'macro_f1_trivial': float(triv.mean()),
        'macro_f1_delta': float((probe - triv).mean()),
        'n_au_beating_trivial': n_beat,
        'majority_beats_trivial': bool(n_beat * 2 > len(probe)),
        'n_au_collapsed': int((probe == 0.0).sum()),
        'corr_presence_vs_f1': float(np.corrcoef(pres, probe)[0, 1]),
    }
    return doc


def print_report(doc, name='probe'):
    print(f'### {name}\n')
    ep = doc['epochs']
    if ep:
        f1 = np.array([e['f1'] for e in ep], dtype=float)
        print(f'val macro-F1 over {len(ep)} epochs: '
              f'first {f1[0]:.2f}, best {f1.max():.2f} '
              f'(epoch {int(np.argmax(f1))}), last {f1[-1]:.2f}')
        print(f'  trivial baseline (constant): {ep[0]["trivial"]:.2f}')
        print(f'  best-vs-last drift: {f1[-1] - f1.max():+.2f} '
              f'-> {"no trend" if abs(f1[-1] - f1.max()) < 2 else "check"}\n')

    if doc['per_au']:
        print('| AU | probe F1 | trivial F1 | delta | presence p | verdict |')
        print('|---|---|---|---|---|---|')
        for r in doc['per_au']:
            verdict = ('collapsed (all-negative)' if r['collapsed_all_negative']
                       else ('beats trivial' if r['beats_trivial']
                             else 'below trivial'))
            print(f"| {r['au']} | {r['probe_f1']:.2f} | {r['trivial_f1']:.2f} | "
                  f"{r['delta']:+.2f} | {r['presence_rate']:.3f} | {verdict} |")
        s = doc['summary']
        print(f"| **macro** | **{s['macro_f1_probe']:.2f}** | "
              f"**{s['macro_f1_trivial']:.2f}** | "
              f"**{s['macro_f1_delta']:+.2f}** | | |")
        print()
        print(f"- AUs beating trivial: **{s['n_au_beating_trivial']}/"
              f"{s['n_au']}** -> majority needed: "
              f"**{'YES' if s['majority_beats_trivial'] else 'NO'}**")
        print(f"- AUs collapsed to all-negative: **{s['n_au_collapsed']}**")
        print(f"- corr(presence rate, probe F1): "
              f"**{s['corr_presence_vs_f1']:+.3f}**")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--log', required=True,
                   help='a run dir train.log OR a logs/*_<jobid>.out')
    p.add_argument('--name', default='', help='label for the printed report')
    p.add_argument('--out', default='', help='write the parsed tables as JSON')
    a = p.parse_args(argv)

    doc = summarise(parse(a.log))
    doc['source'] = a.log
    print_report(doc, a.name or a.log)
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(doc, fh, indent=2)
        print(f'\nwrote {a.out}')
    if not doc['epochs'] and not doc['per_au']:
        print('ERROR: parsed nothing -- is this the right log?', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
