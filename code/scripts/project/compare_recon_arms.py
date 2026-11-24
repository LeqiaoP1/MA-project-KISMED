"""Paired comparison of two ``run_eval_recon.py`` reports.

Both reports must come from the same checkpoint population and the same
``--seed``: rows are joined on ``(subject, session, t_start)``, which is the clip
identity. The two arms differ only in the masking allocation, so the resulting
difference is a within-clip contrast, not a between-corpus one.

Outputs a small JSON (per-clip deltas + summary) and prints a summary. It does not
alter either input.

Usage (from ``code/``)::

    python -m scripts.project.compare_recon_arms \
        --a notes/eval/baselines/tir_rgb2d_resp/artifacts/armA_tir_max_resp_hidden.json \
        --b notes/eval/baselines/tir_rgb2d_resp/artifacts/armB_rgb2d_max_resp_hidden.json \
        --out notes/eval/baselines/tir_rgb2d_resp/artifacts/armA_vs_armB.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if CODE not in sys.path:
    sys.path.insert(0, CODE)


def _key(row):
    return (str(row.get('subject', '')), str(row.get('session', '')),
            round(float(row.get('t_start', float('nan'))), 6))


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    rows = {}
    for r in d.get('per_clip', []):
        k = _key(r)
        if k in rows:
            raise SystemExit(f'{path}: duplicate clip key {k}')
        rows[k] = r
    if not rows:
        raise SystemExit(f'{path}: no per_clip rows')
    return d, rows


def filter_subjects(rows, wanted):
    if not wanted:
        return rows
    keep = {k: r for k, r in rows.items() if k[0] in wanted}
    if not keep:
        raise SystemExit(f'[cmp] --subjects {sorted(wanted)} matches no clip')
    return keep


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--out', default='')
    p.add_argument('--metric', default='r_masked')
    p.add_argument('--subjects', default='',
                   help='comma-separated subject ids to restrict BOTH arms to')
    return p.parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    da, ra = load(a.a)
    db, rb = load(a.b)
    wanted = {x.strip() for x in str(a.subjects).split(',') if x.strip()}
    ra = filter_subjects(ra, wanted)
    rb = filter_subjects(rb, wanted)
    common = sorted(set(ra) & set(rb))
    only_a = set(ra) - set(rb)
    only_b = set(rb) - set(ra)
    print(f'[cmp] A {os.path.basename(a.a)}: {len(ra)} clips')
    print(f'[cmp] B {os.path.basename(a.b)}: {len(rb)} clips')
    print(f'[cmp] common {len(common)}  only-A {len(only_a)}  only-B {len(only_b)}')
    if len(common) < 2:
        raise SystemExit('[cmp] fewer than two paired clips')

    va = np.asarray([ra[k][a.metric] for k in common], dtype=np.float64)
    vb = np.asarray([rb[k][a.metric] for k in common], dtype=np.float64)
    ok = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[ok], vb[ok]
    d = vb - va
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float('nan')

    def arm(v):
        return {
            'mean': float(v.mean()),
            'median': float(np.median(v)),
            'sd': float(v.std(ddof=0)),
            'n': int(v.size),
        }

    # per-subject paired delta (median of clip deltas) for a sanity read
    per_subject = {}
    for k, delta in zip([k for k, keep in zip(common, ok) if keep], d):
        per_subject.setdefault(k[0], []).append(float(delta))
    per_subject = {s: {'n': len(v), 'median_delta': float(np.median(v))}
                   for s, v in sorted(per_subject.items())}

    payload = {
        'schema': 'recon/paired-arms/1',
        'subjects_filter': sorted(wanted) or None,
        'a': {'path': os.path.abspath(a.a), 'label': da.get('label'),
              **arm(va)},
        'b': {'path': os.path.abspath(a.b), 'label': db.get('label'),
              **arm(vb)},
        'metric': a.metric,
        'paired': {
            'n': int(n),
            'mean_delta': mean,
            'sd_delta': sd,
            'se_delta': se,
            't': float(t),
            'delta_positive': int((d > 0).sum()),
            'delta_negative': int((d < 0).sum()),
            'delta_zero': int((d == 0).sum()),
        },
        'per_subject_median_delta': per_subject,
        'only_in_a': [list(k) for k in sorted(only_a)],
        'only_in_b': [list(k) for k in sorted(only_b)],
    }
    print(f'[cmp] A {a.metric}: mean {va.mean():+.4f}  median {np.median(va):+.4f}')
    print(f'[cmp] B {a.metric}: mean {vb.mean():+.4f}  median {np.median(vb):+.4f}')
    print(f'[cmp] paired delta (B - A): {mean:+.4f} ± {se:.4f} (SE), '
          f't = {t:+.2f}, n = {n}; +/{int((d > 0).sum())} '
          f'-/{int((d < 0).sum())} 0/{int((d == 0).sum())}')
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as fh:
            json.dump(payload, fh, indent=2)
        print('[cmp] wrote', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
