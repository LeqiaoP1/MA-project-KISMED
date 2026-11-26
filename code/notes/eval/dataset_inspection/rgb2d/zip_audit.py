#!/usr/bin/env python
"""Audit every ``2D+3D/*.zip``: which archives cannot be read, and why.

A corpus can be delivered **while it is being read**. Two failure modes look
identical to ``zipfile`` but mean different things, and only the file SIZE tells
them apart:

* ``0 bytes``                                  -> placeholder, no source yet
* ``> 0 bytes`` and no end-of-central-directory -> the copy is still running
* readable but already has the EOCD             -> complete

Run this before a long cache build; it takes ~30 s for 140 archives because
``is_zipfile`` only reads the end record.

    cd code
    ../.env/bin/python notes/eval/dataset_inspection/rgb2d/zip_audit.py --raw_root "$RAW_DATA_PATH"
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw_root',
                   default=os.environ.get('RAW_DATA_PATH', '/work/projects/l0003511/test'))
    p.add_argument('--tree', default='2D+3D')
    args = p.parse_args()
    root = os.path.join(args.raw_root, args.tree)
    names = sorted(n for n in os.listdir(root) if n.lower().endswith('.zip'))
    print(f'{len(names)} archives under {root}\n')

    ok, bad, no_jpg, grew = 0, [], [], []
    for name in names:
        path = os.path.join(root, name)
        s0 = os.path.getsize(path)
        if s0 == 0:
            bad.append((name, s0, 'placeholder: 0 bytes, no source', 0))
            continue
        if not zipfile.is_zipfile(path):
            s1 = os.path.getsize(path)
            reason = ('TRUNCATED / still being copied' if s1 != s0
                      else 'no end-of-central-directory (partial or corrupt)')
            bad.append((name, s0, reason, s1 - s0))
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                members = zf.namelist()
            if not any(m.lower().endswith('.jpg') for m in members):
                no_jpg.append((name, s0, len(members)))
            ok += 1
        except Exception as exc:                       # corrupt central dir
            bad.append((name, s0, f'{type(exc).__name__}: {exc}', 0))
        s1 = os.path.getsize(path)
        if s1 != s0:
            grew.append((name, s0, s1))

    print(f'readable: {ok}   bad: {len(bad)}   readable-but-no-jpgs: {len(no_jpg)}')
    if bad:
        print('\nBAD ARCHIVES')
        print(f'{"name":16s} {"size(GB)":>10s} {"grew(B)":>10s}  reason')
        for n, s, r, d in bad:
            print(f'{n:16s} {s / 1e9:10.2f} {d:10d}  {r}')
    if no_jpg:
        print('\nREADABLE BUT WITHOUT JPEGs')
        for n, s, k in no_jpg:
            print(f'{n:16s} {s / 1e9:10.2f} GB  {k} members')
    if grew:
        print('\nSTILL GROWING (copy in progress during this scan)')
        for n, a, b in grew:
            print(f'{n:16s} {a / 1e9:.2f} -> {b / 1e9:.2f} GB')
    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
