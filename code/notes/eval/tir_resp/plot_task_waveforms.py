#!/usr/bin/env python3
"""Plot assembled Stage-3 respiration predictions for one BP4D+ task.

This is a PROBE script in the ``code/notes/eval/tir_resp/`` sense: it turns an
existing evaluation dump into a presentation artefact and adds no new metric.
It re-uses the production code path, so the curves it draws are the same curves
the k-fold report scores:

* clip rows -> session timeline: ``evaluation.assemble.assemble_session``
  (Hann overlap-add; ``nan`` outside the covered span, which is sliced off);
* reference: ``runners.run_evaluate_session.read_reference``
  (raw ``Resp_Volts.txt`` at ``raw_fs``, resampled to ``fs``);
* annotation: ``evaluation.metrics.band_limited_pearson`` on the covered span.

What it shows, and what it does NOT
-----------------------------------
Each panel is ONE subject of ONE task: predicted waveform over the reference.
Because ``physio_norm: zscore`` leaves every clip prediction defined only up to
a per-clip affine map, both signals are z-scored over the displayed window. The
figure compares SHAPE and PHASE only -- amplitude is a separate, oracle-
calibrated number (see ``resp_5fold_cv.md``).

"Representative" is defined, not hand-picked: subjects are ranked by their
``pearson_band`` for that task and four ranks are drawn at ``--quantiles``
(default the 85th, 55th, 30th and 10th percentile), i.e. a strong, a typical, a
weak and a poor subject. Use ``--subjects`` to override.

Usage (from ``code/``)::

    python notes/eval/tir_resp/plot_task_waveforms.py \\
        --root ../output/finetune --run_prefix resp_tir_roi_hpc_fold \\
        --folds 0,1,2,3,4 --task T3 \\
        --out notes/eval/tir_resp/resp_T3_waveforms.svg

Expected dump layout (what ``run_evaluate_kfold.py`` consumes):
``<root>/<prefix><k>/preds.npy`` ``[N_clips, output_len]`` plus
``<root>/<prefix><k>/entries.json`` (``session``, ``t_start``, ``task``,
``signals_file``), with ``preds`` row *i* matching ``entries`` row *i*.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# code/notes/eval/tir_resp/this_file.py -> code/
_CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CODE_DIR))

from evaluation.assemble import assemble_session, group_entries  # noqa: E402
from evaluation.metrics import band_limited_pearson  # noqa: E402
from runners.run_evaluate_session import (  # noqa: E402
    _task_of_name, read_reference)
from runners.run_evaluate_subject import subject_of  # noqa: E402

__all__ = ['collect', 'pick_representatives', 'write_svg', 'main']


# --------------------------------------------------------------------------- io
def _fold_dir(root: str, prefix: str, k: int) -> str:
    """Prefer ``<prefix><k>``; accept ``<prefix>_fold<k>`` too (kfold runner)."""
    for cand in (f'{prefix}{k}', f'{prefix}_fold{k}'):
        d = os.path.join(root, cand)
        if os.path.isdir(d):
            return d
    return os.path.join(root, f'{prefix}{k}')


def collect(root: str,
            prefix: str,
            folds: Sequence[int],
            task: str,
            fs: float,
            raw_fs: float,
            waveform: str,
            band: Tuple[float, float],
            max_seconds: float = 0.0,
            signals_root: str = '') -> Dict[str, dict]:
    """Assemble every session of ``task``; keep the longest one per subject.

    ``signals_root`` rewrites the reference path: ``entries.json`` stores
    ``signals_file`` as an absolute path from the host that produced the dump,
    which usually does not exist here. When given, the reference is looked up as
    ``<signals_root>/<session>/<basename>`` (the canonical dataset layout), so a
    locally mounted copy of ``bp4d_canonical`` lines up with a cluster dump.

    :returns: ``{subject: {session, fold, t, pred, ref, pearson_band, n_clips}}``
    """
    per_subject: Dict[str, dict] = {}
    task = str(task).upper()
    for k in folds:
        d = _fold_dir(root, prefix, k)
        pred_path = os.path.join(d, 'preds.npy')
        ent_path = os.path.join(d, 'entries.json')
        for p in (pred_path, ent_path):
            if not os.path.isfile(p):
                raise SystemExit(f'[waves] fold {k}: missing {p}')
        preds = np.atleast_2d(np.load(pred_path)).astype(float)
        with open(ent_path) as f:
            entries = json.load(f)
        if preds.shape[0] != len(entries):
            raise SystemExit(
                f'[waves] fold {k}: {preds.shape[0]} predictions vs '
                f'{len(entries)} entries -- this looks like a DDP shard, not a '
                f'single-process dump.')

        n_kept = 0
        for name, pairs in sorted(group_entries(entries, fs).items()):
            first = entries[pairs[0][0]]
            label = str(first.get('task') or _task_of_name(name)).upper()
            if label != task:
                continue
            sig = first.get('signals_file') or ''
            if signals_root:
                sig = os.path.join(signals_root, str(name),
                                   os.path.basename(sig) or 'signals.csv')
            if not sig or not os.path.isfile(sig):
                print(f'[waves] {name}: skip (no reference at {sig!r})')
                continue

            rows = [i for i, _ in pairs]
            offs = [o for _, o in pairs]
            t, y, cov = assemble_session(preds[rows], offs, fs)
            ref_full = read_reference(sig, waveform, fs=fs, raw_fs=raw_fs)
            n = int(min(y.size, ref_full.size))
            cov = cov[:n]
            if not cov.any():
                print(f'[waves] {name}: skip (nothing covered)')
                continue

            sel = np.flatnonzero(cov)
            if max_seconds and max_seconds > 0:
                sel = sel[:int(round(max_seconds * fs))]
            tt = t[:n][sel]
            yy = y[:n][sel]
            rr = ref_full[:n][sel]
            r = float(band_limited_pearson(np.atleast_2d(yy),
                                           np.atleast_2d(rr), fs, band))
            cand = dict(subject=subject_of(name) or name,
                        session=name,
                        fold=int(k),
                        t=tt,
                        pred=yy,
                        ref=rr,
                        pearson_band=r,
                        n_clips=len(rows))
            prev = per_subject.get(cand['subject'])
            if prev is None or tt.size > prev['t'].size:
                per_subject[cand['subject']] = cand
            n_kept += 1
        print(f'[waves] fold {k}: {n_kept} {task} session(s)')

    if not per_subject:
        raise SystemExit(
            f'[waves] no {task} session with a readable reference under {root}')
    return per_subject


# ------------------------------------------------------------------- selection
def pick_representatives(per_subject: Dict[str, dict],
                         quantiles: Sequence[float] = (0.85, 0.55, 0.30, 0.10),
                         subjects: Optional[Sequence[str]] = None) -> List[dict]:
    """Rank subjects by ``pearson_band``; take one per requested quantile.

    Quantile 1.0 is the BEST subject. A rank already taken is nudged to the
    nearest free rank, so four distinct panels are always drawn.
    """
    ranked = sorted(per_subject, key=lambda s: per_subject[s]['pearson_band'])
    n = len(ranked)

    if subjects:
        out: List[dict] = []
        for s in subjects:
            if s not in per_subject:
                raise SystemExit(f'[waves] subject {s!r} has no matching session '
                                 f'in this dump (subjects seen: {sorted(per_subject)})')
            rank = ranked.index(s)
            out.append(dict(per_subject[s], rank=rank, n_subjects=n,
                            pct=100.0 * rank / max(1, n - 1)))
        return out

    used: List[int] = []
    out2: List[dict] = []
    for q in quantiles:
        idx = int(round(float(q) * (n - 1)))
        step = 1
        while idx in used:
            down, up = idx - step, idx + step
            if down >= 0 and down not in used:
                idx = down
            elif up <= n - 1 and up not in used:
                idx = up
            else:                                   # pragma: no cover
                raise SystemExit('[waves] not enough subjects for the requested '
                                 'quantiles')
            step += 1
        used.append(idx)
        out2.append(dict(per_subject[ranked[idx]], rank=idx, n_subjects=n,
                         pct=100.0 * idx / max(1, n - 1)))
    return out2


# --------------------------------------------------------------------- drawing
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.zeros_like(x)
    mu, sd = float(np.mean(x[ok])), float(np.std(x[ok]))
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def _nice_ticks(span: float, target: int = 6) -> List[float]:
    """Second ticks at a round step (1/2/5/10/15/20/30/45/60...) covering span."""
    if span <= 0:
        return [0.0]
    raw = span / float(max(1, target))
    step = next((s for s in (1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120)
                 if s >= raw), max(1, int(round(raw))))
    ticks = [0.0]
    while ticks[-1] + step <= span + 1e-9:
        ticks.append(ticks[-1] + step)
    return ticks


def _poly(t: np.ndarray, z: np.ndarray, x0: float, y0: float, w: float,
          h: float, zmax: float, t0: float, span: float,
          max_points: int = 900) -> str:
    """``t``/``z`` -> SVG ``points`` string inside the box ``(x0, y0, w, h)``."""
    n = t.size
    if n < 2:
        return ''
    step = max(1, int(np.ceil(n / float(max_points))))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    xs = x0 + (t[idx] - t0) / span * w
    ys = y0 + h / 2.0 - np.clip(z[idx], -zmax, zmax) / zmax * (h / 2.0 - 4.0)
    ok = np.isfinite(xs) & np.isfinite(ys)
    return ' '.join(f'{x:.1f},{y:.1f}' for x, y in zip(xs[ok], ys[ok]))


def write_svg(panels: List[dict], task: str, band: Tuple[float, float],
              window_s: float, out_path: str) -> str:
    """Render the stacked panels and write ``out_path``; returns that path."""
    W, H = 900, 830
    left, right = 120.0, 860.0
    pw = right - left
    ph = 112.0
    zmax = 2.5
    top0 = 118.0
    gap = 160.0

    L: List[str] = []
    L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
             f' width="{W}" height="{H}" role="img"'
             ' aria-labelledby="figTitle figDesc"'
             ' font-family="Helvetica, Arial, sans-serif">')
    L.append(f'  <title id="figTitle">Stage-3 predicted respiration, task {task}:'
             f' {len(panels)} representative subjects</title>')
    L.append('  <desc id="figDesc">Stacked panels, one subject each. The assembled'
             ' Stage-3 prediction is drawn over the reference respiration waveform;'
             ' both are z-scored over the displayed window, so only shape and phase'
             ' are comparable, not amplitude.</desc>')
    L.append('  <style>')
    L.append('    .bg    { fill: #ffffff; }')
    L.append('    .title { font-size: 16px; font-weight: 700; fill: #1a1a1a; }')
    L.append('    .sub   { font-size: 11.5px; fill: #5a5a5a; }')
    L.append('    .ph    { font-size: 12px; font-weight: 700; fill: #1a1a1a; }')
    L.append('    .pm    { font-size: 10.5px; fill: #7a7a7a; }')
    L.append('    .frame { fill: #fcfcfd; stroke: #dcdcdc; stroke-width: 1; }')
    L.append('    .zero  { stroke: #d0d0d0; stroke-width: 1;'
             ' stroke-dasharray: 3 3; }')
    L.append('    .ref   { fill: none; stroke: #9a9a9a; stroke-width: 2.4;'
             ' stroke-linejoin: round; }')
    L.append('    .pred  { fill: none; stroke: #2f6fb5; stroke-width: 1.6;'
             ' stroke-linejoin: round; }')
    L.append('    .tlab  { font-size: 10px; fill: #777777;'
             ' text-anchor: middle; }')
    L.append('    .ylab  { font-size: 10px; fill: #9a9a9a; text-anchor: end; }')
    L.append('    .axt   { font-size: 11px; font-weight: 700; fill: #333333; }')
    L.append('    .foot  { font-size: 10.5px; fill: #6a6a6a; }')
    L.append('  </style>')
    L.append(f'  <rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')

    L.append(f'  <text class="title" x="40" y="32">Stage-3 predicted respiration,'
             f' task {task} &#183; {len(panels)} representative subjects</text>')
    L.append('  <text class="sub" x="40" y="52">BP4D+ &#183; 5-fold CV'
             ' (subject-disjoint) &#183; Hann overlap-add of the per-clip'
             ' predictions</text>')
    L.append(f'  <text class="sub" x="40" y="68">band {band[0]:g}&#8211;'
             f'{band[1]:g} Hz &#183; both signals z-scored per panel: shape and'
             ' phase only, not amplitude</text>')
    L.append('  <line class="ref" x1="660" y1="86" x2="688" y2="86"/>')
    L.append('  <text class="sub" x="694" y="90">reference</text>')
    L.append('  <line class="pred" x1="772" y1="86" x2="800" y2="86"/>')
    L.append('  <text class="sub" x="806" y="90">prediction</text>')

    spans: List[float] = []
    for i, p in enumerate(panels):
        y0 = top0 + i * gap
        t = np.asarray(p['t'], dtype=float)
        if t.size < 2:
            continue
        t0 = float(t[0])
        span = min(float(window_s), float(t[-1] - t[0])) or 1.0
        spans.append(span)
        keep = t <= t0 + span
        zr = _zscore(np.asarray(p['ref'], dtype=float)[keep])
        zp = _zscore(np.asarray(p['pred'], dtype=float)[keep])

        rank = int(p.get('rank', 0))
        n_sub = int(p.get('n_subjects', 0))
        pct = float(p.get('pct', 0.0))
        L.append(f'  <!-- panel {i + 1}: {p["session"]} -->')
        L.append(f'  <text class="ph" x="{left:.0f}" y="{y0 - 9:.0f}">'
                 f'{p["subject"]} &#183; r_band {float(p["pearson_band"]):+.2f}'
                 f' &#183; rank {rank + 1}/{n_sub}, P{pct:.0f}</text>')
        L.append(f'  <text class="pm" x="{right:.0f}" y="{y0 - 9:.0f}"'
                 f' text-anchor="end">fold {p["fold"]} &#183; {p["n_clips"]}'
                 f' clips &#183; {float(t[-1] - t[0]):.0f} s</text>')

        L.append(f'  <rect class="frame" x="{left:.0f}" y="{y0:.0f}"'
                 f' width="{pw:.0f}" height="{ph:.0f}"/>')
        L.append(f'  <line class="zero" x1="{left:.0f}" y1="{y0 + ph / 2:.0f}"'
                 f' x2="{right:.0f}" y2="{y0 + ph / 2:.0f}"/>')
        L.append(f'  <text class="ylab" x="{left - 8:.0f}" y="{y0 + 16:.0f}">'
                 ' z+2</text>')
        L.append(f'  <text class="ylab" x="{left - 8:.0f}"'
                 f' y="{y0 + ph / 2 + 4:.0f}"> 0</text>')
        L.append(f'  <text class="ylab" x="{left - 8:.0f}"'
                 f' y="{y0 + ph - 6:.0f}"> z&#8722;2</text>')
        pr = _poly(t[keep], zr, left, y0, pw, ph, zmax, t0, span)
        pp = _poly(t[keep], zp, left, y0, pw, ph, zmax, t0, span)
        if pr:
            L.append(f'  <polyline class="ref" points="{pr}"/>')
        if pp:
            L.append(f'  <polyline class="pred" points="{pp}"/>')

    # shared x axis under the last panel
    y_last = top0 + (len(panels) - 1) * gap + ph
    span_last = max(spans) if spans else float(window_s)
    for v in _nice_ticks(span_last):
        x = left + v / span_last * pw if span_last > 0 else left
        L.append(f'  <line class="zero" x1="{x:.0f}" y1="{y_last:.0f}"'
                 f' x2="{x:.0f}" y2="{y_last + 4:.0f}"/>')
        L.append(f'  <text class="tlab" x="{x:.0f}" y="{y_last + 16:.0f}">'
                 f'{v:g}</text>')
    L.append(f'  <text class="axt" x="{(left + right) / 2:.0f}"'
             f' y="{y_last + 36:.0f}" text-anchor="middle">'
             'time within the shown window (s)</text>')

    L.append(f'  <text class="foot" x="40" y="{H - 30}">Subjects ranked by their'
             f' {task} band-limited Pearson and drawn at fixed quantiles: a strong,'
             ' a typical, a weak and a poor subject.</text>')
    L.append(f'  <text class="foot" x="40" y="{H - 14}">Generated by'
             ' code/notes/eval/tir_resp/plot_task_waveforms.py from the fold dumps;'
             ' scored metrics in resp_5fold_cv.md.</text>')
    L.append('</svg>')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')
    return out_path


# ------------------------------------------------------------------------ main
def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser('Plot assembled Stage-3 waveforms for one task')
    p.add_argument('--root', default='../output/finetune')
    p.add_argument('--run_prefix', default='resp_tir_roi_hpc_fold')
    p.add_argument('--folds', default='0,1,2,3,4')
    p.add_argument('--task', default='T3')
    p.add_argument('--waveform', default='resp', choices=['bp', 'resp'])
    p.add_argument('--fs', default=100.0, type=float)
    p.add_argument('--raw_fs', default=1000.0, type=float)
    p.add_argument('--signals_root', default='',
                   help='local dataset root (e.g. ../data/processed/bp4d_canonical);'
                        ' rewrites entries.json signals_file to'
                        ' <signals_root>/<session>/<basename>')
    p.add_argument('--band', default='0.1,0.6')
    p.add_argument('--window_s', default=60.0, type=float,
                   help='seconds shown per panel (default 60)')
    p.add_argument('--max_seconds', default=0.0, type=float,
                   help='keep only this many covered seconds (0 = whole span)')
    p.add_argument('--quantiles', default='0.85,0.55,0.30,0.10')
    p.add_argument('--subjects', default='',
                   help='CSV of subject ids, overrides --quantiles')
    p.add_argument('--out', default='')
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = get_args(argv)
    band = tuple(float(x) for x in str(args.band).split(','))
    folds = [int(k) for k in str(args.folds).split(',') if k.strip()]
    quant = tuple(float(q) for q in str(args.quantiles).split(',') if q.strip())
    forced = [s.strip() for s in str(args.subjects).split(',') if s.strip()]
    out = args.out or str(Path(__file__).with_name(
        f'resp_{str(args.task).lower()}_waveforms.svg'))

    per_subject = collect(args.root, args.run_prefix, folds, args.task, args.fs,
                          args.raw_fs, args.waveform, band, args.max_seconds,
                          args.signals_root)
    panels = pick_representatives(per_subject, quant, forced or None)
    for p in panels:
        print(f'[waves] {p["subject"]:>6s} {p["session"]:<12s}'
              f' fold {p["fold"]}  r_band {float(p["pearson_band"]):+.3f}'
              f'  (rank {int(p["rank"]) + 1}/{int(p["n_subjects"])},'
              f' P{float(p["pct"]):.0f})')
    write_svg(panels, str(args.task).upper(), band, args.window_s, out)
    print(f'[waves] wrote {out}')
    return out


if __name__ == '__main__':
    main()
