#!/usr/bin/env python3
"""Per-task and per-GROUP medians of one pooled k-fold evaluation, as an SVG.

This is a PROBE script in the ``code/notes/eval/tir_resp/`` sense: it renders an
existing evaluation document and adds no new metric. Every number it draws is
read straight out of the k-fold JSON written by
``runners/run_evaluate_kfold.py`` (``per_task`` / ``per_group``), so the figure
can never disagree with the report it illustrates.

Grouping is AGGREGATION ONLY: ``per_task`` does not depend on the tier
definition, and ``per_group`` is a pure re-aggregation of the same sessions. That
is why a re-grouped dump is rendered from
``kfold_resp_tir_roi_hpc_regrouped.json`` (``--kfold``) -- no re-training, no
change to any per-task row.

Layout contract: task rows are in **T1..T10 order** (not sorted by score), and
the tasks of ONE tier (``--bold-tier``, default ``low``) are drawn in bold so the
membership is visible without a caption. Nothing is printed below the axes --
every explanatory line lives in the header, above the diagram.

Usage (from ``code/``)::

    python notes/eval/tir_resp/plot_task_group_medians.py \\
        --kfold ../output/eval/resp_kfold/kfold_resp_tir_roi_hpc_regrouped.json \\
        --out notes/eval/tir_resp/resp_kfold_task_group_medians_v2.svg

``--note`` is OFF by default (empty); pass it only when a one-off figure has to
state the grouping version, e.g. ``--note 'tiers v2, 2026-10-24'``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ['load', 'write_svg', 'main']

#: (key, panel header, axis maximum, ticks, value format)
PANELS: Tuple[Tuple[str, str, float, Tuple[float, ...], str], ...] = (
    ('pearson_band_median', 'band-limited Pearson (0.1\u20130.6 Hz) \u2191',
     0.6, (0.0, 0.2, 0.4, 0.6), '{:+.3f}'),
    ('amp_rel_err_median', 'amplitude rel. error (oracle) \u2193',
     0.8, (0.0, 0.4, 0.8), '{:.3f}'),
    ('mae_bpm_median', 'rate MAE (BPM) \u2193',
     6.0, (0.0, 2.0, 4.0, 6.0), '{:.2f}'),
)

# panel plot ranges (x0, x1) and the x of the numeric value label
PANEL_X: Tuple[Tuple[float, float, float], ...] = (
    (96.0, 326.0, 334.0),
    (402.0, 632.0, 640.0),
    (708.0, 938.0, 946.0),
)
W, LEFT_LABEL = 1000, 80.0
ROW_H, BAR_H = 28.0, 16.0
TASK_TOP = 104.0
GROUP_GAP = 10.0


def load(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _task_key(label: str):
    """``T10`` sorts after ``T9`` -- the rows read T1, T2, ..., T10."""
    m = re.match(r'^[Tt](\d+)$', str(label))
    return (0, int(m.group(1)), '') if m else (1, 0, str(label))


def _rows(doc: dict):
    """``(tasks, groups)`` plus the two blocks.

    Tasks are in LABEL order (T1..T10) -- the figure is a lookup table, not a
    ranking. Groups keep the descending band-limited-Pearson order, which is
    what makes the tier comparison readable at a glance.
    """
    per_task: Dict[str, dict] = doc.get('per_task') or {}
    per_group: Dict[str, dict] = doc.get('per_group') or {}
    if not per_task:
        raise SystemExit('the k-fold document has no "per_task" block')
    tasks = sorted(per_task, key=_task_key)
    groups = sorted(per_group, key=lambda g: -float(per_group[g]['pearson_band_median']))
    return tasks, groups, per_task, per_group


def _bar(value: float, x0: float, x1: float, vmax: float) -> Tuple[float, float]:
    """Clip to ``[0, vmax]`` -> (width, end_x). Negative values draw nothing."""
    w = max(0.0, min(float(value), vmax)) / vmax * (x1 - x0)
    return w, x0 + w


def write_svg(doc: dict, out_path: str, note: str = '',
              bold_tier: str = 'low') -> str:
    """Render the three horizontal-bar panels to ``out_path``.

    ``bold_tier`` names the tier whose TASKS are drawn in bold (label and value).
    Everything explanatory goes in the header; nothing is placed below the axes.
    """
    tasks, groups, per_task, per_group = _rows(doc)
    n_rows = len(tasks) + len(groups)
    group_top = TASK_TOP + len(tasks) * ROW_H + GROUP_GAP
    axis_y = group_top + len(groups) * ROW_H + 6.0
    H = axis_y + 18.0 + 12.0

    def row_top(i: int) -> float:
        return TASK_TOP + i * ROW_H

    def group_row_top(j: int) -> float:
        return group_top + j * ROW_H

    tiers = (doc.get('protocol') or {}).get('task_groups') or {}
    bold_tier = str(bold_tier or '').strip().lower()
    bold_tasks = {str(t).upper() for t in (tiers.get(bold_tier) or ())}


    L: List[str] = []
    L.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000'
             f' {H:.0f}" width="1000" height="{H:.0f}"')
    L.append('     role="img" aria-labelledby="figTitle figDesc"'
             ' font-family="Helvetica, Arial, sans-serif">')
    L.append('  <title id="figTitle">Respiration Stage-3, 5-fold CV:'
             ' per-task and per-group medians'
             + (' (grouping v2)' if note else '') + '</title>')
    L.append('  <desc id="figDesc">Three horizontal bar panels showing band-limited'
             ' Pearson (0.1-0.6 Hz), oracle amplitude relative error and rate MAE'
             ' (BPM) for the ten BP4D+ tasks T1 to T10 and the three score groups.'
             f' Task rows are in T1-T10 order and the {bold_tier or "highlighted"}'
             ' tier&#39;s tasks are drawn in bold; the group rows are a'
             ' re-aggregation of the same sessions under the shipped'
             ' task-tier definition' + (f' ({note})' if note else '') + '.</desc>')
    L.append('  <style>')
    L.append('    .bg    { fill: #ffffff; }')
    L.append('    .title { font-size: 17px; font-weight: 700; fill: #1a1a1a; }')
    L.append('    .sub   { font-size: 12px; fill: #5a5a5a; }')
    L.append('    .hdr   { font-size: 13px; font-weight: 700; fill: #1a1a1a; }')
    L.append('    .lab   { font-size: 13px; fill: #333333; text-anchor: end; }')
    L.append('    .labB  { font-size: 13px; font-weight: 700; fill: #1a1a1a;'
             ' text-anchor: end; }')
    L.append('    .val   { font-size: 11px; fill: #4a4a4a;'
             ' font-variant-numeric: tabular-nums; }')
    L.append('    .valB  { font-size: 11px; font-weight: 700; fill: #1a1a1a;'
             ' font-variant-numeric: tabular-nums; }')
    L.append('    .task  { fill: #3b6ea5; }')
    L.append('    .group { fill: #e07b39; }')
    L.append('    .band  { fill: #f5f8fc; }')
    L.append('    .gband { fill: #fdf3e7; }')
    L.append('    .grid  { stroke: #e2e2e2; stroke-width: 1; }')
    L.append('    .axis  { stroke: #9a9a9a; stroke-width: 1; }')
    L.append('    .tick  { stroke: #9a9a9a; stroke-width: 1; }')
    L.append('    .tlab  { font-size: 10px; fill: #777777;'
             ' text-anchor: middle; }')
    L.append('    .sep   { stroke: #cfc6b8; stroke-width: 1;'
             ' stroke-dasharray: 4 3; }')
    L.append('    .foot  { font-size: 11px; fill: #6a6a6a; }')
    L.append('  </style>')
    L.append(f'  <rect class="bg" x="0" y="0" width="{W}" height="{H:.0f}"/>')

    # ---- header -----------------------------------------------------------
    L.append('  <text class="title" x="92" y="28">Respiration Stage-3'
             ' &#183; 5-fold CV: per-task and per-group medians</text>')
    clips_txt = f'{int(doc["overall"]["n_clips"]):,}'.replace(',', ' ')
    L.append(f'  <text class="sub" x="92" y="48">{int(doc["overall"]["n_subjects"])}'
             f' subjects &#183; {int(doc["overall"]["n_sessions"])} sessions'
             f' &#183; {clips_txt} clips &#183; median over sessions, then'
             ' median over subjects</text>')
    if tiers:
        tier_txt = ' &#183; '.join(
            f'{g} = {", ".join(tiers[g])}' for g in ('low', 'moderate', 'high')
            if g in tiers)
        L.append(f'  <text class="sub" x="92" y="66">score groups: {tier_txt}'
                 + (f' &#183; {note}' if note else '') + '</text>')
    L.append('  <rect class="task" x="742" y="36"'
             ' width="12" height="12" rx="2"/>')
    L.append('  <text class="sub" x="760" y="47">per task (T1&#8211;T10)</text>')
    L.append('  <rect class="group" x="872" y="36" width="12" height="12"'
             ' rx="2"/>')
    L.append('  <text class="sub" x="890" y="47">score group</text>')

    # ---- panel headers ----------------------------------------------------
    for (key, header, _, _, _), (x0, _, _) in zip(PANELS, PANEL_X):
        L.append(f'  <text class="hdr" x="{x0:.0f}" y="86">{header}</text>')

    # ---- row backgrounds --------------------------------------------------
    for i in range(len(tasks)):
        if i % 2:
            L.append(f'  <rect class="band" x="{LEFT_LABEL + 4:.0f}"'
                     f' y="{row_top(i):.0f}" width="{W - 90}"'
                     f' height="{ROW_H - 2:.0f}"/>')
    if groups:
        L.append(f'  <rect class="gband" x="{LEFT_LABEL + 4:.0f}"'
                 f' y="{group_top - 7:.0f}" width="{W - 90}"'
                 f' height="{len(groups) * ROW_H + 8:.0f}" rx="3"/>')

    # ---- vertical grid ----------------------------------------------------
    L.append('  <g class="grid">')
    for (_, _, vmax, ticks, _), (x0, x1, _) in zip(PANELS, PANEL_X):
        for v in ticks:
            x = x0 + (v / vmax) * (x1 - x0)
            L.append(f'    <line x1="{x:.0f}" y1="{TASK_TOP - 10:.0f}"'
                     f' x2="{x:.0f}" y2="{axis_y:.0f}"/>')
    L.append('  </g>')

    # ---- row labels -------------------------------------------------------
    L.append('  <g class="lab">')
    for i, t in enumerate(tasks):
        cls = 'labB' if t in bold_tasks else 'lab'
        L.append(f'    <text class="{cls}" x="{LEFT_LABEL:.0f}"'
                 f' y="{row_top(i) + 12:.0f}">{t}</text>')
    L.append('  </g>')
    L.append('  <g class="lab">')
    for j, g in enumerate(groups):
        cls = 'labB' if g == bold_tier else 'lab'
        L.append(f'    <text class="{cls}" x="{LEFT_LABEL:.0f}"'
                 f' y="{group_row_top(j) + 12:.0f}">{g}</text>')
    L.append('  </g>')
    if groups:
        L.append(f'  <line class="sep" x1="{LEFT_LABEL + 4:.0f}"'
                 f' y1="{group_top - 10:.0f}" x2="{W - 6}"'
                 f' y2="{group_top - 10:.0f}"/>')

    # ---- bars + values ----------------------------------------------------
    for pi, (key, _, vmax, _, fmt) in enumerate(PANELS):
        x0, x1, vx = PANEL_X[pi]
        L.append('  <g class="task">')
        for i, t in enumerate(tasks):
            v = float(per_task[t][key])
            w, _ = _bar(v, x0, x1, vmax)
            L.append(f'    <rect x="{x0:.0f}" y="{row_top(i) + 6:.0f}"'
                     f' width="{w:.1f}" height="{BAR_H:.0f}" rx="2"/>')
        L.append('  </g>')
        L.append('  <g class="group">')
        for j, g in enumerate(groups):
            v = float(per_group[g][key])
            w, _ = _bar(v, x0, x1, vmax)
            L.append(f'    <rect x="{x0:.0f}" y="{group_row_top(j) + 6:.0f}"'
                     f' width="{w:.1f}" height="{BAR_H:.0f}" rx="2"/>')
        L.append('  </g>')
        L.append('  <g class="val">')
        for i, t in enumerate(tasks):
            v = float(per_task[t][key])
            cls = 'valB' if t in bold_tasks else 'val'
            L.append(f'    <text class="{cls}" x="{vx:.0f}"'
                     f' y="{row_top(i) + 12:.0f}">{fmt.format(v)}</text>')
        for j, g in enumerate(groups):
            v = float(per_group[g][key])
            cls = 'valB' if g == bold_tier else 'val'
            L.append(f'    <text class="{cls}" x="{vx:.0f}"'
                     f' y="{group_row_top(j) + 12:.0f}">{fmt.format(v)}</text>')
        L.append('  </g>')

    # ---- axes -------------------------------------------------------------
    L.append('  <g class="axis">')
    for x0, x1, _ in PANEL_X:
        L.append(f'    <line x1="{x0:.0f}" y1="{axis_y:.0f}" x2="{x1:.0f}"'
                 f' y2="{axis_y:.0f}"/>')
    L.append('  </g>')
    L.append('  <g class="tick">')
    for (_, _, vmax, ticks, _), (x0, x1, _) in zip(PANELS, PANEL_X):
        for v in ticks:
            x = x0 + (v / vmax) * (x1 - x0)
            L.append(f'    <line x1="{x:.0f}" y1="{axis_y:.0f}"'
                     f' x2="{x:.0f}" y2="{axis_y + 5:.0f}"/>')
    L.append('  </g>')
    L.append('  <g class="tlab">')
    for (_, _, vmax, ticks, _), (x0, x1, _) in zip(PANELS, PANEL_X):
        for v in ticks:
            x = x0 + (v / vmax) * (x1 - x0)
            L.append(f'    <text x="{x:.0f}" y="{axis_y + 18:.0f}">{v:g}</text>')
    L.append('  </g>')

    # No caption below the axes by design: the tier definition, the ordering and
    # the bold rule are all stated in the header, so nothing can drift out of
    # sync with the rows underneath it.
    L.append('</svg>')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')
    return out_path


# ------------------------------------------------------------------------ main
def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        'Per-task / per-group medians of a pooled k-fold evaluation (SVG)')
    p.add_argument('--kfold', default='../output/eval/resp_kfold/'
                                      'kfold_resp_tir_roi_hpc_regrouped.json',
                   help='k-fold JSON written by runners/run_evaluate_kfold.py')
    p.add_argument('--note', default='',
                   help='annotation shown in the subtitle and the title suffix')
    p.add_argument('--bold-tier', default='low',
                   help='tier whose TASKS are drawn in bold (default low)')
    p.add_argument('--out', default='')
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = get_args(argv)
    doc = load(args.kfold)
    out = args.out or str(Path(__file__).with_name(
        'resp_kfold_task_group_medians_v2.svg'))
    return write_svg(doc, out, note=str(args.note).strip(),
                     bold_tier=str(args.bold_tier).strip())


if __name__ == '__main__':
    print(main())
