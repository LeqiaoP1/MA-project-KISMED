#!/usr/bin/env python3
"""Per-task scatter of rhythm quality: band-limited Pearson vs rate MAE.

PROBE script in the ``code/notes/eval/tir_resp/`` sense: it renders an existing
evaluation document and adds no new metric. Every dot is one BP4D+ task, read
straight out of the k-fold JSON written by ``runners/run_evaluate_kfold.py``:

* ``x`` = ``pearson_band_median``  (0.1-0.6 Hz, higher is better),
* ``y`` = ``mae_bpm_median``       (rate error in BPM, lower is better),
* fill = ``amp_rel_err_median``    (oracle-calibrated, descriptive only).

The figure is PER TASK, so it is unaffected by the task-tier definition: there is
no group row and no tier annotation. Nothing is printed below the axes -- the
plot ends at the x-axis title, so no caption can drift away from the data.

Usage (from ``code/``)::

    python notes/eval/tir_resp/plot_task_scatter.py \\
        --kfold ../output/eval/resp_kfold/kfold_resp_tir_roi_hpc_regrouped.json \\
        --out notes/eval/tir_resp/resp_kfold_task_scatter_v2.svg
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ['load', 'write_svg', 'main']

#: plot rectangle: x0, y0, x1, y1
PANEL = (90.0, 100.0, 660.0, 520.0)
X_MIN, X_MAX = 0.15, 0.60
Y_MIN, Y_MAX = 0.0, 6.0
X_TICKS: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
Y_TICKS: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
W = 800.0

#: blue -> pale yellow -> red, as discrete strips (no gradient support needed)
RAMP = ((0.0, (38, 105, 171)), (0.5, (252, 219, 142)), (1.0, (180, 29, 45)))
STRIPS = 42


def _task_key(label: str):
    """``T10`` sorts after ``T9``."""
    m = re.match(r'^[Tt](\d+)$', str(label))
    return (0, int(m.group(1)), '') if m else (1, 0, str(label))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _colour(t: float) -> str:
    """``t`` in [0, 1] -> ``#rrggbb`` on the blue/yellow/red ramp."""
    t = max(0.0, min(1.0, float(t)))
    if t <= 0.5:
        u = t / 0.5
        a, b = RAMP[0][1], RAMP[1][1]
    else:
        u = (t - 0.5) / 0.5
        a, b = RAMP[1][1], RAMP[2][1]
    rgb = [int(round(_lerp(a[i], b[i], u))) for i in range(3)]
    return '#%02x%02x%02x' % tuple(rgb)


def load(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _points(doc: dict) -> List[dict]:
    """One record per task, skipping tasks without all three metrics."""
    per_task: Dict[str, dict] = doc.get('per_task') or {}
    if not per_task:
        raise SystemExit('the k-fold document has no "per_task" block')
    out = []
    for task in sorted(per_task, key=_task_key):
        rec = per_task[task]
        try:
            out.append({
                'task': task,
                'x': float(rec['pearson_band_median']),
                'y': float(rec['mae_bpm_median']),
                'amp': float(rec['amp_rel_err_median']),
                'n': int(rec.get('n', 0)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise SystemExit('no task carries pearson_band / mae_bpm / amp_rel_err')
    return out


def _overlaps(box, placed) -> bool:
    return any(not (box[2] < p[0] or box[0] > p[2]
                    or box[3] < p[1] or box[1] > p[3]) for p in placed)


def _place_labels(points: List[dict], x0: float, y0: float, x1: float,
                  y1: float) -> Dict[str, Tuple[float, float, str]]:
    """Greedy, deterministic label placement around each dot.

    Candidates are tried in a fixed order (right, below-right, left,
    below-left, above); the first box that hits neither another label nor the
    plot border wins. Ties are resolved by the fixed order, so the same JSON
    always renders the same figure.
    """
    sx = (x1 - x0) / (X_MAX - X_MIN)
    sy = (y1 - y0) / (Y_MAX - Y_MIN)
    pw, ph = 24.0, 15.0
    placed: List[Tuple[float, float, float, float]] = []
    out: Dict[str, Tuple[float, float, str]] = {}
    offsets = ((11.0, -7.0, 'start'), (11.0, 15.0, 'start'),
               (-11.0, -7.0, 'end'), (-11.0, 15.0, 'end'),
               (0.0, -13.0, 'middle'))
    for p in points:
        cx = x0 + (p['x'] - X_MIN) * sx
        cy = y1 - (p['y'] - Y_MIN) * sy
        chosen = None
        for dx, dy, anchor in offsets:
            lx, ly = cx + dx, cy + dy
            if anchor == 'start':
                box = (lx, ly - 11.0, lx + pw, ly + 4.0)
            elif anchor == 'end':
                box = (lx - pw, ly - 11.0, lx, ly + 4.0)
            else:
                box = (lx - pw / 2, ly - 11.0, lx + pw / 2, ly + 4.0)
            inside = (box[0] >= x0 - 2 and box[2] <= x1 + 2
                      and box[1] >= y0 - 2 and box[3] <= y1 + 2)
            if inside and not _overlaps(box, placed):
                chosen = (lx, ly, anchor, box)
                break
        if chosen is None:                       # fall back: right of the dot
            lx, ly, anchor = cx + 11.0, cy - 7.0, 'start'
            chosen = (lx, ly, anchor, (lx, ly - 11.0, lx + pw, ly + 4.0))
        lx, ly, anchor, box = chosen
        placed.append(box)
        out[p['task']] = (lx, ly, anchor)
    return out


def write_svg(doc: dict, out_path: str) -> str:
    """Render the scatter to ``out_path`` (no caption below the axes)."""
    pts = _points(doc)
    x0, y0, x1, y1 = PANEL
    axis_title_y = y1 + 42.0
    H = axis_title_y + 22.0

    amps = [p['amp'] for p in pts]
    a_lo, a_hi = min(amps), max(amps)
    span = (a_hi - a_lo) or 1e-6

    def frac(a: float) -> float:
        return (a - a_lo) / span

    sx = (x1 - x0) / (X_MAX - X_MIN)
    sy = (y1 - y0) / (Y_MAX - Y_MIN)

    L: List[str] = []
    L.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800'
             f' {H:.0f}" width="800" height="{H:.0f}"')
    L.append('     role="img" aria-labelledby="figTitle figDesc"'
             ' font-family="Helvetica, Arial, sans-serif">')
    L.append('  <title id="figTitle">Respiration Stage-3, 5-fold CV: per-task'
             ' scatter of rhythm quality</title>')
    L.append('  <desc id="figDesc">Scatter plot of the ten BP4D+ tasks, one dot'
             ' each. The x axis is band-limited Pearson correlation'
             ' (0.1-0.6 Hz, higher is better), the y axis is rate MAE in breaths'
             ' per minute (lower is better). Dot colour encodes the'
             ' oracle-calibrated amplitude relative error, blue for small error'
             ' through red for large error.</desc>')
    L.append('  <style>')
    L.append('    .bg    { fill: #ffffff; }')
    L.append('    .title { font-size: 17px; font-weight: 700; fill: #1a1a1a; }')
    L.append('    .sub   { font-size: 12px; fill: #5a5a5a; }')
    L.append('    .grid  { stroke: #e8e8e8; stroke-width: 1; }')
    L.append('    .axis  { stroke: #8a8a8a; stroke-width: 1.2; }')
    L.append('    .tick  { stroke: #8a8a8a; stroke-width: 1; }')
    L.append('    .tlab  { font-size: 11px; fill: #666666; text-anchor: end; }')
    L.append('    .tlabx { font-size: 11px; fill: #666666;'
             ' text-anchor: middle; }')
    L.append('    .axt   { font-size: 11px; font-weight: 700; fill: #333333; }')
    L.append('    .dot   { stroke: #3a3a3a; stroke-width: 0.8; }')
    L.append('    .dlab  { font-size: 12px; font-weight: 700; fill: #1a1a1a; }')
    L.append('    .hint  { font-size: 11px; font-style: italic; fill: #9a9a9a; }')
    L.append('    .cbt   { font-size: 10px; fill: #666666; }')
    L.append('    .cbtitle { font-size: 10px; fill: #666666; }')
    L.append('  </style>')
    L.append(f'  <rect class="bg" x="0" y="0" width="{W:.0f}"'
             f' height="{H:.0f}"/>')

    # ---- header -----------------------------------------------------------
    L.append('  <text class="title" x="90" y="34">Respiration Stage-3'
             ' &#183; 5-fold CV: per-task scatter (T1&#8211;T10)</text>')
    clips_txt = f'{int(doc["overall"]["n_clips"]):,}'.replace(',', ' ')
    L.append(f'  <text class="sub" x="90" y="54">'
             f'{int(doc["overall"]["n_subjects"])} subjects &#183;'
             f' {int(doc["overall"]["n_sessions"])} sessions &#183;'
             f' {clips_txt} clips &#183; per-subject medians, three metric'
             ' families on one plane</text>')
    L.append('  <text class="axt" x="90" y="88">rate MAE (BPM), lower is'
             ' better</text>')
    L.append('  <text class="cbtitle" x="686" y="88">amp rel. err'
             ' (oracle)</text>')

    # ---- grid -------------------------------------------------------------
    L.append('  <g class="grid">')
    for v in X_TICKS:
        x = x0 + (v - X_MIN) * sx
        L.append(f'    <line x1="{x:.0f}" y1="{y0:.0f}" x2="{x:.0f}"'
                 f' y2="{y1:.0f}"/>')
    for v in Y_TICKS:
        y = y1 - (v - Y_MIN) * sy
        L.append(f'    <line x1="{x0:.0f}" y1="{y:.0f}" x2="{x1:.0f}"'
                 f' y2="{y:.0f}"/>')
    L.append('  </g>')

    # ---- axes + ticks -----------------------------------------------------
    L.append('  <g class="axis">')
    L.append(f'    <line x1="{x0:.0f}" y1="{y1:.0f}" x2="{x1:.0f}"'
             f' y2="{y1:.0f}"/>')
    L.append(f'    <line x1="{x0:.0f}" y1="{y0:.0f}" x2="{x0:.0f}"'
             f' y2="{y1:.0f}"/>')
    L.append('  </g>')
    L.append('  <g class="tick">')
    for v in X_TICKS:
        x = x0 + (v - X_MIN) * sx
        L.append(f'    <line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}"'
                 f' y2="{y1 + 6:.0f}"/>')
    for v in Y_TICKS:
        y = y1 - (v - Y_MIN) * sy
        L.append(f'    <line x1="{x0 - 6:.0f}" y1="{y:.0f}" x2="{x0:.0f}"'
                 f' y2="{y:.0f}"/>')
    L.append('  </g>')
    L.append('  <g class="tlab">')
    for v in Y_TICKS:
        y = y1 - (v - Y_MIN) * sy
        L.append(f'    <text x="{x0 - 8:.0f}" y="{y + 4:.0f}">{v:g}</text>')
    L.append('  </g>')
    L.append('  <g class="tlabx">')
    for v in X_TICKS:
        x = x0 + (v - X_MIN) * sx
        L.append(f'    <text x="{x:.0f}" y="{y1 + 20:.0f}">{v:g}</text>')
    L.append('  </g>')
    L.append(f'  <text class="axt" x="{(x0 + x1) / 2:.0f}"'
             f' y="{axis_title_y:.0f}" text-anchor="middle">band-limited'
             ' Pearson, 0.1&#8211;0.6 Hz (higher is better)</text>')
    L.append(f'  <text class="hint" x="{x1 - 8:.0f}" y="{y1 - 12:.0f}"'
             ' text-anchor="end">better</text>')

    # ---- dots + labels ----------------------------------------------------
    labels = _place_labels(pts, x0, y0, x1, y1)
    L.append('  <!-- task dots: x = band-limited Pearson, y = rate MAE,'
             ' fill = oracle amp rel. err -->')
    for p in pts:
        cx = x0 + (p['x'] - X_MIN) * sx
        cy = y1 - (p['y'] - Y_MIN) * sy
        L.append(f'  <circle class="dot" cx="{cx:.0f}" cy="{cy:.0f}" r="7"'
                 f' fill="{_colour(frac(p["amp"]))}"/>')
    L.append('  <!-- task labels (greedy de-overlap) -->')
    L.append('  <g class="dlab">')
    for p in pts:
        lx, ly, anchor = labels[p['task']]
        extra = '' if anchor == 'start' else f' text-anchor="{anchor}"'
        L.append(f'    <text class="dlab" x="{lx:.0f}" y="{ly:.0f}"{extra}>'
                 f'{p["task"]}</text>')
    L.append('  </g>')

    # ---- colour scale -----------------------------------------------------
    bar_x, bar_y0, bar_y1 = 686.0, y0, y1
    step = (bar_y1 - bar_y0) / STRIPS
    L.append('  <!-- colour scale for amplitude rel. error: discrete strips'
             ' (no gradient support needed) -->')
    L.append('  <g>')
    for i in range(STRIPS):
        # bottom strip = small error (blue), top strip = large error (red)
        y = bar_y1 - (i + 1) * step
        L.append(f'    <rect x="{bar_x:.0f}" y="{y:.1f}" width="16"'
                 f' height="{step + 0.2:.1f}"'
                 f' fill="{_colour(i / (STRIPS - 1.0))}"/>')
    L.append(f'    <rect x="{bar_x:.0f}" y="{bar_y0:.0f}" width="16"'
             f' height="{bar_y1 - bar_y0:.0f}" fill="none" stroke="#9a9a9a"'
             ' stroke-width="0.6"/>')
    L.append('  </g>')
    for i, a in enumerate((a_lo, (a_lo + a_hi) / 2.0, a_hi)):
        y = bar_y1 - frac(a) * (bar_y1 - bar_y0)
        L.append(f'  <line class="tick" x1="{bar_x - 6:.0f}" y1="{y:.0f}"'
                 f' x2="{bar_x:.0f}" y2="{y:.0f}"/>')
        L.append(f'  <text class="cbt" x="{bar_x + 22:.0f}" y="{y + 3:.0f}">'
                 f'{a:.2f}</text>')

    L.append('</svg>')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text('\n'.join(L) + '\n', encoding='utf-8')
    return out_path


# ------------------------------------------------------------------------ main
def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        'Per-task scatter (Pearson vs rate MAE) of a pooled k-fold evaluation')
    p.add_argument('--kfold', default='../output/eval/resp_kfold/'
                                      'kfold_resp_tir_roi_hpc_regrouped.json',
                   help='k-fold JSON written by runners/run_evaluate_kfold.py')
    p.add_argument('--out', default='')
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = get_args(argv)
    out = args.out or str(Path(__file__).with_name(
        'resp_kfold_task_scatter_v2.svg'))
    return write_svg(load(args.kfold), out)


if __name__ == '__main__':
    print(main())
