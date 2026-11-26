#!/usr/bin/env python3
"""Per-task scatter of rhythm quality: band-limited Pearson vs rate MAE.

PROBE script in the ``code/notes/eval/tir_resp/`` sense: it renders an existing
evaluation document and adds no new metric. Every dot is one BP4D+ task, read
straight out of the k-fold JSON written by ``runners/run_evaluate_kfold.py``:

* ``x`` = ``pearson_band_median``  (0.1-0.6 Hz, higher is better),
* ``y`` = ``mae_bpm_median``       (rate error in BPM, lower is better),
* fill = ``amp_rel_err_median``    (oracle-calibrated, descriptive only).

Axes are FIXED so successive renders stay comparable: x = band-limited Pearson
0.15-0.60, y = rate MAE 0-8 BPM, colour = amplitude relative error 0.40-0.80
(a value outside a fixed range is clipped, never rescaled). One marker carries
two channels: the FILL is the amplitude error and the RING is the task's
distortion tier (low / moderate / high), with an in-plot key naming each tier's
tasks. Nothing is printed below the axes. Nothing is printed below the axes -- the
plot ends at the x-axis title, so no caption can drift away from the data.

Usage (from ``code/``)::

    python notes/eval/tir_resp/plot_task_scatter.py \\
        --kfold ../output/eval/resp_kfold/kfold_resp_tir_roi_hpc_regrouped.json \\
        --out notes/eval/tir_resp/resp_kfold_task_scatter_v2.svg
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ['load', 'write_svg', 'main']

#: plot rectangle: x0, y0, x1, y1
PANEL = (90.0, 100.0, 660.0, 520.0)
X_MIN, X_MAX = 0.15, 0.60
Y_MIN, Y_MAX = 0.0, 8.0
X_TICKS: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
Y_TICKS: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

#: FIXED colour domain for the oracle amplitude relative error. Not derived from
#: the data, so two renders stay directly comparable and a re-aggregated dump
#: cannot silently rescale the colour map.
A_MIN, A_MAX = 0.40, 0.80
A_TICKS: Tuple[float, ...] = (0.40, 0.50, 0.60, 0.70, 0.80)

#: Task-tier membership is drawn as a RING around each dot -- the fill stays
#: reserved for the amplitude error, so one marker carries both channels. The
#: three ring colours are deliberately outside the blue/yellow/red fill ramp.
TIER_ORDER: Tuple[str, ...] = ('low', 'moderate', 'high')
#: Colour alone is not enough here: the dot FILL already uses the whole
#: blue-yellow-red ramp, so a dark purple "moderate" ring read as "another dark
#: ring" next to the near-black "high" one. Moderate therefore also gets a DASH
#: pattern -- a shape cue that survives any hue confusion (and colour blindness).
TIER_STROKE = {'low': '#1a8f3c', 'moderate': '#00a8c0', 'high': '#101010'}
TIER_DASH = {'low': '', 'moderate': ' stroke-dasharray="4.2 2.6"',
             'high': ''}
TIER_DESC = {'low': 'solid green', 'moderate': 'dashed cyan',
             'high': 'solid black'}

#: marker geometry. The tier ring sits OUTSIDE a white separator ring, so its
#: colour never competes with the amplitude fill it surrounds (a thin ring drawn
#: directly on the fill is hard to match against the legend).
DOT_R = 7.5
GAP_R, GAP_W = 8.7, 2.4
RING_R, RING_W = 10.2, 2.6

#: in-plot legend box (x0, y0, width, height); the lower-left of the panel is
#: empty for the shipped dump, and the box is opaque so gridlines do not bleed
#: through the key.
LEGEND_BOX = (98.0, 404.0, 212.0, 86.0)
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


def _overlaps(box, placed) -> bool:    return any(not (box[2] < p[0] or box[0] > p[2]
                    or box[3] < p[1] or box[1] > p[3]) for p in placed)


def _dodge(xy, min_sep: float, passes: int = 60, cap: float = 9.0):
    """Push markers apart until every pair is at least ``min_sep`` px apart.

    Display-only, and deliberately capped. Two tasks whose medians are nearly
    identical -- T4 and T7 differ by 0.09 BPM and 0.001 Pearson -- would
    otherwise draw one ring straight on top of the other, hiding a whole tier.
    The shift is limited to ``cap`` px (well under one grid step), so the dot
    still reads as "T4/T7 overlap" while both rings stay visible.
    """
    xy = [list(p) for p in xy]
    orig = [tuple(p) for p in xy]
    for _ in range(passes):
        moved = False
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                dx, dy = xy[j][0] - xy[i][0], xy[j][1] - xy[i][1]
                dist = math.hypot(dx, dy)
                if dist >= min_sep:
                    continue
                if dist < 1e-6:
                    dx, dy, dist = 0.0, -1.0, 1.0
                push = (min_sep - dist) / 2.0
                ux, uy = dx / dist, dy / dist
                xy[i][0] -= push * ux
                xy[i][1] -= push * uy
                xy[j][0] += push * ux
                xy[j][1] += push * uy
                moved = True
        if not moved:
            break
    for k, (ox, oy) in enumerate(orig):
        dx, dy = xy[k][0] - ox, xy[k][1] - oy
        dist = math.hypot(dx, dy)
        if dist > cap:
            xy[k][0] = ox + dx / dist * cap
            xy[k][1] = oy + dy / dist * cap
    return xy


def _place_labels(points: List[dict], xy, x0: float, y0: float, x1: float,
                  y1: float) -> Dict[str, Tuple[float, float, str]]:
    """Greedy, deterministic label placement around each dot.

    Candidates are tried in a fixed order (right, below-right, left,
    below-left, above); the first box that hits neither another label nor the
    plot border wins. Ties are resolved by the fixed order, so the same JSON
    always renders the same figure. ``xy`` are the (already dodged) marker
    centres.
    """
    pw, ph = 24.0, 15.0
    placed: List[Tuple[float, float, float, float]] = []
    out: Dict[str, Tuple[float, float, str]] = {}
    offsets = ((15.0, -7.0, 'start'), (15.0, 15.0, 'start'),
               (-15.0, -7.0, 'end'), (-15.0, 15.0, 'end'),
               (0.0, -17.0, 'middle'))
    for p, (cx, cy) in zip(points, xy):
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
            lx, ly, anchor = cx + 15.0, cy - 7.0, 'start'
            chosen = (lx, ly, anchor, (lx, ly - 11.0, lx + pw, ly + 4.0))
        lx, ly, anchor, box = chosen
        placed.append(box)
        out[p['task']] = (lx, ly, anchor)
    return out


def _tiers_of(doc: dict) -> Dict[str, Tuple[str, ...]]:
    """``{task: (tier, ...)}`` from the document's recorded ``task_groups``.

    Order follows :data:`TIER_ORDER`, so a task listed in two tiers renders as
    the plain grey ring (the figure never invents a mixed colour).
    """
    groups = (doc.get('protocol') or {}).get('task_groups') or {}
    out: Dict[str, List[str]] = {}
    for name in TIER_ORDER:
        for task in groups.get(name, ()):
            out.setdefault(str(task).upper(), []).append(name)
    return {k: tuple(v) for k, v in out.items()}


def write_svg(doc: dict, out_path: str) -> str:
    """Render the scatter to ``out_path`` (no caption below the axes)."""
    pts = _points(doc)
    tier_map = _tiers_of(doc)
    x0, y0, x1, y1 = PANEL
    axis_title_y = y1 + 42.0
    H = axis_title_y + 22.0

    # fixed colour domain: see A_MIN / A_MAX (values outside are clipped)
    a_lo, a_hi = A_MIN, A_MAX
    span = a_hi - a_lo

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
             ' through red for large error; the ring around each dot encodes the'
             ' task&#39;s distortion tier ('
             + ', '.join(f'{TIER_DESC[t]} = {t}' for t in TIER_ORDER)
             + '), and the in-plot key lists the tasks of each tier.</desc>')
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
    L.append('    .lgbox { fill: #ffffff; fill-opacity: 0.94;'
             ' stroke: #dcdcdc; stroke-width: 0.8; }')
    L.append('    .lgh   { font-size: 10px; font-weight: 700; fill: #777777; }')
    L.append('    .lgd   { font-size: 11px; fill: #333333; }')
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
    xy = _dodge([(x0 + (p['x'] - X_MIN) * sx, y1 - (p['y'] - Y_MIN) * sy)
                 for p in pts], 2.0 * RING_R + 2.0)
    labels = _place_labels(pts, xy, x0, y0, x1, y1)
    L.append('  <!-- task dots: x = band-limited Pearson, y = rate MAE,'
             ' fill = oracle amp rel. err, outer ring = distortion tier -->')
    for p, (cx, cy) in zip(pts, xy):
        tiers = tier_map.get(p['task'], ())
        ring = TIER_STROKE.get(tiers[0], '#3a3a3a') if len(tiers) == 1 else '#3a3a3a'
        L.append(f'  <circle class="dot" cx="{cx:.0f}" cy="{cy:.0f}"'
                 f' r="{DOT_R:g}" fill="{_colour(frac(p["amp"]))}"'
                 ' stroke="none"/>')
        # white separator, then the tier ring OUTSIDE it
        L.append(f'  <circle cx="{cx:.0f}" cy="{cy:.0f}" r="{GAP_R:g}"'
                 f' fill="none" stroke="#ffffff" stroke-width="{GAP_W:g}"/>')
        L.append(f'  <circle class="tier" cx="{cx:.0f}" cy="{cy:.0f}"'
                 f' r="{RING_R:g}" fill="none" stroke="{ring}"'
                 f' stroke-width="{RING_W:g}"'
                 f'{TIER_DASH.get(tiers[0], "") if len(tiers) == 1 else ""}/>')
    L.append('  <!-- task labels (greedy de-overlap) -->')
    L.append('  <g class="dlab">')
    for p in pts:
        lx, ly, anchor = labels[p['task']]
        extra = '' if anchor == 'start' else f' text-anchor="{anchor}"'
        L.append(f'    <text class="dlab" x="{lx:.0f}" y="{ly:.0f}"{extra}>'
                 f'{p["task"]}</text>')
    L.append('  </g>')

    # ---- distortion-tier legend (ring encoding) ---------------------------
    if tier_map:
        bx, by, bw, bh = LEGEND_BOX
        L.append('  <!-- tier legend: ring colour = low / moderate / high -->')
        L.append(f'  <rect class="lgbox" x="{bx:.0f}" y="{by:.0f}"'
                 f' width="{bw:.0f}" height="{bh:.0f}" rx="3"/>')
        L.append(f'  <text class="lgh" x="{bx + 10:.0f}" y="{by + 16:.0f}">'
                 'distortion tier (outer ring)</text>')
        for i, tier in enumerate(TIER_ORDER):
            members = list((doc.get('protocol') or {})
                           .get('task_groups', {}).get(tier, ()))
            if not members:
                continue
            ry = by + 34.0 + i * 18.0
            L.append(f'  <circle cx="{bx + 16:.0f}" cy="{ry - 4:.0f}" r="6"'
                     ' fill="#ffffff"'
                     f' stroke="{TIER_STROKE.get(tier, "#3a3a3a")}"'
                     f' stroke-width="{2.8:g}"'
                     f'{TIER_DASH.get(tier, "")}/>')
            L.append(f'  <text class="lgd" x="{bx + 28:.0f}"'
                     f' y="{ry:.0f}">{tier} = {", ".join(members)}</text>')
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
    for a in A_TICKS:
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
