#!/usr/bin/env python3
"""Slide figure: what "a simple additive model" means, drawn from the real fit.

The figure is one horizontal equation, with no title and no footnotes (it is
meant to be captioned by the speaker, not read):

    observed score  =  overall level  +  person offset  +  task offset  +  noise

with five 12-subject x 10-task grids generated from the ACTUAL fitted model
(``y_ij = mu + alpha_i + beta_j + eps_ij`` over all 1260 sessions), followed by a
stacked bar showing how much of the total variation each part carries
(36.0 % person, 8.6 % task, 55.4 % left over).

Only 12 of the 138 subjects are drawn so the row banding is legible on a slide;
they are taken from subjects that have all ten tasks (no empty cells) and at
evenly spaced ranks of their fitted person effect. Every number on the figure
(ranges, SDs, shares) is computed from the full fit. Each component grid is
colour-scaled to the 95th percentile of its own |value| so a few outlier cells
cannot wash the panel out.

Output is SVG (the deck's other figures are ``figures/<name>.svg.png`` -- convert
with any SVG rasteriser before inserting). Design target: a full-width slide
image, 1500 x 520 px.

Usage (from ``code/``)::

    ../.env/bin/python notes/eval/tir_resp/additive_model_slide.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CODE_DIR))

FONT = 'Helvetica, Arial, sans-serif'


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def fit_additive(M: np.ndarray) -> Dict[str, object]:
    """OLS fit of ``y_ij = mu + alpha_i + beta_j + eps_ij`` on dummy indicators.

    :param M: ``[n_subjects, n_tasks]`` score matrix (NaN where absent).
    :returns: dict with ``mu``, ``alpha`` (centred), ``beta`` (centred),
        ``fitted``, ``resid`` and the R^2 shares.
    """
    rows, cols, y = [], [], []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                rows.append(i)
                cols.append(j)
                y.append(M[i, j])
    rows = np.array(rows)
    cols = np.array(cols)
    y = np.array(y, float)
    n, ns, nt = y.size, M.shape[0], M.shape[1]
    S = np.eye(ns)[rows][:, 1:]
    T = np.eye(nt)[cols][:, 1:]
    X = np.column_stack([np.ones(n), S, T])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta

    def _r2(Z: np.ndarray) -> float:
        b, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return float(1 - np.sum((y - Z @ b) ** 2) / np.sum((y - y.mean()) ** 2))

    r_sub = _r2(np.column_stack([np.ones(n), S]))
    r_task = _r2(np.column_stack([np.ones(n), T]))
    r_both = _r2(X)

    # convert the dummy parameterisation into centred offsets
    alpha_raw = np.concatenate([[0.0], beta[1:1 + ns - 1]])
    beta_raw = np.concatenate([[0.0], beta[1 + ns - 1:]])
    mu = float(beta[0] + alpha_raw.mean() + beta_raw.mean())
    alpha = alpha_raw - alpha_raw.mean()
    bet = beta_raw - beta_raw.mean()

    A = np.full((ns, nt), np.nan)
    B = np.full((ns, nt), np.nan)
    E = np.full((ns, nt), np.nan)
    F = np.full((ns, nt), np.nan)
    for i, j, v, f in zip(rows, cols, y, fitted):
        A[i, j] = alpha[i]
        B[i, j] = bet[j]
        E[i, j] = v - f
        F[i, j] = f
    return dict(mu=mu, alpha=alpha, beta=bet, A=A, B=B, E=E, fitted=F,
                observed=M.copy(), n=n, sd=float(np.std(y)),
                share_person=r_sub, share_task=r_task,
                share_both=r_both,
                share_noise=1.0 - r_both)


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def _lerp(c0: Sequence[int], c1: Sequence[int], t: float) -> str:
    return 'rgb(%d,%d,%d)' % tuple(int(round(a + (b - a) * t))
                                   for a, b in zip(c0, c1))


def _div(v: float, lo: float = -0.6, hi: float = 1.0) -> str:
    """Diverging blue - white - red, shared across every panel of the figure."""
    if not np.isfinite(v):
        return '#f2f2f2'
    if v < 0:
        return _lerp((255, 255, 255), (44, 82, 150), min(1.0, v / lo))
    return _lerp((255, 255, 255), (176, 58, 46), min(1.0, v / hi))


def _grid(s: List[str], x: float, y: float, cw: float, ch: float,
          values: np.ndarray, vmax: float) -> None:
    """Draw a ``values`` matrix as a grid of cells, colour-scaled to ±vmax."""
    vmax = max(float(vmax), 1e-6)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            first_row = i == 0 or not np.isfinite(values[i - 1, j])
            first_col = j == 0 or not np.isfinite(values[i, j - 1])
            s.append(f'<rect x="{x + j * cw:.1f}" y="{y + i * ch:.1f}" '
                     f'width="{cw + 0.3:.2f}" height="{ch + 0.3:.2f}" '
                     f'fill="{_div(v, -vmax, vmax)}" stroke="none"/>')
            if first_row:
                s.append(f'<line x1="{x:.1f}" y1="{y + i * ch:.1f}" '
                         f'x2="{x + values.shape[1] * cw:.1f}" '
                         f'y2="{y + i * ch:.1f}" stroke="#ffffff" '
                         'stroke-width="0.5"/>')
            if first_col:
                s.append(f'<line x1="{x + j * cw:.1f}" y1="{y:.1f}" '
                         f'x2="{x + j * cw:.1f}" '
                         f'y2="{y + values.shape[0] * ch:.1f}" '
                         'stroke="#ffffff" stroke-width="0.5"/>')


def write_svg(fit: Dict[str, object], out_path: str) -> str:
    """Compose the equation figure and write it."""
    alpha = np.asarray(fit['alpha'], float)
    bet = np.asarray(fit['beta'], float)
    mu = float(fit['mu'])
    A = np.asarray(fit['A'])
    B = np.asarray(fit['B'])
    E = np.asarray(fit['E'])
    O = np.asarray(fit['observed'])

    # 12 subjects at evenly spaced ranks of the fitted person effect, taken from
    # the subjects that have ALL ten tasks, so the grids have no empty cells
    order = np.argsort(alpha)
    complete = np.array([bool(np.isfinite(O[i]).all()) for i in range(O.shape[0])])
    order = np.array([i for i in order if complete[i]])
    rank = np.linspace(0, len(order) - 1, 12).round().astype(int)
    pick = order[rank]
    keep = np.zeros(O.shape[0], dtype=bool)
    keep[pick] = True
    rows = np.flatnonzero(keep)
    subj = [f'S{r + 1:02d}' for r in range(len(rows))]     # anonymised row labels

    W, H = 1500.0, 520.0
    CW, CH = 15.0, 20.0
    GX, GY = 200.0, 108.0
    STEP = 246.0
    xs = [GX + k * STEP for k in range(5)]

    def sub(M, rws):
        return M[np.ix_(rws, np.arange(M.shape[1]))]

    s: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}"'
        f' width="{W:.0f}" height="{H:.0f}" role="img"'
        f' font-family="{FONT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    # ---- the equation panels
    titles = ['observed score', 'overall level', 'person', 'task',
              'everything else']
    subs = ['what we measured', 'a constant', 'who the subject is',
            'which task it was', 'residual']
    panels = [O, np.full_like(O, mu), A, B, E]
    notes = [f'mean {np.nanmean(O):+.2f}',
             f'{mu:+.2f} for everyone',
             f'spread {alpha.min():+.2f} \u2026 {alpha.max():+.2f}',
             f'spread {bet.min():+.2f} \u2026 {bet.max():+.2f}',
             f'SD {np.nanstd(E):.2f}']

    for k, (x, M, title, cap, note) in enumerate(
            zip(xs, panels, titles, subs, notes)):
        s.append(f'<text x="{x:.0f}" y="{GY - 44:.0f}" font-size="16" '
                 f'font-weight="700" fill="#1a1a1a">{title}</text>')
        s.append(f'<text x="{x:.0f}" y="{GY - 24:.0f}" font-size="12.5" '
                 f'fill="#7a7a7a">{cap}</text>')
        # colour-scale to the 95th percentile of |value| so a handful of
        # outlier cells cannot wash the panel out
        vis = np.abs(M[np.isfinite(M)])
        vmax = float(np.percentile(vis, 95)) if vis.size else 0.5
        if k == 0:
            vmax = max(vmax, 0.6)
        _grid(s, x, GY, CW, CH, sub(M, rows), vmax)
        s.append(f'<rect x="{x:.1f}" y="{GY:.1f}" width="{10 * CW:.1f}" '
                 f'height="{12 * CH:.1f}" fill="none" stroke="#cfcfcf"/>')
        s.append(f'<text x="{x:.0f}" y="{GY + 12 * CH + 24:.0f}" font-size="12.5"'
                 f' fill="#333">{note}</text>')

    # row labels on the first panel, task ticks above it
    for i, name in enumerate(subj):
        s.append(f'<text x="{GX - 8:.0f}" y="{GY + i * CH + 14:.0f}" '
                 f'font-size="10.5" fill="#9a9a9a" text-anchor="end">{name}'
                 '</text>')
    s.append(f'<text x="{GX - 8:.0f}" y="{GY - 8:.0f}" font-size="10.5" '
             'fill="#9a9a9a" text-anchor="end">subject</text>')
    for j in range(0, 10, 2):
        s.append(f'<text x="{GX + j * CW + CW / 2:.1f}" y="{GY - 8:.0f}" '
                 f'font-size="10.5" fill="#9a9a9a" text-anchor="middle">'
                 f'T{j + 1}</text>')

    # operators: observed = level + person + task + residual
    for k in range(1, 5):
        mid = (xs[k - 1] + 10 * CW + xs[k]) / 2.0
        op = '=' if k == 1 else '+'
        s.append(f'<text x="{mid:.0f}" y="{GY + 12 * CH / 2 + 12:.0f}" '
                 f'font-size="34" fill="#b9b9b9" text-anchor="middle">{op}'
                 '</text>')

    # ---- stacked share bar
    BX, BY, BW, BH = 200.0, 440.0, 1100.0, 46.0
    sh_p = float(fit['share_person'])
    sh_t = float(fit['share_task'])
    sh_n = float(fit['share_noise'])
    segs = [(sh_p, '#b03a2e', f'who the person is  {100 * sh_p:.0f} %'),
            (sh_t, '#2f6fb5', f'which task  {100 * sh_t:.1f} %'),
            (sh_n, '#d9d9d9', f'left over  {100 * sh_n:.0f} %')]
    cx = BX
    s.append(f'<text x="{BX:.0f}" y="{BY - 26:.0f}" font-size="14" '
             'fill="#5a5a5a">share of the total variation</text>')
    for frac, col, lab in segs:
        w = BW * frac
        s.append(f'<rect x="{cx:.1f}" y="{BY:.1f}" width="{w:.1f}" '
                 f'height="{BH:.1f}" fill="{col}"/>')
        dark = col != '#d9d9d9'
        if w >= 150:
            s.append(f'<text x="{cx + w / 2:.1f}" y="{BY + BH / 2 + 6:.1f}" '
                     f'font-size="15" font-weight="700" text-anchor="middle" '
                     f'fill="{"#ffffff" if dark else "#555555"}">{lab}</text>')
        else:                      # too narrow to hold the label inside
            s.append(f'<text x="{cx:.1f}" y="{BY - 6:.0f}" '
                     f'font-size="14" font-weight="700" text-anchor="start" '
                     f'fill="#2f6fb5">{lab}</text>')
        cx += w

    s.append('</svg>')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(s))
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rows', default=(
        'output/eval/resp_kfold/all_tasks_motion_vs_quality.json'))
    p.add_argument('--out', default=(
        'presentation/figures/additive-model-slide.svg'))
    a = p.parse_args()

    repo = Path(_CODE_DIR).parent

    def _p(path: str) -> str:
        return os.path.abspath(path if os.path.isabs(path)
                               else str(repo / path))

    with open(_p(a.rows)) as fh:
        rows = json.load(fh)
    cells: Dict[Tuple[str, str], float] = {}
    for r in rows:
        s, t = r['session'].split('_', 1)
        cells[(s, t)] = float(r['pearson_band'])
    subjects = sorted({s for s, _ in cells})
    tasks = sorted({t for _, t in cells}, key=lambda z: int(z[1:]))
    M = np.full((len(subjects), len(tasks)), np.nan)
    for i, s in enumerate(subjects):
        for j, t in enumerate(tasks):
            if (s, t) in cells:
                M[i, j] = cells[(s, t)]

    fit = fit_additive(M)
    print(f'[slide] {int(fit["n"])} sessions, total SD {fit["sd"]:.3f}')
    print(f'[slide] mu {fit["mu"]:+.3f}, person SD '
          f'{np.std(fit["alpha"], ddof=1):.3f}, task SD '
          f'{np.std(fit["beta"], ddof=1):.3f}, residual SD '
          f'{np.nanstd(fit["E"]):.3f}')
    print(f'[slide] shares: person {100 * fit["share_person"]:.1f} %, '
          f'task {100 * fit["share_task"]:.1f} %, '
          f'noise {100 * fit["share_noise"]:.1f} %')
    print(f'[slide] wrote {write_svg(fit, _p(a.out))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
