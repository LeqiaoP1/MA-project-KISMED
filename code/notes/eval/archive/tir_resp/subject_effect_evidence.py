#!/usr/bin/env python3
"""Evidence that T3 (and every other task) is dominated by WHO the subject is.

Answers the question "what numbers or figures support 'it is mostly about who
the person is'?" with four independent pieces of evidence, all from the existing
5-fold dumps (no training, no GPU):

1. **Additive variance decomposition** over the full 138 x 10 table of
   ``pearson_band`` scores: subject dummies vs task dummies. This is the
   headline -- it asks how much of ALL the variation in the whole evaluation is
   "which person" vs "which task".
2. **Split-half reliability** of the person effect: a subject's median over the
   ODD tasks vs their median over the EVEN tasks. The two sets are disjoint
   recordings, so agreement cannot come from a shared session artefact. Reported
   with the Spearman-Brown correction (each half is only 5 tasks, so the
   raw correlation understates the full-measurement reliability).
3. **Leave-one-task-out prediction**: predict a subject's T3 score from their
   median over the other nine tasks (quartile table + Pearson r^2).
4. **Figure** ``resp_t3_subject_effect.svg``: (a) the T3-vs-other-tasks scatter
   with the quartile step, (b) the subject x task score matrix sorted by subject
   median -- horizontal banding means "person", vertical banding means "task".

Usage (from ``code/``)::

    ../.env/bin/python notes/eval/tir_resp/subject_effect_evidence.py
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


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_matrix(rows_path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """``pearson_band`` as a subject x task matrix (NaN where a cell is absent)."""
    with open(rows_path) as fh:
        rows = json.load(fh)
    cells: Dict[Tuple[str, str], float] = {}
    for r in rows:
        subj, task = r['session'].split('_', 1)
        cells[(subj, task)] = float(r['pearson_band'])
    subjects = sorted({s for s, _ in cells})
    tasks = sorted({t for _, t in cells}, key=lambda z: int(z[1:]))
    M = np.full((len(subjects), len(tasks)), np.nan)
    for i, s in enumerate(subjects):
        for j, t in enumerate(tasks):
            if (s, t) in cells:
                M[i, j] = cells[(s, t)]
    return subjects, tasks, M


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """In-sample R^2 of an OLS with intercept."""
    Xd = np.column_stack([np.ones(y.size), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return float(1.0 - np.sum((y - Xd @ beta) ** 2) / np.sum((y - y.mean()) ** 2))


def decomposition(M: np.ndarray, verbose: bool = False) -> Dict[str, float]:
    """R^2 of an additive subject + task model over the observed cells.

    The model is ``y_ij = mu + alpha_i + beta_j + eps_ij``, fitted by ordinary
    least squares on dummy indicators: one intercept, ``n_subject - 1`` subject
    dummies and ``n_task - 1`` task dummies (the first level of each is folded
    into the intercept). ``alpha_i`` and ``beta_j`` are therefore offsets
    RELATIVE TO THE FIRST LEVEL, not to the grand mean.

    ``unique to subject`` = R^2(subject+task) - R^2(task) is the variation only
    the subject can explain, and vice versa. The design is 91 % complete, so the
    two blocks are nearly orthogonal (max |corr| between a subject dummy and a
    task dummy is 0.04) and the unique shares coincide with the marginal ones.

    There is NO interaction term: with one session per (subject, task) cell an
    interaction is not identifiable, so any genuine person x task interaction is
    inside the residual.
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
    y = np.array(y)
    n = y.size
    S = np.eye(M.shape[0])[rows][:, 1:]
    T = np.eye(M.shape[1])[cols][:, 1:]
    X_s = np.column_stack([np.ones(n), S])
    X_t = np.column_stack([np.ones(n), T])
    X_b = np.column_stack([np.ones(n), S, T])

    def _fit(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        f = X @ beta
        return beta, f, 1.0 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)

    _, _, r_sub = _fit(X_s)
    _, f_s, _ = _fit(X_s)
    _, f_t, r_task = _fit(X_t)
    beta, f_b, r_both = _fit(X_b)
    ss_full = float(np.sum((y - f_b) ** 2))
    df_s, df_t = M.shape[0] - 1, M.shape[1] - 1
    df_r = n - 1 - df_s - df_t
    mse = ss_full / df_r

    from scipy import stats
    out = dict(n_cells=float(n), sd=float(np.std(y)), mean=float(np.mean(y)),
               completeness=float(n) / M.size,
               r2_subject=r_sub, r2_task=r_task, r2_both=r_both,
               uniq_subject=r_both - r_task, uniq_task=r_both - r_sub,
               adj_r2_subject=1 - (1 - r_sub) * (n - 1) / (n - df_s - 1),
               adj_r2_task=1 - (1 - r_task) * (n - 1) / (n - df_t - 1),
               adj_r2_both=1 - (1 - r_both) * (n - 1) / (n - df_s - df_t - 1),
               residual_sd=float(np.sqrt(mse)), df_resid=float(df_r))
    # partial F: drop one block from the full model, compare the residual SS
    F_s = ((float(np.sum((y - f_t) ** 2)) - ss_full) / df_s) / mse
    F_t = ((float(np.sum((y - f_s) ** 2)) - ss_full) / df_t) / mse
    out.update(f_subject=float(F_s),
               p_subject=float(stats.f.sf(F_s, df_s, df_r)),
               f_task=float(F_t), p_task=float(stats.f.sf(F_t, df_t, df_r)))
    if verbose:
        inter = beta[0]
        alpha = beta[1:1 + df_s]
        out['alpha_sd'] = float(np.std(alpha, ddof=1))
        out['alpha_min'] = float(alpha.min())
        out['alpha_max'] = float(alpha.max())
        out['task_effects'] = [float(v) for v in beta[1 + df_s:]]
        out['orthogonality'] = float(np.abs(
            np.corrcoef(np.column_stack([S, T]).T)[:df_s, df_s:]).max())
        out['rank'] = float(np.linalg.matrix_rank(X_b))
        out['n_params'] = float(X_b.shape[1])
        out['intercept'] = float(inter)
    return out


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def _lerp(c0: Sequence[int], c1: Sequence[int], t: float) -> str:
    return 'rgb(%d,%d,%d)' % tuple(int(round(a + (b - a) * t))
                                   for a, b in zip(c0, c1))


def _heat(v: float, lo: float = -0.6, hi: float = 1.0) -> str:
    """Diverging colour: blue (low) - white (0) - red (high)."""
    if not np.isfinite(v):
        return '#f0f0f0'
    if v < 0:
        t = min(1.0, (0.0 - v) / max(-lo, 1e-9))
        return _lerp((255, 255, 255), (49, 87, 160), t)
    t = min(1.0, v / max(hi, 1e-9))
    return _lerp((255, 255, 255), (176, 58, 46), t)


def write_svg(M: np.ndarray, subjects: Sequence[str], tasks: Sequence[str],
              out_path: str) -> str:
    t3 = np.array([M[i, tasks.index('T3')] if 'T3' in tasks else np.nan
                   for i in range(M.shape[0])])
    other = np.array([np.nanmedian([M[i, j] for j in range(len(tasks))
                                    if tasks[j] != 'T3'])
                      for i in range(M.shape[0])])
    ok = np.isfinite(t3) & np.isfinite(other)
    x, y = other[ok], t3[ok]

    W, H = 1000.0, 640.0
    # panel A
    AX, AY, AW, AH = 80.0, 120.0, 380.0, 380.0
    # panel B (heatmap)
    BX, BY, BW, BH = 560.0, 120.0, 300.0, 380.0
    xlo, xhi = -0.25, 0.85
    ylo, yhi = -0.60, 1.00

    def px(v):
        return AX + (v - xlo) / (xhi - xlo) * AW

    def py(v):
        return AY + AH - (v - ylo) / (yhi - ylo) * AH

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}"'
         f' width="{W:.0f}" height="{H:.0f}" role="img"'
         ' font-family="Helvetica, Arial, sans-serif">',
         '<rect width="100%" height="100%" fill="#ffffff"/>',
         '<text x="80" y="36" font-size="17" font-weight="700" fill="#1a1a1a">'
         'The T3 spread is mostly "who the subject is"</text>',
         '<text x="80" y="56" font-size="11.5" fill="#5a5a5a">'
         'band-limited Pearson (0.1-0.6 Hz), 5-fold CV, 138 subjects x 10 tasks '
         '(1260 sessions)</text>',
         '<text x="80" y="74" font-size="11.5" fill="#5a5a5a">'
         'subject identity explains <tspan font-weight="700" fill="#b03a2e">'
         '36.0 %</tspan> of all score variance; the task only '
         '<tspan font-weight="700">8.6 %</tspan> (additive two-way model, '
         'R&#178; 0.446)</text>']

    s.append(f'<rect x="{AX:.0f}" y="{AY:.0f}" width="{AW:.0f}" height="{AH:.0f}"'
             ' fill="#fcfcfd" stroke="#dcdcdc"/>')
    s.append(f'<text x="{AX:.0f}" y="{AY - 12:.0f}" font-size="12" '
             'font-weight="700" fill="#1a1a1a">'
             '(a) T3 score vs the same subject&#8217;s median on the other 9 '
             'tasks</text>')
    for v in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8):
        yy = py(v)
        s.append(f'<line x1="{AX:.0f}" y1="{yy:.1f}" x2="{AX + AW:.0f}" '
                 f'y2="{yy:.1f}" stroke="#ededed"/>')
        s.append(f'<text x="{AX - 6:.0f}" y="{yy + 4:.1f}" font-size="10" '
                 f'fill="#888" text-anchor="end">{v:+.1f}</text>')
    for v in (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8):
        xx = px(v)
        s.append(f'<line x1="{xx:.1f}" y1="{AY:.0f}" x2="{xx:.1f}" '
                 f'y2="{AY + AH:.0f}" stroke="#ededed"/>')
        s.append(f'<text x="{xx:.1f}" y="{AY + AH + 16:.0f}" font-size="10" '
                 f'fill="#888" text-anchor="middle">{v:+.1f}</text>')
    s.append(f'<text x="{AX + AW / 2:.0f}" y="{AY + AH + 34:.0f}" font-size="10.5"'
             ' fill="#666" text-anchor="middle">'
             'median score on the other nine tasks</text>')
    s.append(f'<text x="24" y="{AY + AH / 2:.0f}" font-size="10.5" fill="#666"'
             f' text-anchor="middle" transform="rotate(-90 24 '
             f'{AY + AH / 2:.0f})">T3 score</text>')
    s.append(f'<line x1="{px(0):.1f}" y1="{py(ylo):.0f}" x2="{px(0):.1f}" '
             f'y2="{py(yhi):.0f}" stroke="#c9c9c9" stroke-dasharray="3 3"/>')

    # quartile step + points
    q = np.quantile(x, np.linspace(0, 1, 5))
    steps = []
    for i in range(4):
        m = ((x >= q[i]) & (x < q[i + 1]) if i < 3 else
             (x >= q[i]) & (x <= q[i + 1]))
        if m.any():
            steps.append((float(np.median(x[m])), float(np.median(y[m])), m.sum()))
    for xi, yi, _ in steps:
        s.append(f'<circle cx="{px(xi):.1f}" cy="{py(yi):.1f}" r="5.2" '
                 'fill="#b03a2e" stroke="#ffffff" stroke-width="1.4"/>')
    if len(steps) > 1:
        s.append('<polyline fill="none" stroke="#b03a2e" stroke-width="2.4" '
                 'stroke-linejoin="round" points="'
                 + ' '.join(f'{px(a):.1f},{py(b):.1f}' for a, b, _ in steps)
                 + '"/>')
    for xi, yi, n in steps:
        s.append(f'<text x="{px(xi):.1f}" y="{py(yi) - 11:.1f}" font-size="9.5"'
                 f' fill="#b03a2e" text-anchor="middle">{yi:+.2f}</text>')
    for xi, yi in zip(x, y):
        s.append(f'<circle cx="{px(xi):.1f}" cy="{py(yi):.1f}" r="2.1" '
                 'fill="#2f6fb5" fill-opacity="0.55"/>')

    # regression line
    b, a0 = np.polyfit(x, y, 1)
    s.append(f'<line x1="{px(xlo):.1f}" y1="{py(a0 + b * xlo):.1f}" '
             f'x2="{px(xhi):.1f}" y2="{py(a0 + b * xhi):.1f}" '
             'stroke="#333" stroke-width="1.3" stroke-dasharray="5 4"/>')
    from scipy import stats
    rho = stats.spearmanr(x, y)[0]
    rp = stats.pearsonr(x, y)[0]
    s.append(f'<text x="{AX + 8:.0f}" y="{AY + 16:.0f}" font-size="10.5" '
             f'fill="#333">Spearman rho {rho:+.3f} &#183; Pearson r {rp:+.3f} '
             f'(R&#178; {rp ** 2:.2f}) &#183; n {x.size}</text>')
    s.append(f'<text x="{AX + 8:.0f}" y="{AY + 31:.0f}" font-size="10.5" '
             'fill="#b03a2e">red = quartile medians, from +0.22 (Q1) to +0.79 '
             '(Q4)</text>')

    # ---- panel B: heatmap
    order = np.argsort(np.nan_to_num(np.nanmedian(M, axis=1), nan=-9.0))
    nrow = M.shape[0]
    rh = BH / nrow
    s.append(f'<text x="{BX:.0f}" y="{BY - 12:.0f}" font-size="12" '
             'font-weight="700" fill="#1a1a1a">'
             '(b) every subject (row) x every task (column)</text>')
    s.append(f'<text x="{BX:.0f}" y="{BY - 1:.0f}" font-size="10" '
             'fill="#7a7a7a">sorted by the subject&#8217;s median over its '
             'tasks</text>')
    for rr, i in enumerate(order):
        for j in range(len(tasks)):
            v = M[i, j]
            s.append(f'<rect x="{BX + j * (BW / len(tasks)):.1f}" '
                     f'y="{BY + rr * rh:.1f}" '
                     f'width="{BW / len(tasks):.1f}" height="{rh + 0.4:.1f}" '
                     f'fill="{_heat(v)}"/>')
    for j, t in enumerate(tasks):
        s.append(f'<text x="{BX + (j + 0.5) * (BW / len(tasks)):.1f}" '
                 f'y="{BY + BH + 15:.0f}" font-size="10" fill="#666" '
                 f'text-anchor="middle">{t}</text>')
    # subject-median strip on the right
    sx = BX + BW + 6.0
    for rr, i in enumerate(order):
        v = float(np.nanmedian(M[i]))
        if not np.isfinite(v):
            continue
        w = max(0.0, (v - (-0.6)) / 1.6) * 46.0
        s.append(f'<rect x="{sx:.1f}" y="{BY + rr * rh:.1f}" width="{w:.1f}" '
                 f'height="{rh + 0.4:.1f}" fill="{_heat(v)}" stroke="none"/>')
    s.append(f'<text x="{sx:.0f}" y="{BY + BH + 15:.0f}" font-size="9.5" '
             'fill="#666">subject median</text>')
    for rr, i, lab in ((0, order[0], subjects[order[0]]),
                       (nrow - 1, order[-1], subjects[order[-1]])):
        v = float(np.nanmedian(M[i]))
        s.append(f'<text x="{sx + 50:.0f}" y="{BY + rr * rh + rh + 2:.0f}" '
                 f'font-size="9.5" fill="#666">{lab} {v:+.2f}</text>')
    s.append(f'<text x="{BX:.0f}" y="{BY + BH + 34:.0f}" font-size="10.5" '
             'fill="#5a5a5a">rows read as bands, not speckle.</text>')

    s.append('<text x="80" y="600" font-size="10.5" fill="#6a6a6a">'
             'Split-half check (disjoint recordings): a subject&#8217;s median '
             'over the ODD tasks vs the EVEN tasks, Spearman +0.69 '
             '(Spearman-Brown reliability +0.82, n 137).</text>')
    s.append('<text x="80" y="618" font-size="10.5" fill="#6a6a6a">'
             'Caveat: one session per subject per task, so the person effect is '
             'inferred across tasks, not from repeated T3 recordings. '
             'Full detail: resp_t3_spread_sources.md</text>')
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
    p.add_argument('--svg', default=(
        'code/notes/eval/tir_resp/resp_t3_subject_effect.svg'))
    a = p.parse_args()

    repo = Path(_CODE_DIR).parent

    def _p(path: str) -> str:
        return os.path.abspath(path if os.path.isabs(path)
                               else str(repo / path))

    subjects, tasks, M = load_matrix(_p(a.rows))
    print(f'[subject] {len(subjects)} subjects x {len(tasks)} tasks, '
          f'{int(np.isfinite(M).sum())} sessions')

    d = decomposition(M, verbose=True)
    print('\n=== additive two-way model over ALL cells '
          f'({int(d["n_cells"])} sessions, SD {d["sd"]:.3f}) ===')
    print(f'  design : {len(subjects)} subjects x {len(tasks)} tasks, '
          f'{int(d["n_cells"])} cells observed = '
          f'{100 * d["completeness"]:.1f} % complete')
    print(f'  model  : y_ij = mu + alpha_i + beta_j + eps_ij, OLS on dummies, '
          f'{int(d["n_params"])} parameters, rank {int(d["rank"])}')
    print(f'           intercept {d["intercept"]:+.3f}, alpha SD {d["alpha_sd"]:.3f}, '
          f'range {d["alpha_min"]:+.3f} .. {d["alpha_max"]:+.3f}')
    print(f'           residual df {int(d["df_resid"])}, residual SD '
          f'{d["residual_sd"]:.3f} (total SD {d["sd"]:.3f})')
    print(f'  subject dummies only      R2 = {d["r2_subject"]:+.3f}   '
          f'(adjusted {d["adj_r2_subject"]:+.3f})')
    print(f'  task dummies only         R2 = {d["r2_task"]:+.3f}   '
          f'(adjusted {d["adj_r2_task"]:+.3f})')
    print(f'  subject + task            R2 = {d["r2_both"]:+.3f}   '
          f'(adjusted {d["adj_r2_both"]:+.3f})')
    print(f'  -> unique to SUBJECT      R2 = {d["uniq_subject"]:+.3f}')
    print(f'  -> unique to TASK         R2 = {d["uniq_task"]:+.3f}')
    print(f'  partial F({int(d["df_resid"] * 0 + len(subjects) - 1)},{int(d["df_resid"])})'
          f' subject | task = {d["f_subject"]:.2f}  p = {d["p_subject"]:.1e}')
    print(f'  partial F({len(tasks) - 1},{int(d["df_resid"])}) task | subject = '
          f'{d["f_task"]:.2f}  p = {d["p_task"]:.1e}')
    print(f'  orthogonality: max |corr| subject dummy x task dummy = '
          f'{d["orthogonality"]:.4f}')
    print('  task effects (T1 = reference): ' +
          '  '.join(f'{t}:{v:+.3f}'
                    for t, v in zip(tasks[1:], d['task_effects'])))

    from scipy import stats
    odd = [t for t in tasks if int(t[1:]) % 2 == 1]
    even = [t for t in tasks if int(t[1:]) % 2 == 0]
    a_ = np.array([np.nanmedian([M[i, tasks.index(t)] for t in odd
                                 if np.isfinite(M[i, tasks.index(t)])])
                   for i in range(len(subjects))])
    b_ = np.array([np.nanmedian([M[i, tasks.index(t)] for t in even
                                 if np.isfinite(M[i, tasks.index(t)])])
                   for i in range(len(subjects))])
    ok = np.isfinite(a_) & np.isfinite(b_)
    rho = stats.spearmanr(a_[ok], b_[ok])[0]
    print(f'\n=== split-half reliability (disjoint task sets) ===')
    print(f'  odd tasks {odd}')
    print(f'  even tasks {even}')
    print(f'  Spearman rho = {rho:+.3f} (n={int(ok.sum())})')
    print(f'  Spearman-Brown corrected reliability = {2 * rho / (1 + rho):+.3f}')

    t3 = np.array([M[i, tasks.index('T3')] for i in range(len(subjects))])
    other = np.array([np.nanmedian([M[i, j] for j in range(len(tasks))
                                    if tasks[j] != 'T3'])
                      for i in range(len(subjects))])
    ok = np.isfinite(t3) & np.isfinite(other)
    rp = stats.pearsonr(other[ok], t3[ok])[0]
    print('\n=== predicting T3 from the other nine tasks ===')
    print(f'  Spearman {stats.spearmanr(other[ok], t3[ok])[0]:+.3f}, '
          f'Pearson {rp:+.3f} (R2 {rp ** 2:.3f}), n={int(ok.sum())}')
    q = np.quantile(other[ok], np.linspace(0, 1, 5))
    for i in range(4):
        m = ((other[ok] >= q[i]) & (other[ok] < q[i + 1]) if i < 3 else
             (other[ok] >= q[i]) & (other[ok] <= q[i + 1]))
        print(f'  Q{i + 1} (n={int(m.sum()):2d}) other-task median '
              f'{np.median(other[ok][m]):+.3f} -> T3 median '
              f'{np.median(t3[ok][m]):+.3f}')

    print(f'\n[subject] wrote {write_svg(M, subjects, tasks, _p(a.svg))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
