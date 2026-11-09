"""Stage-2 AMMM pilot report: is it fast enough, and has it converged?

Reads the plain-text train log of a Stage-2 run (``run_pretrain.py``) and
answers the two questions a k-fold sweep must settle BEFORE it is queued:

1. **How fast is an epoch?** From the ``Epoch: [k] Total time: H:MM:SS (x s /
   it)`` lines plus the steady-state ``time:`` / ``data:`` meters, projected to
   a full epoch budget (default 150) including checkpoint writes.
2. **How many epochs are needed?** From the per-epoch
   ``Epoch k stats: {...}`` dict: the change of the last decile of epochs vs
   the decile before it (robust to AMMM's per-step budget noise).
3. **What did the masker actually do?** ``--plot`` draws, per fold, the
   MASK-RATIO distribution of each input modality (TIR-ROI, RESP) together with
   the run's own mean ratio. The log only stores 10-20-iteration WINDOW MEANS
   of ``kept_<s>``, so the per-step masks CANNOT be reconstructed from it: the
   histogram is the plan POLICY the run drew from (``plan_step`` with the
   config's caps/alpha/``min_kept``), which the run's mean validates.

Why not just trust the loss: Stage 2 has no validation split and no eval loop,
and under ``masking: ammm`` the value ALSO moves with the step's Dirichlet
budget draw (fewer visible TIR patches means more masked positions to score),
so the curve is a MONITOR, not a metric -- and it is comparable within this arm
only. The verdict on the encoder is the Stage-3 subject-level reconstruction
metric. This tool refuses to pretend otherwise: it reports the numbers and a
conservative "still descending / flat within the schedule" flag.

Usage (from ``code/``)::

    # a whole run's logs (dir), or the individual .out files
    python notes/eval/tir_resp/ammm_pilot_report.py ../output/pretrain/<run>/logs
    python notes/eval/tir_resp/ammm_pilot_report.py logs/st2_pretrain_12345.out

    # project a different budget, e.g. 150 epochs after a 20-epoch pilot
    python notes/eval/tir_resp/ammm_pilot_report.py <log> --epochs 150

    # one training-curve PNG per fold (Stage 2 itself writes NO figure):
    python notes/eval/tir_resp/ammm_pilot_report.py "$WORK_SCRATCH"/logs --plot

    # the mask-ratio histograms come from the SAME policy; pass the run's knobs
    # if they differ from the shipped AMMM arm:
    ... --plot --video_kept_cap 0.25 --tir_units 36 --resp_units 50
"""
import argparse
import ast
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TOTAL = re.compile(r'^Epoch: \[(\d+)\] Total time: ([0-9:]+) \(([0-9.]+) s / it\)')
PROG = re.compile(r'^Epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\].*?'
                  r'time:\s*([0-9.]+)\s+data:\s*([0-9.]+)')
STATS = re.compile(r'^Epoch (\d+) stats: (\{.*\})\s*$')
SCHED = re.compile(r'^Step-level LR schedule: (\d+) steps')
#: keys printed by engines/pretrain.py (present ones are shown, others skipped)
KEYS = ('loss', 'mse_tir', 'mse_resp', 'kept_tir', 'kept_resp', 'n_batch',
        'grad_norm')


def _hms(text: str) -> float:
    """'0:01:26' / '1:02:03' -> seconds."""
    parts = [int(p) for p in text.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _iter_logs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir()
                              if q.suffix in ('.out', '.log', '.txt')))
        elif p.exists():
            out.append(p)
        else:
            print(f'[warn] not found, skipped: {p}', file=sys.stderr)
    return out


def parse_log(path: Path) -> Dict:
    """Per-epoch seconds / s-it / steady meters / stats, last occurrence wins."""
    epochs: Dict[int, Dict] = {}
    sched_steps: Optional[int] = None
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            m = TOTAL.match(line)
            if m:
                epochs.setdefault(int(m.group(1)), {})['total_s'] = _hms(m.group(2))
                epochs[int(m.group(1))]['s_it'] = float(m.group(3))
                continue
            m = PROG.match(line)
            if m:
                e = int(m.group(1))
                rec = epochs.setdefault(e, {})
                rec['steps'] = int(m.group(3))
                rec['time_s_it'] = float(m.group(4))
                rec['data_s_it'] = float(m.group(5))
                continue
            m = STATS.match(line)
            if m:
                try:
                    stats = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    continue
                if isinstance(stats, dict):
                    epochs.setdefault(int(m.group(1)), {}).update(stats)
                continue
            m = SCHED.match(line)
            if m:
                sched_steps = int(m.group(1))
    return {'path': path, 'epochs': epochs, 'sched_steps': sched_steps}


def _tail_delta(rows: List[Dict], key: str):
    """Relative change of the last decile vs the decile before it.

    Deliberately NOT a short slope: under AMMM the epoch-level loss moves with
    the step's budget draw, so a 5-epoch window can read "+0.9 %/epoch RISING"
    inside a perfectly flat plateau (measured on the first fold: last 25 epochs
    span 0.2036-0.2187). Deciles average that noise out.

    :return: ``(delta, n_epochs_per_window)``; ``(None, None)`` when too short.
    """
    vals = [r[key] for r in rows if key in r]
    if len(vals) < 10:
        return None, None
    k = max(5, len(vals) // 10)
    prev, last = vals[-2 * k:-k], vals[-k:]
    if not prev or not last:
        return None, None
    base, cur = statistics.fmean(prev), statistics.fmean(last)
    if abs(base) < 1e-12:
        return None, None
    return (cur - base) / abs(base), k


def _slope_per_epoch(rows: List[Dict], key: str, window: int) -> Optional[float]:
    """Least-squares slope of the last ``window`` epochs, as % of the mean."""
    vals = [r[key] for r in rows if key in r][-window:]
    if len(vals) < 3:
        return None
    xs = list(range(len(vals)))
    mx, my = statistics.fmean(xs), statistics.fmean(vals)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0 or abs(my) < 1e-12:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, vals)) / den
    return 100.0 * slope / abs(my)


def _plt():
    """Headless matplotlib (login/compute nodes have no display)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def mask_ratio_samples(tir_units: int = 36, resp_units: int = 50,
                       video_cap: float = 0.25, physio_cap: float = 1.0,
                       alpha: float = 1.0, min_kept: int = 0,
                       draws: int = 20000, seed: int = 0) -> Dict[str, List[float]]:
    """Sample the MASK ratio of each stream from the run's own plan policy.

    The train log keeps only window means of ``kept_<s>``, so the realised
    per-step masks are not recoverable. This calls the SAME
    ``core.ammm_masking.plan_step`` with the config's caps/alpha/min_kept and the
    run's geometry, i.e. it reproduces the distribution the run drew from; the
    run's mean mask ratio is the check that the knobs used here are right.

    :return: ``{'tir': [mask ratio per draw], 'resp': [...]}`` where the mask
        ratio is ``1 - kept/n_units`` (a ratio in ``[0, 1]`` on a 1/n grid).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # code/
    import torch                                   # local: keeps --plot lazy
    from core.ammm_masking import StreamSpec, plan_step

    specs = [StreamSpec('tir', 'visual', int(tir_units)),
             StreamSpec('resp', 'signal', int(resp_units))]
    caps = {'tir': float(video_cap), 'resp': float(physio_cap)}
    gen = torch.Generator().manual_seed(int(seed))
    units = {'tir': int(tir_units), 'resp': int(resp_units)}
    out: Dict[str, List[float]] = {'tir': [], 'resp': []}
    for _ in range(int(draws)):
        plan = plan_step(specs, caps=caps, alpha=float(alpha),
                         min_kept=int(min_kept), generator=gen)
        for name in ('tir', 'resp'):
            n = units[name]
            out[name].append(1.0 - plan.kept_units[name] / float(n))
    return out


def _mask_hist_panel(ax, name: str, units: int, ratios: List[float],
                     colour: str, obs_mean: Optional[float],
                     alpha: float, cap: float) -> None:
    """Discrete PMF of the mask ratio (= 1 - kept/n) as a histogram."""
    n = int(units)
    counts = [0] * (n + 1)
    for r in ratios:
        k = int(round(r * n))
        counts[max(0, min(n, k))] += 1
    total = float(len(ratios)) or 1.0
    xs = [i / n for i in range(n + 1)]
    ax.bar(xs, [c / total for c in counts], width=0.85 / n, color=colour,
           align='center')
    if obs_mean is not None:
        ax.axvline(obs_mean, ls='--', lw=1.2, color='k',
                   label='run mean {:.3f}'.format(obs_mean))
        ax.legend(fontsize=8)
    lo = min(xs[min((i for i, c in enumerate(counts) if c), default=0)], 0.99)
    ax.set_xlim(max(0.0, lo - 0.03), 1.02)
    ax.set_title('{} mask ratio  ({} {}, Dir({:g},{:g}), kept cap {:g})'.format(
        name, n, 'patches' if name.startswith('TIR') else 'tokens',
        alpha, alpha, cap), fontsize=10)
    ax.set_xlabel('mask ratio  (1 = 100 % masked)')
    ax.set_ylabel('fraction of steps')
    ax.grid(alpha=.3, axis='y')
    if counts[n]:
        ax.annotate('{:.2%} fully masked'.format(counts[n] / total),
                    xy=(1.0, counts[n] / total), xytext=(-6, 6),
                    textcoords='offset points', ha='right', fontsize=8)


def plot_log(path: Path, rows: List[Dict], steps: Optional[int],
             out_png: Path,
             mask_ratios: Optional[Dict[str, List[float]]] = None,
             policy_note: str = '',
             units: Optional[Dict[str, int]] = None,
             policy: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """2x2 figure per fold log: losses | TIR mask ratio | RESP mask ratio |
    kept units + encoder length.

    ``run_pretrain.py`` writes NO artefact besides checkpoints and stdout (unlike
    Stage 3, which writes ``metrics.jsonl`` + ``training_curves.png`` in the
    loop), so this is the only per-fold figure a Stage-2 sweep can produce.

    :param rows: the per-epoch dicts returned by :func:`report`.
    :param mask_ratios: output of :func:`mask_ratio_samples`; ``None`` (or a
        non-AMMM run) leaves the two histogram panels annotated instead.
    """
    try:
        plt = _plt()
    except ImportError:
        print('[warn] matplotlib is not installed -- cannot plot', file=sys.stderr)
        return None

    nan = float('nan')
    ep = [r['epoch'] for r in rows]

    def col(key):
        return [r.get(key, nan) for r in rows]

    def fin(vals):
        return any(v == v for v in vals)

    def mean_of(key, units):
        """Run's mean mask ratio from the epoch means of ``kept_<s>``."""
        v = [x for x in col(key) if x == x]
        return None if not v else 1.0 - statistics.fmean(v) / float(units)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0), constrained_layout=True)
    fig.suptitle(path.name + ('   ({} steps/epoch)'.format(steps) if steps else ''),
                 fontsize=11)

    ax = axes[0][0]
    for key, c in (('loss', 'tab:red'), ('mse_tir', 'tab:blue'),
                   ('mse_resp', 'tab:green')):
        v = col(key)
        if fin(v):
            ax.plot(ep, v, lw=1.4, color=c, label=key)
    ax.set_title('masked reconstruction loss (a MONITOR, not a metric)')
    ax.set_xlabel('epoch'); ax.grid(alpha=.3)
    if ax.lines:
        ax.legend(fontsize=8)

    units = dict(units or {'tir': 36, 'resp': 50})
    policy = dict(policy or {})
    caps = dict(policy.get('caps') or {'tir': 0.25, 'resp': 1.0})
    alpha = float(policy.get('alpha', 1.0))
    if mask_ratios and any('kept_tir' in r for r in rows):
        _mask_hist_panel(axes[0][1], 'TIR-ROI', units['tir'],
                         mask_ratios['tir'], 'tab:blue',
                         mean_of('kept_tir', units['tir']),
                         alpha, caps['tir'])
        _mask_hist_panel(axes[1][0], 'RESP', units['resp'],
                         mask_ratios['resp'], 'tab:green',
                         mean_of('kept_resp', units['resp']),
                         alpha, caps['resp'])
    else:
        for a in (axes[0][1], axes[1][0]):
            a.text(.5, .5, 'no AMMM plan (masking: ratio)', ha='center',
                   va='center', transform=a.transAxes)
            a.set_title('mask-ratio histogram')

    ax = axes[1][1]
    kt, kr, nb = col('kept_tir'), col('kept_resp'), col('n_batch')
    if fin(kt) or fin(kr):
        if fin(kt):
            ax.plot(ep, kt, marker='.', ms=2, lw=0, color='tab:blue',
                    label='kept_tir (of 36 patches)')
            zero = [e for e, v in zip(ep, kt) if v == 0]
            if zero:
                ax.plot(zero, [0] * len(zero), 'x', ms=7, color='tab:red',
                        label='TIR 100 % masked ({} epoch(s))'.format(len(zero)))
        if fin(kr):
            ax.plot(ep, kr, marker='.', ms=2, lw=0, color='tab:green',
                    label='kept_resp (of 50 tokens)')
        ax.set_ylabel('kept units'); ax.legend(fontsize=8, loc='lower left')
        ax.set_title('AMMM plan per epoch (left) + encoder length (right)')
        if fin(nb):
            twin = ax.twinx()
            twin.plot(ep, nb, lw=1.0, color='tab:purple', alpha=.75,
                      label='n_batch')
            twin.set_ylabel('n_batch', color='tab:purple')
            twin.legend(fontsize=8, loc='upper right')
    else:
        ax.text(.5, .5, 'no AMMM plan logged (masking: ratio)', ha='center',
                va='center', transform=ax.transAxes)
        ax.set_title('AMMM plan')
    ax.set_xlabel('epoch'); ax.grid(alpha=.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print('curve      : {}'.format(out_png))
    return out_png


def report(path: Path, budget: int, ckpt_s: float,
           grid_t: int, ckpt_freq: int):
    """Print one log's report; returns ``(rc, rows, steps)`` for plotting."""
    parsed = parse_log(path)
    epochs = parsed['epochs']
    if not epochs:
        print('{}: no epoch lines found'.format(path), file=sys.stderr)
        return 1, [], None

    rows = [dict(epoch=e, **epochs[e]) for e in sorted(epochs)]
    steps = next((r['steps'] for r in reversed(rows) if 'steps' in r), None)

    print(f'\n=== {path} ===')
    if steps:
        print(f'steps/epoch: {steps}'
              + (f' (LR schedule: {parsed["sched_steps"]} total)' if parsed['sched_steps'] else ''))
    else:
        print('steps/epoch: unknown (no progress line parsed)')

    present = [k for k in KEYS if any(k in r for r in rows)]
    head = ('{:>4} {:>8} {:>7} {:>7} '.format('ep', 'sec', 's/it', 'data')
            + ' '.join('{:>12}'.format(k) for k in present))
    print(head)
    print('-' * len(head))
    for r in rows:
        cells = []
        for k in present:
            cells.append('{:>12}'.format('-' if k not in r else '{:.4f}'.format(r[k])))
        print('{:>4} {:>8} {:>7} {:>7} '.format(
            r['epoch'],
            '-' if 'total_s' not in r else '{:.0f}'.format(r['total_s']),
            '-' if 's_it' not in r else '{:.3f}'.format(r['s_it']),
            '-' if 'data_s_it' not in r else '{:.3f}'.format(r['data_s_it']))
            + ' '.join(cells))

    timed = [r['total_s'] for r in rows if 'total_s' in r]
    if timed:
        steady = timed[1:] or timed           # epoch 0 includes cache warm-up
        per = statistics.median(steady)
        writes = budget // max(1, ckpt_freq) + 1     # +1 = the final checkpoint
        wall = budget * per + writes * ckpt_s
        print('\nepoch time : median {:.1f} s ({:.2f} min) over '
              '{} epoch(s); first epoch {:.0f} s'.format(
                  per, per / 60, len(steady), timed[0]))
        print('projection : {} epochs x {:.0f} s + {} ckpt write(s) x {:.0f} s'
              ' = {:.2f} h per fold  ({:.1f} h for 5 folds)  '
              '[assumed --save_ckpt_freq {}]'.format(
                  budget, per, writes, ckpt_s, wall / 3600, 5 * wall / 3600,
                  ckpt_freq))
        if steps:
            # epoch total / steps is exact; the printed ``time:`` meter is a
            # sliding window and its LAST line of an epoch may cover 1 it.
            print(f'per-step   : {per / steps:.3f} s/it  (ratio twin measured '
                  f'0.31-0.34 s/it on 4x RTX PRO 6000, cache ON)')
    else:
        print('\nepoch time : no "Total time" line yet (run still in epoch 0?)')

    print('\nconvergence (last decile vs the decile before; '
          'negative = still falling):')
    for key in ('loss', 'mse_tir', 'mse_resp'):
        d, k = _tail_delta(rows, key)
        if d is None:
            print('  {:<9}: not enough epochs (need >= 10)'.format(key))
            continue
        vals = [r[key] for r in rows if key in r]
        best = min(range(len(vals)), key=vals.__getitem__)
        best_ep = [r['epoch'] for r in rows if key in r][best]
        if d < -0.02:
            verdict = 'STILL DESCENDING -> extend the budget'
        elif d > 0.02:
            verdict = 'RISING -> check lr / the loss weights'
        else:
            verdict = 'FLAT within the schedule -> budget is enough'
        print('  {:<9}: {:+.2f} % ({} vs {} ep)   best {:.4f} @ ep {}   {}'.format(
            key, 100.0 * d, k, k, vals[best], best_ep, verdict))

    # AMMM sanity: n_batch = kept_tir*grid_t + kept_resp. Both sides are
    # WINDOW/EPOCH MEANS here, so the identity only holds up to the smoothing
    # error (a 0.05 error in the kept_tir mean is 2.5 tokens) -- compare with a
    # tolerance, or a correct run is flagged every epoch.
    tol = 0.15 * grid_t
    bad = [r['epoch'] for r in rows
           if all(k in r for k in ('n_batch', 'kept_tir', 'kept_resp'))
           and abs(r['n_batch']
                   - (r['kept_tir'] * grid_t + r['kept_resp'])) > tol]
    if bad:
        print('\n[warn] n_batch != kept_tir*{} + kept_resp (tol {:.1f}) on '
              'epochs {} -- the plan/gather wiring may be wrong'
              .format(grid_t, tol, bad[:5]))
    if any('kept_tir' in r for r in rows):
        kt = [r['kept_tir'] for r in rows if 'kept_tir' in r]
        kr = [r['kept_resp'] for r in rows if 'kept_resp' in r]
        if kr:
            print('\nAMMM plan  : kept_tir {:.1f}..{:.1f} of 36 spatial units, '
                  'kept_resp {:.1f}..{:.1f} of 50 tokens (draws vary per step '
                  'by design)'.format(min(kt), max(kt), min(kr), max(kr)))

    print('\nNOTE: Stage-2 loss is a MONITOR, not a metric (no val split, no eval'
          '\nloop) and under AMMM its value moves with the per-step budget draw.'
          '\nJudge the encoder by the Stage-3 subject-level pearson_band.')
    return 0, rows, steps


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='+', help='log files or log directories')
    ap.add_argument('--epochs', type=int, default=150,
                    help='epoch budget to project the wall clock for (default 150)')
    ap.add_argument('--ckpt_seconds', type=float, default=20.0,
                    help='cost of one checkpoint write (measured ~21 s / 1.06 GB)')
    ap.add_argument('--save_ckpt_freq', type=int, default=10,
                    help='checkpoint every N epochs (from the config)')
    ap.add_argument('--grid_t', type=int, default=50,
                    help='video tubelets per clip (Gt): 50 at clip 8 s / stride 2 '
                         '/ tubelet_t 2')
    # ---- AMMM policy knobs: what the mask-ratio histograms are drawn from ----
    ap.add_argument('--tir_units', type=int, default=36,
                    help='TIR spatial units (Gh*Gw); 36 at input_size 96')
    ap.add_argument('--resp_units', type=int, default=50,
                    help='RESP tokens (seq_len/sig_kernel); 50 at 8 s / fs 100 '
                         '/ sig_kernel 16')
    ap.add_argument('--video_kept_cap', type=float, default=0.25)
    ap.add_argument('--physio_kept_cap', type=float, default=1.0)
    ap.add_argument('--dirichlet_alpha', type=float, default=1.0)
    ap.add_argument('--min_kept', type=int, default=0)
    ap.add_argument('--mask_draws', type=int, default=20000,
                    help='plan draws behind each mask-ratio histogram')
    ap.add_argument('--plot', action='store_true',
                    help='write one <log>_training_curves.png per log (Stage 2 '
                         'itself writes no figure)')
    ap.add_argument('--plot_dir', default='',
                    help='directory for the PNGs (default: beside each log)')
    a = ap.parse_args(argv)

    files = _iter_logs(a.logs)
    if not files:
        print('no log files found', file=sys.stderr)
        return 2

    mask_ratios = None
    policy = None
    if a.plot:
        try:
            mask_ratios = mask_ratio_samples(
                tir_units=a.tir_units, resp_units=a.resp_units,
                video_cap=a.video_kept_cap, physio_cap=a.physio_kept_cap,
                alpha=a.dirichlet_alpha, min_kept=a.min_kept, draws=a.mask_draws)
            policy = {'alpha': a.dirichlet_alpha,
                      'caps': {'tir': a.video_kept_cap,
                               'resp': a.physio_kept_cap}}
        except Exception as exc:                 # noqa: BLE001 - plotting aid
            print('[warn] mask-ratio histogram unavailable ({}: {})'.format(
                type(exc).__name__, exc), file=sys.stderr)

    units = {'tir': a.tir_units, 'resp': a.resp_units}
    rc = 0
    for path in files:
        code, rows, steps = report(path, a.epochs, a.ckpt_seconds,
                                   a.grid_t, a.save_ckpt_freq)
        rc |= code
        if a.plot and rows:
            name = '{}_training_curves.png'.format(path.stem)
            out = (Path(a.plot_dir) / name) if a.plot_dir else path.with_name(name)
            plot_log(path, rows, steps, out, mask_ratios=mask_ratios,
                     units=units, policy=policy)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
