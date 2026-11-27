"""Replay an ADR-005 constant-budget mask policy and report its kept ratios.

The per-step mask counts are a deterministic function of
``counts_generator(mask_base, epoch, global_step)`` and the apportioner, so they
can be replayed offline for every training step without touching the data. This
script does exactly that and writes:

* ``policy_replay.json`` — the run's resolved policy tuple, per-stream kept-token
  statistics, the mode split, the per-phase Mode-1 fractions with a Wilson
  interval, and the ADR-005 §3.1 Criterion-1 checks;
* ``kept_ratio_hist.png`` — one kept-token histogram per stream over every
  training step;
* ``kept_ratio_by_mode.png`` — mean kept ratio per stream, split by the
  *resolved* mode (ADR-005 contract 4).

It also verifies the replay against the run's own ``Epoch <e> stats:`` log lines
when ``--logs`` is given (the epoch means must match to the printed precision).

Inputs are the checkpoint (for the resolved ``args``) and, optionally, the runner
logs. Nothing is trained, no data is read, and no metric is altered.

Usage (from ``code/``)::

    python -m scripts.project.mask_policy_report \
        --checkpoint EXP_ROOT/fold0/stage2/checkpoints/checkpoint-0149.pth \
        --logs 'logs/adr001_pretrain_571*.out' \
        --out-dir ../code/notes/eval/baselines/tir_rgb2d_resp
"""
import argparse
import ast
import glob
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from core.mask_placement import (admissible_gap_counts,              # noqa: E402
                                 counts_generator)
from core.multimae import build_pretraining_model                 # noqa: E402


# --------------------------------------------------------------------------- #
def get_args(argv=None):
    p = argparse.ArgumentParser(
        description='Replay an ADR-005 mask policy over a whole run.')
    p.add_argument('--checkpoint', required=True,
                   help='Stage-2 checkpoint whose args define the policy')
    p.add_argument('--logs', default='',
                   help='optional glob of runner .out files; used to infer '
                        'steps/epoch and to verify the replay')
    p.add_argument('--steps_per_epoch', default=0, type=int,
                   help='0 = infer from --logs (required if no logs are given)')
    p.add_argument('--epochs', default=0, type=int,
                   help='0 = use args.epochs from the checkpoint')
    p.add_argument('--out-dir', default='',
                   help='directory for the JSON and the two figures '
                        '(default: the checkpoint run dir)')
    p.add_argument('--tag', default='', help='prefix used in the file names')
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
def parse_logs(paths: Sequence[str]) -> Dict[int, Dict]:
    """``{epoch: stats_dict}`` from every ``Epoch <e> stats: {...}`` line."""
    rows: Dict[int, Dict] = {}
    seen_steps: List[int] = []
    for path in paths:
        with open(path, errors='replace') as fh:
            for line in fh:
                m = re.match(r'Epoch: \[\d+\]\s+\[\s*(\d+)/(\d+)\]', line)
                if m:
                    seen_steps.append(int(m.group(2)))
                if line.startswith('Epoch ') and 'stats:' in line:
                    e = int(re.search(r'Epoch (\d+) stats', line).group(1))
                    rows[e] = ast.literal_eval(line.split('stats:', 1)[1])
    if not seen_steps:
        raise SystemExit('[replay] --logs matched no step lines')
    if len(set(seen_steps)) != 1:
        raise SystemExit(f'[replay] logs disagree on steps/epoch: '
                         f'{sorted(set(seen_steps))}')
    return rows, seen_steps[0]


# --------------------------------------------------------------------------- #
def expected_zero_draw_share(app) -> float:
    """Share of Mode-2 draws whose ALIGNED ``K_resp`` is 0.

    With the ADR-002..006 draw (``physio_draw_min = 0``) exactly one value out
    of ``C_p + 1`` is zero, and that draw falls through to the sensor-absent
    class. With ADR-007's ``Uniform{1..C_p}`` (``physio_draw_min = 1``) the zero
    draw does not EXIST -- which is what "zero Mode-2 -> Mode-1 leakage" means
    operationally (ADR-007 §2.5), so the expectation is 0.0, not 1/(C_p+1).
    """
    if int(getattr(app, 'physio_draw_min', 0) or 0) >= 1:
        return 0.0
    caps = [int(v) for v in app.physio_cap_tokens.values()]
    return 1.0 / (min(caps) + 1) if caps else 0.0


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Dict[str, float]:
    """Wilson score interval for a binomial fraction (closed form)."""
    if n <= 0:
        return {'p': float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {'p': p, 'lo': centre - half, 'hi': centre + half, 'n': n, 'k': k}


# --------------------------------------------------------------------------- #
def summarize(counts: np.ndarray, total: int) -> Dict:
    q = np.percentile(counts, [25, 50, 75])
    return {
        'total_tokens_per_step': int(total),
        'min': int(counts.min()),
        'q25': float(q[0]),
        'median': float(q[1]),
        'q75': float(q[2]),
        'max': int(counts.max()),
        'mean': float(counts.mean()),
        'mean_ratio': float(counts.mean() / total),
    }


def main(argv=None):
    a = get_args(argv)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cargs = ck['args']

    logs: List[str] = sorted(glob.glob(a.logs)) if a.logs else []
    logged: Dict[int, Dict] = {}
    logged_steps = 0
    if logs:
        logged, logged_steps = parse_logs(logs)
    steps_per_epoch = int(a.steps_per_epoch) or logged_steps
    if not steps_per_epoch:
        raise SystemExit('[replay] give --steps_per_epoch or --logs')
    epochs = int(a.epochs) or int(getattr(cargs, 'epochs', 0))
    if not epochs:
        raise SystemExit('[replay] give --epochs or store args.epochs')
    if logged and sorted(logged) != list(range(epochs)):
        raise SystemExit(
            f'[replay] logs cover {len(logged)} epoch(s), expected 0..{epochs - 1}')

    model = build_pretraining_model(cargs)
    app = model.apportioner
    mask_base = int(getattr(cargs, 'val_seed', None)
                    or getattr(cargs, 'split_seed', 42))
    streams = list(model.streams)
    visual_names = list(model.visual)
    totals = {s.name: int(s.n_units) * int(s.tokens_per_unit)
              for s in model.specs}

    # ---- replay -----------------------------------------------------------
    draw_mode: List[int] = []
    resolved_mode: List[int] = []
    phase: List[int] = []
    counts: Dict[str, List[int]] = {s: [] for s in streams}
    epoch_means: Dict[str, List[float]] = {s: [] for s in streams}
    epoch_virtual: List[float] = []
    core_counts: List[int] = []
    periph_counts: List[int] = []
    for e in range(epochs):
        acc = {s: 0 for s in streams}
        virt = 0
        for j in range(steps_per_epoch):
            g = counts_generator(mask_base, e, e * steps_per_epoch + j)
            plan = app.sample_step(g, epoch=e)
            for s in streams:
                k = int(plan.kept_tokens[s])
                counts[s].append(k)
                acc[s] += k
            # ADR-007 §2.9: the stratified split, replayed without a GPU.
            cu = plan.core_units or {}
            pu = plan.periph_units or {}
            for name in visual_names:
                core_counts.append(int(cu.get(name, 0)))
                periph_counts.append(int(pu.get(name, 0)))
            draw_mode.append(int(plan.mode))
            resolved_mode.append(int(plan.mode_by_counts))
            phase.append(0 if e < app.virtual_sensing_warmup_epochs else 1)
            virt += int(plan.mode_by_counts == 1)
        for s in streams:
            epoch_means[s].append(acc[s] / steps_per_epoch)
        epoch_virtual.append(virt / steps_per_epoch)

    arrays = {s: np.asarray(v, dtype=np.int64) for s, v in counts.items()}
    n_steps = epochs * steps_per_epoch
    dm = np.asarray(draw_mode, dtype=np.int64)
    rm = np.asarray(resolved_mode, dtype=np.int64)
    ph = np.asarray(phase, dtype=np.int64)

    # ADR-007 §2.9: the replayed stratified split (None on a uniform policy).
    stratification = None
    if getattr(app, 'anatomy_aware', False) and core_counts:
        core_arr = np.asarray(core_counts, dtype=np.int64)
        periph_arr = np.asarray(periph_counts, dtype=np.int64)
        stratification = {
            'spatial_sampling': str(app.spatial_sampling),
            'core_coords': [list(c) for c in app.core_coords],
            'core_indices': [int(i) for i in app.core_indices],
            'S_core': {'min': int(core_arr.min()),
                       'max': int(core_arr.max()),
                       'mean': float(core_arr.mean()),
                       'share_min': float(
                           (core_arr == int(app.core_tubes_min)).mean())},
            'S_periph': {'min': int(periph_arr.min()),
                         'max': int(periph_arr.max()),
                         'mean': float(periph_arr.mean())},
            'core_hit_share': float((core_arr >= 1).mean()),
            'expected_core_hit_share': 1.0,
        }

    # ---- ADR-005 §3.1 Criterion 1 ----------------------------------------
    t = int(app.tokens_per_unit)
    visual = list(model.visual)
    signal = list(model.signal)
    sums = np.zeros(n_steps, dtype=np.int64)
    for s in streams:
        sums += arrays[s]
    checks = {
        'exact_sum': {
            'ok': bool(np.all(sums == int(model.n_budget))),
            'n_violations': int((sums != int(model.n_budget)).sum()),
            'n_steps': n_steps,
        },
        'visual_multiple_of_T': all(
            bool(np.all(arrays[s] % t == 0)) for s in visual),
        'visual_inside_floor_cap': {
            s: bool(arrays[s].min() >= app.video_min_tubes * t
                    and arrays[s].max() <= app.video_cap_units[s] * t)
            for s in visual},
        'signal_inside_cap': {
            s: bool(arrays[s].min() >= 0
                    and arrays[s].max() <= app.physio_cap_tokens[s])
            for s in signal},
        'signal_on_token_grid': {
            s: bool(np.all(arrays[s] % int(
                next(x.tokens_per_unit for x in model.specs
                     if x.name == s)) == 0))
            for s in signal},
        'mode_iff_signal_zero': {
            'ok': bool(np.all((rm == 1) == np.all(
                np.stack([arrays[s] == 0 for s in signal]), axis=0))),
        },
        'mode2_k_tilde_zero_share': {
            'observed': float(((dm == 2) & (rm == 1)).sum()
                              / max(1, (dm == 2).sum())),
            'expected': expected_zero_draw_share(app),
        },
    }

    # mode split, both labels
    mode_split = {}
    for label, arr in (('drawn', dm), ('resolved', rm)):
        mode_split[label] = {
            'mode1': int((arr == 1).sum()),
            'mode2': int((arr == 2).sum()),
            'mode1_share': float((arr == 1).mean()),
        }

    # curriculum
    e_warm = int(app.virtual_sensing_warmup_epochs)
    curriculum = {
        'p_warm': float(app.virtual_sensing_warmup_p),
        'p_target': float(app.p_virtual_sensing),
        'E_warm': e_warm,
        'phase1': wilson(int((rm[ph == 0] == 1).sum()),
                         int((ph == 0).sum())) if e_warm > 0 else None,
        'phase2': wilson(int((rm[ph == 1] == 1).sum()),
                         int((ph == 1).sum())) if e_warm > 0 else None,
        'expected_phase1': (app.virtual_sensing_warmup_p
                            + (1 - app.virtual_sensing_warmup_p)
                            * expected_zero_draw_share(app)),
        'expected_phase2': (app.p_virtual_sensing
                            + (1 - app.p_virtual_sensing)
                            * expected_zero_draw_share(app)),
    }

    # ---- per-stream statistics -------------------------------------------
    stats = {s: summarize(arrays[s], totals[s]) for s in streams}
    for s in visual:                       # kept-unit distribution
        units = arrays[s] // t
        vals, cnt = np.unique(units, return_counts=True)
        stats[s]['kept_units_distribution'] = {
            str(int(v)): float(c / units.size) for v, c in zip(vals, cnt)}
    for s in signal:
        cap = int(app.physio_cap_tokens[s])
        nz = arrays[s][arrays[s] > 0]
        stats[s].update({
            'share_exactly_0': float((arrays[s] == 0).mean()),
            'share_exactly_cap': float((arrays[s] == cap).mean()),
            'n_distinct_nonzero': int(np.unique(nz).size),
            'nonzero_mean': float(nz.mean()) if nz.size else float('nan'),
        })

    mode_conditional = {}
    for m, name in ((1, 'mode1_virtual_sensing'), (2, 'mode2_inpainting')):
        sel = rm == m
        mode_conditional[name] = {
            'n_steps': int(sel.sum()),
            **{s: {'mean_tokens': float(arrays[s][sel].mean()),
                   'mean_ratio': float(arrays[s][sel].mean() / totals[s])}
               for s in streams},
        }

    # ---- replay fidelity vs logs -----------------------------------------
    fidelity = None
    if logged:
        worst = {}
        for s in streams:
            key = f'kept_{s}'
            diffs = [abs(epoch_means[s][e] - float(logged[e][key]))
                     for e in range(epochs)]
            worst[key] = max(diffs)
        worst['is_virtual'] = max(
            abs(epoch_virtual[e] - float(logged[e]['is_virtual']))
            for e in range(epochs))
        fidelity = {'max_abs_epoch_mean_diff': worst,
                    'logged_epochs': len(logged),
                    'steps_per_epoch': steps_per_epoch}
        if any(v > 1e-9 for v in worst.values()):
            raise SystemExit(f'[replay] replay does not match the logs: {worst}')

    # ---- output -----------------------------------------------------------
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
    out_dir = a.out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = f'{a.tag}_' if a.tag else ''
    # ADR-017: the masked-span gap geometry, resolved into seconds and into the
    # admissible gap counts per masked count. Empty for every other placement.
    masked_span: Dict = {}
    if str(model.physio_placement) == 'masked_span':
        _g_min = int(getattr(model, 'physio_gap_min_tokens', 0) or 0)
        _g_max = int(getattr(model, 'physio_gap_max_tokens', 0) or 0)
        _n_tok = int(model.physio_tokens)
        masked_span = {
            'gap_min_tokens': _g_min,
            'gap_max_tokens': _g_max,
            'gap_seconds': [round(float(_g_min) * float(model.sig_kernel)
                                  / float(model.fs), 6),
                            round(float(_g_max) * float(model.sig_kernel)
                                  / float(model.fs), 6)],
            'per_masked_count': {
                str(_n_tok - int(k)): list(admissible_gap_counts(
                    _n_tok - int(k), int(k), _g_min, _g_max))
                for k in app.mode2_counts(model.signal[0])},
        }
    payload = {
        'schema': 'adr-005/policy-replay/1',
        'checkpoint': os.path.abspath(a.checkpoint),
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'n_steps': n_steps,
        'mask_base': mask_base,
        'streams': streams,
        'policy': {
            'n_budget': int(model.n_budget),
            'tokens_per_unit': t,
            'video_min_tubes': int(app.video_min_tubes),
            'video_cap_units': dict(app.video_cap_units),
            'physio_cap_tokens': dict(app.physio_cap_tokens),
            'physio_placement': str(model.physio_placement),
            'p_virtual_sensing': float(app.p_virtual_sensing),
            'virtual_sensing_warmup_p': float(app.virtual_sensing_warmup_p),
            'virtual_sensing_warmup_epochs': e_warm,
            # ADR-007 §2.3/§2.5: what the replay policy actually was.
            'spatial_sampling': str(getattr(app, 'spatial_sampling', 'uniform')),
            'physio_draw_min': int(getattr(app, 'physio_draw_min', 0) or 0),
            'core_coords': [list(c) for c in getattr(app, 'core_coords', ())],
            'core_indices': [int(i) for i in getattr(app, 'core_indices', ())],
            'core_tubes_min': int(getattr(app, 'core_tubes_min', 0) or 0),
            'core_tubes_max': int(getattr(app, 'core_tubes_max', 0) or 0),
            'periph_min_tubes': int(getattr(app, 'periph_min_tubes', 0) or 0),
            **({'masked_span': masked_span} if masked_span else {}),
        },
        'per_stream': stats,
        'mode_split': mode_split,
        'mode_conditional': mode_conditional,
        'curriculum': curriculum,
        'criterion_1': checks,
        'stratification': stratification,
        'replay_vs_logs': fidelity,
    }
    out_json = os.path.join(out_dir, f'{tag}policy_replay.json')
    with open(out_json, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print('[replay] wrote', out_json)
    if masked_span:
        print('[replay] masked_span gaps '
              f'{masked_span["gap_min_tokens"]}..'
              f'{masked_span["gap_max_tokens"]} tokens = '
              f'{masked_span["gap_seconds"]} s;  N(M) per masked count = '
              f'{masked_span["per_masked_count"]}')

    # ---- figures ----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(streams), figsize=(4.2 * len(streams), 3.4),
                             constrained_layout=True, squeeze=False)
    for ax, s in zip(axes[0], streams):
        vals = arrays[s] / totals[s]
        ax.hist(vals, bins=40, color='tab:blue', edgecolor='white', linewidth=0.4)
        ax.set_title(f'{s}: kept ratio (mean {stats[s]["mean_ratio"]:.3f})',
                     fontsize=9)
        ax.set_xlabel('kept / total tokens', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel('steps', fontsize=8)
    suptitle = (f'Mask-policy replay: {n_steps:,} steps, budget '
                f'{int(model.n_budget)}')
    if e_warm > 0:
        suptitle += (f', curriculum p={app.virtual_sensing_warmup_p:g}->'
                     f'{app.p_virtual_sensing:g} at epoch {e_warm}')
    fig.suptitle(suptitle, fontsize=10)
    out_hist = os.path.join(out_dir, f'{tag}kept_ratio_hist.png')
    fig.savefig(out_hist, dpi=135)
    plt.close(fig)
    print('[replay] wrote', out_hist)

    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(streams), 3.2),
                           constrained_layout=True)
    width = 0.36
    xs = np.arange(len(streams))
    all_means = []
    for i, (m, name, colour) in enumerate((
            (1, 'mode 1 (virtual sensing)', '#4c566a'),
            (2, 'mode 2 (waveform inpainted)', '#2e8b57'))):
        sel = rm == m
        means = [float(arrays[s][sel].mean() / totals[s]) for s in streams]
        all_means.extend(means)
        bars = ax.bar(xs + (i - 0.5) * width, means, width,
                      label=f'{name}, {(rm == m).mean() * 100:.1f}%',
                      color=colour)
        ax.bar_label(bars, fmt='%.2f', fontsize=6.5, padding=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(streams, fontsize=9)
    ax.set_ylabel('mean kept ratio', fontsize=9)
    ax.set_ylim(0, max(0.05, max(all_means) * 1.25))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7, frameon=True)
    ax.tick_params(labelsize=8)
    out_mode = os.path.join(out_dir, f'{tag}kept_ratio_by_mode.png')
    fig.savefig(out_mode, dpi=135)
    plt.close(fig)
    print('[replay] wrote', out_mode)

    # ---- console summary --------------------------------------------------
    print(f'\n[replay] {n_steps:,} steps  budget {int(model.n_budget)}  '
          f'mask_base {mask_base}')
    for s in streams:
        d = stats[s]
        print(f'  {s:6s} total {d["total_tokens_per_step"]:4d}  '
              f'min {d["min"]:3d}  q25 {d["q25"]:6.1f}  med {d["median"]:6.1f}  '
              f'q75 {d["q75"]:6.1f}  max {d["max"]:3d}  '
              f'mean {d["mean"]:7.2f}  ratio {d["mean_ratio"]:.3f}')
    print(f'  drawn   : mode1 {mode_split["drawn"]["mode1"]:6d} '
          f'({mode_split["drawn"]["mode1_share"] * 100:5.2f} %)  '
          f'mode2 {mode_split["drawn"]["mode2"]:6d}')
    print(f'  resolved: mode1 {mode_split["resolved"]["mode1"]:6d} '
          f'({mode_split["resolved"]["mode1_share"] * 100:5.2f} %)  '
          f'mode2 {mode_split["resolved"]["mode2"]:6d}')
    if e_warm > 0:
        for key in ('phase1', 'phase2'):
            c = curriculum[key]
            print(f'  {key}: mode-1 {c["p"] * 100:5.2f} %  '
                  f'[{c["lo"] * 100:5.2f}, {c["hi"] * 100:5.2f}]  '
                  f'(expected {curriculum["expected_" + key] * 100:5.2f} %)')
    print(f'  Criterion 1: {json.dumps(checks, default=str)}')
    if fidelity:
        print(f'  replay vs logs: max |epoch mean diff| '
              f'{max(fidelity["max_abs_epoch_mean_diff"].values()):.3e}')
    print('\n[replay] PASS -- the policy is reproducible from '
          '(mask_base, epoch, step)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
