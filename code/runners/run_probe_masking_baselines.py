"""Criterion-3a floor probe: can a NO-VIDEO predictor solve the respiration gap?

ADR-017 §3 Criterion 3a. Data only, no weights, no GPU: for each admissible
Mode-2 kept count ``K_resp`` and each placement rule, hide the respiration
tokens exactly as the masker would and score two predictors that see ONLY the
visible respiration samples:

* **interp** — linear interpolation over the visible samples
  (constant-clamped at the clip edges, ADR-017 §2.1 lets a gap touch an edge);
* **const** — the constant-zero predictor, i.e. the per-clip mean once the
  pipeline's per-clip z-score has been applied.

Pass condition: under ``masked_span`` the interpolator must be WORSE than the
constant for every ``K`` (``mse_interp > mse_const``). Sanity check: under the
default ``scattered`` placement it must be FAR better. If 3a fails, widen the
bounds and repeat — do NOT train (ADR-017 §3).

The run must be on the fold's **validation** role: the gate informs the design,
and ADR-001 reserves ``test`` for the five-fold headline. ``--split val`` reads
``protocol.load_plan(<partition>).roles['val']``; there is no shortcut, the
existing ``_held_out_subjects`` helper only returns ``(test, train)``.

Usage (from ``code/``). Only the checkpoint's ARGS are read, so the ADR-008 arm
— which exists before Criterion 3a is settled — is the natural source. It is a
``scattered`` run and therefore records no gap bounds, so pass the grid::

    python -u runners/run_probe_masking_baselines.py \
        --checkpoint ../output/pretrain/\
stage2_budget_tir_resp_2stream_adr008_fold0/checkpoints/checkpoint-0149.pth \
        --partition <exp_root>/fold0/partition.json --split val \
        --gap_grid 63:125,63:150,75:150 --out /tmp/adr017_floor.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

from core.budget_masking import (SIGNAL, VISUAL, ConstantBudgetApportioner,  # noqa: E402,E501
                                 StreamSpec)
from core.mask_placement import (MASKED_SPAN, SCATTERED, SPAN,  # noqa: E402
                                 place_signal, validate_physio_placement)
from core.multimae import (_SIGNAL_STREAMS, _VISUAL_STREAMS,  # noqa: E402
                           _parse_int_csv)
from runners.run_eval_recon import (build_eval_dataset, _interp_from_visible,  # noqa: E402,E501
                                   _pearson)


def _recorded_args(path: str):
    """The checkpoint's ARGS only: the tensors are memory-mapped, not read."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False,
                          mmap=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'args' not in ckpt:
        raise SystemExit(f'{path}: not a Stage-2 checkpoint (no \'args\')')
    return ckpt['args']


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def get_args(argv=None):
    p = argparse.ArgumentParser(
        'ADR-017 Criterion-3a floor probe (no-video interpolation vs constant)')
    p.add_argument('--checkpoint', required=True,
                   help='Stage-2 checkpoint used ONLY for its recorded args: '
                        'geometry (seq_len / sig_kernel / fs), the masking '
                        'policy tuple and therefore the Mode-2 count set. No '
                        'weights are loaded and nothing runs on the GPU.')
    p.add_argument('--data_checkpoint', default='',
                   help='read the DATASET args from this checkpoint instead '
                        '(the clip inventory then matches that run).')
    p.add_argument('--partition', required=True,
                   help='the fold partition record (<exp_root>/fold<k>/'
                        'partition.json).')
    p.add_argument('--split', default='val', choices=('val', 'test', 'corpus'),
                   help="role to score. 'val' (default) is what Criterion 3a "
                        'asks for; \'corpus\' pools every clip (diagnostic '
                        'only, no role claim).')
    p.add_argument('--repeats', default=5, type=int,
                   help='independent placement draws per clip per K (>= 5, '
                        'Criterion 3a)')
    p.add_argument('--max_clips', default=0, type=int,
                   help='cap the clip count (0 = all); drawn with --seed')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--span_seconds', default=4.0, type=float,
                   help='legacy `span` comparator: kept-block length in '
                        'seconds (converted to tokens with the geometry)')
    p.add_argument('--gap_min_tokens', default=-1, type=int,
                   help='masked_span g_min; -1 = the checkpoint record')
    p.add_argument('--gap_max_tokens', default=-1, type=int,
                   help='masked_span g_max; -1 = the checkpoint record')
    p.add_argument('--gap_grid', default='',
                   help='extra masked_span points to sweep, "gmin:gmax,..." '
                        '(the plan Phase-5 sweep). Empty = only the '
                        'checkpoint pair.')
    p.add_argument('--min_spread', default=1e-8, type=float,
                   help='skip clips whose respiration target has a smaller '
                        'spread than this (a flat clip has no correlation to '
                        'measure). Skipped clips are COUNTED and reported, '
                        'never silently dropped.')
    p.add_argument('--out', default='',
                   help='write the JSON report here (default: stdout only)')
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# geometry / policy
# --------------------------------------------------------------------------- #
def _apportioner_from_args(args) -> Tuple[ConstantBudgetApportioner, int]:
    """Rebuild the run's apportioner from its recorded args (no weights).

    Mirrors ``multimae.build_pretraining_model`` EXACTLY — the temporal
    decimation, the tubelet floor, the kept-duration ``seq_len`` derivation and
    the CSV parsing of the ``A_core`` bounds — because one mismatch moves
    ``grid_t`` and therefore the Mode-2 count set (ADR-017 D4).
    """
    tubelet = _parse_int_csv(getattr(args, 'tubelet', '2,16,16'))
    fps = float(getattr(args, 'fps', 25.0))
    fs = float(getattr(args, 'fs', 100.0))
    stride = max(1, int(getattr(args, 'temporal_stride', 1) or 1))
    frames = max(1, int(round(float(args.clip_duration) * fps / stride)))
    frames -= frames % tubelet[0]          # Conv3d partitions time in t frames
    frames = max(tubelet[0], frames)
    kept_seconds = frames * stride / fps
    seq_len = int(getattr(args, 'seq_len', 0) or 0) or max(
        1, int(round(kept_seconds * fs)))
    n_signal = seq_len // int(args.sig_kernel)
    grid_t = frames // tubelet[0]
    gh = int(args.input_size) // tubelet[1]
    gw = int(args.input_size) // tubelet[2]
    specs = []
    for name in [s.strip() for s in str(args.streams).split(',') if s.strip()]:
        if name in _VISUAL_STREAMS:
            specs.append(StreamSpec(name, VISUAL, gh * gw,
                                    tokens_per_unit=grid_t,
                                    spatial_grid=(gh, gw)))
        elif name in _SIGNAL_STREAMS:
            specs.append(StreamSpec(name, SIGNAL, n_signal, tokens_per_unit=1))
        else:
            raise SystemExit(
                f'[floor] unknown stream {name!r} in the checkpoint args '
                f'(--streams {args.streams!r})')
    app = ConstantBudgetApportioner(
        specs,
        n_budget=int(getattr(args, 'n_budget', 350)),
        video_cap_ratio=float(getattr(args, 'video_cap_ratio', 0.25)),
        physio_cap_ratio=float(getattr(args, 'physio_cap_ratio', 0.5)),
        video_min_tubes=int(getattr(args, 'video_min_tubes', 2)),
        p_virtual_sensing=float(getattr(args, 'p_virtual_sensing', 0.5)),
        virtual_sensing_warmup_p=float(
            getattr(args, 'virtual_sensing_warmup_p', 0.1)),
        virtual_sensing_warmup_epochs=int(
            getattr(args, 'virtual_sensing_warmup_epochs', 0)),
        single_carrier_tubes=int(getattr(args, 'single_carrier_tubes', 0) or 0),
        p_single_modality=float(getattr(args, 'p_single_modality', 0.0) or 0.0),
        single_modality_warmup_p=float(
            getattr(args, 'single_modality_warmup_p', 0.0) or 0.0),
        spatial_sampling=str(getattr(args, 'spatial_sampling', 'uniform')),
        core_row_bounds=_parse_int_csv(
            getattr(args, 'core_row_bounds', (2, 3))),
        core_col_bounds=_parse_int_csv(
            getattr(args, 'core_col_bounds', (2, 3))),
        core_tubes_min=int(getattr(args, 'core_tubes_min', 1)),
        core_tubes_max=int(getattr(args, 'core_tubes_max', 2)),
        periph_min_tubes=int(getattr(args, 'periph_min_tubes', 1)),
        physio_draw_min=int(getattr(args, 'physio_draw_min', 0)))
    return app, n_signal


# --------------------------------------------------------------------------- #
# one placement -> the two predictors' scores
# --------------------------------------------------------------------------- #
def _score(target_tokens: np.ndarray, kept_idx: torch.Tensor,
           sig_kernel: int) -> Dict[str, float]:
    """Masked-position MSE / r of the interp and constant predictors."""
    n_tok = int(target_tokens.shape[0])
    keep = np.zeros(n_tok, dtype=bool)
    keep[kept_idx.numpy()] = True
    masked_tokens = ~keep
    if not masked_tokens.any():
        return {}
    masked = np.repeat(masked_tokens, sig_kernel)
    truth = target_tokens.reshape(-1)
    interp = _interp_from_visible(truth, masked)
    t = truth[masked]
    out = {
        'n_masked': int(masked.sum()),
        'mse_interp': float(((interp[masked] - t) ** 2).mean()),
        'r_masked_interp': _pearson(interp[masked], t),
        'mse_const': float((t ** 2).mean()),
    }
    return out


def _placements(k: int, n_tokens: int, repeats: int, mode: str, gen,
                span_tokens: Optional[int] = None,
                gap_min: Optional[int] = None,
                gap_max: Optional[int] = None) -> torch.Tensor:
    return place_signal(k, n_tokens, generator=gen, mode=mode,
                        span_tokens=span_tokens,
                        gap_min_tokens=gap_min, gap_max_tokens=gap_max,
                        batch_size=repeats)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    a = get_args(argv)
    if int(a.repeats) < 5:
        raise SystemExit('[floor] --repeats must be >= 5 (ADR-017 Criterion 3a)')

    cargs = _recorded_args(a.checkpoint)
    data_cargs = _recorded_args(a.data_checkpoint) if a.data_checkpoint \
        else cargs
    app, n_tokens = _apportioner_from_args(cargs)
    # Criterion 3a scores the RESPIRATION stream: derive its declared name
    # rather than assuming one, and refuse a checkpoint that does not carry it.
    sig_name = app.signal[0].name
    if sig_name != 'resp':
        raise SystemExit(
            '[floor] this probe scores the respiration stream (Criterion 3a); '
            f'the checkpoint declares {sig_name!r} as its physiological '
            'stream.')
    counts = app.mode2_counts(sig_name)
    sig_kernel = int(cargs.sig_kernel)

    from protocol import load_plan
    plan = load_plan(a.partition)
    held = set(plan.roles[a.split]) if a.split in plan.roles else None
    print(f'[floor] role      : {a.split} -> {len(held) if held else "corpus"}'
          f' subject(s) from {os.path.basename(a.partition)}')
    print(f'[floor] geometry  : N_s={n_tokens} tokens x {sig_kernel} samples '
          f'(fs={float(cargs.fs):g} Hz); K_resp={list(counts)} '
          f'(apportioner-derived)')

    # ---- the gap grid is validated BEFORE the dataset is touched ----------
    g_min = int(a.gap_min_tokens) if int(a.gap_min_tokens) >= 0 \
        else int(getattr(cargs, 'physio_gap_min_tokens', 0) or 0)
    g_max = int(a.gap_max_tokens) if int(a.gap_max_tokens) >= 0 \
        else int(getattr(cargs, 'physio_gap_max_tokens', 0) or 0)
    span_tokens = max(1, int(round(float(a.span_seconds) * float(cargs.fs)
                                   / sig_kernel)))
    points: List[Tuple[int, int]] = []
    if g_min >= 1 and g_max >= g_min:
        points.append((g_min, g_max))
    for item in str(a.gap_grid).split(','):
        item = item.strip()
        if not item:
            continue
        lo, hi = (int(x) for x in item.split(':'))
        if (lo, hi) not in points:
            points.append((lo, hi))
    if not points:
        raise SystemExit(
            '[floor] the checkpoint records no gap bounds; pass '
            '--gap_min_tokens/--gap_max_tokens or --gap_grid.')
    for lo, hi in points:
        try:
            validate_physio_placement(
                mode=MASKED_SPAN, n_tokens=n_tokens, gap_min_tokens=lo,
                gap_max_tokens=hi, n_streams=1, kept_counts=counts)
        except ValueError as exc:
            raise SystemExit(
                f'[floor] gap grid point {lo}:{hi} is infeasible for this '
                f'geometry: {exc}') from None

    ds, idx = build_eval_dataset(data_cargs, a, held)
    print(f'[floor] clips     : {len(idx)}')

    # one row per (rule, K, placement point) with the per-clip score lists
    rules: List[Tuple[str, Dict]] = [('scattered', dict(mode=SCATTERED)),
                                     ('span', dict(mode=SPAN,
                                                   span_tokens=span_tokens))]
    for lo, hi in points:
        rules.append((f'masked_span[{lo},{hi}]',
                      dict(mode=MASKED_SPAN, gap_min=lo, gap_max=hi)))

    skipped_flat = 0
    acc: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    gen = torch.Generator().manual_seed(int(a.seed))
    for i in idx:
        target = ds[i]['resp'].detach().float().reshape(n_tokens, sig_kernel)
        if float(target.std()) < float(a.min_spread):
            skipped_flat += 1
            continue
        t_np = target.numpy()
        for name, kw in rules:
            for k in counts:
                rows = []
                kept = _placements(k, n_tokens, int(a.repeats), gen=gen, **kw)
                for r in range(int(a.repeats)):
                    got = _score(t_np, kept[r], sig_kernel)
                    if got:
                        rows.append(got)
                if not rows:
                    continue
                slot = acc.setdefault((name, k), {
                    'mse_interp': [], 'r_masked_interp': [], 'mse_const': []})
                for key in slot:
                    slot[key].append(float(np.nanmean([r[key] for r in rows])))
    if skipped_flat:
        print(f'[floor] skipped {skipped_flat} clip(s) with |spread| < '
              f'{a.min_spread:g} (flat target: no correlation to measure)')

    # ---- report -----------------------------------------------------------
    print()
    print('=' * 78)
    print(f'{"rule":22s} {"K_resp":>6s} {"n_clips":>7s} '
          f'{"mse_interp":>11s} {"mse_const":>10s} {"r_interp":>9s}  verdict')
    verdicts: Dict[str, Dict[str, object]] = {}
    for name, _kw in rules:
        for k in counts:
            slot = acc.get((name, k))
            if not slot:
                continue
            mi = float(np.nanmean(slot['mse_interp']))
            mc = float(np.nanmean(slot['mse_const']))
            ri = float(np.nanmean(slot['r_masked_interp']))
            beats = mi < mc
            print(f'{name:22s} {k:6d} {len(slot["mse_const"]):7d} '
                  f'{mi:11.4f} {mc:10.4f} {ri:+9.4f}  '
                  f'{"interp BEATS const" if beats else "interp loses"}')
            verdicts[f'{name}|K={k}'] = {
                'n_clips': len(slot['mse_const']),
                'mse_interp': mi, 'mse_const': mc,
                'r_masked_interp': ri,
                'interp_beats_const': bool(beats),
            }

    passed_points = {}
    _san = [k for k in counts if f'scattered|K={k}' in verdicts]
    sanity = bool(_san) and all(
        verdicts[f'scattered|K={k}']['interp_beats_const'] for k in _san)
    for lo, hi in points:
        key = f'masked_span[{lo},{hi}]'
        checked = [k for k in counts if f'{key}|K={k}' in verdicts]
        # ``bool(checked)``: every clip skipped as flat would otherwise make
        # ``all([])`` a vacuous PASS.
        ok = bool(checked) and all(
            not verdicts[f'{key}|K={k}']['interp_beats_const']
            for k in checked)
        passed_points[(lo, hi)] = bool(ok)
    passed = any(passed_points.values())
    for lo, hi in points:
        print(f'  point {lo}:{hi}  -> '
              f'{"PASS" if passed_points[(lo, hi)] else "FAIL"}')
    print()
    print(f'Criterion 3a: {"PASS" if passed else "FAIL"} '
          f'(at least one swept point has interp worse than const for every K)')
    print(f'Sanity     : {"PASS" if sanity else "FAIL"} '
          f'(interp far better under scattered)')
    if not passed:
        print('Do NOT train: widen --gap_grid and repeat (ADR-017 §3).')
    else:
        print('Next: freeze the PASSING point in the arm config (plan Phase 6).')

    if a.out:
        payload = {'checkpoint': a.checkpoint, 'partition': a.partition,
                   'split': a.split, 'seed': int(a.seed),
                   'repeats': int(a.repeats), 'n_tokens': n_tokens,
                   'sig_kernel': sig_kernel,
                   'k_resp_counts': list(counts),
                   'span_tokens': span_tokens,
                   'clips': len(idx), 'clips_skipped_flat': skipped_flat,
                   'verdicts': verdicts,
                   'point_passed': {f'{lo}:{hi}': ok
                                    for (lo, hi), ok in passed_points.items()},
                   'criterion_3a_passed': bool(passed),
                   'sanity_scattered_passed': bool(sanity)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f'[floor] wrote {a.out}')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
