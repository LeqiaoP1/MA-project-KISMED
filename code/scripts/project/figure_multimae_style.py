"""MultiMAE Figure-6-style panel for a Stage-2 checkpoint.

Rows = the checkpoint's streams; columns = the original clip and, per kept
density, [masked input | prediction]. The three densities are stated as KEPT
TOKEN COUNTS that sum to the checkpoint's constant budget (350 on the ADR-002/
ADR-005 arms, 200 on the ADR-007/ADR-008 two-stream arm), because the
architecture asserts an encoder length of exactly `n_budget` -- a "fewer
visible tokens overall" sweep like the paper's is not expressible here, so what
varies is how the fixed budget is allocated.

Three presets exist, selected from the checkpoint's own stream set:

* four streams (`tir`, `rgb2d`, `bp`, `resp`) -- the ADR-002 predecessor arm;
* three streams (`tir`, `rgb2d`, `resp`) -- the ADR-005 arm.  Here the
  physiological cap (125 of 250 tokens) plus the 350-token budget leave at
  least 225 visual tokens, so the sparsest expressible visual column is 4 + 5
tubes, not the four-stream panel's 3 + 3;
* two streams (`tir`, `resp`) -- the ADR-007/ADR-008 arm.  Budget 200, TIR cap
  8 of 36 tubes, resp cap 100 of 250 tokens: the sparsest visual column is
  4 tubes (100 tokens) + the full 100-token respiration cap, and the densest is
  8 tubes with respiration fully hidden (Mode 1 virtual sensing).  This preset
  carries a FOURTH column, the remote-sensing / deployment condition: every TIR
  tube visible (900 tokens, i.e. outside the constant budget by construction)
  and no respiration token -- built with
  ``plan_from_counts(..., exempt_budget=True)``, which states its own encoder
  length instead of spending the budget.

Clip: one sample from the fold's HELD-OUT test subjects (never trained on).

    python -m scripts.project.figure_multimae_style \
        --checkpoint EXP_ROOT/fold0/stage2/checkpoints/checkpoint-0149.pth \
        --partition EXP_ROOT/fold0/partition.json --seed 0
"""
import argparse
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from core.multimae import build_pretraining_model            # noqa: E402
from runners.run_visualize_pretrain import (                 # noqa: E402
    _blank_masked_patches, _held_out_subjects, _to_rgb,
    reconstruct_visual)

#: height of the per-column allocation label, in axes fractions. It must clear
#: the two-line ``masked in / kept N/M`` axes title underneath it; 1.20 does not
#: once the figure is only two rows tall (the panel axes are then larger, so the
#: title is too), so the label sits one more line higher.
COLUMN_LABEL_Y = 1.30

#: (label, kept tokens per stream[, plan_kwargs]). Standard rows MUST sum to
#: the model's n_budget; an optional third element with ``exempt_budget=True``
#: marks an evaluation-only column that deliberately leaves the budget.
#: Keyed by the checkpoint's stream set so one script serves every arm.
DENSITIES = {
    frozenset(('tir', 'rgb2d', 'bp', 'resp')): (
        ('sparse  (3/36 tubes, physio 100/250)', {'tir': 75, 'rgb2d': 75,
                                                  'bp': 100, 'resp': 100}),
        ('mid  (6/36 tubes, physio 25/250)', {'tir': 150, 'rgb2d': 150,
                                              'bp': 25, 'resp': 25}),
        ('dense visual  (7/36 tubes, physio 0)', {'tir': 175, 'rgb2d': 175,
                                                  'bp': 0, 'resp': 0}),
    ),
    frozenset(('tir', 'rgb2d', 'resp')): (
        ('resp-max  (5+4/36 tubes, resp 125/250)', {'tir': 125, 'rgb2d': 100,
                                                    'resp': 125}),
        ('mid  (6/36 tubes, resp 50/250)', {'tir': 150, 'rgb2d': 150,
                                            'resp': 50}),
        ('dense visual  (7/36 tubes, physio 0)', {'tir': 175, 'rgb2d': 175,
                                                  'resp': 0}),
    ),
    # ADR-007/ADR-008 two-stream arm: budget 200, TIR cap 8/36 tubes,
    # respiration cap 100/250 tokens. The first three columns each spend the
    # SAME 200 tokens, shifting the split between the visual stream and the
    # waveform. The FOURTH column is the remote-sensing / deployment condition:
    # the whole TIR clip is visible (36 of 36 tubes = 900 tokens) and NO
    # respiration token reaches the encoder. Because that leaves the constant
    # budget, it is built as an exempt plan (``exempt_budget=True``) with the
    # core draw set to the full anatomical core so stratified placement covers
    # every tube; see ``core.budget_masking.ConstantBudgetApportioner.plan_from_counts``.
    frozenset(('tir', 'resp')): (
        ('resp-max  (4/36 tubes, resp 100/250)', {'tir': 100, 'resp': 100}),
        ('mid  (5/36 tubes, resp 75/250)', {'tir': 125, 'resp': 75}),
        ('dense visual  (8/36 tubes, resp 0)', {'tir': 200, 'resp': 0}),
        ('remote sensing  (36/36 tubes visible, resp 0)',
         {'tir': 900, 'resp': 0},
         {'exempt_budget': True, 'kept_core': 4}),
    ),
}


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--partition', required=True)
    p.add_argument('--out', default='')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--frame', default=-1, type=int)
    p.add_argument('--device', default='cuda')
    return p.parse_args(argv)


def pick_clip(cargs, partition, seed):
    from data import build_pretraining_corpus_dataset
    ds = build_pretraining_corpus_dataset(cargs)
    held, train, src = _held_out_subjects(cargs, partition)
    idx = [i for i in range(len(ds))
           if str(ds.entries[i].get('subject', '')) in held]
    if not idx:
        raise SystemExit('[fig] no clip from the held-out subjects')
    overlap = held & train
    if overlap:
        raise SystemExit(f'[fig] partition contract violated: {sorted(overlap)[:3]}')
    g = torch.Generator().manual_seed(int(seed))
    j = idx[torch.randperm(len(idx), generator=g)[0].item()]
    entry = ds.entries[j]
    item = ds[j]
    print(f'[fig] clip: subject {entry.get("subject")} session '
          f'{entry.get("session")} t_start {entry.get("t_start")} '
          f'({len(idx)} held-out clips to choose from)')
    return {k: v.unsqueeze(0) for k, v in item.items()}, entry


def main(argv=None):
    a = get_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cargs = ck['args']
    model = build_pretraining_model(cargs)
    model.load_state_dict(ck['model'], strict=False)
    model.to(dev).eval()
    budget = int(model.n_budget)
    key = frozenset(model.streams)
    if key not in DENSITIES:
        raise SystemExit(
            f'[fig] no allocation preset for streams {sorted(key)}; known sets: '
            f'{[sorted(k) for k in DENSITIES]}')
    densities = DENSITIES[key]

    batch_cpu, entry = pick_clip(cargs, a.partition, a.seed)
    batch = {k: v.to(dev).float() for k, v in batch_cpu.items()}

    plans = []
    for preset in densities:
        label, counts = preset[0], preset[1]
        plan_kwargs = dict(preset[2]) if len(preset) > 2 else {}
        if set(counts) != set(model.streams):
            raise SystemExit(f'[fig] {label}: allocation keys {sorted(counts)} '
                             f'!= model streams {sorted(model.streams)}')
        if not plan_kwargs.get('exempt_budget'):
            if sum(counts.values()) != budget:
                raise SystemExit(f'[fig] {label}: counts sum to '
                                 f'{sum(counts.values())}, budget is {budget}')
        else:
            # An exempt column only ever means "this stream is FULLY visible", so
            # the stated count has to be that stream's whole token count at THIS
            # checkpoint's geometry -- a stale literal must not silently draw a
            # partial frame and label it 100 %.
            full = {s: int(model.n_visual if s in model.visual
                           else model.n_signal) for s in model.streams}
            for s, v in counts.items():
                if v not in (0, full[s]):
                    raise SystemExit(
                        f'[fig] {label}: exempt column states {s}={v} tokens; '
                        f'it must be 0 or all {full[s]} tokens of this '
                        f'checkpoint')
        plans.append((label, counts,
                      model.plan_from_counts(counts, **plan_kwargs)))

    outs = []
    for i, (label, counts, plan) in enumerate(plans):
        gen = torch.Generator().manual_seed(int(a.seed) + i)
        with torch.inference_mode():
            out = model(batch, plan=plan, placement_generator=gen)
        outs.append(out)
        print(f'[fig] {label}: mode={out["mode"]} kept={out["kept"]}')

    frame = a.frame
    sig_kernel = int(model.sig_kernel)
    rows = list(model.streams)
    ncol = 1 + 2 * len(plans)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.35 * ncol, 2.05 * len(rows)),
                             squeeze=False, constrained_layout=True)

    for r, s in enumerate(rows):
        is_visual = s in model.visual
        # ---- original -------------------------------------------------
        ax = axes[r][0]
        if is_visual:
            raw = batch[s]
            fi = raw.shape[2] // 2 if frame < 0 else min(frame, raw.shape[2] - 1)
            ax.imshow(_to_rgb(raw[0, :, fi].detach().cpu()))
            ax.set_title('original\nframe %d' % fi, fontsize=7.5)
        else:
            tgt = batch[s][0, 0].detach().cpu()
            ax.plot(np.arange(tgt.numel()), tgt.numpy(), lw=0.9, color='tab:green')
            ax.set_title('original (z)', fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])

        for g, (out, (label, counts, _)) in enumerate(zip(outs, plans)):
            mask = out['masks'][s][0]
            pred = out['preds'][s][0:1]
            if is_visual:
                raw = batch[s]
                fi = raw.shape[2] // 2 if frame < 0 else min(frame, raw.shape[2] - 1)
                Gt = mask.numel() // (model.grid_h * model.grid_w)
                m_sp = mask.view(Gt, model.grid_h, model.grid_w)[fi // model.tubelet[0]]
                ax_in = axes[r][1 + 2 * g]
                ax_pr = axes[r][2 + 2 * g]
                ax_in.imshow(_to_rgb(_blank_masked_patches(
                    raw[0, :, fi].detach().cpu(), m_sp, model)))
                rec = reconstruct_visual(raw, pred, mask.unsqueeze(0), model)
                ax_pr.imshow(_to_rgb(rec[0, :, fi].detach().cpu()))
                kept_units = int((m_sp == 0).sum())
                ax_in.set_title(f'masked in\nkept {kept_units}/{m_sp.numel()} tubes',
                                fontsize=7.5)
                # A fully visible stream has nothing to reconstruct: its
                # "prediction" panel is the ground truth, so it is labelled as
                # such instead of pretending to show a reconstruction.
                ax_pr.set_title(
                    f'prediction\n{counts[s]} tokens'
                    if kept_units < m_sp.numel() else
                    f'no masked token\n({counts[s]} kept)', fontsize=7.5)
                ax_in.set_xticks([]); ax_in.set_yticks([])
                ax_pr.set_xticks([]); ax_pr.set_yticks([])
            else:
                tgt = batch[s][0, 0].detach().cpu()
                vis = (mask.repeat_interleave(sig_kernel) == 0).cpu()
                prd = pred.reshape(-1).detach().cpu()
                x = np.arange(tgt.numel())
                ax_in = axes[r][1 + 2 * g]
                ax_pr = axes[r][2 + 2 * g]
                y_in = tgt.clone(); y_in[~vis] = float('nan')
                ax_in.plot(x, y_in.numpy(), lw=0.8, color='tab:blue')
                if int(vis.sum()) == 0:
                    ax_in.text(.5, .5, '100% masked', ha='center', va='center',
                               transform=ax_in.transAxes, fontsize=7)
                ax_pr.plot(x, prd.numpy(), lw=0.9, color='tab:red')
                ax_pr.plot(x[vis], tgt[vis].numpy(), '.', ms=1.4, color='k')
                kept_s = int(mask.sum() * 0 + (mask == 0).sum() * sig_kernel)
                ax_in.set_title(f'masked in\nkept {kept_s}/{mask.numel()*sig_kernel}',
                                fontsize=7.5)
                ax_pr.set_title(f'prediction\n{counts[s]} tokens', fontsize=7.5)
                for ax_ in (ax_in, ax_pr):
                    ax_.tick_params(labelsize=6)
                    ax_.set_yticks([])
        axes[r][0].set_ylabel(s, fontsize=9)

    for g, (label, _, _) in enumerate(plans):
        axes[0][1 + 2 * g].annotate(label, xy=(0.5, COLUMN_LABEL_Y),
                                    xycoords='axes fraction',
                                    ha='center', fontsize=8)
    e = entry
    n_streams = len(rows)
    word = {2: 'Two', 3: 'Three', 4: 'Four'}.get(n_streams, str(n_streams))
    n_exempt = sum(1 for _, _, pl in plans
                   if getattr(pl, 'budget_exempt', False))
    how = f'{len(plans) - n_exempt} budget allocations'
    if n_exempt:
        how += ' + a fully visible TIR (remote sensing)'
    fig.suptitle(f'{word}-stream reconstruction at {how} — '
                 f'subject {e.get("subject")}, session {e.get("session")}, '
                 f'held-out (fold 0 test)', fontsize=10)
    out_png = a.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(a.checkpoint))),
        'multimae_style_kept_density.png')
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=135)
    print('[fig] wrote', out_png)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
