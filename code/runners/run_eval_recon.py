"""Zero-shot RESPIRATION reconstruction from a FROZEN Stage-2 checkpoint.

No fine-tuning, no Stage-3: the checkpoint is used as it was saved (shared
encoder + per-stream decoders), the ``resp`` stream is masked with a PINNED
ratio, and the reconstruction is scored on the MASKED positions against the
dataset's own respiration target (the ROI lineage z-scores each clip, so the
target and the prediction live in the same per-clip z space and Pearson is
scale-invariant).

WHY THIS EXISTS -- the RGB2D question
-------------------------------------
"Does the extra 2D-texture stream improve respiration reconstruction?" is
answered with two complementary, controlled comparisons:

1. WITHIN-MODEL (exactly paired, the strongest statement).
   Same weights, same clips, same seeds; only ``mask_ratio_rgb2d`` changes:
   ``0.50`` (visible) vs ``1.00`` (rgb2d contributes NO encoder token). The
   mask draws for the other streams are BIT-IDENTICAL, because
   ``_tube_mask``/``_random_mask`` consume one ``torch.rand(B, N)`` whatever the
   ratio is -- so the only difference is whether rgb2d reached the encoder.
       --checkpoint <tri-modal ckpt> --mask_ratio_rgb2d 0.50 --label with_rgb2d
       --checkpoint <tri-modal ckpt> --mask_ratio_rgb2d 1.00 --label no_rgb2d

2. ACROSS-MODEL (different training runs, same evaluation set).
   The tri-modal checkpoint vs the TIR+RESP twin checkpoint. ``--data_checkpoint``
   pins the DATASET (and therefore the clip set) to one checkpoint's arguments,
   so both arms score exactly the same clips; the 2-stream model simply ignores
   the extra ``rgb2d`` key of the batch.
       --checkpoint .../stage2_hpc_tir_roi_resp_rgb2d_fold0/checkpoints/checkpoint-0149.pth
       --checkpoint .../stage2_hpc_tir_roi_resp_96respgrid_fold0/checkpoints/checkpoint-0149.pth \
           --data_checkpoint .../stage2_hpc_tir_roi_resp_rgb2d_fold0/checkpoints/checkpoint-0149.pth

Metrics (per clip, then aggregated over clips; each clip is scored over
``--repeats`` independent mask draws and averaged FIRST, so a clip is never
weighted more than another):

* ``mse_masked``  -- masked-position MSE, the training objective (per-sample)
* ``r_masked``    -- Pearson over the MASKED samples only. The honest number:
                     those samples are never visible to the model.
* ``r_full``      -- Pearson over the assembled waveform (prediction at masked
                     positions, ground truth at visible ones). This is what a
                     downstream user sees, but it is inflated by the visible
                     part, so it is NOT the headline.
* ``mae``/``rmse`` -- masked-position error, z units.

Usage (from ``code/``)::

    python -u runners/run_eval_recon.py --checkpoint <ckpt> --device cuda \
        --label with_rgb2d --out out.json \\
        --mask_mode ratio --mask_ratio_tir 0.90 --mask_ratio_rgb2d 0.50 \\
        --mask_ratio_resp 0.50 --repeats 5
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

from core.multimae import build_pretraining_model        # noqa: E402
from runners.run_visualize_pretrain import _held_out_subjects  # noqa: E402


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def get_args(argv=None):
    p = argparse.ArgumentParser(
        description='Zero-shot RESP reconstruction from a frozen Stage-2 '
                    'checkpoint (no fine-tuning).')
    p.add_argument('--checkpoint', required=True,
                   help='Stage-2 checkpoint whose WEIGHTS are evaluated')
    p.add_argument('--data_checkpoint', default='',
                   help='optional: build the dataset (and thus the CLIP SET) '
                        'from THIS checkpoint\'s args instead. Use it to make '
                        'two different checkpoints score identical clips.')
    p.add_argument('--split', default='test', choices=('test', 'corpus'),
                   help="'test' (default) = the weight checkpoint's held-out "
                        'fold subjects')
    p.add_argument('--fold_manifest', default='',
                   help='default: <run_dir>/fold_manifest.json of --checkpoint')
    p.add_argument('--label', default='', help='name used in the report')
    p.add_argument('--out', default='', help='JSON report path (default: none)')
    # ---- masking of the evaluation clips (pinned, NOT the training policy) --
    # AMMM kept-ratio overrides: the ONLY mechanism that can mask a stream to
    # ZERO kept units (the ratio path clamps to n_units - 1, so 'mask 1.0' still
    # leaks one patch / one token). All streams are pinned, so no Dirichlet draw
    # happens and the masks of the OTHER streams are bit-identical between two
    # arms that differ only in one stream's value -> an exactly paired ablation.
    p.add_argument('--kept_tir', default=0.10, type=float,
                   help='fraction of the 36 TIR patches kept (0.10 == the '
                        'training mask ratio 0.90)')
    p.add_argument('--kept_resp', default=0.0, type=float,
                   help='fraction of the 50 resp tokens kept. 0.0 (default) = '
                        'the waveform is ENTIRELY hidden, so the reconstruction '
                        'can only come from the visual streams (the cross-modal '
                        'question); 0.5 reproduces the training draw and is '
                        'locally interpolatable (r~0.998) -- a reference only.')
    p.add_argument('--kept_rgb2d', default=0.50, type=float,
                   help='fraction of the 36 rgb2d patches kept. 0.0 = the 2D '
                        'stream contributes NO encoder token (the ablation). '
                        'Inert for a 2-stream checkpoint.')
    p.add_argument('--resp_probe', default='span',
                   choices=('span', 'random', 'hidden'),
                   help="how the waveform is hidden. 'random' keeps "
                        '--kept_resp scattered (the training condition: locally '
                        'interpolatable, r~0.998); \'span\' keeps the same '
                        'budget but as ONE contiguous block (not '
                        "interpolatable -- the informative probe); 'hidden' "
                        'masks it completely (no cross-modal prior at all).')
    p.add_argument('--mask_span_s', default=4.0, type=float,
                   help='--resp_probe span: block length in seconds')
    # ---- evaluation loop ---------------------------------------------------
    p.add_argument('--repeats', default=5, type=int,
                   help='independent mask draws per clip; the masks are seeded '
                        'as seed+r so two runs with the same seed see the SAME '
                        'masks (the paired ablation relies on this)')
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--max_clips', default=0, type=int,
                   help='cap the number of evaluated clips (0 = all); the '
                        'subset is drawn with --seed so it is reproducible')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--device', default='cpu', type=str)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str) -> Dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('model', 'args'):
        if key not in ckpt:
            raise SystemExit(f'{path}: not a Stage-2 checkpoint (no {key!r})')
    return ckpt


def build_model(ckpt: Dict, device: torch.device):
    cargs = ckpt['args']
    model = build_pretraining_model(cargs)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        # A Stage-2 checkpoint always carries every encoder/decoder tensor; a
        # mismatch means the architecture changed under the weights.
        print(f'[eval] WARNING state_dict: {len(missing)} missing / '
              f'{len(unexpected)} unexpected')
        if missing:
            print('[eval]   missing:', missing[:5])
        if unexpected:
            print('[eval]   unexpected:', unexpected[:5])
    return model.to(device).eval()


def pin_masking(model, a, cargs) -> str:
    """Pin the KEPT ratio of every stream; return a caption.

    AMMM overrides are used VERBATIM (``0 <= r <= cap``), are exempt from
    ``min_kept`` and are outside the Dirichlet draw, which is what makes the
    per-stream masks reproducible and the comparison paired.
    """
    pins = {}
    for s in model.streams:
        if s == 'tir':
            v = float(a.kept_tir)
        elif s == 'rgb2d':
            v = float(a.kept_rgb2d)
        elif s == 'resp':
            v = (0.0 if a.resp_probe == 'hidden' else float(a.kept_resp))
        else:                      # unused declared stream: keep it visible
            v = 1.0
        if not 0.0 <= v <= 1.0:
            raise SystemExit(f'--kept_{s} must be in [0, 1]; got {v}')
        pins[s] = v
    if all(v == 0.0 for v in pins.values()):
        raise SystemExit('[eval] every stream pinned at 0 kept units -- the '
                         'encoder would receive nothing')
    # A 'ratio'-trained checkpoint carries no AMMM specs, so derive the caps from
    # its own args (same defaults the model uses) and raise any cap that would
    # reject a pin. configure_ammm() then builds the specs AND validates the
    # pinned set immediately (it runs one plan_step).
    caps = {}
    for s in model.streams:
        if s in model.visual:
            base = float(getattr(cargs, 'video_kept_cap', 0.25) or 0.25)
        else:
            base = float(getattr(cargs, 'physio_kept_cap', 1.0) or 1.0)
        if pins[s] > base:
            print(f'[eval] raising {s} KEPT cap {base:.2f} -> {pins[s]:.2f} '
                  f'to allow the pin')
            base = pins[s]
        caps[s] = base
    model.configure_ammm(video_kept_cap=caps, physio_kept_cap=caps,
                         kept_ratio_override=pins, min_kept=0, verbose=False)
    model.ammm_min_kept = 0
    model.masking = 'ammm'
    # Placement of a 1-D stream: contiguous spans are what remove the
    # interpolate-from-neighbours shortcut. The span COUNT is derived from the
    # MASK ratio, i.e. 1 - kept, so it matches the pinned plan.
    if a.resp_probe == 'span':
        model.physio_mask = 'span'
        model.mask_ratios = {s: 1.0 - pins[s] for s in model.streams}
        model.configure_span_mask(a.mask_span_s)
    else:
        model.physio_mask = 'random'
    return (f'KEPT ' + ' '.join(f'{s}={v:.2f}' for s, v in pins.items())
            + f'  placement={model.physio_mask}')


def build_eval_dataset(data_cargs, a, held: Optional[set]):
    """The held-out clip pool, built from ``data_cargs`` (never mutated)."""
    import copy

    from data import build_pretraining_dataset

    cargs = copy.copy(data_cargs)
    cargs.pretrain_split = 'all'
    cargs.subjects = ''
    ds = build_pretraining_dataset(cargs)
    idx = list(range(len(ds)))
    if held is not None:
        idx = [i for i in idx
               if str(ds.entries[i].get('subject', '')) in held]
    if not idx:
        raise SystemExit('[eval] no clip belongs to the evaluation set -- check '
                         '--split / the fold manifest / the caches')
    if a.max_clips and len(idx) > a.max_clips:
        g = torch.Generator().manual_seed(a.seed)
        order = torch.randperm(len(idx), generator=g).tolist()
        idx = [idx[k] for k in sorted(order[:a.max_clips])]
    return ds, idx


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, or nan when either side is constant."""
    if x.size < 3:
        return float('nan')
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else float('nan')


def score_batch(out: Dict, batch: Dict, sig_kernel: int) -> List[Dict]:
    """Per-clip metrics for the resp stream of one batch."""
    pred = out['preds']['resp'].detach().float().cpu().numpy()
    target = (batch['resp'].detach().float().cpu()
              .reshape(pred.shape[0], -1, sig_kernel).numpy())
    mask = out['masks']['resp'].detach().cpu().numpy().astype(bool)
    B = pred.shape[0]
    rows = []
    for b in range(B):
        m = mask[b]
        if m.sum() == 0:                       # 100 % visible: nothing to score
            rows.append({'n_masked': 0, 'mse': float('nan'),
                         'mae': float('nan'), 'rmse': float('nan'),
                         'r_masked': float('nan'), 'r_full': float('nan')})
            continue
        p_m = pred[b][m].reshape(-1)
        t_m = target[b][m].reshape(-1)
        d = p_m - t_m
        assembled = np.where(m[:, None], pred[b], target[b]).reshape(-1)
        rows.append({
            'n_masked': int(m.sum() * sig_kernel),
            'mse': float((d * d).mean()),
            'mae': float(np.abs(d).mean()),
            'rmse': float(np.sqrt((d * d).mean())),
            'r_masked': _pearson(p_m, t_m),
            'r_full': _pearson(assembled, target[b].reshape(-1)),
        })
    return rows


def mean_std(vals: List[float]) -> Dict[str, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {'mean': float('nan'), 'median': float('nan'), 'std': float('nan'),
                'n': 0}
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'std': float(a.std()), 'n': int(a.size)}


# --------------------------------------------------------------------------- #
def main(argv=None):
    a = get_args(argv)
    dev = torch.device(a.device if (a.device != 'cuda' or
                                    torch.cuda.is_available()) else 'cpu')
    ckpt = load_checkpoint(a.checkpoint)
    cargs = ckpt['args']
    label = a.label or os.path.basename(os.path.dirname(
        os.path.dirname(os.path.abspath(a.checkpoint))))

    if not a.fold_manifest:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
        cand = os.path.join(run_dir, 'fold_manifest.json')
        a.fold_manifest = cand if os.path.isfile(cand) else ''

    model = build_model(ckpt, dev)
    caption = pin_masking(model, a, cargs)
    print(f'[eval] {label}')
    print(f'[eval] weights   : {a.checkpoint} (epoch {ckpt.get("epoch")})')
    print(f'[eval] streams   : {model.streams}  (2-stream ckpts ignore rgb2d)')
    print(f'[eval] masking   : {caption}')
    print(f'[eval] target_norm={getattr(cargs, "target_norm", "?")} '
          f'input_size={getattr(cargs, "input_size", "?")} '
          f'n_signal={model.n_signal} sig_kernel={model.sig_kernel}')

    # ---- data (from --data_checkpoint when given) -------------------------
    data_cargs = cargs
    if a.data_checkpoint:
        dck = load_checkpoint(a.data_checkpoint)
        data_cargs = dck['args']
        print(f'[eval] dataset   : from {a.data_checkpoint} '
              f'(data_set={getattr(data_cargs, "data_set", "?")})')
    held = None
    if a.split == 'test':
        held, train, src = _held_out_subjects(cargs, a.fold_manifest)
        print(f'[eval] split     : test -> fold {getattr(cargs, "fold", 0)}/'
              f'{getattr(cargs, "n_folds", 1)}, {len(held)} held-out subject(s) '
              f'({src}), {len(train)} train')
    ds, idx = build_eval_dataset(data_cargs, a, held)
    subjects = {str(ds.entries[i].get('subject', '')) for i in idx}
    sessions = {str(ds.entries[i].get('session', '')) for i in idx}
    print(f'[eval] clips     : {len(idx)} from {len(sessions)} session(s) / '
          f'{len(subjects)} subject(s); streams in a sample: '
          f'{sorted(ds[idx[0]].keys())}')

    # ---- loop -------------------------------------------------------------
    per_clip: List[Dict] = []
    B = int(a.batch_size)
    for rep in range(max(1, int(a.repeats))):
        torch.manual_seed(a.seed + rep)     # identical masks across arms
        rep_rows: List[Dict] = []
        for start in range(0, len(idx), B):
            chunk = idx[start:start + B]
            items = [ds[i] for i in chunk]
            batch = {k: torch.stack([it[k] for it in items]).to(dev).float()
                     for k in items[0]}
            with torch.inference_mode():
                out = model(batch)
            rep_rows.extend(score_batch(out, batch, model.sig_kernel))
        # average the clip over its draws FIRST, then aggregate over clips
        if rep == 0:
            per_clip = [dict(r) for r in rep_rows]
        else:
            for acc, r in zip(per_clip, rep_rows):
                for k in ('mse', 'mae', 'rmse', 'r_masked', 'r_full'):
                    acc[k] = float(np.nanmean([acc[k], r[k]]))
        print(f'[eval] rep {rep + 1}/{a.repeats}: '
              f'r_masked mean {np.nanmean([r["r_masked"] for r in rep_rows]):+.4f}'
              f'  mse {np.nanmean([r["mse"] for r in rep_rows]):.5f}')

    # ---- aggregate --------------------------------------------------------
    summary = {k: mean_std([r[k] for r in per_clip])
               for k in ('mse', 'mae', 'rmse', 'r_masked', 'r_full')}
    print()
    print('=' * 68)
    print(f'{label}: {len(per_clip)} clip(s) x {a.repeats} draw(s)')
    for k in ('mse', 'rmse', 'mae'):
        s = summary[k]
        print(f'  {k:9s} mean {s["mean"]:.5f}  median {s["median"]:.5f} '
              f'(z units, masked positions)')
    for k in ('r_masked', 'r_full'):
        s = summary[k]
        print(f'  {k:9s} mean {s["mean"]:+.4f}  median {s["median"]:+.4f} '
              f'sd {s["std"]:.4f}  (n={s["n"]})')
    print('=' * 68)

    if a.out:
        with open(a.out, 'w') as fh:
            json.dump({'label': label, 'checkpoint': a.checkpoint,
                       'data_checkpoint': a.data_checkpoint,
                       'split': a.split, 'fold': getattr(cargs, 'fold', None),
                       'masking': caption, 'repeats': a.repeats,
                       'n_clips': len(per_clip),
                       'n_sessions': len(sessions),
                       'n_subjects': len(subjects),
                       'summary': summary,
                       'per_clip': per_clip}, fh, indent=2)
        print(f'[eval] wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
