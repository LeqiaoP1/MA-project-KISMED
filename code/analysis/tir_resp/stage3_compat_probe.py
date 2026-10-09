#!/usr/bin/env python
"""Stage-3 <- Stage-2 compatibility probe (CPU-only, no GPU needed).

Answers ONE question: *will ``configs/finetune/resp_tir_roi_hpc.yaml`` load a
given Stage-2 checkpoint correctly?*

It deliberately calls the REAL production code rather than re-implementing it:

  * ``core.waveform_model.build_waveform_model``   -- the Stage-3 model
  * ``core.waveform_model.load_stage2_encoder``    -- the weight transfer
  * ``runners.run_waveform.check_stage2_roi_contract`` -- the ROI gate

so a PASS here is the same verdict the runner would reach, minus the dataset.

WHAT IT CHECKS
--------------
1. **State-dict transfer.** ``enc_blocks.*`` / ``enc_norm.*`` / ``adapters.*`` /
   ``positions.*`` are copied; a SHAPE mismatch is fatal (the loader raises
   rather than partially loading).
2. **Head transfer.** Stage-2 ``heads.resp`` -> Stage-3 ``waveform_head``. Only
   possible when ``samples_per_token == sig_kernel`` and ``head_hidden == 0``;
   otherwise the head silently starts RANDOM, which is a quality bug, not a
   crash -- so it is reported loudly here.
3. **ROI contract** (``roi_landmarks`` / ``roi_padding`` / ``input_size``).
   These change WHICH PIXELS reach the encoder while leaving every tensor shape
   identical, so nothing downstream can notice -- the runner ABORTS instead.
4. **Geometry mirror** (``tubelet`` / ``fps`` / ``temporal_stride`` /
   ``clip_duration`` / ``sig_kernel`` / ``fs`` / ``seq_len`` / ``input_size``)
   and the corpus-cleaning knobs, compared against the args RECORDED IN THE
   CHECKPOINT by ``utils.checkpoint.save_model``.
5. **MR-STFT window fit** against the Stage-3 target length.

Usage (from ``code/``)::

    python analysis/tir_resp/stage3_compat_probe.py
    python analysis/tir_resp/stage3_compat_probe.py \
        --config configs/finetune/resp_tir_roi_hpc.yaml \
        --finetune /work/scratch/$USER/pretrain/stage2_hpc_tir_roi_resp/checkpoints/checkpoint-0149.pth

Exit code 0 = compatible, 1 = at least one BLOCKER.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_STAGE3 = 'configs/finetune/resp_tir_roi_hpc.yaml'
DEFAULT_STAGE2_DIR = ('/work/scratch/{user}/pretrain/'
                      'stage2_hpc_tir_roi_resp')

#: Geometry / contract keys mirrored from the Stage-2 run. Grouped so the report
#: can distinguish "different pixels" from "different shapes".
#: ``streams`` is deliberately NOT here: Stage 2 feeds ``tir,resp`` while Stage 3
#: feeds ``tir`` and predicts resp, so the two are SUPPOSED to differ. The real
#: invariant (every Stage-3 VISUAL stream existed in Stage 2) is checked on its
#: own below.
ROI_KEYS = ('roi_landmarks', 'roi_padding', 'input_size')
GEOM_KEYS = ('model', 'tubelet', 'fps', 'temporal_stride',
             'clip_duration', 'input_size', 'fs', 'seq_len', 'sig_kernel',
             'head_hidden')
CORPUS_KEYS = ('task_set', 'tasks', 'min_signal_spread', 'rail_touch_v',
               'clip_stride', 'physio_norm', 'spectral_fft_sizes')

OK, BAD, WARN = '  OK  ', 'FAIL  ', ' WARN '


def newest_ckpt(run_dir: str) -> Optional[str]:
    """Newest ``checkpoint-*.pth`` in ``<run_dir>/checkpoints`` (VERSION sort).

    Deliberately NON-recursive: an archived ``checkpoints.precache_*/`` sibling
    must never be picked up as "the newest Stage-2 checkpoint".
    """
    hits = glob.glob(os.path.join(run_dir, 'checkpoints', 'checkpoint-*.pth'))
    return sorted(hits, key=lambda p: os.path.basename(p))[-1] if hits else None


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, 'r') as fh:
        return yaml.safe_load(fh) or {}


def as_namespace(cfg: Dict[str, Any]) -> types.SimpleNamespace:
    """YAML dict -> the attribute namespace the production builders expect."""
    return types.SimpleNamespace(**cfg)


def compare(label: str, s2: Any, s3: Any, marker: str) -> Tuple[bool, str]:
    same = s2 == s3
    tag = OK if same else marker
    return same, (f'[{tag}] {label:<18} stage2={s2!r:<16} stage3={s3!r}')


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_STAGE3,
                    help=f'Stage-3 config (default: {DEFAULT_STAGE3})')
    ap.add_argument('--finetune', default='',
                    help='Stage-2 checkpoint; unset -> newest under --stage2_dir')
    ap.add_argument('--stage2_dir',
                    default=DEFAULT_STAGE2_DIR.format(user=os.environ.get('USER', '')),
                    help='Stage-2 run dir used to auto-resolve --finetune')
    args = ap.parse_args(argv)

    print('=' * 78)
    print('Stage-3 <- Stage-2 compatibility probe')
    print('=' * 78)

    cfg = load_cfg(args.config)
    ns = as_namespace(cfg)
    print(f'config   : {args.config}')
    print(f'data_set : {getattr(ns, "data_set", None)!r}  '
          f'target={getattr(ns, "target", None)!r}  '
          f'model={getattr(ns, "model", None)!r}')

    ckpt_path = args.finetune or newest_ckpt(args.stage2_dir)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f'\n[{BAD}] no Stage-2 checkpoint resolved.\n'
              f'        looked in : {os.path.join(args.stage2_dir, "checkpoints")}\n'
              f'        got       : {ckpt_path!r}')
        return 1
    print(f'finetune : {ckpt_path}')
    print(f'           {os.path.getsize(ckpt_path) / 1e9:.2f} GB')

    import torch

    # ---------------- 1. build the Stage-3 model (real code) -------------- #
    from core.waveform_model import build_waveform_model, load_stage2_encoder

    blockers: List[str] = []
    warnings: List[str] = []

    try:
        model = build_waveform_model(ns)
    except Exception as exc:                       # noqa: BLE001 - report, don't crash
        print(f'\n[{BAD}] build_waveform_model raised: {type(exc).__name__}: {exc}')
        return 1

    print(f'\nStage-3 model: streams={list(model.streams)} '
          f'num_frames={model.num_frames} input={model.input_size} '
          f'output_len={model.output_len} grid_t={model.grid_t} '
          f'samples_per_token={model.samples_per_token} '
          f'sig_kernel={model.sig_kernel} head_style={model.head_style}')

    # ---------------- 2. load the Stage-2 weights (real code) ------------- #
    print('\n--- weight transfer ---')
    try:
        info = load_stage2_encoder(model, ckpt_path, target=getattr(ns, 'target', None))
    except Exception as exc:                       # noqa: BLE001
        print(f'[{BAD}] load_stage2_encoder raised: {type(exc).__name__}: {exc}')
        return 1

    s2_args = info.get('stage2_args')
    print(f'        loaded={info["loaded"]} skipped={info["skipped"]} '
          f'shape_mismatch={info["shape_mismatch"]}')
    if info['head_loaded']:
        print(f'[{OK}] waveform_head transferred from {info["head_source"]} '
              f'({len(info["head_loaded"])} tensors)')
    else:
        blockers.append('waveform_head did NOT transfer (starts random)')
        print(f'[{BAD}] waveform_head NOT transferred: {info.get("head_note")}')

    # ---------------- 3. the ROI contract (real code) --------------------- #
    print('\n--- ROI contract (aborts the runner on mismatch) ---')
    if s2_args is None:
        blockers.append('checkpoint records no `args` -> ROI contract unverifiable')
        print(f'[{BAD}] the checkpoint records no `args` namespace; the runner '
              f'cannot verify the ROI contract at all.')
    else:
        from runners.run_waveform import check_stage2_roi_contract
        try:
            check_stage2_roi_contract(ns, s2_args, allow_mismatch=False)
            print(f'[{OK}] ROI contract satisfied (runner would not abort)')
        except SystemExit as exc:
            blockers.append(f'ROI contract: {exc}')
            print(f'[{BAD}] {exc}')

    # ---------------- 4. geometry / corpus mirror ------------------------- #
    if s2_args is not None:
        print('\n--- geometry mirror (checkpoint args vs Stage-3 config) ---')
        for label, keys, marker in (('GEOMETRY', GEOM_KEYS, BAD),
                                    ('CORPUS', CORPUS_KEYS, WARN)):
            for key in keys:
                s2, s3 = getattr(s2_args, key, None), getattr(ns, key, None)
                if s2 is None or s3 is None:
                    continue
                same, line = compare(key, s2, s3, marker)
                print(line)
                if not same:
                    (blockers if marker == BAD else warnings).append(
                        f'{label} {key}: {s2!r} -> {s3!r}')

        # Stage-2's spectral windows live under a different key name.
        s2_fft = getattr(s2_args, 'spectral_fft_sizes', None)
        s3_fft = getattr(ns, 'fft_sizes', None)
        if s2_fft is not None and s3_fft is not None:
            a = tuple(int(x) for x in str(s2_fft).replace(' ', '').split(',') if x)
            b = tuple(int(x) for x in str(s3_fft).replace(' ', '').split(',') if x)
            print(f'[{OK if a == b else WARN}] spectral_fft_sizes   '
                  f'stage2={a} stage3={b}')
            if a != b:
                warnings.append(f'spectral_fft_sizes {a} -> {b}')

        print(f'\nstage-2 streams were        : '
              f'{getattr(s2_args, "streams", None)!r}')
        print(f'stage-3 feeds               : {list(model.streams)}  '
              f'(the resp stream is the TARGET here, not an input)')

        # The real invariant: every VISUAL stream Stage 3 feeds must have been a
        # Stage-2 stream, else its adapter has no pre-trained weights to inherit
        # (the loader would copy nothing for it and quietly start it random).
        s2_all = [s.strip() for s in
                  str(getattr(s2_args, 'streams', '')).split(',') if s.strip()]
        missing = [s for s in model.streams if s not in s2_all]
        if missing:
            blockers.append(f'Stage-3 visual stream(s) {missing} absent from the '
                            f'Stage-2 run {s2_all} -> their adapters start random')
            print(f'[{BAD}] stage-3 visual stream(s) {missing} are NOT in the '
                  f'Stage-2 run {s2_all}')
        else:
            print(f'[{OK}] every Stage-3 visual stream exists in the Stage-2 run '
                  f'({list(model.streams)} subset of {s2_all})')

    # ---------------- 5. MR-STFT window fit ------------------------------- #
    print('\n--- MR-STFT windows vs the Stage-3 target ---')
    from runners.run_waveform import _parse_fft_sizes, _resolve_fft_sizes
    try:
        # the runner converts the YAML string to a tuple at arg-parse time
        # (run_waveform.py:432) BEFORE _resolve_fft_sizes sees it.
        fft = _parse_fft_sizes(getattr(ns, 'fft_sizes', ''))
        kept, dropped = _resolve_fft_sizes(fft, model.output_len,
                                          fs=float(getattr(ns, 'fs', 100.0)))
        print(f'[{OK}] kept={kept} dropped={dropped} '
              f'for a {model.output_len}-sample target '
              f'({model.output_len / float(getattr(ns, "fs", 100.0)):.2f} s)')
    except SystemExit as exc:
        blockers.append(f'fft_sizes: {exc}')
        print(f'[{BAD}] {exc}')

    # ---------------- verdict --------------------------------------------- #
    print('\n' + '=' * 78)
    for w in warnings:
        print(f'[{WARN}] {w}')
    if blockers:
        print(f'\nRESULT: {len(blockers)} BLOCKER(S) -- the runner would fail:')
        for b in blockers:
            print(f'   - {b}')
        return 1
    print('RESULT: COMPATIBLE -- the Stage-3 config loads this checkpoint '
          '(encoder + head), and the ROI/geometry contracts are satisfied.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
