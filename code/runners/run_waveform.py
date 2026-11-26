"""Stage-3 downstream waveform fine-tuning (BP or RESP branch).

Implements the Stage-3 design (``code/notes/design/ADR-015-stage3-waveform-regression-and-joint-loss.md``;
plan entry ``code/notes/plan/experiments/RGB_BP_plan.md`` §4): simulated full sensor failure
(no contact 1D streams are fed) -- only visual streams drive the prediction.
Two independent task-specialised runs share the Stage-2 encoder checkpoint
(``--finetune``) but use separate regression heads and a unified
spatio-temporal-spectral joint loss (``core.waveform_losses.WaveformJointLoss``).

The Stage-3 model is ``core.waveform_model.MultiModalWaveformRegressor``: the
Stage-2 multimodal encoder (tubelet front-end, joint space-time attention,
identical key names) + a temporal regression head that predicts one waveform
segment per tubelet time step. ``--model project_multimae_<variant>`` selects
it; the clip geometry flags (``--clip_duration/--fps/--tubelet/--temporal_
stride/--sig_kernel/--input_size/--streams``) MUST MIRROR the Stage-2 run the
checkpoint comes from -- both are validated on load, and the head is only
time-aligned when ``sig_kernel/fs == tubelet_t*temporal_stride/fps``.

Usage (from ``code/``)::

    python runners/run_waveform.py -c configs/finetune/bp.yaml
    python runners/run_waveform.py -c configs/finetune/resp.yaml

The ``bp4d+`` dataset (``data/paired_dataset.py``) yields ``(samples, [B, T])``
with ``samples = [B, 3, T, H, W]`` (``use_tir`` off) or ``[B, 6, T, H, W]``.
"""
import argparse
import glob
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from core import WaveformJointLoss
from engines import (evaluate_waveform_loss, predict_waveforms,
                     train_one_epoch_waveform)
from runners._common import (add_common_args, add_protocol_args,
                             check_loader_not_empty, env_or, init_env,
                             load_or_create_plan, make_data_loader,
                             parse_args_with_config)


def _parse_fft_sizes(s):
    return tuple(int(x) for x in str(s).split(',') if x.strip())


def _resolve_fft_sizes(fft_sizes, output_len, fs=100.0):
    """Drop MR-STFT windows longer than the Stage-3 target waveform.

    ``torch.stft`` ZERO-PADS when ``n_fft`` exceeds the signal, so an over-long
    window does NOT raise -- it silently turns part of the magnitude comparison
    into a comparison of padding. This guard therefore drops too-long windows
    rather than letting that happen.

    :returns: ``(kept, dropped)``. Raises ``SystemExit`` when nothing survives.
    """
    output_len = int(output_len)
    sizes = [int(n) for n in fft_sizes]
    if not sizes:
        raise SystemExit('--fft_sizes is empty: the MR-STFT term needs at least '
                         'one window (per-modality defaults: bp 64,128,256 / '
                         'resp 128,256,512).')
    kept = [n for n in sizes if n <= output_len]
    dropped = [n for n in sizes if n > output_len]
    if not kept:
        raise SystemExit(
            f'--fft_sizes {tuple(sizes)}: NO window fits the Stage-3 target '
            f'waveform ({output_len} samples = {output_len / fs:.2f} s at fs '
            f'{fs:g} Hz), and torch.stft would zero-pad instead of raising. Use '
            f'a longer clip (--clip_duration) or smaller windows. Per-modality '
            f'defaults: bp 64,128,256 / resp 128,256,512.')
    return tuple(kept), tuple(dropped)


def _fft_window_advisory(kept, band_lo, output_len, fs=100.0):
    """Report the lowest rate the kept windows can actually resolve.

    A window must span >= ONE period of the band it polices, else the magnitude
    term measures local waveform SHAPE rather than RATE -- precisely how the
    historical respiration set 64,128,256 failed: at fs = 100 Hz the whole
    0.1-0.6 Hz band collapsed into STFT bin 0, so the term could never see
    respiratory rate.

    Returns an INFORMATIONAL string (never an error) when the largest kept window
    cannot reach the slow edge of the band. That is usually a CLIP-LENGTH limit
    rather than a config mistake, because no window may exceed ``output_len`` --
    at 8 s and 0.1 Hz the needed 10 s window cannot exist, so the note says so
    instead of implying the window set should be changed.
    """
    if not kept or not band_lo or float(band_lo) <= 0:
        return None
    need = fs / float(band_lo)                 # samples in one period, slow edge
    if max(kept) >= need:
        return None
    resolvable = fs / max(kept)                # lowest rate the longest window spans
    return (f'largest window {max(kept)} samples = {max(kept) / fs:.2f} s cannot '
            f'span one period at the slow edge of --eval_band '
            f'({float(band_lo):g} Hz -> {need:.0f} samples = {need / fs:.2f} s), '
            f'so RATE is policed only down to ~{resolvable:.3f} Hz '
            f'({resolvable * 60:.0f} bpm). A slower rate needs a LONGER CLIP, not '
            f'a different --fft_sizes: no window may exceed the {output_len}-'
            f'sample target.')


#: Keys that define WHICH PIXELS the encoder sees on the RAW-tree ROI paths
#: (``data_set: tir_roi*``). They do NOT change any tensor shape, so a mismatch
#: cannot be detected downstream: the run trains happily on an input the encoder
#: was never pre-trained on (the checkpoint's positional embeddings and adapter
#: shapes fit either way). They are therefore compared against the args RECORDED
#: IN THE CHECKPOINT -- ``utils.checkpoint.save_model`` stores the whole
#: namespace -- and a mismatch ABORTS the run unless ``--allow_roi_mismatch``
#: declares it deliberate.
ROI_CONTRACT_KEYS = ('roi_landmarks', 'roi_padding', 'input_size',
                     'crop_rule', 'max_sway_px')


def _parse_recorded_groups(ns):
    """``args.task_groups`` -> ``{group: (tasks,)}``; ``None`` when unset.

    Accepts both accepted definitions forms (YAML mapping and the
    ``'low=T2|T3;high=T7'`` string). A malformed definition is treated as
    "nothing to compare" instead of raising: the dataset already fails fast on
    it, and this routine only produces the advisory corpus note.
    """
    from data import task_groups as tgrp

    raw = getattr(ns, 'task_groups', None)
    if raw is None or not str(raw).strip():
        return None
    try:
        return tgrp.parse_task_groups(raw)
    except tgrp.TaskGroupError:
        return None


def check_stage2_roi_contract(args, stage2_args, allow_mismatch=False):
    """Verify that Stage 3 crops the ROI exactly as the Stage-2 run did.

    Compares :data:`ROI_CONTRACT_KEYS` between the args this run will feed to
    the dataset and the args stored in the ``--finetune`` checkpoint.
    ``roi_landmarks`` is compared RESOLVED (1-indexed landmark tuple), so ``''``
    == ``'nose_mouth'`` == the explicit 12-label CSV. Returns the comparison
    table ``{key: {'stage2': .., 'stage3': .., 'match': bool}}``.
    """
    from data import TIR_ROI_DATA_SETS
    from data.tir_resp_dataset import resolve_roi_landmarks

    name = str(getattr(args, 'data_set', '') or '').strip().lower()
    if name not in TIR_ROI_DATA_SETS:
        return {}                       # only this path crops via roi_* args
    if stage2_args is None:
        print('[stage3] ROI contract: the --finetune checkpoint records no '
              '`args` (it is a Stage-1 MAE/ViT init) -> nothing to compare.')
        return {}

    def _read(ns, key):
        raw = getattr(ns, key, None)
        if raw is None:
            return None
        if key == 'roi_landmarks':
            try:
                return resolve_roi_landmarks(raw)
            except Exception:
                return f'<unresolved {raw!r}>'
        if key == 'crop_rule':
            return str(raw).strip().lower()
        return int(raw) if key == 'input_size' else float(raw)

    rows, mismatch = [], []
    for key in ROI_CONTRACT_KEYS:
        stage2, stage3 = _read(stage2_args, key), _read(args, key)
        if stage2 is None or stage3 is None:
            continue
        rows.append((key, stage2, stage3, stage2 == stage3))
        if stage2 != stage3:
            mismatch.append((key, stage2, stage3))

    if not rows:
        print('[stage3] ROI contract: the Stage-2 checkpoint recorded none of '
              f'{list(ROI_CONTRACT_KEYS)} -> nothing to compare.')
        return {}

    # A checkpoint recorded BEFORE the square-crop / motion-gate contract carries
    # none of the newer keys, so the comparison above cannot say anything about
    # them. Say THAT out loud: a square Stage-3 run on an encoder pre-trained
    # with the legacy geometry looks perfectly healthy otherwise.
    missing = [k for k in ('crop_rule', 'max_sway_px')
               if getattr(stage2_args, k, None) is None]
    if missing:
        print('[stage3] WARNING: the Stage-2 checkpoint records none of '
              f'{missing} -- it predates the square-crop / motion-gate '
              f'contract. This run crops with crop_rule='
              f'{getattr(args, "crop_rule", "?")!r}, max_sway_px='
              f'{getattr(args, "max_sway_px", "?")!r} and therefore feeds the '
              'encoder pixels it was NOT pre-trained on. Re-pretrain with this '
              'geometry, or pass --allow_roi_mismatch to acknowledge it.')

    print('[stage3] ROI contract (Stage-2 checkpoint vs this run):')
    for key, stage2, stage3, same in rows:
        print(f'    {key:<14} {stage2!r} {"==" if same else "!="} {stage3!r}'
              + ('' if same else '   <== MISMATCH'))

    if mismatch:
        detail = '; '.join(f'{k}: {a!r} -> {b!r}' for k, a, b in mismatch)
        msg = (f'the Stage-3 ROI does not match the one the --finetune '
               f'checkpoint was pre-trained on ({detail}). These keys change '
               f'which PIXELS reach the encoder while leaving every tensor '
               f'shape unchanged, so the run would look perfectly normal '
               f'while fine-tuning an encoder on an input it never saw. '
               f'Mirror the Stage-2 value(s), re-pretrain with the new ROI, '
               f'or pass --allow_roi_mismatch for a deliberate ablation.')
        if not allow_mismatch:
            raise SystemExit('[stage3] ' + msg)
        print('[stage3] WARNING: --allow_roi_mismatch -- ' + msg)

    # Corpus-level keys: the same hazard class (Stage 3 would be evaluated on a
    # corpus the encoder was not pre-trained on), but a deliberate ablation here
    # is legitimate often enough that it only WARNS. ``clip_stride`` belongs to
    # this class too: with 1 s hops 7/8 of consecutive windows overlap, so it
    # decides WHICH windows exist (it does not change any tensor shape).
    #
    # The respiration-cache lineage REPLACES the two window-content knobs: with
    # a cache the validity statement is the stored per-sample mask and the
    # legacy thresholds are refused, so comparing them would warn about knobs
    # that no longer have any effect.
    uses_resp_cache = bool(str(getattr(args, 'resp_cache', '') or '').strip()
                           or getattr(args, 'resp_cache_key', None))
    corpus_keys = ['task_set', 'tasks', 'clip_stride']
    if uses_resp_cache:
        corpus_keys.append('resp_cache_key')
    else:
        corpus_keys += ['min_signal_spread', 'rail_touch_v']
    for key in corpus_keys:
        stage2 = getattr(stage2_args, key, None)
        stage3 = getattr(args, key, None)
        if stage2 is None or stage3 is None:
            continue
        if str(stage2).strip() != str(stage3).strip():
            print(f'[stage3] note: {key} differs from the Stage-2 run '
                  f'({stage2!r} -> {stage3!r}) -> this run is not evaluated '
                  f'on exactly the pre-trained corpus (allowed).')
    if uses_resp_cache and getattr(stage2_args, 'resp_cache_key', None) is None:
        print('[stage3] WARNING: the Stage-2 checkpoint records no '
              'resp_cache_key -- it predates the respiration-cache lineage, '
              'so its LABEL SPACE is the old raw/whole-file one. The run '
              'continues, but a comparable number needs a retrain.')

    # ``task_groups`` belongs to the same hazard class, but ONLY when a group is
    # actually SELECTED. With ``task_set`` and ``tasks`` both empty the mapping
    # is pure metadata: it labels the eval breakdown and cannot change which
    # sessions exist. So a changed tier definition MUST NOT warn there -- re-
    # grouping the labels of an existing k-fold Stage-3 run is a legitimate,
    # corpus-preserving operation (see data/task_groups.py).
    stage2_groups = _parse_recorded_groups(stage2_args)
    stage3_groups = _parse_recorded_groups(args)
    selecting = any(str(getattr(ns, key, '') or '').strip()
                    for ns in (stage2_args, args)
                    for key in ('task_set', 'tasks'))
    if selecting and stage2_groups is not None and stage3_groups is not None:
        if stage2_groups != stage3_groups:
            print('[stage3] note: task_groups differs from the Stage-2 run '
                  f'({stage2_groups!r} -> {stage3_groups!r}) while a group IS '
                  'selected -> this run is not evaluated on exactly the '
                  'pre-trained corpus (allowed).')

    return {key: {'stage2': stage2, 'stage3': stage3, 'match': same}
            for key, stage2, stage3, same in rows}


def resolve_resp_cache_key(args) -> Optional[str]:
    """The respiration cache DIRECTORY NAME, without building the dataset.

    The Stage-2/Stage-3 contract check runs BEFORE the data loaders exist, so
    the identity has to come from the cache itself. ``None`` when no cache is
    configured. Raises (with the field-by-field diff) on a grid mismatch -- a
    run must never train against a different label space silently.
    """
    root_in = str(getattr(args, 'resp_cache', '') or '').strip()
    if not root_in:
        return None
    from data import resp_cache as respc
    from data import resp_preprocess as rpp
    from data import tir_resp_dataset as trd
    raw = (getattr(args, 'raw_root', None) or getattr(args, 'data_path', None)
           or trd.default_raw_root())
    corpus = trd.discover_sessions(raw)
    want = respc.loader_params(raw, corpus, fs_in=rpp.FS_IN,
                               fs_out=float(getattr(args, 'fs', 100.0)))
    return respc.find_cache(root_in, want).name


def load_stage3_weights(model, output_dir: str, rule: str = 'fixed_epoch',
                        digest: Optional[str] = None) -> str:
    """Load the Stage-3 weights the SELECTION RULE delivered (I7).

    The report-time process is separate from the training process, so nothing
    resumes the weights for it: without this it would score an untrained model
    and report a number that looks plausible. ``fixed_epoch`` delivers the last
    checkpoint; ``best_val`` delivers ``best.pth`` (and nothing else).

    :returns: the path that was loaded.
    """
    if str(rule) == 'best_val':
        path = os.path.join(output_dir, 'best.pth')
        if not os.path.isfile(path):
            raise SystemExit(
                f'[stage3] val_select=best_val but no best.pth under '
                f'{output_dir}; the training run did not select a checkpoint.')
    else:
        candidates = sorted(glob.glob(os.path.join(output_dir, 'checkpoints',
                                                   'checkpoint-*.pth')))
        if not candidates:
            raise SystemExit(
                f'[stage3] no Stage-3 checkpoint under {output_dir}/checkpoints'
                f'; cannot score a run that never trained.')
        path = candidates[-1]

    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload.get('model', payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'[stage3] WARNING: {len(missing)} missing / {len(unexpected)} '
              f'unexpected tensor(s) loaded from {path}')

    recorded = getattr(payload.get('args'), 'partition_digest', None) \
        if isinstance(payload, dict) else None
    if digest and recorded and str(recorded) != str(digest):
        raise SystemExit(
            f'[stage3] partition contract VIOLATED: {path} was trained on '
            f'partition {str(recorded)[:12]}, this run is {str(digest)[:12]}. '
            f'Its test subjects are not the ones this run holds out.')
    print(f'[stage3] loaded {path} (epoch {payload.get("epoch") if isinstance(payload, dict) else "?"}, '
          f'digest {str(recorded)[:12] if recorded else "unrecorded"})')
    return path


def check_stage2_fold_contract(plan, stage2_args) -> dict:
    """Verify the --finetune checkpoint belongs to THIS fold's partition.

    The protocol only holds if fold k's Stage-3 run starts from a Stage-2
    encoder that never saw fold k's test or validation subjects. Nothing in the
    tensor shapes can reveal a violation: the checkpoint loads cleanly either
    way, and the run looks healthy while the headline number is leaked. So
    compare the PARTITION DIGEST the checkpoint records against this run's.

    A digest is the only sound key: two runs can share ``fold == 2`` and still
    hold out different subjects (a different seed or stratifier), which is
    exactly the pair this guard exists to refuse.

    * both records carry a digest and they differ -> hard ``SystemExit``;
    * the checkpoint predates the protocol (no digest) -> loud warning, and the
      verdict is recorded as unverified rather than "OK";
    * this run has no plan -> nothing to compare, nothing claimed.
    """
    if stage2_args is None:
        print('[stage3] partition contract: the checkpoint records no args -> '
              'cannot verify which partition it was pre-trained on.')
        return {'verified': False, 'match': None,
                'reason': 'checkpoint records no args'}

    stage3_digest = plan.digest
    stage2_digest = getattr(stage2_args, 'partition_digest', None)
    if not stage2_digest:
        print(f'[stage3] WARNING: this run is partition {stage3_digest[:12]} '
              f'(fold {plan.fold}/{plan.n_folds}), but the Stage-2 checkpoint '
              f'records NO partition digest (it predates ADR-001). The pairing '
              f'cannot be verified and a cross-validated number from it may be '
              f'leaked; re-pretrain with runners/run_pretrain.py.')
        return {'verified': False, 'match': False,
                'stage3_digest': stage3_digest, 'stage2_digest': None}

    if str(stage2_digest) != str(stage3_digest):
        raise SystemExit(
            f'[stage3] partition contract VIOLATED: the --finetune checkpoint '
            f'was pre-trained on partition {stage2_digest[:12]}, but this run '
            f'uses {stage3_digest[:12]} (fold {plan.fold}/{plan.n_folds}). '
            f'Their held-out subjects differ, so the test role was seen during '
            f'pre-training. Point --finetune at the Stage-2 run of THIS fold.')
    print(f'[stage3] partition contract OK: checkpoint and run share digest '
          f'{stage3_digest[:12]} (fold {plan.fold}/{plan.n_folds}).')
    return {'verified': True, 'match': True, 'stage3_digest': stage3_digest,
            'stage2_digest': str(stage2_digest)}



def get_args():
    parser = argparse.ArgumentParser('Stage-3 waveform fine-tuning', add_help=True)
    add_common_args(parser)
    add_protocol_args(parser)

    # model / task
    parser.add_argument('--model', default='project_multimae_base', type=str,
                        help='project_multimae_{base,large} -> '
                             'Stage-2 multimodal encoder + temporal waveform '
                             'head (the Stage-3 path; the NAME sets the ViT '
                             'geometry and MUST match the Stage-2 run). Any '
                             'project_vit_* keeps the legacy 2-D single-image '
                             'ViT path (cannot consume the clip tensor).')
    parser.add_argument('--streams', default='rgb', type=str,
                        help='VISUAL streams fed at Stage 3 (rgb or rgb,tir) '
                             '-- 1-D physio input is the simulated failure, so '
                             'it is never fed. Must be a subset of the '
                             'Stage-2 streams.')
    parser.add_argument('--use_tir', action='store_true', default=False,
                        help='feed TIR as a second visual stream (default off: '
                             'the Stage-2 run in this project is rgb+bp, so '
                             'the TIR adapter has no trained weights).')
    parser.add_argument('--target', default='bp', choices=['bp', 'resp'],
                        help='which physiological waveform branch to train')
    parser.add_argument('--seq_len', default=0, type=int,
                        help='length of the predicted output waveform (samples). '
                             '0 => the aligned default: num_frames*temporal_'
                             'stride/fps*fs, i.e. the SAME window the input '
                             'clip covers (400 for a 4 s clip at 100 Hz).')
    parser.add_argument('--fs', default=100.0, type=float,
                        help='waveform sampling rate in Hz')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--physio_norm', default='zscore', type=str,
                        choices=['clip', 'zscore'],
                        help='target normalisation, applied by the DATASET: a '
                             'per-clip z-score (one mean/std from the clip\'s '
                             "own samples). 'clip' and 'zscore' are synonyms; "
                             'there is no raw / mean-only / session-relative '
                             'alternative -- a second target space would make '
                             'Stage 2 and Stage 3 incomparable. The recorded '
                             'streams are NOT zero-mean (BP4D BP is raw mmHg, '
                             'mean ~101), which is exactly why the offset has '
                             'to go and the target std is ~1 by construction.')
    # clip geometry -- MUST mirror the Stage-2 Stage-2 run the ckpt comes from
    parser.add_argument('--clip_duration', default=4.0, type=float,
                        help='window length in seconds; mirror Stage 2')
    parser.add_argument('--fps', default=25.0, type=float,
                        help='video frame rate; mirror Stage 2')
    parser.add_argument('--tubelet', default='2,16,16', type=str,
                        help='tubelet (t, ph, pw); mirror Stage 2')
    parser.add_argument('--temporal_stride', default=1, type=int,
                        help='intra-window frame decimation; mirror Stage 2')
    parser.add_argument('--sig_kernel', default=8, type=int,
                        help='signal samples per token; must satisfy '
                             'sig_kernel/fs == tubelet_t*temporal_stride/fps')
    parser.add_argument('--enc_embed_dim', default=0, type=int)
    parser.add_argument('--enc_depth', default=0, type=int)
    parser.add_argument('--enc_num_heads', default=0, type=int)
    parser.add_argument('--head_hidden', default=0, type=int,
                        help='hidden width of a 2-layer waveform head '
                             '(0 = single linear layer per time step)')
    parser.add_argument('--head_style', default='linear', type=str,
                        choices=['linear', 'transposed_conv'],
                        help="Stage-3 waveform head. 'linear' (default) = one "
                             'Linear(D -> samples_per_token) per tubelet step. '
                             "'transposed_conv' = the spec 4.1 learned 1-D "
                             'upsampling head (Conv1d + ConvTranspose1d, stride '
                             'samples_per_token); it cannot inherit the Stage-2 '
                             'head, so combine it with --head_init none.')
    parser.add_argument('--head_init', default='transfer', type=str,
                        choices=['transfer', 'none'],
                        help="how the Stage-3 head starts. 'transfer' "
                             '(default) = copy the Stage-2 heads.<target> '
                             'weights in (single-Linear head only). '
                             "'none' = always RANDOM -- what the spec 4.1 "
                             'discard-the-decoders recipe needs, so the '
                             'Stage-3 head never depends on the Stage-2 '
                             'decoder weights.')
    parser.add_argument('--layer_decay', default=1.0, type=float,
                        help='layer-wise LR decay exponent for fine-tuning '
                             '(spec 4.2, typical 0.65-0.75). 1.0 (default) = '
                             'OFF = a single parameter group. < 1.0 groups the '
                             'parameters by encoder depth (tokenizer most '
                             'decayed, head at 1.0) and multiplies the step LR '
                             'schedule by each group lr_scale.')
    parser.add_argument('--finetune', default=env_or('MODEL_PATH'), type=str,
                        help='Stage-2 pretrained encoder checkpoint to load: a '
                             'local path OR a variant spec (base, mae:large) '
                             'downloaded into <project_root>/models/initial')

    # data (implement BP4D+ in code/data/datasets.py)
    parser.add_argument('--data_set', default=env_or('DATA_SET', 'bp4d+'), type=str)
    parser.add_argument('--data_path', default=env_or('DATA_PATH'), type=str)
    parser.add_argument('--raw_root', default=env_or('RAW_DATA_PATH', ''),
                        type=str,
                        help='RAW BP4D+ tree for the ROI data sets (tir_roi* / '
                             'rgb_roi*); default $RAW_DATA_PATH. The subject '
                             'UNIVERSE is enumerated from this tree, so it must '
                             'be the same one Stage 2 used.')
    parser.add_argument('--tasks', default='', type=str,
                        help='thermal-ROI path (data_set: tir_roi*): explicit '
                             'comma list of task labels, e.g. T1,T2 '
                             '(empty=all); UNIONED with --task_set')
    parser.add_argument('--task_set', default='', type=str,
                        help="thermal-ROI path: named task group(s) from the "
                             "config's task_groups: block, e.g. 'low' or "
                             "'low,high' (empty=all); see data/task_groups.py")
    parser.add_argument('--task_groups', default='', type=str,
                        help='thermal-ROI path: group DEFINITIONS overriding '
                             "the config, e.g. 'low=T1|T2;high=T7|T8|T9'")
    parser.add_argument('--clip_stride', default=0.0, type=float,
                        help='window hop in SECONDS (0 = non-overlapping, i.e. '
                             'hop == clip_duration); < clip_duration gives '
                             'overlapping windows and more samples per session')
    parser.add_argument('--roi_padding', default=0.2, type=float,
                        help='per-side ROI margin for the thermal-ROI path '
                             '(data_set: tir_roi*); must mirror the Stage-2 run')
    parser.add_argument('--crop_rule', default='square',
                        choices=['box', 'square'], type=str,
                        help="thermal-ROI path: how the padded landmark box "
                             "becomes a crop. 'square' (default) squares it "
                             "first, so the rescale has ONE scale factor and "
                             "the anatomy is never deformed; 'box' is the "
                             'legacy per-axis geometry (A/B only). Changes WHICH '
                             'PIXELS reach the encoder -> must mirror Stage 2.')
    parser.add_argument('--max_sway_px', default=40.0, type=float,
                        help='thermal-ROI path: DROP a clip whose target-'
                             'landmark centroid travels further than this many '
                             'SOURCE pixels from its own within-clip median '
                             '(0 = gate disabled; the study value is 40). A '
                             'selection filter: it changes WHICH CLIPS EXIST, '
                             'never a stored pixel, so it is not in the cache '
                             'key -- but it must mirror the Stage-2 run. The '
                             'gate is task-biased (T4 keeps 80 % at 40 px), so '
                             'a per-task result must state how it was gated.')
    parser.add_argument('--roi_landmarks', default='', type=str,
                        help="ROI landmark set for the ROI paths. For the "
                             "thermal-ROI path (data_set: tir_roi*): a preset "
                             "name (nose_mouth | nostrils | nostril_mouth | "
                             "nose_tip) or a comma list of 1-indexed labels, "
                             "e.g. '10,21'; empty = the historical 12-point "
                             "nose+mouth set. For the RGB-ROI path (data_set: "
                             "rgb_roi*): empty or 'face' = ALL 49 landmarks "
                             "(the whole face), or the same presets / CSV.")
    parser.add_argument('--roi_cache', default=env_or('ROI_CACHE', ''), type=str,
                        help='tir_roi_resp only: OUT_ROOT of an offline TIR-ROI '
                             'cache (data/roi_cache.py), i.e. the parent of the '
                             "<CACHE_KEY>/ directories. Empty (default, or "
                             '$ROI_CACHE) = decode .wmv as before. Matched by KEY '
                             "against this run's roi_landmarks / roi_padding / "
                             'corpus, and the run ABORTS on a mismatch rather than '
                             'reading slightly wrong pixels. Shares ONE artifact '
                             'with the Stage-2 run it fine-tunes from (2 s vs 1 s '
                             'hop is irrelevant: the cache is hop-independent).')
    parser.add_argument('--resp_cache', default=env_or('RESP_CACHE', ''),
                        type=str,
                        help='tir_roi_resp only: OUT_ROOT of an offline '
                             'respiration cache (data/resp_cache.py), i.e. the '
                             'parent of the <CACHE_KEY>/ directories. Empty '
                             '(default, or $RESP_CACHE) = read Resp_Volts.txt '
                             'as before. Matched by KEY; the run ABORTS on a '
                             'mismatch. With a cache, --resp_fs IS the '
                             "artefact's output rate and must equal --fs, and "
                             '--rail_touch_v / --min_signal_spread must be 0: '
                             'the stored per-sample validity mask is the only '
                             'rule, so the label space is run-mean-removed '
                             'volts (a DIFFERENT lineage -- retrain).')
    parser.add_argument('--allow_roi_mismatch', action='store_true',
                        default=False,
                        help='run even when the ROI keys (roi_landmarks / '
                             'roi_padding / input_size) differ '
                             'from the ones recorded in the --finetune '
                             'checkpoint. Default OFF: a mismatched ROI '
                             'changes the pixels WITHOUT changing any tensor '
                             'shape, so it trains happily on an input the '
                             'encoder never saw -- refuse it unless the '
                             'deviation is deliberate.')
    parser.add_argument('--min_signal_spread', default=0.0, type=float,
                        help='thermal-ROI path: DROP a clip whose respiration '
                             'window spread (max-min) is below this many VOLTS '
                             '(0.0 = off; for the study use 0.1). The '
                             'DEAD-SIGNAL guard (a channel pinned just inside a '
                             'clamp), complementary to --rail_touch_v. MUST '
                             'equal the Stage-2 value, else the evaluated '
                             'corpus is not the pre-trained one.')
    parser.add_argument('--rail_touch_v', default=0.0, type=float,
                        help='thermal-ROI path: DROP a clip whose respiration '
                             'window contains ANY sample that touched the '
                             'recorder rail, i.e. abs(x) >= this many VOLTS '
                             '(0.0 = off; the study value is 9.90). Removes '
                             'both a dead/pinned sensor and a clipped '
                             'flat-topped trough, so no rail-valued label '
                             'survives. MUST equal the Stage-2 value, else the '
                             'evaluated corpus is not the pre-trained one.')
    parser.add_argument('--decode_scale', default=1, type=int,
                        choices=[1, 2, 4, 8],
                        help='rgb_roi only: JPEG decode scale. 1 = native full '
                             'decode. This is part of the VISUAL FRONT-END '
                             'CONTRACT: it must equal the Stage-2 value, or '
                             'the encoder sees pixels on a different scale.')
    parser.add_argument('--phys_fs', default=1000.0, type=float,
                        help='rgb_roi only: sample rate of the RAW 1-D files '
                             'under Physiology/<S>/<T>/ (1000 Hz nominal); '
                             'they are resampled onto the --fs grid. Must '
                             'equal the Stage-2 value.')

    # training
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    # NOTE: --eval_freq / --val_select / --val_criterion / --fold / seeds come
    # from add_protocol_args (ADR-001 section 2.3). They are declared there and
    # nowhere else, so a default can never disagree with the protocol.

    # optimizer
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=0.0, type=float)

    # joint loss weights (code/notes/plan/experiments/RGB_BP_plan.md §4)
    parser.add_argument('--alpha', default=1.0, type=float, help='L_time weight')
    parser.add_argument('--beta', default=1.0, type=float, help='L_Pearson weight')
    parser.add_argument('--gamma', default=1.0, type=float, help='L_MR-STFT weight')
    parser.add_argument('--fft_sizes', default='64,128,256', type=str,
                        help='MR-STFT FFT window sizes (comma separated)')
    parser.add_argument('--huber_beta', default=1.0, type=float,
                        help='L_time (SmoothL1) transition point in TARGET-STD '
                             'units. The per-clip z-scored target has std ~1, '
                             'so 1.0 is ~1 std: quadratic inside that band, '
                             'linear beyond. Lower it for a heavier (L1-like) '
                             'charge on large errors.')

    # spectral band for evaluation (plan: BP 1.0-2.5 Hz, RESP 0.16-0.4 Hz)
    parser.add_argument('--eval_band', default=None, type=str,
                        help='e.g. "1.0,2.5" to restrict spectral eval')
    parser.add_argument('--save_preds', action='store_true', default=False,
                        help='also write val preds.npy / targets.npy / '
                             'entries.json into --output_dir; required by '
                             'runners/run_evaluate_session.py for the '
                             'session-level assembly metrics')
    parser.add_argument('--eval_role', default='train', type=str,
                        choices=['train', 'val', 'test'],
                        help="which role this PROCESS builds. 'train' (default) "
                             "trains on train and evaluates val; 'test' is the "
                             "report-time process: it scores the fold's test "
                             "role ONCE and never steps an optimizer (I5).")
    parser.add_argument('--overwrite_test', action='store_true', default=False,
                        help='re-score a test role that already has a '
                             'test/metrics.json; the override is recorded in '
                             'the new artefact')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='load the checkpoint in --output_dir/checkpoints '
                             'and run ONLY the final validation pass + the '
                             'artefact dump (metrics_final.json, preds.npy, '
                             'entries.json, ...), then exit. No training, no '
                             'LR schedule. MUST be run SINGLE-PROCESS (1 GPU): '
                             'under DDP each rank only sees 1/world_size of the '
                             'val split, so the dump would be incomplete.')
    return parse_args_with_config(parser)


def main(args):
    from utils import is_main_process, resolve_amp

    device = init_env(args)
    # The fold plan is needed BEFORE any checkpoint is inspected: the Stage-2
    # pairing guard compares partition digests, so a checkpoint from another
    # partition must be refused before a single weight is loaded.
    plan = load_or_create_plan(args, stage='stage3')
    # Persisted into every checkpoint: the report-time process and any resume
    # compare it, so a run can never continue or score another partition.
    args.partition_digest = plan.digest
    amp_config = resolve_amp(getattr(args, 'amp_dtype', 'none'), device)
    print(f'[amp] {amp_config.describe()}')
    args.fft_sizes = _parse_fft_sizes(args.fft_sizes)
    args.eval_band = (tuple(float(x) for x in args.eval_band.split(','))
                      if args.eval_band else None)

    # the dataset (use_tir) and the model (streams) must agree, else the
    # regressor would receive a channel stack that does not match its adapters.
    # The RAW-tree thermal-ROI path (data_set: tir_roi*) is exempt: it delivers
    # the ROI crop of the thermal video ALONE, so `use_tir` (which only controls
    # the canonical paired dataset's rgb+tir concatenation) is irrelevant there.
    # The RGB-ROI path (data_set: rgb_roi*) is the RGB analogue: it delivers the
    # landmark face crop of the visible-light frames alone, and its streams must
    # be exactly ['rgb'] -- checked below.
    _streams = [s.strip() for s in str(args.streams).split(',') if s.strip()]
    from data import RGB_ROI_DATA_SETS, TIR_ROI_DATA_SETS
    _name = str(getattr(args, 'data_set', '') or '').strip().lower()
    _is_roi = _name in TIR_ROI_DATA_SETS
    _is_rgb_roi = _name in RGB_ROI_DATA_SETS
    if _is_roi:
        if _streams != ['tir']:
            raise SystemExit(
                f'data_set {args.data_set!r} feeds the thermal ROI crop alone, '
                f'so --streams must be exactly "tir" (got "{args.streams}").')
    elif _is_rgb_roi:
        if _streams != ['rgb']:
            raise SystemExit(
                f'data_set {args.data_set!r} feeds the RGB face ROI crop '
                f'alone, so --streams must be exactly "rgb" (got '
                f'"{args.streams}").')
    else:
        if args.use_tir and 'tir' not in _streams:
            raise SystemExit(
                f'--use_tir needs --streams to include tir (got '
                f'"{args.streams}").')
        if not args.use_tir and 'tir' in _streams:
            raise SystemExit(
                f'--streams includes tir ("{args.streams}") but --use_tir is '
                f'off, so the dataset returns RGB only. Drop tir or pass '
                f'--use_tir.')

    # ----- model: Stage-2 encoder + waveform regression head --------------- #
    # Resolve the respiration cache IDENTITY before the Stage-2 contract check:
    # it is a corpus-level contract (the LABEL SPACE) and that check runs before
    # the datasets are built.
    if str(getattr(args, 'resp_cache', '') or '').strip():
        try:
            args.resp_cache_key = resolve_resp_cache_key(args)
            print(f'[data] resp_cache_key (pre-flight) = {args.resp_cache_key}')
        except Exception as exc:
            raise SystemExit(f'[stage3] --resp_cache could not be resolved: '
                             f'{exc}')
    _fold_check: dict = {}
    use_multimae = str(args.model).startswith('project_multimae')
    if use_multimae:
        # Stage-3 proper: re-use the Stage-2 multimodal (tubelet, joint
        # space-time) encoder verbatim + a time-aligned regression head.
        from core.waveform_model import build_waveform_model, load_stage2_encoder
        model = build_waveform_model(args)
        print(f'Stage-3 model: {args.model} streams={list(model.streams)} '
              f'geometry num_frames={model.num_frames} input={model.input_size} '
              f'output_len={model.output_len} '
              f'({model.samples_per_token} samples/tubelet token)')
        if args.finetune:
            from models.pretrained import resolve_encoder_weights
            args.finetune = resolve_encoder_weights(args.finetune)
            _stage2 = load_stage2_encoder(model, args.finetune,
                                          target=args.target)
            # the ROI is the one input contract nothing else can check: same
            # tensor shape either way, different pixels.
            check_stage2_roi_contract(args, _stage2.get('stage2_args'),
                                      allow_mismatch=args.allow_roi_mismatch)
            # Same hazard class as the ROI check -- nothing in the tensor shapes
            # reveals it, and the run looks healthy while the number is leaked.
            _fold_check = check_stage2_fold_contract(plan,
                                                     _stage2.get('stage2_args'))
        else:
            print('[stage3] WARNING: no --finetune checkpoint -> the encoder '
                  'starts from random weights (this is NOT Stage-3 fine-tuning).')
    else:
        # legacy 2-D path (project_vit_*): a single patch-embed Conv2d cannot
        # consume the dataset's [B, C, T, H, W] clip tensor.
        from models import create_model
        # RGB (3) + TIR (tir_channels, 3 by default) share one patch-embed conv
        in_chans = 3 + (int(getattr(args, 'tir_channels', 3))
                        if args.use_tir else 0)
        if args.seq_len <= 0:
            args.seq_len = int(round(args.clip_duration * args.fs))
        model = create_model(args.model, num_classes=0, output_len=args.seq_len,
                             in_chans=in_chans)
        print('[stage3] WARNING: project_vit_* is the 2-D legacy path -- it '
              'loads NO Stage-2 encoder weights (key names differ) and expects '
              'a 4-D [B, C, H, W] input. Use project_multimae_* for the real '
              'Stage-3 fine-tune.')
        if args.finetune:
            raise SystemExit(
                'project_vit_* cannot load a Stage-2 multimodal checkpoint '
                '(Stage-2 keys are enc_blocks.*/adapters.*, ProjectViT keys are '
                'blocks.*/patch_embed.*): previously this silently loaded '
                'NOTHING. Use --model project_multimae_base (or drop '
                '--finetune for an intentionally from-scratch baseline).')
    model.to(device)

    # MR-STFT windows must fit the target waveform, and the resolved set must be
    # VISIBLE: WaveformJointLoss.extra_repr prints only alpha/beta/gamma, so a
    # wrong --fft_sizes used to leave no trace in the log at all. Validated HERE
    # (before the dataset scan) so a bad config fails in seconds, not after
    # walking the raw BP4D tree.
    _target_len = int(getattr(model, 'output_len', 0) or args.seq_len)
    args.fft_sizes, _fft_dropped = _resolve_fft_sizes(
        args.fft_sizes, _target_len, fs=args.fs)
    print(f'[stft] fft_sizes {tuple(args.fft_sizes)} for a {_target_len}-sample '
          f'target ({_target_len / args.fs:.2f} s at fs {args.fs:g} Hz)')
    if _fft_dropped:
        print(f'[stft] DROPPED {tuple(_fft_dropped)}: longer than the target, so '
              f'torch.stft would ZERO-PAD')
    _fft_advice = _fft_window_advisory(
        args.fft_sizes, min(args.eval_band) if args.eval_band else None,
        _target_len, fs=args.fs)
    if _fft_advice:
        print(f'[stft] NOTE: {_fft_advice}')

    # ----- data: the ADR-001 roles of this fold --------------------------- #
    from data import build_dataset

    eval_role = str(getattr(args, 'eval_role', 'train') or 'train').lower()
    if eval_role not in ('train', 'val', 'test'):
        raise SystemExit(
            f'--eval_role must be train, val or test; got {eval_role!r}')

    if eval_role == 'test':
        # REPORT TIME — a separate process from training. It builds ONLY the
        # test role and never steps an optimizer (I5), and it scores once: the
        # artefact it writes is the sentinel that refuses a second run.
        import json as _json

        from engines.waveform import predict_waveforms as _predict
        from evaluation import metrics as _metrics_test
        from evaluation.report import save_json as _save_json_test

        test_dir = os.path.join(args.output_dir, 'test')
        sentinel = os.path.join(test_dir, 'metrics.json')
        if os.path.exists(sentinel) and not getattr(args, 'overwrite_test', False):
            raise SystemExit(
                f'[stage3] fold {plan.fold}: the test role has already been '
                f'scored ({sentinel}). ADR-001 evaluates it ONCE; pass '
                f'--overwrite_test to supersede (the override is recorded).')

        dataset_test = build_dataset(args, plan, 'test')
        if len(dataset_test) == 0:
            _reason = getattr(dataset_test, 'empty_reason', 'unknown')
            print(f'[stage3] FOLD NOT SCORABLE: the test role of fold '
                  f'{plan.fold} has no usable clip. Reason: {_reason}')
            _save_json_test(sentinel, {
                'eval_role': 'test', 'skipped': True, 'skip_reason': _reason,
                'n_clips': 0, 'fold': plan.fold, 'n_folds': plan.n_folds,
                'partition_digest': plan.digest,
                'test_subjects': list(plan.roles['test'])})
            return
        # Load the weights the TRAINING process delivered. The rule comes from
        # its selection record, not from this command line: the report process
        # must score what was actually produced (I7).
        from protocol import ProtocolError, read_selection

        _rule = 'fixed_epoch'
        _selection_path = os.path.join(args.output_dir, 'selection.json')
        if os.path.isfile(_selection_path):
            try:
                _rule = str(read_selection(_selection_path).get('rule',
                                                                'fixed_epoch'))
            except ProtocolError as exc:
                raise SystemExit(f'[stage3] {exc}') from None
        else:
            print(f'[stage3] WARNING: no selection record at {_selection_path}; '
                  f'assuming fixed_epoch (the last checkpoint).')
        load_stage3_weights(model, args.output_dir, rule=_rule,
                            digest=plan.digest)

        data_loader_test = make_data_loader(args, dataset_test, shuffle=False,
                                           drop_last=False, num_workers=0)
        check_loader_not_empty(data_loader_test, 'test', args)
        pred, target = _predict(data_loader_test, model, device)
        final = dict(_metrics_test.time_domain_metrics(pred, target))
        try:
            final.update(_metrics_test.spectral_metrics(
                pred, target, fs=args.fs, band=args.eval_band))
        except ImportError:
            final.update({'psd_mae': float('nan'),
                          'hf_power_rel': float('nan')})
        os.makedirs(test_dir, exist_ok=True)
        np.save(os.path.join(test_dir, 'preds.npy'), pred)
        np.save(os.path.join(test_dir, 'targets.npy'), target)
        # The dump must carry the SAME entry schema a training run writes, or
        # the pooler cannot assemble sessions (`t_start`) or re-label tasks.
        entries = []
        for entry in getattr(dataset_test, 'entries', []):
            if isinstance(entry, dict):                 # ROI path: the clip dict
                meta, t_start = entry, entry.get('t_start', 0.0)
                source = 'raw_resp_volts'
                signals_file = meta.get('resp_file') or meta.get('signals_file') or ''
            else:                                       # paired path: (meta, t0)
                meta, t_start = entry[0], entry[1]
                source = 'canonical_signals_csv'
                signals_file = meta.get('signals_file', '')
            entries.append({'session': meta['session'],
                            't_start': float(t_start),
                            'signals_file': signals_file or '',
                            'reference_source': source,
                            'task': meta.get('task'),
                            'task_level': list(meta.get('task_level') or ())})
        with open(os.path.join(test_dir, 'entries.json'), 'w') as handle:
            _json.dump(entries, handle)
        _save_json_test(sentinel, {
            'eval_role': 'test', 'n_clips': int(pred.shape[0]),
            'fold': plan.fold, 'n_folds': plan.n_folds,
            'partition_digest': plan.digest,
            'test_subjects': list(plan.roles['test']),
            'overwritten': bool(getattr(args, 'overwrite_test', False)),
            'per_clip_metrics': final})
        print(f'[stage3] scored the test role of fold {plan.fold}: '
              f'{int(pred.shape[0])} clip(s) -> {sentinel}')
        return

    # Validation is mandatory in every stage (I4) and is the ONLY selection
    # signal (I7). Ordered, single-worker, unpadded: a val pass must be a
    # deterministic function of the weights.
    dataset_val = build_dataset(args, plan, 'val')
    if len(dataset_val) == 0:
        raise SystemExit(
            f'[stage3] the VAL role of fold {plan.fold} is EMPTY. ADR-001 '
            f'requires a validation role in every stage; check the partition '
            f'record and any --subjects filter '
            f'({getattr(dataset_val, "empty_reason", "unknown")}).')
    data_loader_val = make_data_loader(args, dataset_val, shuffle=False,
                                       drop_last=False, num_workers=0)
    check_loader_not_empty(data_loader_val, 'val', args)

    if args.eval_only:
        # No training split: --eval_only never steps an optimiser, and walking
        # the train entries would only give an evaluation-only job a new way
        # to fail.
        dataset_train, data_loader_train = None, None
    else:
        dataset_train = build_dataset(args, plan, 'train')
        if len(dataset_train) == 0:
            raise SystemExit(
                '[stage3] the TRAIN role is EMPTY -- nothing to train on. '
                f'Reason: {getattr(dataset_train, "empty_reason", "unknown")}. '
                'Relax the corpus cleaning (rail_touch_v / min_signal_spread) '
                'or choose another fold.')
        data_loader_train = make_data_loader(args, dataset_train, shuffle=True)
        # fail fast + name the cause instead of crashing later in the logger
        check_loader_not_empty(data_loader_train, 'train', args)

    # Record the respiration cache IDENTITY (directory name, machine-
    # independent) on the args so the checkpoint carries it: corpus-level
    # contract, never a tensor shape -- see the CORPUS_KEYS note above.
    _stats = getattr(dataset_train if dataset_train is not None
                     else dataset_val, 'stats', {}) or {}
    if _stats.get('resp_cache_key'):
        args.resp_cache_key = _stats['resp_cache_key']
        print(f'[data] resp_cache_key = {args.resp_cache_key} '
              f'(recorded in the checkpoint; Stage 3 re-checks it)')
    elif str(getattr(args, 'resp_cache', '') or '').strip():
        print('[data] WARNING: --resp_cache was passed but the dataset '
              'reports no resp_cache_key (not a TIR-ROI dataset?)')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ----- loss / optimizer / scaler -------------------------------------- #
    from utils import (NativeScalerWithGradNormCount, build_layer_decay_assigner,
                       create_optimizer)
    criterion = WaveformJointLoss(alpha=args.alpha, beta=args.beta,
                                  gamma=args.gamma, fft_sizes=args.fft_sizes,
                                  huber_beta=args.huber_beta)
    # layer-wise LR decay (spec 4.2): only when --layer_decay < 1.0. The assigner
    # puts a per-group lr_scale on the param groups and engines/finetune.py
    # multiplies the step schedule by it, so the decay survives the cosine curve.
    layer_assigner = build_layer_decay_assigner(
        model_without_ddp, getattr(args, 'layer_decay', 1.0))
    if layer_assigner is not None:
        optimizer = create_optimizer(
            args, model_without_ddp,
            get_num_layer=layer_assigner.get_layer_id,
            get_layer_scale=layer_assigner.get_scale)
    else:
        optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScalerWithGradNormCount()
    print(f'Criterion: {criterion}')

    # ----- step-level warmup + cosine LR schedule, then resume ------------ #
    # Mirrors run_pretrain: until now --warmup_epochs / --min_lr were parsed but
    # never used (the LR stayed constant at args.lr for every epoch) and
    # --resume was dead, so an interrupted 100-epoch run could not continue.
    from utils import (auto_resume_model, cosine_scheduler, get_world_size,
                       save_model)
    if args.eval_only:
        # Nothing will be optimised, so there is no step schedule to build (and
        # no train split to measure it against).
        steps_per_epoch, lr_schedule_values = 0, None
    else:
        if args.warmup_epochs >= args.epochs:
            raise SystemExit(
                f'--warmup_epochs ({args.warmup_epochs}) must be < --epochs '
                f'({args.epochs}); use --warmup_epochs 0 for a 1-epoch smoke run.')
        steps_per_epoch = max(
            1, (len(dataset_train) // (args.batch_size * get_world_size()))
            // max(1, args.update_freq))
        lr_schedule_values = cosine_scheduler(
            args.lr, args.min_lr, args.epochs, steps_per_epoch,
            warmup_epochs=args.warmup_epochs)
    auto_resume_model(args, model_without_ddp, optimizer, loss_scaler)
    start_epoch = int(getattr(args, 'start_epoch', 0) or 0)
    if args.eval_only:
        # Evaluate the checkpoint already sitting in --output_dir/checkpoints
        # and stop. Starting the loop at --epochs makes it an empty
        # ``range(args.epochs, args.epochs)``, so control falls straight
        # through to the final validation + artefact dump at the end of main().
        # NOTE the guard is on WORLD SIZE, not args.distributed: Slurm exports
        # SLURM_PROCID even for a 1-task batch step, so a legitimate single-GPU
        # eval is 'distributed' with world_size 1 and still sees the whole val
        # set (DistributedSampler with num_replicas=1 is the identity).
        if get_world_size() > 1:
            raise SystemExit(
                '--eval_only must run SINGLE-PROCESS (one GPU, no '
                'torchrun/srun launcher): each rank would only see '
                f'1/{get_world_size()} of the val split, so preds.npy / '
                'entries.json (and the metrics computed from them) would cover '
                'a shard, not the whole val set. Submit with GPUS=1.')
        print(f'[stage3] --eval_only: training skipped; evaluating the '
              f'checkpoint in {args.output_dir}/checkpoints (world size '
              f'{get_world_size()}).')
        start_epoch = args.epochs
    elif start_epoch >= args.epochs:
        raise SystemExit(
            f'nothing to do: the checkpoint in {args.output_dir}/checkpoints is '
            f'already at epoch {start_epoch} and --epochs is {args.epochs}. '
            f'Raise --epochs to train longer, or point --output_dir elsewhere '
            f'/ remove latest_checkpoint.txt to start from scratch.')
    if not args.eval_only:
        print(f'LR schedule: {args.lr:g} -> {args.min_lr:g} over {args.epochs} '
              f'epoch(s), warmup {args.warmup_epochs} epoch(s), '
              f'{steps_per_epoch} steps/epoch; start_epoch={start_epoch}')

    # ----- training loop -------------------------------------------------- #
    # artefact helpers are needed INSIDE the loop (metrics.jsonl + the refreshed
    # training_curves.png), so import them before it starts
    from evaluation.report import (append_jsonl, plot_training_curves,
                                   plot_waveform_panel, save_json)
    from protocol import SelectionTracker, make_eval_record, write_selection

    # Selection is one object shared with Stage 2 and the ONLY writer of
    # best.pth, so a delivered checkpoint can only come from the VAL role (I7).
    tracker = SelectionTracker(
        epochs=args.epochs, rule=getattr(args, 'val_select', 'fixed_epoch'),
        criterion=getattr(args, 'val_criterion', 'joint_loss'),
        eval_freq=int(getattr(args, 'eval_freq', 0) or 0),
        n_val_clips=len(dataset_val),
        n_train_clips=len(dataset_train) if dataset_train is not None else 0,
        dev_smoke=bool(getattr(args, 'dev_smoke', False)))
    history = []                    # one dict per evaluated epoch (see below)
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_waveform(
            model=model, criterion=criterion, data_loader=data_loader_train,
            optimizer=optimizer, device=device, epoch=epoch,
            loss_scaler=loss_scaler, max_norm=args.clip_grad,
            update_freq=args.update_freq,
            lr_schedule_values=lr_schedule_values,
            start_steps=epoch * steps_per_epoch,
            amp_config=amp_config, loss_in_fp32=True)

        if tracker.should_eval(epoch):
            val = evaluate_waveform_loss(data_loader_val, model, criterion,
                                         device)
            value = (val['val_loss'] if tracker.criterion == 'joint_loss'
                     else val['components'].get('pearson', float('nan')))
            improved = tracker.observe(epoch, value)
            print(f'[epoch {epoch}] {args.target}: val_loss '
                  f'{val["val_loss"]:.4f} ('
                  + ', '.join(f'{k}={v:.4f}' for k, v in
                              val['components'].items()
                              if k in ('time', 'pearson', 'stft'))
                  + f') over {val["n_clips"]} clip(s)'
                  + ('  <- best' if improved else ''))
            record = {'epoch': epoch, 'target': args.target,
                      'train_loss': (train_stats or {}).get('loss'),
                      'lr': (train_stats or {}).get('lr'),
                      **val['components']}
            history.append(record)
            if is_main_process():
                # the ADR section 2.3 evaluation record, then the plotting line
                append_jsonl(os.path.join(args.output_dir, 'metrics.jsonl'),
                             make_eval_record(
                                 epoch=epoch, value_kind=tracker.criterion,
                                 criterion_value=value,
                                 counts={'clips': val['n_clips'],
                                         'subjects': len(plan.roles['val'])},
                                 seed={'val_seed': plan.val_seed},
                                 components=val['components'],
                                 pinned={}, role_digest=plan.digest))
                plot_training_curves(
                    history,
                    os.path.join(args.output_dir, 'training_curves.png'),
                    title=f'{args.target} | {args.model} | input {args.input_size} '
                          f'| {args.epochs} epochs | fold {plan.fold} '
                          f'({tracker.rule})')
            if improved and tracker.writes_best_checkpoint and is_main_process():
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save({'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'partition_digest': plan.digest},
                           os.path.join(args.output_dir, 'best.pth'))

        if is_main_process() and (epoch % args.save_ckpt_freq == 0
                                  or epoch + 1 == args.epochs):
            save_model(args, epoch, model, model_without_ddp, optimizer,
                       loss_scaler)

    # ----- final evaluation artefacts: figures + JSON (not just stdout) ----- #
    if is_main_process():
        from evaluation import metrics as _metrics
        
        # A TRAINING run reports its VAL role; the test role is scored by its
        # own process (--eval_role test), so it is never built here (I5).
        pred, target = predict_waveforms(data_loader_val, model, device)
        eval_role = 'val'

        if pred.size:
            final = dict(_metrics.time_domain_metrics(pred, target))
            try:
                final.update(_metrics.spectral_metrics(
                    pred, target, fs=args.fs, band=args.eval_band))
            except ImportError:
                final.update({'psd_mae': float('nan'),
                              'hf_power_rel': float('nan')})
            n_pred = int(pred.shape[0])
            _ds_eval = dataset_val
            _tsel = getattr(_ds_eval, 'task_selection', None)
            selection = tracker.record()
            selection['partition_digest'] = plan.digest
            write_selection(os.path.join(args.output_dir, 'selection.json'),
                            selection)
            save_json(os.path.join(args.output_dir, 'metrics_final.json'), {
                'target': args.target, 'model': args.model,
                'task_set': list(getattr(_tsel, 'requested', ()) or ()),
                'task_groups': {k: list(v) for k, v in
                                getattr(_tsel, 'levels', {}).items()},
                'split_stratify': plan.split_stratify, 'n_clips': n_pred,
                'fs': args.fs, 'clip_duration': args.clip_duration,
                'input_size': args.input_size, 'sig_kernel': args.sig_kernel,
                'physio_norm': args.physio_norm, 'finetune': args.finetune,
                'epochs': args.epochs,
                'selection': selection['rule'],
                'selection_record': selection,
                'eval_role': eval_role,
                # fold provenance: the fold this run executed, the partition it
                # used, and whether the Stage-2 encoder came from that same
                # partition (a mismatch is a hard error, not a warning).
                'fold': plan.fold, 'n_folds': plan.n_folds,
                'partition_digest': plan.digest,
                'val_subjects': list(plan.roles['val']),
                'fold_contract': _fold_check,
                # loss provenance: the L_time shape must be readable off the
                # artefact, not only off the job log.
                'loss': {'time': 'smoothl1',
                         'huber_beta': criterion.huber_beta,
                         'alpha': args.alpha, 'beta': args.beta,
                         'gamma': args.gamma,
                         'fft_sizes': list(args.fft_sizes)},
                'per_clip_metrics': final})
            # Session labels let the panel SPREAD its clips over different
            # sessions. Without them it plotted rows 0..max_clips-1, i.e. always
            # the first session -- which made a run look solved when only that
            # session worked (measured 2026-09-26: +0.961 for the session owning
            # rows 0-3 vs +0.139 for the whole val split).
            try:
                _val_sessions = [
                    (e['session'] if isinstance(e, dict) else e[0]['session'])
                    for e in _ds_eval.entries]
            except Exception as _exc:
                _val_sessions = None
                print(f'[stage3] WARNING: no {eval_role} session labels '
                      f'({_exc}); the prediction panel will show the FIRST '
                      f'clips, which may all come from ONE session.')
            plot_waveform_panel(
                pred, target, fs=args.fs,
                out_png=os.path.join(args.output_dir, 'predictions_final.png'),
                title=(f'{args.target}: predicted vs target, {eval_role} split '
                       f'(z-scored)'),
                band=args.eval_band, metrics=final,
                sessions=_val_sessions)
            print(f'[stage3] wrote metrics_final.json + predictions_final.png to '
                  f'{args.output_dir}')
            if args.save_preds:
                np.save(os.path.join(args.output_dir, 'preds.npy'), pred)
                np.save(os.path.join(args.output_dir, 'targets.npy'), target)

                # ``entries`` has TWO layouts: PairedSessionDataset stores
                # (session_meta_dict, t_start) 2-tuples, the RAW-tree thermal-ROI
                # dataset stores the clip dicts directly. Normalise both to the
                # {'session', 't_start', 'signals_file'} record that
                # run_evaluate_session.py consumes.
                entries = []
                for e in _ds_eval.entries:
                    if isinstance(e, dict):                    # thermal-ROI path
                        meta, t_start = e, e.get('t_start', 0.0)
                        sig = (meta.get('resp_file')
                               or meta.get('signals_file') or '')
                        source = 'raw_resp_volts'
                    else:                                      # paired path
                        meta, t_start = e[0], e[1]
                        sig = meta.get('signals_file', '')
                        source = 'canonical_signals_csv'
                    entries.append({'session': meta['session'],
                                    't_start': float(t_start),
                                    'signals_file': sig or '',
                                    'reference_source': source,
                                    'task': meta.get('task'),
                                    'task_level': list(meta.get('task_level')
                                                       or ())})
                if len(entries) != n_pred:
                    print(f'[stage3] WARNING: {len(entries)} dataset entries vs '
                          f'{n_pred} predictions -- the loader was sharded '
                          f'(DDP), so the dump is incomplete; re-run the '
                          f'evaluation single-process for session assembly.')
                save_json(os.path.join(args.output_dir, 'entries.json'), entries)
                print(f'[stage3] saved preds.npy / targets.npy / entries.json '
                      f'({n_pred} clips) for session-level evaluation')


if __name__ == '__main__':
    args = get_args()
    main(args)
