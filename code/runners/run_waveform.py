"""Stage-3 downstream waveform fine-tuning (BVP or RESP branch).

Implements docs/ImplementationPlan.md Stage 3: simulated full sensor failure
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

    python runners/run_waveform.py -c configs/finetune/bvp.yaml
    python runners/run_waveform.py -c configs/finetune/resp.yaml

The ``bp4d+`` dataset (``data/paired_dataset.py``) yields ``(samples, [B, T])``
with ``samples = [B, 3, T, H, W]`` (``use_tir`` off) or ``[B, 6, T, H, W]``.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core import WaveformJointLoss
from engines import evaluate_waveforms, train_one_epoch_waveform
from runners._common import (add_common_args, env_or, init_env,
                             make_data_loader, parse_args_with_config)


def _parse_fft_sizes(s):
    return tuple(int(x) for x in str(s).split(',') if x.strip())


def get_args():
    parser = argparse.ArgumentParser('Stage-3 waveform fine-tuning', add_help=False)
    add_common_args(parser)

    # model / task
    parser.add_argument('--model', default='project_multimae_base', type=str,
                        help='project_multimae_{tiny,small,base,large,huge} -> '
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
                             'the Stage-2 run in this project is rgb+bvp, so '
                             'the TIR adapter has no trained weights).')
    parser.add_argument('--target', default='bvp', choices=['bvp', 'resp', 'eda'],
                        help='which physiological waveform branch to train')
    parser.add_argument('--seq_len', default=0, type=int,
                        help='length of the predicted output waveform (samples). '
                             '0 => the aligned default: num_frames*temporal_'
                             'stride/fps*fs, i.e. the SAME window the input '
                             'clip covers (400 for a 4 s clip at 100 Hz).')
    parser.add_argument('--fs', default=100.0, type=float,
                        help='waveform sampling rate in Hz')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--signal_norm', default='zscore', type=str,
                        choices=['none', 'ac', 'zscore'],
                        help='per-clip target normalisation. The recorded '
                             'streams are NOT zero-mean (BP4D BVP is raw mmHg, '
                             'mean ~101): with "none" the head must fit a ~100 '
                             'DC offset and the MR-STFT term is dominated by '
                             'the 0 Hz bin instead of the pulsatile band. '
                             '"zscore" (default) = zero-mean/unit-std per '
                             'window, the standard rPPG convention; "ac" '
                             'removes the mean only.')
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
    parser.add_argument('--finetune', default=env_or('MODEL_PATH'), type=str,
                        help='Stage-2 pretrained encoder checkpoint to load: a '
                             'local path OR a variant spec (base, mae:large) '
                             'downloaded into <project_root>/models/initial')

    # data (implement BP4D+ in code/data/datasets.py)
    parser.add_argument('--data_set', default=env_or('DATA_SET', 'bp4d+'), type=str)
    parser.add_argument('--data_path', default=env_or('DATA_PATH'), type=str)

    # training
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)

    # optimizer
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=0.0, type=float)

    # joint loss weights (docs/ImplementationPlan.md)
    parser.add_argument('--alpha', default=1.0, type=float, help='L_time weight')
    parser.add_argument('--beta', default=1.0, type=float, help='L_Pearson weight')
    parser.add_argument('--gamma', default=1.0, type=float, help='L_MR-STFT weight')
    parser.add_argument('--fft_sizes', default='64,128,256', type=str,
                        help='MR-STFT FFT window sizes (comma separated)')

    # spectral band for evaluation (plan: BVP 1.0-2.5 Hz, RESP 0.16-0.4 Hz)
    parser.add_argument('--eval_band', default=None, type=str,
                        help='e.g. "1.0,2.5" to restrict spectral eval')
    return parse_args_with_config(parser)


def main(args):
    from utils import is_main_process

    device = init_env(args)
    args.fft_sizes = _parse_fft_sizes(args.fft_sizes)
    args.eval_band = (tuple(float(x) for x in args.eval_band.split(','))
                      if args.eval_band else None)

    # the dataset (use_tir) and the model (streams) must agree, else the
    # regressor would receive a channel stack that does not match its adapters
    _streams = [s.strip() for s in str(args.streams).split(',') if s.strip()]
    if args.use_tir and 'tir' not in _streams:
        raise SystemExit(
            f'--use_tir needs --streams to include tir (got "{args.streams}").')
    if not args.use_tir and 'tir' in _streams:
        raise SystemExit(
            f'--streams includes tir ("{args.streams}") but --use_tir is off, '
            f'so the dataset returns RGB only. Drop tir or pass --use_tir.')

    # ----- model: Stage-2 encoder + waveform regression head --------------- #
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
            load_stage2_encoder(model, args.finetune)
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

    # ----- data (implement bp4d+ first) ----------------------------------- #
    from data import build_dataset
    dataset_train = build_dataset(is_train=True, test_mode=False, args=args)
    dataset_val = build_dataset(is_train=False, test_mode=False, args=args)
    data_loader_train = make_data_loader(args, dataset_train, shuffle=True)
    data_loader_val = make_data_loader(args, dataset_val, shuffle=False,
                                       drop_last=False)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ----- loss / optimizer / scaler -------------------------------------- #
    from utils import NativeScalerWithGradNormCount, create_optimizer
    criterion = WaveformJointLoss(alpha=args.alpha, beta=args.beta,
                                  gamma=args.gamma, fft_sizes=args.fft_sizes)
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScalerWithGradNormCount()
    print(f'Criterion: {criterion}')

    # ----- training loop -------------------------------------------------- #
    from utils import save_model
    best_pearson = -float('inf')
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch_waveform(
            model=model, criterion=criterion, data_loader=data_loader_train,
            optimizer=optimizer, device=device, epoch=epoch,
            loss_scaler=loss_scaler, max_norm=args.clip_grad,
            update_freq=args.update_freq)

        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            stats = evaluate_waveforms(data_loader_val, model, device,
                                       fs=args.fs, band=args.eval_band)
            print(f'[epoch {epoch}] {args.target}: {stats}')
            pearson = stats.get('pearson', -1.0)
            if is_main_process() and pearson > best_pearson:
                best_pearson = pearson
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save({'model': model_without_ddp.state_dict(),
                            'epoch': epoch}, os.path.join(args.output_dir, 'best.pth'))

        if is_main_process() and (epoch % args.save_ckpt_freq == 0
                                  or epoch + 1 == args.epochs):
            save_model(args, epoch, model, model_without_ddp, optimizer,
                       loss_scaler)

    print(f'Best Pearson ({args.target}): {best_pearson:.4f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
