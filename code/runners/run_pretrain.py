"""Masked pre-training entry point.

Usage (from ``code/``)::

    python runners/run_pretrain.py -c configs/pretrain/example.yaml

NOTE: masked pre-training needs your (a) dataset in ``data/datasets.py`` and
(b) an encoder+decoder that returns ``(pred, target, mask)`` (port from
``tmp/videomae/modeling_pretrain.py`` / ``tmp/MultiMAE/multimae/multimae.py``).
Until then this script wires everything except those two thesis-specific parts.
"""
import argparse
import os
import sys

# allow running either `python runners/run_pretrain.py` or `python -m runners.run_pretrain`
# from the code/ root by ensuring code/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engines import train_one_epoch_pretrain
from runners._common import (add_common_args, env_or, init_env,
                             make_data_loader, parse_args_with_config)


def get_args():
    parser = argparse.ArgumentParser('Project MAE pre-training', add_help=False)
    add_common_args(parser)

    # model
    parser.add_argument('--model', default='project_multimae_base', type=str,
                        help='registered model name. project_multimae_{base,'
                             'large} run the multimodal masked pre-training '
                             '(the NAME sets the ViT geometry); any project_vit_* '
                             'runs the single-stream path')
    # ViT geometry comes from the --model variant; 0 = "unset" => take it from
    # the name. Set a value only to OVERRIDE (e.g. an ablation).
    parser.add_argument('--enc_embed_dim', default=0, type=int)
    parser.add_argument('--enc_depth', default=0, type=int)
    parser.add_argument('--enc_num_heads', default=0, type=int)
    # data / clip geometry (the multimodal dataset & model read these)
    parser.add_argument('--data_path', default=env_or('DATA_PATH'), type=str)
    parser.add_argument('--data_set', default='bp4d+', type=str,
                        help="Stage-2 data source: 'bp4d+' (default) = the "
                             "canonical paired layout; 'tir_roi' = the ADD-ON "
                             "thermal-ROI + respiration dataset built from the "
                             "RAW tree (see code/TirROI_Resp_plan.md).")
    parser.add_argument('--raw_root', default=env_or('RAW_DATA_PATH', ''),
                        type=str,
                        help='raw BP4D root for data_set=tir_roi; default: '
                             '$RAW_DATA_PATH, else <repo>/data/raw/BP4D')
    parser.add_argument('--roi_padding', default=0.2, type=float,
                        help='tir_roi only: fraction of the landmark-box '
                             'extent added on EACH side before the crop')
    parser.add_argument('--subjects', default='', type=str,
                        help='tir_roi only: comma list of subjects (empty=all)')
    parser.add_argument('--tasks', default='', type=str,
                        help='tir_roi only: comma list of tasks (empty=all)')
    parser.add_argument('--fs', default=100.0, type=float,
                        help='signal sample rate (Hz)')
    parser.add_argument('--fps', default=25.0, type=float,
                        help='RGB/TIR frame rate (Hz)')
    parser.add_argument('--clip_duration', default=4.0, type=float,
                        help='window length in seconds per MAE sample')
    parser.add_argument('--clip_stride', default=0.0, type=float,
                        help='window stride in seconds. < clip_duration yields '
                             'OVERLAPPING windows (more samples/session, e.g. '
                             'to reach 10-20k clips on the full HPC data). '
                             '0 => stride = clip_duration (non-overlapping).')
    parser.add_argument('--temporal_stride', default=1, type=int,
                        help='decimate frames INSIDE a window: keep 1 frame '
                             'every N (1 = every frame; 2/4/8 -> 12.5/6.25/'
                             '3.125 fps at 25 fps nominal). Distinct from '
                             '--clip_stride (window hop in seconds). Sets the '
                             'video geometry num_frames = clip_duration*fps/N.')
    parser.add_argument('--seq_len', default=0, type=int,
                        help='signal samples per window (0 => clip_duration*fs)')
    parser.add_argument('--input_size', default=64, type=int,
                        help='frame short-side resize/crop (square)')
    # multimodal MAE (Stage 2)
    parser.add_argument('--streams', default='rgb,tir,bp', type=str,
                        help='comma list of pretraining streams. Stage-2 '
                             'requires >=2: >=1 of rgb,tir (video) AND >=1 '
                             'of bp,resp,eda (1-D physio, Stage-3 target), '
                             'e.g. rgb,bp or rgb,tir,bp,resp,eda')
    parser.add_argument('--tubelet', default='2,16,16', type=str,
                        help='tubelet (t, ph, pw) for the video tokenizer')
    parser.add_argument('--mask_ratio_rgb', default=0.75, type=float)
    parser.add_argument('--mask_ratio_tir', default=0.50, type=float)
    parser.add_argument('--mask_ratio_bp', default=0.90, type=float)
    parser.add_argument('--mask_ratio_resp', default=0.90, type=float)
    parser.add_argument('--mask_ratio_eda', default=0.90, type=float)
    # 1-D masking PATTERN. 'random' (default) = the historical scattered
    # dropout, byte-identical for every existing config. 'span' = contiguous
    # blocks, which removes the "interpolate the gap from the visible
    # neighbours" shortcut of a scattered mask (see
    # code/SpanMask_PhysioSignals.md). Span geometry is per stream and derived
    # from mask_span_s + the stream's ratio; only the span PLACEMENT is random.
    parser.add_argument('--physio_mask', default='random', type=str,
                        choices=['random', 'span'],
                        help="1-D masking pattern: 'random' (default, "
                             "scattered) or 'span' (contiguous blocks; then "
                             "span length x count are derived from "
                             "--mask_span_s and the per-stream mask ratio)")
    parser.add_argument('--mask_span_s', default='', type=str,
                        help="span masking only: span length in SECONDS, "
                             "either one value for all physio streams, a "
                             "stream=seconds CSV ('resp=4.0,bp=1.0'), or a "
                             "YAML mapping. Empty = per-stream defaults "
                             "(bp 1.0, resp 4.0, eda 8.0 = one target period)"
                             ". The span COUNT is derived (n_spans = "
                             "round(ratio * n_signal / span_tokens)) so that "
                             "ratio x clip length is preserved.")
    parser.add_argument('--loss_weights', default='', type=str,
                        help='FULL per-stream override of the per-modality '
                             'masked-MSE weights: comma list, ONE value per '
                             '--streams modality in order. Empty (default) = '
                             'policy: lambda 1.0 for visual streams and '
                             'lambda --signal_weight for physio streams.')
    parser.add_argument('--signal_weight', default=0.5, type=float,
                        help='weight of each physio (1-D) stream in the '
                             'masked-MSE weighted sum (visual streams fixed '
                             'at 1.0). Tune in ~[0.5, 1.0] from the NORMALIZED '
                             'variance of the physio stream\'s masked target '
                             'tokens: high variance -> ~0.5 so the 1-D signal '
                             'does not dominate the gradient; low variance -> '
                             '~1.0 so it is not ignored.')
    parser.add_argument('--target_norm', default='token', type=str,
                        choices=['token', 'clip'],
                        help="normalisation of the 1-D reconstruction target. "
                             "'token' (default) = per-token z-score: every "
                             'sig_kernel window is rescaled by its OWN '
                             "mean/std. 'clip' = per-clip z-score: ONE "
                             'mean/std per (sample, stream), so the token '
                             'windows reassemble into a coherent waveform and '
                             'the Stage-2 target space matches Stage 3. REQUIRED '
                             'to be clip when --spectral_weight > 0.')
    parser.add_argument('--spectral_weight', default=0.0, type=float,
                        help='weight of the multi-resolution STFT MAGNITUDE '
                             'loss (see --spectral_fft_sizes) on the ASSEMBLED '
                             '1-D waveform of every physio stream. 0.0 '
                             '(default) = OFF = the previous masked-MSE-only '
                             'objective. Requires --target_norm clip. The '
                             'per-token masked MSE constrains amplitude only, '
                             'so a low-frequency surrogate can lower it '
                             'without modelling the cardiac/respiratory cycle; '
                             'this term gives the shared encoder a direct '
                             'gradient on periodicity. Start ~0.1 and tune.')
    parser.add_argument('--spectral_weights', default='', type=str,
                        help='FULL per-stream override of the spectral '
                             'weights: comma list, ONE value per --streams '
                             'modality in order (same convention as '
                             '--loss_weights; empty = --spectral_weight for '
                             'every physio stream, 0.0 for the video streams). '
                             'A positive value on a video stream is rejected.')
    parser.add_argument('--spectral_fft_sizes', '--fft_sizes',
                        dest='spectral_fft_sizes', default='', type=str,
                        help='MR-STFT FFT window sizes in SAMPLES, resolved PER '
                             'PHYSIO STREAM (a window must span >= 1 period of '
                             "the band it polices, so BP 1-2.5 Hz and RESP "
                             '0.16-0.4 Hz cannot share one set). Forms: '
                             "'64,128,256' = the SAME windows for every physio "
                             "stream; 'resp=128/256/512,bp=64/128/256' = per "
                             'stream (the COMMA separates streams, "/" '
                             'separates the windows of one stream); or a YAML '
                             'mapping {resp: [128, 256, 512]}. Empty/"auto" = '
                             'the per-modality defaults (SPECTRAL_FFT_DEFAULTS: '
                             'bp/resp 64,128,256, eda 256,512,1024), which '
                             'mirror configs/finetune/*.yaml one for one so a '
                             'stream gets the same spectral objective in Stage '
                             '2 and Stage 3. --fft_sizes is accepted as the '
                             'Stage-3 alias. At fs=100 Hz: 64 -> 1.56 Hz, '
                             '128 -> 0.78 Hz, 256 -> 0.39 Hz bins. Windows '
                             'longer than the clip are DROPPED and logged (the '
                             'model raises only if none remain, so eda needs a '
                             'clip of >= 2.56 s).')
    parser.add_argument('--spectral_hop_ratio', default=0.25, type=float,
                        help='STFT hop as a fraction of the FFT window '
                             '(0.25 = 75 %% overlap).')
    parser.add_argument('--dec_depth', default=2, type=int)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--sig_kernel', default=8, type=int,
                        help='signal token window (samples per token). MUST be '
                             'time-aligned with the video tubelet: one token '
                             'must cover the same seconds in every stream, '
                             'i.e. sig_kernel/fs == tubelet_t*temporal_stride/'
                             'fps (default 8/100 == 2*1/25 = 80 ms). The model '
                             'refuses to build on a misaligned geometry.')
    parser.add_argument('--pos_init', default='sincos3d', type=str,
                        choices=['sincos3d', 'random'],
                        help="positional-embedding init: 'sincos3d' (default) "
                             'initialises the video positions with a 3-D '
                             '(t, h, w) sincos grid and the physio positions '
                             'with the same 1-D temporal sincos (a space-time '
                             'prior at init, since the Stage-1 MAE pos_embed '
                             'cannot be reused); \'random\' = trunc_normal.')
    parser.add_argument('--pretrained_encoder', default='', type=str,
                        help='Stage-1 ViT checkpoint to initialise the shared '
                             'encoder from: a local path OR a variant spec, '
                             'e.g. base = videomae:base (downloads into '
                             '<project_root>/models/initial) or mae:base as '
                             'the 2-D control. VideoMAE is the default for '
                             'base/large: its patch embed IS a tubelet '
                             'Conv3d(3, D, (2,16,16)), so the tokenizer '
                             'transfers verbatim (a 2-D MAE source is '
                             'boxcar-inflated and leaves the model '
                             'motion-blind at init)')
    parser.add_argument('--inflate_rgb_patch', default=1, type=int,
                        choices=[0, 1],
                        help='1 (default) = transfer the RGB patch embed: a '
                             '3-D Conv3d source is copied verbatim, a 2-D '
                             'Conv2d source is averaged over the tubelet. '
                             '0 = leave the tokenizer random (ablation: '
                             'encoder blocks only)')
    # training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    # optimizer / lr  (official-MAE semantics)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', default=None, type=float,
                        help='peak (absolute) learning rate. Omit it to derive '
                             'the LR from --blr via lr = blr * batch_size * '
                             'world_size / 256 (large-batch MAE convention). '
                             'For small-batch LOCAL runs pass --lr explicitly: '
                             'the scaled LR would be ~1e-6 and the model would '
                             'not learn.')
    parser.add_argument('--blr', default=1.5e-4, type=float,
                        help='base lr, only used when --lr is not set: '
                             'lr = blr * batch_size * world_size / 256')
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--warmup_epochs', default=40, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=0.0, type=float)
    return parse_args_with_config(parser)


def main(args):
    from utils import get_world_size, is_main_process

    device = init_env(args)

    # MAE linear-scaling rule: only derive the absolute LR from --blr when the
    # user did not pass --lr explicitly. Otherwise blr*batch/256 on a batch-1/2
    # local run silently yields ~1e-6 and the weights never move (flat loss).
    if args.lr is None:
        args.lr = args.blr * args.batch_size * get_world_size() / 256
        print(f'LR derived from blr (blr*batch*world/256): {args.lr:.3e}')
    else:
        print(f'LR set explicitly (--lr): {args.lr:.3e}')

    # ----- model ---------------------------------------------------------- #
    if args.model.startswith('project_multimae'):
        from core.multimae import build_pretraining_model, load_pretrained_encoder
        model = build_pretraining_model(args)
        if getattr(args, 'pretrained_encoder', ''):
            from models.pretrained import resolve_encoder_weights
            ckpt = resolve_encoder_weights(args.pretrained_encoder)
            load_pretrained_encoder(
                model, ckpt,
                inflate_rgb_patch=bool(getattr(args, 'inflate_rgb_patch', 1)))
    else:
        from models import create_model
        model = create_model(args.model)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model = {args.model}, params = {n_params:,}')

    # ----- data ----------------------------------------------------------- #
    # Implement code/data/datasets.py::build_pretraining_dataset first.
    from data import build_pretraining_dataset
    dataset_train = build_pretraining_dataset(args)
    data_loader_train = make_data_loader(args, dataset_train, shuffle=True)

    # step-level warmup + cosine LR schedule (mirrors MultiMAE/VideoMAE)
    from utils import cosine_scheduler
    steps_per_epoch = max(
        1, (len(dataset_train) // (args.batch_size * get_world_size()))
        // max(1, args.update_freq))
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, steps_per_epoch,
        warmup_epochs=args.warmup_epochs)
    print(f'Step-level LR schedule: {len(lr_schedule_values)} steps, '
          f'peak {args.lr:.3e} -> min {args.min_lr:.3e}, '
          f'warmup {args.warmup_epochs} epoch(s)')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ----- optimizer / scaler -------------------------------------------- #
    from utils import NativeScalerWithGradNormCount, create_optimizer
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScalerWithGradNormCount()

    # ----- training loop -------------------------------------------------- #
    from utils import save_model
    print(f'Start training for {args.epochs} epochs')
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        stats = train_one_epoch_pretrain(
            model=model, data_loader=data_loader_train, optimizer=optimizer,
            device=device, epoch=epoch, loss_scaler=loss_scaler,
            max_norm=args.clip_grad, update_freq=args.update_freq,
            lr_schedule_values=lr_schedule_values,
            start_steps=epoch * steps_per_epoch)
        print(f'Epoch {epoch} stats: {stats}')

        if is_main_process() and (epoch % args.save_ckpt_freq == 0
                                  or epoch + 1 == args.epochs):
            save_model(args, epoch, model, model_without_ddp, optimizer,
                       loss_scaler)


if __name__ == '__main__':
    args = get_args()
    main(args)
