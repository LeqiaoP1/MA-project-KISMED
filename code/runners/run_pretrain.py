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
    parser.add_argument('--model', default='project_vit_base_patch16_224',
                        type=str, help='registered model name')
    # data / clip geometry (the multimodal dataset & model read these)
    parser.add_argument('--data_path', default=env_or('DATA_PATH'), type=str)
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
    parser.add_argument('--seq_len', default=0, type=int,
                        help='signal samples per window (0 => clip_duration*fs)')
    parser.add_argument('--input_size', default=64, type=int,
                        help='frame short-side resize/crop (square)')
    # multimodal MAE (Stage 2)
    parser.add_argument('--streams', default='rgb,tir,bvp', type=str,
                        help='comma list of pretraining streams')
    parser.add_argument('--tubelet', default='2,16,16', type=str,
                        help='tubelet (t, ph, pw) for the video tokenizer')
    parser.add_argument('--mask_ratio_rgb', default=0.75, type=float)
    parser.add_argument('--mask_ratio_tir', default=0.50, type=float)
    parser.add_argument('--mask_ratio_bvp', default=0.90, type=float)
    parser.add_argument('--enc_embed_dim', default=192, type=int)
    parser.add_argument('--enc_depth', default=6, type=int)
    parser.add_argument('--enc_num_heads', default=6, type=int)
    parser.add_argument('--dec_depth', default=2, type=int)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--sig_kernel', default=8, type=int,
                        help='signal token window (samples per token)')
    parser.add_argument('--pretrained_encoder', default='', type=str,
                        help='MAE/ImageNet ViT checkpoint to initialise the '
                             'shared encoder from (Stage-1 spatial priors)')
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
            load_pretrained_encoder(model, args.pretrained_encoder)
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
