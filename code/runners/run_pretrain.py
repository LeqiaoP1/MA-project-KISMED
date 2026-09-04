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
    # data
    parser.add_argument('--data_path', default=env_or('DATA_PATH'), type=str)
    # training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    # optimizer / lr
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', default=1.5e-4, type=float)
    parser.add_argument('--blr', default=1.5e-4, type=float,
                        help='base lr = lr * batch_size / 256 (takes precedence)')
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--warmup_epochs', default=40, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=0.0, type=float)
    return parse_args_with_config(parser)


def main(args):
    from utils import get_world_size, is_main_process

    device = init_env(args)

    # scale LR by the effective batch size (MAE convention)
    args.lr = args.blr * args.batch_size * get_world_size() / 256

    # ----- model ---------------------------------------------------------- #
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
            max_norm=args.clip_grad, update_freq=args.update_freq)
        print(f'Epoch {epoch} stats: {stats}')

        if is_main_process() and (epoch % args.save_ckpt_freq == 0
                                  or epoch + 1 == args.epochs):
            save_model(args, epoch, model, model_without_ddp, optimizer,
                       loss_scaler)


if __name__ == '__main__':
    args = get_args()
    main(args)
