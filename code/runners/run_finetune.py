"""Fine-tuning entry point (classification).

Usage (from ``code/``)::

    python runners/run_finetune.py -c configs/finetune/example.yaml

Requires a registered dataset (see ``data/datasets.py``) and, typically, a
pretrained checkpoint passed with ``--finetune`` (load the encoder weights,
then fine-tune with a new classification head).
"""
import argparse
import os
import sys

# allow running either `python runners/run_finetune.py` or `python -m runners.run_finetune`
# from the code/ root by ensuring code/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engines import evaluate, train_one_epoch
from runners._common import (add_common_args, env_or, init_env,
                             make_data_loader, parse_args_with_config)


def get_args():
    parser = argparse.ArgumentParser('Project fine-tuning', add_help=False)
    add_common_args(parser)

    # model
    parser.add_argument('--model', default='project_vit_base_patch16_224',
                        type=str, help='registered model name')
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--finetune', default=env_or('MODEL_PATH'), type=str,
                        help='pretrained checkpoint to fine-tune from')
    # data
    parser.add_argument('--data_set', default=env_or('DATA_SET', ''), type=str)
    parser.add_argument('--data_path', default=env_or('DATA_PATH'), type=str)
    # training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    # optimizer / lr
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=0.0, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float)
    return parse_args_with_config(parser)


def main(args):
    from utils import is_main_process

    device = init_env(args)

    # ----- model ---------------------------------------------------------- #
    from models import create_model
    model = create_model(args.model, num_classes=args.nb_classes)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model = {args.model}, params = {n_params:,}')

    if args.finetune:
        ckpt = torch.load(args.finetune, map_location='cpu')
        state = ckpt['model'] if 'model' in ckpt else ckpt
        # drop incompatible head weights (new nb_classes)
        for k in ['head.weight', 'head.bias']:
            if k in state and state[k].shape != model.state_dict()[k].shape:
                print(f'Dropping incompatible key {k}')
                del state[k]
        model.load_state_dict(state, strict=False)
        print(f'Loaded fine-tune checkpoint {args.finetune}')

    # ----- data ----------------------------------------------------------- #
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

    # ----- optimizer / criterion ----------------------------------------- #
    from utils import NativeScalerWithGradNormCount, create_optimizer
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScalerWithGradNormCount()
    if args.smoothing > 0.:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ----- training loop -------------------------------------------------- #
    from utils import auto_resume_model, save_model
    auto_resume_model(args, model_without_ddp, optimizer, loss_scaler)
    best_acc1 = 0.0
    for epoch in range(args.start_epoch if hasattr(args, 'start_epoch') else 0,
                       args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model=model, criterion=criterion, data_loader=data_loader_train,
            optimizer=optimizer, device=device, epoch=epoch,
            loss_scaler=loss_scaler, max_norm=args.clip_grad,
            update_freq=args.update_freq)

        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            test_stats = evaluate(data_loader_val, model, device)
            acc1 = test_stats.get('acc1', 0.0)
            print(f'[epoch {epoch}] Acc@1 {acc1:.2f}')

            if is_main_process() and acc1 > best_acc1:
                best_acc1 = acc1
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save({'model': model_without_ddp.state_dict(),
                            'epoch': epoch}, os.path.join(args.output_dir, 'best.pth'))

        if is_main_process() and (epoch % args.save_ckpt_freq == 0
                                  or epoch + 1 == args.epochs):
            save_model(args, epoch, model, model_without_ddp, optimizer,
                       loss_scaler)

    print(f'Best Acc@1: {best_acc1:.2f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
