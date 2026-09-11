"""AU-occurrence probe -- Semantic Representation Quality (classification).

ADD-ON diagnostic, separate from the Stage-1/2/3 runners. Probes the Stage-2
shared-encoder representations with FACS AU *occurrence* detection: does the
frozen (linear) or fine-tuned encoder linearly separate facial-action
semantics? Compare controls by pointing ``--finetune`` at different
checkpoints with an identical probe protocol:
  * random init          (no --finetune)                -- lower bound C0
  * Stage-1 MAE/ImageNet (--finetune base)              -- C1
  * Stage-2 BP4D multimodal MAE (output/pretrain/...)   -- C2 (headline)

Usage (from ``code/``)::

    python runners/run_au_probe.py -c configs/finetune/au_local.yaml

Data: canonical ``rgb/`` frames + raw ``AUCoding/AU_OCC`` csv (see
``data/au_dataset.py``). Splits are subject-disjoint.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from runners._common import (add_common_args, env_or, init_env,
                             make_data_loader, parse_args_with_config)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_AU_ROOT = os.path.join(_REPO, 'data', 'raw', 'BP4D', 'AUCoding', 'AU_OCC')


def get_args():
    parser = argparse.ArgumentParser('AU occurrence probe', add_help=False)
    add_common_args(parser)

    # --- data ------------------------------------------------------------ #
    parser.add_argument('--data_path', default=env_or('DATA_PATH'), type=str,
                        help='canonical session root (bp4d_canonical layout)')
    parser.add_argument('--au_root', default=os.environ.get('AU_ROOT')
                        or _DEFAULT_AU_ROOT, type=str,
                        help='raw BP4D+ AUCoding/AU_OCC directory')
    parser.add_argument('--au_list', default='1,2,4,6,7,10,12,14,15,17,23,24',
                        type=str,
                        help='explicit target AUs, comma list. Overrides the '
                             'default BP4D 12-AU subset, e.g. "6,7,10,12,14" '
                             'for the most frequent AUs only. Leave empty and '
                             'use --au_freq_topk to select by frequency.')
    parser.add_argument('--au_freq_topk', default=0, type=int,
                        help='auto-select the N most frequent AUs by presence '
                             'rate over the AU-usable sessions (overrides '
                             '--au_list when it is empty). e.g. 5 => the most '
                             'frequent AU6/7/10/12/14 set.')
    parser.add_argument('--streams', default='rgb', type=str,
                        help='visual stream(s) fed to the encoder: rgb[,tir] '
                             '(AU probe is visual-only by design)')
    # subject-disjoint split
    parser.add_argument('--train_ratio', default=0.8, type=float,
                        help='fraction of subjects for train (used when no '
                             'explicit --train_subjects/--val_subjects)')
    parser.add_argument('--train_subjects', default='', type=str,
                        help='explicit train subject ids, comma list')
    parser.add_argument('--val_subjects', default='', type=str,
                        help='explicit val subject ids, comma list')

    # --- probe geometry (MUST reproduce the Stage-2 checkpoint) ---------- #
    parser.add_argument('--tubelet', default='2,16,16', type=str,
                        help='tubelet (t, ph, pw) of the Stage-2 encoder')
    parser.add_argument('--fps', default=25.0, type=float)
    parser.add_argument('--clip_duration', default=4.0, type=float,
                        help='clip length in seconds == the Stage-2 '
                             'pretraining clip (drives num_frames)')
    parser.add_argument('--num_frames', default=0, type=int,
                        help='clip length in frames (0 => clip_duration*fps)')
    parser.add_argument('--input_size', default=64, type=int)
    parser.add_argument('--enc_embed_dim', default=192, type=int)
    parser.add_argument('--enc_depth', default=6, type=int)
    parser.add_argument('--enc_num_heads', default=6, type=int)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--drop_path_rate', default=0.0, type=float)

    # --- probe mode ------------------------------------------------------ #
    parser.add_argument('--probe', default='linear',
                        choices=['linear', 'ft'],
                        help='linear = freeze encoder, train head only '
                             '(formal diagnostic); ft = fine-tune all')
    parser.add_argument('--pool', default='mean', type=str,
                        help='token pooling (only "mean" implemented)')
    parser.add_argument('--finetune', default=env_or('MODEL_PATH'), type=str,
                        help='checkpoint to probe: Stage-2 MAE or Stage-1 '
                             'MAE/ImageNet. Either a local path or a variant '
                             'spec (base, mae:large, deit:small) that is '
                             'downloaded into <project_root>/models/initial. '
                             'Empty => random init (C0).')

    # --- training -------------------------------------------------------- #
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=0.0, type=float)
    return parse_args_with_config(parser)


def _print_au_table(au_list, f1_per_au):
    print('\nPer-AU F1 @0.5:')
    for au, f1 in zip(au_list, f1_per_au):
        print(f'  AU{au:>2}: {f1 * 100:6.2f}')
    print(f'  ----\n  macro: {sum(f1_per_au) / len(f1_per_au) * 100:6.2f}')


def main(args):
    from utils import get_world_size, is_main_process

    from data.au_dataset import resolve_au_list
    au_list = resolve_au_list(args)
    # pin the resolved set so build_au_datasets reuses it (no duplicate scan)
    args.au_list = ','.join(str(a) for a in au_list)
    args.au_freq_topk = 0
    n_aus = len(au_list)

    device = init_env(args)

    # ---- model ---------------------------------------------------------- #
    from core.au_probe import build_au_probe_model, load_au_probe_weights
    model = build_au_probe_model(args, num_classes=n_aus)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'AU probe: streams={args.streams} AUs={au_list} '
          f'params={n_params:,}')

    if args.finetune:
        from models.pretrained import resolve_encoder_weights
        load_au_probe_weights(model, resolve_encoder_weights(args.finetune))
    if args.probe == 'linear':
        model.freeze_features()
        n_head = sum(p.numel() for p in model.head.parameters())
        n_train = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
        print(f'Linear probe: frozen encoder; trainable = head only '
              f'({n_train:,} / {n_head:,})')

    # ---- data (subject-disjoint split) ---------------------------------- #
    from data.au_dataset import build_au_datasets
    dataset_train, dataset_val = build_au_datasets(args)
    print(f'Train subjects {sorted(dataset_train.subjects)} -> '
          f'{len(dataset_train)} samples | Val subjects '
          f'{sorted(dataset_val.subjects)} -> {len(dataset_val)} samples')
    data_loader_train = make_data_loader(args, dataset_train, shuffle=True)
    data_loader_val = make_data_loader(args, dataset_val, shuffle=False,
                                       drop_last=False)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ---- optimizer / criterion ----------------------------------------- #
    from utils import NativeScalerWithGradNormCount, create_optimizer
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScalerWithGradNormCount()
    criterion = torch.nn.BCEWithLogitsLoss()

    # step-level warmup + cosine (parity with the Stage runners)
    from utils import cosine_scheduler
    steps_per_epoch = max(
        1, (len(dataset_train) // (args.batch_size * get_world_size()))
        // max(1, args.update_freq))
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs,
                                   steps_per_epoch,
                                   warmup_epochs=args.warmup_epochs)

    # ---- train / eval --------------------------------------------------- #
    from engines.au_probe import evaluate_au, train_one_epoch_au
    best_f1 = -1.0
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch_au(
            model=model, data_loader=data_loader_train, optimizer=optimizer,
            device=device, epoch=epoch, criterion=criterion,
            loss_scaler=loss_scaler, max_norm=args.clip_grad,
            update_freq=args.update_freq, lr_schedule_values=lr_schedule,
            start_steps=epoch * steps_per_epoch)

        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            stats = evaluate_au(data_loader_val, model, device)
            f1 = stats['f1']
            print(f'[epoch {epoch}] val macro-F1 {f1:.2f}  loss '
                  f'{stats["loss"]:.4f}')
            if is_main_process() and f1 > best_f1:
                best_f1 = f1
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save({'model': model_without_ddp.state_dict(),
                            'epoch': epoch, 'au_list': au_list},
                           os.path.join(args.output_dir, 'best_au.pth'))
        if is_main_process() and (epoch % args.save_ckpt_freq == 0
                                  or epoch + 1 == args.epochs):
            ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({'model': model_without_ddp.state_dict(),
                        'epoch': epoch, 'au_list': au_list},
                       os.path.join(ckpt_dir,
                                    f'au_checkpoint-{epoch:04d}.pth'))

    print(f'\nBest val macro-F1: {best_f1:.2f}')
    # final per-AU breakdown on the val split
    stats = evaluate_au(data_loader_val, model, device)
    _print_au_table(au_list, stats['f1_per_au'])


if __name__ == '__main__':
    args = get_args()
    main(args)
