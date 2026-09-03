"""Stage-3 downstream waveform fine-tuning (BVP or RESP branch).

Implements docs/ImplementationPlan.md Stage 3: simulated full sensor failure
(no contact 1D streams are fed) -- only visual streams drive the prediction.
Two independent task-specialised runs share the Stage-2 encoder checkpoint
(``--finetune``) but use separate regression heads and a unified
spatio-temporal-spectral joint loss (``core.waveform_losses.WaveformJointLoss``).

Usage (from ``code/``)::

    python runners/run_waveform.py -c configs/finetune/bvp.yaml
    python runners/run_waveform.py -c configs/finetune/resp.yaml

Requires an implemented ``bp4d+`` dataset in ``data/datasets.py`` yielding
``(samples, target_waveform)`` where ``target_waveform`` is ``[B, T]``.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core import WaveformJointLoss
from engines import evaluate_waveforms, train_one_epoch_waveform
from runners._common import (add_common_args, init_env, make_data_loader,
                             parse_args_with_config)


def _parse_fft_sizes(s):
    return tuple(int(x) for x in str(s).split(',') if x.strip())


def get_args():
    parser = argparse.ArgumentParser('Stage-3 waveform fine-tuning', add_help=False)
    add_common_args(parser)

    # model / task
    parser.add_argument('--model', default='project_vit_base_patch16_224', type=str)
    parser.add_argument('--target', default='bvp', choices=['bvp', 'resp'],
                        help='which physiological waveform branch to train')
    parser.add_argument('--seq_len', default=1000, type=int,
                        help='length of the predicted output waveform (samples)')
    parser.add_argument('--fs', default=100.0, type=float,
                        help='waveform sampling rate in Hz')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--finetune', default='', type=str,
                        help='Stage-2 pretrained encoder checkpoint to load')

    # data (implement BP4D+ in code/data/datasets.py)
    parser.add_argument('--data_set', default='bp4d+', type=str)
    parser.add_argument('--data_path', default='', type=str)

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

    # ----- model: shared ViT encoder + waveform regression head ----------- #
    from models import create_model
    model = create_model(args.model, num_classes=0, output_len=args.seq_len)
    model.to(device)

    if args.finetune:
        ckpt = torch.load(args.finetune, map_location='cpu')
        state = ckpt['model'] if 'model' in ckpt else ckpt
        # drop incompatible keys (task heads / decoders of the pretrain model)
        drop = []
        for k in state:
            if k.startswith('head.') or k.startswith('decoder.'):
                drop.append(k)
        for k in drop:
            del state[k]
        model.load_state_dict(state, strict=False)
        print(f'Loaded Stage-2 encoder from {args.finetune} '
              f'(dropped {len(drop)} head/decoder keys)')

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
