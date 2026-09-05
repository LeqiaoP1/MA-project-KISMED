"""Checkpointing helpers (save / resume from ``output_dir``).

Adapted from MultiMAE ``tmp/MultiMAE/utils/checkpoint.py`` and the MAE
codebase. The run scripts should call these inside the training loop.
"""
import os

import torch

from .dist import is_main_process, save_on_master

__all__ = ['save_model', 'load_model', 'auto_resume_model']


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler,
               model_ema=None):
    output_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'checkpoint-{epoch:04d}.pth')
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
        'args': args,
    }
    if model_ema is not None:
        to_save['model_ema'] = model_ema.state_dict()

    save_on_master(to_save, checkpoint_path)
    if is_main_process():
        # write a pointer file that auto_resume_model can find
        with open(os.path.join(output_dir, 'latest_checkpoint.txt'), 'w') as f:
            f.write(checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """Load a checkpoint (from ``args.resume``) into model/optimizer/scaler."""
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        if loss_scaler is not None and 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        if model_ema is not None and 'model_ema' in checkpoint:
            model_ema.module.load_state_dict(checkpoint['model_ema'])


def auto_resume_model(args, model_without_ddp, optimizer, loss_scaler,
                      model_ema=None):
    """Resume from ``args.resume`` or the latest checkpoint in output_dir."""
    output_dir = os.path.join(args.output_dir, 'checkpoints')
    latest = os.path.join(output_dir, 'latest_checkpoint.txt')
    if os.path.isfile(latest):
        with open(latest) as f:
            args.resume = f.read().strip()
        print(f'Auto-resuming from {args.resume}')

    if args.resume:
        load_model(args, model_without_ddp, optimizer, loss_scaler, model_ema=model_ema)
