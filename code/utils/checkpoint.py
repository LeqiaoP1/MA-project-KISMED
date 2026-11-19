"""Checkpointing helpers (save / resume from ``output_dir``).

Adapted from MultiMAE ``tmp/MultiMAE/utils/checkpoint.py`` and the MAE
codebase. The run scripts should call these inside the training loop.
"""
import os

import torch

from .dist import is_main_process, save_on_master

__all__ = ['save_model', 'load_model', 'auto_resume_model',
           'check_resume_fold', 'check_resume_loss']


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
        # the AMP precision mode, so a resume in a DIFFERENT mode is visible
        'amp_dtype': getattr(args, 'amp_dtype', 'none'),
        'args': args,
    }
    if model_ema is not None:
        to_save['model_ema'] = model_ema.state_dict()

    save_on_master(to_save, checkpoint_path)
    if is_main_process():
        # write a pointer file that auto_resume_model can find
        with open(os.path.join(output_dir, 'latest_checkpoint.txt'), 'w') as f:
            f.write(checkpoint_path)


def check_resume_fold(cur_args, ck_args) -> dict:
    """Refuse to resume ONE fold's state from a checkpoint of ANOTHER partition.

    Isolation is a property of the output directory: fold k's run may only ever
    continue fold k's own state. Nothing in the tensor shapes reveals a
    violation -- the checkpoint loads perfectly -- so the PARTITION DIGEST the
    checkpoint records is compared with the digest this process is running.

    A digest, not a fold index: two runs can share ``fold == 2`` and still hold
    out different subjects (a different seed or stratifier), which is exactly
    the pair this check exists to refuse.

    * both digests recorded and different -> ``SystemExit``;
    * checkpoint records no digest (pre-ADR-001) -> warning, verdict unverified;
    * this run has no digest -> nothing to compare, nothing claimed.

    :returns: a small verdict dict for logging.
    """
    cur_digest = getattr(cur_args, 'partition_digest', None)
    if not cur_digest:
        return {}
    if not ck_args:
        print('[resume] the checkpoint records no args -> cannot verify which '
              'partition it belongs to.')
        return {'checkpoint_digest': None, 'run_digest': str(cur_digest),
                'match': False}
    ck_digest = getattr(ck_args, 'partition_digest', None)
    if not ck_digest:
        print(f'[resume] WARNING: this run is partition '
              f'{str(cur_digest)[:12]} but the checkpoint records NO partition '
              f'digest (it predates ADR-001). Continuing may mix another '
              f'partition\'s state into this run -- use a fresh --output_dir '
              f'for the fold.')
        return {'checkpoint_digest': None, 'run_digest': str(cur_digest),
                'match': False}

    if str(ck_digest) != str(cur_digest):
        raise SystemExit(
            f'[resume] PARTITION MISMATCH: the checkpoint belongs to partition '
            f'{str(ck_digest)[:12]}, but this run is {str(cur_digest)[:12]}. '
            f'Resuming would continue another fold\'s weights, and that fold '
            f'saw these held-out subjects. Point --output_dir at this fold\'s '
            f'own run dir (or start fresh).')
    print(f'[resume] partition contract OK: checkpoint and run share digest '
          f'{str(cur_digest)[:12]}.')
    return {'checkpoint_digest': str(ck_digest), 'run_digest': str(cur_digest),
            'match': True}


def check_resume_loss(cur_args, ck_args) -> dict:
    """Report when a resume changes the shape of the Stage-3 ``L_time`` term.

    ``L_time`` is a Huber term, so ``huber_beta`` IS the objective: inside the
    band the error is charged quadratically, beyond it linearly. A run started
    with one beta and continued with another loads perfectly and trains without
    complaint (the weights and the optimiser state are shape-compatible) -- it
    simply changes the objective mid-run, and ``metrics_final.json`` then records
    ONE loss for a run that had two. Nothing else in the pipeline notices, which
    is exactly why this is said out loud.

    A warning, not a ``SystemExit``: retuning ``huber_beta`` on a live run is
    legitimate when it is meant. Runners with no ``huber_beta`` (Stage 1/2) are
    skipped, so it costs them nothing.

    :returns: a small verdict dict for logging.
    """
    cur_raw = getattr(cur_args, 'huber_beta', None)
    if cur_raw is None:
        return {}                       # not a Stage-3 waveform run
    cur = float(cur_raw)
    if not ck_args:
        return {}                       # unverifiable; check_resume_fold said so
    ck_raw = getattr(ck_args, 'huber_beta', None)
    if ck_raw is None:
        return {}                       # predates the key; nothing to compare
    ck = float(ck_raw)

    if ck != cur:
        print(f'[resume] WARNING: huber_beta (the L_time transition point) '
              f'changes across this resume -- the checkpoint was trained with '
              f'{ck!r}, this run uses {cur!r}. The weights and the optimizer '
              f'state load fine, but the objective changes mid-run; pass a '
              f'fresh --output_dir unless that is what you meant.')
        return {'checkpoint_huber_beta': ck, 'run_huber_beta': cur,
                'match': False}
    return {'checkpoint_huber_beta': ck, 'run_huber_beta': cur, 'match': True}


def load_model(args, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """Load a checkpoint (from ``args.resume``) into model/optimizer/scaler."""
    if args.resume:
        # weights_only=False: our own checkpoints store the `args` namespace,
        # which pickles numpy scalars -- torch>=2.6 defaults to
        # weights_only=True and refuses to unpickle them, so resume used to die
        # with a _pickle.UnpicklingError. Same policy as
        # core.waveform_model.load_stage2_encoder.
        checkpoint = torch.load(args.resume, map_location='cpu',
                                weights_only=False)
        # k-fold: a fold run may only resume its OWN fold (see the helper).
        check_resume_fold(args, checkpoint.get('args'))
        # Stage 3: say so when the L_time term changes across the resume.
        check_resume_loss(args, checkpoint.get('args'))
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        if loss_scaler is not None and 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        # AMP mode metadata (added 2026-10-13): a checkpoint written in one
        # precision mode and resumed in another reuses the optimizer/scaler
        # momentum as-is, which is legitimate but easy to do by accident -> say
        # it out loud. Old checkpoints simply have no key and resume silently.
        ckpt_amp = checkpoint.get('amp_dtype')
        cur_amp = str(getattr(args, 'amp_dtype', 'none'))
        if ckpt_amp is not None and str(ckpt_amp) != cur_amp:
            print(f'[amp] WARNING: {args.resume} was written with '
                  f'amp_dtype={ckpt_amp!r} but this run uses '
                  f'amp_dtype={cur_amp!r}; the optimizer/scaler state is '
                  f'reused as-is and training continues in {cur_amp!r}.')
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
