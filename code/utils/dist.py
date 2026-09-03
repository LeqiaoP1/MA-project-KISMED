"""Distributed training helpers (DDP).

Adapted from MultiMAE ``tmp/MultiMAE/utils/dist.py`` and the MAE codebase.
Requires launching with ``torch.distributed.launch`` / ``torchrun``.
"""
import os
import pickle
import subprocess
import socket
from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    'init_distributed_mode', 'setup_for_distributed', 'is_dist_avail_and_initialized',
    'get_world_size', 'get_rank', 'is_main_process', 'save_on_master',
    'all_gather', 'get_model',
]


def setup_for_distributed(is_master):
    """Disable printing/wandb when not the master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_model(model):
    if is_dist_avail_and_initialized():
        model = model.module
    return model


def init_distributed_mode(args):
    """Initialise DDP from CLI args (device, local rank, seed offset, ...)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url),
          flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def all_gather(data):
    """Gather ``data`` (a tensor or picklable object) across all ranks."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    if isinstance(data, torch.Tensor):
        tensor_list = [torch.zeros_like(data) for _ in range(world_size)]
        dist.all_gather(tensor_list, data)
        return tensor_list

    # generic fallback: move object to CPU and all_gather_object
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')
    local_size = torch.LongTensor([tensor.numel()]).to('cuda')
    size_list = [torch.LongTensor([0]).to('cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    tensor_list = [torch.empty(size, dtype=torch.uint8).to('cuda')
                   for size in size_list]
    dist.all_gather(tensor_list, tensor)
    return [pickle.loads(t.cpu().numpy().tobytes()) for t in tensor_list]
