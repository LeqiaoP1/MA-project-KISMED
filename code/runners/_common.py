"""Shared CLI/config plumbing for the ``runners/*.py`` entry points.

Implements the MultiMAE pattern:
  * YAML config file (``-c/--config``) provides argparse *defaults*
  * explicit command-line flags override the YAML values
"""
import argparse
import os
import random

import numpy as np
import torch
import yaml

from utils import get_rank, get_world_size, init_distributed_mode


def env_or(name: str, default: str = '') -> str:
    """Resolve a CLI default from an environment variable (e.g. DATA_PATH)."""
    return os.environ.get(name, default)


def add_common_args(parser: argparse.ArgumentParser):
    """Flags shared by every runner."""
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    # distributed
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    # output / resume  (paths may be injected via env for local <-> HPC switching)
    parser.add_argument('--output_dir', default=env_or('OUTPUT_DIR', './output'),
                        type=str,
                        help='root folder for checkpoints/logs')
    parser.add_argument('--resume', default=env_or('RESUME', ''), type=str,
                        help='checkpoint path to resume from')
    parser.add_argument('--log_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', default='thesis-project', type=str)
    parser.add_argument('--num_workers', default=int(env_or('NUM_WORKERS', '8')), type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    # quick/dev runs: cap the number of sessions / clips (see PairedSessionDataset)
    parser.add_argument('--max_sessions', default=None, type=int,
                        help='limit number of sessions (smoke tests)')
    parser.add_argument('--max_clips', default=None, type=int,
                        help='limit number of clips taken per session '
                             '(smoke tests)')
    parser.add_argument('--max_entries', default=None, type=int,
                        help='limit number of clips per split (smoke tests)')


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None):
    """Parse ``argv``; if ``-c/--config`` given, its YAML supplies defaults."""
    config_parser = argparse.ArgumentParser('Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str,
                               metavar='FILE',
                               help='YAML config file specifying defaults')
    cfg_args, remaining = config_parser.parse_known_args(argv)
    if cfg_args.config:
        with open(cfg_args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    return parser.parse_args(remaining)


def init_env(args):
    """DDP init + reproducible seed. Returns the compute device."""
    init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    return device


def make_data_loader(args, dataset, shuffle=True, drop_last=True,
                     batch_size=None):
    """Distributed-aware DataLoader."""
    batch_size = batch_size or args.batch_size
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=get_world_size(), rank=get_rank(),
            shuffle=shuffle)
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem,
        drop_last=drop_last)
    return loader
