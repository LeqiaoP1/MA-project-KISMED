"""Shared CLI/config plumbing for the ``runners/*.py`` entry points.

Implements the MultiMAE pattern:
  * YAML config file (``-c/--config``) provides argparse *defaults*
  * explicit command-line flags override the YAML values
"""
import argparse
import difflib
import os
import random
import sys

import numpy as np
import torch
import yaml

from utils import (add_amp_args, get_rank, get_world_size,
                   init_distributed_mode)

# ``code/runners/_common.py`` -> [0]=runners, [1]=code, [2]=project root
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)


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
    # ABSOLUTE fallback: a bare './output' would land inside code/ when the env
    # profile is not sourced (env_local.sh exports OUTPUT_DIR=<repo>/output)
    parser.add_argument('--output_dir',
                        default=env_or('OUTPUT_DIR',
                                       os.path.join(_PROJECT_ROOT, 'output')),
                        type=str,
                        help='root folder for checkpoints/logs; default: '
                             '$OUTPUT_DIR, else <project_root>/output')
    parser.add_argument('--resume', default=env_or('RESUME', ''), type=str,
                        help='checkpoint path to resume from')
    parser.add_argument('--log_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', default='thesis-project', type=str)
    parser.add_argument('--num_workers', default=int(env_or('NUM_WORKERS', '8')), type=int)
    parser.add_argument('--tir_channels',
                        default=int(env_or('TIR_CHANNELS', '3')),
                        type=int, choices=[1, 3],
                        help='TIR input channels. 3 (default) treats the '
                             'thermal stream as what it is: a false-colour '
                             '(rainbow) rendering with real chroma '
                             '(wmv3/yuv420p, verified with OpenCV + PyAV). '
                             '1 = legacy luma-only surrogate; it changes the '
                             'TIR adapter geometry, so use it only to match '
                             'Stage-2 checkpoints trained before 2026-09.')
    parser.add_argument('--pin_mem', action='store_true', default=True)
    # quick/dev runs: cap the number of sessions / clips (see PairedSessionDataset)
    parser.add_argument('--max_sessions', default=None, type=int,
                        help='limit number of sessions (smoke tests)')
    parser.add_argument('--max_clips', default=None, type=int,
                        help='limit number of clips taken per session '
                             '(smoke tests)')
    parser.add_argument('--max_entries', default=None, type=int,
                        help='limit number of clips per split (smoke tests)')
    # mixed precision (AMP): see utils/native_scaler.py + notes/design/ADR-016-mixed-precision-and-ddp-accumulation.md
    add_amp_args(parser)


#: Stream names a run can declare (the model's own taxonomy).
#: Declared once so every runner's parser exposes the same keys and a YAML uses
#: exactly the names the taxonomy lists.
PROTOCOL_STREAMS = ('rgb', 'tir', 'rgb2d', 'bp', 'resp')


def partition_path(exp_root: str, fold: int, dev_smoke: bool = False) -> str:
    """``<exp_root>/fold<k>/partition.json`` (``.dev.json`` for a dev run)."""
    name = 'partition.dev.json' if dev_smoke else 'partition.json'
    return os.path.join(exp_root, f'fold{int(fold)}', name)


def add_protocol_args(parser: argparse.ArgumentParser):
    """Flags of ADR-001 section 2.3 — exactly those, and no fold-count knob.

    ``K`` is a module constant (``protocol.constants.K``); the absence of a
    ``--n_folds`` flag is deliberate, because a value that must never change
    should not be expressible.
    """
    from protocol import CRITERIA, K, SELECTION_RULES, SPLIT_STRATEGIES

    parser.add_argument('--exp_root', default='', type=str,
                        help='experiment root; the partition record lives at '
                             '<exp_root>/fold<k>/partition.json and every run '
                             'of that fold shares it. Required.')
    parser.add_argument('--fold', default=None, type=int,
                        help=f'fold index in [0, {K}); block k is the test '
                             f'role of fold k')
    parser.add_argument('--split_seed', default=42, type=int,
                        help='partition seed (ADR-001 I2)')
    parser.add_argument('--split_stratify', default='sex', type=str,
                        choices=list(SPLIT_STRATEGIES),
                        help="partition strategy: 'sex' shuffles within each "
                             'stratum then interleaves proportionally')
    parser.add_argument('--val_seed', default=None, type=int,
                        help='seed of the validation draw (default: split_seed)')
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='epochs between validation evaluations; 0 = off '
                             '(legal only with --val_select fixed_epoch)')
    parser.add_argument('--val_select', default='fixed_epoch', type=str,
                        choices=list(SELECTION_RULES),
                        help='selection rule; fixed_epoch never reads a metric')
    parser.add_argument('--val_criterion', default='joint_loss', type=str,
                        choices=list(CRITERIA),
                        help='what best_val minimises/maximises')
    parser.add_argument('--dev_smoke', action='store_true', default=False,
                        help='plumbing runs only: tolerates a corpus that '
                             'cannot hold 70/10/20 and writes '
                             'partition.dev.json, which can never be pooled')


def load_or_create_plan(args, *, stage: str = ''):
    """Read ``<exp_root>/fold<k>/partition.json``, creating it on first use (D4).

    Both stages and the standalone utility converge on ONE file, so a Stage-3
    run can never be paired with a Stage-2 encoder from a different partition:
    a differing digest is a hard error inside ``protocol.save_plan``.
    """
    from data.universe import corpus_subjects
    from protocol import build_plan, load_plan, save_plan

    exp_root = str(getattr(args, 'exp_root', '') or '').strip()
    if not exp_root:
        raise SystemExit(
            '--exp_root is required: ADR-001 keeps the partition record at '
            '<exp_root>/fold<k>/partition.json and every run of that fold '
            'shares it.')
    fold = getattr(args, 'fold', None)
    if fold is None:
        raise SystemExit('--fold is required (0 .. K-1).')
    dev_smoke = bool(getattr(args, 'dev_smoke', False))
    path = partition_path(exp_root, fold, dev_smoke)

    def _describe(plan) -> str:
        return (f'fold {plan.fold}/{plan.n_folds}: '
                + ', '.join(f'{role}={plan.counts[role]}'
                            for role in ('train', 'val', 'test'))
                + f'  digest {plan.digest[:12]}')

    if os.path.exists(path):
        plan = load_plan(path)
        print(f'[protocol] partition {path} -> {_describe(plan)}')
        return plan

    universe = corpus_subjects(args)
    plan = build_plan(universe, int(fold),
                      split_seed=int(getattr(args, 'split_seed', 42)),
                      split_stratify=str(getattr(args, 'split_stratify', 'sex')),
                      val_seed=getattr(args, 'val_seed', None),
                      dev_smoke=dev_smoke)
    written = save_plan(path, plan,
                        writer={'stage': stage or 'unknown',
                                'argv': sys.argv[1:]})
    print(f'[protocol] wrote {path} -> {_describe(written)}')
    return written



def _check_config_keys(parser: argparse.ArgumentParser, cfg: dict, path: str):
    """Fail loudly on YAML keys the target parser does not define.

    ``parser.set_defaults(**cfg)`` accepts ARBITRARY keywords, so a typo
    (``roi_paddingg``) or a key that was renamed/removed in code becomes a
    silent no-op: the run proceeds with the argparse default instead and looks
    perfectly healthy. A key that is misspelled into a *different valid* default
    is the worst case -- e.g. dropping ``physio_cap_ratio: 0.25`` silently falls
    back to 0.50, or a lost ``pretrained_encoder`` silently trains from scratch.

    So the one soft spot of the "YAML supplies defaults" design is turned into a
    hard error here. Keys with a close match get a spelling hint.
    """
    known = {a.dest for a in parser._actions}
    unknown = sorted(k for k in cfg if k not in known)
    if not unknown:
        return
    lines = [f'[config] {path}: {len(unknown)} key(s) are not options of '
             f'{parser.prog!r} and would be SILENTLY IGNORED:']
    for key in unknown:
        near = difflib.get_close_matches(key, known, n=1, cutoff=0.7)
        lines.append(f'    {key}' + (f"   (did you mean {near[0]!r}?)" if near else ''))
    lines.append('        A key here is either a typo or a leftover from an '
                 'older config. Fix the YAML (or delete the key) and re-run.')
    raise SystemExit('\n'.join(lines))


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None):
    """Parse ``argv``; if ``-c/--config`` given, its YAML supplies defaults.

    NOTE: the ``-c`` parser below is built with ``add_help=False`` so that it
    only ever steals ``-c/--config``; every flag it does not know -- including
    ``--help`` -- falls through to the caller's ``parser``. The runner parsers
    must therefore use ``add_help=True`` (argparse's default) or ``--help``
    dies with "unrecognized arguments: --help". Every training/eval runner had
    ``add_help=False`` and no ``-h`` action until 2026-10-01.

    Resolution order (highest first): explicit CLI flag > YAML value > argparse
    hardcoded default. Only keys PRESENT in the YAML are set; anything absent
    keeps the hardcoded default from the runner's ``get_args()``. Unknown keys
    are rejected (see :func:`_check_config_keys`).
    """
    config_parser = argparse.ArgumentParser('Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str,
                               metavar='FILE',
                               help='YAML config file specifying defaults')
    cfg_args, remaining = config_parser.parse_known_args(argv)
    if cfg_args.config:
        with open(cfg_args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg is None:                      # empty file -> keep all defaults
            cfg = {}
        if not isinstance(cfg, dict):
            raise SystemExit(
                f'[config] {cfg_args.config}: expected a YAML mapping at the '
                f'top level, got {type(cfg).__name__}.')
        _check_config_keys(parser, cfg, cfg_args.config)
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
                     batch_size=None, num_workers=None):
    """Distributed-aware DataLoader.

    ``num_workers=0`` is passed explicitly by the validation loaders: an
    ordered, single-process pass is what makes two evaluations of one
    checkpoint comparable bit-for-bit (ADR-001 I6).
    """
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


def check_loader_not_empty(loader, name, args, extra=''):
    """Fail fast when a split yields ZERO batches, naming the actual cause.

    A ``DataLoader`` silently yields nothing when ``drop_last`` eats the whole
    split (fewer samples than ``batch_size``) or when the dataset is empty after
    filtering. The next symptom used to be a bare
    ``ZeroDivisionError: float division by zero`` inside the progress logger,
    which says nothing about the cause -- and the LR schedule, which clamps
    ``steps_per_epoch`` to >= 1, happily pretends the epoch exists.
    """
    if len(loader) > 0:
        return
    n = len(loader.dataset)
    batch_size = int(loader.batch_size or 0)
    lines = [f'the {name} split yields 0 batches: {n} sample(s), '
             f'batch_size {batch_size}, drop_last={bool(loader.drop_last)}']
    if n == 0:
        lines.append('the dataset is EMPTY: check --tasks/--task_set/'
                     '--task_groups, --min_signal_spread and the split keys. '
                     '(A subject-disjoint val split only holds the held-out '
                     'subject, so that subject must have surviving clips.)')
    elif batch_size and n < batch_size:
        lines.append(f'{n} sample(s) < batch_size {batch_size} with '
                     f'drop_last=True gives 0 batches: use --batch_size {n} '
                     f'or less, or collect more clips (--max_entries 0, '
                     f'wider --tasks/--task_set)')
    else:
        lines.append('lower --batch_size or collect more clips '
                     '(--max_entries 0, wider --tasks/--task_set)')
    if extra:
        lines.append(extra)
    raise SystemExit('[data] ' + '\n       '.join(lines))
