"""Create the ADR-001 partition record for one fold (write-once, D4).

The record is what Stage 2, Stage 3 and the evaluator share, so it can also be
created before any training — the case that matters for a from-scratch baseline,
which has no Stage-2 run to inherit it from.

Usage (from ``code/``)::

    python runners/run_partition.py --exp_root ../output/protocol_v1 \\
        --fold 0 --data_set tir_roi --raw_root $RAW_DATA_PATH

    # inspect the corpus without writing anything
    python runners/run_partition.py --fold 0 --data_set tir_roi \\
        --raw_root $RAW_DATA_PATH --dry_run
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.universe import (available_data_sets, corpus_sessions,
                           corpus_subjects, resolve_corpus)
from protocol import (K, ProtocolError, build_plan, load_plan, save_plan,
                      to_record)

__all__ = ['get_args', 'main']


def get_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'ADR-001 partition record', add_help=True,
        description='Derive the train/val/test roles of one fold from the '
                    'corpus and write <exp_root>/fold<k>/partition.json once.')
    parser.add_argument('--exp_root', default='', type=str,
                        help='experiment root; the record goes to '
                             '<exp_root>/fold<k>/partition.json. Required '
                             'unless --dry_run.')
    parser.add_argument('--fold', default=0, type=int,
                        help=f'fold index in [0, {K}); block k is the test role')
    parser.add_argument('--split_seed', default=42, type=int,
                        help='partition seed (ADR-001 I2)')
    parser.add_argument('--split_stratify', default='sex', type=str,
                        choices=['none', 'sex'],
                        help="partition strategy; 'sex' shuffles within each "
                             'stratum and interleaves proportionally')
    parser.add_argument('--val_seed', default=None, type=int,
                        help='seed of the validation draw (default: split_seed)')
    parser.add_argument('--data_set', default='tir_roi', type=str,
                        choices=list(available_data_sets()),
                        help='dataset family that defines the subject universe')
    parser.add_argument('--raw_root', default=None, type=str,
                        help='raw BP4D+ tree (default: $RAW_DATA_PATH)')
    parser.add_argument('--dev_smoke', action='store_true', default=False,
                        help='report, instead of refusing, a corpus that cannot '
                             'hold the 70/10/20 shares. The record is written to '
                             'partition.dev.json and can never be pooled.')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='print the partition and exit without writing')
    return parser.parse_args(argv)


def _git_head(repo_root: str) -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=repo_root, capture_output=True, text=True,
                             check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def main(argv=None) -> int:
    args = get_args(argv)
    if not args.dry_run and not args.exp_root:
        raise SystemExit('--exp_root is required unless --dry_run.')

    root, name = resolve_corpus(args)
    sessions = corpus_sessions(root, name)
    universe = corpus_subjects(raw_root=root, data_set=name)
    print(f'[partition] data_set={name} universe={len(universe)} subject(s), '
          f'{len(sessions)} usable session(s) under {root}')

    plan = build_plan(universe, args.fold, split_seed=args.split_seed,
                      split_stratify=args.split_stratify,
                      val_seed=args.val_seed, dev_smoke=args.dev_smoke)
    print(f'[partition] fold {plan.fold}/{plan.n_folds}: '
          + ', '.join(f'{role}={plan.counts[role]}' for role in plan.roles)
          + f'  share='
          + '/'.join(f'{plan.share[role]:.3f}' for role in ('train', 'val', 'test')))
    print(f'[partition] digest {plan.digest}  universe digest '
          f'{plan.universe_digest[:16]}')

    if args.dry_run:
        print('[partition] --dry_run: nothing written')
        return 0

    rel = os.path.join('fold%d' % plan.fold,
                       'partition.dev.json' if plan.dev_smoke else 'partition.json')
    path = os.path.join(args.exp_root, rel)
    writer = {'stage': 'run_partition', 'argv': sys.argv[1:],
              'git': _git_head(os.path.dirname(os.path.dirname(
                  os.path.dirname(os.path.abspath(__file__)))))}
    existed = os.path.exists(path)
    try:
        stored = save_plan(path, plan, writer=writer)
    except ProtocolError as exc:
        raise SystemExit(f'[partition] {exc}') from None
    print(f'[partition] {"reused" if existed else "wrote"} {path} '
          f'(digest {stored.digest})')
    if load_plan(path).digest != plan.digest:          # read-back check
        raise SystemExit('[partition] read-back mismatch; the record is unusable')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
