"""Regressions for the four defects the first end-to-end smoke exposed.

Each of these passed the unit suite and still broke a real fold, which is why
they live here as named cases rather than as a comment in the plan.
"""
import argparse
import sys

import pytest

from data import datasets as ds
from protocol import ProtocolError, build_plan, make_eval_record
from protocol.constants import SCHEMA_EVAL


def _plan(fold=0):
    return build_plan([f'S{i:03d}' for i in range(1, 7)], fold,
                      split_seed=42, split_stratify='none', dev_smoke=True)


# --------------------------------------------------------------------------- #
# 1. one args namespace, several roles
# --------------------------------------------------------------------------- #
def test_building_two_roles_does_not_mutate_the_shared_args(monkeypatch):
    """A role build must not leak its subject list into the NEXT role.

    The shipped bug set ``args.subjects`` to the train role, so the val build
    intersected the val role against those subjects and produced an empty
    validation set -- after a perfectly healthy train build.
    """
    seen = []

    def fake_source(view_args):
        seen.append(list(view_args.subjects))
        return 'dataset'

    monkeypatch.setattr(ds, '_build_pretrain_source', fake_source)
    args = argparse.Namespace(data_set='tir_roi', subjects='')
    plan = _plan()

    ds.build_pretraining_dataset(args, plan, role='train')
    ds.build_pretraining_dataset(args, plan, role='val')

    assert seen[0] == list(plan.roles['train'])
    assert seen[1] == list(plan.roles['val']), 'the val build saw the train list'
    assert args.subjects == '', 'the shared args namespace was mutated'


def test_corpus_view_does_not_mutate_the_shared_args(monkeypatch):
    monkeypatch.setattr(ds, '_build_pretrain_source', lambda view: 'dataset')
    args = argparse.Namespace(data_set='tir_roi', subjects=['F001'])
    ds.build_pretraining_corpus_dataset(args)
    assert args.subjects == ['F001']


# --------------------------------------------------------------------------- #
# 2./3. the Stage-3 entry point
# --------------------------------------------------------------------------- #
def test_stage3_parser_declares_raw_root():
    """`--raw_root` is read by the builders, so it must be spellable."""
    import runners.run_waveform as rw

    argv = sys.argv
    try:
        sys.argv = ['run_waveform.py', '--exp_root', '/tmp/x', '--fold', '0',
                    '--raw_root', '/data/raw']
        args = rw.get_args()
    finally:
        sys.argv = argv
    assert args.raw_root == '/data/raw'


def test_stage3_parser_declares_the_test_role():
    import runners.run_waveform as rw

    argv = sys.argv
    try:
        sys.argv = ['run_waveform.py', '--exp_root', '/tmp/x', '--fold', '0',
                    '--eval_role', 'test']
        args = rw.get_args()
    finally:
        sys.argv = argv
    assert args.eval_role == 'test'
    assert args.overwrite_test is False


# --------------------------------------------------------------------------- #
# 4. the evaluation record keeps only numbers
# --------------------------------------------------------------------------- #
def test_eval_record_drops_non_numeric_components():
    """`spectral_metrics` returns `psd_band_bins=None`; a record is numeric."""
    record = make_eval_record(
        epoch=0, value_kind='joint_loss', criterion_value=1.5,
        counts={'clips': 4, 'subjects': 2}, seed={'val_seed': 42},
        components={'time': 0.5, 'pearson': 0.25, 'psd_band_bins': None,
                    'psd_skipped': 3, 'label': ['low']})
    assert record['schema'] == SCHEMA_EVAL
    assert record['components'] == {'time': 0.5, 'pearson': 0.25,
                                    'psd_skipped': 3.0}


def test_eval_record_still_requires_counts_and_seeds():
    with pytest.raises(ProtocolError, match='clips and subjects'):
        make_eval_record(epoch=0, value_kind='joint_loss', criterion_value=1.0,
                         counts={'clips': 1}, seed={'val_seed': 1})
