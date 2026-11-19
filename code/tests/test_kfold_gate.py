"""The pooler's ADR-001 gate: five compliant folds of ONE partition, or nothing.

Five individually-correct folds can still be summed into a meaningless number:
different partitions, a dev plan, a fold that was never scored. Each of those is
a silent error at aggregation time, so they are refused by
``runners/run_evaluate_kfold.py`` before any table is printed.
"""
import json
import os

import pytest

from protocol import build_plan, save_plan, write_selection
from runners.run_evaluate_kfold import (gate_protocol_partitions,
                                        gate_test_artefacts)


def _universe():
    return ([f'F{i:03d}' for i in range(1, 83)]
            + [f'M{i:03d}' for i in range(1, 59) if i != 49])


def _write_fold(root, fold, *, seed=42, dev_smoke=False, scored=True,
                rule='fixed_epoch'):
    plan = build_plan(_universe(), fold, split_seed=seed,
                      split_stratify='sex', dev_smoke=dev_smoke)
    fold_dir = os.path.join(str(root), f'fold{fold}')
    save_plan(os.path.join(fold_dir, 'partition.json'), plan)
    if scored:
        test_dir = os.path.join(fold_dir, 'stage3', 'test')
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, 'metrics.json'), 'w') as handle:
            json.dump({'eval_role': 'test', 'n_clips': 10,
                       'partition_digest': plan.digest}, handle)
        write_selection(os.path.join(fold_dir, 'stage3', 'selection.json'), {
            'schema': 'adr-001/selection/1', 'rule': rule, 'eval_freq': 5,
            'selected_epoch': 149, 'criterion': 'joint_loss',
            'direction': 'min', 'criterion_value': 0.1, 'n_evals': 30,
            'pinned': {}, 'dev_smoke': False})
    return plan


def test_five_compliant_folds_pass_the_gate(tmp_path):
    for fold in range(5):
        _write_fold(tmp_path, fold)
    plans = gate_protocol_partitions(str(tmp_path), list(range(5)))
    assert len(plans) == 5
    assert gate_test_artefacts(str(tmp_path), list(range(5))) == \
        {fold: 'fixed_epoch' for fold in range(5)}
    # the five test roles are a partition of the corpus
    covered = [s for plan in plans.values() for s in plan.roles['test']]
    assert len(covered) == len(set(covered)) == len(_universe())


def test_partial_sweep_is_refused_unless_marked(tmp_path):
    for fold in range(4):
        _write_fold(tmp_path, fold)
    with pytest.raises(SystemExit, match='exactly K=5'):
        gate_protocol_partitions(str(tmp_path), list(range(4)))
    assert len(gate_protocol_partitions(str(tmp_path), list(range(4)),
                                        allow_partial=True)) == 4


def test_missing_partition_record_is_refused(tmp_path):
    for fold in range(4):
        _write_fold(tmp_path, fold)
    with pytest.raises(SystemExit, match='missing partition record'):
        gate_protocol_partitions(str(tmp_path), list(range(5)))


def test_a_dev_plan_can_never_be_pooled(tmp_path):
    for fold in range(5):
        _write_fold(tmp_path, fold, dev_smoke=True)
    with pytest.raises(SystemExit, match='dev_smoke'):
        gate_protocol_partitions(str(tmp_path), list(range(5)), allow_partial=True)


def test_mixed_partition_rules_are_refused(tmp_path):
    for fold in range(5):
        _write_fold(tmp_path, fold, seed=42 if fold < 4 else 7)
    with pytest.raises(SystemExit, match='ONE'):
        gate_protocol_partitions(str(tmp_path), list(range(5)))


def test_unscored_fold_is_refused(tmp_path):
    for fold in range(5):
        _write_fold(tmp_path, fold, scored=(fold != 2))
    with pytest.raises(SystemExit, match='no test artefact'):
        gate_test_artefacts(str(tmp_path), list(range(5)))


def test_skipped_test_role_is_refused(tmp_path):
    for fold in range(5):
        _write_fold(tmp_path, fold)
    path = os.path.join(str(tmp_path), 'fold1', 'stage3', 'test', 'metrics.json')
    with open(path, 'w') as handle:
        json.dump({'eval_role': 'test', 'skipped': True,
                   'skip_reason': 'no usable clip'}, handle)
    with pytest.raises(SystemExit, match='not scorable'):
        gate_test_artefacts(str(tmp_path), list(range(5)))


def test_mixed_selection_rules_are_refused(tmp_path):
    for fold in range(5):
        _write_fold(tmp_path, fold, rule='best_val' if fold == 3
                    else 'fixed_epoch')
    with pytest.raises(SystemExit, match='different selection rules'):
        gate_test_artefacts(str(tmp_path), list(range(5)))
