"""ADR-001 protocol tests — one per invariant and falsifiable criterion.

Every test here is derived from
``code/notes/design/ADR-001-subject-disjoint-split-and-selection.md``; the plan
(``code/notes/plan/ADR-001-subject-kfold-plan.md``) maps clause -> test name.
Run from ``code/``::

    python -m pytest tests/test_protocol.py -q
"""
import dataclasses
import json
import os
import subprocess
import sys

import pytest

from protocol import (DEFAULT_VAL_SEED, K, ROLE_SHARE, FoldPlan, LeakGuard,
                      ProtocolError, SelectionTracker, append_jsonl,
                      assert_roles_disjoint, build_plan, load_plan,
                      make_eval_record, read_jsonl, save_plan, split_universe,
                      to_record, val_mask_seed, validate_plan)
from protocol.constants import ROLES
from protocol.partition import allocate_strata


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _universe():
    """The real corpus shape: 82 female + 57 male = 139 subjects (M049 unusable)."""
    female = [f'F{i:03d}' for i in range(1, 83)]
    male = [f'M{i:03d}' for i in range(1, 59) if i != 49]
    return female + male


SEED = 42


def _plan(fold=0, **kwargs):
    kwargs.setdefault('split_seed', SEED)
    kwargs.setdefault('split_stratify', 'sex')
    return build_plan(_universe(), fold, **kwargs)


def _small_universe(n=6):
    return [f'F{i:03d}' for i in range(1, n + 1)]


# --------------------------------------------------------------------------- #
# I1 / I3 — disjointness and completeness
# --------------------------------------------------------------------------- #
def test_roles_are_pairwise_disjoint():
    """I1: a subject belongs to exactly one role, for every fold."""
    for fold in range(K):
        roles = _plan(fold).roles
        seen = set()
        for role in ROLES:
            members = set(roles[role])
            assert not (seen & members), f'fold {fold}: {role} overlaps'
            seen |= members


def test_roles_partition_the_universe():
    """I3: the three roles cover the universe exactly once."""
    universe = _universe()
    for fold in range(K):
        plan = _plan(fold)
        union = set().union(*(set(plan.roles[role]) for role in ROLES))
        assert union == set(universe)
        assert sum(plan.counts.values()) == len(universe)


def test_roles_are_sorted_and_unique():
    """Records are canonical: sorted, duplicate-free role lists."""
    plan = _plan(1)
    for role in ROLES:
        members = list(plan.roles[role])
        assert members == sorted(members)
        assert len(members) == len(set(members))


def test_assert_roles_disjoint_rejects_an_overlap():
    with pytest.raises(ProtocolError, match='disjoint'):
        assert_roles_disjoint({'train': ['F001'], 'val': ['F001'], 'test': []})


# --------------------------------------------------------------------------- #
# I2 / Criterion 1 — the partition is a pure function
# --------------------------------------------------------------------------- #
def test_partition_is_a_pure_function():
    """The same arguments give the same partition, whatever the input order."""
    base = _plan(2)
    shuffled = list(reversed(_universe()[7:] + _universe()[:7]))
    for variant in (shuffled, set(_universe()), tuple(_universe())):
        other = build_plan(variant, 2, split_seed=SEED, split_stratify='sex')
        assert other.digest == base.digest
        assert other.roles == base.roles


def test_caps_and_filters_cannot_move_a_boundary():
    """`build_plan` has no cap/filter/task argument: only (universe, fold, seeds).

    The universe is the corpus, so a run-time cap narrows what a loader reads,
    never who is held out. This asserts the signature, because a future
    ``max_sessions=`` parameter here is exactly how I2 would regress.
    """
    import inspect
    params = set(inspect.signature(build_plan).parameters)
    assert params == {'universe', 'fold', 'split_seed', 'split_stratify',
                      'val_seed', 'dev_smoke'}


def test_two_constructions_are_byte_identical(tmp_path):
    """Criterion 1: two independent constructions produce one partition.

    With the same ``writer`` the files are byte-identical. A different writer
    (Stage 2 vs the standalone utility) legitimately changes only the provenance
    block, so the second half compares what the partition IS: roles, counts,
    shares, universe and digest -- never the writer.
    """
    first = tmp_path / 'a' / 'partition.json'
    twin = tmp_path / 'a_twin' / 'partition.json'
    second = tmp_path / 'b' / 'partition.json'
    save_plan(str(first), _plan(3), writer={'stage': 2})
    save_plan(str(twin), _plan(3), writer={'stage': 2})
    save_plan(str(second), _plan(3), writer={'stage': 3})
    assert first.read_bytes() == twin.read_bytes()

    left = json.loads(first.read_text())
    right = json.loads(second.read_text())
    assert left['roles'] == right['roles']
    assert left['digest'] == right['digest']
    assert left['counts'] == right['counts']
    assert left['share'] == right['share']
    assert left['universe'] == right['universe']


def test_hashseed_does_not_change_the_record(code_dir):
    """A set-derived universe must not leak iteration order into the record."""
    script = (
        'from protocol import build_plan;'
        "u=['F%03d'%i for i in range(1,83)]+"
        "['M%03d'%i for i in range(1,59) if i!=49];"
        "p=build_plan(set(u),0,split_seed=42,split_stratify='sex');"
        'print(p.digest, p.counts)'
    )
    outputs = set()
    for hashseed in ('0', '1', '12345'):
        env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=code_dir)
        result = subprocess.run([sys.executable, '-c', script], cwd=code_dir,
                                env=env, capture_output=True, text=True,
                                check=True)
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, outputs


def test_stratified_partition_requires_a_seed():
    with pytest.raises(ValueError, match='requires an explicit split_seed'):
        split_universe(_universe(), K, split_seed=None, split_stratify='sex')
    # Through the public entry point the same refusal is a ProtocolError.
    with pytest.raises(ProtocolError, match='requires an explicit split_seed'):
        build_plan(_universe(), 0, split_stratify='sex')


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match='split_stratify'):
        split_universe(_universe(), K, split_seed=SEED, split_stratify='gender')


# --------------------------------------------------------------------------- #
# Criterion 1 — realised shares, and the dev escape
# --------------------------------------------------------------------------- #
def test_share_within_two_points():
    """Realised 70/10/20 within +/-2 pp on the 139-subject corpus."""
    universe = _universe()
    assert [len(_plan(fold).roles['test']) for fold in range(K)] == \
        [28, 28, 28, 28, 27]
    for fold in range(K):
        plan = _plan(fold)
        assert plan.counts['val'] == 14
        assert plan.counts['train'] == len(universe) - plan.counts['test'] - 14
        for role in ROLES:
            drift = abs(plan.share[role] - ROLE_SHARE[role])
            assert drift <= 0.02 + 1e-12, (fold, role, plan.share)


def test_dev_smoke_relaxes_the_share_tolerance():
    """A 6-subject corpus cannot reach 70/10/20; it is allowed only as dev."""
    with pytest.raises(ProtocolError, match='Criterion 1'):
        build_plan(_small_universe(6), 0)
    plan = build_plan(_small_universe(6), 0, dev_smoke=True)
    assert plan.dev_smoke is True
    assert sum(plan.counts.values()) == 6


def test_val_draw_is_stratified():
    """With stratify='sex', the val draw is proportional, not just random."""
    for fold in range(K):
        plan = _plan(fold)
        test = set(plan.roles['test'])
        pool = [s for s in plan.universe if s not in test]
        female_pool = sum(1 for s in pool if s.startswith('F'))
        expected = len(plan.roles['val']) * female_pool / len(pool)
        got = sum(1 for s in plan.roles['val'] if s.startswith('F'))
        assert abs(got - expected) <= 1.0, (fold, got, expected)


def test_allocate_strata_sums_exactly():
    allocation = allocate_strata(9, {'F': 82, 'M': 57})
    assert sum(allocation.values()) == 9
    assert abs(allocation['F'] - 9 * 82 / 139) <= 1
    assert allocate_strata(0, {'F': 1}) == {'F': 0}


# --------------------------------------------------------------------------- #
# I2 — seeds, constants, bounds
# --------------------------------------------------------------------------- #
def test_val_seed_defaults_to_split_seed():
    assert _plan(0, split_seed=7).val_seed == 7
    assert build_plan(_universe(), 0).val_seed == DEFAULT_VAL_SEED
    assert _plan(0, val_seed=99).val_seed == 99


def test_fold_out_of_range_is_rejected():
    for fold in (-1, K, K + 3):
        with pytest.raises(ProtocolError, match='fold'):
            _plan(fold)


def test_n_folds_is_the_constant():
    plan = _plan(0)
    assert plan.n_folds == K
    with pytest.raises(ProtocolError, match='new ADR'):
        validate_plan(dataclasses.replace(plan, n_folds=4))


def test_universe_smaller_than_k_is_rejected():
    with pytest.raises(ProtocolError, match='at least K'):
        build_plan(_small_universe(4), 0, dev_smoke=True)


# --------------------------------------------------------------------------- #
# I4 / D4 — one write-once record
# --------------------------------------------------------------------------- #
def test_save_plan_is_write_once(tmp_path):
    path = str(tmp_path / 'partition.json')
    first = save_plan(path, _plan(0), writer={'stage': 2})
    again = save_plan(path, _plan(0), writer={'stage': 3})   # same partition
    assert first.digest == again.digest
    with pytest.raises(ProtocolError, match='different partition'):
        save_plan(path, _plan(1))


def test_partition_utility_matches_stage_2(tmp_path):
    """A standalone creator and Stage 2 must produce one identical record."""
    path = str(tmp_path / 'partition.json')
    save_plan(path, _plan(4), writer={'stage': 'run_partition'})
    before = open(path, 'rb').read()
    save_plan(path, _plan(4), writer={'stage': 2})
    assert open(path, 'rb').read() == before, 'the record was rewritten'
    assert load_plan(path).digest == _plan(4).digest


def test_load_rejects_an_edited_record(tmp_path):
    path = tmp_path / 'partition.json'
    save_plan(str(path), _plan(0))
    record = json.loads(path.read_text())
    record['roles']['train'] = record['roles']['train'][1:]     # drop a subject
    path.write_text(json.dumps(record))
    with pytest.raises(ProtocolError, match='edited'):
        load_plan(str(path))


def test_load_rejects_an_unknown_schema(tmp_path):
    path = tmp_path / 'partition.json'
    path.write_text(json.dumps({'schema': 'adr-001/partition/999'}))
    with pytest.raises(ProtocolError, match='schema'):
        load_plan(str(path))


def test_record_carries_counts_shares_and_universe():
    record = to_record(_plan(2))
    assert record['schema'] == 'adr-001/partition/1'
    assert record['counts'] == {'train': 97, 'val': 14, 'test': 28}
    assert record['universe']['size'] == 139
    assert record['universe']['subjects'] == sorted(_universe())
    assert set(record['roles']) == set(ROLES)


# --------------------------------------------------------------------------- #
# I6 — validation determinism
# --------------------------------------------------------------------------- #
def test_mask_seed_varies_per_batch_not_per_epoch():
    """Same base -> same seed for the same batch, different across batches."""
    base = 1234
    first_pass = [val_mask_seed(base, index) for index in range(8)]
    second_pass = [val_mask_seed(base, index) for index in range(8)]
    assert first_pass == second_pass                       # comparable epochs
    assert len(set(first_pass)) == len(first_pass)          # distinct masks
    assert val_mask_seed(base, 0) == base


# --------------------------------------------------------------------------- #
# I7 — selection rule
# --------------------------------------------------------------------------- #
def test_fixed_epoch_selects_the_last_epoch():
    tracker = SelectionTracker(epochs=150, eval_freq=5)
    for epoch, value in ((0, 0.9), (5, 0.7), (145, 0.4), (149, 0.5)):
        tracker.observe(epoch, value)
    record = tracker.record()
    assert record['rule'] == 'fixed_epoch'
    assert record['selected_epoch'] == 149
    assert record['criterion_value'] == 0.5
    assert record['n_evals'] == 4
    assert tracker.writes_best_checkpoint is False


def test_best_val_respects_the_criterion_direction():
    loss = SelectionTracker(epochs=150, rule='best_val', criterion='joint_loss')
    loss.observe(0, 0.9)
    loss.observe(1, 0.3)
    loss.observe(2, 0.4)
    assert loss.best_epoch == 1 and loss.record()['direction'] == 'min'

    corr = SelectionTracker(epochs=150, rule='best_val',
                            criterion='pearson_band')
    corr.observe(0, 0.40)
    corr.observe(1, 0.60)
    corr.observe(2, 0.55)
    assert corr.best_epoch == 1 and corr.record()['direction'] == 'max'


def test_evaluation_schedule():
    tracker = SelectionTracker(epochs=150, eval_freq=5)
    assert tracker.should_eval(0)
    assert not tracker.should_eval(1)
    assert tracker.should_eval(5)
    assert tracker.should_eval(149)          # last epoch is always evaluated
    off = SelectionTracker(epochs=150, eval_freq=0)
    assert not off.should_eval(149)


def test_best_val_without_evaluations_is_refused():
    tracker = SelectionTracker(epochs=10, rule='best_val', eval_freq=5)
    with pytest.raises(ProtocolError, match='nothing to select on'):
        tracker.record()


def test_best_val_with_evaluation_off_is_refused():
    with pytest.raises(ProtocolError, match='eval_freq=0'):
        SelectionTracker(epochs=10, rule='best_val', eval_freq=0)


def test_eval_freq_below_the_cost_budget_is_refused():
    with pytest.raises(ProtocolError, match='Criterion 4'):
        SelectionTracker(epochs=10, eval_freq=1)
    assert SelectionTracker(epochs=10, eval_freq=1, dev_smoke=True).eval_freq == 1


def test_selection_record_is_self_describing():
    tracker = SelectionTracker(epochs=3, eval_freq=3,
                               n_val_clips=2000, n_train_clips=13000)
    tracker.observe(2, 0.25)
    record = tracker.record(pinned={'kept_tir': 0.25})
    assert record['schema'] == 'adr-001/selection/1'
    for key in ('rule', 'eval_freq', 'selected_epoch', 'criterion',
                'criterion_value', 'n_val_clips', 'pinned'):
        assert key in record
    assert record['pinned'] == {'kept_tir': 0.25}


# --------------------------------------------------------------------------- #
# Criterion 3 — the leakage abort
# --------------------------------------------------------------------------- #
def test_leak_guard_allows_only_the_train_role():
    plan = _plan(0)
    guard = LeakGuard(plan)
    guard.check([plan.roles['train'][0]], epoch=0, batch_idx=0)   # allowed
    with pytest.raises(ProtocolError, match='LEAK'):
        guard.check([plan.roles['test'][0]], epoch=0, batch_idx=1)
    with pytest.raises(ProtocolError, match='held-out'):
        guard.check([plan.roles['val'][0]], epoch=3, batch_idx=10)


def test_leak_guard_sampling_schedule():
    guard = LeakGuard(_plan(0), every=10)
    assert guard.should_check(0, 1)          # every batch of the first epoch
    assert guard.should_check(1, 10)         # one in ten afterwards
    assert not guard.should_check(1, 11)
    guard.check(['NOT_A_SUBJECT'], epoch=1, batch_idx=11)   # skipped, no raise


# --------------------------------------------------------------------------- #
# records — schema discipline
# --------------------------------------------------------------------------- #
def test_eval_record_requires_counts_and_seeds():
    with pytest.raises(ProtocolError, match='clips and subjects'):
        make_eval_record(epoch=0, value_kind='joint_loss', criterion_value=1.0,
                         counts={'clips': 10}, seed={'val_seed': 42})
    with pytest.raises(ProtocolError, match='seeds'):
        make_eval_record(epoch=0, value_kind='joint_loss', criterion_value=1.0,
                         counts={'clips': 10, 'subjects': 2}, seed={})


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    record = make_eval_record(epoch=5, value_kind='joint_loss',
                              criterion_value=0.42,
                              counts={'clips': 120, 'subjects': 14},
                              seed={'val_seed': 42, 'mask_base': 7})
    append_jsonl(str(path), record)
    append_jsonl(str(path), dict(record, epoch=10))
    lines = read_jsonl(str(path))
    assert [line['epoch'] for line in lines] == [5, 10]
    assert lines[0]['schema'] == 'adr-001/eval/1'

