"""The shipped distortion tiers (v2) and the corpus-selection contract.

Two invariants that used to be comments only:

* every ``tir_roi*`` config -- and the three code defaults the survey / gate
  probe / eval runners fall back on -- defines the SAME task -> group mapping.
  A drift there silently changes every per-group table, because the Stage-2 /
  Stage-3 agreement check compared ``task_set`` but never ``task_groups``.
* the mapping is a PARTITION of ``T1``..``T10`` (each task in exactly one tier).
  ``runners/run_survey_tir_resp._aggregate`` and
  ``notes/eval/tir_resp/gate_model_input.py::group_of`` assume single
  membership, so a task in two groups would be double-counted and a task in
  none would vanish from the tally (``breakdown_by_label`` skips an empty
  label list).

Rationale of record: ``code/notes/design/TIR_ROI_RESP_TasksDistortion.md``.
The plan: ``code/notes/plan/task_tiers_regroup_plan.md``.

Run from ``code/``:  ``python -m pytest tests/test_task_group_configs.py -q``
"""
import argparse
import glob
import importlib
import importlib.util
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: v2 (2026-10-24): T5 high->low, T1 high->moderate, T10 moderate->high.
EXPECTED_TIERS = {
    'low': ('T2', 'T3', 'T5'),
    'moderate': ('T1', 'T4'),
    'high': ('T6', 'T7', 'T8', 'T9', 'T10'),
}

#: every config that ships a ``task_groups:`` block.
TIR_ROI_CONFIGS = [
    'configs/pretrain/stage2_local_tir_roi_resp.yaml',
    'configs/pretrain/stage2_local_tir_roi_crossmae.yaml',
    'configs/pretrain/stage2_hpc_tir_roi_resp.yaml',
    'configs/finetune/resp_tir_roi_local.yaml',
    'configs/finetune/resp_tir_roi_local_matched.yaml',
    'configs/finetune/resp_tir_roi_crossmae.yaml',
    'configs/finetune/resp_tir_roi_hpc.yaml',
    'configs/finetune/resp_tir_roi_hpc_smoothl1.yaml',
]

REGROUP_STRING = 'low=T2|T3|T5;moderate=T1|T4;high=T6|T7|T8|T9|T10'


def _args_for(rel, monkeypatch):
    """Parse a shipped config with its REAL runner parser."""
    monkeypatch.setattr(sys, 'argv', ['prog', '-c', os.path.join(CODE_DIR, rel)])
    runner = ('runners.run_pretrain' if rel.startswith('configs/pretrain/')
              else 'runners.run_waveform')
    return importlib.import_module(runner).get_args()


def _gate_probe():
    """Import ``notes/eval/tir_resp/gate_model_input.py`` by path (not a package)."""
    path = os.path.join(CODE_DIR, 'notes/eval/tir_resp/gate_model_input.py')
    spec = importlib.util.spec_from_file_location('_gate_model_input', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# the canonical definition
# --------------------------------------------------------------------------- #
def test_shipped_constant_is_the_v2_partition():
    from data.task_groups import SHIPPED_TASK_GROUPS, parse_task_groups

    assert parse_task_groups(SHIPPED_TASK_GROUPS) == EXPECTED_TIERS


def test_shipped_constant_covers_every_task_exactly_once():
    """A partition, not just a set of subsets -- see the module docstring."""
    from data.task_groups import SHIPPED_TASK_GROUPS, parse_task_groups

    levels = parse_task_groups(SHIPPED_TASK_GROUPS)
    seen = [task for tasks in levels.values() for task in tasks]
    assert sorted(seen, key=lambda t: int(t[1:])) == \
        [f'T{i}' for i in range(1, 11)]
    assert len(seen) == len(set(seen)), f'duplicated task in {levels!r}'


def test_tir_roi_config_inventory_is_complete():
    """A new ``task_groups:`` config must be registered in this test's list."""
    found = sorted(
        os.path.relpath(path, CODE_DIR).replace(os.sep, '/')
        for path in glob.glob(os.path.join(CODE_DIR, 'configs/*/*.yaml'))
        if 'task_groups:' in open(path).read())
    assert found == sorted(TIR_ROI_CONFIGS)


@pytest.mark.parametrize('cfg', TIR_ROI_CONFIGS)
def test_config_ships_the_shipped_tiers(cfg, monkeypatch):
    from data.task_groups import parse_task_groups

    a = _args_for(cfg, monkeypatch)
    assert parse_task_groups(a.task_groups) == EXPECTED_TIERS, (
        f'{cfg}: task_groups drifted from the shipped tiers '
        f'({a.task_groups!r})')
    assert str(a.task_set).strip() == '', (
        f'{cfg}: task_set must stay empty -- the tiers are LABELS by default, '
        f'so a non-empty default would silently retrain on a tier subset')


# --------------------------------------------------------------------------- #
# the three code defaults (the only places the mapping was hardcoded)
# --------------------------------------------------------------------------- #
def test_survey_default_matches_the_shipped_tiers():
    from data.task_groups import parse_task_groups
    from runners import run_survey_tir_resp as survey

    assert parse_task_groups(survey.DEFAULT_TASK_GROUPS) == EXPECTED_TIERS


def test_gate_probe_default_matches_the_shipped_tiers():
    from data.task_groups import parse_task_groups

    assert parse_task_groups(_gate_probe().DEFAULT_GROUPS) == EXPECTED_TIERS


def test_gate_sbatch_default_matches_the_shipped_tiers():
    """bash cannot import Python, so the literal is pinned here instead."""
    from data.task_groups import parse_task_groups

    path = os.path.join(CODE_DIR, 'scripts/hpc/gate_model_input.sbatch')
    with open(path) as fh:
        text = fh.read()
    m = re.search(r'^TASK_GROUPS="\$\{TASK_GROUPS:-([^}]*)\}"', text, re.M)
    assert m, 'gate_model_input.sbatch no longer exports a TASK_GROUPS default'
    assert parse_task_groups(m.group(1)) == EXPECTED_TIERS


# --------------------------------------------------------------------------- #
# tier-restricted corpus selection (already plumbed; now pinned)
# --------------------------------------------------------------------------- #
def test_low_tier_selects_exactly_its_three_tasks():
    from data.task_groups import SHIPPED_TASK_GROUPS, resolve_task_selection

    sel = resolve_task_selection(task_set='low', task_groups=SHIPPED_TASK_GROUPS)
    assert sel.tasks == ('T2', 'T3', 'T5')
    assert sel.requested == ('low',)


def test_low_plus_moderate_selects_five_tasks():
    from data.task_groups import SHIPPED_TASK_GROUPS, resolve_task_selection

    sel = resolve_task_selection(task_set='low,moderate',
                                 task_groups=SHIPPED_TASK_GROUPS)
    assert sel.tasks == ('T1', 'T2', 'T3', 'T4', 'T5')


def test_empty_task_set_keeps_the_whole_corpus():
    from data.task_groups import SHIPPED_TASK_GROUPS, resolve_task_selection

    sel = resolve_task_selection(task_set='', task_groups=SHIPPED_TASK_GROUPS)
    assert sel.tasks is None and not sel.active


def test_every_task_maps_to_exactly_one_tier():
    from data.task_groups import (SHIPPED_TASK_GROUPS, levels_for_task,
                                  parse_task_groups, levels_by_task)

    override = levels_by_task(parse_task_groups(SHIPPED_TASK_GROUPS))
    for i in range(1, 11):
        assert len(levels_for_task(override, f'T{i}')) == 1, f'T{i}'
    assert levels_for_task(override, 'T5') == ('low',)
    assert levels_for_task(override, 'T1') == ('moderate',)
    assert levels_for_task(override, 'T10') == ('high',)
    # an unparsable label is empty, never an exception
    assert levels_for_task(override, 'not-a-task') == ()


def test_unmapped_tasks_flags_an_incomplete_override():
    from data.task_groups import parse_task_groups, unmapped_tasks

    partial = {'low': ('T2', 'T3', 'T5')}       # the other seven are missing
    override = {t: ('low',) for t in partial['low']}
    assert unmapped_tasks(['T3', 'T7', 'T10'], override) == ['T10', 'T7']
    assert unmapped_tasks(['T3'], override) == []
    # no override -> nothing to warn about, even for an empty map
    assert unmapped_tasks(['T7'], None) == []
    assert parse_task_groups('') == {}


# --------------------------------------------------------------------------- #
# BDD 2: relabel an EXISTING dump (aggregation-only, no re-training)
# --------------------------------------------------------------------------- #
def _write_run(tmp_path, task, old_level):
    """One 8 s clip whose prediction IS the reference (pearson 1.0)."""
    fs = 100.0
    ref = np.sin(2 * np.pi * 0.3 * np.arange(1600) / fs)
    sig = tmp_path / 'Resp_Volts.txt'
    np.savetxt(sig, np.repeat(ref, 10))            # 1000 Hz native
    np.save(tmp_path / 'preds.npy', ref[:800][None, :])
    np.save(tmp_path / 'targets.npy', ref[:800][None, :])
    with open(tmp_path / 'entries.json', 'w') as fh:
        json.dump([{'session': f'S001_{task}', 't_start': 0.0,
                    'signals_file': str(sig), 'task': task,
                    'task_level': [old_level]}], fh)


def _run_session(tmp_path, task_groups=''):
    from runners.run_evaluate_session import main as session_main

    args = argparse.Namespace(
        pred_dir=str(tmp_path), pred_path='', target_path='', entries_path='',
        waveform='resp', fs=100.0, tier='1', band='', smooth='',
        out=str(tmp_path / 'm.json'), fig_dir=str(tmp_path / 'fig'),
        zoom_s=20.0, max_sessions=0, window_s=30.0, raw_fs=1000.0,
        task_groups=task_groups)
    session_main(args)
    with open(tmp_path / 'm.json') as fh:
        return json.load(fh)


def test_override_relabels_a_dump_written_under_the_old_tiers(tmp_path):
    """T5 was recorded as 'high'; the override must move it to 'low'."""
    _write_run(tmp_path, task='T5', old_level='high')
    doc = _run_session(tmp_path, task_groups=REGROUP_STRING)

    assert set(doc['per_group']) == {'low'}
    assert doc['per_group']['low']['n'] == 1
    assert doc['task_group_source'] == 'cli'
    assert doc['task_levels']['T5'] == ['low']
    assert doc['task_groups']['low'] == ['T2', 'T3', 'T5']


def test_recorded_labels_are_kept_when_no_override_is_given(tmp_path):
    """The default path is bit-for-bit the pre-v2 behaviour."""
    _write_run(tmp_path, task='T5', old_level='high')
    doc = _run_session(tmp_path)

    assert set(doc['per_group']) == {'high'}
    assert doc['task_group_source'] == 'entries'
    assert doc['task_groups'] == {} and doc['task_levels'] == {}


def test_override_warns_about_tasks_it_does_not_cover(tmp_path, capsys):
    _write_run(tmp_path, task='T7', old_level='high')
    _run_session(tmp_path, task_groups='low=T2|T3|T5')   # T7 unmapped

    out = capsys.readouterr().out
    assert 'WARNING' in out and 'T7' in out and 'per_group' in out


# --------------------------------------------------------------------------- #
# BDD 3 + 4: the Stage-2/Stage-3 corpus note
# --------------------------------------------------------------------------- #
def _contract_args(task_set='', task_groups=REGROUP_STRING, tasks=''):
    return SimpleNamespace(data_set='tir_roi_resp', roi_landmarks='',
                           roi_padding=0.2, input_size=112, tasks=tasks,
                           task_set=task_set, task_groups=task_groups)


OLD_GROUPS = ('low=T2|T3;moderate=T4|T7|T8|T10;high=T1|T5|T6|T9')


def test_contract_notes_a_task_set_mismatch(capsys):
    """BDD 3: Stage 2 on 'low', Stage 3 on 'low,moderate' -> a note fires."""
    from runners.run_waveform import check_stage2_roi_contract

    check_stage2_roi_contract(_contract_args(task_set='low,moderate'),
                              _contract_args(task_set='low'))
    out = capsys.readouterr().out
    assert 'task_set differs' in out
    assert 'not evaluated on exactly the pre-trained corpus' in out


def test_contract_ignores_task_groups_when_no_group_is_selected(capsys):
    """BDD 4: label-only regrouping of an existing dump must NOT warn."""
    from runners.run_waveform import check_stage2_roi_contract

    check_stage2_roi_contract(_contract_args(task_groups=REGROUP_STRING),
                              _contract_args(task_groups=OLD_GROUPS))
    out = capsys.readouterr().out
    assert 'task_groups differs' not in out
    assert 'note:' not in out


def test_contract_notes_task_groups_when_a_group_is_selected(capsys):
    """Same task_set, different definitions -> the corpus DID change."""
    from runners.run_waveform import check_stage2_roi_contract

    check_stage2_roi_contract(
        _contract_args(task_set='low', task_groups=REGROUP_STRING),
        _contract_args(task_set='low', task_groups=OLD_GROUPS))
    out = capsys.readouterr().out
    assert 'task_groups differs' in out
    assert 'low' in out


def test_contract_raises_on_a_real_roi_mismatch():
    """Unchanged behaviour for the keys that DO abort."""
    from runners.run_waveform import check_stage2_roi_contract

    stage3 = _contract_args()
    stage3.input_size = 64
    with pytest.raises(SystemExit, match='ROI does not match'):
        check_stage2_roi_contract(stage3, _contract_args())
