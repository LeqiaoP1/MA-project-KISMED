"""BDD acceptance criteria for the AU-Occurrence Probe (TIR-ROI + Resp).

Protocol of record: ``analysis/tir_resp/AU_OccurenceProbe.md`` §10.2. Each test
maps to a Gherkin scenario there. They are designed to run WITHOUT the raw BP4D
tree (no media): they exercise the pure geometry/label/gating helpers, the
metric engine on synthetic tensors, and the runner's plumbing/config.
"""
import argparse
import importlib
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.au_probe import build_au_probe_model
from data.au_tir_roi_dataset import (
    ROI_BORDERLINE_AUS, ROI_FORBIDDEN_AUS, TIER_A_AU_LIST, au_probe_num_frames,
    centre_frame_index, check_au_gate, classify_au_set, load_au_table,
    resolve_labels)
from engines.au_probe import (_gather_predictions, evaluate_au, trivial_f1,
                              tune_per_au_thresholds)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================================================================== #
# Feature: Geometry and Temporal Stride Contract
# =========================================================================== #
def _probe_args(**over):
    base = dict(streams='tir', tir_channels=3, tubelet='2,16,16', fps=25.0,
                clip_duration=8.0, temporal_stride=2, num_frames=0, pool='mean',
                enc_embed_dim=768, enc_depth=12, enc_num_heads=12, mlp_ratio=4.0,
                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                input_size=64)
    base.update(over)
    return argparse.Namespace(**base)


def test_geometry_helper_matches_the_checkpoint_math():
    # 8 s @ 25 fps, stride 2 -> 100 frames; stride 1 -> 200 (the mis-centre trap)
    assert au_probe_num_frames(8.0, 25.0, 2, 2) == 100
    assert au_probe_num_frames(8.0, 25.0, 1, 2) == 200
    assert centre_frame_index(200) == 100


def test_probe_model_uses_temporal_stride():
    """Given a stride-2 checkpoint, the probe builds with num_frames 100."""
    m2 = build_au_probe_model(_probe_args(temporal_stride=2), num_classes=7)
    assert m2.num_frames == 100
    assert m2.n_visual == (100 // 2) * (64 // 16) ** 2      # 50 * 16 = 800

    m1 = build_au_probe_model(_probe_args(temporal_stride=1), num_classes=7)
    assert m1.num_frames == 200                              # the wrong geometry
    assert m1.n_visual == 1600


def test_runner_exposes_the_geometry_flags():
    """The stride/ROI plumbing must exist, or the label silently mis-centres."""
    with open(os.path.join(CODE_DIR, 'runners', 'run_au_probe.py')) as fh:
        src = fh.read()
    for flag in ('--temporal_stride', '--roi_landmarks', '--roi_padding',
                 '--data_set', '--raw_root', '--strict_au_gate'):
        assert flag in src, f'{flag} missing from run_au_probe.py'


# =========================================================================== #
# Feature: Anatomical AU Gating
# =========================================================================== #
def test_classify_au_set_splits_the_tiers():
    rep = classify_au_set([1, 2, 4, 5, 7, 10, 12, 6, 13])
    assert rep['forbidden'] == [1, 2, 4, 5, 7]
    assert rep['borderline'] == [6, 13]
    assert rep['core'] == [10, 12]
    assert set(TIER_A_AU_LIST) == {10, 12, 14, 15, 17, 23, 24}
    assert set(ROI_FORBIDDEN_AUS) == {1, 2, 4, 5, 7}
    assert set(ROI_BORDERLINE_AUS) == {6, 13}


def test_au_gate_warns_on_forbidden_and_aborts_when_strict():
    clean = check_au_gate(list(TIER_A_AU_LIST))
    assert clean == ''                                       # Tier A is clean

    warn = check_au_gate([1, 2, 10])
    assert warn and '1' in warn and 'OUTSIDE' in warn        # strong warning
    with pytest.raises(ValueError):
        check_au_gate([1, 2, 10], strict=True)               # or a hard error


# =========================================================================== #
# Feature: Centre-frame AU labelling (protocol §4)
# =========================================================================== #
def test_load_au_table_maps_frames_and_drops_unknowns(tmp_path):
    csv = tmp_path / 'F001_T1.csv'
    csv.write_text('frame,1,10,12,14\n'
                   '1,0,1,0,1\n'
                   '2,0,0,0,0\n'
                   '3,1,9,0,0\n'          # AU10 == 9 (unknown) -> row dropped
                   '4,0,1,1,0\n')
    tab = load_au_table(str(csv), [10, 12])
    assert tab['frame_to_label'][0].tolist() == [1.0, 0.0]   # csv frame 1 -> 0
    assert 2 not in tab['frame_to_label']                    # the '9' row
    assert tab['frame_to_label'][3].tolist() == [1.0, 1.0]
    assert tab['max_frame_1based'] == 4


def test_load_au_table_errors_on_missing_target(tmp_path):
    csv = tmp_path / 'F001_T1.csv'
    csv.write_text('frame,1,2\n1,0,0\n')
    with pytest.raises(ValueError):
        load_au_table(str(csv), [99])


def test_resolve_labels_keeps_only_centre_frame_labelled_clips():
    entries = [
        {'session': 'F001_T1', 'frame_start': 0},      # centre 100 -> labelled
        {'session': 'F001_T1', 'frame_start': 200},    # centre 300 -> missing
        {'session': 'F002_T1', 'frame_start': 0},      # no table -> dropped
    ]
    tables = {
        'F001_T1': {'frame_to_label': {100: np.array([1, 0], np.float32)},
                    'max_frame_1based': 400},
        'F002_T1': None,
    }
    kept, labels = resolve_labels(entries, 200, tables)
    assert len(kept) == 1
    assert kept[0]['frame_start'] == 0
    assert labels[0].tolist() == [1.0, 0.0]


# =========================================================================== #
# Feature: Trivial Baseline Reporting
# =========================================================================== #
def test_trivial_f1_matches_the_closed_form():
    assert trivial_f1([0.6])[0] == pytest.approx(2 * 0.6 / 1.6)   # ~0.75
    assert trivial_f1([0.0])[0] == 0.0
    assert trivial_f1([1.0])[0] == pytest.approx(1.0)


class _ConstantLogits(nn.Module):
    """A probe stub that returns fixed logits for any input."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, x):                                       # noqa: D401
        return self.logits


def test_evaluate_au_reports_probe_trivial_and_delta():
    # N=4, A=2; both AUs have presence 0.5 -> trivial F1 = 0.667 each.
    targets = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=torch.float32)
    probs = targets.clone()                                     # perfect predictor
    logits = torch.where(probs > 0.5, 8.0, -8.0)                # sigmoid -> {~1,~0}
    model = _ConstantLogits(logits)
    batches = [(torch.zeros(4, 1), targets)]
    stats = evaluate_au(batches, model, torch.device('cpu'))

    assert stats['f1'] == pytest.approx(100.0, abs=0.05)
    assert stats['trivial_f1'] == pytest.approx(200.0 / 3.0, abs=0.1)
    assert stats['f1_minus_trivial'] == pytest.approx(
        stats['f1'] - stats['trivial_f1'], abs=1e-6)
    assert stats['n'] == 4


def test_tune_thresholds_recovers_a_hard_low_confidence_case():
    # all positives with probs < 0.5: @0.5 F1 = 0, but a lower threshold is perfect
    probs = np.array([[0.4], [0.35], [0.3]])
    targets = np.array([[1], [1], [1]])
    thr, best = tune_per_au_thresholds(probs, targets)
    assert best[0] == pytest.approx(1.0)
    assert thr[0] <= 0.4


def test_evaluate_au_oracle_upper_bound_is_opt_in():
    targets = torch.tensor([[1, 1], [0, 1]], dtype=torch.float32)
    logits = torch.tensor([[0.2, 0.9], [-0.1, 0.8]])           # AU0 below 0.5
    model = _ConstantLogits(logits)
    batches = [(torch.zeros(2, 1), targets)]
    base = evaluate_au(batches, model, torch.device('cpu'))
    tuned = evaluate_au(batches, model, torch.device('cpu'), tune_thresholds=True)
    assert 'oracle_f1' not in base
    assert 'oracle_f1' in tuned
    assert tuned['oracle_f1'] >= base['f1']                     # an upper bound


# =========================================================================== #
# Feature: Distributed Evaluation (DDP) Safety
# =========================================================================== #
def test_gather_is_a_noop_when_not_distributed():
    probs = torch.rand(5, 3)
    tgt = torch.randint(0, 2, (5, 3)).float()
    p2, t2 = _gather_predictions(probs, tgt)
    assert torch.equal(p2, probs) and torch.equal(t2, tgt)


def test_train_engine_unwraps_ddp_for_set_train():
    from engines.au_probe import _unwrap

    probe = nn.Linear(2, 2)
    assert _unwrap(probe) is probe

    class _Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    assert _unwrap(_Wrapper(probe)) is probe


# =========================================================================== #
# Feature: shipped config is wired to the protocol
# =========================================================================== #
def test_au_tir_roi_config_resolves_the_protocol(monkeypatch):
    cfg = os.path.join(CODE_DIR, 'configs', 'finetune', 'au_tir_roi.yaml')
    monkeypatch.setattr(sys, 'argv', ['prog', '-c', cfg])
    args = importlib.import_module('runners.run_au_probe').get_args()
    assert args.data_set == 'tir_roi'
    assert args.streams == 'tir'
    assert args.temporal_stride == 2
    assert args.input_size == 64
    assert args.probe == 'linear'
    # the Tier-A headline set (NOT the default BP4D 12-AU subset)
    assert args.au_list.replace(' ', '') == '10,12,14,15,17,23,24'


def test_au_tir_roi_hpc_config_resolves_the_protocol(monkeypatch):
    cfg = os.path.join(CODE_DIR, 'configs', 'finetune', 'au_tir_roi_hpc.yaml')
    monkeypatch.setattr(sys, 'argv', ['prog', '-c', cfg])
    args = importlib.import_module('runners.run_au_probe').get_args()
    assert args.data_set == 'tir_roi'
    assert args.streams == 'tir'
    assert args.input_size == 112          # the HPC lineage
    assert args.temporal_stride == 2


# =========================================================================== #
# Feature: builder arity contract (regression: job 55427926, 2026-10-14)
# The runner unpacks 3 values; build_au_tir_roi_datasets used to return 2 when
# no test split was requested -> "not enough values to unpack (3, got 2)".
# =========================================================================== #
class _FakeDataset:
    def __init__(self, subjects=None, **kw):
        self.subjects = sorted(subjects or [])

    def __len__(self):
        return len(self.subjects)


def _builder_args(**over):
    base = dict(au_root='ignored', au_list='10,12', raw_root='ignored',
                input_size=64, clip_duration=8.0, fps=25.0, temporal_stride=2,
                tubelet='2,16,16', roi_padding=0.2, max_sessions=None,
                max_clips=None, max_entries=None, roi_cache='',
                roi_cache_lru=2, train_ratio=0.5, train_subjects='',
                val_subjects='', test_subjects='')
    base.update(over)
    return argparse.Namespace(**base)


def _patch_builder(monkeypatch):
    import data.au_tir_roi_dataset as m
    sessions = [{'session': f'{s}_T1', 'subject': s, 'task': 'T1', 'path': 'x'}
                for s in ('F001', 'F002', 'F003')]
    monkeypatch.setattr(m, 'discover_au_sessions', lambda root: sessions)
    monkeypatch.setattr(m, 'AuTirRoiDataset', _FakeDataset)
    return m


def test_builder_always_returns_a_three_tuple(monkeypatch):
    m = _patch_builder(monkeypatch)
    train, val, test = m.build_au_tir_roi_datasets(_builder_args())
    assert test is None                     # no --test_subjects
    assert train.subjects == ['F001', 'F002'] and val.subjects == ['F003']


def test_builder_returns_a_test_dataset_when_requested(monkeypatch):
    m = _patch_builder(monkeypatch)
    train, val, test = m.build_au_tir_roi_datasets(
        _builder_args(train_subjects='F001', val_subjects='F002',
                      test_subjects='F003'), with_test=True)
    assert train.subjects == ['F001'] and val.subjects == ['F002']
    assert test is not None and test.subjects == ['F003']


# =========================================================================== #
# Feature: bare-tensor input maps to the configured stream (regression: job
# 55450025, 2026-10-15 -> 'stream "tir" missing from input; got keys ["rgb"]')
# =========================================================================== #
def test_probe_maps_a_bare_tensor_to_the_single_stream():
    m = build_au_probe_model(_probe_args(temporal_stride=2), num_classes=7)
    assert m.streams == ['tir']
    out = m(torch.zeros(2, 3, 100, 64, 64))          # bare tensor
    assert out.shape == (2, 7)


def test_probe_maps_a_bare_tensor_to_rgb_when_rgb_streamed():
    m = build_au_probe_model(_probe_args(streams='rgb'), num_classes=3)
    assert m(torch.zeros(2, 3, 100, 64, 64)).shape == (2, 3)
