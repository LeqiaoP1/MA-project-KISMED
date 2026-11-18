"""Regression guard: the EDA modality is removed from the project.

EDA was dropped as a physiological stream/target on 2026-11-18. These cases
pin the removal at every public entry point so it cannot silently creep back in:

* the stream registries no longer contain ``'eda'``;
* ``MultiModalMAE`` and ``PairedPretrainDataset`` reject ``streams=('rgb','eda')``;
* the runners reject ``--target eda``;
* the raw-physiology inspector rejects ``--channel EDA`` and no longer exposes
  the ``eda`` channel.

The Stage-2 config side (``mask_ratio_eda``) is covered in
``test_config_loading.py::test_stage2_carries_no_eda_keys``.
"""
import importlib
import sys

import pytest

from core import multimae as mm
from data.paired_dataset import PairedPretrainDataset
from runners import run_inspect_physio


def test_stream_registries_have_no_eda():
    assert mm._SIGNAL_STREAMS == ('bp', 'resp')
    assert 'eda' not in mm.STREAM_CHANNELS
    assert set(mm.MASK_SPAN_DEFAULTS) == {'bp', 'resp'}


def test_multimodal_mae_rejects_eda_stream():
    with pytest.raises(ValueError, match=r'unknown stream\(s\).*eda'):
        mm.MultiModalMAE(streams=('rgb', 'eda'))


def test_paired_pretrain_rejects_eda_stream():
    with pytest.raises(ValueError, match=r'unknown stream\(s\).*eda'):
        PairedPretrainDataset(data_path='/nonexistent', streams=('rgb', 'eda'))


@pytest.mark.parametrize('runner', ['run_waveform', 'run_inspect_data'])
def test_runners_reject_eda_target(runner, monkeypatch):
    module = importlib.import_module(f'runners.{runner}')
    monkeypatch.setattr(sys, 'argv', ['prog', '--target', 'eda'])
    with pytest.raises(SystemExit):
        module.get_args()


def test_inspector_rejects_eda_channel():
    with pytest.raises(SystemExit):
        run_inspect_physio._parse_channels('EDA')
    assert 'eda' not in run_inspect_physio.CHANNEL_META
    assert 'eda' not in run_inspect_physio.CHANNEL_ALIASES
