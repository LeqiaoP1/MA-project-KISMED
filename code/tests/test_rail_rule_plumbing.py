"""The rail-touch cleaning rule must reach EVERY pipeline stage.

``rail_touch_v`` is opt-in (``0.0`` = off) and is threaded from the YAML/CLI
through ``args`` into the dataset. That makes two failure modes possible, and
both are silent:

* an entry point that can build the TIR-ROI dataset but does not expose the flag
  (so a config either cannot set it, or sets it and is rejected);
* a builder that forgets to forward ``args.rail_touch_v``.

These tests pin the contract: the dataset DISPATCH layer (``data/datasets.py``,
i.e. what ``run_pretrain.py`` and ``run_waveform.py``/``run_finetune.py`` really
call) hands the value to the dataset in both the Stage-2 and the Stage-3 view,
the effective policy is LOGGED, and every entry point exposes the flag.

The synthetic raw tree is the same minimal one ``test_tir_resp_rail_filter.py``
uses: one session, one IR track, one ``Resp_Volts.txt``, and a stubbed video
container probe (no frame is ever decoded).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pytest
import yaml

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import tir_resp_dataset as trd
from data import video_io as vio
from data.datasets import build_dataset, build_pretraining_dataset

FPS = 25.0
RESP_FS = 1000.0
N_FRAMES = 400                             # 16 s
CLIP_S = 8.0
STRIDE_S = 1.0                             # the shipped hop (synthetic tests)
#: per-config window-hop OVERRIDES (s): the HPC runs deliberately sample
#: transients at 2 s (4x redundant decode instead of 8x, ~halves the epoch).
#: Stage 3 must MATCH its Stage-2 encoder's hop (the config says so), and the
#: rest of the TIR-ROI study stays at the 1 s hop. The `_smoothl1` arm is a
#: clone of resp_tir_roi_hpc.yaml, so it inherits the same 2 s hop.
EXPECTED_STRIDE_S = {'stage2_hpc_tir_roi_resp.yaml': 2.0,
                     'resp_tir_roi_hpc.yaml': 2.0,
                     'resp_tir_roi_hpc_smoothl1.yaml': 2.0}
N_RESP = int(N_FRAMES / FPS * RESP_FS)     # 16000 samples
RAIL = trd.RESP_RAIL_V

#: the 7 experiment configs that drive the TIR-ROI/RESP study
TIR_CONFIGS = sorted(glob.glob(os.path.join(CODE_DIR, 'configs', 'pretrain',
                                            'stage2_*tir_roi*.yaml'))
                     + glob.glob(os.path.join(CODE_DIR, 'configs', 'finetune',
                                              'resp_tir_roi_*.yaml')))
#: entry points that can build the TIR-ROI/RESP dataset
RAIL_ENTRY_POINTS = ('run_pretrain.py', 'run_finetune.py', 'run_waveform.py',
                     'run_inspect_tir_resp.py', 'run_survey_tir_resp.py')


class _FakeReader:
    """Container-probe stand-in (``num_frames`` / ``fps`` only)."""

    def __init__(self, num_frames, fps):
        self.num_frames = num_frames
        self.fps = fps

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _stub_video(monkeypatch):
    monkeypatch.setattr(vio, 'open_video',
                        lambda path: _FakeReader(N_FRAMES, FPS))
    yield


def _write_raw(root, resp: np.ndarray) -> str:
    (root / 'Thermal' / 'S001').mkdir(parents=True)
    (root / 'Thermal' / 'S001' / 'T1.wmv').write_bytes(b'\x00')
    (root / 'IRFeatures').mkdir()
    line = ' '.join(['100.0', '100.0'] * 28)          # 28 landmarks, no sentinel
    (root / 'IRFeatures' / 'S001_T1.txt').write_text(
        '\n'.join([line] * N_FRAMES) + '\n')
    phys = root / 'Physiology' / 'S001' / 'T1'
    phys.mkdir(parents=True)
    (phys / 'Resp_Volts.txt').write_text(
        '\n'.join(f'{v:.4f}' for v in resp) + '\n')
    return str(root)


def _resp_with_rail_in_the_first_second() -> np.ndarray:
    """Healthy breathing plus a 1 s rail stretch at t=0.

    With an 8 s window and a 1 s hop the windows start at 0..8 s, so the stretch
    is inside the FIRST window only -> exactly 1 of the 9 clips is dropped, and
    the count itself proves the hop is 1 s (a 4 s hop would give 3 windows).
    """
    y = np.full(N_RESP, -4.0)
    y += 3.0 * np.sin(2 * np.pi * 0.3 * np.arange(N_RESP) / RESP_FS)
    y[:int(1.0 * RESP_FS)] = RAIL
    return y


def _args(root: str, **over) -> argparse.Namespace:
    """An ``args`` namespace as the runners build it (only the keys we touch)."""
    kw = dict(raw_root=root, data_path=root, data_set='tir_roi',
              subjects=['S001'], tasks=None, task_set=None, task_groups=None,
              clip_duration=CLIP_S, clip_stride=STRIDE_S, temporal_stride=1,
              tubelet='2,16,16', input_size=64, roi_padding=0.2,
              roi_landmarks='', fps=FPS, resp_fs=RESP_FS,
              fs=100.0, physio_norm='zscore', target='resp',
              train_ratio=1.0, split_by='session', val_subject='',
              min_signal_spread=0.0, rail_touch_v=9.90,
              max_clips=None, max_entries=None, verbose=False)
    kw.update(over)
    return argparse.Namespace(**kw)


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def test_stage2_dispatch_threads_the_rail_rule(tmp_path, capsys):
    """``build_pretraining_dataset`` (run_pretrain.py) applies the rule."""
    root = _write_raw(tmp_path, _resp_with_rail_in_the_first_second())
    ds = build_pretraining_dataset(_args(root, pretrain_split='all'))
    assert ds.stats['rail_touch_v'] == pytest.approx(9.90)
    assert len(ds) == 8                       # 9 windows, the railed one dropped
    assert ds.stats['clips_dropped_rail'] == 1
    assert ds.base.rail_touch_v == pytest.approx(9.90)
    out = capsys.readouterr().out
    assert 'stage2 corpus cleaning' in out and 'TOUCHED' in out


def test_stage2_dispatch_leaves_the_corpus_alone_when_off(tmp_path):
    root = _write_raw(tmp_path, _resp_with_rail_in_the_first_second())
    ds = build_pretraining_dataset(_args(root, rail_touch_v=0.0))
    assert len(ds) == 9
    assert ds.stats['clips_dropped_rail'] == 0


# --------------------------------------------------------------------------- #
# Stage 3
# --------------------------------------------------------------------------- #
def test_stage3_dispatch_threads_the_rail_rule(tmp_path, capsys):
    """``build_dataset`` (run_waveform.py / run_finetune.py) applies the rule."""
    root = _write_raw(tmp_path, _resp_with_rail_in_the_first_second())
    args = _args(root, data_set='tir_roi_resp', is_train=True, test_mode=False)
    ds = build_dataset(is_train=True, test_mode=False, args=args)
    assert ds.stats['rail_touch_v'] == pytest.approx(9.90)
    assert len(ds) == 8
    assert ds.stats['clips_dropped_rail'] == 1
    out = capsys.readouterr().out
    assert 'stage3 corpus cleaning' in out and 'TOUCHED' in out


def test_an_absent_key_is_off_AND_said_out_loud(tmp_path, capsys):
    """A builder call without the key must not silently filter -- and must not
    silently do NOTHING either: the effective policy is printed either way."""
    root = _write_raw(tmp_path, _resp_with_rail_in_the_first_second())
    args = _args(root)
    del args.rail_touch_v                       # the silent-off failure mode
    ds = build_pretraining_dataset(args)
    assert len(ds) == 9                         # unfiltered, as documented
    out = capsys.readouterr().out
    assert 'rail_touch_v=0 V (OFF' in out


# --------------------------------------------------------------------------- #
# the flags and the shipped values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('runner', RAIL_ENTRY_POINTS)
def test_every_entry_point_exposes_the_flag(runner):
    """A stage that can build the dataset MUST be able to set the knob."""
    with open(os.path.join(CODE_DIR, 'runners', runner)) as fh:
        src = fh.read()
    assert "'--rail_touch_v'" in src, f'{runner} cannot set --rail_touch_v'


def test_the_tir_configs_use_the_strict_rule_and_a_consistent_hop():
    """Every TIR-ROI/RESP config: the touch rule ON, an 8 s window, and the
    DOCUMENTED window hop (the HPC Stage-2/Stage-3 runs use 2 s, the rest 1 s).

    The count is the study's config list: the 7 originals + the Stage-3
    `_smoothl1` L_time A/B arm (a clone of resp_tir_roi_hpc.yaml).
    """
    assert len(TIR_CONFIGS) == 8
    for path in TIR_CONFIGS:
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        name = os.path.basename(path)
        rel = os.path.relpath(path, CODE_DIR)
        assert cfg['rail_touch_v'] == pytest.approx(9.90), rel
        assert cfg['min_signal_spread'] == pytest.approx(0.1), rel
        assert cfg['clip_stride'] == pytest.approx(
            EXPECTED_STRIDE_S.get(name, STRIDE_S)), rel
        assert cfg['clip_duration'] == pytest.approx(CLIP_S), rel


def test_the_dataset_builders_forward_the_rule():
    """The builders must read the key off ``args`` (getattr default 0.0)."""
    for fn in (trd.build_tir_roi_pretrain_dataset,
               trd.build_tir_roi_finetune_dataset):
        src = __import__('inspect').getsource(fn)
        assert "getattr(args, 'rail_touch_v'" in src, fn.__name__
        assert '_log_corpus_cleaning' in src, fn.__name__
