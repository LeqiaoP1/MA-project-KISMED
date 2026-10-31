"""The respiration cache as the EVALUATION reference (design C8).

The legacy evaluator re-read ``Resp_Volts.txt`` at ``--raw_fs`` and resampled it
with ``np.interp`` -- a different preprocessing than the training label. These
tests pin the artefact route: one shard supplies BOTH sides, its per-sample mask
restricts scoring, and the resolution works without knowing the training corpus.
"""
import json
import os
import sys

import numpy as np
import pytest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import resp_cache as respc
from data import tir_resp_dataset as trd
from runners import run_build_resp_cache as rb
from runners import run_evaluate_kfold as rk
from runners.run_evaluate_session import read_reference_from_cache

FS = 1000.0
OUT = 100.0
N_RESP = 24000                                   # 24 s at 1 kHz
N_OUT = 2400


def _breathing(seconds: float = 24.0) -> np.ndarray:
    t = np.arange(int(round(seconds * FS))) / FS
    return 3.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)


def _write_raw(root, traces: dict) -> str:
    for name, x in traces.items():
        subj, task = name.split('_')
        (root / 'Thermal' / subj).mkdir(parents=True, exist_ok=True)
        (root / 'Thermal' / subj / f'{task}.wmv').write_bytes(b'\x00')
        phys = root / 'Physiology' / subj / task
        phys.mkdir(parents=True, exist_ok=True)
        (phys / 'Resp_Volts.txt').write_text(
            '\n'.join(f'{v:.5f}' for v in x) + '\n')
    return str(root)


def _build(raw, out, traces):
    _write_raw(raw, traces)
    assert rb.main(['--init', '--build', '--out_root', str(out),
                    '--raw_root', str(raw)]) == 0
    sessions = trd.discover_sessions(str(raw))
    params = rb.respc.build_params(str(raw), sessions)
    return respc.cache_root(out, respc.cache_key(params))


def _dump(pred_dir, session, shard, offsets_s=(0.0, 8.0), gain=1.0):
    """A tiny prediction dump whose rows are slices of the shard's signal."""
    rows = []
    for t0 in offsets_s:
        s = int(round(t0 * OUT))
        rows.append(np.asarray(shard['signal'][s:s + 800], dtype=np.float64))
    tgts = np.stack(rows)
    preds = tgts * gain
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / 'preds.npy', preds)
    np.save(pred_dir / 'targets.npy', tgts)
    entries = [{'session': session, 't_start': float(t0),
                # deliberately NOT a real file: only the cache may be used
                'signals_file': '/nonexistent/Resp_Volts.txt',
                'task': 'T1', 'task_level': ['low']}
               for t0 in offsets_s]
    with open(pred_dir / 'entries.json', 'w') as fh:
        json.dump(entries, fh)
    return pred_dir


# --------------------------------------------------------------------------- #
# the helper
# --------------------------------------------------------------------------- #
def test_read_reference_from_cache_returns_the_grid_and_mask(tmp_path):
    root = _build(tmp_path / 'raw', tmp_path / 'out',
                  {'S001_T1': _breathing()})
    t, y, mask, resolved = read_reference_from_cache(tmp_path / 'out',
                                                    'S001_T1', fs=OUT)
    assert resolved == str(root)
    assert y.size == N_OUT and t.size == N_OUT
    assert np.allclose(t, np.arange(N_OUT) / OUT)
    assert np.array_equal(np.isnan(y), ~mask)
    assert mask.any()
    # the same artefact the labels came from
    sh = respc.read_shard(respc.shard_path(root, 'S001_T1'))
    assert np.array_equal(y, np.asarray(sh['signal'], dtype=float), equal_nan=True)


def test_read_reference_pins_an_explicit_key(tmp_path):
    root = _build(tmp_path / 'raw', tmp_path / 'out',
                  {'S001_T1': _breathing()})
    got = read_reference_from_cache(tmp_path / 'out', 'S001_T1', fs=OUT,
                                    cache_key=root.name)
    assert got[3] == str(root)
    with pytest.raises(FileNotFoundError):
        read_reference_from_cache(tmp_path / 'out', 'S001_T1', fs=OUT,
                                  cache_key='not_a_cache')


def test_unknown_session_is_refused(tmp_path):
    root = _build(tmp_path / 'raw', tmp_path / 'out',
                  {'S001_T1': _breathing()})
    with pytest.raises(FileNotFoundError):
        read_reference_from_cache(tmp_path / 'out', 'S999_T9', fs=OUT)


def test_eval_lookup_ignores_the_training_corpus(tmp_path):
    """An evaluator cannot know the corpus, so it matches the GRID contract."""
    root = _build(tmp_path / 'raw', tmp_path / 'out',
                  {'S001_T1': _breathing()})
    want = respc.eval_params(fs_out=OUT)
    assert respc.find_cache(tmp_path / 'out', want,
                            keys=respc.EVAL_COMPARE_KEYS,
                            require_session='S001_T1') == root
    # a grid mismatch is still refused
    with pytest.raises(ValueError):
        respc.find_cache(tmp_path / 'out', respc.eval_params(fs_out=50.0),
                         keys=respc.EVAL_COMPARE_KEYS)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_score_fold_uses_the_cache_and_its_mask(tmp_path):
    root = _build(tmp_path / 'raw', tmp_path / 'out',
                  {'S001_T1': _breathing()})
    sh = respc.read_shard(respc.shard_path(root, 'S001_T1'))
    # punch a hole into the artefact so the mask restriction is observable
    sig = np.array(sh['signal'], copy=True)
    mask = np.array(sh['mask'], copy=True)
    sig[1000:1100] = np.nan
    mask[1000:1100] = False
    respc.write_shard(root, 'S001_T1', dict(sh, signal=sig, mask=mask,
                                            valid_frac=float(mask.mean())))

    dump = _dump(tmp_path / 'fold0', 'S001_T1', sh, gain=1.0)
    recs, _ = rk.score_fold(str(dump), OUT, FS, (0.1, 0.6), 'resp',
                            0.0, 0.10, 0, resp_cache=str(tmp_path / 'out'))
    assert len(recs) == 1
    rec = recs[0]
    assert rec['reference_source'] == 'resp_cache'
    assert rec['pearson'] == pytest.approx(1.0, abs=1e-6)
    assert rec['pearson_band'] == pytest.approx(1.0, abs=1e-3)
    assert rec['reference_coverage'] < 1.0
    # the hole is NOT scored: the two 8 s windows cover 1600 output samples and
    # the 100-sample hole sits inside that span, so coverage is 1500/1600
    assert rec['coverage'] == pytest.approx((1600 - 100) / 1600, abs=1e-6)


def test_score_fold_legacy_path_is_unchanged_without_the_flag(tmp_path):
    root = _build(tmp_path / 'raw', tmp_path / 'out',
                  {'S001_T1': _breathing()})
    sh = respc.read_shard(respc.shard_path(root, 'S001_T1'))
    dump = _dump(tmp_path / 'fold0', 'S001_T1', sh)
    recs, _ = rk.score_fold(str(dump), OUT, FS, (0.1, 0.6), 'resp', 0.0, 0.10, 0)
    assert recs == []          # the recorded signals_file does not exist
