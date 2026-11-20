"""Phase 4 gates — the window contract every consumer shares (ADR-003).

Gate map: G9 (bounds before slicing), G10 (bounded clips, loud mismatch),
G14 (window arithmetic round-trip), I5/I6/I8.
"""
import numpy as np
import pytest

from data import resp_cache as respc
from data import resp_window as rw

FS_OUT = 100.0


def shard(mask_pattern):
    """A shard-like mapping with a signal that is NaN exactly where mask is False."""
    mask = np.asarray(mask_pattern, dtype=bool)
    signal = np.where(mask, 1.0, np.nan).astype(np.float32)
    return {'signal': signal, 'mask': mask,
            'cause': np.where(mask, 0, 2).astype(np.uint8)}


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #
def test_window_indices_round_trip():
    s, n = rw.window_indices(1.234, 8.0, FS_OUT)
    assert (s, n) == (123, 800)
    assert rw.window_indices(0.0, 1.0, FS_OUT) == (0, 100)
    with pytest.raises(ValueError, match='clip_seconds'):
        rw.window_indices(0.0, 0.0, FS_OUT)
    with pytest.raises(ValueError, match='t_start'):
        rw.window_indices(-0.1, 8.0, FS_OUT)
    with pytest.raises(ValueError, match='finite'):
        rw.window_indices(float('nan'), 8.0, FS_OUT)


def test_window_indices_does_not_assert_the_fit():
    """A generating loop must be able to probe past the end and reject cleanly."""
    s, n = rw.window_indices(10_000.0, 8.0, FS_OUT)     # far outside any session
    assert (s, n) == (1_000_000, 800)


# --------------------------------------------------------------------------- #
# usability — bounds FIRST, slice second
# --------------------------------------------------------------------------- #
def test_usable_requires_every_sample():
    mask = np.ones(1000, dtype=bool)
    assert rw.usable(mask, 0, 800)
    mask[500] = False
    assert not rw.usable(mask, 0, 800)
    assert rw.usable(mask, 501, 400)


def test_usable_rejects_a_truncated_all_true_slice():
    """``mask[s:s+n]`` truncates silently; a short window is never usable."""
    mask = np.ones(100, dtype=bool)
    assert not rw.usable(mask, 50, 800), (
        'a window running past the end must be unusable even though every '
        'remaining sample is True')
    assert not rw.usable(mask, 50, 50) is None
    assert rw.usable(mask, 50, 50)


def test_usable_rejects_degenerate_windows():
    mask = np.ones(100, dtype=bool)
    assert not rw.usable(mask, 0, 0)
    assert not rw.usable(mask, -1, 10)
    with pytest.raises(ValueError):
        rw.usable(np.ones((2, 2), dtype=bool), 0, 1)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def test_read_window_returns_the_slice():
    sh = shard(np.ones(1000, dtype=bool))
    seg = rw.read_window(sh, 10, 20)
    assert seg.size == 20 and np.all(seg == 1.0)


def test_read_window_raises_on_a_hole_instead_of_returning_zeros():
    mask = np.ones(1000, dtype=bool)
    mask[100] = False
    sh = shard(mask)
    assert np.isnan(sh['signal'][100]), 'the shard must store NaN, not 0'
    with pytest.raises(ValueError, match='not usable'):
        rw.read_window(sh, 0, 200)


def test_read_window_raises_out_of_bounds():
    sh = shard(np.ones(1000, dtype=bool))
    with pytest.raises(ValueError):
        rw.read_window(sh['signal'], 950, 100)


def test_window_cause_reports_the_most_severe_nonzero_code():
    sh = shard(np.ones(1000, dtype=bool))
    sh['cause'][40] = 2          # flatline
    sh['cause'][41] = 1          # rail outranks flatline in the ADR precedence
    assert rw.window_cause(sh['cause'], 0, 100) == 1
    assert rw.window_cause(sh['cause'], 200, 100) == 0, (
        'a clean window reports no defect, not the first sample')
    sh['cause'][300] = 4         # non-finite outranks everything
    assert rw.window_cause(sh['cause'], 250, 100) == 4


# --------------------------------------------------------------------------- #
# G10 — clip length is bounded by the minimum run
# --------------------------------------------------------------------------- #
def test_assert_clip_fits_accepts_the_bound_itself():
    params = {'min_run_s': 8.0}
    rw.assert_clip_fits(8.0, params)                  # exactly at the bound
    rw.assert_clip_fits(7.9, params)
    with pytest.raises(ValueError, match='min_run_s'):
        rw.assert_clip_fits(8.5, params)


def test_assert_clip_fits_names_the_rebuild_command():
    with pytest.raises(ValueError) as exc:
        rw.assert_clip_fits(10.0, {'min_run_s': 8.0})
    assert '--min_run_s 10' in str(exc.value), str(exc.value)


# --------------------------------------------------------------------------- #
# I6 — identity verification is the only way in
# --------------------------------------------------------------------------- #
def _corpus(n=2):
    return [{'session': f'F{i:03d}_T1'} for i in range(1, n + 1)]


def test_verify_identity_accepts_a_keyed_directory(tmp_path):
    params = respc.build_params('raw', _corpus())
    root = respc.cache_root(tmp_path, respc.cache_key(params))
    respc.write_params(root, params)
    assert rw.verify_identity(root, params) == root


def test_verify_identity_searches_an_out_root(tmp_path):
    params = respc.build_params('raw', _corpus())
    root = respc.cache_root(tmp_path, respc.cache_key(params))
    respc.write_params(root, params)
    assert rw.verify_identity(tmp_path, params) == root


def test_verify_identity_refuses_a_mismatch(tmp_path):
    params = respc.build_params('raw', _corpus())
    other = respc.build_params('raw', _corpus(), fs_out=50)
    root = respc.cache_root(tmp_path, respc.cache_key(other))
    respc.write_params(root, other)
    with pytest.raises((ValueError, FileNotFoundError)):
        rw.verify_identity(root, params)
