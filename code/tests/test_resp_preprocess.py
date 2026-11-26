"""Phase 2 gates — per-run offset, resampling and the emitted grid (ADR-003).

Measured quantities this file pins (recorded in ``code/notes/eval/archive/tir_resp/``):

* 130 Hz interior aliasing residual: RMS 1.3e-4 V, max 2.9e-4 V for a 1 V tone
  (the ADR's 2.4e-4 V is used here as the RMS bound).
* Run-edge second difference with edge replication: 4.7e-4 V at the start and
  7.9e-5 V at the end, against 2.3e-2 / 1.2e-1 V for zero padding — the
  operational meaning of the ADR's "run-edge gradient below 1e-3 V".
* Interior error against an analytic 0.5 Hz tone: 3.5e-5 V; a slice-local
  (phase-unaware) resample of the same run is off by 2.2e-2 V.

Gate map: G4 (no transient), G4b (zero phase error), G9 (holes widened, runs not
shrunk), G1b (mask <=> cause), G13 (anti-alias is real).
"""
import numpy as np
import pytest
from scipy.signal import resample_poly

from data import resp_preprocess as rpp
from data import resp_window as rw

FS = 1000.0
FS_OUT = 100.0
DOWN = 10


def tone(n_samples: int, freq: float, amp: float = 1.0, phase0: int = 0):
    """A tone evaluated at ABSOLUTE sample indices ``phase0 .. phase0+n-1``."""
    idx = np.arange(phase0, phase0 + n_samples)
    return amp * np.sin(2 * np.pi * freq * idx / FS)


def single_run(signal: np.ndarray, start: int, n_total: int) -> np.ndarray:
    x = np.full(n_total, np.nan)
    x[start:start + signal.size] = signal
    return x


# --------------------------------------------------------------------------- #
# per-run offset
# --------------------------------------------------------------------------- #
def test_run_mean_is_removed_to_float64_precision():
    x = single_run(tone(20000, 0.3) + 4.5, 0, 20000)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    assert abs(float(out['signal'][out['mask']].mean())) < 1e-6, (
        'the per-run DC offset must be gone; nothing is estimated across a break')


def test_amplitude_and_shape_survive():
    """18 whole periods of a 1 V sine: std must stay 1/sqrt(2) within 1 %."""
    n = 60000
    x = single_run(tone(n, 0.3) + 4.5, 0, n)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    y = out['signal'][out['mask']]
    assert abs(float(np.std(y)) / (1.0 / np.sqrt(2.0)) - 1.0) < 0.01, (
        'no high-pass, no detrend, no scaling: amplitude must survive')


# --------------------------------------------------------------------------- #
# G4 — no run-edge transient (and the zero-padded alternative DOES show one)
# --------------------------------------------------------------------------- #
def _edge_curvature(y: np.ndarray) -> tuple:
    """``(|d2| at start, |d2| at end)`` — a transient is a curvature spike."""
    d2 = np.abs(np.diff(y, n=2))
    return float(d2[:4].max()), float(d2[-4:].max())


def test_edge_replication_leaves_no_transient():
    start, n = 1503, 9000
    x = single_run(4.5 + tone(n, 0.1, phase0=start), start, 24000)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT, min_run_s=8.0)
    y = out['signal'][out['mask']]
    d2_start, d2_end = _edge_curvature(y)
    assert d2_start < 1e-3 and d2_end < 1e-3, (
        f'run-edge gradient must stay below 1e-3 V (ADR Criterion 4); got '
        f'{d2_start:.2e} / {d2_end:.2e} V')
    interior = np.abs(np.diff(y, n=2))[10:-10].max()
    assert d2_start < 10 * max(interior, 1e-9), (
        'the edge must not be an order of magnitude rougher than the interior')


def test_zero_padding_would_show_a_transient():
    """The gate is falsifiable: the naive alternative violates the same bound."""
    start, n = 1503, 9000
    sig = 4.5 + tone(n, 0.1, phase0=start)
    zero_padded = resample_poly(sig - sig.mean(), 1, DOWN)
    d2_start, d2_end = _edge_curvature(zero_padded)
    assert max(d2_start, d2_end) > 1e-2, (
        'zero padding must be visibly worse, otherwise G4 tests nothing')


def test_run_starting_near_sample_zero_keeps_its_full_pad():
    """``a = 3`` is not special: the pad is built in memory, never read from file."""
    start, n = 3, 8000
    x = single_run(tone(n, 0.1, phase0=start), start, n + 4000)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT, min_run_s=8.0)
    idx = np.flatnonzero(out['mask'])
    assert idx.size == 800 and idx[0] == 1 and idx[-1] == 800
    d2_start, _ = _edge_curvature(out['signal'][idx])
    assert d2_start < 1e-3


# --------------------------------------------------------------------------- #
# G4b — no sub-sample phase error
# --------------------------------------------------------------------------- #
def test_off_grid_run_lands_on_exact_grid_instants():
    """0.5 Hz (4 whole periods in 8 s) so the removed mean is zero by construction."""
    n = 8000
    ref = None
    for start in (1503, 1504, 1507, 1510, 1523):
        x = single_run(tone(n, 0.5, phase0=start), start, 24000)
        out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT, min_run_s=8.0)
        idx = np.flatnonzero(out['mask'])
        assert idx.size == 800
        y = out['signal'][idx]
        # the analytic tone SAMPLED ON THE GRID: source index = grid index * DOWN
        analytic = np.sin(2 * np.pi * 0.5 * (idx.astype(np.float64) * DOWN) / FS)
        # interior only: the first/last outputs legitimately see the edge pad
        assert np.abs(y[12:-12] - analytic[12:-12]).max() < 1e-3, (
            f'start={start}: the run must land at lag 0, not a sub-sample shift')
        if ref is None:
            ref = (idx, y)
        else:
            common, ia, ib = np.intersect1d(ref[0], idx, return_indices=True)
            sel = (common > ref[0][0] + 12) & (common < ref[0][-1] - 12)
            assert np.abs(ref[1][ia[sel]] - y[ib[sel]]).max() < 1e-3, (
                'the same trace at a different start residue must give the same '
                'grid values')


def test_slice_local_resampling_would_be_measurably_shifted():
    """Falsifiability: resampling the slice from index 0 is orders worse."""
    start, n = 1503, 8000
    sig = tone(n, 0.5, phase0=start)
    naive = resample_poly(sig - sig.mean(), 1, DOWN)
    analytic = tone(n // DOWN, 0.5, phase0=1500)
    assert np.abs(naive[12:-12] - analytic[12:-12]).max() > 1e-2, (
        'the phase-aware path must be distinguishable from the phase-unaware one')


# --------------------------------------------------------------------------- #
# G13 — the anti-alias filter is real
# --------------------------------------------------------------------------- #
def test_130hz_folds_to_30hz_only_through_a_strong_filter():
    n = 30000
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * 0.3 * t) + np.sin(2 * np.pi * 130.0 * t)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    idx = np.flatnonzero(out['mask'])
    y = out['signal'][idx]
    tk = idx * DOWN / FS
    design = np.c_[np.sin(2 * np.pi * 0.3 * tk), np.cos(2 * np.pi * 0.3 * tk),
                   np.ones_like(tk)]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = (y - design @ coef)[12:-12]          # interior: no edge pad involved
    rms = float(np.sqrt((resid ** 2).mean()))
    assert rms <= 2.4e-4, (
        f'the 130 Hz tone must be attenuated, not folded in; interior RMS '
        f'{rms:.2e} V exceeds the ADR residual 2.4e-4 V')
    assert np.abs(resid).max() <= 5e-4


def test_in_band_content_is_not_attenuated():
    """30-50 Hz is legitimately IN band at fs_out = 100 Hz."""
    n = 30000
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 40.0 * t)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    y = out['signal'][out['mask']]
    spectrum = np.abs(np.fft.rfft(y - y.mean()))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / FS_OUT)
    peak40 = spectrum[np.argmin(np.abs(freqs - 40.0))]
    energy = np.sqrt((spectrum ** 2).sum())
    assert peak40 / energy > 0.05, '40 Hz must survive: it is below the 50 Hz edge'


# --------------------------------------------------------------------------- #
# G9 — a hole narrower than one output period must still be visible
# --------------------------------------------------------------------------- #
def test_subperiod_hole_is_widened_and_neighbours_keep_their_samples():
    n = 24000
    x = tone(n, 0.3) + 4.5
    x[12002:12007] = np.nan                   # 5 source samples, no grid instant
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT, min_run_s=8.0)
    mask = out['mask']
    assert not mask[1200], 'grid instant 1200 (source 12000) must be in the hole'
    assert mask[1199] and mask[1201], (
        'widening costs at most one grid sample: the neighbours must survive')
    assert (out['cause'][~mask] != 0).all()


def test_window_spanning_a_widened_hole_is_rejected():
    n = 24000
    x = tone(n, 0.3) + 4.5
    x[12002:12007] = np.nan
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT, min_run_s=8.0)
    s, k = rw.window_indices(t_start=11.5, clip_seconds=1.0, fs_out=FS_OUT)
    assert not rw.usable(out['mask'], s, k), (
        'a window covering the hole must be unusable even though the hole is '
        'narrower than one output period')


def test_exact_minimum_run_still_yields_a_full_length_clip():
    """The ADR's §2.4 bound is satisfiable AT the bound: 8 s run -> 800 samples."""
    start, n = 1503, 8000
    x = single_run(tone(n, 0.3, phase0=start), start, 24000)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT, min_run_s=8.0)
    assert int(out['mask'].sum()) == 800
    s, k = rw.window_indices(0.0, 8.0, FS_OUT)
    assert k == 800
    assert rw.usable(out['mask'], 151, 800), (
        'an 8 s window must fit inside an exactly 8 s run (plan G9)')


def test_run_shorter_than_the_minimum_produces_no_samples():
    start, n = 1503, 7999
    x = single_run(tone(n, 0.3, phase0=start), start, 20000)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT, min_run_s=8.0)
    assert out['mask'].sum() == 0 and out['n_runs'] == 0


# --------------------------------------------------------------------------- #
# the emitted contract
# --------------------------------------------------------------------------- #
def test_invalid_samples_are_nan_and_never_zero():
    x = tone(20000, 0.3) + 4.5
    x[5000:6500] = -10.0                      # a 1.5 s saturation block
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    assert np.array_equal(np.isnan(out['signal']), ~out['mask'])
    assert not np.any(out['signal'][~out['mask']] == 0)


def test_mask_cause_and_grid_identity():
    x = tone(24000, 0.3) + 4.5
    x[5000:6500] = -10.0
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    assert np.array_equal(out['mask'], out['cause'] == rpp.CAUSE_VALID)
    assert out['n_src'] == x.size
    assert out['signal'].size == int(np.ceil(x.size / DOWN))
    assert out['fs_in'] == 1000 and out['fs_out'] == 100
    assert out['signal'].dtype == np.float32
    assert 0.0 <= out['valid_frac'] <= 1.0


def test_preprocess_is_deterministic():
    x = tone(24000, 0.3) + 4.5
    x[5000:6500] = -10.0
    a = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    b = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    for key in ('signal', 'mask', 'cause'):
        np.testing.assert_array_equal(a[key], b[key])


def test_an_all_invalid_session_is_still_a_payload():
    """The stage never rules a session unusable (ADR Criterion 7)."""
    x = np.full(20000, -10.0)
    out = rpp.preprocess_session(x, fs_in=FS, fs_out=FS_OUT)
    assert out['valid_frac'] == 0.0 and out['n_runs'] == 0
    assert np.isnan(out['signal']).all() and not out['mask'].any()


# --------------------------------------------------------------------------- #
# build-time refusals (plan A7 / A8)
# --------------------------------------------------------------------------- #
def test_non_divisor_output_rate_is_refused():
    with pytest.raises(ValueError, match='must divide'):
        rpp.resolve_geometry(FS, 128.0)
    with pytest.raises(ValueError, match='above fs_in'):
        rpp.resolve_geometry(FS, 2000.0)


def test_too_few_taps_are_refused():
    with pytest.raises(ValueError, match='taps_per_phase'):
        rpp.resolve_geometry(FS, FS_OUT, taps_per_phase=5)


def test_kernel_radius_is_the_filter_support_not_a_guess():
    up, down, radius = rpp.resolve_geometry(FS, FS_OUT)
    assert (up, down, radius) == (1, 10, 100), (
        'resample_poly designs half_len = 10 * max(up, down) taps, i.e. 100 '
        'SOURCE samples; replicating 20 would leave 80 taps convolving with zeros')
