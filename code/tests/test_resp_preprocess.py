"""Unit tests for ``data/resp_preprocess.py`` (pure preprocessing, no IO).

Every case builds its own synthetic session so the thresholds and boundaries
are exact and the test states its own intent; no binary fixture is checked in.
Real ``Resp_Volts.txt`` files are only used by the builder's ``--check`` parity
gate.

Design: ``code/notes/design/RESP_session_preprocess.md`` §3.4.
"""
import os
import sys

import numpy as np
import pytest
from scipy.signal import resample_poly

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import resp_preprocess as rp

FS = 1000.0
OUT = 100.0


def _sine(seconds: float, f: float, amp: float = 1.0, dc: float = 0.0,
          fs: float = FS) -> np.ndarray:
    t = np.arange(int(round(seconds * fs))) / fs
    return dc + amp * np.sin(2 * np.pi * f * t)


def _valid_window_ok(mask: np.ndarray, s: int, n: int) -> bool:
    return bool(mask[s:s + n].all())


# --------------------------------------------------------------------------- #
# saturation is EVENT-based
# --------------------------------------------------------------------------- #
def test_short_rail_contact_is_kept():
    """0.5 s at 12 V is shorter than ``sat_min_s`` (1 s) -> stays valid."""
    x = _sine(20.0, 0.3, amp=2.0)
    x[2000:2500] = 12.0                      # 0.5 s
    mask, cause = rp.validity_mask(x, FS)
    assert mask[2000:2500].all()
    assert (cause[2000:2500] == rp.CAUSE_VALID).all()
    assert mask.sum() == x.size


def test_rail_event_at_the_alarm_time_invalidates():
    """1.5 s at -12 V IS an event at the design's 1 s alarm time.

    Design §2 (2026-10-31) shortened ``sat_min_s`` from 2 s to 1 s, so a
    contact that used to be kept is now invalidated.
    """
    x = _sine(30.0, 0.3, amp=2.0)
    x[10000:11500] = -12.0                   # 1.5 s
    mask, cause = rp.validity_mask(x, FS)
    assert not mask[10000:11500].any()
    assert (cause[10000:11500] == rp.CAUSE_SAT).all()


def test_long_rail_event_invalidates_exactly_the_event():
    """2.5 s at -12 V is an event -> exactly its samples are cause 1."""
    x = _sine(30.0, 0.3, amp=2.0)
    x[10000:12500] = -12.0                   # 2.5 s, mid-file
    mask, cause = rp.validity_mask(x, FS)
    assert not mask[10000:12500].any()
    assert (cause[10000:12500] == rp.CAUSE_SAT).all()
    # outside the event nothing is invalidated (both runs are >= 8 s)
    assert (cause[:10000] == rp.CAUSE_VALID).all()
    assert (cause[12500:] == rp.CAUSE_VALID).all()


def test_many_short_grazes_are_not_an_event():
    """A large rail FRACTION made of short grazes is not a 2 s event.

    This is the F004 pattern that the old whole-window rule dropped entirely.
    """
    x = _sine(20.0, 0.3, amp=2.0)
    for start in range(0, x.size - 400, 1000):
        x[start:start + 400] = 10.5          # 0.4 s graze every second
    mask, cause = rp.validity_mask(x, FS)
    assert (cause != rp.CAUSE_SAT).all()
    # every run is still >= 8 s because the grazes never split the signal
    assert mask.all()


# --------------------------------------------------------------------------- #
# flatline
# --------------------------------------------------------------------------- #
def test_ripple_above_the_threshold_stays_valid():
    """0.4 V ptp (2 s period) is above the 0.3 V threshold -> valid.

    This is the M058 shallow-breathing case: real, low-amplitude respiration
    that the 0.5 V contract used to throw away.
    """
    x = _sine(30.0, 0.5, amp=0.2)            # ptp 0.4 V
    mask, cause = rp.validity_mask(x, FS)
    assert mask.all(), f'{int((~mask).sum())} sample(s) flagged flat'


def test_shallow_breathing_below_the_threshold_is_flat():
    """0.24 V ptp is below ``flat_ptp_v`` 0.3 V -> every sample cause 2.

    A 5 s peak-to-peak probe cannot distinguish a channel that rings a little
    from one that is frozen, so the residual band below the threshold is still
    thrown away.
    """
    x = _sine(30.0, 0.5, amp=0.12)           # ptp 0.24 V
    mask, cause = rp.validity_mask(x, FS)
    assert not mask.any()
    assert (cause == rp.CAUSE_FLAT).all()


def test_small_ripple_is_flat():
    """0.04 V ptp (the old dead-channel case) is still flat."""
    x = _sine(30.0, 0.5, amp=0.02)
    mask, cause = rp.validity_mask(x, FS)
    assert not mask.any()
    assert (cause == rp.CAUSE_FLAT).all()


def test_flat_window_matches_a_brute_force_loop():
    """The filtered implementation equals an explicit ``np.ptp`` window loop."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    n_w = 137
    got = rp._window_ptp(x, n_w)
    want = np.array([np.ptp(x[j:j + n_w])
                     for j in range(x.size - n_w + 1)])
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=0, atol=1e-12)


def test_flatline_covers_exactly_the_stretch():
    """A 6 s dead stretch is flat; the live breathing on either side is not."""
    x = _sine(40.0, 0.5, amp=1.0)
    x[10000:16000] = 0.0                     # 6 s, longer than the 5 s probe
    mask, cause = rp.validity_mask(x, FS)
    assert (cause[10000:16000] == rp.CAUSE_FLAT).all()
    assert not mask[10000:16000].any()
    # A 5 s probe that catches a SLIVER of the sine right at the boundary stays
    # below the threshold, so the flat span reaches a little past the stretch.
    # State the bound instead of hard-coding the threshold: < 0.5 s on each side.
    idx = np.flatnonzero(cause == rp.CAUSE_FLAT)
    assert idx.min() <= 10000 and idx.max() >= 15999
    assert 10000 - int(idx.min()) < 0.5 * FS
    assert int(idx.max()) - 15999 < 0.5 * FS


def test_flat_stretch_shorter_than_the_probe_is_not_flat():
    """3 s of dead signal never fills a 5 s window -> still valid."""
    x = _sine(20.0, 0.5, amp=1.0)
    x[10000:13000] = 0.0
    mask, cause = rp.validity_mask(x, FS)
    assert (cause == rp.CAUSE_VALID).all()


# --------------------------------------------------------------------------- #
# short runs
# --------------------------------------------------------------------------- #
def test_short_run_becomes_a_hole():
    """6 s of breathing + a 3 s rail event -> the 6 s run is cause 3."""
    x = _sine(9.0, 0.3, amp=2.0)
    x[6000:] = 12.0                          # 3 s >= sat_min_s
    mask, cause = rp.validity_mask(x, FS)
    assert not mask.any()
    assert (cause[:6000] == rp.CAUSE_SHORT).all()
    assert (cause[6000:] == rp.CAUSE_SAT).all()


def test_run_exactly_at_the_minimum_is_kept():
    x = _sine(8.0, 0.3, amp=2.0)
    mask, _ = rp.validity_mask(x, FS)
    assert mask.all()


# --------------------------------------------------------------------------- #
# DC removal, resampling, amplitude
# --------------------------------------------------------------------------- #
def test_dc_offset_is_removed_per_run():
    """Adding 4.5 V to a run must not change the stored signal."""
    x = _sine(30.0, 0.5, amp=1.0)
    a = rp.preprocess_session(x, FS, OUT)
    b = rp.preprocess_session(x + 4.5, FS, OUT)
    m = a['mask'] & b['mask']
    assert m.sum() > 1000
    assert np.allclose(a['signal'][m], b['signal'][m], rtol=0, atol=1e-5)


def test_two_runs_have_independent_offsets():
    """A break separates two DC levels: each run is centred on its own mean."""
    x = np.concatenate([_sine(12.0, 0.5, amp=1.0, dc=0.0),
                        _sine(12.0, 0.5, amp=1.0, dc=5.0)])
    x[11500:14000] = -12.0                   # 2.5 s event -> a real break
    out = rp.preprocess_session(x, FS, OUT)
    sig, m = out['signal'], out['mask']
    assert out['n_runs'] == 2
    # interior of run 1 and run 2, away from the run edges
    n = sig.size
    r1 = np.zeros(n, bool)
    r1[30:1100] = True
    r2 = np.zeros(n, bool)
    r2[1450:] = True
    m1, m2 = m & r1, m & r2
    assert m1.sum() > 500 and m2.sum() > 500
    # each run is centred on ITS OWN mean; the residual is the conservative
    # edge trim (a few samples at each run edge), not the 5 V DC gap
    assert abs(float(sig[m1].mean())) < 5e-2
    assert abs(float(sig[m2].mean())) < 5e-2
    # premise: the two runs really did sit at different DC levels
    assert abs(float(x[100:11000].mean()) - float(x[14500:23000].mean())) > 4.0


def test_amplitude_survives_the_resample():
    """0.3 Hz, 1 V amplitude -> std 0.707 within 1 %."""
    x = _sine(30.0, 0.3, amp=1.0, dc=3.0)
    out = rp.preprocess_session(x, FS, OUT)
    sig, m = out['signal'], out['mask']
    assert m.sum() > 1000
    assert abs(float(sig[m].std()) - 1.0 / np.sqrt(2.0)) < 0.01 * 0.70710678


def test_grid_mapping_contract():
    x = _sine(12.345, 0.3)
    out = rp.preprocess_session(x, FS, OUT)
    assert out['n_out'] == rp.grid_length(x.size, FS, OUT)
    assert out['n_out'] == int(np.floor((x.size - 1) * OUT / FS)) + 1
    assert out['signal'].shape == (out['n_out'],)
    assert out['mask'].shape == (out['n_out'],)


def test_final_grid_sample_is_filled_for_a_run_ending_at_eof():
    """The grid is ``floor((n-1)*fo/fi)+1`` samples long -- the last must exist.

    Regression: the per-run output length was ``round(len*fo/fi)``, so a run
    reaching EOF (92382 -> 9238.2 -> 9238) left the session's LAST grid sample
    unfilled: ``NaN`` with cause 0, mask False, and the session's final window
    silently dropped.
    """
    x = _sine(92.382, 0.3, amp=1.0, dc=3.0)      # exactly 92 382 samples
    assert x.size == 92382
    out = rp.preprocess_session(x, FS, OUT)
    assert out['n_out'] == rp.grid_length(x.size, FS, OUT) == 9239
    assert out['mask'].all()
    assert out['valid_frac'] == 1.0
    assert np.isfinite(out['signal']).all()
    # and the filled tail is the real waveform, not garbage
    assert abs(float(out['signal'][-1])) < 1.5


# --------------------------------------------------------------------------- #
# aliasing gate (design C5 / plan G2)
# --------------------------------------------------------------------------- #
def test_aliasing_tone_is_removed_but_np_interp_keeps_it():
    """A 130 Hz tone (above the 50 Hz output Nyquist) must not fold in.

    Point sampling (``np.interp``) aliases 130 Hz to 30 Hz; the anti-aliased
    ``resample_poly`` path removes it. This is the analytic proof that C5 is
    real.
    """
    t = np.arange(int(30.0 * FS)) / FS
    clean = np.sin(2 * np.pi * 0.3 * t) + 3.0
    noisy = clean + 0.1 * np.sin(2 * np.pi * 130.0 * t)
    a = rp.preprocess_session(clean, FS, OUT)
    b = rp.preprocess_session(noisy, FS, OUT)
    m = a['mask'] & b['mask']
    resid = b['signal'][m].astype(np.float64) - a['signal'][m].astype(np.float64)
    assert np.sqrt(np.mean(resid ** 2)) < 1e-2      # >= 40 dB down

    # the naive path the design retires
    n_out = a['n_out']
    naive = np.interp(np.arange(n_out) / OUT * FS, np.arange(t.size), noisy)
    naive_clean = np.interp(np.arange(n_out) / OUT * FS, np.arange(t.size), clean)
    naive_err = np.sqrt(np.mean((naive - naive_clean)[m] ** 2))
    assert naive_err > 5e-2                          # aliasing is NOT removed


# --------------------------------------------------------------------------- #
# conservative grid mask
# --------------------------------------------------------------------------- #
def test_40ms_gap_leaves_a_hole_on_the_grid():
    x = _sine(20.0, 0.3, amp=2.0)
    x[10000:10040] = np.nan                  # 40 ms non-finite gap
    out = rp.preprocess_session(x, FS, OUT)
    sig, m, cause = out['signal'], out['mask'], out['cause']
    assert np.array_equal(np.isnan(sig), ~m)
    assert (~m).sum() >= 1
    # the gap maps to output sample ~1000; no 8 s window may straddle it
    gap_out = int(round(10000 * OUT / FS))
    n = int(round(8.0 * OUT))
    for s in range(0, sig.size - n + 1):
        if s <= gap_out < s + n:
            assert not m[s:s + n].all()
    assert (cause[~m] != 0).all()


def test_nonfinite_samples_are_never_valid_and_are_nan():
    x = _sine(20.0, 0.3, amp=2.0)
    x[[100, 5000, 15000]] = np.inf
    out = rp.preprocess_session(x, FS, OUT)
    assert out['n_nonfinite'] == 3
    assert np.array_equal(np.isnan(out['signal']), ~out['mask'])


def test_invalid_samples_are_nan_not_zero():
    x = _sine(20.0, 0.3, amp=2.0)
    x[5000:8000] = 12.0                      # 3 s saturation event
    out = rp.preprocess_session(x, FS, OUT)
    assert np.isnan(out['signal'][~out['mask']]).all()
    assert not np.any(out['signal'][~out['mask']] == 0.0)


# --------------------------------------------------------------------------- #
# determinism + degenerate inputs
# --------------------------------------------------------------------------- #
def test_second_call_is_bit_identical():
    x = _sine(25.0, 0.3, amp=1.5, dc=1.0)
    x[12000:12500] = np.nan
    a = rp.preprocess_session(x, FS, OUT)
    b = rp.preprocess_session(x, FS, OUT)
    assert np.array_equal(a['signal'], b['signal'], equal_nan=True)
    assert np.array_equal(a['mask'], b['mask'])
    assert np.array_equal(a['cause'], b['cause'])


def test_empty_and_flat_sessions_do_not_crash():
    out = rp.preprocess_session(np.zeros(0), FS, OUT)
    assert out['signal'].size == 0 and out['mask'].size == 0
    out = rp.preprocess_session(np.full(20000, 5.0), FS, OUT)
    assert out['mask'].sum() == 0
    assert (out['cause'] == rp.CAUSE_FLAT).all()


def test_non_integral_rates_are_refused():
    with pytest.raises(ValueError):
        rp.preprocess_session(_sine(12.0, 0.3), 1000.0, 99.5)


def test_resample_run_matches_scipy_directly():
    y = _sine(10.0, 0.3, amp=1.0)
    want = resample_poly(np.pad(y, (rp.RESAMPLE_HALF_TAPS,) * 2, mode='edge'),
                         up=1, down=10)
    want = want[rp.RESAMPLE_HALF_TAPS // 10:
                rp.RESAMPLE_HALF_TAPS // 10 + y.size // 10]
    got = rp.resample_run(y, FS, OUT)
    assert np.allclose(got, want, rtol=0, atol=1e-12)
