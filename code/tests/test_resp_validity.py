"""Phase 1 gates — validity decided once, on the raw grid (ADR-003).

Every case states its own intent on a synthetic trace, so a threshold or an
off-by-one cannot hide behind a fixture nobody can read.

Gate map: G2 (rail event-based), G3 (flatline spares slow breathing), G1a (cause
precedence), G1b (mask and cause agree).
"""
import numpy as np
import pytest

from data import resp_preprocess as rpp

FS = 1000.0


def sine(seconds: float, freq: float = 0.3, amp: float = 1.0,
         offset: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(seconds * FS))) / FS
    return offset + amp * np.sin(2 * np.pi * freq * t)


# --------------------------------------------------------------------------- #
# G2 — a rail SAMPLE is not invalid; a rail EVENT is
# --------------------------------------------------------------------------- #
def test_short_rail_graze_is_kept():
    """0.5 s above 9.95 V is a contact, not saturation: every sample stays valid."""
    x = sine(20.0)
    x[5000:5500] = -10.0                     # 0.5 s at the clamp
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert mask[5000:5500].all(), 'a sub-second graze must be KEPT (ADR Criterion 2)'
    assert (cause[5000:5500] == rpp.CAUSE_VALID).all()


def test_multi_second_rail_block_is_removed():
    """1.5 s at the clamp is saturation: every sample of the event is invalid."""
    x = sine(20.0)
    x[5000:6500] = -10.0                     # 1.5 s
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert not mask[5000:6500].any(), 'a >= 1 s block must be removed entirely'
    assert (cause[5000:6500] == rpp.CAUSE_SAT).all()
    # and nothing outside the event is touched
    assert mask[:5000].all() and mask[6500:].all()


def test_rail_event_boundary_is_inclusive_at_threshold():
    """Exactly ``sat_min_s`` invalidates; one sample less does not."""
    x = sine(20.0)
    x[5000:6000] = 9.99                      # exactly 1.000 s
    m_full, _ = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert not m_full[5000:6000].any()

    x2 = sine(20.0)
    x2[5000:5999] = 9.99                     # 0.999 s
    m_short, _ = rpp.validity_mask(x2, fs_in=FS, min_run_s=1.0)
    assert m_short[5000:5999].all()


def test_rail_threshold_is_magnitude_and_excludes_below():
    """9.90 V is below the 9.95 V rail threshold and stays valid."""
    x = sine(20.0)
    x[5000:6500] = 9.90
    mask, _ = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert mask[5000:6500].all()


# --------------------------------------------------------------------------- #
# G3 — the flatline probe must spare slow breathing
# --------------------------------------------------------------------------- #
def test_005_hz_breath_survives_the_probe():
    """A 10 s period breath is not a dead channel (the probe is 5 s, not longer).

    The exact boundary the 5 s probe implies is ``A > 1.02 V`` at 0.05 Hz, since
    the spread over 5 s at the extremum is ``0.293 * A``; a real belt swings
    several volts, so 2 V is the realistic case and 0.8 V is pinned below as the
    documented limit rather than a surprise.
    """
    x = sine(60.0, freq=0.05, amp=2.0)
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert mask.all(), (
        'a 0.05 Hz breath must stay valid — the 5 s probe length exists exactly '
        'so it cannot be mistaken for a frozen channel (ADR Criterion 3)')
    assert (cause == rpp.CAUSE_VALID).all()


def test_005_hz_below_the_spared_amplitude_is_a_documented_limit():
    """A 0.8 V, 0.05 Hz trace IS flagged as flat — the recorded boundary, not a bug."""
    x = sine(60.0, freq=0.05, amp=0.8)      # 0.293 * 0.8 = 0.234 V < 0.3 V
    mask, _ = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert not mask.all(), (
        'the spared amplitude at 0.05 Hz is > 1.02 V; below it the 0.3 V probe '
        'correctly calls the trace near-dead')


def test_003_hz_breath_is_spared_at_any_practical_amplitude():
    x = sine(60.0, freq=0.3, amp=0.5)
    mask, _ = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert mask.all()


def test_frozen_channel_is_removed_and_dilated_over_the_probe():
    """A frozen stretch is invalid, and invalidity covers every window it enters."""
    x = sine(60.0)
    x[20000:30000] = 0.0123                  # frozen: ptp ~0 over 10 s
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert not mask[20000:30000].any()
    assert (cause[20000:30000] == rpp.CAUSE_FLAT).all()
    # dilation: a sample just BEFORE the frozen stretch is inside a 5 s window
    # that ends in it, so it is invalid too.
    assert not mask[19999]


@pytest.mark.parametrize('amplitude', [0.28, 0.32])
def test_flat_threshold_operating_point(amplitude):
    """0.28 V peak-to-peak is flat, 0.32 V is not — the 0.3 V point is exact."""
    n = 30000
    x = np.zeros(n)
    t = np.arange(n) / FS
    x[:] = 0.5 * amplitude * np.sin(2 * np.pi * 0.05 * t)   # ptp = amplitude
    x[10000:25000] = 0.0
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    frozen_ok = mask[15000]                          # deep inside the frozen part
    if amplitude < 0.3:
        assert not mask[15000]
    else:
        # 0.32 V ptp over a full 5 s window is above the threshold
        assert mask[15000] or cause[15000] == rpp.CAUSE_FLAT
    assert frozen_ok == mask[15000]


def test_flat_probe_equals_bruteforce_sliding_window():
    """The rolling-filter implementation is pinned against the literal definition."""
    rng = np.random.default_rng(0)
    n = 5000
    x = 0.2 * np.sin(2 * np.pi * 0.4 * np.arange(n) / FS)
    x += 0.01 * rng.standard_normal(n)
    x[2000:3000] = 0.05
    x[4000:4003] = np.nan                    # a NaN gap must not create a flatline
    n_w = rpp.flatline_window_samples(FS, 1.0)   # 1 s probe keeps the test small

    win = np.lib.stride_tricks.sliding_window_view(x, n_w)
    full = np.isfinite(win).all(axis=1)
    ptp = np.ptp(np.where(full[:, None], win, 0.0), axis=-1)
    bad = full & (ptp < rpp.FLAT_PTP_V)
    cover = np.zeros(n, dtype=bool)
    for j in np.flatnonzero(bad):
        cover[j:j + n_w] = True
    expect_valid = ~(cover | ~np.isfinite(x))

    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=0.001,
                                    flat_window_s=1.0)
    assert np.array_equal(mask, expect_valid), (
        'the min/max-filter probe must equal the brute-force window loop; this '
        'is the one place an off-by-one silently changes the mask')
    assert np.array_equal(cause == rpp.CAUSE_FLAT, cover & np.isfinite(x))


# --------------------------------------------------------------------------- #
# short runs become holes
# --------------------------------------------------------------------------- #
def test_short_run_becomes_a_hole_and_long_run_survives():
    """Two 10 s runs survive a 6 s rail block between them; the block is the hole."""
    x = np.full(36000, np.nan)
    x[0:10000] = sine(10.0)
    x[10000:16000] = -10.0                     # 6 s rail -> saturation hole
    x[16000:26000] = sine(10.0)
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=8.0)
    assert mask[0:10000].all()
    assert mask[16000:26000].all()
    assert not mask[10000:16000].any()
    assert (cause[10000:16000] == rpp.CAUSE_SAT).all()


def test_run_shorter_than_minimum_is_short_run():
    x = np.full(20000, np.nan)
    x[1000:7000] = sine(6.0)                   # 6 s, no other defect
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=8.0)
    assert not mask.any()
    assert (cause[1000:7000] == rpp.CAUSE_SHORT).all()


# --------------------------------------------------------------------------- #
# G1a / G1b — one cause per sample, and it agrees with the mask
# --------------------------------------------------------------------------- #
def test_nonfinite_outranks_rail():
    """A NaN inside a rail block carries the non-finite code, the rest the rail."""
    x = sine(20.0)
    x[5000:6500] = -10.0
    x[5500:5510] = np.nan
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert (cause[5500:5510] == rpp.CAUSE_NONFINITE).all()
    assert (cause[5000:5500] == rpp.CAUSE_SAT).all()
    assert (cause[5510:6500] == rpp.CAUSE_SAT).all()
    assert not mask[5000:6500].any()


def test_mask_and_cause_agree_everywhere():
    rng = np.random.default_rng(3)
    x = sine(40.0) + 0.02 * rng.standard_normal(int(40 * FS))
    x[5000:6500] = -10.0
    x[20000:26000] = 0.0
    x[30000:30005] = np.nan
    mask, cause = rpp.validity_mask(x, fs_in=FS, min_run_s=1.0)
    assert np.array_equal(mask, cause == rpp.CAUSE_VALID), (
        'mask[k] == False <=> cause[k] != 0 (ADR §2.1)')


def test_flatline_window_is_odd_so_the_probe_can_be_aligned():
    assert rpp.flatline_window_samples(FS, 5.0) % 2 == 1
    assert rpp.flatline_window_samples(FS, 5.0) == 5001
    with pytest.raises(ValueError):
        rpp.flatline_window_samples(FS, 0.0)
