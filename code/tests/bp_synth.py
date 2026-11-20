"""Synthetic blood-pressure traces for the ADR-004 gates.

**No binary fixture is committed.** Every trace is generated here, and each test
states its own window of it, so an off-by-one cannot hide behind a stored file.

``sine`` deliberately mixes two incommensurate sinusoids: a pure 1 Hz sine at
1 kHz can repeat an exact value on neighbouring samples at its extrema, which
would make a fixture accidentally test Gate 2 or Gate 3. ``max_equal_run`` lets a
test assert that a fixture is what it claims to be.
"""
import numpy as np

FS = 1000.0


def sine(n, fs=FS, freq=1.0, amp=1.0, level=90.0, phase=0.0) -> np.ndarray:
    """A pulsatile trace around ``level`` with peak-to-peak ≈ 2.62 · ``amp``."""
    t = np.arange(int(n), dtype=np.float64) / float(fs)
    w = 2.0 * np.pi * float(freq) * t
    return (float(level)
            + float(amp) * (np.sin(w + phase)
                            + 0.31 * np.sin(0.37 * w + 0.7)))


def constant(n, value) -> np.ndarray:
    """``n`` samples of exactly ``value`` — a flatline / dwell / plateau block."""
    return np.full(int(n), float(value), dtype=np.float64)


def concat(*parts) -> np.ndarray:
    """Concatenate trace blocks into one float64 trace."""
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])


def max_equal_run(x) -> int:
    """Longest run of exactly-equal consecutive samples (fixture self-check)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0
    best = run = 1
    for i in range(1, x.size):
        run = run + 1 if x[i] == x[i - 1] else 1
        best = max(best, run)
    return best


def brute_equal_runs(x, alive) -> list:
    """Reference implementation of ``data.bp_preprocess.equal_runs``.

    A plain loop, deliberately naive: the vectorised form is pinned against it.
    """
    x = np.asarray(x)
    alive = np.asarray(alive, dtype=bool)
    out = []
    i, n = 0, x.size
    while i < n:
        if not alive[i]:
            i += 1
            continue
        j = i + 1
        while j < n and alive[j] and x[j] == x[i]:
            j += 1
        out.append((i, j))
        i = j
    return out
