"""Post-processing evaluation (docs/ImplementationPlan.md).

Tier 1 -- temporal alignment:   MAE, RMSE, Pearson r
Tier 2 -- spectral fidelity:    Welch PSD consistency, MR-STFT tracking
Tier 3 -- clinical fidelity:    offline HRV via NeuroKit2 (RMSSD, pNN50,
                                MedianNN, Shannon entropy)

These are computed **offline** on predicted continuous waveforms; the
clinical Tier-3 metrics are non-differentiable and never part of training.
"""
from .metrics import aggregate_metrics, breakdown_by_label
from .metrics import bandpass_fft, band_limited_pearson, bootstrap_ci
from .metrics import estimate_rate_bpm, rate_metrics, rate_summary
from .metrics import single_tone_template, time_domain_metrics, spectral_metrics
from .clinical import extract_hrv_metrics
