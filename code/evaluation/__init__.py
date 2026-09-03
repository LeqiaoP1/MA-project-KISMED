"""Post-processing evaluation (docs/ImplementationPlan.md).

Tier 1 -- temporal alignment:   MAE, RMSE, Pearson r
Tier 2 -- spectral fidelity:    Welch PSD consistency, MR-STFT tracking
Tier 3 -- clinical fidelity:    offline HRV via NeuroKit2 (RMSSD, pNN50,
                                MedianNN, Shannon entropy)

These are computed **offline** on predicted continuous waveforms; the
clinical Tier-3 metrics are non-differentiable and never part of training.
"""
from .metrics import time_domain_metrics, spectral_metrics
from .clinical import extract_hrv_metrics
