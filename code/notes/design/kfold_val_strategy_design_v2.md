# Technical Design: GroupK-Fold Cyclic Rotation & 3-Stage Pipeline Strategy

## Executive Summary
This document defines the technical architecture for subject-exclusive cross-validation and stage-specific dataset allocation across the 3-stage model lifecycle on the **BP4D+ dataset** (140 subjects, multi-clip continuous thermal/physiological waveforms). 

It explicitly standardizes the 3-stage training pipeline nomenclature:
1. **Stage 1**: **Initial Backbone Weight Ingestion** — Loading initial pretrained weights provided by the VideoMAE authors (pretrained on Kinetics-400/SSv2).
2. **Stage 2**: **Domain SSL Pretraining** — Self-Supervised Learning (Solution A / Asymmetric Cross-MAE) on the BP4D+ domain dataset.
3. **Stage 3**: **Task Fine-Tuning** — Supervised downstream fine-tuning for continuous respiratory waveform prediction and Action Unit (AU) detection.

---

## 1. Pipeline Stage Definition & Lifecycle Overview

| Stage Index | Stage Name & Scope | Input Weights Source | Dataset Allocation (per Fold) | Validation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | **Initial Weights Ingestion** | VideoMAE Author Checkpoint (Kinetics-400 / SSv2) | N/A (Global Initialization) | N/A |
| **Stage 2** | **Domain SSL Pretraining** | Stage 1 VideoMAE Weights | **80% Train** (112 Subjects)<br>**20% Test** (28 Subjects Locked) | **0% Val** (Early stopping inapplicable; maximizes subject diversity) |
| **Stage 3** | **Task Fine-Tuning** | Stage 2 Domain Pretrained Weights | **Option A**: 80% Train / 20% Test<br>**Option B**: 70% Train / 10% Val / 20% Test | **Option A**: Fixed-epoch Cosine Annealing<br>**Option B**: 14-subject small validation subset |

---

## 2. GroupK-Fold with Cyclic 5-Fold Subject Rotation

### 2.1 Problem Statement
In physiological dataset processing (e.g., BP4D+ with 140 subjects and 1,400 clips), clips from the same subject exhibit high temporal and biometric correlation. Random splitting leads to severe **data leakage**. Furthermore, implementing a nested `GroupKFold` (`gkf_outer` + `gkf_inner`) introduces complex relative index mapping, slice offsets, and potential alignment bugs across parallel execution workers.

### 2.2 Core Mechanism: Cyclic Rotation
Instead of nesting `GroupKFold` instances, a single outer `n_folds` `GroupKFold` partitions all subjects into **mutually exclusive, subject-isolated folds** ($F_0, F_1, \dots, F_{n-1}$). For the full BP4D+ dataset, $n=5$ yields ~28 subjects per fold (~20%). For the local dev environment with 4 subjects, $n=4$ yields 1 subject per fold (Leave-One-Subject-Out).

For a given parallel worker executing Fold $k \in \{0, \dots, n-1\}$, subject sets are deterministically assigned via cyclic modular arithmetic:

$$\begin{aligned}
\text{Test Fold } (T_k) &= F_k \\
\text{Validation Fold } (V_k) &= F_{(k + 1) \bmod n} && (\text{if required in Stage 3}) \\
\text{Training Folds } (R_k) &= \{F_{(k + 2) \bmod n}, \dots, F_{(k + n - 1) \bmod n}\}
\end{aligned}$$

When utilizing **0% Validation (Stage 2 SSL & Stage 3 Option A)**, the training pool expands to include all $n-1$ non-testing folds ($F_{(k+1)\bmod n}$ through $F_{(k+n-1)\bmod n}$).

```
Fold 0: [ Test: F0 (28 subs) | Train Pool: F1, F2, F3, F4 (112 subs) ]
Fold 1: [ Test: F1 (28 subs) | Train Pool: F2, F3, F4, F0 (112 subs) ]
Fold 2: [ Test: F2 (28 subs) | Train Pool: F3, F4, F0, F1 (112 subs) ]
Fold 3: [ Test: F3 (28 subs) | Train Pool: F4, F0, F1, F2 (112 subs) ]
Fold 4: [ Test: F4 (28 subs) | Train Pool: F0, F1, F2, F3 (112 subs) ]
```

---

## 3. Detailed Stage-Specific Validation Strategies

### 3.1 Stage 1: Initial Weights Download & Ingestion
* **Objective**: Load official ViT-Base VideoMAE weights pretrained on general spatio-temporal video benchmarks (Kinetics-400 or Something-Something v2).
* **Dataset**: No BP4D+ data processing required at this initialization step.
* **Role**: Serves as the base initialization ($W_{\text{stage1}}$) fed into Stage 2 domain pretraining.

---

### 3.2 Stage 2: Domain SSL Pretraining (Solution A / Asymmetric Cross-MAE)
* **Objective**: Pretrain the ViT-Base encoder on domain-specific thermal infrared (TIR) video sequences using cross-modal frame alignment and InfoNCE loss against physiological signals.
* **Allocation**: **112 Subjects (80%) Train**, **0 Subjects (0%) Val**, **28 Subjects (20%) Test Holdout**.

```
       140 Subjects (BP4D+ Dataset)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
112 Subjects (80%)   28 Subjects (20%)
 Stage 2 SSL Train   Strictly Locked
 (100% Pretrain)     (Test Holdout)
```

* **Scientific Rationale**:
  1. **Inapplicability of Early Stopping**: SSL pretext tasks optimize contrastive/reconstruction loss over fixed step/epoch schedules (e.g., 100–300 epochs). No downstream target labels exist to evaluate validation accuracy during pretext training.
  2. **Subject Diversity Primacy**: ViT-Base (~86M parameters) requires maximum subject diversity to learn generalized facial thermal/physiological feature manifolds. Withholding 28 subjects for an SSL validation set unnecessarily strips 25% of available subject variance.
  3. **Uncompromised Test Firewall**: As long as the 28 test subjects remain strictly untouched throughout pretraining, the evaluation protocol remains 100% leak-free and scientifically sound.

---

### 3.3 Stage 3: Supervised Task Fine-Tuning

During downstream task fine-tuning (e.g., continuous respiratory waveform prediction with Negative Pearson Correlation Loss), two scientifically valid options exist depending on hyperparameter optimization needs:

#### Option A: Fixed-Epoch Schedule (Recommended: 80% Train / 0% Val / 20% Test)
* **Train Set**: 112 subjects (80%)
* **Val Set**: 0 subjects (0%)
* **Test Set**: 28 subjects (20%, locked holdout)
* **Mechanism**: Fine-tune for a predetermined number of epochs (e.g., 30–50) with Cosine Annealing learning rate decay, evaluating directly on the test set. Maximize training subjects to avoid data starvation on ViT-Base.

#### Option B: Asymmetric Small Validation Subset (70% Train / 10% Val / 20% Test)
* **Train Set**: ~98 subjects (~70%)
* **Val Set**: ~14 subjects (~10%)
* **Test Set**: 28 subjects (20%, locked holdout)
* **Mechanism**: If early stopping or validation loss tracking is strictly required, carve a small 10% subject-isolated validation subset from the 112-subject training pool rather than a full 20% fold. This recovers 14 subjects back into the training pool compared to standard double-fold slicing.

---

## 4. Reference Python Implementation

```python
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold

def get_stage_splits(samples: np.ndarray, subjects: np.ndarray, fold: int, n_folds: int = 5, stage: int = 3, use_small_val: bool = False):
    """
    Generates subject-exclusive splits for Stage 2 (SSL) and Stage 3 (Task Fine-Tuning).
    
    Args:
        samples: Dataset sample array.
        subjects: Subject ID array aligned with samples.
        fold: Current fold index (0 to n_folds - 1).
        n_folds: Total number of folds (e.g., 5 for HPC, 4 for local LOSO).
        stage: Pipeline stage (2 for Domain SSL, 3 for Task Fine-Tuning).
        use_small_val: If True in Stage 3, uses 10% val instead of 0%.
    """
    # Handle edge case: 1 fold means train on everything, test on everything (or specific holdout)
    if n_folds <= 1:
        all_idx = np.arange(len(samples))
        return all_idx, np.array([], dtype=int), all_idx

    gkf = GroupKFold(n_splits=n_folds)
    all_folds = [test_idx for _, test_idx in gkf.split(samples, groups=subjects)]
    
    test_idx = all_folds[fold]
    
    if stage == 2:
        # Stage 2 (Domain SSL): 0% Val
        train_fold_indices = [(fold + i) % n_folds for i in range(1, n_folds)]
        train_idx = np.concatenate([all_folds[f] for f in train_fold_indices])
        val_idx = np.array([], dtype=int)
        
    elif stage == 3:
        if not use_small_val:
            # Option A: Fixed Epoch Fine-tuning (0% Val)
            train_fold_indices = [(fold + i) % n_folds for i in range(1, n_folds)]
            train_idx = np.concatenate([all_folds[f] for f in train_fold_indices])
            val_idx = np.array([], dtype=int)
        else:
            # Option B: Small Validation Set (~10% Val)
            raw_train_fold_indices = [(fold + i) % n_folds for i in range(1, n_folds)]
            raw_train_subjects = np.unique(subjects[np.concatenate([all_folds[f] for f in raw_train_fold_indices])])
            
            # Calculate 10% of total subjects, minimum 1 subject
            total_subjects = len(np.unique(subjects))
            n_val_subs = max(1, int(round(total_subjects * 0.10)))
            
            # Sub-split training subjects
            np.random.seed(42 + fold)
            val_subs = np.random.choice(raw_train_subjects, size=n_val_subs, replace=False)
            train_subs = np.setdiff1d(raw_train_subjects, val_subs)
            
            raw_train_idx = np.concatenate([all_folds[f] for f in raw_train_fold_indices])
            val_idx = raw_train_idx[np.isin(subjects[raw_train_idx], val_subs)]
            train_idx = raw_train_idx[np.isin(subjects[raw_train_idx], train_subs)]
            
    return train_idx, val_idx, test_idx
```

---

## 5. Summary Matrix

| Stage | Stage Description | Train Subjects | Val Subjects | Test Subjects | Validation Strategy Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | **Initial Weights Ingestion** | N/A | N/A | N/A | Downloading official VideoMAE author weights (Kinetics-400 / SSv2). |
| **Stage 2** | **Domain SSL Pretraining** | **112 (80%)** | **0 (0%)** | **28 (20%)** | Maximizes pretraining subject diversity; early stopping is inapplicable without target labels. |
| **Stage 3 (Option A)** | **Task Fine-Tuning (Fixed Epoch)** | **112 (80%)** | **0 (0%)** | **28 (20%)** | Fixed-epoch training with Cosine Annealing; avoids data starvation on ViT-Base backbone. |
| **Stage 3 (Option B)** | **Task Fine-Tuning (Small Val)** | **98 (70%)** | **14 (10%)** | **28 (20%)** | Enables early stopping while recovering 14 subjects back to training compared to 20% val. |

---

## 6. Acceptance Criteria (BDD-Style Tests)

To ensure the implementation strictly adheres to the cyclic rotation and stage-specific validation rules, the following Behavior-Driven Development (BDD) scenarios must pass.

### Scenario 1: Stage 2 (Domain SSL) Subject Allocation
**Given** a dataset of 140 subjects
**And** the pipeline is configured for Stage 2 (Domain SSL)
**When** the split generator is called for Fold 0
**Then** the Test set must contain exactly 28 subjects (Fold 0)
**And** the Validation set must be empty (0 subjects)
**And** the Train set must contain exactly 112 subjects (Folds 1, 2, 3, 4)
**And** the intersection of Train and Test subjects must be empty

### Scenario 2: Stage 3 (Task Fine-Tuning) Option A - Fixed Epoch
**Given** a dataset of 140 subjects
**And** the pipeline is configured for Stage 3 (Task Fine-Tuning) with Option A (0% Val)
**When** the split generator is called for Fold 2
**Then** the Test set must contain exactly 28 subjects (Fold 2)
**And** the Validation set must be empty (0 subjects)
**And** the Train set must contain exactly 112 subjects (Folds 3, 4, 0, 1)
**And** the intersection of Train and Test subjects must be empty

### Scenario 3: Stage 3 (Task Fine-Tuning) Option B - Small Validation
**Given** a dataset of 140 subjects
**And** the pipeline is configured for Stage 3 (Task Fine-Tuning) with Option B (10% Val)
**When** the split generator is called for Fold 4
**Then** the Test set must contain exactly 28 subjects (Fold 4)
**And** the Validation set must contain exactly 14 subjects
**And** the Train set must contain exactly 98 subjects
**And** the Validation subjects must be a subset of the non-test folds (Folds 0, 1, 2, 3)
**And** the intersection of Train, Validation, and Test subjects must be mutually empty

### Scenario 4: Cyclic Rotation Determinism
**Given** a dataset of 140 subjects
**When** the split generator is called for Fold 1 across multiple independent execution runs
**Then** the exact same 28 subjects must be assigned to the Test set every time
**And** the exact same subjects must be assigned to the Train (and Val) sets every time

### Scenario 5: No Overlapping Clips Across Folds
**Given** a dataset of continuous video sessions
**And** clips are generated using a sliding window with overlap (e.g., 8s window, 2s stride)
**When** the dataset is partitioned into folds
**Then** all clips originating from Subject X must reside entirely within a single fold
**And** no two overlapping clips from the same session can ever be split across Train, Val, or Test sets

### Scenario 6: Dynamic Fold Count Compatibility (Local vs HPC)
**Given** a local dataset of only 4 subjects
**And** the pipeline is configured for `n_folds = 4`
**When** the split generator is called for Fold 0
**Then** the Test set must contain exactly 1 subject
**And** the Train set must contain exactly 3 subjects (if 0% Val is used)
**And** the cyclic rotation logic must correctly wrap around using modulo 4 instead of modulo 5
