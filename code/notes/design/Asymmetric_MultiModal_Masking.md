# Software Design Specification: Asymmetric Multi-Modal Masking Engine

## 1. System Overview & Architectural Motivation

In contactless multi-modal physiological estimation systems—combining facial RGB video ($128 \times 128$), thermal infrared (TIR) video ($96 \times 96$), 1D continuous Blood Pressure (BP) waveforms, and 1D continuous Respiration (Resp) waveforms—designing an optimal self-supervised pre-training framework requires balancing five core structural demands:

1. **Formal Architecture Name: Asymmetric Multi-Modal Masking (AMMM)**:
   This specification formalizes the engine under its official publication title: **Asymmetric Multi-Modal Masking (AMMM)**. The name highlights the core architectural innovation: applying domain-asymmetric kept/masked ratio constraints across visual video tubelets ($\le 25\%$ kept / $\ge 75\%$ masked) and 1D continuous physiological signals (uncapped up to $100\%$ kept) while maintaining scale-invariant Dirichlet budget allocation.
2. **Visuo-Physiological Co-Existence Constraint (At Least 1 Video + At Least 1 Physio)**:
   Contactless physiological estimation relies fundamentally on cross-modal representation learning—mapping spatial facial color dynamics (cardiac rPPG) and thermal nostril variations (breathing airflow) to continuous 1D biological waveforms. To ensure every pre-training mini-batch engages cross-modal attention between vision and physiology, the Level-1 Dropout Engine enforces a strict **Visuo-Physiological Co-Existence Constraint**:
   $$\left(\sum_{v \in \{\text{RGB}, \text{TIR}\}} a_v \ge 1\right) \quad \text{AND} \quad \left(\sum_{p \in \{\text{BP}, \text{Resp}\}} a_p \ge 1\right)$$
   Any dropout state violating this condition (e.g., pure-video batches or pure-physio batches) is automatically revoked by the engine's fallback safeguard. Note this constrains **presence and supervision**, not visibility: an active stream may still be fully masked on a given step (§3.2.3), so cross-modal attention is guaranteed in expectation across steps rather than in every individual step.
3. **Decoupled 4-Modality Streams ($N=4$)**:
   Separating continuous Blood Pressure (BP) and Respiration into two independent 1-channel streams (instead of a combined 2-channel stream) enables the Vision Transformer to learn both **Intra-Physio reconstruction** ($\text{BP} \leftrightarrow \text{Respiration}$) and clean biophysical cross-modal attention pairings: $(\text{RGB} \leftrightarrow \text{BP})$ for cardiac rPPG pulses and $(\text{TIR} \leftrightarrow \text{Respiration})$ for nostril thermal airflow.
4. **Video 25% Kept Ratio Upper Cap ($r_{\text{video}} \le 0.25$)**:
   As established in VideoMAE, 3D video streams exhibit extreme temporal redundancy, rendering mask ratios of **75% to 95%+** highly effective for learning spatiotemporal features without quality loss. Even if raw Dirichlet sampling assigns high weight to visual streams (e.g., $\lambda_{\text{RGB}} = 0.70$), visual streams are strictly capped at a **maximum kept ratio of 25%** ($r_{\text{video}} = \min(\lambda_{\text{video}}, 0.25)$). This enforces a minimum **75% mask ratio for video modalities**, suppressing spatiotemporal shortcuts.
5. **Uncapped 1D Physiological Waveforms ($r_{\text{physio}} \le 1.00$)**:
   Unlike video streams, 1D physiological continuous waveforms (BP and Respiration) contain delicate temporal landmarks (systolic peaks, dicrotic notches, respiratory phase) across only 25 temporal tokens ($T=25$). Physiological streams carry **no upper cap limit** and can be kept up to 100% ($r_{\text{physio}} = \lambda_{\text{physio}} \in [0.0, 1.0]$).
6. **Pure Visible Tokens in ViT Encoder & Dense Mini-Batch Execution**:
   The ViT encoder receives **only unmasked, surviving signal patches** (`[B, N_batch, D]`), completely eliminating the quadratic self-attention FLOPs and VRAM overhead of filler `[MASK]` placeholder vectors during pre-training. All samples within a single mini-batch share an identical Dirichlet draw, guaranteeing **100% CUDA warp utilization**. The visible-token count per modality is therefore **constant within a batch and free to change between batches**: only the COUNT is shared across the samples of a step, never the mask placement (§5.5).

---

## 2. Geometry & Input Stream Specifications ($N=4$)

* **RGB Video Stream**:
  * **Crop Region**: Facial area ($128 \times 128$ px, $3$ channels).
  * **Temporal Decimation**: Decimated to 12.5 FPS ($100$ frames across an 8-second clip).
  * **Tubelet Geometry**: $P_t = 4$ frames, $P_h = 16$ px, $P_w = 16$ px ($T=25$ steps).
  * **Spatial Patches ($S_{\text{RGB}}$)**: $8 \times 8 = 64$ spatial patches ($1600$ raw tokens).
  * **Max Kept Patches**: Capped at $25\% \times 64 = \mathbf{16 \text{ spatial patches}}$ ($\mathbf{400 \text{ tokens}}$ max).

* **Thermal Infrared (TIR) Video Stream**:
  * **Crop Region**: Mouth + Nose ROI ($96 \times 96$ px, $3$ pseudo-color channels).
  * **Temporal Decimation**: Decimated to 12.5 FPS ($100$ frames across an 8-second clip).
  * **Tubelet Geometry**: $P_t = 4$ frames, $P_h = 16$ px, $P_w = 16$ px ($T=25$ steps).
  * **Spatial Patches ($S_{\text{TIR}}$)**: $6 \times 6 = 36$ spatial patches ($900$ raw tokens).
  * **Max Kept Patches**: Capped at $25\% \times 36 = \mathbf{9 \text{ spatial patches}}$ ($\mathbf{225 \text{ tokens}}$ max).

* **1D Continuous Blood Pressure (BP) Waveform**:
  * **Sampling Rate**: 100 Hz ($800$ samples across 8s, $1$ channel).
  * **1D Conv Projection**: Stride = 32, Kernel = 32 ($320\text{ ms}$ temporal window).
  * **Temporal Token Steps ($T$)**: $25$ steps ($25$ raw tokens).
  * **Max Kept Tokens**: Uncapped up to $100\% = \mathbf{25 \text{ tokens}}$ max.

* **1D Continuous Respiration (Resp) Waveform**:
  * **Sampling Rate**: 100 Hz ($800$ samples across 8s, $1$ channel).
  * **1D Conv Projection**: Stride = 32, Kernel = 32 ($320\text{ ms}$ temporal window).
  * **Temporal Token Steps ($T$)**: $25$ steps ($25$ raw tokens).
  * **Max Kept Tokens**: Uncapped up to $100\% = \mathbf{25 \text{ tokens}}$ max.

---

## 3. Mathematical Formulation of AMMM with Co-Existence Safeguard

Let $\mathcal{M} = \{\text{RGB}, \text{TIR}, \text{BP}, \text{Resp}\}$ be the **superset** of input modalities, partitioned into visual streams $\mathcal{V} = \{\text{RGB}, \text{TIR}\}$ and physiological streams $\mathcal{P} = \{\text{BP}, \text{Resp}\}$. A run declares a non-empty subset $\mathcal{M}_{\text{decl}} \subseteq \mathcal{M}$ with $\mathcal{V}_{\text{decl}} = \mathcal{M}_{\text{decl}} \cap \mathcal{V}$ and $\mathcal{P}_{\text{decl}} = \mathcal{M}_{\text{decl}} \cap \mathcal{P}$; every rule below is stated over $\mathcal{M}_{\text{decl}}$, so a two-stream run (e.g. $\{\text{TIR}, \text{Resp}\}$, or $\{\text{RGB}, \text{BP}\}$) is the *same* rule with a smaller index set.

**Declared-set precondition.** $|\mathcal{V}_{\text{decl}}| \ge 1$ **and** $|\mathcal{P}_{\text{decl}}| \ge 1$. A run that declares only visual or only physiological streams is INVALID and MUST be rejected at construction time, before any sampling. This is a precondition, **not** something the fallback of §3.1.3 repairs: a fallback that must re-activate a stream of a kind the run does not contain has no valid target and would fail at runtime (index error / undefined behaviour) instead of at configuration time.

**Presence is not visibility.** $a_m = 1$ ("active") means the modality is part of the step and has a loss term; it does **not** imply $K_m \ge 1$. Every modality may be $100\%$ masked ($K_m = 0$, §3.2.3), so the co-existence safeguard of §3.1 guarantees cross-modal *supervision* is possible, not that every step contains a visible token of each kind. A fully masked stream is a legal, supported state.

### 3.1 Availability Vector & Fallback Safeguard Rule

1. **Batch-Level Modality Availability Vector ($\vec{a}_{\text{batch}}$)**:
   $$\vec{a}_{\text{batch}} = (a_m)_{m \in \mathcal{M}_{\text{decl}}} \in \{0, 1\}^{|\mathcal{M}_{\text{decl}}|}$$
   One vector is drawn per training **step** and shared by every sample of the mini-batch. A per-sample availability vector is explicitly **not** used: it would make the visible-token count differ between samples, which the dense-gather contract (§5.5) forbids. Requirement 2 of §1 ("all samples share one Dirichlet draw") and this vector therefore have the same cardinality: the step.

2. **Visuo-Physiological Co-Existence Predicate ($\mathcal{C}(\vec{a})$)**, over the declared subsets:
   $$\mathcal{C}(\vec{a}) = \left(\sum_{v \in \mathcal{V}_{\text{decl}}} a_v \ge 1\right) \land \left(\sum_{p \in \mathcal{P}_{\text{decl}}} a_p \ge 1\right)$$

3. **Fallback Revocation Mechanism**:
   If $\mathcal{C}(\vec{a}_{\text{batch}}) = \mathbf{False}$, the engine forms $\vec{a}_{\text{valid}}$ by re-activating the **first stream of the missing kind in the declared order** (this target provably exists, by the declared-set precondition above):
   * **If no visual stream is present** ($\sum_{v \in \mathcal{V}_{\text{decl}}} a_v = 0$): force the first $v \in \mathcal{V}_{\text{decl}}$ to $1$ (RGB when declared, otherwise TIR).
   * **If no physiological stream is present** ($\sum_{p \in \mathcal{P}_{\text{decl}}} a_p = 0$): force the first $p \in \mathcal{P}_{\text{decl}}$ to $1$ (BP when declared, otherwise Resp).
   * **No modality name is ever forced.** RGB / BP appear above only because they are FIRST in the superset's declared order; a `tir,resp` run re-activates `tir` / `resp` and never introduces a stream the run does not contain.
   * **Consequence for a single-member kind**: if $|\mathcal{V}_{\text{decl}}| = 1$, that stream can never be dropped (the fallback reinstates it), so modality dropout only varies a kind that has more than one member. This must be logged once at construction so the effective distribution is visible.

### 3.2 Dirichlet Proportion Sampling & Asymmetric Mapping

1. **Dirichlet Raw Proportion Sampling** (over the DRAWN streams only):
   Let $\mathcal{M}_{\text{overridden}} \subseteq \mathcal{M}_{\text{active}}$ be the streams whose kept ratio is FIXED by configuration (the experiment knob `kept_ratio_override`), and let the remainder be the drawn set:
   $$\mathcal{M}_{\text{drawn}} = \mathcal{M}_{\text{active}} \setminus \mathcal{M}_{\text{overridden}}$$
   $$\vec{\lambda} = (\lambda_m)_{m \in \mathcal{M}_{\text{drawn}}} \sim \text{Dir}(\vec{\alpha}_{\text{drawn}}) \quad \text{where } \alpha_m = 1.0 \;\; \forall m, \quad \sum_{m \in \mathcal{M}_{\text{drawn}}} \lambda_m = 1.0$$
   The simplex constraint holds on $\mathcal{M}_{\text{drawn}}$ **only**: an overridden stream contributes its own fixed $r_m$ outside the draw, so the total kept fraction of a step is not bounded by $1$ when overrides are used. An overridden ratio is used verbatim — never rescaled by the draw — subject to $0 \le r_m \le \operatorname{cap}_m$ (a violation is a configuration error, checked at construction, not silently clamped).

   **Degenerate draw ($|\mathcal{M}_{\text{drawn}}| = 1$).** A one-dimensional Dirichlet is a point mass at $1.0$, which would silently pin the sole free stream to its maximum kept ratio on every step. That stream is therefore drawn from $\text{Uniform}(0,1)$ instead — the $\text{Beta}(1,1)$ marginal — so its ratio keeps varying, and the substitution is logged once at construction.

   **Empty draw ($|\mathcal{M}_{\text{drawn}}| = 0$).** Every active stream is pinned, so no distribution is sampled and every $r_m$ is the configured value; the global-visibility guard of §3.2.3 then applies.
2. **Asymmetric Kept Ratio Mapping ($r_m$)**, with $u_m$ the drawn $\lambda_m$ or the pinned override (used verbatim):
   * **For 3D Video Streams ($m \in \{\text{RGB}, \text{TIR}\}$)**:
     $$r_m = \min(u_m, 0.25)$$
   * **For 1D Physiological Waveforms ($m \in \{\text{BP}, \text{Resp}\}$)**:
     $$r_m = u_m \in [0.0, 1.0]$$
3. **Discrete Token / Patch Conversion** (over the ACTIVE streams $\mathcal{M}_{\text{active}} = \{m : a_m = 1\}$, with $\operatorname{clamp}(x, lo, hi) = \max(lo, \min(x, hi))$ and lower bound $0$; `min_kept` raises it for **drawn** streams only, see below):
   * $S_{\text{RGB}} = \operatorname{clamp}(\lfloor r_{\text{RGB}} \cdot 64 \rceil, 0, 16) \implies K_{\text{RGB}} = S_{\text{RGB}} \times 25 \le 400 \text{ tokens}$
   * $S_{\text{TIR}} = \operatorname{clamp}(\lfloor r_{\text{TIR}} \cdot 36 \rceil, 0, 9) \implies K_{\text{TIR}} = S_{\text{TIR}} \times 25 \le 225 \text{ tokens}$
   * $K_{\text{BP}} = \operatorname{clamp}(\lfloor r_{\text{BP}} \cdot 25 \rceil, 0, 25) \text{ tokens}$
   * $K_{\text{Resp}} = \operatorname{clamp}(\lfloor r_{\text{Resp}} \cdot 25 \rceil, 0, 25) \text{ tokens}$

   **$K_m = 0$ is legal for every modality** (i.e. a $0\%$ kept / $100\%$ masked stream): the whole stream becomes reconstruction target with zero encoder tokens. The decoder then sees only that stream's mask tokens plus their positional embeddings, so its prediction is a positional prior and the stream contributes **no gradient to the shared encoder on that step** (its adapter is unused too). This is an accepted state, not an error; it is logged as `kept_<m> = 0` / `masked_<m> = n_m` so a run that keeps doing it is visible in the log rather than silently degraded.

   **Upper bound is kind-specific**: $S_{\text{RGB}} \le 16$ and $S_{\text{TIR}} \le 9$ are the 25 % kept caps of a 64- / 36-patch spatial grid, whereas $K_{\text{BP}}, K_{\text{Resp}} \le 25$ is only the token count of the stream (i.e. no physiological cap — a physio stream may also keep 100 %). A stream with $K_m = |\text{units}_m|$ has **no masked position**: its masked-MSE term is exactly $0$ and MUST be reported as an explicit $0$, never by omitting the key (§5.9).

   **Global-visibility guard (required).** Even though a single stream may be invisible, the mini-batch must not be: the shared encoder is fed the concatenation of all visible tokens, so $\sum_{m} K_m = 0$ yields an empty `[B, 0, D]` sequence. The engine MUST guarantee $\sum_{m \in \mathcal{M}_{\text{active}}} K_m \ge 1$ by granting one unit when every draw rounds to zero. The recipient is chosen by a rule that is TOTAL (it never depends on a quantity that may not exist): largest $\lambda_m$; if a stream's proportion is pinned by a configured override instead of a draw, rank by that pinned $r_m$; if no comparable number exists for any active stream (all pinned to an equal value), take the FIRST active stream in declared order. This is a degenerate-case safeguard for the encoder input, **not** a per-stream floor: it fires only when the entire step has nothing to encode. In addition, the forward pass MUST assert the concatenated length $\sum_m k_m \ge 1$ before the encoder (a cheap runtime check that also catches a hand-built mask), so the invariant cannot be violated by bypassing the sampler.

   **Optional strict variant.** A run that wants the historical "every active stream reaches the encoder" premise can set `min_kept: 1`; the default is `0` (per-stream masking may be total). `min_kept` applies to **drawn** streams only: an explicitly overridden stream is exempt, so `kept_ratio_override: {tir: 0.0}` means 0 % kept even when `min_kept: 1` — an ablation must never be silently overridden.

### Maximum ViT Encoder Upper Bound ($N_{\text{batch}}^{\max}$) for the §2 geometry
$$N_{\text{batch}}^{\max} = K_{\text{RGB}}^{\max} + K_{\text{TIR}}^{\max} + K_{\text{BP}}^{\max} + K_{\text{Resp}}^{\max} = 400 + 225 + 25 + 25 = \mathbf{675 \text{ tokens}}$$

This is a property of the $G_t = 25$ configuration (tubelet_t 4 @ 12.5 fps), not of the engine — see invariant 3 in §5.

---

## 4. Illustrative PyTorch Implementation

> The sketch below hardcodes the **4-stream superset** (`N=4`, fixed indices
> `[0..3]`, fixed 64/36/25 unit counts). It illustrates the mapping, **not** the
> engine: the normative rule is §3, which is stated over
> $\mathcal{M}_{\text{decl}}$ and must not assume which streams exist, how many
> there are, or their unit counts. Do not port the index arithmetic.

```python
# Not as implementation reference

import torch
import torch.nn as nn
from typing import Dict, Tuple

class AsymmetricMultiModalMaskingEngineV15(nn.Module):
    """
    # Not as implementation reference
    
    Asymmetric Multi-Modal Masking Engine (AMMM v15)
    
    Features:
    - Visuo-Physiological Co-Existence Safeguard: Requires >= 1 Video AND >= 1 Physio signal.
    - 3D Video Modalities (RGB & TIR): Kept ratio capped at <= 25% (r_video <= 0.25, i.e. >= 75% masked).
    - 1D Physio Modalities (BP & Resp): Uncapped up to 100% (r_physio <= 1.00).
    - ViT Encoder Input: Pure visible tokens [B, N_batch, D] (Max 675 tokens).
    - Execution: Batch-level sampling for dense rectangular GPU execution.
    """
    def __init__(
        self,
        T: int = 25,
        S_rgb_max: int = 64,   # 8x8 spatial patches
        S_tir_max: int = 36,   # 6x6 spatial patches
        video_cap_ratio: float = 0.25, # 25% kept max (75% masked min)
        alpha: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    ):
        super().__init__()
        self.T = T
        self.S_rgb_max = S_rgb_max
        self.S_tir_max = S_tir_max
        self.video_cap_ratio = video_cap_ratio
        self.alpha_base = torch.tensor(alpha)

    def _enforce_visuo_physio_constraint(self, active_modalities: torch.Tensor) -> torch.Tensor:
        """
        # Not as implementation reference
        Enforces that active_modalities has at least 1 video signal AND at least 1 physio signal.
        active_modalities order: [RGB, TIR, BP, Resp]
        """
        valid_a = active_modalities.clone()
        has_video = (valid_a[0] > 0) or (valid_a[1] > 0)
        has_physio = (valid_a[2] > 0) or (valid_a[3] > 0)
        
        # Fallback 1: If no video is present, force RGB active
        if not has_video:
            valid_a[0] = 1.0
            
        # Fallback 2: If no physio is present, force BP active
        if not has_physio:
            valid_a[2] = 1.0
            
        return valid_a

    def forward(
        self, 
        active_modalities: torch.Tensor, # Shape [4] tensor of 1s and 0s (RGB, TIR, BP, Resp)
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        # Not as implementation reference
        
        # 1. Enforce Visuo-Physiological Co-Existence Safeguard
        valid_a = self._enforce_visuo_physio_constraint(active_modalities)
        
        # 2. Mask Dirichlet alpha by valid sensor availability
        alpha_masked = self.alpha_base.to(device) * valid_a.to(device)
        active_indices = torch.where(alpha_masked > 0)[0]
        
        dirichlet = torch.distributions.Dirichlet(alpha_masked[active_indices])
        sampled_l = dirichlet.sample()
        
        lambdas = torch.zeros(4, device=device)
        lambdas[active_indices] = sampled_l
        
        # 3. Apply Asymmetric Mapping: 25% Video Upper Cap vs. Uncapped Physio
        r_rgb = min(lambdas[0].item(), self.video_cap_ratio)
        r_tir = min(lambdas[1].item(), self.video_cap_ratio)
        r_bp = lambdas[2].item()   # Uncapped
        r_resp = lambdas[3].item() # Uncapped
        
        # 4. Convert to Discrete Patch / Token Counts
        S_rgb_kept = min(int(round(r_rgb * self.S_rgb_max)), int(self.video_cap_ratio * self.S_rgb_max)) # Max 16
        S_tir_kept = min(int(round(r_tir * self.S_tir_max)), int(self.video_cap_ratio * self.S_tir_max)) # Max 9
        K_bp_kept = int(round(r_bp * self.T))     # Max 25
        K_resp_kept = int(round(r_resp * self.T)) # Max 25
        
        # 5. Spatiotemporal Tubelet Patch Gathering (3D Video)
        rgb_noise = torch.rand(self.S_rgb_max, device=device)
        rgb_spatial_idx = torch.topk(rgb_noise, S_rgb_kept, largest=False).indices
        rgb_keep_idx = (rgb_spatial_idx.unsqueeze(-1) * self.T + torch.arange(self.T, device=device)).flatten()
        
        tir_noise = torch.rand(self.S_tir_max, device=device)
        tir_spatial_idx = torch.topk(tir_noise, S_tir_kept, largest=False).indices
        tir_keep_idx = (tir_spatial_idx.unsqueeze(-1) * self.T + torch.arange(self.T, device=device)).flatten()
        
        # 6. Temporal Sequence Gathering (1D Waveforms)
        bp_noise = torch.rand(self.T, device=device)
        bp_keep_idx = torch.topk(bp_noise, K_bp_kept, largest=False).indices
        
        resp_noise = torch.rand(self.T, device=device)
        resp_keep_idx = torch.topk(resp_noise, K_resp_kept, largest=False).indices
        
        total_visible_tokens = len(rgb_keep_idx) + len(tir_keep_idx) + len(bp_keep_idx) + len(resp_keep_idx)
        
        return {
            "r_kept": (r_rgb, r_tir, r_bp, r_resp),
            "rgb_keep_idx": rgb_keep_idx,
            "tir_keep_idx": tir_keep_idx,
            "bp_keep_idx": bp_keep_idx,
            "resp_keep_idx": resp_keep_idx,
            "N_batch": total_visible_tokens
        }
```

---

## 5. Verification & Acceptance Protocols

1. **Visuo-Physiological Co-Existence Invariant** — a **presence** invariant (an active stream may be fully masked, §3.2.3), stated over the declared subsets so it holds for `tir,resp` as well as for the 4-stream superset:
   $$\forall \text{ steps } t, \quad \Big(\sum_{v \in \mathcal{V}_{\text{decl}}} a_v(t) \ge 1\Big) \;\land\; \Big(\sum_{p \in \mathcal{P}_{\text{decl}}} a_p(t) \ge 1\Big)$$
   The corresponding *visibility* guarantee is deliberately weaker and applies to the step as a whole, not per kind (invariant 7).
2. **25% Video Upper Cap Invariant** (for the design's 128 / 96 geometry; another geometry has its own numbers):
   $$\forall \vec{\lambda}, \quad r_{\text{RGB}} \le 0.25 \quad \text{and} \quad r_{\text{TIR}} \le 0.25$$
   enforcing $S_{\text{RGB}} \le 16$ spatial patches and $S_{\text{TIR}} \le 9$ spatial patches (guaranteeing $\ge 75\%$ mask ratio).
3. **Encoder Sequence Cap for the §2 Geometry** (NOT an engine invariant):
   For the 4-stream superset at $T = 25$ the ViT encoder tensor length is bounded by:
   $$N_{\text{batch}} \le K_{\text{RGB}}^{\max} + K_{\text{TIR}}^{\max} + K_{\text{BP}}^{\max} + K_{\text{Resp}}^{\max} = \mathbf{675 \text{ tokens}}$$
   This number belongs to the $G_t = 25$ / 128-96 configuration of §2, not to the engine: with $G_t = 50$ the same 16 kept spatial patches are 800 rgb tokens. For any other configured geometry the bound is recomputed as
   $$N_{\text{batch}}^{\max} = \sum_{v \in \mathcal{V}} \lfloor \operatorname{cap}_v \cdot |\text{spatial}_v| \rfloor \cdot G_t \; + \; \sum_{p \in \mathcal{P}} |\text{units}_p|$$
   The engine-level invariants are only 6 (per-stream cap) and 7 (non-empty encoder input); a run with $G_t = 50$ legitimately has a larger bound.
4. **Pure-Visible-Token Encoder Invariant**:
   The ViT encoder tensor shape is strictly `[B, N_batch, D]` containing **zero `[MASK]` tokens**.
5. **Dense Mini-Batch Invariant** (constant *within* a batch, free *across* batches):
   $$\forall \text{ step } t, \; \forall b, b' \in \text{batch}_t, \quad K_m^{(t)}(b) = K_m^{(t)}(b') = K_m^{(t)} \quad\Longrightarrow\quad N_{\text{batch}}^{(t)} = \sum_m K_m^{(t)}$$
   Every sample of one mini-batch has the SAME visible-token count per modality, so the encoder input is a dense `[B, N_batch, D]` tensor and no sample is padded — that is what preserves 100 % CUDA warp utilization. The count is a property of the **step**, not of the run: $K_m^{(t)}$ and therefore $N_{\text{batch}}^{(t)}$ may differ from one step to the next (the Dirichlet draw changes). What is shared within a batch is the **count only**; the *placement* of the kept units stays independently random per sample, otherwise every sample in a batch would be masked identically and the step would carry the statistical weight of one sample.
6. **Per-Stream Masking Range Invariant**:
   $$\forall m \in \mathcal{M}_{\text{active}}, \quad 0 \le K_m \le \operatorname{cap}_m \cdot |\text{units}_m|$$
   with the cap taken as $0.25$ for a visual stream and $1.0$ for a physiological stream. Both extremes are legal: $K_m = 0$ ($100\%$ masked) and $K_m = |\text{units}_m|$ ($0\%$ masked) — the latter has no masked position, so its masked-MSE term is exactly $0$ and MUST be reported as an explicit $0$, never by omitting the key (invariant 9).
7. **Global Visibility Invariant** (the concatenated encoder input is never empty):
   $$\sum_{m \in \mathcal{M}_{\text{active}}} K_m \ge 1$$
   so the encoder is never fed an empty `[B, 0, D]` sequence; the engine grants one unit if every draw rounds to zero — recipient = largest $\lambda_m$, else the largest pinned override $r_m$, else the first active stream in declared order (see §3.2.3) — and `forward` asserts the realised concatenated length is $\ge 1$. A *single* modality may still be $K_m = 0$.
8. **Declared-Set Validity Invariant**:
   $$|\mathcal{V}_{\text{decl}}| \ge 1 \;\land\; |\mathcal{P}_{\text{decl}}| \ge 1$$
   checked at construction time; an invalid declared set raises instead of relying on the fallback.
9. **Loss-Key Stability Invariant** (DDP safety):
   Every declared stream reports its loss key on **every** step and on **every** rank — value $0$ when the stream is inactive or has no masked position, plus its masked-token count. The metric logger synchronises meters with one collective per key (`MetricLogger.synchronize_between_processes`), so a key set that depends on the step or the rank desynchronises the collectives. Keys are a function of the **configuration only**.
