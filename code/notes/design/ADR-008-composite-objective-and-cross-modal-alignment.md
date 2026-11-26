# ADR-008: Composite Waveform Objective and Cross-Modal InfoNCE Alignment for Stage-2 Pre-Training (`tir`, `resp`)

- **Status**: Implemented (Pending Eval)
- **Date**: 2026-11-25
- **Author**: KISMED MA research team
- **Related Spec/Plan**: `code/notes/plan/features/ADR-008-composite-objective-and-cross-modal-alignment-plan.md` (build order and conformance gates), `code/notes/design/ADR-007-two-stream-anatomy-aware-constant-budget-masking.md` (companion two-stream masking, geometry, and curriculum specification that this ADR builds upon), `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md` (normative for the headline downstream metric, respiratory passband, and subject-level aggregation used in §3), `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md` (normative for subject-disjoint 5-fold partitioning and validation-only model selection), `code/notes/GLOSSARY.md` (canonical terminology). All objective definitions, masking semantics, alignment equations, feasibility rules, and acceptance criteria in this ADR are self-contained.
- **Supersedes**: `N/A` — extends Stage-2 pre-training on top of `code/notes/design/ADR-007-two-stream-anatomy-aware-constant-budget-masking.md` by replacing pure masked time-domain MSE on physiological streams with a composite time-correlation-spectral objective and introducing a Mode-2-gated cross-modal InfoNCE latent regularizer, while leaving ADR-007's masking policy, geometry, and curriculum unchanged.

**Normative status.** This ADR is the self-contained architectural specification for the Stage-2 composite physiological waveform objective and cross-modal latent alignment regularizer. Every mathematical definition, masking rule, numerical precision invariant, distributed-training contract, feasibility precondition, observability requirement, logical parameter, and acceptance criterion is stated independently of any software implementation, module layout, runner script, or configuration file, so the decision survives arbitrary rewrites of the training stack.

---

## 1. Context & Problem Statement

### 1.1 Objective and scope

Stage-2 masked multimodal pre-training under the two-stream ADR-007 policy (`tir`, `resp`, fixed $N = 200$ visible encoder tokens, anatomy-aware stratified spatial tube sampling) aims to learn a shared representation from which thermal-infrared facial ROI video (`tir`) alone can reconstruct the continuous respiration waveform (`resp`) of an unseen subject under complete contact-sensor absence (**Mode 1: pure virtual sensing**, $K_{\text{resp}} = 0$).

ADR-007 resolves multi-stream interference, nasal-plume spatial blindspots, and token-budget oversizing, but retains pure masked time-domain Mean Squared Error (MSE) as the sole Stage-2 reconstruction objective (with physiological loss weight $\lambda_{\text{resp}} = 0.5$ against visual weight $1.0$). This ADR addresses the optimization-landscape deficiencies that cause Mode-1 virtual sensing to collapse under pure masked MSE.

The downstream scientific headline metric, its respiratory passband ($0.10\text{--}0.60\text{ Hz}$), its session-assembly and per-session $\to$ per-subject aggregation, and its bootstrap confidence interval are governed by `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md` and are not redefined here.

**In scope.** The unified Stage-2 pre-training objective; the three-term composite respiration waveform loss ($\mathcal{L}_{\text{MSE}}$, $\mathcal{L}_{\text{Pearson}}$, $\mathcal{L}_{\text{spectral}}$) and its Mode-1 / Mode-2 masking semantics; the Mode-2-gated symmetric cross-modal InfoNCE latent regularizer ($\mathcal{L}_{\text{align}}$) and its anatomy-weighted pooling operator; physiological loss-weight rebalancing; numerical precision, degenerate-target, and distributed-training invariants; backward-compatibility neutrality; observability requirements; and pre-registered falsifiable acceptance criteria.

**Out of scope.** Specific source modules, class hierarchies, CLI plumbing, or configuration file schemas; token-budget apportionment, spatial stratification, and curriculum schedules (governed verbatim by ADR-007); raw data curation, ROI extraction, and offline caching; and Stage-3 fine-tuning architectures or losses.

### 1.2 Measured repository evidence vs. theoretical diagnosis

Empirical measurements and theoretical hypotheses are kept strictly separate below.

**Measured predecessor context (evidence):**
1. **Four-stream Stage-2 pure-MSE collapse (`code/notes/eval/baselines/tir_rgb2d_bp_resp/resp_recon_arms_ab.md`):** On Fold 0 (3 held-out test subjects, 199 clips), frozen zero-shot Mode-1 reconstruction ($K_{\text{resp}} = 0$) scored `r_masked` mean **`+0.0062`** (TIR-max, `mse = 1.0254 z²`) and **`+0.0140`** (RGB2D-max, `mse = 1.0218 z²`), with symmetric per-clip correlation tails — the signature of flat conditional-mean collapse ($\hat{y} \approx 0$).
2. **Three-stream Stage-2 pure-MSE inpainting-vs-virtual-sensing gap (`code/notes/eval/baselines/tir_rgb2d_resp/README.md`):** On Fold 0 (epoch 149, ViT-Base with pretrained video-MAE initialization), the exact same encoder-decoder achieved **`r_masked = +0.996`** and **`MSE = 0.009 z²`** in Mode 2 when $50\%$ of `resp` tokens were visible (125 of 250 tokens, 210 clips), yet dropped in Mode 1 (1,594 clips / 27 test subjects) to **`r_masked = +0.109`** (subject median **`+0.113 [+0.078, +0.136]`**) with **`MSE = 1.041 z²`** under TIR-max (and `+0.080`, subject median `+0.092 [+0.071, +0.108]`, `MSE = 1.059 z²` under RGB2D-max). An `MSE ≈ 1.04 z²` on unit-variance z-scored targets confirms heavy amplitude shrinkage toward zero.
3. **Stage-3 composite-loss feasibility (`code/notes/eval/tir_resp/resp_5fold_cv_96respgrid.md`, historical off-protocol ADR-001 record):** Supervised TIR-only Stage-3 fine-tuning trained with a composite time-correlation-spectral objective achieved a pooled 5-fold subject-level band-limited Pearson median of **`+0.542 [+0.493, +0.576]`** against a **`+0.537`** single-tone template baseline (rate MAE `3.75 bpm [2.81, 4.34]`), proving that strong respiratory information exists in the 96 px TIR ROI.
4. **ADR-007 pure-MSE baseline status:** No archived Fold-0 evaluation artifact exists yet under `code/notes/eval/` for the pure-MSE ADR-007 two-stream arm; that matched pure-MSE ADR-007 run serves as the primary direct comparator for ADR-008.

**Mathematical diagnosis (hypotheses motivating this ADR):**
The extreme gap between Mode-2 inpainting ($r = +0.996$) and Mode-1 virtual sensing ($r \in [+0.006, +0.113]$, $\text{MSE} \approx 1.02\text{--}1.04\,z^2$) demonstrates that neither encoder capacity nor decoder width is the binding bottleneck. Instead, pure masked MSE suffers from two structural deficiencies in the cross-modal optimization landscape:

#### 1.2.1 The conditional mean trap under phase uncertainty
Let $y \in \mathbb{R}^{L}$ ($L = 8000$ samples over $8.0\text{ s}$ at $F_s = 1000\text{ Hz}$ under ADR-007 geometry) be a continuous, per-clip z-scored ground-truth respiration waveform ($\mathbb{E}[y] = 0$, $\text{Var}(y) = 1$). Modeling a representative breathing cycle as a unit-variance oscillation:
$$y(t) = \sqrt{2}\sin(\omega_0 t + \phi), \qquad \omega_0 = 2\pi f_0, \quad f_0 \in [0.10, 0.60]\text{ Hz}, \quad \phi \sim \text{Uniform}(-\pi, \pi).$$
In Mode 1 ($K_{\text{resp}} = 0$), the respiration decoder receives only masked queries conditioned on the encoded thermal latents $z_{\text{tir}}$. Because convective airflow heat transfer at the nostril boundary exhibits physiological thermal inertia and contact-belt phase lag, visual estimation carries temporal phase uncertainty $\Delta\phi = \hat{\phi} - \phi$. Under pure time-domain MSE, $\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{L}\sum_{t=1}^{L}(y_t - \hat{y}_t)^2$:

- **Strategy A — Oscillatory prediction with phase offset $\Delta\phi$** ($\hat{y}(t) = \sqrt{2}\sin(\omega_0 t + \phi + \Delta\phi)$):
  $$\mathbb{E}_{\phi}[\mathcal{L}_{\text{MSE}}] = 2 - 2\cos(\Delta\phi),$$
  which evaluates to **$2.0$** under uniform phase uncertainty $\Delta\phi \sim \text{Uniform}(-\pi, \pi)$ (and up to $4.0$ when out of phase, $\Delta\phi = \pi$).
- **Strategy B — Conditional dataset mean (trivial flat zero line)** ($\hat{y}(t) \equiv 0$):
  $$\mathbb{E}_{\phi}[\mathcal{L}_{\text{MSE}}] = \text{Var}(y) = \mathbf{1.0}.$$

Because $1.0 < 2 - 2\cos(\Delta\phi)$ whenever $\cos(\Delta\phi) < 0.5$ ($|\Delta\phi| > 60^\circ$), **pure time-domain MSE actively penalizes predicting an oscillatory waveform whenever phase confidence drops below $60^\circ$**. The global minimizer under weak early-epoch conditioning is the conditional expectation $\hat{y} \equiv \mathbb{E}[y \mid z_{\text{tir}}] = 0$, explaining the empirically observed $\text{MSE} \approx 1.02\text{--}1.04\,z^2$.

#### 1.2.2 Cross-modal gradient dilution and loss-weight asymmetry
Under the baseline two-stream objective $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{tir}} + \lambda_{\text{resp}}\mathcal{L}_{\text{resp}}$ with $\lambda_{\text{resp}} = 0.5$, the shared encoder latents $z \in \mathbb{R}^{N \times D}$ ($N = 200, D = 768$) receive gradients from two highly asymmetric reconstruction paths:
- **Visual reconstruction path (`tir`):** With $C = 3$ false-colour channels, $G_t = 25$ temporal tubelets per spatial tube, $36$ spatial tubes, and $(2, 16, 16)$ tubelets ($3 \times 2 \times 16 \times 16 = 1536$ scalars per tubelet), each step decodes $(36 - S_{\text{tir}}) \times 25 \in [700, 800]$ masked tubelets — **$1{,}075{,}200\text{--}1{,}228{,}800$ scalar pixel outputs** backed by strong same-modality spatial correlation across neighbouring thermal patches.
- **Physiological reconstruction path (`resp`):** Each step decodes $n_{\text{masked}} \in [150, 250]$ tokens $\times$ $32$ samples/token — **$4{,}800\text{--}8{,}000$ scalar outputs**, further down-weighted by $\lambda_{\text{resp}} = 0.5$.

Even with mean reduction per modality, visual self-reconstruction dominates the gradient flowing into the shared encoder ($\|\nabla_{z_{\text{tir}}}\mathcal{L}_{\text{tir}}\|_2 \gg \|\nabla_{z_{\text{tir}}}(\lambda_{\text{resp}}\mathcal{L}_{\text{resp}})\|_2$), especially once `resp` predictions settle near the flat conditional-mean stationary point.

### 1.3 Requirements

The decision in §2 satisfies seven verifiable architectural requirements:
- **R1 — Collapse-resistant physiological objective.** On Mode 1 ($K_{\text{resp}} = 0$), a flat zero prediction ($\hat{y} \equiv 0$) must incur high correlation loss ($\mathcal{L}_{\text{Pearson}} = 1.0$) with repelling variance gradients and high in-band spectral loss ($\mathcal{L}_{\text{spectral}} \approx 1.0$), eliminating the flat-line local minimum.
- **R2 — Exact Mode-1 and Mode-2 masking semantics.** Neither $\mathcal{L}_{\text{Pearson}}$ nor $\mathcal{L}_{\text{spectral}}$ may be corrupted by unmasked placeholder zeros during Mode 2 ($K_{\text{resp}} > 0$); gradients must flow exclusively through masked prediction tokens.
- **R3 — Direct cross-modal latent alignment.** On Mode-2 steps ($K_{\text{resp}} > 0 \land K_{\text{tir}} > 0$), a symmetric contrastive regularizer must align pooled `tir` and `resp` encoder representations prior to the decoders, injecting direct supervisory gradients from respiratory dynamics into the thermal visual encoder blocks.
- **R4 — Numerical precision, degenerate-target, and distributed safety.** Correlation, spectral rFFT, and contrastive normalization must operate in 32-bit floating point (FP32) with degenerate-target guards; Mode-1 gating of alignment must preserve loss-key stability and distributed parameter-reduction safety across ranks.
- **R5 — Strict backward compatibility (neutral defaults).** Every logical parameter introduced by this ADR must have a neutral default that recovers pure masked MSE with zero extra model parameters and identical behaviour for prior arms.
- **R6 — Downstream transfer isolation.** Auxiliary alignment projection heads belong solely to Stage-2 pre-training and must be cleanly excluded when transferring the pretrained visual encoder to Stage-3 downstream regression.
- **R7 — Full observability and zero hidden constants.** All loss weights, passband bounds, FFT length rules, projection dimensions, temperatures, pooling weights, and diagnostic cadences must be explicitly parameterized, logged, and recorded in the run manifest.

---

## 2. Decision & Parameter Exposure

**Adopt a three-term Composite Respiration Waveform Objective ($\mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{Pearson}} + \mathcal{L}_{\text{spectral}}$) combined with elevated physiological loss weighting ($\lambda_{\text{resp}} = 5.0$) and a Mode-2-gated Explicit Cross-Modal InfoNCE Regularizer ($\lambda_{\text{align}} = 0.5$) over the two-stream (`tir`, `resp`) ADR-007 pre-training architecture.**

> **Abstraction boundary.** In accordance with this ADR's high-level design mandate, no existing source modules, class/function identifiers, runner scripts, or configuration file paths are cited below. Section 2.8 defines the normative logical parameter surface that any conforming implementation must expose and record.

### 2.1 Unified Stage-2 Pre-Training Objective

On every pre-training step over the declared two-stream modality set $\mathcal{M} = \{\text{tir}, \text{resp}\}$ (inherited from ADR-007):
$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{tir}} + \lambda_{\text{resp}}\,\mathcal{L}_{\text{resp}} + \lambda_{\text{align}} \cdot \mathbb{I}(K_{\text{resp}} > 0 \land K_{\text{tir}} > 0) \cdot \mathcal{L}_{\text{align}}$$
where:
- $\mathcal{L}_{\text{tir}}$ is standard masked time-space pixel MSE over masked `tir` tubelets (`mask == 1`), with implicit weight $1.0$.
- $\lambda_{\text{resp}}$ is the physiological stream weight (ADR-008 arm value **`5.0`**, raised from the baseline `0.5` to counter visual gradient dilution).
- $\mathcal{L}_{\text{resp}}$ is the three-term composite respiration waveform objective (§2.2).
- $\mathcal{L}_{\text{align}}$ is the symmetric cross-modal InfoNCE loss evaluated directly on pooled encoder latents (§2.3), scaled by $\lambda_{\text{align}}$ (ADR-008 arm value **`0.5`**) and gated by $\mathbb{I}(K_{\text{resp}} > 0 \land K_{\text{tir}} > 0)$ — active strictly on Mode-2 steps ($90\%$ of Phase-I steps and $50\%$ of Phase-II steps under ADR-007's curriculum).

### 2.2 Composite Respiration Waveform Objective ($\mathcal{L}_{\text{resp}}$)

```text
                      Composite Respiration Waveform Objective
                    L_resp = alpha * L_MSE + beta * L_Pearson + gamma * L_spectral
                                           │
          ┌────────────────────────────────┼────────────────────────────────┐
          ▼                                ▼                                ▼
 Time-Domain Shape (MSE)       Masked Linear Correlation         Band-Limited Spectral PSD
   alpha = 1.0 (neutral 1.0)     beta = 2.0 (neutral 0.0)          gamma = 1.0 (neutral 0.0)
 • Anchors baseline & scale    • Computed strictly on masked     • Spliced full clip [B, L]
 • Masked tokens only            tokens [B, n_masked * k]          (visible tokens detached)
 • Preserves mse_resp parity   • Repels flat zero-variance       • DC-centered Hann FP32 rFFT
                               • Degenerate-target guarded       • Unit-sum L1 over [0.10, 0.60] Hz
```

Let the physiological stream have $N_s$ temporal tokens of $k$ samples each ($L = N_s \cdot k$, with ADR-007 geometry $N_s = 250, k = 32, L = 8000, F_s = 1000\text{ Hz}$). Let $\hat{Y}, Y \in \mathbb{R}^{B \times N_s \times k}$ denote the per-token decoder prediction and per-clip z-scored target, $M \in \{0, 1\}^{B \times N_s}$ the binary mask (`1 = masked`, `0 = visible`), and $n_{\text{masked}}$ the batch-uniform masked token count (ADR-007 Contract 6).

$$\mathcal{L}_{\text{resp}} = \alpha\,\mathcal{L}_{\text{MSE}} + \beta\,\mathcal{L}_{\text{Pearson}} + \gamma\,\mathcal{L}_{\text{spectral}}$$

with ADR-008 arm weights $(\alpha, \beta, \gamma) = (1.0, 2.0, 1.0)$ and neutral defaults $(\alpha, \beta, \gamma) = (1.0, 0.0, 0.0)$. If $n_{\text{masked}} = 0$, all three terms emit explicit scalar `0.0`.

#### 2.2.1 Term 1 — Masked Time-Domain MSE ($\mathcal{L}_{\text{MSE}}$)
$$\mathcal{L}_{\text{MSE}} = \frac{\sum_{b=1}^{B}\sum_{n=1}^{N_s} M_{b,n}\cdot \frac{1}{k}\sum_{j=1}^{k}\left(\hat{Y}_{b,n,j} - Y_{b,n,j}\right)^2}{\sum_{b=1}^{B}\sum_{n=1}^{N_s} M_{b,n} + \epsilon_{\text{mask}}}, \qquad \epsilon_{\text{mask}} = 10^{-6}.$$
Retaining $\alpha = 1.0$ anchors DC amplitude, bounds extreme oscillations, and keeps raw `mse_resp` comparable across all Stage-2 arms.

#### 2.2.2 Term 2 — Masked Differentiable Negative Pearson Correlation Loss ($\mathcal{L}_{\text{Pearson}}$)
**Masking rule (prevents Mode-2 zero-hole corruption).** In masked autoencoders where only masked positions are decoded, unmasked slots of $\hat{Y}$ carry placeholder zeros. Computing Pearson over the full un-spliced tensor would corrupt mean, variance, and covariance on Mode-2 steps. Therefore, gather strictly the $n_{\text{masked}}$ masked tokens for both $\hat{Y}$ and $Y$, flatten to dense rectangular FP32 tensors $p, y \in \mathbb{R}^{B \times M_{\text{pts}}}$ where $M_{\text{pts}} = n_{\text{masked}} \cdot k$ (Mode 1: $M_{\text{pts}} = 8000$; Mode 2: $M_{\text{pts}} \in \{4800, 5600, 6400, 7200\}$).

Define centered vectors $\tilde{p}_b = p_b - \bar{p}_b$ and $\tilde{y}_b = y_b - \bar{y}_b$. With stabilizer $\epsilon = 10^{-8}$ placed inside **each** square root (preventing `NaN` gradients at exact zero variance) and degenerate-target variance floor $\epsilon_{\text{var}} = 10^{-6}$:
$$r_b = \frac{\sum_{i=1}^{M_{\text{pts}}} \tilde{p}_{b,i}\,\tilde{y}_{b,i}}{\sqrt{\sum_{i=1}^{M_{\text{pts}}} \tilde{p}_{b,i}^2 + \epsilon}\,\sqrt{\sum_{i=1}^{M_{\text{pts}}} \tilde{y}_{b,i}^2 + \epsilon}}, \qquad \mathcal{V} = \left\{b \in \{1,\dots,B\} \;\middle|\; \sum_{i=1}^{M_{\text{pts}}} \tilde{y}_{b,i}^2 > \epsilon_{\text{var}}\right\},$$
$$\mathcal{L}_{\text{Pearson}} = \begin{cases} \frac{1}{|\mathcal{V}|}\sum_{b \in \mathcal{V}} (1 - r_b) & \text{if } |\mathcal{V}| > 0 \text{ and } M_{\text{pts}} \ge 2 \\ 0.0 & \text{otherwise.} \end{cases}$$

**Anti-collapse mechanism:** $\mathcal{L}_{\text{Pearson}} \in [0.0, 2.0]$ attains its minimum $0.0$ at $r = +1.0$. As the decoder collapses toward a constant ($\|\tilde{p}_b\| \to 0$), $\mathcal{L}_{\text{Pearson}} \to 1.0$ and the gradient magnitude with respect to $\tilde{p}_b$ scales proportionally to $\|\tilde{p}_b\|^{-1}$, strongly repelling the network from the zero-variance subspace.

#### 2.2.3 Term 3 — Band-Limited Respiratory Spectral Power Loss ($\mathcal{L}_{\text{spectral}}$)
**Continuous-grid splicing rule (prevents Mode-2 rectangular-window artifacts).** A Discrete Fourier Transform requires a uniformly sampled time grid of length $L = N_s \cdot k$ at sampling rate $F_s$. Omitting unmasked tokens breaks uniform spacing, while zero-filling unmasked slots injects broadband rectangular-window harmonics. Assemble the full-clip waveform in FP32 by splicing detached ground truth into visible (`M == 0`) tokens:
$$\hat{y}_{\text{full}} = \operatorname{reshape}\!\left(M \odot \hat{Y} + (1 - M) \odot \operatorname{stopgrad}(Y),\; [B, L]\right), \qquad y_{\text{full}} = \operatorname{reshape}(Y, [B, L]).$$
On Mode 1 ($M \equiv 1$), $\hat{y}_{\text{full}} \equiv \hat{Y}$ (pure prediction). On Mode 2, gradients flow solely through masked tokens of $\hat{Y}$, forcing masked segments to match the frequency and phase continuity of the visible waveform.

**Windowed FP32 rFFT and unit-sum PSD matching:**
1. **Per-clip DC removal & Hann windowing:** Subtract per-sample temporal means from $\hat{y}_{\text{full}}$ and $y_{\text{full}}$ before multiplying by a length-$L$ Hann window $w(t) = 0.5 - 0.5\cos(2\pi t / L)$ to prevent DC leakage into low-frequency respiratory bins.
2. **FFT length $N_{\text{fft}}$:** Determined by logical parameter `spectral_n_fft` (default `0` = auto):
   $$N_{\text{fft}} = \begin{cases} 4 \cdot 2^{\lceil \log_2 L \rceil} & \text{if } \texttt{spectral\_n\_fft} = 0 \\ \texttt{spectral\_n\_fft} & \text{otherwise (must satisfy } N_{\text{fft}} \ge L \text{).} \end{cases}$$
   At ADR-007 geometry ($L = 8000, F_s = 1000\text{ Hz}$), auto $N_{\text{fft}} = 4 \times 8192 = 32768$, giving interpolated bin spacing $\Delta f = 1000 / 32768 \approx 0.0305\text{ Hz}$ ($\approx 1.83\text{ bpm}$) and 16 bins inside $[0.10, 0.60]\text{ Hz}$.
3. **Physiological respiratory passband $\Omega_{\text{resp}}$:** Default $[f_{\min}, f_{\max}] = [0.10\text{ Hz}, 0.60\text{ Hz}]$ ($6\text{--}36\text{ bpm}$), aligned to the normative evaluation passband (`ADR-009-resp-waveform-evaluation-protocol.md` §4). Define $\mathcal{K}_{\text{resp}} = \{k \mid f_{\min} \le \frac{k F_s}{N_{\text{fft}}} \le f_{\max}\}$; startup validation requires $|\mathcal{K}_{\text{resp}}| \ge 1$.
4. **Power spectrum & bin-invariant total-variation L1 distance:** Compute real-FFT spectra $\hat{Y}_{\text{fft}}, Y_{\text{fft}}$ in FP32 and power densities $P_{\hat{y}}(k) = \operatorname{Re}(\hat{Y}_{\text{fft}}(k))^2 + \operatorname{Im}(\hat{Y}_{\text{fft}}(k))^2$ and $P_y(k) = \operatorname{Re}(Y_{\text{fft}}(k))^2 + \operatorname{Im}(Y_{\text{fft}}(k))^2$ over $k \in \mathcal{K}_{\text{resp}}$. Normalize into unit-sum distributions ($\epsilon = 10^{-8}$):
   $$\tilde{P}_{\hat{y},b}(k) = \frac{P_{\hat{y},b}(k)}{\sum_{j \in \mathcal{K}_{\text{resp}}} P_{\hat{y},b}(j) + \epsilon}, \qquad \tilde{P}_{y,b}(k) = \frac{P_{y,b}(k)}{\sum_{j \in \mathcal{K}_{\text{resp}}} P_{y,b}(j) + \epsilon}.$$
   Guard against zero-energy targets via $\mathcal{V}_{\text{spec}} = \{b \mid \sum_{j \in \mathcal{K}_{\text{resp}}} P_{y,b}(j) > \epsilon_{\text{spec}}\}$ ($\epsilon_{\text{spec}} = 10^{-8}$):
   $$\mathcal{L}_{\text{spectral}} = \begin{cases} \frac{1}{|\mathcal{V}_{\text{spec}}|}\sum_{b \in \mathcal{V}_{\text{spec}}}\sum_{k \in \mathcal{K}_{\text{resp}}} \left|\tilde{P}_{\hat{y},b}(k) - \tilde{P}_{y,b}(k)\right| & \text{if } |\mathcal{V}_{\text{spec}}| > 0 \\ 0.0 & \text{otherwise.} \end{cases}$$

**Anti-collapse & scale invariance:** Summing $|\tilde{P}_{\hat{y}} - \tilde{P}_y|$ over $\mathcal{K}_{\text{resp}}$ (total variation L1 distance, bounded in $[0.0, 2.0]$) keeps the loss scale invariant to zero-padding factor and bin count $|\mathcal{K}_{\text{resp}}|$. On Mode 1, a flat zero prediction yields $P_{\hat{y}} \equiv 0 \implies \mathcal{L}_{\text{spectral}} = \sum_{k} \tilde{P}_y(k) \approx \mathbf{1.0}$, while concentrating spectral energy at the true breathing rate reduces the loss even under phase offset $\Delta\phi$.

### 2.3 Explicit Cross-Modal InfoNCE Alignment Regularizer ($\mathcal{L}_{\text{align}}$)

```text
                Shared Encoder Output Z: [B, N, D]  (post-encoder norm)
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
    Z_tir: [B, K_tir, D]                              Z_resp: [B, K_resp, D]
              │                                                 │
   Anatomy-Weighted Pooling                          Temporal Mean Pooling
   (spatial tube in A_core -> w_core)                           │
              ▼                                                 ▼
       u_tir: [B, D]                                     u_resp: [B, D]
              │                                                 │
   2-Layer MLP (D -> D -> d_proj)                    2-Layer MLP (D -> D -> d_proj)
   + FP32 L2 Normalization                           + FP32 L2 Normalization
              ▼                                                 ▼
    v_tir: [B, d_proj]                                v_resp: [B, d_proj]
              │                                                 │
              └──────────► [Optional Cross-Rank Gather] ◄───────┘
                                       │
                                       ▼
                        Symmetric InfoNCE (tau = 0.07)
                     Gated strictly to Mode 2 (K_resp > 0)
```

#### 2.3.1 Latent slicing and pooling operators
After the shared encoder (post-normalization) produces $Z \in \mathbb{R}^{B \times N \times D}$, slice the contiguous visible-token blocks belonging to each stream: $Z_{\text{tir}} \in \mathbb{R}^{B \times K_{\text{tir}} \times D}$ and $Z_{\text{resp}} \in \mathbb{R}^{B \times K_{\text{resp}} \times D}$ (defined on every Mode-2 step where $K_{\text{tir}} > 0$ and $K_{\text{resp}} > 0$).

1. **Temporal respiration pooling:**
   $$u_{\text{resp}}^{(b)} = \frac{1}{K_{\text{resp}}} \sum_{j=1}^{K_{\text{resp}}} Z_{\text{resp}}^{(b, j)} \in \mathbb{R}^{D}.$$
2. **Configurable anatomy-weighted thermal pooling:** Let $s(b, i) \in \{0, \dots, N_{\text{spatial}} - 1\}$ denote the spatial tube index of visible `tir` token $i$ in sample $b$ (for a $(t, h, w)$-flattened tubelet grid, $s(b, i) = \text{idx}_{\text{tir}}(b, i) \bmod N_{\text{spatial}}$), and let $\mathcal{A}_{\text{core}}$ be the anatomical core tube set defined by ADR-007 (empty under `spatial_sampling = uniform`). With configurable core weight $w_{\text{core}} = \texttt{align\_core\_weight} > 0$:
   $$w_{b,i} = \begin{cases} w_{\text{core}} & \text{if } s(b,i) \in \mathcal{A}_{\text{core}} \\ 1.0 & \text{otherwise,} \end{cases} \qquad u_{\text{tir}}^{(b)} = \frac{\sum_{i=1}^{K_{\text{tir}}} w_{b,i}\, Z_{\text{tir}}^{(b,i)}}{\sum_{i=1}^{K_{\text{tir}}} w_{b,i}} \in \mathbb{R}^{D}.$$
   Setting $w_{\text{core}} = 1.0$ (default) recovers unweighted global mean pooling; setting $w_{\text{core}} > 1.0$ (e.g., `2.0`) emphasizes the nostril-plume core tokens during cross-modal alignment.

#### 2.3.2 Non-linear projection heads
To prevent contrastive dimensional collapse from eroding spatial detail required by the visual reconstruction decoder, each modality's pooled latent passes through a dedicated 2-layer MLP ($D \to D \to d_{\text{proj}}$, default $d_{\text{proj}} = 128$, adding $2(D^2 + D + D d_{\text{proj}} + d_{\text{proj}}) = 1{,}378{,}048$ parameters at $D = 768$) followed by FP32 L2 normalization onto the unit hypersphere:
$$g_m(u) = W_2^{(m)}\operatorname{GELU}\!\left(W_1^{(m)} u + b_1^{(m)}\right) + b_2^{(m)}, \qquad v_m^{(b)} = \frac{g_m(u_m^{(b)})}{\max\!\left(\|g_m(u_m^{(b)})\|_2,\; \epsilon\right)} \in \mathbb{R}^{d_{\text{proj}}}.$$

#### 2.3.3 Symmetric InfoNCE formulation and distributed negative gathering
For an effective batch of $B_{\text{eff}} = W \cdot B$ synchronized clips (where $W$ is the distributed world size when cross-rank gathering is enabled during training, and $W = 1$ on single-device runs or unpadded validation), let $V_{\text{tir}}, V_{\text{resp}} \in \mathbb{R}^{B_{\text{eff}} \times d_{\text{proj}}}$ denote the gathered embeddings with gradients preserved on the local rank's slice $[rB : (r+1)B]$ and detached on remote slices. For each local sample $b \in \{0, \dots, B - 1\}$ with global positive index $p(b) = rB + b$ and temperature $\tau > 0$ (default `0.07`):
$$\mathcal{L}_{\text{align}} = -\frac{1}{2B}\sum_{b=0}^{B-1} \left[ \log \frac{\exp\!\left(\langle v_{\text{tir}}^{(b)}, V_{\text{resp}}^{(p(b))} \rangle / \tau\right)}{\sum_{k=0}^{B_{\text{eff}}-1} \exp\!\left(\langle v_{\text{tir}}^{(b)}, V_{\text{resp}}^{(k)} \rangle / \tau\right)} + \log \frac{\exp\!\left(\langle v_{\text{resp}}^{(b)}, V_{\text{tir}}^{(p(b))} \rangle / \tau\right)}{\sum_{k=0}^{B_{\text{eff}}-1} \exp\!\left(\langle v_{\text{resp}}^{(b)}, V_{\text{tir}}^{(k)} \rangle / \tau\right)} \right].$$

**Gradient directivity:** The gradient with respect to $v_{\text{tir}}^{(b)}$ is $\frac{1}{\tau}\left(\sum_{k} P_{bk} V_{\text{resp}}^{(k)} - V_{\text{resp}}^{(p(b))}\right)$, injecting a direct biological supervisory signal into the visual Transformer encoder blocks without passing through either reconstruction decoder.

If $B_{\text{eff}} < 2$ (e.g., batch size 1 on a single device), $\mathcal{L}_{\text{align}}$ evaluates to `0.0`.

### 2.4 Normative contracts

These ten contracts completely define the objective and alignment behaviour:

1. **Unified loss composition.** Every pre-training step optimizes $\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{tir}} + \lambda_{\text{resp}}\mathcal{L}_{\text{resp}} + \lambda_{\text{align}} \cdot \mathbb{I}(K_{\text{resp}} > 0 \land K_{\text{tir}} > 0) \cdot \mathcal{L}_{\text{align}}$ with $\mathcal{L}_{\text{resp}} = \alpha\mathcal{L}_{\text{MSE}} + \beta\mathcal{L}_{\text{Pearson}} + \gamma\mathcal{L}_{\text{spectral}}$ (R1, R3).
2. **Masked-only correlation.** $\mathcal{L}_{\text{Pearson}}$ is computed exclusively over the $n_{\text{masked}} \cdot k$ masked samples; altering predictions at unmasked positions (`mask == 0`) must produce zero change in $\mathcal{L}_{\text{Pearson}}$ or its gradients (R2).
3. **Detached-spliced spectral continuity.** $\mathcal{L}_{\text{spectral}}$ operates on the full-clip length-$L$ waveform formed by splicing detached ground truth at visible positions (`mask == 0`) with decoded predictions at masked positions (`mask == 1`); altering unmasked prediction slots must produce zero change in $\mathcal{L}_{\text{spectral}}$ or its gradients (R2).
4. **FP32 numerical stability & degenerate guards.** $\mathcal{L}_{\text{Pearson}}$, $\mathcal{L}_{\text{spectral}}$ (DC centering, Hann window, rFFT, PSD normalization), and $\mathcal{L}_{\text{align}}$ (L2 normalization and cosine logits) execute in FP32 regardless of mixed-precision autocast mode; zero-variance / zero-energy targets are excluded cleanly rather than producing `NaN`/`Inf` (R4).
5. **Strict Mode-2 alignment gate.** $\mathcal{L}_{\text{align}}$ is computed strictly when $K_{\text{resp}} > 0$ and $K_{\text{tir}} > 0$. On Mode-1 steps ($K_{\text{resp}} = 0$), logged `loss_align == 0.0` and `align_active == 0` (R3).
6. **Distributed reduction & loss-key stability.** On Mode-1 steps during distributed training, the alignment projection parameters must still participate in the autograd graph with exact zero gradient (via a zero-multiplier dummy term) so gradient all-reduce never hangs even when unused-parameter detection is disabled; all logged metric keys must remain fixed across modes and ranks (R4).
7. **Neutral backward compatibility.** When $(\alpha, \beta, \gamma, \lambda_{\text{align}}) = (1.0, 0.0, 0.0, 0.0)$, no alignment projection heads are instantiated (zero added parameters) and the training loss is bit-identical to pure masked MSE (R5).
8. **Two-stream cardinality guard for alignment.** Enabling $\lambda_{\text{align}} > 0$ requires exactly one visual stream and one physiological stream (`len(visual) == 1` and `len(signal) == 1`); multi-visual or multi-signal runs must fail loudly at startup rather than silently aligning an arbitrary stream pair (R3, R7).
9. **Downstream transfer isolation.** Stage-2 alignment projection heads are auxiliary pre-training modules and must be excluded cleanly when transferring encoder weights to Stage-3 downstream waveform regression (R6).
10. **Zero hidden constants.** Every weight, passband edge, FFT length rule, projection dimension, temperature, pooling weight, and diagnostic interval is governed solely by the exposed parameter surface of §2.8 and recorded in the run manifest (R7).

### 2.5 Language-agnostic reference algorithm

```text
# Executed per pre-training step given encoder latents Z [B, N, D],
# per-stream predictions/targets/masks, kept indices, and step counts (K_tir, K_resp).

function COMPOSITE_RESP_LOSS(pred, target, mask, ids_masked, cfg):
    # pred, target: [B, N_s, k]; mask: [B, N_s] in {0, 1}; ids_masked: [B, n_masked]
    if n_masked == 0:
        return {total: 0.0, mse: 0.0, pearson: 0.0, spectral: 0.0}

    l_mse = MASKED_MSE(pred, target, mask)

    if cfg.resp_loss_pearson_weight > 0:
        p_m = reshape(gather_tokens(pred,   ids_masked), [B, n_masked * k])   # FP32
        t_m = reshape(gather_tokens(target, ids_masked), [B, n_masked * k])   # FP32
        l_pearson = MASKED_NEG_PEARSON_FP32(p_m, t_m, eps=1e-8, var_floor=1e-6)
    else:
        l_pearson = 0.0

    if cfg.resp_loss_spectral_weight > 0:
        # Splice detached ground truth at visible positions (mask == 0)
        p_full = reshape(mask * fp32(pred) + (1 - mask) * stopgrad(fp32(target)), [B, L])
        t_full = reshape(stopgrad(fp32(target)), [B, L])
        l_spectral = BANDLIMITED_SPECTRAL_L1_FP32(
            p_full, t_full, fs=cfg.fs, n_fft=resolved_n_fft,
            band=[cfg.spectral_f_min, cfg.spectral_f_max], eps=1e-8, energy_floor=1e-8
        )
    else:
        l_spectral = 0.0

    total = (cfg.resp_loss_mse_weight      * l_mse
           + cfg.resp_loss_pearson_weight  * l_pearson
           + cfg.resp_loss_spectral_weight * l_spectral)
    return {total: total, mse: l_mse, pearson: l_pearson, spectral: l_spectral}


function CROSS_MODAL_INFONCE(Z_tir, Z_resp, ids_keep_tir, core_indices, cfg, is_training):
    # Z_tir: [B, K_tir, D]; Z_resp: [B, K_resp, D]
    u_tir  = ANATOMY_WEIGHTED_POOL(Z_tir, ids_keep_tir, core_indices, cfg.align_core_weight)
    u_resp = MEAN(Z_resp, dim=1)

    v_tir  = L2_NORMALIZE_FP32(proj_tir(u_tir))          # [B, d_proj]
    v_resp = L2_NORMALIZE_FP32(proj_resp(u_resp))        # [B, d_proj]

    if cfg.align_gather_ddp and is_training and world_size > 1:
        V_tir,  offset = ALL_GATHER_KEEP_LOCAL_GRAD(v_tir)
        V_resp, _      = ALL_GATHER_KEEP_LOCAL_GRAD(v_resp)
    else:
        V_tir, V_resp, offset = v_tir, v_resp, 0

    if rows(V_tir) < 2:
        return ZERO_TOUCH_ALIGN_PARAMS() if is_training else 0.0

    logits_t2r = (v_tir  @ transpose(V_resp)) / cfg.align_temperature
    logits_r2t = (v_resp @ transpose(V_tir))  / cfg.align_temperature
    labels     = arange(0, B) + offset
    return 0.5 * (CROSS_ENTROPY(logits_t2r, labels) + CROSS_ENTROPY(logits_r2t, labels))


function STEP_OBJECTIVE(...):
    l_tir  = MASKED_MSE(pred_tir, target_tir, mask_tir)
    r_out  = COMPOSITE_RESP_LOSS(pred_resp, target_resp, mask_resp, ids_masked_resp, cfg)
    total  = l_tir + cfg.signal_weight * r_out.total

    if cfg.align_weight > 0:
        if K_resp > 0 and K_tir > 0:
            l_align      = CROSS_MODAL_INFONCE(Z_tir, Z_resp, ids_keep_tir, core_indices, cfg, is_training)
            align_active = 1
            total        = total + cfg.align_weight * l_align
        else:
            l_align      = 0.0
            align_active = 0
            if is_training:
                total    = total + ZERO_TOUCH_ALIGN_PARAMS()
    return total, {l_tir, r_out, l_align, align_active}
```

### 2.6 Feasibility preconditions & startup validation

Every precondition below must be validated before training step 0; any violation must fail loudly at startup:

1. **Non-negative, non-degenerate composite weights:** $\alpha \ge 0$, $\beta \ge 0$, $\gamma \ge 0$, $\lambda_{\text{align}} \ge 0$, and $\alpha + \beta + \gamma > 0$.
2. **Valid respiratory passband (when $\gamma > 0$):** $0 < f_{\min} < f_{\max} \le F_s / 2$. *(Check: $0 < 0.10 < 0.60 \le 500.0$ ✓)*
3. **Valid FFT length & non-empty band (when $\gamma > 0$):** Either `spectral_n_fft == 0` (auto) or `spectral_n_fft >= L`, and $|\mathcal{K}_{\text{resp}}| \ge 1$. *(Check: $N_{\text{fft}} = 32768 \ge 8000$, $|\mathcal{K}_{\text{resp}}| = 16 \ge 1$ ✓)*
4. **Valid InfoNCE parameters (when $\lambda_{\text{align}} > 0$):** $\tau > 0$, $d_{\text{proj}} \ge 16$, $w_{\text{core}} > 0$, and exact two-stream cardinality $|\mathcal{M}_{\text{visual}}| = 1 \land |\mathcal{M}_{\text{signal}}| = 1$. *(Check: $\tau = 0.07 > 0$, $d_{\text{proj}} = 128 \ge 16$, $w_{\text{core}} = 1.0 > 0$, $|\{\text{tir}\}| = |\{\text{resp}\}| = 1$ ✓)*
5. **Valid gradient-balance cadence:** `grad_balance_freq >= 0`.
6. **Coherent composite target & single physiological stream (when $\beta > 0$ or $\gamma > 0$):** the resolved target normalisation must be `none` or `clip` (a per-token normalisation inserts a step at every token boundary, so the spliced full-clip waveform of §2.2.3 is discontinuous and the passband would describe a splicing artefact), and exactly one physiological stream must be declared ($|\mathcal{M}_{\text{signal}}| = 1$), because the per-step objective observables of §2.7 are singular per stream (a second physiological stream would overwrite another stream's `resp_pearson` / `resp_spectral`). *(Check: `target_norm = none` via the ROI-lineage pin, $|\{\text{resp}\}| = 1$ ✓)*

*Rule 6 was added on 2026-11-25 during implementation planning (decision **D3** of `code/notes/plan/features/ADR-008-composite-objective-and-cross-modal-alignment-plan.md`) so that the ADR and its plan state the same precondition set.*

### 2.7 Observability contract

Any conforming run executing this ADR must record:

- **Per step (training):** raw `mse_tir`, raw `mse_resp` (preserving baseline comparability), raw unweighted `resp_pearson`, `resp_spectral`, `loss_align`, and `align_active` $\in \{0, 1\}$, alongside all ADR-007 masking observables. Key sets must be fixed by configuration (never conditional on the sampled mode).
- **Periodic gradient-balance diagnostic (opt-in via `grad_balance_freq > 0`):** Every `grad_balance_freq` steps, compute $\|\nabla_{z_{\text{tir}}} \mathcal{L}_{\text{tir}}\|_2$, $\|\nabla_{z_{\text{tir}}}(\lambda_{\text{resp}}\mathcal{L}_{\text{resp}})\|_2$, and their ratio $\|\nabla_{z_{\text{tir}}}(\lambda_{\text{resp}}\mathcal{L}_{\text{resp}})\|_2 / \|\nabla_{z_{\text{tir}}} \mathcal{L}_{\text{tir}}\|_2$ via non-destructive autograd differentiation (without mutating parameter gradient buffers). When `grad_balance_freq == 0`, zero extra differentiation cost is incurred.
- **Per validation pass (deterministic FP32):** Preserve per-stream raw MSE components intact and report sample-weighted composite and alignment validation averages (`resp_mse`, `resp_pearson`, `resp_spectral`, `loss_align`, `align_active`) in a dedicated objective summary block, aggregated across all distributed ranks.
- **Per run (manifest):** Record the full resolved ADR-008 objective parameter tuple of §2.8 alongside ADR-007's masking record. Resuming a run whose stored objective parameters disagree with the current configuration must fail loudly. When all ADR-008 parameters remain at their neutral defaults, legacy manifests stay unchanged.

### 2.8 Logical parameter surface

These logical parameters completely specify ADR-008. Any conforming implementation must expose them without hidden constants and record their resolved values in the run manifest:

| Logical Parameter | Symbol | Type | Neutral Default | ADR-008 Arm Value | Admissible Values | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `signal_weight` *(reused)* | $\lambda_{\text{resp}}$ | float | `0.5` | `5.0` | $> 0.0$ | Weight on $\mathcal{L}_{\text{resp}}$ relative to visual weight `1.0` |
| `resp_loss_mse_weight` | $\alpha$ | float | `1.0` | `1.0` | $\ge 0.0$ | Weight of masked time-domain MSE inside $\mathcal{L}_{\text{resp}}$ |
| `resp_loss_pearson_weight` | $\beta$ | float | `0.0` | `2.0` | $\ge 0.0$ | Weight of masked negative Pearson loss (`0.0` = disabled) |
| `resp_loss_spectral_weight` | $\gamma$ | float | `0.0` | `1.0` | $\ge 0.0$ | Weight of band-limited spectral power loss (`0.0` = disabled) |
| `spectral_f_min` | $f_{\min}$ | float | `0.10` | `0.10` | $(0.0,\; f_{\max})$ | Lower respiratory passband edge (Hz) |
| `spectral_f_max` | $f_{\max}$ | float | `0.60` | `0.60` | $(f_{\min},\; F_s / 2]$ | Upper respiratory passband edge (Hz) |
| `spectral_n_fft` | $N_{\text{fft}}$ | int | `0` | `0` ($\to 32768$) | `0` (auto $4 \cdot 2^{\lceil\log_2 L\rceil}$) or $\ge L$ | rFFT length for $\mathcal{L}_{\text{spectral}}$ |
| `align_weight` | $\lambda_{\text{align}}$ | float | `0.0` | `0.5` | $\ge 0.0$ | Weight of symmetric InfoNCE loss (`0.0` = disabled, no heads created) |
| `align_proj_dim` | $d_{\text{proj}}$ | int | `128` | `128` | $\ge 16$ | Output dimensionality of the 2-layer InfoNCE projection MLPs |
| `align_temperature` | $\tau$ | float | `0.07` | `0.07` | $> 0.0$ | InfoNCE cosine-similarity temperature |
| `align_core_weight` | $w_{\text{core}}$ | float | `1.0` | `1.0` | $> 0.0$ | Pooling weight on $\mathcal{A}_{\text{core}}$ visible `tir` tokens vs. `1.0` on $\mathcal{A}_{\text{periph}}$ |
| `align_gather_ddp` | — | bool | `true` | `true` | `true`, `false` | Gather cross-rank embeddings for global InfoNCE negatives in distributed training |
| `grad_balance_freq` | — | int | `0` | `50` | $\ge 0$ | Step interval for gradient-balance ratio logging (`0` = disabled) |

Fixed geometry inherited from ADR-007 (not redefined here): `streams = {tir, resp}`, `n_budget = 200`, `clip_duration = 8.0 s`, `fs = 1000 Hz`, `seq_len = 8000`, `sig_kernel = 32` ($N_s = 250$ tokens), `spatial_patches = 36` ($6 \times 6$), `tokens_per_unit = 25`, `spatial_sampling = anatomy_aware`, `E_warm = 20`.

### 2.9 Rejected alternatives

- **Raising $\lambda_{\text{resp}}$ alone while keeping pure masked MSE:** Rejected — scaling MSE preserves its flat conditional-mean global minimum under phase uncertainty and cannot cure zero-variance collapse.
- **Applying broadband multi-resolution STFT loss in Stage 2:** Rejected — broadband log-magnitude STFT penalizes energy up to Nyquist ($500\text{ Hz}$), whereas Stage-2 respiratory supervision should concentrate spectral matching on the physiological $[0.10, 0.60]\text{ Hz}$ passband.
- **Applying InfoNCE to decoder outputs or forcing InfoNCE on Mode-1 steps:** Rejected — decoder-side alignment fails to inject direct gradients into shared encoder blocks; on Mode-1 steps $K_{\text{resp}} = 0$, so no physiological encoder latents exist.
- **Computing Pearson or rFFT on raw zero-filled predictions during Mode 2:** Rejected — placeholder zeros at unmasked positions distort both Pearson variance/covariance and FFT power spectra; masked-only gathering (Pearson) and detached-target splicing (Spectral) eliminate both artefacts.
- **Computing rFFT on token-averaged 250-point signals at a hardcoded $31.25\text{ Hz}$ rate:** Rejected — operating on the full-resolution spliced `[B, L]` waveform at the declared sampling rate $F_s$ generalizes cleanly across arbitrary sampling rates and clip lengths without intra-token aliasing.

---

## 3. Falsifiable Criteria

### 3.1 Criterion 1 — Mathematical, masking, precision, and distributed conformance (pre-training gate; zero GPU hours)

All seven deterministic checks must pass before launching any full training run:
1. **Neutral-default bit-identity:** With $(\alpha, \beta, \gamma, \lambda_{\text{align}}) = (1.0, 0.0, 0.0, 0.0)$, no alignment projection heads are created (zero added parameters) and forward/backward outputs match pure masked MSE bit-for-bit.
2. **Anti-collapse response (Mode 1):** On a synthetic unit-variance $0.25\text{ Hz}$ sinusoid ($L = 8000, F_s = 1000\text{ Hz}$, all tokens masked), flat prediction $\hat{y} \equiv 0$ yields $\mathcal{L}_{\text{Pearson}} = 1.0 \pm 10^{-5}$, $\mathcal{L}_{\text{spectral}} = 1.0 \pm 10^{-4}$, and $\|\partial \mathcal{L}_{\text{resp}} / \partial \hat{y}\|_2 > 0$; exact target prediction $\hat{y} = y$ yields $\mathcal{L}_{\text{MSE}}, \mathcal{L}_{\text{Pearson}}, \mathcal{L}_{\text{spectral}} < 10^{-6}$.
3. **Mode-2 masking integrity & degenerate-target guard:** On Mode 2 ($K_{\text{resp}} > 0$), modifying predicted values at unmasked positions (`mask == 0`) produces zero change in $\mathcal{L}_{\text{MSE}}$, $\mathcal{L}_{\text{Pearson}}$, $\mathcal{L}_{\text{spectral}}$, and their gradients; constant all-zero targets return finite `0.0` with zero `NaN`/`Inf`.
4. **InfoNCE gate, directivity, and anatomy weighting:** On Mode 2 ($K_{\text{resp}} > 0, B \ge 2$), $\mathcal{L}_{\text{align}} > 0$ produces strictly non-zero gradients on both visual and physiological encoder paths, and synchronized positive pairs yield strictly lower loss than permuted pairs; with $w_{\text{core}} > 1.0$, core-token gradient norms exceed peripheral-token gradient norms. On Mode 1 ($K_{\text{resp}} = 0$), `loss_align == 0.0`, `align_active == 0`, and projection heads receive exact zero gradient while remaining connected in the distributed autograd graph.
5. **Mixed-precision (FP16 / BF16) safety:** Forward and backward passes under FP16 and BF16 autocast produce finite losses and gradients with no complex-FFT dtype failures.
6. **Downstream transfer isolation:** Loading an ADR-008 Stage-2 checkpoint into a Stage-3 visual-only regressor transfers all encoder weights cleanly and ignores auxiliary alignment projection parameters.
7. **Startup feasibility rejection:** Each violated precondition in §2.6 raises a startup validation error before step 0.

### 3.2 Criterion 2 — Stage-2 pretext anti-collapse & gradient-balance diagnostics (Fold-0 held-out `test`, non-gating for Stage-3 adoption)

Per `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md` §1, Stage-2 pretext metrics verify optimization health and collapse elimination but are **non-gating** for final Stage-3 adoption. Measured on Fold-0 held-out `test` subjects (subject-disjoint per ADR-001) under Mode 1 ($K_{\text{resp}} = 0$, $K_{\text{tir}} = 200$):

| Diagnostic Indicator | Verification Condition & Aggregation | Baseline Reference | Failure Threshold (Rejection) | Success Target |
| :--- | :--- | :--- | :--- | :--- |
| **Variance Retention** | Mode 1 ($K_{\text{resp}} = 0$), per-clip $\text{Var}(\hat{y}) = \frac{1}{L}\sum_{t=1}^{L}(\hat{y}_t - \bar{\hat{y}})^2$, median across held-out subjects | Predecessor pure-MSE `MSE ≈ 1.02–1.04 z²` (heavy shrinkage); matched pure-MSE ADR-007 Fold 0 | $\text{Var}(\hat{y}) < 0.10$ (conditional mean collapse) | $\text{Var}(\hat{y}) \ge 0.60$ |
| **Spectral Peak Error** | Mode 1 ($K_{\text{resp}} = 0$), $|\arg\max_{\omega \in \Omega_{\text{resp}}} P_{\hat{y}}(\omega) - \arg\max_{\omega \in \Omega_{\text{resp}}} P_y(\omega)|$, subject median | Matched pure-MSE ADR-007 Fold 0 | $> 3.0\text{ bpm}$ ($> 0.05\text{ Hz}$) | $\le 1.5\text{ bpm}$ ($\le 0.025\text{ Hz}$) |
| **Zero-Shot Correlation** | Mode 1 ($K_{\text{resp}} = 0$), masked Pearson $r$, subject median across Fold-0 test subjects | Four-stream `+0.0062` (`code/notes/eval/baselines/tir_rgb2d_bp_resp/resp_recon_arms_ab.md`); three-stream `+0.113 [+0.078, +0.136]` (`code/notes/eval/baselines/tir_rgb2d_resp/README.md`); matched ADR-007 Fold 0 | $r < +0.10$ (noise floor) | $r \ge +0.35$ |
| **Gradient Balance** | Stage-2 Phase-II periodic logs (`grad_balance_freq = 50`), median $\|\nabla_{z_{\text{tir}}}(\lambda_{\text{resp}}\mathcal{L}_{\text{resp}})\|_2 / \|\nabla_{z_{\text{tir}}}\mathcal{L}_{\text{tir}}\|_2$ | Matched pure-MSE ADR-007 Fold 0 ($\lambda_{\text{resp}} = 0.5$) | $< 10^{-3}$ (gradient dilution) | $\in [0.05,\; 0.50]$ |

### 3.3 Criterion 3 — Stage-3 downstream scientific Go/No-Go gate (normative protocol)

All scientific adoption claims follow `code/notes/design/ADR-009-resp-waveform-evaluation-protocol.md` and `code/notes/design/ADR-001-subject-disjoint-split-and-selection.md`:

- **Split & aggregation:** Subject-disjoint 5-fold cross-validation ($70\%$ train / $10\%$ val / $20\%$ test by subject per fold; model selection strictly on `val`, one-shot evaluation on `test`). Clip predictions are assembled into continuous per-session waveforms via Hann overlap-add and scored against the shared respiration-cache reference.
- **Headline metric (gating):** Band-limited Pearson correlation ($0.10\text{--}0.60\text{ Hz}$), aggregated per session $\to$ median per subject $\to$ pooled median across held-out subjects with bootstrap $95\%$ CI. Secondary diagnostic metric: breath-rate MAE (`bpm`). Amplitude metrics are excluded from gating by protocol.
- **Go/No-Go threshold:** After identical Stage-3 TIR-only fine-tuning across all 5 folds, the ADR-008 pre-trained encoder must exceed the matched pure-MSE ADR-007 pre-trained encoder on the pooled subject-level band-limited Pearson median by **$\ge +0.02$** (with bootstrap $95\%$ CIs reported), and beat the single-tone template baseline on $> 50\%$ of held-out subjects. *(Verified historical off-protocol Stage-3 reference for context only: `+0.542 [+0.493, +0.576]` vs. template `+0.537`, `code/notes/eval/tir_resp/resp_5fold_cv_96respgrid.md`.)*

### 3.4 Criterion 4 — Capped smoke execution & resource record

- **Smoke run:** A single-device capped Stage-2 smoke run (1 epoch, warm-up disabled, corpus capped at $\le 100$ clips, small batch size, Mode-1 probability $0.50$ so both modes execute, periodic gradient-balance probe enabled) must complete with zero assertion errors or `NaN`/`Inf` losses, record the resolved ADR-008 parameter tuple in the run manifest, and pass a deterministic validation sweep emitting both raw MSE components and ADR-008 objective diagnostics.
- **Resource threshold:** Peak allocated and reserved device memory must be measured and recorded alongside batch size, device model, and precision mode. The two 2-layer projection MLPs ($D = 768 \to 768 \to 128$) add exactly $1{,}378{,}048$ parameters ($\sim 1.3\%$ of a ViT-Base encoder) and negligible activation memory (`[B, 128]` embeddings plus FP32 length-$L$ rFFT); peak allocated VRAM must remain within $+5\%$ of the matched pure-MSE ADR-007 run on identical hardware. `[TODO: verify exact peak allocated/reserved VRAM, batch size, device model, precision mode, and step wall-time on first ADR-008 execution]`.

### 3.5 Acceptance artefacts

- **Per fold:** Immutable fold partition record (with subject-disjoint digest), validation selection record, per-epoch training/validation metric log (including raw MSE, composite terms, InfoNCE, and gradient-balance ratio), run manifest capturing the resolved §2.8 parameter tuple, and Fold-0 frozen Mode-1 pretext evaluation report.
- **Per experiment:** Pooled 5-fold Stage-3 evaluation summary containing per-subject metrics, pooled median band-limited Pearson with bootstrap $95\%$ CI, breath-rate MAE, template-win fraction, and paired fold-level deltas against the matched pure-MSE ADR-007 comparator.

---

## 4. Consequences & Trade-offs

- **Positive:**
  - **Eliminates the conditional mean trap at Stage 2:** $\mathcal{L}_{\text{Pearson}}$ actively repels zero-variance flat collapse and $\mathcal{L}_{\text{spectral}}$ rewards true respiratory-band energy concentration even under temporal phase uncertainty $\Delta\phi$.
  - **Overcomes cross-modal gradient dilution:** Pairing $\lambda_{\text{resp}} = 5.0$ with Mode-2-gated symmetric InfoNCE ($\lambda_{\text{align}} = 0.5$) injects direct physiological supervisory gradients into the shared visual encoder rather than letting pixel reconstruction dominate.
  - **Zero regression risk to existing arms or Stage 3:** Neutral parameter defaults keep pure-MSE baselines bit-identical, and auxiliary alignment heads are cleanly discarded on Stage-3 encoder transfer.
- **Negative / Technical Debt:**
  - **Expanded hyperparameter surface:** Introduces 12 additional logical parameters (all neutrally defaulted) that must be tracked consistently in run manifests and ablations.
  - **Non-comparable raw total validation loss across objectives:** Because total validation loss combines composite waveform terms and InfoNCE, cross-arm comparisons against pure-MSE ADR-007 must use raw `mse_resp`, Mode-1 pretext diagnostics (§3.2), and Stage-3 subject-level band-limited Pearson (§3.3).
  - **Mini-batch negative sensitivity in InfoNCE:** Small local batch sizes restrict negative diversity, and random mini-batches can occasionally place two clips from the same subject/session in the same batch as false negatives. Cross-rank negative gathering (`align_gather_ddp = true`) enlarges the negative pool in distributed runs, but instance-level false-negative collisions remain an inherent trade-off.
  - **Potential visual representation degradation if over-weighted:** Excessively large $\lambda_{\text{resp}}$ or $\lambda_{\text{align}}$ can starve visual tubelet reconstruction (`mse_tir` divergence); monitoring `mse_tir` and keeping `grad_ratio_resp_to_tir` within $[0.05, 0.50]$ mitigates this risk.
- **Rollback / Rejection Condition:** Reject, disable (via neutral defaults $\beta = 0, \gamma = 0, \lambda_{\text{align}} = 0, \lambda_{\text{resp}} = 0.5$), or retune this decision if: (i) any Criterion-1 mathematical, precision, or distributed invariant fails; (ii) Fold-0 Mode-1 pretext diagnostics still collapse ($\text{Var}(\hat{y}) < 0.10$ or subject median $r < +0.10$), gradient balance remains diluted ($< 10^{-3}$), or `mse_tir` diverges ($> 2\times$ pure-MSE ADR-007); or (iii) 5-fold Stage-3 downstream evaluation fails to exceed the matched pure-MSE ADR-007 baseline on pooled subject-level band-limited Pearson median by the pre-registered margin ($\ge +0.02$).

---

## 5. Reproducibility Record

- **Run / Job IDs**: `N/A` — proposed decision; not yet executed.
- **Evidence Artifacts**:
  - `code/notes/eval/baselines/tir_rgb2d_bp_resp/resp_recon_arms_ab.md` (measured Fold-0 four-stream Mode-1 collapse: `r_masked = +0.0062`, `mse = 1.0254 z²`).
  - `code/notes/eval/baselines/tir_rgb2d_resp/README.md` (measured Fold-0 three-stream Mode-2 `+0.996` / `MSE = 0.009` vs. Mode-1 `+0.113 [+0.078, +0.136]` / `MSE = 1.041`).
  - `code/notes/eval/tir_resp/resp_5fold_cv_96respgrid.md` (historical off-protocol 5-fold Stage-3 TIR-ROI reference `+0.542 [+0.493, +0.576]` vs. template `+0.537`).
  - `[TODO: link Fold-0 ADR-007 vs. ADR-008 pretext diagnostics and 5-fold Stage-3 evaluation artefacts once executed]`.
- **Tests / Smoke Checks**:
  - Deterministic conformance suite covering all seven Criterion-1 invariants (§3.1).
  - Single-device capped smoke run (§3.4).
  - `[TODO: record verification suite results and peak VRAM measurements after implementation]`.
- **Rollback / Rejection Condition**: Defined in §4.
- **Deferred, non-blocking decisions:** (1) Ablating anatomy-weighted InfoNCE pooling ($w_{\text{core}} = 2.0$ vs. uniform $w_{\text{core}} = 1.0$); (2) sensitivity sweep over $\lambda_{\text{resp}} \in \{2.0, 5.0, 10.0\}$ and $\lambda_{\text{align}} \in \{0.1, 0.5, 1.0\}$ if Fold-0 gradient balance falls outside $[0.05, 0.50]$; (3) comparing the canonical $[0.10, 0.60]\text{ Hz}$ passband against the narrower $[0.15, 0.50]\text{ Hz}$ clinical band.

---

## Appendix A — Formal Corrections Applied to the Earlier Withdrawn Draft

For audit traceability, the following formal discrepancies in the earlier withdrawn draft were corrected in this authoritative specification:

1. **Empirical baseline attribution (§1.2):** The draft attributed `r_masked ≈ +0.006` and `r = 0.9967 / MSE = 0.0072` to Fold-0 trials of ADR-007. No Fold-0 ADR-007 evaluation artifact is yet archived; the verified repository baselines are four-stream Fold 0 (`+0.0062`, `mse = 1.0254 z²`) and three-stream Fold 0 (`+0.996 / 0.009` Mode 2 vs. `+0.113 / 1.041` Mode 1).
2. **Waveform sample length $L$ and sampling rate $F_s$ (§2.2):** The draft confused the 250 physiological *tokens* with raw waveform samples, writing $T_p = 250$, $F_s = 31.25\text{ Hz}$, and $N_{\text{fft}} = 256$ (and misstating bin index $k = 5$, which is $0.610\text{ Hz} > 0.50\text{ Hz}$). Under ADR-007 geometry, the continuous waveform has $L = 8000$ samples at $F_s = 1000\text{ Hz}$ ($250$ tokens $\times$ $32$ samples/token); ADR-008 operates directly on `[B, L]` at the declared $F_s$ with auto zero-padded $N_{\text{fft}} = 4 \cdot 2^{\lceil\log_2 L\rceil}$ ($32768$ at $L = 8000$).
3. **Mode-2 zero-filled prediction hazard (§2.2.2–§2.2.3):** Computing Pearson or rFFT directly on predictions whose unmasked slots are zero-filled corrupts Mode-2 losses; ADR-008 mandates masked-only gathering for $\mathcal{L}_{\text{Pearson}}$ and detached-target splicing for $\mathcal{L}_{\text{spectral}}$.
4. **Spectral loss reduction and respiratory passband (§2.2.3):** The draft's equation averaged across frequency bins while its reference code summed across bins, and used $[0.15, 0.50]\text{ Hz}$. ADR-008 standardizes on the bin-count-invariant **sum** (total variation L1) over the canonical $[0.10, 0.60]\text{ Hz}$ passband (`ADR-009-resp-waveform-evaluation-protocol.md`) with pre-window DC removal and a degenerate-target energy guard.
5. **Masked visual scalar dimension (§1.2.2):** The draft computed $(36 - 6) \times 2 \times 16 \times 16 = 15{,}360$, omitting both $C = 3$ false-colour channels and $G_t = 25$ temporal tubelets per tube. The true masked visual output count is $(36 - S_{\text{tir}}) \times 25 \times 3 \times 2 \times 16 \times 16 \in [1{,}075{,}200,\; 1{,}228{,}800]$ scalars.
6. **Latent tensor layout, anatomy-weighted pooling, mixed precision, and distributed training (§2.3–§2.4):** Resolved the mismatch between prose anatomy-weighted pooling and unweighted reference code via the explicit `align_core_weight` parameter over flat `[B, K_tir, D]` encoder slices, and added mandatory FP32 loss casting, cross-rank negative gathering, and Mode-1 distributed parameter-reduction safety.
7. **Evaluation protocol alignment (§3.2–§3.3):** Separated Stage-2 pretext diagnostics (non-gating) from the normative 5-fold subject-level Stage-3 band-limited Pearson Go/No-Go gate required by `ADR-009-resp-waveform-evaluation-protocol.md` and `ADR-001`.
