# Project Research Summary

**Project:** IDSP EQ Estimator -- Accuracy Improvement
**Domain:** Differentiable DSP, blind parametric EQ parameter estimation
**Researched:** 2026-04-05
**Confidence:** HIGH

## Executive Summary

This is a differentiable DSP system that estimates multi-band parametric EQ parameters (gain, frequency, Q, filter type) from wet audio alone, without access to the dry signal. The domain is well-established in recent literature: DDSP (Engel et al., ICLR 2020) demonstrated that traditional DSP operations can be embedded as differentiable layers, DETR (Carion et al., ECCV 2020) established Hungarian matching for set-prediction problems, and DiffVox (2025) showed gradient-based parameter estimation through differentiable EQ chains. The project's conceptual architecture -- differentiable biquad cascade with TCN encoder and Gumbel-Softmax type selection -- is sound. The accuracy plateau (gain MAE ~6 dB, frequency MAE ~2.4 octaves, type accuracy ~48%) is not caused by a fundamental architectural limitation but by eight specific, codebase-verified bugs in the gain prediction mechanism, loss weighting, activation functions, and data distribution.

The recommended approach is systematic bug-fixing in dependency order, not architectural experimentation. Research across all four areas converges on a single conclusion: the gain prediction head uses tanh activation (attenuating gradients by 20-40% at moderate gains), receives noisy input from a mel-residual auxiliary path, and gets diluted gradients from Gumbel-Softmax type uncertainty. The fix is direct MLP regression with STE clamp, removal of the mel-residual path, and hard-typed forward pass for parameter loss. This must be preceded by fixing validation metrics (currently inflated by missing Hungarian matching) and balanced gain distribution (Beta(2,5) concentrates samples near zero). No new frameworks or libraries are required -- the existing PyTorch 2.8 + scipy stack is appropriate, with only auraloss as an optional addition.

The key risk is that the eight identified issues compound: weak gain gradients flow through a broken mechanism from skewed data with wrong targets and misleading metrics. No single fix breaks the plateau because all eight reinforce each other. The mitigation is strict fix ordering: metrics first (to measure), then data distribution and gain mechanism (to provide signal), then loss architecture (to direct signal), then Q parameterization and curriculum refinement. Research flags this as high-confidence -- pitfalls are derived from 14+ failed training runs, not theoretical speculation.

## Key Findings

### Recommended Stack

The existing technology stack is correct and should be kept without changes. PyTorch 2.8 with bf16-mixed AMP, scipy for Hungarian matching, and the custom differentiable biquad cascade are the right tools for this domain. The custom DifferentiableBiquadCascade implements Robert Bristow-Johnson Audio EQ Cookbook formulas entirely in PyTorch with verified gradient flow -- no external library needed. The only recommended addition is auraloss (Steinmetz, 2020) for MelSTFTLoss, which applies perceptual frequency weighting that could improve spectral matching. The log-cosh loss function already prototyped in `fixes/modified_loss.py` should replace Huber for gain regression due to its smooth MSE-to-L1 transition without the kink at delta.

**Core technologies:**
- PyTorch 2.8+cu128: Differentiable training pipeline with autograd through custom biquad computation, bf16-mixed AMP, torch.compile for kernel fusion -- already in use, no change needed.
- Custom DifferentiableBiquadCascade: 5 filter types with soft/hard type selection and verified gradient flow -- purpose-built for this task, do not replace with external libraries.
- scipy.optimize.linear_sum_assignment: Hungarian matching for permutation-invariant band assignment -- exact algorithm, negligible CPU overhead at batch_size=1024.
- STE clamp (existing in codebase): Straight-through estimator for bounded parameter output -- identity gradient within bounds, strictly superior to tanh for regression.

### Expected Features

**Must have (table stakes) -- P1:**
- Hungarian-matched validation metrics: Current MAE is inflated by unmatched band comparison. Must be fixed first to establish a trustworthy baseline.
- Balanced gain distribution: Switch from Beta(2,5) to Uniform(-24,24) or Beta(2,2) so the model sees the full gain range with equal frequency.
- Gradient-safe activations (STE clamp): Replace tanh-based gain activation with piecewise-linear or STE clamp to stop gradient attenuation at moderate-to-large gains.
- Correct loss weighting with phased schedule: Parameter regression loss must dominate over spectral and auxiliary losses, especially in early training. Phase losses rather than activating all 8+ components simultaneously.
- Stable Gumbel-Softmax gradient protection: During peaking-only curriculum stages, bypass soft typing or use hard-typed path for parameter loss to prevent type uncertainty from diluting gain gradients.

**Should have (competitive) -- P2:**
- Audio-domain reconstruction loss: Peladeau and Peeters (2023) showed reconstruction loss produces better perceptual results than parameter-domain loss alone. Add as auxiliary after core accuracy is established.
- Inference-time parameter refinement: ST-ITO (Steinmetz, 2024) showed 30-50% accuracy improvement via gradient-free optimization at inference using the existing differentiable biquad cascade as the objective.
- Metric-gated curriculum transitions: Replace epoch-based stage advancement with metric-triggered thresholds (e.g., advance when gain MAE < 3 dB for 3 consecutive validations).
- Per-band confidence estimation: Sigmoid confidence head per band trained with expected calibration error loss -- critical for product UX.

**Defer (v2+):**
- Self-supervised encoder pretraining: Requires designing a pretraining objective and training schedule. High value but high complexity.
- Learned style embedding for genre priors: Requires labeled or clustered data. Nice-to-have for future product differentiation.
- ONNX export and DAW plugin integration: Deployment concern, not accuracy concern.

### Architecture Approach

The system uses a FrequencyAwareEncoder (2D spectral front-end, grouped TCN with dilated causal convolutions, attention temporal pooling) feeding a MultiTypeEQParameterHead that predicts per-band gain, frequency, Q, and filter type. The parameter head outputs feed a DifferentiableBiquadCascade for frequency response computation, with a MultiTypeEQLoss combining Hungarian-matched parameter regression, type classification, frequency response matching, and regularization terms. The architecture is sound but the parameter head has three interacting failures in the gain path: (1) tanh activation attenuates gradients, (2) mel-residual auxiliary path injects noise, (3) Gumbel-Softmax soft blending dilutes gain gradients. The fix replaces the gain path with direct MLP regression (LayerNorm, Linear+GELU, Linear, scale, STE clamp), removes the mel-residual path entirely, and uses hard-typed forward pass for parameter regression loss while keeping soft forward for spectral losses.

**Major components:**
1. FrequencyAwareEncoder: Produces discriminative embedding from mel spectrogram via 2D Conv front-end, grouped 1D TCN, and attention temporal pooling. Keep as-is -- addresses previous encoder collapse.
2. MultiTypeEQParameterHead (gain path): Currently broken due to tanh clamp and mel-residual noise. Replace with direct MLP regression + STE clamp. Remove blend gate and auxiliary path.
3. MultiTypeEQParameterHead (frequency, Q, type paths): Frequency attention mechanism is sound (keep). Q head needs sigmoid-to-exp replacement with log-linear + STE clamp. Type head (Gumbel-Softmax) is correct but needs hard-typed dual path for loss computation.
4. DifferentiableBiquadCascade: Differentiable DSP core with verified gradients. Keep as-is.
5. MultiTypeEQLoss: Currently 10+ competing components. Restructure with phased activation (gain-only early, add spectral later) and log-cosh for gain regression.

### Critical Pitfalls

1. **Validation metrics computed without Hungarian matching** -- MAE is inflated by permutation errors, making progress impossible to track. Fix before any model changes. (PITFALLS.md Pitfall 3, HIGH confidence)
2. **Tanh soft-clamp gradient attenuation on gain output** -- 20-40% gradient loss at moderate gains (12-18 dB), the exact range where accuracy matters. Replace with STE clamp. (PITFALLS.md Pitfall 2, HIGH confidence)
3. **Uncalibrated mel-residual readout for gain prediction** -- Mel amplitude does not deterministically correspond to dB gain. Remove auxiliary path, use direct MLP regression only. (PITFALLS.md Pitfall 1, HIGH confidence)
4. **Gumbel-Softmax gradient dilution during early training** -- At temperature 1.0 with 5 types, ~80% of gain gradient flows to types where gain is irrelevant. Use hard-typed path for parameter loss. (PITFALLS.md Pitfall 4, HIGH confidence)
5. **Loss component competition** -- 8+ loss terms let the optimizer satisfy spectral matching without fixing gain. Phase losses so gain regression dominates early training. (PITFALLS.md Pitfall 5, HIGH confidence)

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Metrics and Data Foundation
**Rationale:** Must establish correct measurement and balanced training data before any model changes. Without correct metrics, subsequent fixes cannot be evaluated. Without balanced data, the model is incentivized to predict near-zero gain regardless of architecture.
**Delivers:** Trustworthy validation baseline (likely gain MAE 3-4 dB, not 6 dB), balanced gain distribution across full range.
**Addresses:** Hungarian-matched validation metrics (FEATURES.md P1), balanced gain distribution (FEATURES.md P1).
**Avoids:** Wasting training runs with misleading metrics (PITFALLS.md Pitfall 3), systematic underestimation of large gains (PITFALLS.md Pitfall 7).

### Phase 2: Gain Prediction Fix
**Rationale:** The gain prediction mechanism is the primary accuracy bottleneck. Three interacting failures (tanh clamp, mel-residual noise, Gumbel dilution) must be fixed together in the same commit because partial fixes leave compounding issues.
**Delivers:** Gain MAE drops below 3 dB, gradient flow to gain improved by 2-3x.
**Uses:** STE clamp (already in codebase), direct MLP regression pattern, hard-typed dual-path forward.
**Implements:** New gain head (ARCHITECTURE.md Fix 2a: LayerNorm + Linear/GELU/Linear + scale + STE clamp), hard-typed parameter loss path (ARCHITECTURE.md Fix 2b), removal of mel-residual auxiliary path.
**Avoids:** Gradient attenuation from saturating activations (PITFALLS.md Pitfall 2), mel-residual noise injection (PITFALLS.md Pitfall 1), Gumbel dilution of gain gradients (PITFALLS.md Pitfall 4).

### Phase 3: Loss Architecture Restructuring
**Rationale:** With a working gain mechanism, the loss function must be restructured to prioritize gain regression over spectral matching. The current 10+ competing components allow the optimizer to satisfy easy losses at the expense of gain accuracy.
**Delivers:** Gain MAE approaches 1.5-2 dB, stable training dynamics.
**Uses:** Log-cosh loss for gain (prototyped in fixes/modified_loss.py), phased loss activation schedule, separated parameter vs. spectral loss paths.
**Implements:** Phased loss schedule (ARCHITECTURE.md Fix 3a: gain-heavy early, add spectral/Q later), reduced anti-collapse weight (ARCHITECTURE.md Fix 3b).
**Avoids:** Loss competition where spectral matching satisfies optimizer without fixing gain (PITFALLS.md Pitfall 5), Hungarian matcher cost misalignment (PITFALLS.md Pitfall 6).

### Phase 4: Q Parameterization and Curriculum Refinement
**Rationale:** With gain and loss structure fixed, address Q parameterization (sigmoid-to-exp creates narrow gradient corridor) and improve curriculum scheduling. These are secondary fixes that polish the core mechanism.
**Delivers:** All metrics approaching target thresholds (gain < 1 dB, freq < 0.25 octaves, type > 95%).
**Uses:** Log-linear Q parameterization with STE clamp, metric-gated curriculum transitions.
**Implements:** Q head replacement (ARCHITECTURE.md Fix 4: log-linear + STE clamp instead of sigmoid-to-exp), metric-gated curriculum (FEATURES.md differentiator).
**Avoids:** Q gradient dead zones at extreme values (PITFALLS.md, narrow sigmoid corridor), catastrophic forgetting at curriculum stage boundaries (PITFALLS.md Pitfall 11).

### Phase 5: Accuracy Optimization and Validation
**Rationale:** After core mechanisms work, apply literature-validated techniques to push beyond the plateau to professional quality. These are independent enhancements that compound on a solid foundation.
**Delivers:** Professional quality confirmed across all parameters, confidence estimation per band.
**Uses:** Differentiable biquad cascade for inference-time refinement, auraloss for perceptual spectral losses.
**Implements:** Inference-time parameter refinement (FEATURES.md P2: Nelder-Mead on differentiable EQ), audio-domain reconstruction loss (FEATURES.md P2), per-band confidence estimation.
**Avoids:** Premature deployment of inaccurate model (FEATURES.md anti-feature), ensemble overhead (FEATURES.md anti-feature).

### Phase Ordering Rationale

- Phase 1 must come first because it provides the measurement instrument -- without correct metrics, no other change can be validated. This is the single most common failure mode in iterative ML: chasing improvements you cannot measure.
- Phase 2 depends on Phase 1 and should fix all three gain failures simultaneously. Partial fixes (e.g., only replacing tanh but keeping the mel-residual path) leave compounding issues that prevent plateau breakthrough.
- Phase 3 depends on Phase 2 because loss tuning is meaningless with a broken gain head -- the optimizer will find new local minima that satisfy spectral losses regardless of weight adjustments.
- Phase 4 can partially overlap with Phase 3 since the Q fix follows the same pattern as the gain fix (replace sigmoid with direct regression + STE clamp) but operates on an independent parameter.
- Phase 5 is polish that only makes sense after the core mechanism works. Each technique in this phase is independently validated in the literature and can be added incrementally.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Loss weight schedule is derived from analysis, not empirically validated. The specific lambda values (gain=5.0 early, 3.0 mid) are starting points that need tuning based on observed gradient magnitudes during training. Recommend running per-component gradient norm diagnostics before finalizing weights.
- **Phase 4:** Metric-gated curriculum transitions need threshold criteria defined. Research identified the pattern from Diff-MST (Sober, 2024) but specific thresholds for gain/freq/type advancement require experimentation on the corrected system.
- **Phase 5:** Inference-time refinement latency budget for streaming use cases. The technique adds 50-200ms per sample, which is acceptable for offline but may conflict with real-time requirements. Need to profile against streaming latency targets.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Pure implementation fix. Hungarian matching in validation loop and uniform gain distribution are well-understood changes with no design ambiguity.
- **Phase 2:** The STE clamp vs tanh comparison is definitive (identity gradient vs 0.42-0.60 gradient at operating range). The direct MLP regression pattern is well-established. Implementation guidance in ARCHITECTURE.md is specific enough to proceed without additional research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Existing stack verified from runtime (PyTorch 2.8, scipy 1.11.4). Alternatives (JAX, DDSP, torchaudio in training) evaluated and rejected with clear rationale. Only addition is auraloss (low-risk, well-maintained). |
| Features | MEDIUM | Table stakes are codebase-verified (8 issues from root cause analysis with diagnostic measurements). Differentiators are from literature (Peladeau 2023, DiffVox 2025, ST-ITO 2024) with MEDIUM confidence -- these papers address related but not identical problems, so transfer effectiveness needs empirical validation. |
| Architecture | HIGH | Root cause analysis is thorough -- gradient flow computed analytically for each failure mode. Fix pattern (STE clamp direct regression) is well-established in the literature and already partially implemented in the codebase. Fix ordering derived from dependency analysis between interacting failures. |
| Pitfalls | HIGH | Pitfalls are derived from 14+ failed training runs and detailed code inspection, not theoretical speculation. Each pitfall has been observed in practice with documented warning signs. The cascading failure model (8 issues compounding) explains why individual fixes have failed. |

**Overall confidence:** HIGH

### Gaps to Address

- **Encoder embedding discriminability post-fix:** The root cause analysis identified encoder collapse (cosine distance 0.006) as a historical issue. The 2D spectral front-end was added to address this, but it is unclear whether the encoder now produces sufficiently discriminative embeddings for gain prediction once the head is fixed. This can only be validated empirically after Phase 2 is implemented. If gain MAE does not drop below 3 dB after the gain head fix, encoder quality should be investigated.

- **Log-cosh vs tuned Huber for gain:** Both are theoretically sound for gain regression. Log-cosh has no kink and smooth MSE-to-L1 transition. A Huber with smaller delta (1.0 instead of 5.0) might work equally well. This needs empirical comparison during Phase 3 -- recommend running both for 5 epochs each and comparing gradient norm profiles.

- **Product-of-bands gradient scaling:** The frequency response product `H_total = prod(H_bands)` creates up to 16x gradient amplification at 5 bands. ARCHITECTURE.md recommends switching to log-space sum. This is mathematically correct but changes gradient dynamics for the entire system. Should be evaluated as an optional change during Phase 2 or Phase 3, not as a prerequisite.

- **Exact loss weights for phased schedule:** The recommended weights (lambda_gain=5.0 in Stage 1, etc.) are analytical starting points. These will need empirical tuning based on observed loss magnitudes and gradient norms during training. The research provides the structure (what to prioritize when), not the exact numbers.

- **bf16 underflow for small loss components:** bf16 has only 7 bits of mantissa. Small-weight losses (lambda_spread=0.05, lambda_contrastive=0.1) may underflow. This is a plausible concern but likely not the primary bottleneck. Mitigate by accumulating total loss in fp32 and monitoring per-component gradient magnitudes.

## Sources

### Primary (HIGH confidence)
- Root cause analysis: `insight/diagnostics/root_cause_analysis.md` -- 8 identified issues with gradient flow analysis, project-specific
- Codebase inspection: `insight/differentiable_eq.py`, `insight/model_tcn.py`, `insight/loss_multitype.py`, `insight/train.py`, `insight/dataset.py` -- direct verification of failure modes
- Existing fix attempts: `insight/fixes/gain_fixes.py`, `insight/fixes/modified_loss.py`, `insight/fixes/modified_head.py` -- prototyped solutions analyzed for feasibility
- Engel et al., "DDSP: Differentiable Digital Signal Processing", ICLR 2020 -- foundational differentiable DSP approach
- Carion et al., "End-to-End Object Detection with Transformers" (DETR), ECCV 2020 -- Hungarian matching for set prediction
- PyTorch 2.8 runtime verification: torch==2.8.0+cu128, scipy==1.11.4

### Secondary (MEDIUM confidence)
- Peladeau and Peeters, "Blind Estimation of Audio Effects Parameters Using an Auto-Encoder Approach", arXiv:2310.11781 (2023) -- audio-domain loss superiority
- DiffVox, "Differentiable Parametric EQ for Vocal Effects Estimation", arXiv:2504.14735 (2025) -- gradient-based parameter estimation
- ST-ITO, "Inference-Time Optimization for Audio Effects", arXiv:2410.21233 (2024) -- inference-time refinement
- Diff-MST, "Differentiable Mixing Style Transfer", arXiv:2407.08889 (2024) -- metric-gated curriculum
- Steinmetz et al., "micro-tcn: Efficient Temporal Convolutional Networks for Analog Audio Effect Modeling", AES 2022 -- TCN validation with auraloss
- Steinmetz and Reiss, "auraloss: Audio-focused loss functions in PyTorch", DMRN+15, 2020 -- perceptually-motivated spectral losses

### Tertiary (LOW confidence)
- nnAudio (Cheuk et al., IEEE Access 2020) -- differentiable CQT, only if mel resolution proves insufficient
- NablAFx, arXiv:2502.11668 (2025) -- learned style embedding, deferred to v2+
- bf16 precision underflow concern -- plausible but not empirically verified in this system

---
*Research completed: 2026-04-05*
*Ready for roadmap: yes*
