# Domain Pitfalls: Differentiable DSP Blind Parametric EQ Estimation

**Domain:** Differentiable DSP, blind audio parameter estimation, multi-band parametric regression
**Researched:** 2026-04-05
**Scope:** Pitfalls specific to improving accuracy in a differentiable DSP blind EQ estimator that has plateaued across 14+ training runs

---

## Critical Pitfalls

Mistakes that cause training plateaus, silent accuracy inflation, or require fundamental rewrites.

### Pitfall 1: Uncalibrated Readout from Spectral Residual for Gain Prediction

**What goes wrong:** Using the amplitude of the mel-spectral residual (raw mel minus smoothed mel) as the primary signal for gain prediction. The assumption is that a spectral bump/dip magnitude maps deterministically to dB gain. It does not. Mel-bin bandwidth varies with frequency (wider at higher frequencies), the smoothing kernel width is fixed in mel bins but EQ bandwidth varies with Q and center frequency, and HP/LP filters create non-zero mel residuals even though their gain target is 0 dB.

**Why it happens:** The spectral residual visually correlates with gain for simple peaking EQ cases, creating a false sense of validity. Engineers implement it because it provides a strong inductive bias (gain should be related to spectral shape). But the mapping from log-mel residual amplitude to dB gain is nonlinear, frequency-dependent, Q-dependent, type-dependent, and signal-dependent. A single per-band scale+bias (10 parameters total for 5 bands) cannot capture this variation.

**Consequences:** Gain predictions are systematically miscalibrated. The model compensates by tuning the scale/bias to the mean gain, which reduces variance in predictions toward zero. Large gains are systematically underestimated. This was the primary driver of the 6-7 dB gain MAE plateau.

**Prevention:** Use a dedicated regression head (2-layer MLP) from the trunk embedding for gain prediction. The spectral residual can provide auxiliary information but must not be the primary gain source. The MLP should output directly in dB space or in a normalized space with a learnable output scale.

**Warning signs:**
- Gain MAE plateaus while frequency MAE continues to improve
- Predicted gain variance is much lower than target gain variance
- The gain head has very few learnable parameters (scale + bias = 10 params)
- Gain calibration parameters drift slowly and never converge

**Phase:** Must be fixed in Phase 1 (foundation) before any other gain optimization. The existing gain_mlp in the codebase is the right approach but the tanh clamp (see Pitfall 2) and blend gate (see Pitfall 4) may undermine it.

**Confidence:** HIGH -- confirmed by root cause analysis, diagnostic measurements, and codebase inspection of `differentiable_eq.py:762-784`.

---

### Pitfall 2: Tanh Soft-Clamp Gradient Attenuation on Regression Outputs

**What goes wrong:** Using `tanh(x / max_val) * max_val` to clamp predicted parameters (gain, Q) to a valid range. The derivative of this function is `1 - tanh(x / max_val)^2`, which means any prediction approaching the boundary loses gradient signal. For gain at 12 dB with max=24 dB, gradient is attenuated to 0.79. At 18 dB, attenuation is 0.60. At 24 dB, attenuation is 0.42. Even moderate gains lose 20-40% of their gradient signal.

**Why it happens:** Tanh is a familiar bounded activation that prevents unbounded outputs. Engineers apply it to the gain output layer thinking "the valid range is [-24, 24], so tanh ensures this." But the gradient attenuation is insidious because it is worst exactly where the model most needs gradient signal to correct large errors.

**Consequences:** The model can learn small gains easily (near-linear region) but systematically underestimates large gains. Combined with a Beta-concentrated gain distribution (see Pitfall 7), the model converges to predicting safe near-zero values. This creates a plateau that looks like "the model almost works" but cannot be broken by more training or different loss weights.

**Prevention:** Use a straight-through estimator (STE) clamp: forward pass uses hard clamp, backward pass passes gradients through unmodified. Alternatively, use a much wider tanh range (e.g., `tanh(x / 36) * 24`) so the operating range stays in the near-linear region. The codebase's `ste_clamp` function in `differentiable_eq.py:784` is the right fix -- verify it is used consistently for all gain paths.

**Warning signs:**
- Large gain targets (> 12 dB) are systematically underestimated
- Gain gradient magnitudes decrease as predictions approach targets
- Gain MAE improves for small gains but never for large gains
- The gain distribution of predictions is narrower than the target distribution

**Phase:** Phase 1 (foundation). Must be verified before any loss weight tuning, because loss weight tuning cannot compensate for systematic gradient attenuation.

**Confidence:** HIGH -- mathematical analysis of gradient flow confirmed in root cause analysis Issue 2. The codebase already has `ste_clamp` at line 784, but the `gain_mlp` still uses `nn.Tanh()` internally at line 566.

---

### Pitfall 3: Validation Metrics Computed Without Hungarian Matching

**What goes wrong:** Computing validation MAE by directly comparing predicted band i to target band i, without first solving the assignment problem. Since the model's band ordering is arbitrary, predicted band 0 may correspond to target band 2. Direct comparison inflates every metric by the typical inter-band parameter distance. For 5 bands with random ordering, the expected gain MAE from permutation error alone can be several dB.

**Why it happens:** It is easy to write `(pred - target).abs().mean()` and not realize that the Hungarian matching applied during training must also be applied during validation. The training loss function handles matching internally, but the validation metric code is separate and may not replicate this step.

**Consequences:** Reported metrics are misleadingly bad. The model may actually be performing significantly better than reported. This causes engineers to chase phantom problems and apply unnecessary fixes, wasting training runs. Conversely, if matching is applied inconsistently (some metrics matched, others not), different metrics tell contradictory stories about progress.

**Prevention:** Extract the matching logic into a shared utility. Apply identical Hungarian matching in both the training loss and the validation metric computation. Log both matched and unmatched MAE to detect the gap. The codebase has fixed this (train.py:673-696), but any new metric must also use matching.

**Warning signs:**
- Validation MAE is much higher than training loss would suggest
- Gain MAE and frequency MAE have very different sensitivity to the same changes
- Type accuracy is near-random (~20% for 5 types) despite type loss decreasing
- Adding more bands makes validation metrics dramatically worse (permutation error grows)

**Phase:** Phase 0 (pre-requisite). Must be fixed before any metric-driven development. The codebase appears to have this fixed but it should be verified that no new metric paths bypass the matcher.

**Confidence:** HIGH -- confirmed in codebase at `train.py:673-696`, which now applies Hungarian matching. This was root cause analysis Issue 8.

---

### Pitfall 4: Gumbel-Softmax Gradient Dilution of Continuous Parameters

**What goes wrong:** When using soft (weighted) type probabilities during training, the gradient for continuous parameters (gain, Q) flows through a weighted average of all filter types. At high temperature (tau=1.0, 5 filter types), the gradient for peaking-specific parameters is diluted to ~20% of its full strength because the other 4 types (which do not use gain in the same way) receive 80% of the gradient weight.

**Why it happens:** The soft forward pass computes biquad coefficients for all 5 types and takes a convex combination weighted by `type_probs`. This is necessary for differentiability through the type decision. But it means the gain gradient is `sum_k(type_prob_k * d_coeff_k/d_gain)`. For peaking, `d_coeff/d_gain` is significant. For HP/LP, `d_coeff/d_gain` is zero or very small. When type is uncertain early in training, 80% of the gradient goes to types where gain does not matter.

**Consequences:** During curriculum stage 1 (peaking-only, tau=1.0), the model should be learning gain-frequency relationships but only 20% of the gain gradient reaches the peaking coefficients. This slows gain learning dramatically at the exact stage when the foundation should be established. The effect diminishes as temperature drops, but early damage is done.

**Prevention:** During peaking-only curriculum stages, bypass the Gumbel-Softmax and use hard peaking type directly. Only enable soft typing when multiple filter types are introduced. Alternatively, use a straight-through estimator for type selection (hard forward, soft backward) so the gain gradient always flows through the argmax type. A third option: compute gain loss on the peaking-type output only (detaching type uncertainty from gain regression).

**Warning signs:**
- Gain MAE barely improves during early curriculum stages despite frequency improving
- Type probabilities remain near-uniform for many epochs
- Gain gradient magnitude correlates inversely with Gumbel temperature
- Reducing Gumbel temperature early causes type accuracy to drop (premature hardening)

**Phase:** Phase 1 (foundation). The curriculum schedule and Gumbel temperature must be coordinated. The codebase's curriculum in `config.yaml` starts with `gumbel_temperature: 1.0` for peaking-only stages -- this should be irrelevant during peaking-only and can be bypassed entirely.

**Confidence:** HIGH -- mathematical analysis of gradient flow through `compute_biquad_coeffs_multitype_soft` confirms the dilution. This was root cause analysis Issue 5.

---

### Pitfall 5: Loss Component Competition and Weight Imbalance

**What goes wrong:** When 8+ loss terms compete for gradient signal, the optimizer finds local minima where one or two dominant losses are minimized at the expense of the critical loss. Specifically, spectral losses (hmag, MR-STFT) can achieve low values by adjusting frequency and Q to match the overall spectral shape, without correctly predicting gain. The model learns to "paint the right spectrum" with wrong parameters.

**Why it happens:** The spectral loss gradient with respect to gain flows through `10^(gain_db/40)` and then through the product of all band responses (`prod(H_mag_bands)`). The gradient through the product scales with other bands' magnitudes, creating band-dependent gradient amplification. Meanwhile, the direct gain regression loss (Huber) provides a cleaner gradient but may be overwhelmed by the spectral loss if its weight is too low. With 8+ loss terms, manual weight tuning has an exponential search space.

**Consequences:** Training loss decreases steadily but individual parameter MAEs plateau. The model is "good enough" spectrally but individual parameters (especially gain) remain inaccurate. Adding more loss terms or adjusting weights marginally helps but never breaks through, because each new configuration introduces a different set of trade-offs rather than solving the fundamental competition.

**Prevention:** (a) Detach gain from spectral loss computation -- let gain learn purely from regression loss. The codebase already does this (train.py:427-433, `pred_gain.detach()` for hmag). (b) Separate loss weights for gain vs. freq vs. Q rather than a combined `lambda_param`. (c) Reduce spectral loss weight during gain-focused training phases. (d) Monitor per-component gradient magnitudes, not just loss values, to detect when one loss dominates.

**Warning signs:**
- Total loss decreases but individual parameter MAEs plateau
- Loss components move in opposite directions when tuning weights
- Adding more weight to gain loss improves gain MAE but degrades frequency or type accuracy
- The ratio of gradient norms between loss components varies by more than 100x

**Phase:** Phase 1 (foundation). Loss structure must be established before curriculum training. The current codebase already separates gain weight (`lambda_gain=2.0`) and detaches gain from spectral loss, which is correct. But verify that gradient norms are actually balanced.

**Confidence:** HIGH -- confirmed by codebase inspection of `loss_multitype.py` and `train.py:420-433`.

---

### Pitfall 6: Hungarian Matching Cost Weight Misalignment with Loss Weights

**What goes wrong:** The cost matrix used for Hungarian matching (band assignment) and the loss function applied after matching use different weightings. If the matcher strongly prefers frequency-based assignment (high `lambda_freq` relative to `lambda_gain`), two bands with similar frequencies but different gains may be swapped. The regression loss then pushes gain toward the wrong target. The loss still decreases (the assignment is optimal for the matcher's cost function), but the gain MAE reflects the misassignment.

**Why it happens:** The matching cost and the regression loss serve different purposes. The matcher should find the assignment that makes the learning problem easiest (which often means matching by frequency, since frequency is easier to learn). The loss should then push all parameters toward their matched targets. But if the matcher consistently pairs by frequency, gain errors accumulate because the optimizer receives the wrong gain target for some bands.

**Consequences:** Subtle but persistent gain MAE inflation from incorrect band assignments. This is hard to detect because the training loss looks fine (it is computed on matched targets). Only a per-band analysis of assignment accuracy reveals the problem. DETR-style architectures face the same issue and address it with matching cost warmup and denoising training.

**Prevention:** (a) Ramp up the gain weight in the matching cost over training (start frequency-focused, add gain later). (b) Log the percentage of samples where matching differs between a gain-inclusive and gain-exclusive cost. (c) Consider using the same weights for matching cost as for the regression loss. The codebase's `ImprovedHungarianMatcher` in `fixes/modified_loss.py` implements gain-aware matching with `set_gain_weight()` warmup.

**Warning signs:**
- Gain MAE is high even when frequency and Q MAE are low
- Per-band gain error is highly variable (some bands accurate, others very wrong)
- Changing `lambda_freq` in the matcher significantly changes gain MAE without changing frequency MAE
- Matching assignments change frequently across training steps (matching instability)

**Phase:** Phase 1 (foundation). The matcher and loss weights must be co-designed. The `ImprovedHungarianMatcher` in the fixes directory should be evaluated.

**Confidence:** MEDIUM -- the theoretical concern is well-established from DETR literature, but its quantitative impact on this specific system has not been measured. Root cause analysis Issue 4 ranks it MEDIUM-HIGH.

---

### Pitfall 7: Skewed Training Data Distribution for Critical Parameters

**What goes wrong:** Using a `Beta(2,5)` distribution for gain sampling concentrates ~70% of training samples at gains below 7 dB (mean = 6.9 dB). The model sees far more small-gain examples than large-gain examples and converges to predicting near-zero gain as the optimal strategy -- it minimizes expected loss by predicting the prior mean. Even with a balanced distribution like `Beta(2,2)`, uniform sampling over a wide range (-24 to +24 dB) may not provide enough resolution at perceptually important gain values.

**Why it happens:** Audio engineers often use moderate EQ settings in practice, so a concentrated distribution seems realistic. But for training, the model needs to see the full range of gains with sufficient frequency to learn the mapping. The prior distribution of real-world EQ settings should be handled by the loss function or inference-time calibration, not by the training data distribution.

**Consequences:** Systematic underestimation of large gains. The model "plays it safe" by predicting near-zero, which gives low average error on the training distribution but fails on large-gain test cases. This is invisible if validation uses the same skewed distribution -- all metrics look mediocre but the root cause is unclear.

**Prevention:** Use a near-uniform distribution for gain sampling (e.g., `Beta(2,2)` which is already in the current codebase, or even `Uniform(-24, 24)`). If using Beta, use symmetric parameters with shape >= 1. Consider stratified sampling to ensure each gain range is equally represented. Log the empirical gain distribution and compare to the target distribution.

**Warning signs:**
- Predicted gain distribution is narrower than target gain distribution
- Mean predicted gain is close to zero regardless of target gain
- Gain MAE is dominated by large-gain errors
- The model predicts the same gain for all bands (regression toward the mean)

**Phase:** Phase 1 (foundation) for data generation. Must be fixed before training begins. The current codebase uses `Beta(2,2)` which is symmetric and near-uniform -- an improvement over the original `Beta(2,5)`.

**Confidence:** HIGH -- well-understood bias-variance tradeoff in regression with skewed targets. Root cause analysis Issue 7. Codebase inspection confirms `Beta(2,2)` in `dataset.py:217`.

---

### Pitfall 8: Product-of-Bands Gradient Scaling in Frequency Response Loss

**What goes wrong:** When computing the total frequency response as the product of individual band responses (`H_total = prod(H_band_i)`), the gradient of the loss with respect to band i's gain scales with the product of all other bands' responses. If 4 out of 5 bands have high gain (H_mag ~ 2.0), the gradient for the 5th band is amplified by 2^4 = 16x. Conversely, if most bands have low gain (H_mag ~ 1.0), the gradient is at unity scale. This creates wildly different effective learning rates for different bands and different samples.

**Why it happens:** This is a fundamental property of cascaded filter systems -- the total response is a product, not a sum. The gradient chain rule naturally produces this amplification. It is mathematically correct but creates an optimization landscape where the same gain error receives a 16x different gradient depending on what other bands are doing.

**Consequences:** Training instability and inconsistent convergence across bands. Bands that happen to be in a high-gain context receive large gradients and may overshoot. Bands in low-gain contexts receive small gradients and learn slowly. This makes the training plateau resistant to learning rate or loss weight changes because the effective learning rate varies per-sample and per-band.

**Prevention:** (a) Use sum-of-band responses in log-space instead of product: `log(H_total) = sum(log(H_band_i))`. This decomposes the gradient evenly. (b) Normalize the gradient by the number of active bands. (c) Use per-band frequency response loss rather than total response loss. The codebase already detaches gain from the spectral loss (train.py:427-433), which mitigates this for the spectral path, but the hmag loss may still have this issue.

**Warning signs:**
- Training is unstable when multiple bands have large gains
- Gain gradient norms vary by 10x+ across batches
- Adding more bands makes training harder (should not be the case with proper scaling)
- Gradient clipping is needed frequently and at low thresholds

**Phase:** Phase 1 (foundation). The gradient scaling must be understood and addressed before scaling up the number of bands or gain range. Current codebase mitigates by detaching gain from spectral loss.

**Confidence:** MEDIUM -- the mathematical analysis is sound (root cause analysis Issue 6), but the practical impact depends on how often high-gain multi-band configurations appear in training data.

---

## Moderate Pitfalls

### Pitfall 9: Encoder Collapse in Multi-Task Architectures

**What goes wrong:** The TCN encoder produces near-identical embeddings for all inputs (cosine distance ~0.006), making the parameter head's task impossible. The head receives no discriminative information and can only predict parameter averages. This manifests as "the model learns something but not enough" -- frequency may be approximately correct (due to the spectral bypass) but gain is essentially random.

**Why it happens:** The encoder optimizes for a combination of reconstruction/regression losses. If the parameter head can partially solve the task using the spectral bypass (mel_profile passed directly to the head), there is no pressure on the encoder to produce informative embeddings. The encoder finds a shortcut: produce a constant embedding, let the spectral bypass handle frequency, and the parameter head predicts average gain from the bypass. The anti-collapse losses (embed_var, contrastive) help but may not be strong enough.

**Prevention:** Monitor embedding variance per batch. If variance drops below a threshold, increase anti-collapse loss weight. Use gradient-based probes to verify the encoder embedding carries information about parameters (not just the spectral bypass). Consider removing the spectral bypass temporarily to force the encoder to learn.

**Warning signs:**
- Embedding variance near zero (cosine distance < 0.01 between random pairs)
- Performance is similar with and without the encoder (spectral bypass alone achieves similar metrics)
- Anti-collapse losses plateau at non-zero values
- Removing the spectral bypass causes performance to collapse entirely

**Phase:** Phase 2 (encoder quality). After fixing gain mechanism and loss weights, verify that the encoder is actually contributing meaningful information beyond the spectral bypass.

**Confidence:** HIGH -- encoder collapse was diagnosed with cosine distance 0.006. The spectral bypass was explicitly added to mitigate it, which is correct as a safety net but may enable encoder laziness.

---

### Pitfall 10: Matching Instability Across Training Steps (DETR-style Oscillation)

**What goes wrong:** The Hungarian matching can flip between training steps as predictions change slightly. A predicted band that was matched to target band 2 in step t may be matched to target band 3 in step t+1. The regression target for that band changes discontinuously, creating an oscillating optimization target. This is analogous to the well-documented assignment instability in DETR.

**Why it happens:** The matching cost is computed using current model predictions. As the model updates, nearly-equivalent cost assignments can swap. With 5 bands, there are 120 possible permutations. If the cost difference between the optimal and second-optimal assignment is small, even tiny prediction changes can flip the assignment.

**Consequences:** Slower convergence, training loss fluctuations, and sensitivity to hyperparameters. The model receives inconsistent supervision signals, making it harder to learn fine-grained parameter differences. This is particularly harmful for gain, which has the smallest signal-to-noise ratio in the cost matrix.

**Prevention:** (a) Use deterministic tie-breaking in the Hungarian algorithm. (b) Apply matching cost warmup (start with frequency-only matching, gradually add gain). (c) Log the fraction of samples where assignments change between consecutive steps. (d) Consider using soft matching (Sinkhorn) instead of hard Hungarian during early training. From the DETR literature: DN-DETR's denoising training approach stabilizes matching by adding noise to ground-truth targets during training.

**Warning signs:**
- Training loss has high-frequency oscillations (step-to-step variation)
- Matching assignments change for > 10% of samples between consecutive steps
- Validation metrics fluctuate significantly between epochs
- The model converges much slower than expected for the dataset size

**Phase:** Phase 1-2. Monitor from the beginning. If instability is detected, implement warmup or soft matching.

**Confidence:** MEDIUM -- well-documented in DETR literature (DN-DETR, DAB-DETR, Deformable DETR all address this). Specific impact on this system depends on the actual assignment flip rate, which should be measured.

---

### Pitfall 11: Curriculum Stage Boundary Forgetting

**What goes wrong:** When transitioning between curriculum stages (e.g., peaking-only to multi-type), the model catastrophically forgets skills from the previous stage. The peaking gains that were learned in stage 1 degrade when shelf and HP/LP types are introduced, because the loss landscape changes and the optimizer focuses on the new challenges.

**Why it happens:** Neural networks learn incrementally. When the data distribution changes at a stage boundary (new filter types, wider gain range), the model's existing weights are optimized for the old distribution. The optimizer immediately starts adjusting for the new distribution, potentially overwriting the learned representations. Without replay or regularization, the model must re-learn what it already knew.

**Prevention:** (a) Use experience replay: mix a fraction (10-20%) of previous-stage data into each new stage. (b) Apply elastic weight consolidation (EWC) or L2 regularization toward the end-of-stage weights. (c) Use gradual stage transitions rather than hard switches (linearly blend old and new distributions). (d) Reduce learning rate at stage boundaries. The codebase's curriculum already has 5 stages with gradual parameter range expansion, which helps, but there is no explicit replay mechanism.

**Warning signs:**
- Metrics from stage N degrade significantly at the start of stage N+1
- The model needs many epochs at each new stage just to recover previous performance
- Later curriculum stages show no improvement over earlier stages
- Training loss spikes at stage boundaries

**Phase:** Phase 2-3 (curriculum training). Once the foundation is solid, the curriculum schedule must be designed to preserve learning across stages.

**Confidence:** MEDIUM -- catastrophic forgetting in curriculum learning is well-established (Bengio et al., 2009; Kirkpatrick et al., 2017). Specific impact depends on how different the stages are and whether replay is used.

---

### Pitfall 12: bf16 Mixed Precision Gradient Underflow for Small Loss Components

**What goes wrong:** Using bf16 mixed precision training can cause small loss components (e.g., gain regression loss when lambda_gain is small, or anti-collapse losses with low weights) to underflow to zero in bf16 representation. bf16 has only 7 bits of mantissa (vs. 10 for fp16, 23 for fp32), giving a minimum representable positive value of ~1.2e-7. If a loss component's contribution to the total gradient is smaller than this, it is silently dropped.

**Why it happens:** The total loss is computed as a weighted sum of 8+ components. Small-weight components (lambda_spread=0.05, lambda_contrastive=0.1) multiplied by small loss values can produce contributions below bf16 precision. The autocast context may convert intermediate computations to bf16, losing the small signals.

**Prevention:** (a) Accumulate the total loss in fp32 (force the summation to happen in fp32). (b) Use `torch.clamp` on individual loss components to prevent them from going below a minimum threshold. (c) Monitor per-component gradient magnitudes in fp32 vs. bf16 to detect underflow. (d) Consider using fp16 instead of bf16 if the hardware supports it (fp16 has better mantissa precision).

**Warning signs:**
- Loss components with small weights show zero gradient
- Removing small-weight loss components has no effect on training
- Training behaves identically with and without anti-collapse losses
- Gradient norms for specific parameters are consistently zero

**Phase:** Phase 1 (foundation). Precision settings affect all subsequent training. The codebase uses bf16-mixed precision (config.yaml `precision: "16-mixed"`).

**Confidence:** LOW-MEDIUM -- bf16 precision limitations are well-documented, but the specific impact on this system's loss components has not been measured. The concern is plausible but may not be the primary bottleneck.

---

## Minor Pitfalls

### Pitfall 13: Shared Trunk with Competing Parameter Heads

**What goes wrong:** The gain, frequency, Q, and type heads all share the same trunk embedding. When these parameters require different types of information (gain needs amplitude information, frequency needs spectral peak location, type needs spectral shape), the shared representation may be pulled in conflicting directions. The optimizer compromises, producing a representation that is mediocre for all parameters rather than excellent for any one.

**Prevention:** Use separate projection layers for each parameter (already done in the codebase with separate heads). Consider adding a small parameter-specific adapter (1-2 layers) after the shared trunk but before each head. Monitor gradient alignment between heads -- if gradients consistently point in different directions for the shared trunk, the representation is being torn.

**Phase:** Phase 2 (architecture refinement).

**Confidence:** MEDIUM -- multi-task learning gradient conflicts are well-studied (GradNorm, PCGrad). The codebase uses separate heads but shares the trunk.

---

### Pitfall 14: Gradient Checkpointing Hiding Gradient Bugs

**What goes wrong:** Gradient checkpointing (recomputing activations during backward instead of storing them) can hide gradient bugs because it changes the numerical precision and ordering of gradient computation. A bug that manifests without checkpointing (e.g., NaN from a specific operation) may not appear with checkpointing, and vice versa.

**Prevention:** Test gradient flow both with and without checkpointing. If checkpointing is enabled, verify that gradient magnitudes match the non-checkpointed case within a small tolerance.

**Phase:** Ongoing (during debugging).

**Confidence:** LOW -- specific to implementation details, not a domain-level pitfall.

---

### Pitfall 15: Frequency Prediction Coupled to Gain via Attention Readout

**What goes wrong:** The frequency prediction uses attention over the mel profile, and the gain prediction uses attention-weighted readout of the mel residual. If the attention weights (which determine frequency) are wrong, the gain readout samples from the wrong mel bins. This creates a coupling where frequency errors cause gain errors, even though the gain head itself is functioning correctly.

**Prevention:** Use the gain MLP (from trunk embedding) as the primary gain path, independent of the frequency attention mechanism. The current codebase does this with `gain_mlp` but still blends in the attention-based mel readout via `gain_blend_gate`. Consider reducing or eliminating the blend during early training.

**Phase:** Phase 1 (foundation). Verify that the primary gain path does not depend on frequency attention.

**Confidence:** MEDIUM -- the coupling is visible in the codebase at `differentiable_eq.py:770-782`. The blend gate starts at 0.7/0.3 (primary/aux), which means 30% of gain signal still comes from the attention-coupled path.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation | Priority |
|-------------|---------------|------------|----------|
| Gain mechanism replacement | Pitfall 1 (uncalibrated readout), Pitfall 2 (tanh clamp) | Use MLP regression head with STE clamp | CRITICAL |
| Loss weight tuning | Pitfall 5 (loss competition), Pitfall 6 (matcher alignment) | Separate gain weight, monitor gradient norms | HIGH |
| Validation metrics | Pitfall 3 (unmatched metrics) | Apply Hungarian matching to all validation metrics | CRITICAL |
| Gumbel temperature schedule | Pitfall 4 (gradient dilution) | Bypass soft typing during peaking-only stages | HIGH |
| Data distribution | Pitfall 7 (skewed gains) | Use uniform or Beta(2,2) gain distribution | HIGH |
| Curriculum transitions | Pitfall 11 (catastrophic forgetting) | Add replay buffer, gradual stage blending | MEDIUM |
| Encoder quality | Pitfall 9 (encoder collapse) | Monitor embedding variance, probe information content | MEDIUM |
| Multi-band scaling | Pitfall 8 (product gradient scaling) | Use log-space loss, per-band response loss | MEDIUM |
| Training stability | Pitfall 10 (matching oscillation), Pitfall 12 (bf16 underflow) | Monitor assignment flip rate, accumulate loss in fp32 | LOW-MEDIUM |
| Multi-task architecture | Pitfall 13 (competing heads), Pitfall 15 (freq-gain coupling) | Parameter-specific adapters, reduce attention-based gain blend | LOW-MEDIUM |

---

## Most Important Takeaway

The 8 critical and high-priority pitfalls form a cascade:

1. **Pitfall 3** (unmatched metrics) means you cannot trust your measurements
2. **Pitfall 1** (uncalibrated readout) means the gain mechanism is fundamentally broken
3. **Pitfall 2** (tanh clamp) means even a fixed readout would struggle at large gains
4. **Pitfall 4** (Gumbel dilution) means early training gains nothing for gain
5. **Pitfall 7** (skewed data) means the model is incentivized to predict near-zero
6. **Pitfall 5** (loss competition) means spectral losses fight gain regression
7. **Pitfall 6** (matcher weights) means some bands get wrong targets
8. **Pitfall 8** (product scaling) means gradient magnitudes are unpredictable

These 8 issues compound: the model receives weak gain gradients (2,4,5,8), from a broken mechanism (1), with skewed data (7), some bands get wrong targets (6), and you cannot tell because metrics are wrong (3). No single fix breaks the plateau because all 8 must be addressed simultaneously or in the right order.

**Recommended fix order:**
1. Fix validation metrics (Pitfall 3) -- establish ground truth
2. Fix gain mechanism (Pitfall 1) -- make gain learnable at all
3. Fix tanh clamp (Pitfall 2) -- preserve gradient signal
4. Fix data distribution (Pitfall 7) -- present the model with hard examples
5. Fix loss weights and Gumbel schedule (Pitfalls 4, 5) -- let the optimizer work
6. Fix matcher alignment (Pitfall 6) -- give correct targets
7. Address product scaling (Pitfall 8) -- stabilize gradient magnitudes

---

## Sources

- Root cause analysis: `insight/diagnostics/root_cause_analysis.md` (HIGH confidence, project-specific)
- Codebase inspection: `insight/differentiable_eq.py`, `insight/model_tcn.py`, `insight/loss_multitype.py`, `insight/train.py`, `insight/dataset.py` (HIGH confidence)
- Attempted fixes: `insight/fixes/gain_fixes.py`, `insight/fixes/modified_loss.py`, `insight/fixes/modified_head.py` (HIGH confidence, project-specific)
- DETR Hungarian matching instability: Carion et al. (2020), DN-DETR (Li et al., 2022), DAB-DETR (Liu et al., 2022) (HIGH confidence, peer-reviewed)
- Gumbel-Softmax gradient properties: Jang et al. (2017), Maddison et al. (2017) (HIGH confidence, peer-reviewed)
- DDSP failure modes: Engel et al. (ICLR 2020) (MEDIUM confidence, literature)
- Multi-task loss balancing: Kendall et al. (2018), GradNorm (Chen et al., 2018) (HIGH confidence, peer-reviewed)
- Curriculum learning: Bengio et al. (2009), catastrophic forgetting: Kirkpatrick et al. (2017, EWC) (HIGH confidence, peer-reviewed)
- Tanh gradient attenuation: standard calculus, derivative = 1 - tanh(x)^2 (HIGH confidence, mathematical fact)
