# Feature Research

**Domain:** Differentiable DSP blind parametric EQ parameter estimation
**Researched:** 2026-04-05
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Techniques any working blind EQ estimator must have. Missing these = system cannot produce reliable results.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Hungarian-matched validation metrics | Users (and developers) cannot trust metrics computed on unmatched band assignments. The current system reports inflated error because bands are compared arbitrarily. | LOW | Pure implementation fix. No architectural change needed. Cost matrix construction in validation loop must mirror training Hungarian matcher. |
| Balanced gain distribution in training data | Beta(2,5) distribution concentrates 80% of gain samples below 2 dB, making the model unable to learn from moderate and high-gain examples. Any working system trains on uniform or perceptually-weighted gain sampling. | LOW | Change sampling from Beta(2,5) to Uniform(-12, 12) or triangular with heavier tails in dataset generation. |
| Gradient-safe activations (no tanh attenuation) | tanh soft-clamp attenuates gradients by 20-40% at moderate gains (3-8 dB), the exact range where accuracy matters most for audio professionals. Any production system uses piecewise-linear or softplus-based clamps. | LOW | Replace tanh-based gain activation with piecewise linear or softplus with learnable bounds. Minimal code change in parameter head. |
| Correct loss weighting (gain priority) | When spectral loss and type classification loss dominate, the gain gradient signal is too weak to learn. Any working parametric estimator gives parameter regression loss equal or greater weight than auxiliary losses. | MEDIUM | Requires systematic ablation: baseline with parameter-only loss, then add spectral and type losses at controlled weights. Not a single config change -- needs a schedule. |
| Perceptual frequency parameterization | Linear frequency spacing wastes model capacity on inaudible high frequencies and under-represents musically critical low-mid range. Log-frequency or Bark-scale spacing is standard in audio ML. | LOW | Already implemented (log-freq parameterization exists). Verify it is active and not bypassed. |
| Stable Gumbel-Softmax gradient flow | During early training when type probabilities are uncertain, Gumbel-Softmax distributes gradients across all filter types, diluting gain and frequency gradients. Any multi-type system must protect parameter gradients from type uncertainty. | MEDIUM | Two approaches from literature: (1) Straight-through estimator with detach trick during warmup, (2) Separate parameter prediction per type branch with type-gated combination. |
| Differentiable biquad cascade with verified gradients | The DSP core must produce correct gradients from frequency response back through filter coefficients to input parameters. Any DDSP system validates gradient correctness with finite-difference checks. | LOW | Already exists. Verify with test_eq.py that gradient flow is numerically correct after any changes. |

### Differentiators (Competitive Advantage)

Techniques that separate a professional-grade system from a basic one. These are where accuracy gains beyond "working" live.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Audio-domain reconstruction loss (auto-encoder approach) | Peladeau & Peeters (2023) showed that training with audio quality metrics produces better perceptual results than parameter-domain loss alone, even when parameter accuracy is similar. Combined parameter + reconstruction loss can break the current plateau. | MEDIUM | Add STFT reconstruction loss between re-synthesized EQ output and target wet signal. The differentiable biquad cascade already supports forward pass; add spectral distance as auxiliary loss. Must be weighted carefully -- parameter loss should remain primary to preserve interpretability. |
| Self-supervised pretraining of encoder | Steinmetz et al. (2022) demonstrated that pretraining the encoder on audio reconstruction before fine-tuning for parameter estimation prevents encoder collapse (the exact issue diagnosed here: cosine distance 0.006). Pretrained encoders produce discriminative embeddings. | HIGH | Two-stage training: (1) Pretrain TCN on masked spectral reconstruction or contrastive audio tasks, (2) Freeze encoder, train parameter head, (3) End-to-end fine-tuning. Requires designing a pretraining objective that produces frequency-sensitive embeddings. |
| Inference-time parameter refinement | ST-ITO (Steinmetz et al., 2024) showed that gradient-free optimization at inference time can improve parameter estimates by 30-50% without any training changes. Given a coarse prediction from the network, run Nelder-Mead or CMA-ES on the differentiable biquad cascade to minimize spectral error. | MEDIUM | Add optional refinement pass after network prediction. Uses the existing differentiable EQ as the objective function. No training changes needed. Trade-off: adds ~50-200ms latency per inference, acceptable for offline use but must be optional for streaming. |
| Band-aware curriculum learning with proper scheduling | The current 5-stage curriculum is designed but poorly calibrated -- transitions happen too early or too late. Diff-MST (Sober et al., 2024) uses metric-triggered curriculum transitions where stage advancement is gated by validation metric thresholds, not epoch counts. | MEDIUM | Replace epoch-based curriculum transitions with metric-gated transitions. Define threshold criteria (e.g., "advance to stage 3 when stage 2 gain MAE < 3 dB for 3 consecutive validations"). Prevents premature difficulty increase. |
| Per-band confidence estimation | Predict calibration confidence for each band's parameters. DiffVox (2025) uses this to weight loss per-band during training and to flag uncertain predictions at inference. Critical for product: users need to know which bands to trust. | MEDIUM | Add a confidence head (sigmoid output per band) trained with expected calibration error loss. Uncertain bands get lower weight in loss during training and are flagged in the UI at inference. |
| Multi-resolution spectral loss with perceptual weighting | Standard STFT loss treats all frequencies equally. Multi-resolution losses (from neural vocoder literature) with perceptual frequency weighting give more gradient signal in musically critical bands. | MEDIUM | Already partially implemented (MultiResolutionSTFTLoss exists). Add perceptual weighting: louder frequency regions get more loss weight. Aligns with how audio professionals evaluate EQ quality. |
| Learned style embedding for genre-dependent EQ priors | NablAFx (2025) showed that a style embedding can capture mixing tendencies and improve parameter estimation by providing informative priors. For EQ estimation, knowing the genre provides strong prior on likely EQ shapes. | HIGH | Add conditional input to parameter head based on genre/style embedding. Requires genre-labeled training data or unsupervised clustering. Deferred to v2. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Switch to spectral model as primary approach | Spectral model achieves 0.20 dB MAE on H_db prediction -- seems like a solved problem | Parametric extraction from predicted H_db via scipy optimization takes 5-30 seconds per sample, far too slow for real-time product use. Also loses gradient flow -- cannot fine-tune end-to-end. | Fix the parametric pipeline. The spectral model's success proves the information exists in the audio; the parametric pipeline just needs to extract it correctly. |
| End-to-end black-box neural EQ (no DSP structure) | Remove the biquad cascade entirely, let a neural network predict EQ directly from audio | Loses interpretability (users cannot see which band does what), loses gradient-based debugging, makes streaming inference harder, and requires vastly more training data. DDSP literature consistently shows that incorporating DSP structure improves data efficiency and generalization. | Keep the differentiable biquad cascade as the decoder. Structure is a feature, not a limitation. |
| New architecture (Transformer, Conformer, etc.) before fixing known bugs | "Maybe a better encoder would solve everything" | The root cause analysis identified 8 specific bugs in the existing pipeline. Changing architecture introduces new unknowns while leaving known problems unsolved. This is the most common failure mode in iterative ML projects. | Fix the 8 identified issues first. Only consider architectural changes if metrics plateau after all known bugs are fixed. |
| Real-time deployment / API serving before accuracy is solved | Stakeholders want a demo | Deploying a system with 6 dB gain MAE creates negative first impressions and wastes engineering time on serving infrastructure that will need to change when the model changes. | Accuracy first, deployment second. Use offline batch evaluation for demos. |
| Multi-band compression or dynamic EQ support | "While we're at it, let's support more effects" | Scope creep. Dynamic effects require time-varying parameter estimation, fundamentally different architecture. Would dilute focus from the core EQ problem. | Parametric EQ only for v1. Dynamic effects are a future product, not a feature addition. |
| Ensemble of models | "Average predictions from multiple models for better accuracy" | 5x inference cost, 5x memory, no streaming support. In a real-time DAW plugin, this is a non-starter. | Single model with inference-time refinement (see differentiators) achieves similar gains without the cost. |
| Weighted random search for hyperparameters | "Maybe the learning rate or batch size is wrong" | The 8 identified issues are structural bugs, not hyperparameter tuning problems. No amount of LR adjustment fixes a broken activation function or incorrect metric computation. | Fix bugs systematically, then tune hyperparameters on the corrected system. |

## Feature Dependencies

```
[Hungarian-matched validation metrics]
    (no dependencies -- standalone fix, must come FIRST)

[Balanced gain distribution]
    (no dependencies -- standalone fix)

[Gradient-safe activations]
    (no dependencies -- standalone fix)

[Correct loss weighting]
    └──requires──> [Hungarian-matched validation metrics]
    └──requires──> [Balanced gain distribution]
    (need correct metrics to measure loss effect; need balanced data so loss signal is meaningful)

[Stable Gumbel-Softmax gradients]
    └──requires──> [Gradient-safe activations]
    (type gradient fix interacts with gain activation -- fix activations first)

[Audio-domain reconstruction loss]
    └──requires──> [Correct loss weighting]
    (adding another loss term before fixing weighting makes the problem worse)
    └──requires──> [Gradient-safe activations]

[Self-supervised encoder pretraining]
    └──requires──> [Audio-domain reconstruction loss]
    (pretraining objective is reconstruction-based; need the reconstruction loss designed first)
    └──enhances──> [All subsequent training]

[Inference-time refinement]
    └──requires──> [Working differentiable biquad cascade] (already exists)
    (independent of training improvements -- can be added anytime after base system works)

[Band-aware curriculum learning]
    └──requires──> [Hungarian-matched validation metrics]
    (metric-gated transitions require correct metrics)
    └──requires──> [Correct loss weighting]

[Per-band confidence estimation]
    └──requires──> [Stable Gumbel-Softmax gradients]
    (confidence is meaningless if type predictions are random)

[Multi-resolution spectral loss]
    └──enhances──> [Audio-domain reconstruction loss]
    (can be added independently but works best alongside reconstruction loss)

[Learned style embedding]
    └──requires──> [All table stakes fixed]
    (adding conditioning to a broken model helps nothing)

[Gumbel-Softmax fix] ──conflicts──> [End-to-end black-box neural EQ]
(structured type selection is only meaningful with structured DSP output)

[Inference-time refinement] ──conflicts──> [Ensemble of models]
(both attempt to improve accuracy at inference; refinement is strictly better for this use case)
```

### Dependency Notes

- **Correct loss weighting requires Hungarian-matched validation metrics:** Without correct metrics, you cannot measure whether a loss weight change helped or hurt. This is why the validation bug must be fixed first -- it is the measurement instrument.
- **Correct loss weighting requires balanced gain distribution:** If training data concentrates gains near zero, the loss signal from moderate/high gains is statistically weak regardless of weighting.
- **Audio-domain reconstruction loss enhances all training:** Peladeau & Peeters showed this loss provides gradient signal that parameter-domain loss alone cannot, because it captures the perceptual effect of parameter interactions across bands.
- **Self-supervised pretraining enhances all subsequent training:** A pretrained encoder that already understands spectral shapes will learn parameter estimation faster and be more robust to the multi-band permutation problem.
- **Gumbel-Softmax fix conflicts with black-box approach:** If you remove the DSP structure, Gumbel-Softmax has no purpose. The fix only makes sense within the differentiable DSP paradigm.
- **Inference-time refinement conflicts with ensemble:** Both add inference cost. Refinement leverages the existing differentiable DSP structure; ensembles ignore it. They serve the same goal but refinement is architecturally aligned.

## MVP Definition

### Launch With (v1)

Minimum viable product -- accurate parametric EQ estimation from wet audio.

- [ ] Hungarian-matched validation metrics -- cannot track progress without correct measurement
- [ ] Balanced gain distribution (Uniform or heavy-tailed) -- training data must represent the target space
- [ ] Gradient-safe activations (replace tanh with piecewise-linear) -- stop gradient attenuation
- [ ] Correct loss weighting (parameter regression >= spectral loss) -- ensure gain gets learning signal
- [ ] Stable Gumbel-Softmax gradient protection during warmup -- prevent type uncertainty from diluting parameter gradients
- [ ] Gain MAE < 1 dB on validation set -- the core value proposition
- [ ] Frequency MAE < 0.25 octaves on validation set -- bands must be at the right frequency
- [ ] Filter type accuracy > 95% -- correct filter shape is essential for parameter interpretation

### Add After Validation (v1.x)

Features to add once core accuracy is proven.

- [ ] Audio-domain reconstruction loss -- breaks accuracy plateau beyond MVP targets
- [ ] Metric-gated curriculum transitions -- stabilizes training across difficulty stages
- [ ] Per-band confidence estimation -- critical for product UX (which bands to trust)
- [ ] Inference-time parameter refinement -- 30-50% accuracy boost with no training changes

### Future Consideration (v2+)

Features to defer until core product is proven.

- [ ] Self-supervised encoder pretraining -- requires designing a pretraining objective and training schedule
- [ ] Learned style embedding for genre priors -- requires labeled or clustered data
- [ ] Multi-resolution spectral loss with perceptual weighting -- marginal gain over base reconstruction loss
- [ ] ONNX export and DAW plugin integration -- deployment concern, not accuracy concern

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Hungarian-matched validation metrics | HIGH | LOW | P1 |
| Balanced gain distribution | HIGH | LOW | P1 |
| Gradient-safe activations | HIGH | LOW | P1 |
| Correct loss weighting | HIGH | MEDIUM | P1 |
| Stable Gumbel-Softmax gradients | HIGH | MEDIUM | P1 |
| Audio-domain reconstruction loss | HIGH | MEDIUM | P2 |
| Inference-time refinement | HIGH | MEDIUM | P2 |
| Metric-gated curriculum transitions | MEDIUM | MEDIUM | P2 |
| Per-band confidence estimation | MEDIUM | MEDIUM | P2 |
| Self-supervised encoder pretraining | HIGH | HIGH | P3 |
| Multi-resolution spectral loss | MEDIUM | MEDIUM | P3 |
| Learned style embedding | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch -- these are the 8 known bugs that must be fixed
- P2: Should have, add after P1 fixes are verified working -- these break the plateau
- P3: Nice to have, future consideration -- these push beyond professional quality

## Competitor Feature Analysis

| Feature | Peladeau & Peeters (2023) | DiffVox (2025) | Diff-MST (2024) | Our Approach |
|---------|---------------------------|-----------------|-----------------|--------------|
| Parameter estimation method | Auto-encoder with audio quality metric optimization | Differentiable parametric EQ with gradient descent from single vocal | End-to-end differentiable mixing console | Differentiable biquad cascade with TCN encoder |
| Loss function | Audio-domain (reconstruction) | Audio-domain (spectral) | Multi-objective (spectral + parameter) | Parameter regression + spectral + type classification |
| Training strategy | Optimization at inference (no separate training) | Gradient descent from target audio | Self-supervised from paired mixes | Curriculum learning (5 stages) |
| Filter type support | Not specified (parametric EQ) | Parametric EQ (shelf, peak) | Full mixing chain | 5 types: peaking, lowshelf, highshelf, highpass, lowpass |
| Multi-band handling | Single EQ chain | Single vocal chain | N/A (full mix) | Hungarian matching for permutation invariance |
| Inference speed | Slow (optimization-based) | Fast (single forward pass + gradient steps) | Real-time | Fast (single forward pass, streaming capable) |
| Known weakness | Slow inference, no streaming | Single source type (vocals only) | Requires paired mixes | Encoder collapse, gain calibration |

### Key Insight from Competitor Analysis

The closest successful system (DiffVox) achieves its results by keeping the problem constrained (vocals only, known effect chain) and using audio-domain optimization. Our system is more ambitious (blind, multi-type, multi-band) but can adopt their validated techniques: audio-domain loss, gradient-based refinement at inference, and structured DSP priors.

The most important takeaway from Peladeau & Peeters is that their auto-encoder approach produces better audio quality than parameter-based approaches even when parameter accuracy is comparable. This suggests our system should optimize for perceptual audio quality (reconstruction loss) alongside parameter accuracy, not parameter accuracy alone.

## Sources

- arXiv:2310.11781 -- Peladeau & Peeters, "Blind Estimation of Audio Effects Parameters Using an Auto-Encoder Approach" (2023). Audio-domain loss superiority over parameter-domain loss.
- arXiv:2504.14735 -- DiffVox, "Differentiable Parametric EQ for Vocal Effects Estimation" (2025). Gradient-based parameter estimation through differentiable EQ.
- arXiv:2407.08889 -- Diff-MST, "Differentiable Mixing Style Transfer" (2024). Self-supervised training with differentiable effects chain.
- arXiv:2410.21233 -- ST-ITO, "Inference-Time Optimization for Audio Effects" (2024). Gradient-free refinement at inference time.
- arXiv:2502.11668 -- NablAFx, "Differentiable Audio Effects Framework" (2025). Framework for differentiable plugin-based effects.
- arXiv:2207.08759 -- Steinmetz et al., "Style Transfer for Audio Effects with Differentiable Signal Processing" (2022). Self-supervised pretraining prevents encoder collapse.
- arXiv:2010.10291 -- Steinmetz et al., "Automatic Multitrack Mixing with Differentiable DSP" (2020). Multi-resolution spectral loss design.
- Project root cause analysis -- 8 identified issues in current system (from `.planning/PROJECT.md`)

---
*Feature research for: differentiable DSP blind parametric EQ estimation*
*Researched: 2026-04-05*
