# Architecture Research

**Domain:** Differentiable DSP blind parametric EQ estimation -- parameter head redesign
**Researched:** 2026-04-05
**Confidence:** HIGH (codebase-verified, root cause analysis confirmed, existing fix attempts analyzed)

## System Overview (Current Architecture)

```
mel_spectrogram (B, n_mels, T)
    |
    v
+-------------------------------------------+
|  FrequencyAwareEncoder                    |
|  +-------------------------------------+  |
|  | 2D SpectralFrontend (3-layer Conv2D)|  |
|  +-------------------------------------+  |
|              |                             |
|  +-------------------------------------+  |
|  | Reshape + 1x1 Conv Projection       |  |
|  +-------------------------------------+  |
|              |                             |
|  +-------------------------------------+  |
|  | FrequencyPreservingTCN (grouped 1D) |  |
|  +-------------------------------------+  |
|              |                             |
|  +-------------------------------------+  |
|  | AttentionTemporalPool                |  |
|  +-------------------------------------+  |
+-------------------------------------------+
    |                        |
    | embedding (B, 128)     | mel_profile (B, n_mels) -- spectral bypass
    v                        v
+-------------------------------------------+
|  MultiTypeEQParameterHead                 |
|                                           |
|  trunk: Linear(128 -> 5*64) -> ReLU      |
|              |                            |
|  +-----------+----------+---------+       |
|  |           |          |         |       |
|  v           v          v         v       |
| gain_mlp  freq_attn  q_head  type_head   |
| (primary) + attn    (trunk)  (trunk+mel)  |
|  |     mel_aux                         |
|  v                                     |
|  gain_blend_gate -> blended gain_db     |
+-------------------------------------------+
    |          |           |          |
    v          v           v          v
  gain_db    freq         q     type_logits
    |          |           |          |
    +----------+-----------+----------+
               |
               v
+-------------------------------------------+
|  DifferentiableBiquadCascade              |
|  BiquadCoeffs -> FreqResponse -> H_mag    |
+-------------------------------------------+
               |
               v
+-------------------------------------------+
|  MultiTypeEQLoss                          |
|  Hungarian matching + weighted sum of     |
|  gain/freq/Q/type/hmag/spectral losses    |
+-------------------------------------------+
```

### Current Parameter Head Data Flow (Gain Path -- the broken part)

```
trunk_out (B, 5, 64)
    |
    +---> gain_mlp(Linear->ReLU->Linear->Tanh) ---> primary_gain (B, 5)
    |                                                    |
    |                                         * gain_output_scale (learnable, init 24)
    |                                                    |
    |                                              primary_gain_scaled
    |                                                    |
    +---> [mel_residual = mel - AvgPool(mel)]            |
    |         |                                          |
    |    attn weighted readout (uses freq attention)     |
    |         |                                          |
    |    gain_mel_aux(Linear->ReLU->Linear->Tanh)        |
    |         |                                          |
    |    * gain_aux_scale (learnable, init 24)           |
    |         |                                          |
    |    aux_gain_scaled                                 |
    |                                                    |
    +---> gain_blend_gate (sigmoid, init ~0.7) <---------+
              |
         blended gain_db
              |
         ste_clamp(-24, 24)
```

## Identified Architectural Failures

### Failure 1: Gain MLP produces underutilized output (CRITICAL)

The current `gain_mlp` uses Tanh as final activation, producing output in [-1, 1] scaled by `gain_output_scale` (learnable, initialized at 24.0). This is better than the original Gaussian readout but still problematic:

- **Tanh gradient attenuation persists** at moderate gains. At gain=18 dB (0.75 of range), gradient is 0.60. At gain=24 dB, gradient is 0.42.
- The `gain_output_scale` being a learnable scalar means the MLP must produce appropriately-scaled raw values for ALL bands simultaneously.
- The aux mel-residual path adds noise. The blend gate starts at 0.7/0.3 but the mel residual amplitude does not deterministically correspond to dB gain (confirmed in root cause analysis Issue 1).

### Failure 2: Q parameterization creates narrow gradient corridor

```python
q = exp(sigmoid(q_raw) * (log(10) - log(0.1)) + log(0.1))
```

This maps sigmoid output [0,1] to Q range [0.1, 10.0] exponentially. The sigmoid derivative peaks at 0.25 (at midpoint), so maximum gradient through Q is 0.25 * (log(10/0.1)) = 0.25 * 4.6 = 1.15. But near the edges (Q near 0.1 or 10.0), the sigmoid gradient drops to near zero, creating dead zones.

### Failure 3: Gumbel-Softmax dilutes gain gradients early in training

During soft forward, biquad coefficients are a convex combination weighted by type probabilities. When type uncertainty is high (temperature=1.0), gain only affects the peaking/shelf types (~60% of coefficient weight). The remaining ~40% of gradient flows into HP/LP coefficient paths where gain is irrelevant, diluting the gain learning signal during the critical early training phase.

### Failure 4: Multi-band product gradient scaling is unstable

```
H_mag_total = product(H_mag_bands, dim=1)
```

The gradient of the product w.r.t. band i is the product of all OTHER bands. If 4 bands each contribute H_mag=2.0, band 0's gradient is amplified 16x. This makes gradient magnitude depend on the other bands' behavior, creating inter-band coupling that prevents stable per-band gain optimization.

## Recommended Architecture: Direct Regression with STE Clamping

### Design Principle

The parameter head should produce parameter values through a **direct regression** path where:
1. The encoder embedding provides sufficient discriminative information
2. A per-band MLP decodes each parameter independently
3. Activation functions avoid gradient attenuation within the operating range
4. Bounding uses STE clamp (already in codebase) rather than saturating activations
5. The mel profile provides frequency-location signal only (not gain magnitude)

### Proposed Gain Head

```
trunk_out (B, 5, 64)
    |
    v
LayerNorm(64)
    |
    v
Linear(64 -> 64) + GELU
    |
    v
Linear(64 -> 1)   (raw gain logit)
    |
    v
raw * max_gain_db   (scale to [-max, +max] range; no tanh)
    |
    v
STE_clamp(-24, 24)  (hard bounds, identity gradient)
    |
    v
gain_db (B, 5)
```

**Why this works:**
- The MLP (64 -> 64 -> 1) has 4160 parameters per band (20,800 total), vs the old Gaussian readout's 10 parameters. This is sufficient capacity to learn the gain mapping.
- GELU activation provides smooth, non-saturating nonlinearity.
- Raw output scaled by `max_gain_db` means the MLP only needs to produce values in roughly [-1, +1] for normal operating range, matching typical Xavier initialization.
- STE clamp provides hard bounds at inference with no gradient loss during training -- the gradient passes through as identity within bounds.
- No mel-residual dependency for gain. The mel profile informs frequency location only.

### Proposed Frequency Head (unchanged from current)

The current frequency attention mechanism is sound. It uses attention over the mel profile with position-aware queries, which is the right approach for blind frequency estimation. The blended attention + direct regression path is appropriate.

**Keep as-is.**

### Proposed Q Head

Replace the sigmoid-to-exponential mapping with a simpler log-linear parameterization:

```python
# Current (narrow gradient corridor):
q = exp(sigmoid(q_raw) * (log(10) - log(0.1)) + log(0.1))

# Proposed (direct log-space regression):
q_log = ste_clamp(q_raw * 2.0, math.log(0.1), math.log(10.0))
q = exp(q_log)
```

The raw linear output scaled by 2.0 maps to the log-Q range [log(0.1), log(10)] = [-2.3, 2.3]. STE clamp provides bounds. No sigmoid means no gradient bottleneck.

### Proposed Type Classification (keep Gumbel-Softmax, adjust temperature schedule)

Gumbel-Softmax is the right approach for differentiable type selection. The issue is not the mechanism but the **gradient interaction with gain during early training**.

**Fix:** Decouple the gain gradient from type uncertainty by computing a **hard-typed gain loss** alongside the soft spectral loss:

```python
# Already partially done in current model_tcn.py forward():
H_mag_hard = self.dsp_cascade(gain_db, freq, q, n_fft, filter_type)  # hard types
H_mag_soft = self.dsp_cascade.forward_soft(gain_db, freq, q, type_probs, n_fft)  # soft

# Use H_mag_hard for parameter regression loss (gain, freq, Q)
# Use H_mag_soft for spectral/H_mag loss (shape matching)
```

This ensures gain gradients flow through the hard-typed path (no type dilution) while spectral gradients flow through the soft path (for differentiable type learning).

## Loss Function Architecture for Multi-Parameter Regression

### Current Problem

The total loss has 10+ components. The gain signal is a small fraction of total gradient magnitude:

```
total = lambda_gain * loss_gain        # 2.0 * ~3.0 = 6.0
      + lambda_freq * loss_freq        # 1.0 * ~2.0 = 2.0
      + lambda_q * loss_q              # 0.5 * ~0.5 = 0.25
      + lambda_type * loss_type        # 0.5 * ~1.5 = 0.75
      + lambda_spectral * loss_spectral # 1.0 * ~0.8 = 0.8
      + lambda_hmag * loss_hmag        # 0.3 * ~1.2 = 0.36
      + lambda_activity * loss_activity # 0.1 * ~0.1 = 0.01
      + lambda_spread * loss_spread    # 0.05 * ~0.3 = 0.015
      + lambda_embed_var * embed_var   # 0.5 * ~0.5 = 0.25
      + lambda_contrastive * contrast  # 0.1 * ~0.5 = 0.05
```

Even with lambda_gain=2.0, the gain loss (6.0) competes with spectral+hmag (1.16) and all other terms. More critically, the spectral loss provides an alternative gradient path that can satisfy the optimizer without fixing gain.

### Recommended Loss Architecture

**Phase the losses.** Not all loss components should be active at all times:

#### Stage 1: Foundation (Epochs 1-5)
| Loss | Weight | Purpose |
|------|--------|---------|
| loss_gain (log-cosh) | 5.0 | Primary: gain convergence |
| loss_freq (Huber, log-space) | 2.0 | Secondary: frequency |
| loss_type (CE) | 1.0 | Type classification |
| loss_hmag_hard | 1.0 | Frequency response via hard types |
| loss_embed_var | 0.3 | Anti-collapse |

**Not active:** spectral, contrastive, spread, activity. These compete with gain and are not needed early.

#### Stage 2: Refinement (Epochs 6-15)
| Loss | Weight | Purpose |
|------|--------|---------|
| loss_gain (log-cosh) | 3.0 | Maintain gain accuracy |
| loss_freq | 2.0 | Continue frequency |
| loss_type | 1.0 | Type classification |
| loss_hmag_soft | 0.5 | Soft spectral matching |
| loss_q (Huber, log-space) | 1.0 | Q now that gain is stable |
| loss_embed_var | 0.1 | Reduced anti-collapse |

#### Stage 3: Full Optimization (Epochs 16+)
All losses active with reduced gain weight since it should be converged.

### Recommended Gain Loss Function: Log-Cosh

Already implemented in `fixes/modified_loss.py`. Properties:
- Near zero error: ~x^2/2 (MSE-like, strong gradients)
- Large error: ~|x| - log(2) (L1-like, robust to outliers)
- No kink at delta (smoother than Huber)
- Avoids the issue where Huber delta=5.0 gives weak gradients for errors < 5 dB (which is where we want the STRONGEST gradients)

### Hungarian Matching: Gain-Weighted Cost

The current matcher uses `lambda_gain=2.0` but frequency cost is in log-octaves (range ~6.9) while gain is in dB (range 48). One octave of frequency error = 1.0 cost; one dB of gain error = 2.0 cost. A 5 dB gain error (2.0*5 = 10.0) roughly equals a 5-octave frequency error. This is reasonable but should be validated.

**Recommendation:** Keep current gain weight in matcher. The fix should focus on the gain prediction mechanism itself, not the matching.

## Component Responsibilities

| Component | Responsibility | Current State | Recommended Change |
|-----------|----------------|---------------|-------------------|
| FrequencyAwareEncoder | Produce discriminative embedding from mel spectrogram | Works (2D front-end + grouped TCN + attention pool) | Keep |
| MultiTypeEQParameterHead.trunk | Shared per-band features from embedding | 128 -> 5*64, ReLU | Add LayerNorm after ReLU |
| gain_mlp | Predict dB gain from trunk features | Tanh + learnable scale | Replace with GELU MLP + STE clamp |
| gain_mel_aux | Auxiliary gain from mel residual | Blend gate 0.7/0.3 | REMOVE entirely -- adds noise |
| freq_attn + freq_direct | Predict center/cutoff frequency | Blended attention + regression | Keep as-is |
| q_head | Predict Q factor | Sigmoid->exp mapping | Replace with log-linear + STE clamp |
| type_head | Classify filter type | Gumbel-Softmax | Keep, use hard types for gain loss |
| DifferentiableBiquadCascade | Differentiable DSP layer | Works well | Keep |
| MultiTypeEQLoss | Training objective | 10 components, equal priority | Phase losses, use log-cosh for gain |

## Gain Activation Functions Comparison

| Activation | Gradient at 50% range | Gradient at 90% range | Bounded | Recommendation |
|------------|----------------------|----------------------|---------|----------------|
| tanh * scale | 0.79 | 0.42 | Yes | AVOID -- saturates |
| STE clamp | 1.0 (identity) | 1.0 (identity) | Hard yes | USE -- no gradient loss |
| softplus-based | ~0.7 | ~0.3 | Soft yes | AVOID -- still attenuates |
| tanh * 1.5*scale | 0.93 | 0.56 | Soft yes | Acceptable fallback |
| Linear + clamp | 1.0 | 1.0 (within bounds) | Hard yes | Equivalent to STE clamp |

**Verdict:** STE clamp (already in codebase as `ste_clamp`) is strictly better than tanh for bounded regression. Use it.

## Data Flow for Proposed Fix

```
mel_spectrogram (B, 128, T)
    |
    v
FrequencyAwareEncoder
    |
    +---> embedding (B, 128)
    +---> mel_profile (B, 128)
    |
    v
trunk: Linear(128 -> 320) -> ReLU -> LayerNorm -> reshape(B, 5, 64)
    |
    +---> gain_path:
    |    Linear(64,64) -> GELU -> Linear(64,1) -> * 24.0 -> STE_clamp(-24,24)
    |    = gain_db (B, 5)
    |
    +---> freq_path:
    |    [unchanged: attention over mel_profile + direct regression blend]
    |    = freq (B, 5)
    |
    +---> q_path:
    |    Linear(64,1) -> * 2.0 -> STE_clamp(log(0.1), log(10)) -> exp
    |    = q (B, 5)
    |
    +---> type_path:
         [unchanged: trunk + mel features -> 3-layer MLP -> Gumbel-Softmax]
         = type_logits, type_probs, filter_type

gain_db, freq, q, type_probs, filter_type
    |
    v
DifferentiableBiquadCascade
    |
    +---> forward_soft(gain, freq, q, type_probs) -> H_mag_soft  (for hmag_loss)
    +---> forward(gain, freq, q, filter_type)     -> H_mag_hard  (for param_loss)
    |
    v
Phased loss computation (gain prioritized early)
```

## Fix Order (Dependencies Between Fixes)

The fixes have strict ordering dependencies. Implementing them out of order will waste training runs.

### Phase 1: Metrics Foundation (no model changes)

**Fix 1a: Hungarian-matched validation metrics**

The reported gain MAE of ~6 dB includes permutation errors. Fix the validation metric first to know the true baseline.

- File: `insight/train.py`, validation loop
- Change: Apply Hungarian matching before computing MAE
- Dependency: None -- pure metric fix
- Expected outcome: Reported MAE may drop to 3-4 dB (metric was inflated)

**Fix 1b: Uniform gain distribution in dataset**

The Beta(2,5) distribution concentrates gains near zero. Switch to uniform for the full-range training stage.

- File: `insight/dataset.py`, `_beta_gain` method
- Change: Use `np.random.uniform(-24, 24)` for gain sampling (or at minimum Beta(2,2))
- Dependency: None
- Expected outcome: Model sees more large-gain examples, reducing systematic underestimation

### Phase 2: Gain Prediction Fix (the critical change)

**Fix 2a: Replace gain head with direct regression + STE clamp**

Remove `gain_mlp` (with Tanh), remove `gain_mel_aux` entirely, remove `gain_blend_gate`. Replace with:

```python
self.gain_head = nn.Sequential(
    nn.LayerNorm(hidden_dim),
    nn.Linear(hidden_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, 1),
)
# In forward:
gain_raw = self.gain_head(trunk_out).squeeze(-1) * 24.0
gain_db = ste_clamp(gain_raw, -24.0, 24.0)
```

- Dependency: Fix 1a (need correct metrics to validate)
- Files: `insight/differentiable_eq.py` (MultiTypeEQParameterHead)
- Expected outcome: Gradient flow to gain improved by 2-3x within operating range

**Fix 2b: Use hard-typed path for parameter loss**

Ensure the gain/freq/Q regression loss uses `H_mag_hard` (hard argmax types), not the soft path. The soft path should only feed `hmag_loss` and `spectral_loss`.

- Dependency: Fix 2a (same commit)
- File: `insight/model_tcn.py` forward method
- Expected outcome: Gain gradients no longer diluted by type uncertainty

### Phase 3: Loss Architecture Fix

**Fix 3a: Phase losses, use log-cosh for gain**

Implement the staged loss schedule described above. Use log-cosh for gain loss. Separate gain from the combined param loss.

- Dependency: Fix 2a (need working gain head before tuning loss)
- File: `insight/loss_multitype.py` (or adopt `fixes/modified_loss.py`)
- Expected outcome: Gain optimization no longer competes with spectral matching

**Fix 3b: Reduce anti-collapse weight after encoder stabilizes**

The `lambda_embed_var=0.5` and `lambda_contrastive=0.1` add noise to parameter head gradients. Reduce after epoch 5 once encoder is stable.

- Dependency: Fix 3a (part of loss restructuring)
- Expected outcome: Cleaner gradient signal to parameter heads

### Phase 4: Q Parameterization Fix

**Fix 4: Replace sigmoid-to-exp Q mapping with log-linear + STE clamp**

Lower priority than gain since Q MAE is currently 0.49 decades (usable). Fix after gain is stable.

- Dependency: Fix 2a (same pattern, but can be done later)
- File: `insight/differentiable_eq.py` (MultiTypeEQParameterHead)
- Expected outcome: Better gradient flow for extreme Q values

### Dependency Graph

```
Fix 1a (metrics) -----> Fix 2a (gain head) -----> Fix 3a (phased loss) -----> Fix 4 (Q head)
Fix 1b (gain dist) --/        |
                          Fix 2b (hard-type path)
                               |
                          Fix 3b (reduce anti-collapse)
```

Fixes 1a and 1b are independent and can be done in parallel. Fix 2a and 2b should be in the same commit. Fix 3a depends on Fix 2a being validated. Fix 4 is lowest priority.

## Patterns to Follow

### Pattern 1: Direct MLP Regression for Bounded Parameters

**What:** Use a small MLP (2 layers) followed by linear scaling and STE clamp for bounded parameter prediction.
**When:** Any parameter with a known range (gain, Q, frequency).
**Why:** Avoids gradient attenuation from saturating activations (tanh, sigmoid).

```python
class DirectRegressionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, min_val, max_val):
        super().__init__()
        self.range = max_val - min_val
        self.center = (max_val + min_val) / 2
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        raw = self.net(x).squeeze(-1)
        # Scale so net only needs to produce [-1, +1] for normal range
        return ste_clamp(raw * (self.range / 2) + self.center, self.min_val, self.max_val)
```

### Pattern 2: Hard-Type Path for Parameter Regression

**What:** Compute frequency response twice -- once with soft types (for spectral losses) and once with hard argmax types (for parameter losses).
**When:** Using Gumbel-Softmax for differentiable type selection with multi-parameter regression.
**Why:** Prevents type uncertainty from diluting gain/Q gradients through the soft coefficient blending.

```python
# In model forward():
H_mag_soft = self.dsp_cascade.forward_soft(gain, freq, q, type_probs, n_fft)
H_mag_hard = self.dsp_cascade(gain, freq, q, n_fft, filter_type=filter_type)

# Param loss uses hard types; spectral loss uses soft types
```

### Pattern 3: Phased Loss Activation

**What:** Enable loss components progressively during training rather than all at once.
**When:** Training with many competing loss terms where early optimization of one blocks another.
**Why:** Prevents the optimizer from finding local minima where easy losses are satisfied at the expense of harder ones.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Mel-Residual Gain Readout

**What:** Reading mel-spectrogram residual amplitude at a frequency location as gain estimate.
**Why it fails:** Mel residual magnitude is not proportional to dB gain. It depends on Q, bandwidth, signal spectrum, and mel bin spacing. A single scale+bias cannot calibrate this.
**Do instead:** Direct MLP regression from encoder trunk features. Let the encoder learn what gain looks like.

### Anti-Pattern 2: Saturating Activations for Bounded Regression

**What:** Using tanh or sigmoid to bound parameter output.
**Why it fails:** These activations attenuate gradients near the boundaries. For gain at 18 dB (75% of range), tanh gradient is only 0.60. At 24 dB it is 0.42. The model needs full gradient signal at all gain values.
**Do instead:** Linear output + STE clamp. Identity gradient within bounds, zero outside (which is acceptable since we want to stay within bounds).

### Anti-Pattern 3: Product-of-Band Gradient Without Normalization

**What:** Computing total frequency response as `product(H_mag_bands, dim=1)`.
**Why it fails:** Gradient for band i scales with the product of all other bands' magnitudes. When other bands have large gains, this amplifies gradient by up to 16x. When they are near unity, gradient is normal. This creates band-dependent gradient noise.
**Do instead:** Compute H_mag loss per-band (L1 between individual band responses) in addition to the total product loss. Per-band loss provides stable, decoupled gradient. Alternatively, use log-space: `log(H_total) = sum(log(H_bands))`, which decomposes the product into a sum.

### Anti-Pattern 4: Mel-Residual Auxiliary Path for Gain

**What:** Blending a trunk-based gain prediction with a mel-residual readout using a learned gate.
**Why it fails:** The mel residual provides noisy, frequency-dependent information that does not correlate with dB gain. The blend gate may learn to trust the mel path (especially if initialized at 0.3), injecting noise into gain predictions.
**Do instead:** Remove the auxiliary path entirely. Use only the trunk-based direct regression. The mel profile is valuable for frequency prediction, not gain.

## Scalability Considerations

| Concern | Current (5 bands) | If extended to 10+ bands |
|---------|-------------------|--------------------------|
| Hungarian matching | O(N^3) per batch, negligible | Still negligible for N<20 |
| Biquad cascade product | 5 bands, gradient ~16x amplification | 10 bands, ~512x amplification -- switch to log-space sum |
| Parameter head params | ~40K (5 bands * 64 hidden) | ~80K, linear scaling |
| Gumbel-Softmax | 5 types, manageable | 5 types still, but more bands = more soft blending |

### Key Scaling Risk

The `product(H_mag_bands, dim=1)` gradient amplification is already problematic at 5 bands. For more bands, use log-space:

```python
# Instead of:
H_total = torch.prod(H_bands, dim=1)

# Use:
log_H_total = torch.log(H_bands + 1e-8).sum(dim=1)
H_total = torch.exp(log_H_total)
```

This decomposes the product into a sum in log-space, giving each band an additive gradient contribution. Apply this change now even for 5 bands.

## Sources

- Carion et al., "End-to-End Object Detection with Transformers" (DETR), ECCV 2020 -- Hungarian matching for set prediction
- Engel et al., "DDSP: Differentiable Digital Signal Processing", ICLR 2020 -- differentiable DSP parameter estimation
- Root cause analysis: `insight/diagnostics/root_cause_analysis.md` (8 identified issues with gradient flow analysis)
- Existing fix attempts: `insight/fixes/gain_fixes.py`, `insight/fixes/modified_head.py`, `insight/fixes/modified_loss.py`
- Current parameter head: `insight/differentiable_eq.py` lines 531-830

---
*Architecture research for: differentiable DSP blind EQ estimation parameter head redesign*
*Researched: 2026-04-05*
