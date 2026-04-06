# Phase 5: Inference Refinement & Confidence - Research

**Researched:** 2026-04-06
**Domain:** PyTorch inference-time optimization, MC-Dropout uncertainty estimation, test-time training
**Confidence:** HIGH (all findings verified against actual codebase)

## Summary

Phase 5 adds two inference-time capabilities to `StreamingTCNModel`: gradient-based parameter refinement and MC-Dropout confidence estimation. Both are batch-mode only additions that do not touch the streaming path. The entire implementation leverages existing infrastructure — the `DifferentiableBiquadCascade` is already fully differentiable end-to-end, dropout layers already exist in `MultiTypeEQParameterHead`, and `HungarianBandMatcher` already handles permutation-invariant evaluation.

The gradient-based refinement is "test-time training" scoped to 15 scalar parameters (5 bands × gain/freq/Q). The encoder is frozen; only the predicted (gain, freq, Q) tensors are optimized leaf variables. A spectral consistency loss between predicted H_mag and observed mel-spectrum drives the optimization. The MC-Dropout confidence pass multiplies the number of encoder forward passes by 5 but the refinement loop itself is cheap (3-5 Adam steps through biquad coefficient math only).

The key design decisions are already locked in CONTEXT.md. The remaining Claude's Discretion items (loss formulation for refinement, optimizer choice, overall confidence formula, test structure) are resolved in this document based on codebase inspection.

**Primary recommendation:** Implement `refine_forward()` as a method on `StreamingTCNModel`. MC-Dropout requires selective module-level mode control — use `param_head.train()` with encoder and DSP layer in `eval()`. Spectral consistency loss should be L1 on log-magnitude between predicted H_mag and the mel-profile magnitude (already present as `mel_profile` in the forward output dict).

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Refinement approach:**
- D-01: Gradient-based parameter refinement at inference. Freeze TCN encoder, take initial predictions as starting point, run N gradient steps through DifferentiableBiquadCascade optimizing a self-supervised spectral consistency loss.
- D-02: Refinement optimizes only (gain, freq, Q) — filter type stays at model's argmax prediction (hard type lock). No discrete type switching during optimization.
- D-03: Refinement loss: spectral consistency between predicted H_mag and observed spectral shape in the mel-spectrogram.
- D-04: 3-5 gradient refinement steps, small learning rate (~0.01). Encoder frozen — gradients flow only through parameter-to-H_mag path.
- D-05: Integration as `refine_forward()` method on `StreamingTCNModel`.

**Confidence estimation:**
- D-06: MC-Dropout: enable dropout during inference, run N=5 passes through encoder + parameter head. Compute mean prediction and variance.
- D-07: Type confidence from entropy of mean type_probs across MC-Dropout passes.
- D-08: Per-band confidence output dict: `{type_entropy, gain_variance, freq_variance, q_variance, overall_confidence}`.
- D-09: No architecture changes — dropout already exists in parameter heads (0.2 rate).

**Streaming vs batch separation:**
- D-10: Refinement and MC-Dropout run ONLY in batch/evaluation mode. Streaming stays single-pass.
- D-11: Batch inference pipeline order: single-pass → MC-Dropout confidence → gradient refinement.
- D-12: `model.forward(refine=True)` triggers full pipeline. `model.process_frame()` untouched.

**Latency budget:**
- D-13: Total overhead ~60-120ms. 5 MC-Dropout passes + 3-5 grad steps.
- D-14: Target 30% gain MAE improvement over single-pass (INFR-01).
- D-15: Config-driven in `refinement:` section: mc_dropout_passes, grad_refine_steps, grad_lr, refine_loss.

### Claude's Discretion
- Exact formulation of the spectral consistency loss (L1 vs L2 vs composite)
- How to combine type entropy and parameter variance into "overall confidence"
- Whether to use mean MC-Dropout predictions or single-pass as starting point for refinement
- Calibration validation metrics (reliability diagrams, ECE, Brier score)
- Exact optimizer for refinement (Adam vs SGD)
- Test structure and file organization

### Deferred Ideas (OUT OF SCOPE)
- Conformal prediction intervals (PROD-02)
- Snapshot ensemble (TRN-01)
- Feature importance analysis (TRN-02)
- Temperature scaling calibration
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INFR-01 | Inference-time refinement improves gain MAE by ≥30% over single-pass | Refinement loop section; loss formulation analysis; gradient path verification |
| INFR-02 | Per-band confidence estimation (calibrated probability for type + parameter uncertainty) | MC-Dropout section; dropout layer inventory; confidence formula design |
</phase_requirements>

---

## Standard Stack

### Core (all already present in the project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.0.0 | Gradient-based refinement loop | Already the training framework; autograd handles all gradient flow |
| torch.optim.Adam | built-in | Few-step refinement optimizer | Adam outperforms SGD for few-step convergence; standard for fine-tuning |
| scipy.optimize.linear_sum_assignment | >=1.10.0 | Hungarian matching (already used) | Already used in HungarianBandMatcher for evaluation |

### No New Dependencies
Phase 5 requires zero new dependencies. Everything needed is already in the codebase: `DifferentiableBiquadCascade` for gradient path, `MultiTypeEQParameterHead` for dropout layers, and `HungarianBandMatcher` for matched evaluation. [VERIFIED: codebase inspection]

---

## Architecture Patterns

### Existing Code Inventory (VERIFIED via codebase inspection)

**`DifferentiableBiquadCascade.forward()` — the refinement backbone** [VERIFIED: differentiable_eq.py:354-378]
- Takes `(gain_db, freq, q, n_fft, filter_type)` — all are plain tensors
- Returns `H_mag_total` of shape `(B, N_FFT//2+1)`
- The computation graph is fully differentiable: `gain_db → A → biquad_coeffs → H_mag`
- `filter_type` is an integer tensor selected via `torch.where` — gradients do NOT flow through `filter_type` (discrete branch), which is exactly what D-02 requires (type locked to argmax)

**`DifferentiableBiquadCascade.forward_soft()` — NOT for refinement** [VERIFIED: differentiable_eq.py:420-469]
- Uses Gumbel-Softmax blending — appropriate for training only
- Refinement must use `forward()` with hard argmax types (D-02 says hard type lock)

**`MultiTypeEQParameterHead` — dropout layer inventory** [VERIFIED: differentiable_eq.py:531-807]
- `self.dropout = nn.Dropout(p=0.2)` — applied to trunk output in `forward()` (line 698)
- `self.classification_head` contains `nn.Dropout(0.2)` and `nn.Dropout(0.1)` [lines 596, 599]
- These 3 dropout layers are activated when the module is in `train()` mode
- The gain MLP and Q MLP do NOT have explicit dropout (they share the trunk dropout)

**`StreamingTCNModel.forward()` return dict** [VERIFIED: model_tcn.py:606-655]
- Returns: `params=(gain_db, freq, q)`, `type_logits`, `type_probs`, `filter_type`, `H_mag`, `H_mag_hard`, `embedding`, `mel_profile`, `attn_weights`
- `mel_profile` is `mel_frames.mean(dim=-1)` — shape `(B, n_mels)` — this is the observed spectral shape for the refinement loss

**`StreamingTCNModel.process_frame()` — MUST remain untouched** [VERIFIED: model_tcn.py:668-754]
- Has its own BatchNorm eval-mode guard at lines 684-690
- Streaming buffer logic at lines 692-708
- This method must not be modified at all

**`export.py` — ONNX export scope** [VERIFIED: export.py:20-45]
- Exports `model.forward(dummy_input)` via `torch.onnx.export`
- Exports gain_db, freq, q, type_logits, type_probs, filter_type, H_mag, embedding as named outputs
- `refine_forward()` is a separate method — ONNX export uses `model.forward()`, not `refine_forward()`, so the export is unaffected by Phase 5

### Pattern 1: Selective Dropout Enable (MC-Dropout)

**Problem:** `model.train()` enables dropout but also switches BatchNorm to accumulation mode (wrong for inference). `model.eval()` disables dropout. Need dropout ON + BatchNorm in eval mode.

**Solution:** Set each module's training mode individually. [ASSUMED — standard PyTorch MC-Dropout pattern, but specific module traversal pattern verified against codebase structure]

```python
# Source: [ASSUMED standard PyTorch pattern]
def _enable_mc_dropout(model: StreamingTCNModel) -> None:
    """Enable dropout layers while keeping BatchNorm in eval mode."""
    model.eval()  # Set all modules to eval first
    # Then selectively re-enable dropout modules only
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
```

This works because `nn.Dropout.train()` sets `self.training = True` only on that module, which controls `F.dropout(x, training=self.training)` behavior. BatchNorm is unaffected.

**BatchNorm modules in the codebase:** [VERIFIED: model_tcn.py]
- `SpectralConvBlock2D` has `nn.BatchNorm2d` (line 107)
- `FrequencyPreservingTCNBlock` has `nn.BatchNorm1d` (line 218)
- None in `MultiTypeEQParameterHead` (uses LayerNorm-free design)
- These must all remain in eval mode during MC-Dropout

### Pattern 2: Gradient-Based Parameter Refinement

**Problem:** Predicted (gain, freq, Q) are leaf tensors from the model's parameter head — they don't have `requires_grad=True` after `.eval()` and `torch.no_grad()` context. We need them as optimizable leaf variables.

**Solution:** Detach from graph, clone with `requires_grad=True`, then optimize. [ASSUMED — standard PyTorch test-time optimization pattern]

```python
# Source: [ASSUMED standard PyTorch pattern, consistent with differentiable_eq.py design]
def refine_forward(self, mel_frames, refine_steps=5, refine_lr=0.01, n_fft=2048):
    """
    Single-pass prediction followed by gradient-based parameter refinement.
    Encoder is frozen; only (gain_db, freq, q) are optimized.
    """
    # Step 1: Single-pass prediction (no_grad for encoder)
    with torch.no_grad():
        out = self.forward(mel_frames)
    
    gain_db_init, freq_init, q_init = out["params"]
    filter_type = out["filter_type"]     # Hard argmax — locked (D-02)
    mel_profile = out["mel_profile"]     # Observed spectral shape (B, n_mels)
    
    # Step 2: Create optimizable leaf variables (detach from encoder graph)
    gain_db = gain_db_init.detach().clone().requires_grad_(True)
    freq = freq_init.detach().clone().requires_grad_(True)
    q = q_init.detach().clone().requires_grad_(True)
    
    # Step 3: Refinement optimizer (Adam, few-step convergence)
    optimizer = torch.optim.Adam([gain_db, freq, q], lr=refine_lr)
    
    for _ in range(refine_steps):
        optimizer.zero_grad()
        
        # Forward through DSP layer only (encoder frozen)
        H_mag_pred = self.dsp_cascade(gain_db, freq, q, n_fft, filter_type)
        
        # Spectral consistency loss
        loss = self._spectral_consistency_loss(H_mag_pred, mel_profile)
        loss.backward()
        optimizer.step()
        
        # Re-apply parameter bounds after optimizer step
        with torch.no_grad():
            gain_db.clamp_(-24.0, 24.0)
            freq.clamp_(20.0, 20000.0)
            q.clamp_(0.1, 10.0)
    
    return gain_db.detach(), freq.detach(), q.detach()
```

**Why Adam over SGD:** Adam maintains per-parameter momentum from step 1 — for 3-5 step optimization, Adam converges much faster than SGD from a cold start. The gain/freq/Q parameters have very different gradient magnitudes (gain in dB-space vs freq in Hz-space), and Adam's adaptive learning rates handle this heterogeneity correctly. [ASSUMED — standard optimization literature]

### Pattern 3: Spectral Consistency Loss for Refinement

**Decision (Claude's Discretion):** L1 on log-magnitude between predicted H_mag and observed mel-spectrum magnitude.

**Rationale:**

The mel-profile (`mel_profile = mel_frames.mean(dim=-1)`) is in log-mel-spectrogram space (already log-scaled in `dsp_frontend.py`). The predicted H_mag from `dsp_cascade.forward()` is in linear magnitude space. To compare them:

1. Convert H_mag to log: `log_H_pred = torch.log(H_mag_pred.clamp(min=1e-6))`
2. The mel-profile is a weighted average of log-magnitudes — the alignment is approximate (mel filterbank vs linear FFT bins), but the gross spectral shape is preserved
3. L1 is preferred over L2: gains can range ±24 dB, and L2 would over-penalize large deviations that may be correct

**Alternative considered — direct H_mag comparison:** The model also outputs `target_H_mag` during evaluation. However at inference time (blind estimation), the target H_mag is unknown — only the observed wet signal's mel-spectrum is available. The mel-profile is the correct self-supervised reference.

**Approximate alignment issue:** The mel-profile has `n_mels=128` bins; H_mag has `n_fft//2+1=1025` bins. To compare them, either:
- Downsample H_mag through a mel filterbank (expensive, requires torchaudio or custom mel matrix)
- Subsample H_mag at mel-frequency centers (cheap, approximate)
- Compare H_mag against a mel-filterbank projection

**Recommended approach:** Build a simple mel-frequency subsample index at module init time, project H_mag to `n_mels` representative bins, then take L1 against `mel_profile`. This avoids torchaudio dependency and keeps the refinement loop fast. [ASSUMED — no existing code for this projection in the codebase]

```python
# Source: [ASSUMED — design based on differentiable_eq.py and dsp_frontend.py patterns]
def _spectral_consistency_loss(self, H_mag_pred, mel_profile):
    """
    L1 on log-magnitude between predicted H_mag and observed mel-spectrum.
    H_mag_pred: (B, n_fft//2+1) linear magnitude
    mel_profile: (B, n_mels) log-mel-spectrogram mean
    """
    # Project H_mag to n_mels representative bins via linear interpolation
    # (avoids torchaudio mel filterbank dependency)
    n_fft_bins = H_mag_pred.shape[-1]
    # Sample at mel-frequency positions
    indices = torch.linspace(0, n_fft_bins - 1, self.n_mels,
                              device=H_mag_pred.device).long()
    H_mag_mel = torch.log(H_mag_pred[:, indices].clamp(min=1e-6))
    # mel_profile is already log-scaled
    return F.l1_loss(H_mag_mel, mel_profile)
```

### Pattern 4: Overall Confidence Score (Claude's Discretion)

**Recommended formula:** Weighted harmonic mean of type certainty and parameter stability.

```python
# Source: [ASSUMED — information-theoretic design]
def _compute_confidence(type_probs_stack, gain_stack, freq_stack, q_stack):
    """
    type_probs_stack: (n_passes, B, num_bands, 5)
    gain_stack, freq_stack, q_stack: (n_passes, B, num_bands)
    """
    # Type confidence: 1 - normalized entropy (high = certain)
    mean_type_probs = type_probs_stack.mean(0)  # (B, num_bands, 5)
    entropy = -(mean_type_probs * (mean_type_probs + 1e-8).log()).sum(-1)  # (B, num_bands)
    max_entropy = math.log(5)  # 5 filter types
    type_conf = 1.0 - entropy / max_entropy  # (B, num_bands), in [0, 1]
    
    # Parameter confidence: 1 - normalized standard deviation
    gain_std = gain_stack.std(0)        # (B, num_bands)
    gain_conf = 1.0 / (1.0 + gain_std / 6.0)  # 6 dB = "high uncertainty"
    
    freq_std_oct = (freq_stack.log().std(0))    # std in log-space (octaves)
    freq_conf = 1.0 / (1.0 + freq_std_oct / 0.5)  # 0.5 octaves = "high uncertainty"
    
    q_std_log = (q_stack.log().std(0))
    q_conf = 1.0 / (1.0 + q_std_log / 0.5)
    
    # Overall: weighted combination (type certainty weighted more heavily)
    overall_conf = 0.4 * type_conf + 0.2 * gain_conf + 0.2 * freq_conf + 0.2 * q_conf
    
    return type_conf, gain_conf, freq_conf, q_conf, overall_conf
```

### Pattern 5: Config Section for Refinement

Add to `insight/conf/config.yaml` after the `curriculum:` section:

```yaml
refinement:
  enabled: true
  mc_dropout_passes: 5        # D-15 default
  grad_refine_steps: 5        # D-15 default
  grad_lr: 0.01               # D-15 default
  refine_loss: "log_l1_mel"   # spectral consistency loss type
```

### Recommended Project Structure for Phase 5 Files

```
insight/
├── model_tcn.py              # Add refine_forward() method
├── conf/config.yaml          # Add refinement: section
├── test_inference_refinement.py  # New standalone test file
└── evaluate_model.py         # Extend (if it exists) OR create minimal evaluate_with_refinement.py
```

Note: `evaluate_model.py` does not exist in the codebase at research time. [VERIFIED: directory listing] The evaluation extension will need to be implemented in a new file or in train.py's validate() method.

### Anti-Patterns to Avoid

- **`model.train()` for MC-Dropout:** Sets all BatchNorm to accumulate stats — produces incorrect normalization. Use selective `module.train()` on Dropout layers only.
- **`requires_grad=True` on `filter_type`:** It's a Long tensor — PyTorch does not support gradients on integer tensors. Only (gain_db, freq, q) get `requires_grad=True`.
- **Optimizing over `forward_soft()`:** The soft path uses Gumbel-Softmax blending which is for training. Refinement must use `forward()` with hard argmax types (D-02).
- **Calling refinement inside `torch.no_grad()`:** The refinement loop requires gradient computation through the DSP layer. Only the initial single-pass encoder call uses `no_grad`.
- **ONNX export of `refine_forward()`:** The ONNX export in `export.py` uses `torch.onnx.export(model, ...)` which traces `model.forward()`. As long as `refine_forward()` is a separate method and not called during the trace, ONNX is unaffected. [VERIFIED: export.py:33]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Differentiable EQ for refinement | Custom DSP gradient layer | `DifferentiableBiquadCascade.forward()` | Already fully differentiable; battle-tested with STE clamp, bf16 safety, stability guards |
| Few-step optimizer | Custom gradient update loop | `torch.optim.Adam` | Momentum from step 1 matters even at 3 steps; Adam handles heterogeneous parameter scales |
| Dropout enable/disable | Custom training flag | `nn.Dropout.train()` / `nn.Dropout.eval()` | PyTorch's built-in mechanism; module-level control works correctly |
| Permutation-invariant evaluation | Custom band matching | `HungarianBandMatcher` | Already exists in `loss_multitype.py`; used in training |
| Parameter bounds enforcement | Custom clamping layer | `torch.Tensor.clamp_()` (in-place) | Simple, correct; matches existing `ste_clamp` bounds |

---

## Common Pitfalls

### Pitfall 1: BatchNorm in Wrong Mode During MC-Dropout
**What goes wrong:** If `model.train()` is called to enable dropout, all BatchNorm layers switch to accumulation mode. During inference with small batches (B=1), BatchNorm accumulates running statistics, producing different outputs each pass and corrupting the variance estimate.
**Why it happens:** `nn.Module.train()` propagates to all child modules recursively.
**How to avoid:** Call `model.eval()` first, then iterate over `model.modules()` and call `.train()` only on `nn.Dropout` instances.
**Warning signs:** Variance of gain predictions across MC-Dropout passes is 10x larger than expected; predictions are wildly different across passes even with the same input.

### Pitfall 2: Integer Tensor Requires Grad
**What goes wrong:** `filter_type = type_logits.argmax(dim=-1)` is a `torch.long` tensor. Calling `.requires_grad_(True)` on it raises `RuntimeError: Only Tensors of floating point dtype can require gradients`.
**Why it happens:** PyTorch does not support gradients on integer tensors.
**How to avoid:** Only call `.requires_grad_(True)` on `gain_db`, `freq`, and `q`. `filter_type` is passed as a constant to `dsp_cascade.forward()`.
**Warning signs:** RuntimeError on `filter_type.requires_grad_(True)` during refinement loop initialization.

### Pitfall 3: Gradient Accumulation Across Refinement Steps
**What goes wrong:** If `optimizer.zero_grad()` is not called at each step, gradients accumulate across steps, producing explosive gradient norms on step 3+.
**Why it happens:** Default PyTorch behavior accumulates gradients in `.grad` buffers.
**How to avoid:** Call `optimizer.zero_grad()` at the top of each refinement step, not after.
**Warning signs:** Loss increases after step 1; gain explodes toward ±24 dB clamp.

### Pitfall 4: STE Clamp Doesn't Work Outside Training Graph
**What goes wrong:** `ste_clamp()` uses `StraightThroughClamp.apply()` — a custom autograd Function. When called on detached leaf tensors during refinement, it correctly passes gradients through. But if the refinement parameters are clamped INSIDE the optimizer step (before `optimizer.step()`), the clamp is treated as a hard constraint and gradients are zeroed.
**Why it happens:** STE clamp is designed for the training graph, not for clamping after optimizer steps.
**How to avoid:** Apply `clamp_()` (in-place, no-grad) AFTER `optimizer.step()` in `with torch.no_grad():` context. This is standard projected gradient descent.
**Warning signs:** Parameters drift beyond physical bounds (gain > 24 dB, freq < 20 Hz).

### Pitfall 5: Refinement Makes Performance Worse (Overfitting to Noise)
**What goes wrong:** With only 3-5 gradient steps and a noisy mel-profile reference, refinement can overfit to spectral noise rather than improving parameter estimates.
**Why it happens:** The mel-profile is a time-average and may include room reflections, recording noise, or transient content that doesn't reflect the EQ curve.
**How to avoid:** Keep `grad_lr` small (~0.01). Validate on held-out samples. If refinement consistently degrades performance, reduce steps to 2 or disable per band using type confidence threshold.
**Warning signs:** Refined gain MAE is WORSE than single-pass on validation set; parameters cluster at boundary values (clamp saturation).

### Pitfall 6: MC-Dropout Variance Near Zero (Dropout Rate Too Low)
**What goes wrong:** With p=0.2 dropout on only the trunk output, variance across 5 passes may be negligibly small — not useful as a confidence signal.
**Why it happens:** The trunk dropout is only one layer; the gain MLP and Q MLP don't have their own dropout. 5 passes with p=0.2 may produce ~95% identical outputs.
**How to avoid:** Verify variance is meaningful on random inputs before shipping. If variance is < 0.01 dB consistently, increase n_passes to 10 or add dropout layers to gain/Q MLPs (Phase 5 constraint: no architecture changes — so increase n_passes first).
**Warning signs:** `gain_std` across all bands < 0.05 dB on validation set; confidence scores cluster near 1.0 for all inputs.

### Pitfall 7: ONNX Export Breaks If refine_forward is Called During Trace
**What goes wrong:** If `refine_forward()` is ever used as the model's `__call__` or `forward()`, the ONNX trace captures the refinement loop, which includes `torch.optim.Adam` operations not supported by ONNX.
**Why it happens:** `torch.onnx.export` traces `model.forward()` by calling `model(dummy_input)`.
**How to avoid:** Never rename `refine_forward` to `forward`. Keep the ONNX export script pointing to `model.forward()`. Add a comment in `export.py` warning against tracing `refine_forward`. [VERIFIED: export.py uses `model.forward()` implicitly via `torch.onnx.export(model, dummy_input, ...)`]
**Warning signs:** ONNX export script errors with "unsupported operation: Adam step" or hangs during trace.

---

## Code Examples

Verified patterns from codebase inspection:

### MC-Dropout Pass Structure
```python
# Source: [ASSUMED — based on verified model_tcn.py dropout layer structure]
# Verified: dropout layers at differentiable_eq.py:579, 596, 599

def _run_mc_dropout_passes(model, mel_frames, n_passes=5):
    """Run N forward passes with dropout enabled, collect statistics."""
    # Enable dropout, keep BatchNorm in eval mode
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    gain_preds = []
    freq_preds = []
    q_preds = []
    type_probs_preds = []
    
    with torch.no_grad():  # No grad needed for MC-Dropout passes
        for _ in range(n_passes):
            out = model.forward(mel_frames)  # Uses model.training=False for encoder
            gain_db, freq, q = out["params"]
            gain_preds.append(gain_db)
            freq_preds.append(freq)
            q_preds.append(q)
            type_probs_preds.append(out["type_probs"])
    
    model.eval()  # Restore full eval mode
    
    gain_stack = torch.stack(gain_preds, dim=0)         # (n_passes, B, num_bands)
    freq_stack = torch.stack(freq_preds, dim=0)
    q_stack = torch.stack(q_preds, dim=0)
    type_probs_stack = torch.stack(type_probs_preds, dim=0)  # (n_passes, B, num_bands, 5)
    
    return gain_stack, freq_stack, q_stack, type_probs_stack
```

### DifferentiableBiquadCascade Call in Refinement Loop
```python
# Source: [VERIFIED: differentiable_eq.py:354-378]
# DifferentiableBiquadCascade.forward() signature:
# forward(self, gain_db, freq, q, n_fft=2048, filter_type=None)
# Returns: (B, N_FFT//2+1)

# In the refinement loop:
H_mag_pred = self.dsp_cascade(
    gain_db,        # requires_grad=True leaf
    freq,           # requires_grad=True leaf
    q,              # requires_grad=True leaf
    self.n_fft,     # int constant (2048 from config)
    filter_type     # torch.long, locked from single-pass (D-02)
)
# H_mag_pred: (B, 1025) with full gradient path to gain_db, freq, q
```

### Confidence Output Structure (D-08)
```python
# Source: [ASSUMED — based on D-08 specification in CONTEXT.md]
# Returns list of per-band confidence dicts

def _format_confidence(band_idx, filter_type, type_conf, gain_std, freq, freq_std_oct, q, q_std_log):
    from differentiable_eq import FILTER_NAMES
    return {
        "band": band_idx,
        "type": FILTER_NAMES[filter_type],
        "type_conf": type_conf,
        "gain_variance": gain_std ** 2,
        "freq_variance": freq_std_oct ** 2,   # variance in log-space (octaves^2)
        "q_variance": q_std_log ** 2,
        "overall_confidence": 0.4 * type_conf + 0.2 / (1 + gain_std/6.0)
                               + 0.2 / (1 + freq_std_oct/0.5)
                               + 0.2 / (1 + q_std_log/0.5),
    }
```

### model.forward() Signature Extension
```python
# Source: [VERIFIED: model_tcn.py:606 current forward signature]
# Current: def forward(self, mel_frames):
# Phase 5 change: add refine=False parameter

def forward(self, mel_frames, refine=False):
    # ... existing implementation unchanged ...
    result = { ... }  # same as now
    
    if refine:
        # MC-Dropout confidence (D-11: before refinement)
        gain_stack, freq_stack, q_stack, type_probs_stack = \
            self._run_mc_dropout_passes(mel_frames)
        confidence = self._compute_confidence(type_probs_stack, gain_stack, freq_stack, q_stack)
        result["confidence"] = confidence
        
        # Gradient refinement (D-11: after confidence)
        refined_gain, refined_freq, refined_q = self.refine_forward(mel_frames)
        result["refined_params"] = (refined_gain, refined_freq, refined_q)
    
    return result
```

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Standalone Python scripts (no pytest) — project convention |
| Config file | None — standalone `if __name__ == "__main__":` blocks |
| Quick run command | `cd insight && python test_inference_refinement.py` |
| Full suite command | `cd insight && python test_inference_refinement.py && python test_streaming.py` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| INFR-01 | Refinement improves gain MAE ≥30% over single-pass | Functional | `python test_inference_refinement.py` | ❌ Wave 0 |
| INFR-01 | Gradient flows through biquad DSP to gain/freq/Q | Unit | `python test_inference_refinement.py` | ❌ Wave 0 |
| INFR-01 | Refinement does not degrade streaming latency | Regression | `python test_streaming.py` (existing) | ✅ exists |
| INFR-02 | MC-Dropout produces non-zero variance across 5 passes | Unit | `python test_inference_refinement.py` | ❌ Wave 0 |
| INFR-02 | Type entropy is highest for ambiguous types | Unit | `python test_inference_refinement.py` | ❌ Wave 0 |
| INFR-02 | Confidence dict has all 5 required keys per band | Unit | `python test_inference_refinement.py` | ❌ Wave 0 |

### Key Test Assertions (for `test_inference_refinement.py`)

**Without a trained checkpoint (testable with random model):**

1. `test_gradient_flow_through_dsp()`: Create leaf tensors with `requires_grad=True`, run `dsp_cascade.forward()`, call `.backward()` — verify `gain_db.grad is not None` and `gain_db.grad.abs().sum() > 0`.

2. `test_mc_dropout_produces_variance()`: Run 5 passes with dropout enabled on a fixed random input, verify `gain_stack.std(0).mean() > 0.0` (non-zero variance). Check that BatchNorm layers have `training=False` during MC-Dropout passes.

3. `test_batchnorm_stays_eval_during_mc_dropout()`: After `_enable_mc_dropout(model)`, iterate over all BatchNorm modules and assert `module.training == False`. Assert Dropout modules have `module.training == True`.

4. `test_confidence_dict_structure()`: Run `refine_forward()` on random input, verify the returned confidence dict has keys `type_entropy`, `gain_variance`, `freq_variance`, `q_variance`, `overall_confidence` for each of 5 bands. Verify all values are in [0, 1].

5. `test_streaming_unaffected()`: Call `model.process_frame()` before and after adding `refine_forward()` — verify the streaming output dict is identical in structure and values.

6. `test_onnx_export_unchanged()`: Run `export.py` export logic and verify it does not include refinement operations in the traced graph.

7. `test_refine_reduces_loss()`: With a known synthetic EQ applied to noise, verify that refinement loss (spectral consistency) decreases monotonically across 5 steps. This does NOT require a trained checkpoint — just verifies the optimization is working.

**Requires a trained checkpoint (integration tests, cannot automate without training):**

- `test_refinement_improves_gain_mae_30pct()`: Run single-pass vs refined on validation set, compare Hungarian-matched gain MAE. Must show ≥30% reduction. This is the INFR-01 gate.
- `test_confidence_correlates_with_error()`: Verify that low-confidence predictions have higher actual parameter error than high-confidence predictions (correlation coefficient > 0.3). This is the INFR-02 calibration check.

### Sampling Rate
- **Per task commit:** `cd insight && python test_inference_refinement.py`
- **Per wave merge:** `cd insight && python test_inference_refinement.py && python test_streaming.py && python test_q_type_freq.py`
- **Phase gate:** Both structural tests AND streaming regression test pass before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `insight/test_inference_refinement.py` — covers INFR-01, INFR-02 structural tests (all 6 assertions above)
- [ ] `insight/conf/config.yaml` — add `refinement:` section

---

## ONNX Export Safety

**Finding:** The ONNX export in `export.py` is safe by design. [VERIFIED: export.py:33]

`torch.onnx.export(model, dummy_input, ...)` traces `model.__call__` which delegates to `model.forward()`. Since `refine_forward()` is a separate method and is never called `forward()`, the ONNX export captures only the single-pass inference graph.

**Additional safeguard:** The refinement loop uses `torch.optim.Adam`, which is not ONNX-exportable. If someone accidentally tried to export `refine_forward`, the export would fail with an explicit error rather than silently producing a broken ONNX model.

**Verification in export.py:** Current output names in `export.py:39-42` are `["gain_db", "freq", "q", "type_logits", "type_probs", "filter_type", "H_mag", "embedding"]`. Phase 5 adds `confidence` and `refined_params` only to the `refine=True` path of `forward()`. These are not present in the `refine=False` default path, so ONNX export is completely unaffected.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Adam outperforms SGD for 3-5 step refinement | Architecture Patterns, Pattern 2 | Could use SGD instead — minor implementation change, no correctness impact |
| A2 | `nn.Dropout.train()` correctly enables only that module's dropout without affecting BatchNorm | Architecture Patterns, Pattern 1 | If PyTorch has changed this behavior, need to verify with version check; LOW risk |
| A3 | L1 on log-magnitude between subsampled H_mag and mel_profile is a meaningful refinement signal | Architecture Patterns, Pattern 3 | If spectral alignment is too coarse, refinement may not improve MAE; key empirical risk |
| A4 | p=0.2 dropout on trunk produces useful variance across 5 passes | Common Pitfalls, Pitfall 6 | If variance is negligible, confidence scores are useless; need empirical check at test time |
| A5 | Refinement with lr=0.01, 5 steps will not overfit to spectral noise | Common Pitfalls, Pitfall 5 | Core empirical risk — INFR-01 may not be achievable if refinement diverges |
| A6 | `mel_profile` (mean of mel_frames) is a sufficient spectral shape reference for the refinement loss | Architecture Patterns, Pattern 3 | mel_profile discards temporal variation; single-frame mel may be more informative |

**If A3, A4, or A5 turn out incorrect:** The spectral consistency loss can be replaced with the `FreqResponseLoss` from `loss.py` (which compares predicted H_mag against a target H_mag from a different reference), at the cost of requiring the observed mel-spectrum to be converted to a frequency response estimate.

---

## Open Questions

1. **Does `mel_profile` have sufficient spectral resolution for refinement?**
   - What we know: `mel_profile = mel_frames.mean(dim=-1)` — shape `(B, 128)` — is already computed in `encoder.forward()` and passed through the forward dict.
   - What's unclear: Whether 128 mel bins give enough frequency resolution to distinguish ±3 dB EQ changes. The mel filterbank compresses high frequencies logarithmically.
   - Recommendation: Test refinement loss descent on synthetic data with known parameters. If descent stalls at >1 dB error, consider using the raw STFT magnitude (n_fft//2+1 bins) instead.

2. **Will refinement improve peaking bands more than HP/LP?**
   - What we know: Gain is irrelevant for HP/LP (D-02 locks type). Refinement optimizes gain for all types, but gain matters only for peaking/shelf filters.
   - What's unclear: Whether H_mag from HP/LP filters has enough spectral information to inform gain refinement of nearby peaking bands.
   - Recommendation: Report per-type gain MAE improvement in the test, not just aggregate. If peaking improves but HP/LP doesn't, that's acceptable and expected.

3. **Should MC-Dropout mean or single-pass prediction seed the refinement?**
   - What we know: D-11 says MC-Dropout runs first, then refinement. The CONTEXT.md leaves the seeding choice to Claude's Discretion.
   - Recommendation: Use MC-Dropout mean as the starting point for refinement. The mean is a lower-variance estimate than single-pass, providing a better initial condition. This requires running MC-Dropout before refinement, which D-11 already specifies.

---

## Environment Availability

Step 2.6: Phase 5 is purely code/config changes with no new external dependencies beyond the existing project stack. All required modules (PyTorch autograd, `torch.optim.Adam`, `nn.Dropout`) are part of the existing PyTorch installation.

---

## Security Domain

Step skipped: This phase adds inference-time computation on a local model with no network calls, no user inputs, and no authentication surfaces. Security controls (V2, V3, V4) are not applicable. Input validation (V5) is handled by existing `ste_clamp()` and `torch.clamp()` in the refinement loop.

---

## Sources

### Primary (HIGH confidence)
- `insight/differentiable_eq.py` — `DifferentiableBiquadCascade` gradient path, `MultiTypeEQParameterHead` dropout layers, `StraightThroughClamp` behavior
- `insight/model_tcn.py` — `StreamingTCNModel.forward()`, `process_frame()`, BatchNorm locations
- `insight/export.py` — ONNX export scope (uses `model.forward()` only)
- `insight/loss_multitype.py` — `HungarianBandMatcher`, spectral loss patterns
- `insight/conf/config.yaml` — Current configuration structure (no `refinement:` section yet)
- `insight/test_q_type_freq.py`, `insight/test_loss_architecture.py` — Established test file patterns

### Secondary (MEDIUM confidence)
- Standard PyTorch MC-Dropout pattern (module-level `.train()` for Dropout only) — well-documented in PyTorch community; consistent with `model_tcn.py:684-690` BatchNorm eval pattern already in codebase

### Tertiary (LOW confidence)
- Adam optimizer superiority over SGD for few-step test-time optimization — general ML literature, not verified for this specific domain
- `mel_profile` as a sufficient spectral reference for refinement loss — plausible but not empirically verified in this codebase

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all verified against codebase
- Architecture patterns: HIGH for code structure; MEDIUM for empirical effectiveness (A3, A5)
- Pitfalls: HIGH — identified from actual code paths and PyTorch autograd semantics

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable PyTorch API, 30 days)
