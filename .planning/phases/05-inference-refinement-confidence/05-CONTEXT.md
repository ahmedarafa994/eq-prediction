# Phase 5: Inference Refinement & Confidence - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Add inference-time parameter refinement and per-band confidence estimation to push accuracy beyond single-pass results. Refinement runs gradient-based optimization through the differentiable DSP layer (freeze encoder, optimize predicted parameters). Confidence uses MC-Dropout variance across multiple stochastic forward passes. Both features are batch-mode only — streaming inference stays single-pass with unchanged latency. No changes to model encoder architecture, loss function, or training pipeline (those are Phases 2-4).

</domain>

<decisions>
## Implementation Decisions

### Refinement approach
- **D-01:** Gradient-based parameter refinement at inference time. Freeze the TCN encoder, take initial predictions as starting point, run N gradient steps through the differentiable DSP layer (DifferentiableBiquadCascade) optimizing a self-supervised spectral consistency loss.
- **D-02:** Refinement optimizes only (gain, freq, Q) parameters — filter type stays at the model's argmax prediction (hard type lock). This avoids discrete type switching during optimization.
- **D-03:** Refinement loss: spectral consistency between predicted H_mag and the observed spectral shape in the mel-spectrogram. The differentiable DSP layer computes H_mag from parameters, enabling gradient flow back to (gain, freq, Q).
- **D-04:** 3-5 gradient refinement steps with small learning rate (~0.01). Encoder is frozen — gradients only flow through the parameter-to-H_mag path (biquad coefficient computation + frequency response). Computationally cheap since the encoder is not re-run.
- **D-05:** Integration as a method on existing StreamingTCNModel (e.g., `refine_forward()`). Takes the same input as `forward()`, runs single-pass, then refines. Keeps API simple.

### Confidence estimation
- **D-06:** MC-Dropout for parameter uncertainty: enable dropout during inference, run N=5 forward passes through the encoder + parameter head. Compute mean prediction (refined params) and variance across passes (uncertainty estimate).
- **D-07:** Type confidence from entropy of mean type_probs across MC-Dropout passes. Low entropy = high confidence in filter type. High entropy = uncertain type prediction.
- **D-08:** Per-band confidence output: for each of the 5 predicted bands, output a confidence dict with {type_entropy, gain_variance, freq_variance, q_variance, overall_confidence}. Overall confidence is a weighted combination of type certainty and parameter variance.
- **D-09:** No architecture changes for confidence — dropout already exists in the parameter heads (0.2 rate in gain/freq MLPs). Just need to ensure dropout layers are present and can be enabled during inference.

### Streaming vs batch separation
- **D-10:** Refinement and MC-Dropout run ONLY in batch/evaluation mode. Streaming mode (init_streaming/process_frame) stays single-pass with no changes to latency (~5.8ms per frame).
- **D-11:** Batch inference pipeline order: (1) single-pass prediction → (2) MC-Dropout for confidence → (3) gradient refinement for accuracy. MC-Dropout runs first to get uncertainty estimates on the unrefined prediction, then refinement improves the point estimate.
- **D-12:** Two inference paths: `model.forward()` gets a `refine=True` option that triggers the full pipeline. `model.process_frame()` remains untouched. Clean separation, no streaming impact.

### Latency budget
- **D-13:** Total batch-mode overhead: ~60-120ms over single-pass. Breakdown: 5 MC-Dropout passes (~50-100ms) + 3-5 gradient steps (~10-20ms). Acceptable for offline analysis and evaluation.
- **D-14:** Target: 30% gain MAE improvement over single-pass (INFR-01). If single-pass is ~3 dB, refined should be ~2.1 dB or better.
- **D-15:** Iteration counts configurable in config.yaml under a `refinement` section (mc_dropout_passes, grad_refine_steps, grad_lr, refine_loss). Defaults: N=5, steps=5, lr=0.01.

### Claude's Discretion
- Exact formulation of the spectral consistency loss for refinement (L1 vs L2 vs composite)
- How to combine type entropy and parameter variance into a single "overall confidence" score
- Whether to use the mean of MC-Dropout predictions or the single-pass prediction as the starting point for refinement
- Calibration validation metrics (reliability diagrams, ECE, Brier score)
- Exact optimizer for refinement (Adam vs SGD — Adam likely best for few steps)
- Test structure and file organization

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Differentiable DSP layer (refinement backbone)
- `insight/differentiable_eq.py` — DifferentiableBiquadCascade.forward() and forward_soft(), StraightThroughClamp (STE clamp), biquad coefficient computation. This is the gradient path for refinement.
- `insight/differentiable_eq.py` lines 562-597 — MultiTypeEQParameterHead with dropout layers (gain MLP, freq MLP, Q MLP, type head). MC-Dropout will use these existing dropout layers.

### Model inference paths
- `insight/model_tcn.py` lines 606-655 — StreamingTCNModel.forward(): batch inference, returns params + H_mag + H_mag_hard + embedding. This is the entry point for batch refinement.
- `insight/model_tcn.py` lines 657-754 — init_streaming() and process_frame(): streaming inference. Must remain UNCHANGED.
- `insight/model_tcn.py` lines 684-690 — BatchNorm eval mode in streaming — relevant for MC-Dropout (must keep eval mode for BatchNorm while enabling dropout).

### Export and evaluation
- `insight/export.py` — ONNX export scope. Refinement is PyTorch-only (not exported to ONNX). Need to understand what's exported to keep ONNX path unchanged.
- `insight/evaluate_model.py` — Evaluation pipeline. Refinement and confidence metrics will extend this.

### Loss / optimization patterns
- `insight/loss_multitype.py` — HungarianBandMatcher and MultiTypeEQLoss. Spectral loss patterns for refinement loss design.
- `insight/loss.py` — MultiResolutionSTFTLoss (MR-STFT). Spectral loss pattern that could inform refinement loss.

### Configuration
- `insight/conf/config.yaml` — Loss weights, model config. Will add `refinement:` section.

### Prior phase context
- `.planning/phases/04-q-type-freq-refinement/04-CONTEXT.md` — Phase 4 decisions (Q head, focal loss, metric-gated curriculum)
- `.planning/phases/03-loss-architecture-restructuring/03-CONTEXT.md` — Phase 3 decisions (dual forward path, warmup, MR-STFT loss)
- `.planning/phases/02-gain-prediction-fix/02-CONTEXT.md` — Phase 2 decisions (STE clamp, mel-residual removal)
- `.planning/phases/01-metrics-data-foundation/01-CONTEXT.md` — Phase 1 decisions (metrics, data distribution)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DifferentiableBiquadCascade.forward()`: Fully differentiable parameter → H_mag computation. This IS the refinement backbone — no new DSP code needed.
- `ste_clamp()` (differentiable_eq.py:32): Already used for gain and Q clamping. Refinement will produce parameters that naturally stay within STE clamp bounds.
- Dropout layers in MultiTypeEQParameterHead: gain MLP (0.2 dropout), freq MLP (0.2 dropout). These exist and can be activated during MC-Dropout inference.
- `HungarianBandMatcher` in loss_multitype.py: Already used for evaluation. Confidence metrics can be computed alongside matched metrics.
- H_mag_hard computation in model_tcn.py:637 — model already computes hard-type frequency response, which refinement can use as initialization.

### Established Patterns
- Model returns dict with `params`, `type_logits`, `type_probs`, `filter_type`, `H_mag`, `embedding`
- Standalone test scripts: `test_*.py` with `if __name__ == "__main__"` blocks
- Config-driven parameters: all hyperparameters in config.yaml
- Two inference modes: forward() for batch, process_frame() for streaming

### Integration Points
- `model_tcn.py` forward() returns dict with params + H_mag + H_mag_hard + embedding — refinement reads params, optimizes them, returns refined params
- `model_tcn.py` process_frame() — must remain untouched for streaming compatibility
- `evaluate_model.py` — extend with confidence metrics and refinement comparison
- `config.yaml` — add `refinement:` section with mc_dropout_passes, grad_refine_steps, grad_lr
- `train.py` validate() — add confidence metric logging (type entropy, param variance)

</code_context>

<specifics>
## Specific Ideas

- The refinement is essentially "test-time training" but scoped to just the 5 band parameters (15 scalar values: 5×gain + 5×freq + 5×Q). The optimization problem is tiny — 3-5 gradient steps through a handful of biquad coefficient formulas. The expensive part is the MC-Dropout encoder passes.
- MC-Dropout in this model is natural because the parameter heads already have dropout. Just need to call selective training mode (enable dropout, keep BatchNorm in eval mode) — a standard pattern.
- Confidence output structured as: `{"band_i": {"type": str, "type_conf": float, "gain_db": float, "gain_std": float, "freq_hz": float, "freq_std": float, "q": float, "q_std": float}}` for downstream consumers.
- The ONNX export should NOT include refinement — it stays in PyTorch. ONNX exports the single-pass model for deployment. Refinement is for analysis/evaluation.

</specifics>

<deferred>
## Deferred Ideas

- Conformal prediction intervals (PROD-02) — full prediction intervals deferred; confidence score is sufficient for v1
- Snapshot ensemble (TRN-01) — could complement gradient refinement but adds complexity
- Feature importance analysis (TRN-02) — interesting for debugging but not part of inference refinement
- Temperature scaling calibration — may add post-hoc if MC-Dropout entropy alone is insufficient for well-calibrated confidence

</deferred>

---

*Phase: 05-inference-refinement-confidence*
*Context gathered: 2026-04-06*
