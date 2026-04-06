# Phase 5: Inference Refinement & Confidence - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Inference-time optimization pushes accuracy beyond single-pass results, and each band prediction carries a calibrated confidence estimate. This phase adds a `refine_forward()` method to StreamingTCNModel that takes raw single-pass predictions and iteratively refines them using gradient-based optimization through the differentiable biquad cascade. It also adds per-band confidence estimation via type probability + MC dropout uncertainty, calibrated with temperature scaling post-hococ No changes to model encoder architecture or training pipeline.

</domain>

<decisions>
## Implementation Decisions

### Inference-time refinement (INFR-01)
- **D-01:** Gradient-based parameter refinement — use the biquad cascade as a differentiable layer, backprop through predicted params to minimize spectral reconstruction loss (L1 between predicted H_mag and target H_db from a single reference pass). Each step runs a forward + backward pass through the DSP cascade.
- **D-02:** 5-10 refinement steps per inference call. Sufficient for ≥30% gain MAE improvement target without excessive latency. Expected ~50-200ms on GPU.
- **D-03:** Refinement optimization target: spectral reconstruction L1 loss — compare predicted H_mag (from refined params applied to biquad cascade) against the target H_db from the single-pass model prediction. This is self-supervised — no ground truth needed at inference.
- **D-04:** Integration as a method on existing StreamingTCNModel (e.g., `refine_forward()`). Takes the same input as `forward()`, runs single-pass, then refines. Keeps API simple and avoids coupling issues.

### Confidence estimation (INFR-02)
- **D-05:** Per-band confidence = max(type_softmax_probability) as type confidence, Parameter uncertainty via MC dropout (16 forward passes with dropout enabled, variance over predictions). Combines both type certainty and parameter uncertainty.
- **D-06:** Temperature scaling calibration post-hoc — learn a single scalar temperature on a held-out validation set. Scale type logits before softmax to sharpen/soften distributions to match observed accuracy. Standard approach, proven effective.
- **D-07:** Confidence output format: per-band dict with `{type_confidence: float, param_uncertainty: float, overall_confidence: float}`. Type_confidence = max(type_probs after temp scaling), param_uncertainty = std of MC dropout parameter variance, overall_confidence = type_confidence * (1 - normalized_param_uncertainty).

### Streaming support
- **D-08:** Support both batch and streaming (frame-by-frame) modes for refinement. In streaming mode, refinement accumulates state across frames using the existing streaming buffer. Bounded optimization steps count per frame (e.g., max 3 steps per frame in streaming vs 5-10 in batch).
- **D-09:** Streaming refinement reuses the causal buffer already in `_streaming_buffer`. No new state management needed — refinement operates on the embedding + mel_profile that process_frame already produces.

### Claude's Discretion
- Exact optimizer for refinement (Adam vs SGD vs L-BFGS — Adam likely best for few steps)
- Learning rate for refinement steps
- MC dropout implementation details (which layers get dropout, how many passes)
- Temperature scaling validation approach
- Test structure and file organization

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Model architecture
- `insight/model_tcn.py` lines 498-755 — StreamingTCNModel: forward(), init_streaming(), process_frame(), output dict structure
- `insight/model_tcn.py` lines 530-650 — MultiTypeEQParameterHead: param prediction heads, type classification, Gumbel-Softmax
- `insight/differentiable_eq.py` — DifferentiableBiquadCascade: forward(), forward_soft(), full differentiable DSP graph

### Loss / optimization
- `insight/loss_multitype.py` — MultiTypeEQLoss: spectral loss, Hungarian matching, loss components
- `insight/train.py` lines 618-700 — validate(): per-parameter metrics, Hungarian matching in validation, gain_mae computation

### Export / streaming
- `insight/export.py` — ONNX export, model benchmarking, verify_onnx_pytorch_match()
- `insight/model_tcn.py` lines 668-754 — process_frame(): streaming inference path

### Configuration
- `insight/conf/config.yaml` — Loss weights, model params, streaming config

### Prior phase context
- `.planning/phases/04-q-type-freq-refinement/04-CONTEXT.md` — Phase 4 decisions (Q head, focal loss, metric-gated curriculum)
- `.planning/phases/03-loss-architecture-restructuring/03-CONTEXT.md` — Phase 3 decisions (loss warmup, dual forward path)
- `.planning/phases/02-gain-prediction-fix/02-CONTEXT.md` — Phase 2 decisions (gain head cleanup)

- `.planning/phases/01-metrics-data-foundation/01-CONTEXT.md` — Phase 1 decisions (metrics, data)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DifferentiableBiquadCascade.forward()` — Already fully differentiable in PyTorch. Takes (gain_db, freq, q, n_fft, filter_type) → H_mag. This IS the refinement optimization target.
- `process_frame()` streaming path — Produces embedding + mel_profile + params. Refinement can hook into this same path.
- `type_logits` / `type_probs` in model output — Already computed via Gumbel-Softmax. Can be temperature-scaled for confidence.
- `ste_clamp()` (differentiable_eq.py:7) — STE gradient flow already used for gain and Q, can be reused in refinement parameter bounds.

### Established Patterns
- Model returns dict with `params`, `type_logits`, `type_probs`, `filter_type`, `H_mag`, `embedding`
- Standalone test scripts: `test_*.py` with `if __name__ == "__main__"` blocks
- Config-driven hyperparameters via YAML
- Streaming mode: causal buffer + cumulative skip connection mean

### Integration Points
- `StreamingTCNModel.forward()` output dict → input to refinement
- `StreamingTCNModel.process_frame()` output dict → input to streaming refinement
- `DifferentiableBiquadCascade` — refinement optimization target (forward is the differentiable H_mag computation)
- `export.py` — ONNX export will need updating if refinement adds new outputs

</code_context>

<specifics>
## Specific Ideas

- The biquad cascade is already differentiable — refinement is essentially "gradient descent on the spectral reconstruction loss surface". This is the simplest possible refinement approach because no new computation graph needs to be built.
- MC dropout is confidence: enable dropout in the param head MLP layers during inference, run 16 forward passes, compute variance of predictions. Standard technique from deep learning uncertainty estimation.
- Temperature scaling: learn temperature T on validation set by minimizing NLL on type predictions. T is a single scalar that sharpens (T < 1) or softens (T > 1) the softmax distribution.
- STATE.md blocker notes 50-200ms latency for refinement — need to verify acceptable for streaming use case.

</specifics>

<deferred>
## Deferred Ideas

- Conformal prediction intervals (PROD-02) — full prediction intervals, deferred; confidence score is sufficient for v1
- Snapshot ensemble (TRN-01) — could complement gradient refinement but adds complexity
- Feature importance analysis (TRN-02) — interesting for debugging but not part of inference refinement

</deferred>

---

*Phase: 05-inference-refinement-confidence*
*Context gathered: 2026-04-06*
