# Phase 5: Inference Refinement & Confidence - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 05-inference-refinement-confidence
**Areas discussed:** Inference-time refinement, Confidence estimation, Streaming support, Integration approach

---

## Inference-time refinement

| Option | Description | Selected |
|--------|-------------|----------|
| Gradient-based param refinement | Use biquad cascade as differentiable layer, backprop through params to minimize spectral reconstruction loss. ~30-200ms per step. | ✓ |
| Ensemble averaging | Multiple forward passes with perturbed inputs, average predictions. ~1-5ms. | |
| Temperature scaling sharpening | Test-time augmentation from type logits temperature scaling. No iterative optimization. | |

**User's choice:** Gradient-based param refinement
**Notes:** User selected recommended option. Biquad cascade is already differentiable in PyTorch, making this approach natural.

**Refinement steps:** 5-10 steps (selected from options)

---

## Confidence estimation

| Option | Description | Selected |
|--------|-------------|----------|
| Type probability + MC dropout uncertainty | Confidence = max(type_softmax) + parameter uncertainty via MC dropout (16 passes). Temperature scaling calibration. | ✓ |
| Type probability only | Single confidence score from type probability. Simpler but no parameter uncertainty. | |
| Analytical uncertainty propagation | Gaussian error propagation through biquad cascade. Most principled but complex. | |

**User's choice:** Type probability + MC dropout uncertainty (recommended)
**Notes:** Combines type certainty with parameter uncertainty. Temperature scaling calibration post-hoc.

---

## Streaming support

| Option | Description | Selected |
|--------|-------------|----------|
| Batch only | Full file analysis only. Simpler. | |
| Batch + streaming | Both modes. Bounded optimization steps in streaming. | ✓ |

**User's choice:** Batch + streaming (recommended)
**Notes:** Reuse existing causal buffer for streaming refinement. Bounded step count per frame (3 steps streaming vs 5-10 batch).

---

## Integration approach

| Option | Description | Selected |
|--------|-------------|----------|
| Method on existing model | refine_forward() on StreamingTCNModel. Simple API. | ✓ |
| Separate module | InferenceRefiner class. Clean separation but more files. | |

**User's choice:** Method on existing model (recommended)
**Notes:** Keeps API simple. Method takes same input as forward(), runs single-pass then refines.

---

## Claude's Discretion

- Exact optimizer for refinement (Adam vs SGD)
- Learning rate for refinement
- MC dropout implementation details
- Temperature scaling validation approach
- Test structure and file organization
