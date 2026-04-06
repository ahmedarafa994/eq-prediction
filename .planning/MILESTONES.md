# Milestones

## v1.0 IDSP EQ Estimator Accuracy Improvement (Shipped: 2026-04-06)

**Phases completed:** 5 phases, 12 plans, 26 tasks

**Key accomplishments:**

1. Fixed validation measurement: Hungarian-matched metrics, uniform gain distribution, gradient norm monitoring (Phase 1)
2. Removed broken mel-residual gain path, replaced with direct MLP + STE clamp for clean gradient flow (Phase 2)
3. Dual forward path (hard argmax + soft Gumbel), gain-only warmup, log-cosh loss, independent loss weights (Phase 3)
4. Q 3-layer MLP with STE clamp, class-balanced focal loss for rare filter types, metric-gated curriculum (Phase 4)
5. Gradient-based parameter refinement and MC-Dropout confidence estimation for inference-time accuracy (Phase 5)

**Known Gaps:**

- All 27 v1 requirements show "Pending" in traceability table — implementation complete but live training verification deferred
- Pre-fix baseline: gain MAE 5.60 dB, type accuracy 48.1% — post-fix metrics require multi-epoch training run
- Validation sign-offs approved pending live training verification

---
