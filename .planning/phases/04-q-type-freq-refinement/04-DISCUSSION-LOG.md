# Phase 4: Q, Type & Frequency Refinement - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in 04-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 04-q-type-freq-refinement
**Areas discussed:** Q parameterization, Hungarian cost balance, Metric-gated curriculum, Type accuracy strategy

---

## Q Parameterization

| Option | Description | Selected |
|--------|-------------|----------|
| Deep MLP + log-linear STE | 3-layer MLP matching gain head pattern, output log(Q) directly, STE clamp | ✓ |
| Minimal swap only | Keep single Linear, swap sigmoid→exp for STE clamp | |
| Per-type Q heads | Separate heads per filter type | |

**User's choice:** Deep MLP + log-linear STE (Recommended)
**Notes:** Matches the gain head pattern established in Phase 2. Identity gradients within bounds fix the sigmoid saturation problem.

---

## Hungarian Cost Balance

| Option | Description | Selected |
|--------|-------------|----------|
| Equal unit weights | lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0 | ✓ |
| Range-normalized equalization | Normalize each cost by its range for [0,1] scaling | |
| Partial adjustment | Bump lambda_freq from 0.5 to 1.0 only | |

**User's choice:** Equal unit weights (Recommended)
**Notes:** Simplest fix, directly satisfies FREQ-02 requirement. Slight risk of gain-accurate assignment degradation but acceptable trade-off for balanced freq learning.

---

## Metric-Gated Curriculum

| Option | Description | Selected |
|--------|-------------|----------|
| Per-stage threshold dicts | Each stage defines metric thresholds for all relevant params | ✓ |
| Composite score gating | Single weighted sum threshold per stage | |
| Epoch-primary + metric skip | Epoch counts primary, skip stages if metrics already met | |

**User's choice:** Per-stage threshold dicts (Recommended)
**Follow-up — Threshold source:** Start from baselines (Derived from Phase 1-3 metrics, stored in config.yaml)
**Notes:** Hard epoch cap per stage prevents infinite stall. Thresholds tunable without code changes via config.

---

## Type Accuracy Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Class-balanced + focal loss | Inverse-frequency weights + focal loss for hard samples | ✓ |
| Stronger CE loss only | Increase lambda_type, faster Gumbel annealing | |
| Two-stage classification | Separate classifier then conditioned param regression | |

**User's choice:** Class-balanced + focal loss (Recommended)
**Notes:** Addresses 5:1 type imbalance (peaking 50% vs HP/LP 10%). Focal loss gamma=2.0 focuses on hard-to-classify samples.

---

## Claude's Discretion

- Exact MLP layer initialization
- Focal loss gamma value (start 2.0, tunable)
- How to compute class weights from type_weights config
- Test structure and file organization

## Deferred Ideas

None — discussion stayed within phase scope.
