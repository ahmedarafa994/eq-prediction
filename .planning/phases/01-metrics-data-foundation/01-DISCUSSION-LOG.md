# Phase 1: Metrics & Data Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-05
**Phase:** 01-metrics-data-foundation
**Areas discussed:** Validation metric baseline, Test/regression strategy

---

## Validation metric baseline

| Option | Description | Selected |
|--------|-------------|----------|
| Measure first, then fix | Run 1 epoch baseline with current code, capture matched vs raw MAE, then fix | ✓ |
| Fix first, measure after | Fix everything, then measure. Simpler but no before/after | |
| Skip baseline, trust prior runs | Use ~6 dB from prior runs, just verify new metrics | |

**User's choice:** Measure first, then fix
**Notes:** Want clean before/after comparison to quantify each fix's impact

| Option | Description | Selected |
|--------|-------------|----------|
| Full logging | Log all loss components + per-parameter-group gradient norms during validation | ✓ |
| Components only, no gradients | Log loss components but skip per-parameter-group gradient monitoring | |

**User's choice:** Full logging
**Notes:** Want full observability — both loss components during validation and gradient norms broken down by gain/freq/Q/type

## Test/regression strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Unit tests for metrics | Write test_metrics.py verifying Hungarian matching, MAE computation, per-parameter breakdown | ✓ |
| Smoke test only | Skip unit tests, manually verify printed metrics | |
| Unit tests + before/after script | Both unit tests AND comparison script | |

**User's choice:** Unit tests for metrics
**Notes:** Follow existing test_*.py standalone pattern

## Claude's Discretion

- Validation log output formatting
- Gradient norm monitoring code structure
- Whether to keep "raw" unmatched MAE alongside matched

## Deferred Ideas

None — discussion stayed within phase scope
