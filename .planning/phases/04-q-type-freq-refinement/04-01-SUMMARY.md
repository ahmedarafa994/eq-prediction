---
phase: 04-q-type-freq-refinement
plan: 01
subsystem: model
tags: [pytorch, mlp, ste-clamp, q-parameter, gradient-flow]

# Dependency graph
requires:
  - phase: 03-loss-architecture-restructuring
    provides: MultiTypeEQParameterHead with gain MLP + STE clamp pattern
provides:
  - 3-layer Q MLP with log-linear STE clamp output in MultiTypeEQParameterHead
  - Identity gradient flow within [log(0.1), log(10.0)] bounds for Q parameter
affects: [04-02, training, loss]

# Tech tracking
tech-stack:
  added: []
  patterns: [3-layer-mlp-for-parameter-head, log-linear-ste-clamp-output]

key-files:
  created: []
  modified:
    - insight/differentiable_eq.py

key-decisions:
  - "Q parameter uses same STE clamp pattern as gain (per D-01, D-02) - log-linear output with identity gradients within bounds"
  - "3-layer MLP for Q (64->64->64->1) matching gain head depth pattern"

patterns-established:
  - "Parameter head MLP pattern: trunk embedding -> multi-layer MLP -> log-linear output -> STE clamp -> exp() for downstream"

requirements-completed: [QP-01]

# Metrics
duration: 5min
completed: 2026-04-06
---

# Phase 4 Plan 1: Q MLP Summary

**3-layer Q MLP replacing single Linear head with log-linear STE clamp output for identity gradient flow within [0.1, 10.0] bounds**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-06T05:47:54Z
- **Completed:** 2026-04-06T05:53:41Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Replaced single Linear Q head with 3-layer MLP (Linear->ReLU->Linear->ReLU->Linear, 64->64->64->1)
- Replaced sigmoid->exp Q mapping with log-linear + STE clamp, eliminating gradient saturation at extreme Q values
- Q output remains in Q space (not log-Q) for backward compatibility with loss_multitype.py (which computes log(pred_q))
- All existing tests pass (test_eq.py, test_model.py)

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace single Linear Q head with 3-layer MLP and log-linear STE clamp** - `ca95cc4` (feat)

## Files Created/Modified
- `insight/differentiable_eq.py` - MultiTypeEQParameterHead: q_head replaced with q_mlp, forward Q computation uses STE clamp

## Decisions Made
- Log-linear STE clamp pattern applied to Q, matching the gain head approach (per design decisions D-01, D-02)
- Q values are exp(q_log) so downstream loss code (`torch.log(pred_q + 1e-8)`) works unchanged

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Q MLP is ready for training with the improved gradient flow
- Plan 04-02 (focal loss for type classification) can proceed independently
- Old checkpoints with q_head weights will be incompatible - training from scratch required

## Self-Check: PASSED

- FOUND: 04-01-SUMMARY.md
- FOUND: insight/differentiable_eq.py
- FOUND: commit ca95cc4

---
*Phase: 04-q-type-freq-refinement*
*Completed: 2026-04-06*
