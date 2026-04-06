---
phase: 01-metrics-data-foundation
plan: 03
subsystem: metrics
tags: [validation, baseline, hungarian-matching, eq-parameters]

requires: []
provides:
  - Pre-fix baseline validation metrics (gain, freq, Q, type accuracy)
  - Loss component breakdown from current buggy code
  - Reference point for Plan 04 delta computation (D-09)
affects: [01-04-PLAN, validation, metrics]

tech-stack:
  added: []
  patterns: [baseline-validation-before-fixes]

key-files:
  created: [insight/pre_fix_baseline.md]
  modified: []

key-decisions:
  - "Disabled torch.compile to load checkpoint state_dict without OptimizedModule key prefix mismatch"

patterns-established:
  - "Pre-fix baseline: run validation before any code changes to capture the before-state"

requirements-completed: [METR-01]

duration: 6min
completed: 2026-04-05
---

# Phase 1 Plan 03: Pre-Fix Baseline Summary

**Pre-fix baseline validation metrics captured with current buggy code: Gain MAE 5.60 dB, Freq MAE 2.378 oct, Type accuracy 48.1%**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-05T23:48:32Z
- **Completed:** 2026-04-05T23:54:49Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Ran full validation epoch with existing best.pt checkpoint on 2000-sample synthetic validation set
- Captured pre-fix baseline metrics: Gain MAE 5.60 dB (matched), Freq MAE 2.378 octaves, Q MAE 0.472 decades, Type accuracy 48.1%
- Recorded all loss component values for reference (param_loss 50.18, gain_loss 18.39, hmag_loss 2.73, etc.)
- Documented raw vs matched gain MAE difference (13.02 dB raw vs 5.60 dB matched) confirming permutation penalty inflation

## Task Commits

Each task was committed atomically:

1. **Task 1: Run pre-fix baseline validation with current code** - `488fcd0` (docs)

## Files Created/Modified
- `insight/pre_fix_baseline.md` - Pre-fix baseline validation results with matched and raw MAE values, loss component breakdown

## Decisions Made
- Disabled torch.compile in config to avoid OptimizedModule state_dict key mismatch when loading checkpoint -- the compiled model wraps keys with `_orig_mod.` prefix that doesn't match the saved checkpoint

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Initial attempt to load checkpoint failed because Trainer constructor applies torch.compile, causing state_dict key mismatch with `_orig_mod.` prefix. Resolved by manually constructing Trainer components with torch.compile disabled.

## Next Phase Readiness
- Pre-fix baseline is captured and documented (D-01 complete)
- Plan 04 can now compute the delta between pre-fix and post-fix metrics (D-09)
- Baseline confirms the known issues: 5.60 dB gain MAE (target < 1 dB), 2.378 oct freq MAE (target < 0.25), 48.1% type accuracy (target > 95%)

## Self-Check: PASSED

- FOUND: insight/pre_fix_baseline.md
- FOUND: 01-03-SUMMARY.md
- FOUND: commit 488fcd0
- pre_fix_baseline.md contains Pre-Fix Baseline header, actual values (5.60 dB), D-01 reference

---
*Phase: 01-metrics-data-foundation*
*Completed: 2026-04-05*
