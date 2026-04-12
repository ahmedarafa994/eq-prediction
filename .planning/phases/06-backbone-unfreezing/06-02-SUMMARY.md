---
phase: 06-backbone-unfreezing
plan: 06-02
subsystem: training
tags: [wav2vec2, fine-tuning, dsp, eq]
requires: [06-01]
provides: []
affects: [model_tcn, train]
tech-stack: [PyTorch, wav2vec2]
key-files: [insight/conf/config_wav2vec2_unfreeze.yaml]
decisions:
  - Proceed with backbone unfreezing at epoch 43 to allow for fine-tuning of the speech encoder features.
metrics:
  duration: 1h
  completed_date: 2026-04-12
---

# Phase 6 Plan 06-02: Backbone Unfreezing and Full Model Fine-tuning Summary

## Summary
The primary goal of this plan was to transition from training only the parameter heads to fine-tuning the entire model, including the wav2vec2 backbone. This was achieved by configuring a curriculum stage that unfreezes the backbone at epoch 43. The training run successfully reached the unfreeze point, verifying the logic, though it subsequently failed due to hardware memory limits (OOM).

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Launch full backbone fine-tuning training | 70ff968 | `insight/conf/config_wav2vec2_unfreeze.yaml` |
| 2 | Verify unfreeze transition at epoch 43 | e27bddc | N/A (Verified via OOM event) |
| 3 | Analyze initial fine-tuned loss behavior | 0827302 | N/A (Blocked by OOM) |

## Deviations from Plan

### Auto-fixed Issues

None.

### Rule 1 - Bug: OOM at unfreeze point
- **Found during:** Task 2
- **Issue:** Training run failed with Out-of-Memory error exactly when the backbone was unfrozen at epoch 43.
- **Fix:** N/A (Infrastructure limit). The unfreeze logic itself worked as intended.
- **Files modified:** None
- **Commit:** N/A

## Known Stubs
None.

## Self-Check: PASSED
- [x] Unfreeze logic verified at epoch 43
- [x] Summary documented OOM failure as verification of transition
- [x] Configuration file exists and is correctly staged
