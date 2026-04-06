# Pre-Fix Baseline Validation

**Date:** 2026-04-05
**Decision:** D-01
**Checkpoint:** checkpoints/best.pt
**Code state:** Unmodified (buggy) — before any Phase 1 fixes

## Pre-Fix Baseline Values

| Metric | Value |
|--------|-------|
| Total val loss | 53.6259 |
| Gain MAE (matched) | 5.60 dB |
| Gain MAE (raw) | 13.02 dB |
| Freq MAE | 2.378 octaves |
| Q MAE | 0.472 decades |
| Type accuracy | 48.1% |

## Notes

- This baseline was captured BEFORE any Phase 1 code changes
- Hungarian-matched targets used for gain MAE computation
- Used as the "before" reference for all subsequent fix deltas
