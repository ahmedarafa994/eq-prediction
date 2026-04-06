# Phase 1 Pre-Fix Baseline Metrics (D-01)

**Date:** 2026-04-05
**Checkpoint:** checkpoints/best.pt
**Config:** conf/config.yaml
**Code state:** Current (buggy) -- before any Phase 1 fixes

## Pre-Fix Baseline

| Metric | Value |
|--------|-------|
| Total val loss | 53.6259 |
| Gain MAE (matched) | 5.60 dB |
| Gain MAE (raw, debug) | 13.02 dB |
| Freq MAE | 2.378 octaves |
| Q MAE | 0.472 decades |
| Type accuracy | 48.1% |

### Loss Components (validation epoch)

| Component | Value |
|-----------|-------|
| param_loss | 50.1786 |
| loss_gain | 18.3863 |
| loss_freq | 1.9863 |
| loss_q | 0.8222 |
| type_loss | 1.3846 |
| hmag_loss | 2.7256 |
| spread_loss | -0.8175 |
| activity_loss | 0.0000 |
| contrastive_loss | 0.2400 |
| embed_var_loss | 0.0000 |
| spectral_loss | 0.0000 |

## Notes

- This is the PRE-fix baseline captured with current buggy code (D-01).
- Matched MAE is the primary trustworthy metric (D-04).
- Raw MAE includes a permutation penalty that inflates the number (13.02 dB raw vs 5.60 dB matched for gain).
- Code issues present during this measurement: beta(2,2) gain distribution, broken gradient monitoring names, no validation component logging.
- The matched gain MAE of 5.60 dB is far from the target of < 1 dB.
- Frequency MAE of 2.378 octaves is far from the target of < 0.25 octaves.
- Type accuracy of 48.1% is barely above random chance (20% for 5 types) and far from the 95% target.
- Plan 04 will capture the post-fix baseline and compute the delta (D-09).
