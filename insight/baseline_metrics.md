# Baseline Metrics — Consolidated Reference

**Date:** 2026-04-06
**Purpose:** Single source of truth for all baseline measurements across phases

---

## Phase 1 Pre-Fix Baseline (Before Any Code Changes)

| Metric | Value | Source |
|--------|-------|--------|
| Total val loss | 53.6259 | pre_fix_baseline.md |
| Gain MAE (matched) | 5.60 dB | pre_fix_baseline.md |
| Gain MAE (raw) | 13.02 dB | pre_fix_baseline.md |
| Freq MAE | 2.378 octaves | pre_fix_baseline.md |
| Q MAE | 0.472 decades | pre_fix_baseline.md |
| Type accuracy | 48.1% | pre_fix_baseline.md |

## Phase 2 Post-Code-Change Measurements

### Existing Checkpoint Results (Old Architecture, Epochs 11-18)

| Epoch | Gain MAE (matched) | Gain MAE (raw) | Freq MAE | Q MAE | Type Acc | Val Loss |
|-------|-------------------|----------------|----------|-------|----------|----------|
| 11 | 4.50 dB | 11.47 dB | 2.025 oct | 0.465 dec | 58.5% | 27.50 |
| 12 | 4.49 dB | 11.48 dB | 2.027 oct | 0.466 dec | 58.5% | 27.38 |
| 13 | 4.50 dB | 11.45 dB | 2.026 oct | 0.466 dec | 58.6% | 27.47 |
| 14 | 4.50 dB | 11.43 dB | 2.025 oct | 0.465 dec | 58.6% | 27.51 |
| 15 | 4.49 dB | 11.49 dB | 2.029 oct | 0.466 dec | 58.7% | 27.40 |
| 16 | 4.50 dB | 11.46 dB | 2.029 oct | 0.466 dec | 58.6% | 27.43 |
| 17 | 4.50 dB | 11.45 dB | 2.028 oct | 0.466 dec | 58.6% | 27.46 |
| 18 | 4.49 dB | 11.51 dB | 2.031 oct | 0.466 dec | 58.7% | 27.39 |

**Best: 4.49 dB matched** (19.8% improvement from 5.60 dB pre-fix)

### Fresh Training Start (Cleaned Architecture)

| Epoch | Gain MAE (matched) | Val Loss | Notes |
|-------|-------------------|----------|-------|
| 1 | 6.62 dB | 16.51 | Cold start, gain learning rapidly |
| 2 | 6.03 dB | 14.80 | 0.59 dB improvement in 1 epoch |

*Fresh run interrupted before convergence*

## Delta Table (Pre-Fix vs Best Post-Change)

| Metric | Pre-Fix | Post-Change | Delta | Improvement |
|--------|---------|-------------|-------|-------------|
| Gain MAE (matched) | 5.60 dB | 4.49 dB | -1.11 dB | 19.8% |
| Gain MAE (raw) | 13.02 dB | 11.43 dB | -1.59 dB | 12.2% |
| Freq MAE | 2.378 oct | 2.025 oct | -0.353 oct | 14.8% |
| Q MAE | 0.472 dec | 0.465 dec | -0.007 dec | 1.5% |
| Type accuracy | 48.1% | 58.7% | +10.6 pp | 22.0% |

## Target: < 3 dB Gain MAE

- **Current best:** 4.49 dB (old architecture ceiling)
- **Target:** < 3.0 dB
- **Gap:** 1.49 dB remaining
- **Path to target:** Full retraining with Phase 3 loss restructuring (independent weights, warmup, log-cosh)

## Notes

- Post-fix baseline with new uniform data was **deferred** — old checkpoint trained on beta(2,2) distribution
- Next full training run with uniform data + Phase 3 loss architecture will produce definitive baseline
- All measurement instrumentation (Hungarian matching, per-param MAE, gradient norms) is now correct per Phase 1
