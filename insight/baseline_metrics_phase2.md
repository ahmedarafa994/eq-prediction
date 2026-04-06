# Phase 2 Baseline Metrics — Gain Prediction Fix

**Date:** 2026-04-06
**Config:** conf/config.yaml (primary), conf/config_phase2_val.yaml (validation run)
**Architecture:** StreamingTCNModel with cleaned MultiTypeEQParameterHead (mel-residual gain path removed in Plan 02-1)

---

## Phase 1 Pre-Fix Baseline (Reference)

| Metric | Value |
|--------|-------|
| Gain MAE (matched) | 5.60 dB |
| Gain MAE (raw) | 13.02 dB |
| Freq MAE | 2.378 octaves |
| Q MAE | 0.472 decades |
| Type accuracy | 48.1% |

## Phase 2 Post-Code-Change Measurements

### Existing Checkpoint Results (Epochs 11-18, Old Architecture)

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

### Fresh Training Start (Cleaned Architecture)

| Epoch | Gain MAE (matched) | Val Loss | Notes |
|-------|-------------------|----------|-------|
| 1 | 6.62 dB | 16.51 | Cold start |
| 2 | 6.03 dB | 14.80 | Interrupted |

## Analysis

- **Pre-fix baseline:** 5.60 dB matched
- **Best post-change:** 4.49 dB matched (19.8% improvement)
- **Target (GAIN-04):** < 3.0 dB — **NOT MET** (gap: 1.49 dB)
- **Fresh start:** Shows learning trajectory (6.62 → 6.03 in 2 epochs) but interrupted

## Target Assessment

The gain MAE < 3 dB target (GAIN-04) is **NOT MET**. The 4.49 dB plateau reflects the old architecture's ceiling under combined loss weights. The cleaned gain head (no mel-residual path, STE clamp instead of Tanh) is architecturally sound but requires:

1. Full retraining from scratch with 30+ epochs (not interrupted)
2. Phase 3 loss restructuring: independent per-parameter weights, gain-only warmup, log-cosh loss
3. Higher lambda_gain relative to spectral/type losses

## What's Needed

- Full retraining with cleaned architecture + Phase 3 loss changes
- See: `conf/config_phase2_val.yaml` (50k samples, 30 epochs max)
- Train.py fixed for `strict=False` checkpoint loading (architecture migration compat)
