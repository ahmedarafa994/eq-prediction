# Phase 1 Baseline Metrics

**Date:** 2026-04-05
**Checkpoint:** checkpoints/best.pt (same checkpoint used for pre-fix measurement)
**Config:** conf/config.yaml

## Pre-Fix Baseline (D-01)

Measured BEFORE any Phase 1 code changes (captured by Plan 03).

| Metric | Value |
|--------|-------|
| Total val loss | 53.6259 |
| Gain MAE (matched) | 5.60 dB |
| Gain MAE (raw, debug) | 13.02 dB |
| Freq MAE | 2.378 octaves |
| Q MAE | 0.472 decades |
| Type accuracy | 48.1% |

### Pre-Fix Loss Components

| Component | Value |
|-----------|-------|
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

## Post-Fix Baseline (D-09)

Post-fix validation requires running a validation epoch with the fixed code and newly generated uniform data distribution. The existing checkpoint was trained on the old beta(2,2) distribution — running validation on uniform data produces numbers that reflect the data shift, not model improvement.

**Post-fix measurement deferred to next training run.** The metrics code is now correct (D-02 component logging, D-03 gradient monitoring), so any future `python train.py` will automatically log trustworthy component-level metrics.

## What Changed (Plans 01-01, 01-02, 01-03)

| Fix | Decision ID | Impact |
|-----|-------------|--------|
| Validation component logging | D-02 | All 10 loss components now logged per validation epoch |
| Gradient norm monitoring | D-03 | Correct parameter name matching (gain_mlp, classification_head, etc.) |
| Uniform gain distribution | D-05 | random.uniform(gain_range[0], gain_range[1]) replaces beta(2,2) |
| HP/LP full gain range | D-06 | HP/LP use _sample_gain() instead of random.uniform(-1, 1) |
| Cache deletion | D-07 | All precomputed .pt caches deleted for regeneration |

## Validation for Phase 2

Phase 2 (Gain Prediction Fix) will compare its results against this baseline.
Target: Gain MAE < 3 dB (down from 5.60 dB matched pre-fix).

**Note:** The pre-fix baseline is trustworthy because Plan 03 captured it before any code changes. Matched MAE is the primary metric (D-04). Raw MAE (13.02 dB) is inflated by permutation penalty and kept for debug comparison only.
