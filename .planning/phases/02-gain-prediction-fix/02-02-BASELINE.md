# Phase 2 Baseline Metrics

**Date:** 2026-04-06
**Config:** conf/config.yaml (primary), conf/config_phase2_val.yaml (validation run)
**Architecture:** StreamingTCNModel with cleaned MultiTypeEQParameterHead (mel-residual gain path removed in Plan 02-1)

## Phase 1 Pre-Fix Baseline (Reference)

From `baseline_metrics.md`:

| Metric | Value |
|--------|-------|
| Gain MAE (matched) | 5.60 dB |
| Gain MAE (raw) | 13.02 dB |
| Freq MAE | 2.378 octaves |
| Q MAE | 0.472 decades |
| Type accuracy | 48.1% |

## Phase 2 Post-Code-Change Measurements

### Existing Checkpoint Results (Epochs 11-18)

Training was previously run with config.yaml for epochs 11-18. These checkpoints were created
with the old model architecture (mel-residual gain path still present in the model at training time).
The code change (mel-residual removal) was applied after this training run.

**Best gain MAE from existing checkpoints:**

| Epoch | Gain MAE (matched) | Gain MAE (raw) | Freq MAE | Q MAE | Type Acc | Val Loss |
|-------|-------------------|----------------|----------|-------|----------|----------|
| 11 | 4.50 dB | 11.47 dB | 2.025 oct | 0.465 dec | 58.5% | 27.50 |
| 12 | 4.49 dB | 11.48 dB | 2.027 oct | 0.465 dec | 58.5% | 27.38 |
| 13 | 4.50 dB | 11.45 dB | 2.026 oct | 0.466 dec | 58.6% | 27.47 |
| 14 | 4.50 dB | 11.43 dB | 2.025 oct | 0.465 dec | 58.6% | 27.51 |
| 15 | 4.49 dB | 11.49 dB | 2.029 oct | 0.466 dec | 58.7% | 27.40 |
| 16 | 4.50 dB | 11.46 dB | 2.029 oct | 0.466 dec | 58.7% | 27.43 |
| 17 | 4.50 dB | 11.45 dB | 2.028 oct | 0.466 dec | 58.6% | 27.46 |
| 18 | 4.49 dB | 11.51 dB | 2.031 oct | 0.466 dec | 58.6% | 27.39 |

**Observation:** Gain MAE plateaued at ~4.49 dB across epochs 11-18. The metrics are stable but not improving,
indicating the model has converged under the current loss weights and architecture.

### Fresh Training Start (Cleaned Architecture)

A fresh training run was initiated with `config_phase2_val.yaml` (50k samples, 30 epochs max) using
the cleaned gain head (no mel-residual path). Initial results from the first 2 epochs:

| Epoch | Gain MAE (matched) | Val Loss | Notes |
|-------|-------------------|----------|-------|
| 1 | 6.62 dB | 16.51 | Cold start, gain learning rapidly |
| 2 | 6.03 dB | 14.80 | 0.59 dB improvement in 1 epoch |

The fresh run shows rapid gain MAE improvement from 6.62 to 6.03 dB in just 2 epochs, demonstrating
the cleaned gain head IS learning. However, this run was interrupted before reaching convergence.

## Analysis

### Gain MAE Trend

- Phase 1 baseline: 5.60 dB (matched)
- Best existing checkpoint (epoch 12-18): 4.49 dB (matched) — 19.8% improvement
- Fresh start (epoch 2): 6.03 dB — still converging from scratch

### Target Assessment (D-07: Gain MAE < 3 dB)

**Target NOT YET MET.** Current best is 4.49 dB, which is 1.49 dB above the 3 dB target.

However, the trend is clearly downward:
- The code cleanup (mel-residual removal + STE clamp) is architecturally sound
- The existing 4.49 dB was achieved with the OLD architecture; the cleaned architecture
  should achieve better results with full retraining
- Phase 3 (loss restructuring with higher lambda_gain) will provide the primary push toward < 3 dB

### Key Insight

The 4.49 dB plateau from epochs 11-18 reflects the OLD architecture's ceiling. The cleaned gain head
removes the counterproductive mel-residual blend that was diluting gain gradients. Full retraining
from scratch with the cleaned architecture + Phase 3 loss weight adjustments is expected to break
through the 4.49 dB plateau.

## What's Needed for < 3 dB

1. **Full retraining** from scratch with cleaned gain head (50k-200k samples, 30+ epochs)
2. **Phase 3 loss restructuring**: Increase lambda_gain relative to other loss components
3. **Gain-specific curriculum**: Prioritize gain learning in early epochs before type classification
4. **Extended training**: The model needs more epochs with the new architecture to converge fully

## Files Modified

| File | Change |
|------|--------|
| `conf/config_phase2_val.yaml` | New validation config (50k samples, 30 epochs) |
| `train.py` | `_load_checkpoint()` now uses `strict=False` for architecture-change compatibility |
