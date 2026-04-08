---
phase: quick
plan: 260408-3fg
subsystem: loss
tags: [loss, focal-loss, class-weights, sign-penalty, gain]
tech-stack:
  added: []
  patterns: [per-class-multipliers-after-normalization, sign-mismatch-penalty]
key-files:
  modified:
    - insight/loss_multitype.py
    - insight/conf/config.yaml
    - insight/train.py
decisions:
  - "5x lowshelf multiplier applied after inverse-frequency normalization + re-normalization to preserve mean=1"
  - "sign_penalty_weight=0.5 adds half the log-cosh contribution for each sign-flipped band"
  - "Sign penalty incorporated into loss_gain before lambda_gain scaling so lambda_gain still controls overall gain loss scale"
metrics:
  duration: "109s"
  completed: "2026-04-08"
  tasks: 2
  files_modified: 3
---

# Phase quick Plan 260408-3fg: Apply Lowshelf Class Weight Increase and Gain Sign Penalty Summary

Two surgical loss function fixes targeting the two highest-leverage accuracy problems identified at epoch 20: lowshelf type collapse (5.5% accuracy) and gain sign flip errors (contributing to 5.78 dB gain MAE).

## What Was Done

### Task 1: Loss function changes (insight/loss_multitype.py)

**A — Per-class weight multipliers (class_weight_multipliers parameter):**

Added `class_weight_multipliers: list[float] | None = None` constructor parameter to `MultiTypeEQLoss`. When provided, multipliers are applied *after* the existing inverse-frequency normalization, then the combined weights are re-normalized to mean=1. This keeps the overall focal loss scale stable while giving targeted emphasis to underperforming classes.

With `class_weight_multipliers=[1.0, 5.0, 1.0, 1.0, 1.0]`, the resulting type_class_weights are:

```
peaking=0.161, lowshelf=2.688, highshelf=0.538, highpass=0.807, lowpass=0.807
```

Lowshelf (index 1) is now the highest-weighted class by a factor of ~17x over peaking, ensuring the focal loss genuinely up-weights lowshelf samples.

**B — Gain sign penalty (sign_penalty_weight parameter):**

Added `sign_penalty_weight: float = 0.0` constructor parameter. In `forward()`, immediately after computing `loss_gain = log_cosh_loss(...)`, a sign mismatch penalty is computed and added to `loss_gain`:

```python
sign_mismatch = (pred_gain * matched_gain) < 0.0
sign_penalty = (sign_mismatch.float() * (pred_gain - matched_gain).abs()).mean()
loss_gain = loss_gain + self.sign_penalty_weight * sign_penalty
```

The penalty is proportional to absolute error so large flips (e.g., +6 dB predicted vs -15 dB GT) cost significantly more than small near-zero sign ambiguities. `components["sign_penalty"]` is logged every forward pass.

Both parameters default to `None`/`0.0` preserving full backward compatibility.

### Task 2: Config and train.py wiring (insight/conf/config.yaml, insight/train.py)

**Why 5x for lowshelf:** At epoch 20, lowshelf classification accuracy was 5.5% — essentially random. The base inverse-frequency weight already gives lowshelf ~3.3x over peaking (1/0.15 vs 1/0.5), but that is insufficient. An additional 5x multiplier (giving ~17x total over peaking after re-normalization) is needed to make the loss gradient genuinely dominate when the model misclassifies lowshelf samples.

**Why 0.5 for sign_penalty_weight:** At 0.5, each sign-flipped band contributes an additional 50% of its absolute error to the gain loss. For a +6 dB / -15 dB mismatch, this adds 10.5 dB extra penalty on top of the log-cosh loss, creating strong signal without overwhelming the other loss components.

## Smoke-Check Results

```
lowshelf weight: 2.688
sign_penalty_weight: 0.5
Integration smoke-check passed
```

```
sign_penalty: 0.25   (for +6 dB predicted vs -15 dB GT on 1 of 10 bands)
All checks passed
```

Import check: `Import OK`

## Deviations from Plan

None — plan executed exactly as written.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 58b6235 | feat(260408-3fg): add class_weight_multipliers and sign_penalty_weight to MultiTypeEQLoss |
| 2 | a647de2 | feat(260408-3fg): wire class_weight_multipliers and sign_penalty_weight through config and train.py |

## Self-Check: PASSED

- insight/loss_multitype.py modified: FOUND
- insight/conf/config.yaml modified: FOUND
- insight/train.py modified: FOUND
- Commit 58b6235: FOUND
- Commit a647de2: FOUND
