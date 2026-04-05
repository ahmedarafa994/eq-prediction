# Fix: Gain MAE stuck at ~6 dB, val loss regression across curriculum stages

## Context

Training completed 165 epochs but **gain MAE is stuck at ~6 dB** and **type accuracy collapsed to 0.5%** (from 65% at epoch 51). The model learns frequency well (1.39 oct in stage 1) but cannot learn gain magnitude or filter type discrimination. Root causes:

1. **40% of training gains are near 0 dB** (`_beta_gain` uses beta(2,5) for subtle EQ) — model optimizes for predicting ~0 dB
2. **Huber delta=3.0 for gain** is too large — most errors >3 dB get constant linear gradients, providing weak learning signal
3. **`min_gain_db=0.0`** allows 0 dB gains in training, creating trivially "inactive" bands that the model learns to predict as 0 dB
4. **Stage transitions lack warmup** — when HP/LP filters introduced, the optimizer hasn't adapted and all stage 1-2 learning is destabilized
5. **`lambda_param` stays at 1.0 while `recon_loss` grows to ~3.0** — the reconstruction loss dominates total loss, drowning out parameter regression gradients

## Changes

### 1. Fix gain sampling: set `min_gain_db=3.0` globally (`insight/train.py`)

Every `_beta_gain()` call must use `min_gain_db >= 3.0`. This ensures every band has a meaningful, audible gain that the model must learn to predict. No more trivially-zero bands.

**File**: `insight/train.py` lines ~315, ~329, ~382, ~407
- Stage config default: `min_gain_db=3.0` (was `0.0`)
- Replay dataset: `min_gain_db=3.0` (was `0.0`)
- Validation dataset: `min_gain_db=3.0` (was `0.0`)

Also update `insight/conf/config.yaml`:
- Add `min_gain_db: 3.0` to the global `data` section

### 2. Tighten Huber delta for gain from 3.0 → 1.5 (`insight/loss_multitype.py`)

**File**: `insight/loss_multitype.py` line ~308
```python
# BEFORE:
self.huber_gain = nn.HuberLoss(delta=3.0)
# AFTER:
self.huber_gain = nn.HuberLoss(delta=1.5)
```

With gains now in [3, 24] dB range, a delta of 1.5 dB keeps more errors in the quadratic (strong gradient) regime. Large errors still get linear treatment but the transition point is lower, giving more gradient signal for moderate errors.

### 3. Add LR warmup at each curriculum stage transition (`insight/train.py`)

When a new curriculum stage starts, reset the LR to warmup from a lower value over ~3 epochs before resuming cosine schedule. This prevents the destabilization seen at stage 2→3 and 3→4 transitions.

**File**: `insight/train.py` — in `_setup_stage()` method or at the start of each stage loop
- At stage start: save current LR, reset optimizer momentum, set LR to `base_lr * 0.1` and warm up over 3 epochs
- The existing cosine scheduler continues from the warmup end point

### 4. Increase `lambda_param` weight relative to recon (`insight/conf/config.yaml`)

The reconstruction loss (~2.5-3.0) now dominates the total loss while `lambda_param=1.0` and `param_loss` contributes ~10. But the gradient from recon doesn't effectively flow back to individual gain parameters — it's a spectral-level signal. The param loss directly optimizes gain, freq, Q.

**File**: `insight/conf/config.yaml`
- Stage 1: `lambda_param: 1.0` → `2.0`
- Stage 2: `lambda_param: 1.0` → `2.0`  
- Stage 3: `lambda_param: 1.0` → `2.0`
- Stage 4: `lambda_param: 1.0` → `1.5`
- Stage 5: `lambda_param: 0.5` → `1.0`
- Reduce `lambda_recon` from 0.1/0.5/1.0 → 0.05/0.1/0.2 across stages

## Files to modify

1. `insight/train.py` — min_gain_db default, LR warmup at stage transitions
2. `insight/loss_multitype.py` — Huber delta for gain: 3.0 → 1.5
3. `insight/conf/config.yaml` — min_gain_db, lambda_param, lambda_recon

## Verification

1. Run all existing tests: `python test_eq.py && python test_model.py && python test_multitype_eq.py`
2. Start training: `bash /teamspace/studios/this_studio/resume_training.sh`
3. After 5 epochs, check:
   - Gain MAE should be < 5.5 dB (was 5.7 at epoch 1 before)
   - Type accuracy should start improving (>25% by epoch 5)
   - Val loss should be monotonically decreasing in stage 1
4. After epoch 32 (stage 1 end), gain MAE should be < 4.5 dB
