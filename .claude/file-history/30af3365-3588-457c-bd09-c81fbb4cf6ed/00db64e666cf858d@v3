# Plan: Fix NaN Loss at Curriculum Stage Transition (Epoch 21)

## Context

At epoch 21, the curriculum transitions from `easy_peaking` → `medium_peaking`. This triggers:
- Q range widens from [0.3, 5.0] → [0.2, 8.0]
- Gumbel temperature drops from 1.0 → 0.5

The model produces `train_loss: 0.0` and `val_loss: NaN` at this transition, meaning training effectively dies.

## Root Cause Analysis

### Primary Bug: NaN × 0.0 = NaN

**File:** `insight/loss_multitype.py` lines 580–583

```python
if not torch.isfinite(total_loss):
    total_loss = total_loss * 0.0  # BUG: NaN * 0.0 = NaN in IEEE 754
```

This "guard" doesn't work. When `total_loss` is NaN, multiplying by 0.0 still produces NaN. The comment says "Maintains graph, returns 0 with correct dtype" but this is mathematically wrong.

**Consequence:** The loss function returns NaN instead of a recoverable zero. In training, the outer NaN guard (train.py line 577) skips the batch → all batches skipped → `epoch_loss=0.0, n_batches=0` → `train_loss=0.0`. In validation, the `math.isnan()` guard skips NaN batches → `val_loss=0.0, n_batches=0` → `val_loss=0.0` OR the NaN propagates differently depending on accumulation order.

### Secondary Cause: Stage Transition Numerical Instability

The Gumbel temperature drop from 1.0 → 0.5 makes type probability distributions sharper. Combined with wider Q range, the biquad coefficient computation in `differentiable_eq.py` can produce extreme values:
- `compute_biquad_coeffs_multitype_soft` computes weighted mixtures of coefficients
- At Q=8.0, the peaking filter coefficients can be very large
- The weighted sum with sharp type_probs can amplify these extremes
- Division by `a0_safe` then creates overflow → NaN

The `nan_to_num` guard at line 463 of `differentiable_eq.py` patches the output but doesn't prevent gradient corruption.

## Fixes

### Fix 1: Correct the NaN-to-zero replacement in loss_multitype.py

**File:** `insight/loss_multitype.py` line 580–583

Replace:
```python
if not torch.isfinite(total_loss):
    total_loss = total_loss * 0.0
```

With:
```python
if not torch.isfinite(total_loss):
    total_loss = torch.zeros_like(total_loss)
```

This correctly produces 0.0 instead of NaN.

### Fix 2: Add safety guard in validate() for all-NaN case

**File:** `insight/train.py` line 834

Add a guard after computing `avg_val_loss`:
```python
avg_val_loss = val_loss / max(n_batches, 1)
if math.isnan(avg_val_loss) or math.isinf(avg_val_loss):
    avg_val_loss = float('inf')  # Treat as worst-case for checkpoint logic
```

### Fix 3: Add gradient-level NaN protection in differentiable_eq.py

**File:** `insight/differentiable_eq.py` after line 468 (the log-domain product)

Add NaN guards on the gradient path:
```python
H_mag_total = torch.nan_to_num(H_mag_total, nan=1.0, posinf=1e4, neginf=1e-4)
```

This already exists at line 463 for the blended bands, but NOT for the final total product.

### Fix 4: Clamp Gumbel temperature to prevent overly sharp distributions

**File:** `insight/train.py` line 442–446 (inside `_setup_stage`)

Add a minimum temperature floor:
```python
gumbel_temp = stage_cfg.get("gumbel_temperature", None)
if gumbel_temp is not None and hasattr(self.model, "dsp_cascade"):
    if hasattr(self.model.dsp_cascade, "gumbel_temperature"):
        safe_temp = max(gumbel_temp, 0.3)  # Floor at 0.3 to prevent extreme sharpness
        self.model.dsp_cascade.gumbel_temperature.fill_(safe_temp)
```

### Fix 5: Add warmup for stage transitions (smooth dataset blending)

**File:** `insight/train.py` inside `train_one_epoch`

Add a NaN recovery mechanism at the batch level — if the loss is NaN, reduce learning rate temporarily:
```python
if not torch.isfinite(total_loss):
    nan_batches += 1
    # Reduce LR to help recovery
    for pg in self.optimizer.param_groups:
        pg['lr'] *= 0.5
    ...
```

## Files to Modify

1. `insight/loss_multitype.py` — Fix NaN×0.0 bug (line 580–583)
2. `insight/train.py` — Validate NaN guard (line 834), Gumbel temp floor (line 442–446), LR recovery
3. `insight/differentiable_eq.py` — Final H_mag_total NaN guard (after line 468)

## Verification

1. Run `python train.py` and monitor epoch 21 transition
2. Check that `train_loss` and `val_loss` remain finite at the stage boundary
3. Verify that the NaN batch count in training logs drops to zero after fixes
4. Run existing tests: `python test_eq.py`, `python test_multitype_eq.py`
