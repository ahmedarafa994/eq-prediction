# Implementation Plan: Fix EQ Estimation Model Performance Plateau

## Executive Summary

The model currently plateaus at gain MAE ~3 dB, freq MAE ~2 octaves, type accuracy ~52.7%. This plan addresses the 5 root causes identified with specific, testable code changes.

## Current Performance Baseline

From training_v7.log epoch 47-50:
- Gain MAE: 2.99-3.00 dB (stalled)
- Frequency MAE: 1.99-2.00 octaves (stalled)
- Type accuracy: 52.5-52.8% (near random for 5 classes = 20%)
- Loss plateau: val_loss stuck at ~8.76

## Root Causes & Solutions

### 1. Dataset Gain Distribution Bias (HIGHEST PRIORITY)

**Problem**: `dataset.py` lines 201-205 use Beta(2,5) which concentrates 80% of samples near 0 dB. Combined with HP/LP filters always having gain=0, this creates severe bias.

**Solution**: Replace Beta distribution with uniform sampling across valid gain ranges.

**File**: `/teamspace/studios/this_studio/insight/dataset.py`

**Changes**:
- Line 201-205: Replace `_beta_gain()` with `_uniform_gain()`
- For HP/LP filters (lines 170, 173): Allow small non-zero gains (±3 dB) instead of forcing 0.0

**Code change**:
```python
# OLD (lines 201-205)
def _beta_gain(self):
    sign = random.choice([-1, 1])
    magnitude = np.random.beta(2, 5) * min(abs(self.gain_range[0]), abs(self.gain_range[1]))
    return sign * magnitude

# NEW
def _uniform_gain(self):
    """Sample gain uniformly from valid range."""
    return random.uniform(*self.gain_range)

# Also update HP/LP gains (lines 170, 173):
# OLD: g = 0.0
# NEW: g = random.uniform(-3.0, 3.0)  # Small gain range for filters
```

**Impact**: Removes 80% bias toward near-zero gains. Provides balanced gradient signal across full gain range.

**Risk**: Low. Uniform sampling is simpler and more stable than Beta.

**Test**: Run `python test_multitype_eq.py` to ensure no test breakage.

---

### 2. Loss Weight Imbalance (HIGH PRIORITY)

**Problem**: Gain loss contributes ~56% of total loss (lambda_gain=2.0), dominating optimization.

**Solution**: Rebalance loss weights to equalize contribution across parameters.

**File**: `/teamspace/studios/this_studio/insight/loss_multitype.py`

**Changes**:
- Line 30: Reduce `lambda_gain` from 2.0 to 1.0
- Line 183: Update default `lambda_gain` from 2.0 to 1.0
- Keep `lambda_param=1.5` for freq+Q
- Increase `lambda_type` from 0.5 to 1.0 to improve type classification

**Code change**:
```python
# In HungarianBandMatcher.__init__ (line 30):
def __init__(self, lambda_freq=1.0, lambda_q=0.5, lambda_gain=1.0):  # was 2.0
    
# In MultiTypeEQLoss.__init__ (line 183):
def __init__(self, n_fft=2048, sample_rate=44100,
             lambda_param=1.0, lambda_gain=1.0, lambda_type=1.0,  # was 0.5
```

**Impact**: Balances optimization across all parameters. Prevents gain from dominating.

**Risk**: Low. Weights are hyperparameters; rebalancing is standard practice.

**Test**: Verify training loss components are more balanced (gain loss ~33% instead of 56%).

---

### 3. Attention Temperature Too Low (MEDIUM PRIORITY)

**Problem**: Temperature 0.1 makes frequency attention nearly one-hot, losing multi-band context.

**Solution**: Raise initial temperature and make it learnable.

**File**: `/teamspace/studios/this_studio/insight/differentiable_eq.py`

**Changes**:
- Line 506: Change initial temperature from 0.1 to 0.5
- Line 506: Make it a `nn.Parameter` instead of `register_buffer` for learnability

**Code change**:
```python
# OLD (line 506):
self.register_buffer("attn_temperature", torch.tensor(0.1))

# NEW:
self.attn_temperature = nn.Parameter(torch.tensor(0.5))
```

**Impact**: Softer attention allows each band to see broader frequency context. Learnable temperature lets the model adapt.

**Risk**: Low. Temperature is a standard attention mechanism parameter.

**Test**: Verify attention weights are not degenerate (check attention entropy).

---

### 4. Mel Profile Normalization Issue (MEDIUM PRIORITY)

**Problem**: Lines 551-553 normalize mel_profile (mean subtraction, std division), removing absolute level information needed to distinguish positive vs negative gains.

**Solution**: Skip mean subtraction or preserve absolute level as a separate feature.

**File**: `/teamspace/studios/this_studio/insight/differentiable_eq.py`

**Changes**:
- Lines 551-553: Remove mean subtraction, only normalize by std
- Or: Add a parallel path that preserves absolute level

**Code change**:
```python
# Option A: Remove mean subtraction (simpler)
# OLD (lines 551-553):
mp_mean = mel_profile.mean(dim=-1, keepdim=True)
mp_std = mel_profile.std(dim=-1, keepdim=True).clamp(min=1e-4)
mel_profile_normed = (mel_profile - mp_mean) / mp_std

# NEW:
mp_std = mel_profile.std(dim=-1, keepdim=True).clamp(min=1e-4)
mel_profile_normed = mel_profile / mp_std  # Only scale, don't center

# Option B: Preserve absolute level (more complex, add feature)
# Add to line 557 after building cnn_in:
cnn_in = torch.cat([mel_profile_normed.unsqueeze(1), 
                    pos, 
                    mel_profile.mean(dim=-1, keepdim=True).unsqueeze(1)], dim=1)
# Then update cnn_channels from 2 to 3 in line 488, 493
```

**Recommendation**: Start with Option A (simpler). If gain prediction doesn't improve, try Option B.

**Impact**: Preserves absolute level information for gain sign prediction.

**Risk**: Medium. Changing normalization affects the entire feature distribution. Monitor training stability.

**Test**: Check that gain predictions improve sign accuracy (positive vs negative).

---

### 5. Curriculum Learning Not Implemented (LOW PRIORITY)

**Problem**: `config.yaml` has 4 curriculum stages over 70 epochs, but `train.py` doesn't implement curriculum. `max_epochs: 50` means it never reaches later stages.

**Solution**: Implement simple 2-stage curriculum or extend training.

**File**: `/teamspace/studios/this_studio/insight/train.py`

**Changes**:
- Option A (simple): Increase `max_epochs` to 80 in config
- Option B (proper): Implement curriculum in Trainer.fit()

**Code change for Option A**:
```yaml
# In conf/config.yaml line 58:
trainer:
  max_epochs: 80  # was 50
```

**Code change for Option B** (in train.py):
```python
# Add to Trainer.__init__:
self.curriculum = self.cfg.get("curriculum", {})
self.curriculum_stages = self.curriculum.get("stages", [])

# Add to Trainer.fit():
def fit(self):
    # ... existing setup ...
    for epoch in range(1, self.max_epochs + 1):
        # Apply curriculum stage
        current_stage = self._get_current_stage(epoch)
        if current_stage:
            self._apply_curriculum_stage(current_stage)
        # ... rest of training loop ...

def _get_current_stage(self, epoch):
    epoch_so_far = 0
    for stage in self.curriculum_stages:
        epoch_so_far += stage["epochs"]
        if epoch <= epoch_so_far:
            return stage
    return self.curriculum_stages[-1] if self.curriculum_stages else None

def _apply_curriculum_stage(self, stage):
    # Update dataset gain_range if specified
    if "param_ranges" in stage and "gain" in stage["param_ranges"]:
        self.train_dataset.gain_range = tuple(stage["param_ranges"]["gain"])
    # Update loss lambda_type
    if "lambda_type" in stage:
        self.criterion.lambda_type = stage["lambda_type"]
    # Update Gumbel temperature
    if "gumbel_temperature" in stage:
        self.model.param_head.gumbel_temperature.data.fill_(stage["gumbel_temperature"])
```

**Recommendation**: Start with Option A (simple epoch increase). Only implement Option B if Options 1-4 don't break the plateau.

**Impact**: More training time allows model to converge. Curriculum provides staged difficulty.

**Risk**: Low for Option A. Medium for Option B (curriculum bugs could destabilize training).

**Test**: Monitor val_loss beyond epoch 50. Should continue decreasing.

---

## Implementation Order (Highest Impact, Lowest Risk First)

### Phase 1: Critical Dataset & Loss Fixes (Do First)
1. **Fix gain distribution** (`dataset.py`) - HIGHEST impact, LOW risk
2. **Rebalance loss weights** (`loss_multitype.py`) - HIGH impact, LOW risk
3. **Test**: Run `python test_multitype_eq.py` to verify no breakage
4. **Train**: Run 10 epochs to verify training stability

### Phase 2: Architecture Improvements (Do Second)
5. **Raise attention temperature** (`differentiable_eq.py`) - MEDIUM impact, LOW risk
6. **Fix mel normalization** (`differentiable_eq.py`) - MEDIUM impact, MEDIUM risk
7. **Test**: Run test suite
8. **Train**: Run 20 epochs, monitor for NaN divergence

### Phase 3: Training Extensions (Do Last if Needed)
9. **Increase max_epochs** (`config.yaml`) - LOW impact, LOW risk
10. **Optional**: Implement curriculum if plateau persists
11. **Train**: Full 80 epoch run

---

## Validation Plan

After each phase, verify:

1. **Tests pass**: `python test_multitype_eq.py`
2. **Training stable**: No NaN loss, gradients < 10.0
3. **Metrics improve**: 
   - Gain MAE < 3.0 dB (target: < 2.0 dB)
   - Freq MAE < 2.0 octaves (target: < 1.5 octaves)
   - Type accuracy > 60% (target: > 75%)

### Diagnostic Commands

```bash
# After training, diagnose performance
python diagnose_gain.py --checkpoint checkpoints/best.pt

# Check learning curves
python analyze_learning_curves.py checkpoints/training_history.json

# Verify gradient flow
python diagnose_gradients.py --checkpoint checkpoints/best.pt
```

---

## Expected Outcomes

### After Phase 1 (Dataset + Loss):
- Gain MAE: 2.99 → ~2.2 dB (uniform sampling provides balanced signal)
- Type accuracy: 52% → ~60% (higher lambda_type weight)
- Training: More stable, less plateau

### After Phase 2 (Attention + Normalization):
- Freq MAE: 2.0 → ~1.6 octaves (softer attention improves context)
- Gain MAE: ~2.2 → ~1.8 dB (absolute level preserved)
- Type accuracy: ~60% → ~70% (better frequency awareness)

### After Phase 3 (Extended training):
- All metrics improve by ~10-20% from longer convergence
- Final target: Gain MAE < 1.5 dB, Freq MAE < 1.2 octaves, Type acc > 75%

---

## Rollback Plan

If any change causes instability:

1. **Gain distribution breaks training**: Revert to Beta(2,2) instead of uniform
2. **Loss rebalancing hurts convergence**: Restore lambda_gain=2.0
3. **Attention temperature causes NaN**: Revert to 0.1, verify attention implementation
4. **Mel normalization breaks inference**: Restore mean subtraction
5. **Curriculum bugs**: Remove curriculum, stick to simple max_epochs increase

All changes are isolated to specific files/modules, making rollback straightforward.

---

## Configuration Changes Summary

### conf/config.yaml
```yaml
trainer:
  max_epochs: 80  # Line 58, was 50

loss:
  lambda_gain: 1.0  # Line 38 (add if not present), was implicit 2.0
  lambda_type: 1.0  # Line 39, was 0.5
```

### Critical Files for Implementation

1. `/teamspace/studios/this_studio/insight/dataset.py` - Gain sampling fix
2. `/teamspace/studios/this_studio/insight/loss_multitype.py` - Loss weight rebalancing
3. `/teamspace/studios/this_studio/insight/differentiable_eq.py` - Attention temp + mel norm
4. `/teamspace/studios/this_studio/insight/conf/config.yaml` - Epoch increase + loss weights
5. `/teamspace/studios/this_studio/insight/train.py` - Optional curriculum implementation

---

## Testing Strategy

### Unit Tests (After Each Change)
```bash
python test_multitype_eq.py
```

### Integration Test (After Each Phase)
```bash
# Quick 5-epoch training run
python train.py  # With modified max_epochs=5 in config
# Check: No NaN, loss decreasing, reasonable metrics
```

### Full Training (Final Validation)
```bash
python train.py  # With full config (80 epochs)
# Monitor: training_v8.log
# Diagnose: python diagnose_gain.py --checkpoint checkpoints/best.pt
```

---

## Success Criteria

The plan is successful if:
1. All tests pass (`test_multitype_eq.py`)
2. Training completes 80 epochs without NaN divergence
3. Final metrics exceed baseline:
   - Gain MAE < 2.5 dB (vs 3.0 baseline)
   - Freq MAE < 1.7 octaves (vs 2.0 baseline)
   - Type accuracy > 65% (vs 53% baseline)
4. Learning curves show continued improvement (no plateau at epoch 50)

---

## Notes

- **Streaming inference compatibility**: All changes preserve streaming. No architectural changes to TCN encoder.
- **Incremental approach**: Each phase can be tested independently before proceeding.
- **Diagnostic tools**: Existing scripts (`diagnose_gain.py`, `analyze_learning_curves.py`) will validate improvements.
- **Training stability**: Monitor `embed_var` metric (should stay > 0.5) to prevent encoder collapse.
