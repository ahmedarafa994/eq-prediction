# Fix: Model plateau at gain MAE ~3 dB, freq MAE ~2 oct, type acc ~52%

## Context

After replacing the broken Gaussian readout with a gain MLP, matched gain MAE dropped from ~7 dB to ~3 dB (epoch 50). But the model plateaus: frequency MAE is stuck at ~2 octaves, type accuracy at ~52.7% (barely above random). The gain MLP proved the parameter head can learn; the bottleneck is now the training signal and architecture tuning.

## Root Causes

1. **Gain distribution biased toward 0 dB**: `Beta(2,5)` concentrates 80% of peaking gains near 0; HP/LP filters always have gain=0. The model sees mostly zero-gain examples.
2. **Loss weight imbalance**: `lambda_gain=2.0` makes gain contribute 56% of total loss, drowning out freq/type gradients.
3. **Attention temperature 0.1**: Too sharp — nearly one-hot attention loses frequency context.
4. **Mel normalization removes absolute level**: Mean-subtraction erases gain sign information.
5. **No curriculum applied**: Config has 4 stages but `train.py` ignores them. `max_epochs=50` is insufficient.

## Changes

### Phase 1: Dataset + Loss (highest impact, lowest risk)

#### 1. Fix gain distribution in `insight/dataset.py`

Replace `_beta_gain()` (line 201-205) with uniform sampling:
```python
def _sample_gain(self):
    sign = random.choice([-1, 1])
    magnitude = random.uniform(0.5, min(abs(self.gain_range[0]), abs(self.gain_range[1])))
    return sign * magnitude
```

Also fix HP/LP to allow small gains (lines 170, 173):
```python
# HP: g = 0.0 → small random gain
g = random.uniform(-3.0, 3.0)
# LP: same
g = random.uniform(-3.0, 3.0)
```

#### 2. Rebalance loss weights in `insight/conf/config.yaml` and `insight/loss_multitype.py`

In `config.yaml`:
```yaml
loss:
  lambda_param: 1.5    # keep
  lambda_type: 1.0     # was 0.5 — type accuracy is critical
  lambda_hmag: 0.5     # keep
  lambda_embed_var: 0.3 # was 0.5 — reduce anti-collapse (it's working)
  lambda_contrastive: 0.05  # was 0.1 — reduce
```

In `loss_multitype.py` constructor defaults (line 31), match config changes. Also change `lambda_gain` default from 2.0 to 1.0.

In `loss_multitype.py` `forward()` (line ~333), update total_loss formula:
```python
self.lambda_gain * loss_gain  # lambda_gain now 1.0 instead of 2.0
```

### Phase 2: Architecture tuning (medium risk)

#### 3. Make attention temperature learnable in `insight/differentiable_eq.py`

Line 506: Change from buffer to parameter:
```python
# OLD: self.register_buffer("attn_temperature", torch.tensor(0.1))
# NEW:
self.attn_temperature = nn.Parameter(torch.tensor(0.5))
```

Line 571: Already has `.clamp(min=0.01)` — this is fine for a learnable parameter.

#### 4. Fix mel profile normalization in `insight/differentiable_eq.py`

Lines 551-553: Remove mean subtraction, keep std normalization:
```python
# OLD:
mp_mean = mel_profile.mean(dim=-1, keepdim=True)
mp_std = mel_profile.std(dim=-1, keepdim=True).clamp(min=1e-4)
mel_profile_normed = (mel_profile - mp_mean) / mp_std

# NEW:
mp_std = mel_profile.std(dim=-1, keepdim=True).clamp(min=1e-4)
mel_profile_normed = mel_profile / mp_std
```

This preserves absolute level (needed to distinguish boost vs cut).

### Phase 3: Training duration

#### 5. Increase epochs in `insight/conf/config.yaml`

```yaml
trainer:
  max_epochs: 80  # was 50
```

## Files to modify

1. `insight/dataset.py` — `_beta_gain()` → uniform; HP/LP gains
2. `insight/conf/config.yaml` — loss weights, max_epochs
3. `insight/loss_multitype.py` — lambda_gain default, forward weights
4. `insight/differentiable_eq.py` — attn_temperature, mel normalization

## Verification

1. `python test_multitype_eq.py` — all tests pass (shapes, bounds, gradient flow)
2. Train 10 epochs: loss should be stable, gain MAE should decrease faster than v7
3. Train 80 epochs: target gain MAE < 2 dB, freq MAE < 1.5 oct, type acc > 60%
4. Check that `attn_temperature` parameter moves from its initial 0.5 value
