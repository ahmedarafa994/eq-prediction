# Training Stability Fix Plan

## Context

Training run v8 showed catastrophic NaN collapse at epoch 5 — validation loss becomes NaN and training loss drops to 0.0000. Root cause: soft-blended biquad coefficients in `forward_soft` produce unstable frequency responses when type accuracy is low (~50%), causing NaN in `hmag_loss` → NaN gradients → corrupted optimizer state (Adam momentum/variance buffers) → permanent weight corruption.

**Already applied** (verified in current codebase):
- Config: loss weights rebalanced, LR=1e-4, weight_decay=0.01, curriculum temps adjusted, precision=fp32
- `forward_soft`: pole stability check (pole_check > 1.95), per-band clamp [1e-6, 1e3], total clamp [1e-6, 1e4]
- `train.py`: NaN batch skipping with component ID, per-param gradient zeroing, grad clip at 1.0
- Parameter head dropout at 0.2 (both trunk and classification head)
- Head LR multiplier already 3.0x (not 5.0x)

**Why it still fails**: NaN protections stop NaN *forward output* but NaN enters optimizer state through intermediate backward nodes. When `backward()` runs, NaN propagates through the computational graph *before* gradients are zeroed. Adam's `.step()` then writes NaN into momentum/variance buffers. Even if gradients are zeroed afterward, the next `.step()` reads corrupted optimizer state → NaN weights. The `_recover_from_nan` only triggers when weights are NaN, but the optimizer state can be corrupted while weights are still finite.

---

## Changes

### 1. Optimizer State NaN Sanitization (CRITICAL)
**File: `insight/train.py` — after `self.optimizer.step()` (~line 469)**

After each optimizer step, check Adam state for NaN and reset corrupted entries:

```python
self.optimizer.step()

# Sanitize optimizer state: reset any Adam momentum/variance buffers that contain NaN
for state in self.optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            v.zero_()
```

This is the highest-impact fix — it directly addresses the optimizer state corruption that causes permanent training death.

### 2. Add Anomaly Detection for First 2 Epochs
**File: `insight/train.py` — at start of `train_one_epoch`**

Enable PyTorch anomaly detection during warmup to get exact stack traces for any NaN:

```python
if epoch <= 2:
    torch.autograd.set_detect_anomaly(True)
else:
    torch.autograd.set_detect_anomaly(False)
```

Add after the existing NaN check (~line 394), replacing `torch.autograd.set_detect_anomaly` calls. This adds ~2x overhead for 2 epochs only, but gives precise traceback of where NaN originates in backward.

### 3. Reduce BatchNorm Momentum
**File: `insight/model_tcn.py` — `SpectralConvBlock2D.__init__` (line 107)**

```python
# Before:
self.bn = nn.BatchNorm2d(out_channels)
# After:
self.bn = nn.BatchNorm2d(out_channels, momentum=0.05)
```

Slower running stat tracking (0.05 vs default 0.1) reduces BatchNorm's susceptibility to accumulating corrupted statistics from NaN batches. Single-line change.

### 4. Add Config Key for `lambda_type_match`
**File: `insight/conf/config.yaml` — in loss section (after line 50)**

Config reads `lambda_type_match` with default 0.5 (train.py:213), but it's not in the YAML. The Hungarian matcher uses it for cost weighting. Add it explicitly:

```yaml
lambda_type_match: 1.0  # Match config to Hungarian matcher (was default 0.5)
```

### 5. Add SpecAugment to Training
**File: `insight/train.py` — add method to Trainer class**

Add frequency and time masking to mel-spectrograms during training. Insert as a method on the Trainer, called after `_prepare_input` in `train_one_epoch`:

```python
def _spec_augment(self, mel_spec, freq_mask_param=15, time_mask_param=30, num_masks=2):
    """Apply SpecAugment: frequency and time masking for regularization."""
    B, n_mels, T = mel_spec.shape
    augmented = mel_spec.clone()
    for _ in range(num_masks):
        f = torch.randint(0, freq_mask_param, (1,)).item()
        f0 = torch.randint(0, max(n_mels - f, 1), (1,)).item()
        augmented[:, f0:f0+f, :] = 0
    for _ in range(num_masks):
        t = torch.randint(0, time_mask_param, (1,)).item()
        t0 = torch.randint(0, max(T - t, 1), (1,)).item()
        augmented[:, :, t0:t0+t] = 0
    return augmented
```

Call in `train_one_epoch` after `_prepare_input`, before model forward:
```python
if self.training:
    mel_frames = self._spec_augment(mel_frames)
```

### 6. Increase Dataset Size
**File: `insight/conf/config.yaml` (line 18)**

```yaml
dataset_size: 30000   # Increase from 10000 (1.8M params needs more data)
```

1.8M params on 8K training samples is ~228 params/sample. 30K gives ~24K train samples (~76 params/sample).

---

## Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `insight/train.py` | Optimizer state NaN sanitization (#1) | After ~469 |
| `insight/train.py` | Anomaly detection first 2 epochs (#2) | Start of train_one_epoch |
| `insight/model_tcn.py` | BatchNorm momentum 0.05 (#3) | Line 107 |
| `insight/conf/config.yaml` | lambda_type_match + dataset_size (#4, #6) | Lines 18, ~50 |
| `insight/train.py` | SpecAugment method + call (#5) | New method + in train_one_epoch |

## Execution Order

1. **Changes #1-4** (stability/core fixes) → run 10 epochs → verify no NaN
2. **Changes #5-6** (regularization/capacity) → run 20 epochs → verify convergence
3. **Full 80-epoch run**

## Verification

1. `python train.py` for 5 epochs:
   - No NaN in val_loss
   - No NaN batches logged
   - Type accuracy starts improving
2. Existing tests: `python test_eq.py && python test_model.py && python test_multitype_eq.py`
3. 20-epoch checkpoint: type accuracy > 60%, train/val gap < 20%
4. 80-epoch full run: val loss < 10, type accuracy > 70%
