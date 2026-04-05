# Plan: Reducing High Training Loss in IDSP EQ Estimator

## Context

Training loss (32.65) is ~2x validation loss (16.19) at epoch 28 of a 165-epoch curriculum. This is abnormal — normally train loss < val loss. The primary causes are: (1) EMA weights used for validation but not training, (2) over-weighted `freq_anchor` loss (lambda=5.0 contributing 26.5% of total loss), and (3) diffuse attention over 256 mel bins causing 1.69-octave frequency MAE. The model also OOM-crashes at epoch 29.

Training crashed and needs to resume from epoch 28 checkpoint at `checkpoints/idsp-eq-v3-improved/epoch_028.pt`.

## Changes

### 1. Fix train/val loss comparison (ROOT CAUSE of 2x gap)

**File:** `insight/train.py` line 669

Validation uses EMA weights (`self._load_ema_weights()`) while training uses raw weights. This alone explains most of the 2x gap.

**Fix:** Also compute an EMA-weighted training loss for logging (don't change actual training). After each training epoch, run a few batches through the model with EMA weights loaded and log `train_ema_loss` alongside `train_loss`. This gives a fair comparison with `val_loss`.

Alternatively (simpler): just stop swapping EMA weights for validation. Use raw weights for both. EMA weights are still saved and can be loaded for inference/export.

**Priority:** HIGH | **Risk:** LOW

### 2. Rebalance loss weights

**File:** `insight/conf/config.yaml`

The `freq_anchor` loss (lambda=5.0, raw=1.73, weighted=8.65) dominates the total loss. Frequency error is penalized three times: in `param_loss`, in `freq_anchor`, and in `hmag`. This creates gradient conflict.

Changes:
- `lambda_freq_anchor`: 5.0 → **3.0** (stages 1-2) / **2.0** (stages 3-6)
- `lambda_hmag`: 2.0 → **1.0**
- All 6 curriculum stages: update `lambda_freq_anchor` accordingly

**File:** `insight/loss_multitypy.py`, `PermutationInvariantParamLoss.forward` line 290

Add internal frequency weighting inside param_loss:
- `loss = loss_gain + 3.0 * loss_freq + loss_q` (currently equal weighting)
- This provides strong frequency supervision within param_loss, reducing reliance on freq_anchor

**Expected impact:** Total loss drops from ~32 to ~22. freq_anchor contribution: 8.65 → 5.2.

**Priority:** HIGH | **Risk:** LOW

### 3. Sharpen frequency attention

**File:** `insight/differentiable_eq.py`

The attention mechanism (temperature=0.1 over 256 mel bins) is too soft — effective focus spans ~10 bins. Octave precision requires 1-3 bins.

Changes:
- Add `attn_temperature_target` buffer to `MultiTypeEQParameterHead` (target=0.02)
- **File:** `insight/conf/config.yaml` — add `attn_temperature` to each curriculum stage:
  - easy_peaking: 0.1 → medium_peaking: 0.05 → frequency_spread: 0.03 → shelf_types+: 0.02
- **File:** `insight/train.py` `_setup_stage` — update `attn_temperature` alongside `gumbel_temperature`
- Initialize `freq_blend_weight` from 0.0 → **2.0** (sigmoid(2)=0.88, favoring attention path)

**Expected impact:** freq_mae improves from 1.69 to ~1.0-1.2 octaves.

**Priority:** HIGH | **Risk:** MEDIUM (too-sharp early attention may hurt gradient flow — mitigated by direct regression path)

### 4. Fix OOM crashes

**File:** `insight/conf/config.yaml`

- `batch_size`: 2048 → **1024**
- `accumulate_grad_batches`: 2 → **4** (keeps effective batch size at 4096)
- **File:** `insight/model_tcn.py` — add `use_gradient_checkpointing` option, enable in `GatedResidualBlock`

**Expected impact:** Eliminates OOM. ~30% more compute per epoch but stable training.

**Priority:** HIGH | **Risk:** LOW

### 5. Fix mel_profile normalization inconsistency

**File:** `insight/train.py` lines 502-503 and `insight/differentiable_eq.py` lines 801-803

The encoder's `freq_profile_proj` receives un-normalized mel_profile, while the param head normalizes it internally. This inconsistency harms spectral shape learning.

**Fix:** Normalize `mel_profile` in `train.py` before passing to `_predict_from_embedding`. Remove duplicate normalization inside `MultiTypeEQParameterHead.forward`.

**Priority:** MEDIUM | **Risk:** LOW

### 6. Per-stage warmup for learning rate

**File:** `insight/train.py` `_setup_stage` lines 444-459

Already implemented with warmup_sched + cosine_sched via SequentialLR. This is correct — verify it's working properly after resuming from checkpoint.

**Priority:** LOW (already implemented) | **Risk:** N/A

## Files to Modify

| File | Changes |
|------|---------|
| `insight/conf/config.yaml` | Loss weights, batch_size, attn_temperature per stage |
| `insight/loss_multitype.py` | Add freq weight inside PermutationInvariantParamLoss |
| `insight/differentiable_eq.py` | attn_temperature annealing, freq_blend_weight init |
| `insight/train.py` | EMA logging fix, attn_temperature update in _setup_stage, mel_profile normalization, gradient checkpointing |
| `insight/model_tcn.py` | Gradient checkpointing in GatedResidualBlock |

## Implementation Order

1. **Config changes** (loss weights, batch size, attn_temperature) — no code dependencies
2. **loss_multitype.py** — add freq weighting in param_loss
3. **differentiable_eq.py** — attn_temperature target, freq_blend_weight init
4. **train.py** — wire attn_temperature updates, fix EMA logging, fix mel_profile norm
5. **model_tcn.py** — gradient checkpointing support

## Verification

1. Resume training from epoch 28 checkpoint: `python train.py --resume checkpoints/idsp-eq-v3-improved/epoch_028.pt`
2. After 3 epochs, check:
   - `train_loss` should be closer to `val_loss` (EMA logging fix)
   - `freq_mae` should start decreasing (attention sharpening)
   - No OOM errors (batch size reduction)
3. After full `medium_peaking` stage (epoch 40):
   - `freq_mae` target: < 1.2 octaves
   - `gain_mae` target: < 4.0 dB
   - Total loss target: < 25 (train), < 15 (val)
4. Run existing tests to verify no regressions:
   ```bash
   python test_eq.py && python test_model.py && python test_multitype_eq.py && python test_streaming.py
   ```
