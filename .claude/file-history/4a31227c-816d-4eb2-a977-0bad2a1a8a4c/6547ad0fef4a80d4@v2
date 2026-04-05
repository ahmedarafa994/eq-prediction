# Plan: Improve Blind EQ Parameter Prediction

## Context

The IDSP model trains a TCN encoder + `MultiTypeEQParameterHead` to blindly estimate EQ parameters (gain, freq, Q, filter type) from processed audio — without the dry signal. Training is running and progressing through curriculum stages, but current validation metrics show room for improvement:

- **gain_mae ~4.6 dB** — nearly 5 dB error on a ±24 dB range
- **freq_mae ~1.76 octaves** — very large frequency error
- **q_mae ~0.40 decades** — moderate
- **type_acc 100%** — but only because early stages use peaking-only data

The core problem is that the model's frequency prediction is still inaccurate. The architecture has several bottlenecks that limit blind estimation quality. This plan addresses the highest-impact issues in order.

## Key Files

- `insight/model_tcn.py` — TCN encoder + parameter head architecture
- `insight/differentiable_eq.py` — `MultiTypeEQParameterHead` (gain/freq/Q/type prediction)
- `insight/loss_multitype.py` — `MultiTypeEQLoss`, Hungarian matching
- `insight/train.py` — training loop, curriculum stages
- `insight/dsp_frontend.py` — STFT/mel frontend
- `insight/dataset.py` — data generation
- `insight/conf/config.yaml` — all hyperparameters

---

## Step 1: Fix mel_profile input to use wet-dry difference instead of raw wet mel

**Problem**: `train.py:481-482` passes `mel_diff = (mel_frames - dry_mel).mean(dim=-1)` as `mel_profile`. But the model's `_predict_from_embedding` also receives `mel_profile` and the parameter head's attention operates on it. The attention-based frequency predictor needs to see the **EQ effect** (wet - dry difference), not the raw wet spectrum. Currently the mel_profile is the *mean* over time of the difference, losing temporal detail but that's acceptable — however the `mel_profile` passed during validation at `train.py:630` uses `mel_frames.mean(dim=-1)` which is NOT the difference. This inconsistency hurts validation metrics.

**Change in `train.py`**: The validate function should compute mel_profile consistently:
- `train.py:630`: Change `mel_diff = (mel_frames - dry_mel).mean(dim=-1)` and pass that as `mel_profile` (same as training).

Actually looking more carefully at lines 630-631, the validate function already computes `mel_diff = (mel_frames - dry_mel).mean(dim=-1)` and passes it as `mel_profile`. This is consistent. **Skip this step** — the code is already correct.

## Step 2: Increase TCN capacity for better spectral feature extraction

**Problem**: The TCN has `channels=128`, `num_blocks=4`, `num_stacks=2`, `kernel_size=3` → receptive field = 121 frames = 121×256/44100 ≈ 0.7s. With 1.5s audio, this covers about half the signal. The model only has **1.56M parameters** — quite small for this task.

**Changes**:
- `config.yaml`: Increase `kernel_size` from 3 to 5 (better frequency resolution per frame)
- `config.yaml`: Increase `num_blocks` from 4 to 6 (deeper temporal reasoning)
- This gives receptive field = 1 + 2×(5-1)×(2^0 + 2^1 + 2^2 + 2^3 + 2^4 + 2^5) × 2 stacks = 1 + 2×4×63×2 = 1009 frames → ~5.8s — more than enough

**Impact**: Medium-high. Better temporal modeling of spectral evolution.

## Step 3: Add spectral reconstruction loss (cycle consistency)

**Problem**: The model predicts EQ parameters but never verifies whether applying those parameters to the input actually reconstructs the spectral shape. The `hmag_loss` compares predicted vs target frequency responses, but this is in the parameter space — not in the actual spectral effect on the audio.

**Change in `loss_multitype.py`**: Add a loss term that computes:
1. Apply predicted EQ to the dry signal's mel-spectrogram → `reconstructed_wet_mel`
2. Compare `reconstructed_wet_mel` vs actual `wet_mel`
3. This creates a direct "does my prediction actually explain the observation?" signal

This is the most important missing piece — the model currently has no direct spectral feedback loop for blind estimation.

**Implementation**:
- In `train.py` training loop, after getting predictions, compute `pred_H_mag` applied to `dry_mel` and compare to `wet_mel`
- Add L1 loss on log-mel difference: `L_recon = |log(wet_mel) - log(dry_mel * pred_H)|`
- Weight: start with `lambda_recon: 0.5` in config

**Impact**: **High** — this is the key blind-estimation signal.

## Step 4: Add warmup learning rate schedule

**Problem**: Each curriculum stage resets the LR scheduler to CosineAnnealing but doesn't warm up. When transitioning from one stage to the next (especially when the dataset distribution changes), the model can be destabilized by large gradients on unfamiliar data.

**Change in `train.py`**: Add a linear warmup for the first 5% of steps in each stage before switching to cosine decay.

**Impact**: Medium. Prevents loss spikes at stage transitions.

## Step 5: Improve frequency prediction with multi-scale mel input

**Problem**: The parameter head's attention-based frequency prediction uses a 2-layer CNN on the mel profile. With 256 mel bins spanning 20–20000 Hz, the CNN has 7×5 receptive field = 11 bins — far too narrow to capture the full EQ bump shape. The attention weights tend toward uniformity (entropy loss helps but doesn't fix the root cause).

**Change in `differentiable_eq.py` `MultiTypeEQParameterHead`**:
- Add a dilated convolution path: `Conv1d(2, 32, kernel_size=3, dilation=4)` alongside the existing CNN, then concatenate features. This expands the receptive field without more parameters.
- Alternatively, add a global pooling path (mean, max) that gives the attention a summary of the full spectrum.

**Impact**: **High** for frequency accuracy. The current narrow-receptive-field CNN is a bottleneck.

## Step 6: Add Exponential Moving Average (EMA) of model weights

**Problem**: Training with large batch sizes (2048) and cosine annealing can produce noisy final weights. EMA smoothing gives a more stable model for inference.

**Change in `train.py`**: Maintain an EMA copy of model weights with decay=0.999. Use EMA weights for validation and checkpointing.

**Impact**: Medium. Often gives 5-15% improvement on regression tasks.

## Step 7: Validate with spectral reconstruction metric

**Problem**: Current validation metrics (gain_mae, freq_mae, q_mae, type_acc) measure parameter accuracy but not whether the predicted EQ curve actually matches the true EQ curve. A model could have ok parameter MAE but produce a poor frequency response match.

**Change in `train.py` validate function**: Add:
- `H_mag_mae`: L1 distance between predicted and target frequency response curves (in dB)
- `spectral_cosine`: cosine similarity between predicted and target mel-spectra after applying predicted EQ

**Impact**: Medium. Better visibility into actual model quality.

---

## Implementation Order

1. **Step 2** (TCN capacity) — config change only, immediate
2. **Step 3** (spectral reconstruction loss) — highest impact, new loss component
3. **Step 5** (multi-scale mel in parameter head) — high impact for frequency
4. **Step 6** (EMA weights) — moderate impact, easy to add
5. **Step 4** (LR warmup) — stability improvement
6. **Step 7** (better validation metrics) — observability

## Verification

After each step:
1. Run `python train.py` for at least 2 curriculum stages
2. Check that val_loss is decreasing and gain_mae/freq_mae improve vs baseline
3. Run `python test_model.py` and `python test_multitype_eq.py` to verify gradient flow
4. After all steps, run full 90-epoch training and compare final metrics

## Current Training Status

The background training task completed successfully. The updated config has already been applied with:
- `audio_duration: 1.5` (reduced from 3.0 for faster iteration)
- `batch_size: 2048` (auto-calibrated)
- `lambda_freq_anchor: 5.0` (increased from 2.0)
- 6 curriculum stages with all 5 filter types in later stages
- `early_stopping_patience: 25`
- `lr_schedule: plateau` with patience=5
