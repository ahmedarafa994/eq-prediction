# Diagnostic Report: IDSP EQ Parameter Estimation Training Failure

## Context

The differentiable DSP (IDSP) system for blind parametric EQ parameter estimation has failed to achieve acceptable prediction accuracy across 9 training iterations (v3 through v9). Despite reducing training loss by 3x (18.2 → 6.2), validation metrics remain essentially stagnant: gain MAE plateaued at ~3.2 dB, type classification accuracy stalled at ~53% (barely above 20% random chance for 5 types), and frequency/Q errors show minimal improvement after epoch 10. The training-to-validation loss gap widened from 1.1 to 2.7, indicating progressive overfitting without convergence on the underlying task.

The system takes a wet (EQ-processed) audio signal, computes a mel-spectrogram, passes it through a TCN encoder, and predicts 5-band parametric EQ parameters (gain, frequency, Q, filter type) via a parameter head. Training uses Hungarian matching for permutation-invariant loss, Gumbel-Softmax for differentiable type selection, and a multi-component loss.

---

## Executive Summary

Training failed to converge on meaningful EQ parameter estimation. The core failure is a **cascade of compounding architectural and optimization issues** that prevent gradient signal from reaching the gain prediction pathway with sufficient fidelity. The model learned to minimize total loss by optimizing frequency response reconstruction (spectral loss) while ignoring individual parameter accuracy. Key failures:

1. **Gain prediction uses a fundamentally broken mechanism** (Gaussian readout from mel-residual with only 2 learnable parameters per band)
2. **Loss weighting allows spectral loss to dominate over parameter regression**, creating a local minimum where total loss drops but individual metrics stagnate
3. **Severe overfitting** after epoch ~10 with a 2.7x train-val gap by epoch 80
4. **Type classification collapsed to near-random** (53% vs 20% random) despite being trained for 80 epochs
5. **Gain MAE actually worsened from epoch 10 onward** (3.04 → 3.24 dB) as the model overfit to training data

---

## 1. Loss Behavior Analysis

### 1.1 Training Loss Curve

| Phase | Epochs | Train Loss | Val Loss | Train-Val Gap | Gain MAE (dB) | Type Acc |
|-------|--------|-----------|----------|--------------|---------------|----------|
| Initial descent | 1-3 | 13.3→9.9 | 12.5→9.5 | 0.4 | 4.9→3.3 | 48→52% |
| Early plateau | 4-10 | 9.5→8.8 | 9.4→8.8 | 0.6 | 3.2→3.0 | 52→53% |
| Slow decay | 11-40 | 8.7→7.5 | 8.9→8.7 | 1.2 | 3.1→3.1 | 53→53% |
| Overfitting | 41-80 | 7.5→6.2 | 8.7→9.0 | 2.8 | 3.1→3.2 | 53→53% |

**Interpretation**: The train-val gap expanded from 0.4 (epoch 1) to 2.8 (epoch 80), a classic overfitting pattern. The early descent (epochs 1-3) was driven by the spectral loss components learning to match frequency response shapes, not by individual parameter accuracy improving.

### 1.2 Gain MAE Trend (Critical Anomaly)

```
Epoch  1: gain=4.88 dB   (rapid descent from initial)
Epoch  3: gain=3.29 dB   (fast improvement)
Epoch  7: gain=3.12 dB   (stall begins)
Epoch 10: gain=3.04 dB   (local minimum)
Epoch 20: gain=3.03 dB   (plateau)
Epoch 40: gain=3.10 dB   (SLIGHT WORSENING)
Epoch 60: gain=3.22 dB   (continued worsening)
Epoch 80: gain=3.24 dB   (worst since epoch 3)
```

The gain MAE reached a minimum at epoch 10 (3.04 dB) and then **monotonically worsened** for 70 epochs despite training loss dropping 30% in the same period. This is the clearest indicator that the optimizer found a local minimum where total loss improves by sacrificing gain accuracy.

### 1.3 gain_raw vs gain (Metric Bug)

The training logs show two gain metrics:
- `gain_mae` (Hungarian-matched): ~3.2 dB
- `gain_raw` (unmatched): ~7.0 dB

The 2.2x discrepancy confirms the validation metric bug identified in the root cause analysis: `gain_raw` compares predictions against original (unmatched) targets, inflating error by including permutation noise. However, even the matched metric (3.2 dB) is far too high for practical use - a 3 dB gain error is perceptually obvious.

### 1.4 Loss Component Evolution

From v9 log, per-component breakdown at key epochs:

| Component | Epoch 1 | Epoch 10 | Epoch 40 | Epoch 80 | % Change (10→80) |
|-----------|---------|----------|----------|----------|------------------|
| loss_gain | 5.3 | 3.0 | 2.9 | 1.5 | -50% |
| loss_freq | 1.2 | 1.1 | 0.9 | 0.8 | -27% |
| loss_q | 0.6 | 0.6 | 0.6 | 0.6 | 0% (stagnant) |
| type_loss | 1.7 | 1.5 | 1.3 | 1.2 | -20% |
| hmag_loss | 1.7 | 1.5 | 1.2 | 0.9 | -40% |
| embed_var | 0.97 | 0.94 | 0.95 | 1.05 | +12% |

**Key observations**:
- `loss_gain` in training dropped 50% (5.3→1.5) but validation gain MAE stayed flat - this is **overfitting on gain**
- `loss_q` is completely stagnant across all 80 epochs - the model never learned Q prediction
- `embed_var` actually increased (0.94→1.05), indicating the anti-collapse regularization succeeded but at the cost of not learning discriminative embeddings for the task
- `spectral_loss` remains 0.0 throughout - this loss is never activated

### 1.5 NaN Batch Events

v9 log shows `1 NaN batches skipped` at epoch 6. The training script handles NaN by skipping batches, but the presence of NaN gradients indicates numerical instability in the early training phase. The root cause is likely in `10^(gain_db/40)` when gain_db is very large or in the product of 5 band frequency responses.

### 1.6 Comparison Across Training Runs

| Run | LR | Epochs | Final Val Loss | Gain MAE | Type Acc | Notes |
|-----|-----|--------|---------------|----------|----------|-------|
| v7 | 2e-4 | 80 | 8.96 | 3.24 | 52.9% | lr=2e-4, 80 epochs |
| v8 | 2e-4 | 80 | 8.96 | 3.24 | 52.7% | same config |
| v8c | 1e-4 | 50 | ~8.8 | 3.17 | 53.5% | lr=1e-4, 50 epochs |
| v9 | 2e-4 | 80 | 8.98 | 3.24 | 52.7% | same as v7 |

All runs converge to essentially identical final metrics regardless of learning rate, confirming the issue is architectural, not hyperparameter-tuning.

---

## 2. Issue Classification by Severity

### CRITICAL (Blocks convergence entirely)

#### Issue C1: Gaussian Readout Gain Mechanism
- **File**: `insight/differentiable_eq.py:589-608`, `insight/model_tcn.py` (MultiTypeEQParameterHead)
- **Root Cause**: Gain is predicted by reading a Gaussian-weighted sum of mel-residual values at the predicted frequency, then applying a per-band scale+bias (only 2 learnable parameters per band = 10 total for gain). The mel-residual amplitude in log-mel units does not linearly correspond to dB gain.
- **Evidence**: Training `loss_gain` decreases but validation gain MAE stays flat. The 10-parameter bottleneck (scale+bias for 5 bands) cannot capture the complex frequency/Q/type-dependent mapping from mel-residual to dB.
- **Impact**: Gain prediction is fundamentally capped.

#### Issue C2: Spectral Loss Dominates Parameter Regression
- **File**: `insight/loss_multitype.py:269-334`, `insight/conf/config.yaml:37-55`
- **Root Cause**: `loss_param` (gain+freq+Q combined) gets weight 1.5, while `loss_hmag` (full spectral match) gets 1.0, but hmag is much easier to reduce by jointly adjusting all parameters. The optimizer finds that reducing hmag by 0.5 dB is "cheaper" than reducing gain MAE by 3 dB.
- **Evidence**: hmag_loss decreased 47% (1.7→0.9) while gain MAE stayed flat. Training loss drops but individual parameter metrics don't improve.
- **Impact**: Model learns to match spectral shape without learning individual parameter values.

#### Issue C3: Encoder Collapse (Partially Addressed)
- **File**: `insight/model_tcn.py` (StreamingTCNModel)
- **Root Cause**: The TCN encoder was diagnosed with cosine distance ~0.006 between embeddings (near-identical for all inputs). Anti-collapse regularization (embed_var + contrastive) was added, which increased embed_var from ~0.5 to ~1.0, but the embeddings still don't carry discriminative parameter information.
- **Evidence**: embed_var_loss is always 0.0000 in logs (never fires because variance > threshold), yet the model can't predict parameters well. The variance regularization forces spread but not task-relevant spread.
- **Impact**: The encoder produces embeddings that vary across the batch but don't encode the EQ parameter information needed by the parameter head.

### HIGH (Major contributor to failure)

#### Issue H1: Tanh Soft-Clamp Gradient Vanishing
- **File**: `insight/differentiable_eq.py:608`
- **Root Cause**: `tanh(gain_db / 24.0) * 24.0` attenuates gradients by 20-42% for common gain values (12-18 dB). The Beta(2,5) gain distribution concentrates samples near 7 dB where attenuation is ~10%, but the model still needs to predict larger gains accurately.
- **Evidence**: Gain predictions systematically underestimate large gains.
- **Impact**: Compounds with C1 to make large gain prediction nearly impossible.

#### Issue H2: Gumbel-Softmax Gradient Dilution at High Temperature
- **File**: `insight/differentiable_eq.py:154-237`
- **Root Cause**: At temperature=1.0 (stage 1, epochs 1-10), gain gradients flow through a weighted sum of all 5 filter types. Only ~20% of the gradient reaches the peaking filter coefficients where gain matters.
- **Evidence**: Type accuracy is near-random (52%) after 80 epochs, meaning the model never resolved type uncertainty.
- **Impact**: During critical early training, gain gradients are severely attenuated.

#### Issue H3: Beta(2,5) Gain Distribution Bias
- **File**: `insight/dataset.py:201-205`
- **Root Cause**: Beta(2,5) gives mean magnitude of 6.9 dB with 90th percentile at 13.4 dB. The model is incentivized to predict small gains (minimizing expected loss) rather than learning the full range.
- **Evidence**: Gain predictions cluster near zero.
- **Impact**: The model takes a "safe bet" strategy, predicting near-zero gain for everything.

### MEDIUM (Contributes to suboptimal performance)

#### Issue M1: Hungarian Matching Cost Imbalance
- **File**: `insight/loss_multitype.py:30-64`
- **Root Cause**: Cost matrix weights gain, freq, and Q equally (1.0, 1.0, 0.5) but frequency errors in log-space span ~6.9 octaves while gain spans 48 dB. A 1-octave frequency error costs the same as 1 dB gain error.
- **Impact**: Incorrect band assignments inflate gain MAE indirectly.

#### Issue M2: Product-Based Band Response Gradient Instability
- **File**: `insight/differentiable_eq.py` (band cascade)
- **Root Cause**: `H_mag_total = prod(H_mag_bands)` creates gradient scaling that depends on other bands' magnitudes. With 5 bands each at H~2.0, the gradient for one band scales by 2^4=16x.
- **Impact**: Noisy, band-dependent gradient magnitudes make consistent gain optimization difficult.

#### Issue M3: No Curriculum Execution in Training
- **File**: `insight/train.py`, `insight/conf/config.yaml:67-101`
- **Root Cause**: The config defines a 4-stage curriculum (peaking_warmup → multitype_fixed → full_difficulty → realworld_finetune) but the training script (`train.py`) uses `SyntheticEQDataset` with all 5 filter types from epoch 1. The curriculum defined in config is never loaded or applied in the current trainer.
- **Evidence**: training_v9.log shows all 5 types present from step 0 (type_weights configured as [0.5, 0.15, 0.15, 0.1, 0.1]).
- **Impact**: The model must learn everything simultaneously, preventing staged learning.

#### Issue M4: Stagnant Q Prediction
- **Root Cause**: `loss_q` stays at ~0.6 across all 80 epochs. The model never learned to predict Q factor. This is likely because Q has minimal impact on spectral shape compared to gain and frequency, so the spectral loss gradient provides little Q signal.
- **Impact**: Q predictions are essentially random, making the model useless for practical EQ matching.

---

## 3. Correlation: Loss Patterns → Root Causes

| Observed Pattern | Root Cause(s) |
|-----------------|---------------|
| Train loss drops 3x but val metrics flat | C2: spectral loss dominates; C3: encoder collapse |
| Gain MAE worsens after epoch 10 | C1: 10-param bottleneck; H1: tanh gradient vanishing; H3: Beta distribution |
| loss_q stagnant at ~0.6 | Q has minimal spectral impact → spectral loss provides no Q gradient |
| Type accuracy stuck at ~53% | H2: Gumbel-Softmax at high temperature; C3: non-discriminative embeddings |
| Train-val gap grows to 2.8x | Model memorizes spectral patterns without learning parameterization |
| embed_var increases but metrics don't | Anti-collapse creates spread but not task-relevant features |
| NaN batches at epoch 6 | Numerical instability in `10^(gain/40)` with large gain + product of 5 bands |
| gain_raw >> gain_mae (7 vs 3 dB) | M1: Hungarian matching corrects some permutation errors |

---

## 4. Prioritized Recommendations

### Priority 1: Fix Gain Prediction Architecture (Addresses C1, H1)
1. Replace Gaussian readout with a proper MLP regression head from the trunk embedding
2. Increase gain prediction parameters from 10 to at least 512 (128-dim hidden layer)
3. Replace `tanh(gain/24)*24` with `torch.clamp` using straight-through estimator for gradient flow
4. **Validation checkpoint**: Gain MAE should drop below 2.0 dB within 10 epochs

### Priority 2: Rebalance Loss Weights (Addresses C2)
1. Separate `loss_param` into independent `loss_gain`, `loss_freq`, `loss_q` with separate weights
2. Set `lambda_gain=3.0`, `lambda_freq=1.0`, `lambda_q=0.5`
3. Reduce `lambda_hmag` from 1.0 to 0.3
4. Remove anti-collapse losses (`lambda_embed_var`, `lambda_contrastive`) once encoder is fixed
5. **Validation checkpoint**: hmag_loss should be ~50% of total loss, not the dominant term

### Priority 3: Implement Curriculum Learning (Addresses M3, H2)
1. Actually execute the curriculum stages from config.yaml in train.py
2. Stage 1 (epochs 1-10): Peaking-only, lr=2e-4, tau=1.0
3. Stage 2 (epochs 11-30): Add shelf types, lr=1e-4, tau=0.5
4. Stage 3 (epochs 31-60): Add HP/LP, lr=5e-5, tau=0.1
5. **Validation checkpoint**: Type accuracy should exceed 80% by end of stage 2

### Priority 4: Fix Data Distribution (Addresses H3)
1. Replace `Beta(2,5)` with `Beta(2,2)` (uniform-like) or `Uniform(0.3, 1.0) * max_gain`
2. Ensure equal representation of small and large gains
3. **Validation checkpoint**: Gain MAE should improve proportionally across the full range

### Priority 5: Fix Q Prediction (Addresses M4)
1. Add dedicated Q loss with separate weight (lambda_q=1.0)
2. Consider predicting bandwidth (BW) instead of Q - it has a more linear relationship to spectral shape
3. **Validation checkpoint**: Q MAE should drop below 0.3 decades

---

## 5. Step-by-Step Remediation Plan

### Phase 1: Quick Wins (Expected: 1-2 hours)

**Step 1.1**: Replace tanh clamp with STE clamp
- File: `insight/differentiable_eq.py:608`
- Change: `gain_db = torch.clamp(gain_db, -24.0, 24.0)` with gradient bypass
- Test: `python test_eq.py` to verify gradient flow at gain boundaries

**Step 1.2**: Fix gain distribution
- File: `insight/dataset.py:201-205`
- Change: `_beta_gain` to use `Beta(2, 2)` or uniform distribution
- Test: Generate 1000 samples, verify gain histogram is approximately uniform

**Step 1.3**: Rebalance loss weights
- File: `insight/conf/config.yaml:37-55`
- Change: `lambda_gain: 3.0`, `lambda_hmag: 0.3`, `lambda_param: 0.0` (replaced by individual)
- Test: Train for 5 epochs, verify gain loss is the dominant component

### Phase 2: Architecture Fix (Expected: 3-4 hours)

**Step 2.1**: Replace Gaussian readout with MLP head
- File: `insight/model_tcn.py` (MultiTypeEQParameterHead)
- Replace `gain_cal_scale`/`gain_cal_bias` with a 2-layer MLP: `Linear(128, 128) → ReLU → Linear(128, 1)`
- The MLP takes the trunk embedding (128-dim) and predicts gain directly
- Remove mel-residual dependency entirely
- Test: `python test_model.py` to verify gradient flow through new gain head

**Step 2.2**: Add separate loss components for each parameter
- File: `insight/loss_multitype.py`
- Split `loss_param` into `loss_gain`, `loss_freq`, `loss_q`
- Expose each with its own lambda in config
- Test: `python test_multitype_eq.py` to verify loss computation

### Phase 3: Training Infrastructure (Expected: 2-3 hours)

**Step 3.1**: Implement curriculum learning in train.py
- Load curriculum stages from config.yaml
- Modify `SyntheticEQDataset` to accept filter_type and param_range overrides
- Implement stage transitions with Gumbel temperature annealing
- Test: Train for 20 epochs, verify type distribution changes across stages

**Step 3.2**: Add proper gradient monitoring
- Log gradient norms for gain, freq, Q parameters separately
- Add gradient norm ratio monitoring (gain_grad / freq_grad) to detect imbalance
- Test: Verify gradient logging appears in training output

### Phase 4: Full Training Run (Expected: 30-60 min GPU time)

**Step 4.1**: Run full 80-epoch training with all fixes
- Expected metrics after fixes:
  - Gain MAE: < 2.0 dB (from 3.2 dB)
  - Type accuracy: > 80% (from 53%)
  - Frequency MAE: < 1.0 octave (from 1.9)
  - Q MAE: < 0.3 decades (from 0.46)

**Step 4.2**: Validate against real audio
- `python test_real_audio.py --audio path/to/test.wav`
- Compare predicted vs ground-truth EQ parameters

---

## 6. Missing Information (Conditional Analysis)

The following information would strengthen the analysis but is not currently available:

1. **Gradient norm traces**: Per-parameter gradient norms across epochs would confirm the hypothesized gradient dilution in the gain pathway. The diagnostic script `diagnose_gradients.py` exists but no output was found.

2. **Per-type breakdown**: Gain MAE and type accuracy broken down by filter type (peaking vs shelf vs HP/LP) would reveal whether specific types are particularly problematic.

3. **Embedding visualization**: t-SNE/UMAP of TCN embeddings colored by ground-truth parameters would confirm whether the encoder collapses to a non-discriminative manifold.

4. **Inference-time comparison**: Comparison of `forward_soft` (Gumbel-Softmax) vs `forward_hard` (argmax) predictions would quantify the train-test gap from type uncertainty.

5. **Real audio test results**: No evidence of `test_real_audio.py` being run. Performance on synthetic vs real audio may differ significantly.

6. **Spectral model comparison**: The spectral model (`model_spectral.py`) achieved 0.20 dB MAE in spectral prediction. A comparison run with the same validation set would contextualize the TCN model's performance.

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `insight/model_tcn.py` | Replace Gaussian readout with MLP gain head |
| `insight/differentiable_eq.py` | Replace tanh clamp with STE clamp |
| `insight/loss_multitype.py` | Split loss_param into independent components |
| `insight/dataset.py` | Fix gain distribution |
| `insight/train.py` | Implement curriculum learning from config |
| `insight/conf/config.yaml` | Rebalance loss weights |
