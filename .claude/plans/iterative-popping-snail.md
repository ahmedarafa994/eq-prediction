# IDSP EQ Estimator: Comprehensive Optimization Roadmap

## Context

The IDSP blind EQ parameter estimation model is stuck in a severe training plateau. After 165 epochs across 6 curriculum stages, the model achieves:
- **Gain MAE: 5.32 dB** (target: <1.5 dB)
- **Freq MAE: 1.607 octaves** (target: <0.25 oct)
- **Q MAE: 0.365 decades**
- **Type accuracy: 40.6%** (target: >90%, near-random for 5 types)
- **Val loss: 24.62** — flat for the last 40+ epochs

The model catastrophically forgot peaking accuracy when shelf types were introduced (type_acc dropped from 100% → 46% at epoch 56 and never recovered). Frequency error is worst at low frequencies (2.12 oct) suggesting the model cannot detect bass EQ changes.

### Root Causes Identified

1. **Frequency head architecture is flawed** — soft attention weighted mean produces diffuse, mid-frequency-biased estimates; the direct regression path fights it at 50/50 blend
2. **No mel-spectrogram normalization** — raw log-mel has extreme scale differences (low-freq bins 10x larger than high-freq), crippling the TCN encoder
3. **Catastrophic forgetting** — each curriculum stage completely replaces data; no rehearsal of previous types
4. **Loss function has 10 competing components** — two different frequency losses (`lambda_freq_internal=3.0` Huber + `lambda_freq_anchor=3.0` L1) produce conflicting gradients
5. **Gain tanh activation saturates** — `tanh(x)*24` gives near-zero gradients for |x|>3, stalling extreme gain predictions

---

## Phase 0: Diagnostic Experiments (no retraining)

Before changing code, confirm root causes with 4 quick experiments:

### 0A: Attention Visualization
Write a script loading `best.pt`, running 100 samples, and visualizing attention weights per band. If attention is near-uniform (entropy > 70% of max), confirms RC1.

### 0B: Direct Regression Ablation
Set `freq_blend_weight` to -10 (sigmoid ≈ 0, pure direct regression), run validation. If freq MAE drops, confirms attention is harmful.

### 0C: Mel Normalization Ablation
Add instance normalization to `STFTFrontend.mel_spectrogram()`, run validation without retraining. If low-freq error improves, confirms RC2.

### 0D: Loss Gradient Analysis
For a fixed batch, compute gradient norms from each loss component w.r.t. `freq_direct` and `gainq` weights. Reveals which losses provide useful gradients.

---

## Phase 1: Critical Architecture Fixes (retrain from scratch)

### Step 1: Fix Frequency Regression Head
**File:** `insight/differentiable_eq.py`, `MultiTypeEQParameterHead.__init__()` and `.forward()`

- Change `freq_blend_weight` initialization from `torch.tensor(0.0)` → `torch.tensor(-6.0)` so sigmoid(-6) ≈ 0.002, making direct regression dominant
- The direct sigmoid-mapped regression path already exists and works — it's just being drowned out by the attention path
- Widen prior centers from `linspace(0.15, 0.85)` to `linspace(0.1, 0.9)` for better frequency coverage

**Impact:** Freq MAE 1.6 → ~0.8 oct (biggest single improvement)
**Cost:** Zero — actually reduces compute by de-emphasizing attention

### Step 2: Add Mel-Spectrogram Normalization
**File:** `insight/dsp_frontend.py`, `STFTFrontend.mel_spectrogram()` (line 144)

Add instance normalization after computing log-mel:
```python
# After: log_mel = torch.log(mel_spec)
log_mel = (log_mel - log_mel.mean(dim=-2, keepdim=True)) / (log_mel.std(dim=-2, keepdim=True) + 1e-4)
```

Also apply the same normalization in `MultiTypeEQParameterHead.forward()` where `mel_profile = mel_frames.mean(dim=-1)` is computed (line ~803).

**Impact:** Freq MAE low-freq 2.12 → ~0.8 oct; overall gain estimation improves
**Cost:** Negligible

### Step 3: Fix Gain Activation
**File:** `insight/differentiable_eq.py`, `MultiTypeEQParameterHead.forward()` (line ~783)

Replace `tanh(x) * 24.0` with `tanh(x / 3.0) * 24.0` — pushes saturation from |x|>3 to |x|>9, maintaining gradients across the full range.

**Impact:** Gain MAE 5.3 → ~3.5 dB
**Cost:** Zero

### Step 4: Simplify Loss Function
**File:** `insight/loss_multitype.py`, `PermutationInvariantParamLoss` and `MultiTypeEQLoss`

- Remove internal `lambda_freq_internal` multiplier (set to 0.0) — let `lambda_freq_anchor=5.0` be the sole frequency loss
- Disable `L_coverage`, `L_spread`, `L_attn_entropy`, `L_activity` for Phase 1 (set all to 0.0)
- Reduce `L_hmag` from 1.0 → 0.5, `L_type` from 3.0 → 2.0, `L_recon` from 0.5 → 0.1
- Keep: `L_param` (1.0), `L_freq_anchor` (5.0), `L_type` (2.0), `L_hmag` (0.5)

Phase 1 config.yaml loss section:
```yaml
loss:
  lambda_param: 1.0
  lambda_type: 2.0
  lambda_spectral: 0.0
  lambda_hmag: 0.5
  lambda_activity: 0.0
  lambda_spread: 0.0
  lambda_coverage: 0.0
  lambda_freq_anchor: 5.0
  lambda_attn_entropy: 0.0
  lambda_freq_match: 1.0
  lambda_q_match: 0.5
  lambda_freq_internal: 0.0
  lambda_recon: 0.1
```

**Impact:** Cleaner gradient signal, faster convergence, breaks plateau
**Cost:** Slightly faster per step

### Step 5: Simplify Curriculum (2 stages only for Phase 1)
**File:** `insight/conf/config.yaml`

Replace 6 curriculum stages with 2:
```yaml
curriculum:
  stages:
  - name: peaking_foundation
    epochs: 30
    filter_types: [peaking]
    param_ranges:
      gain: [-12.0, 12.0]
      q: [0.3, 5.0]
    freq_range: [100.0, 8000.0]
    min_gain_db: 3.0
    lambda_type: 0.0      # no type loss when only 1 type
    gumbel_temperature: 1.0
  - name: full_multitype
    epochs: 100
    filter_types: [peaking, lowshelf, highshelf, highpass, lowpass]
    param_ranges:
      gain: [-24.0, 24.0]
      q: [0.1, 10.0]
    min_gain_db: 0.5
    lambda_type: 2.0
    gumbel_temperature: 0.5
```

**Impact:** Eliminates the shelf_types catastrophic forgetting event
**Cost:** None

**Phase 1 Success Criteria:** Gain MAE < 3.0 dB, Freq MAE < 0.8 oct, Type acc > 60%

---

## Phase 2: Curriculum & Training Strategy

### Step 6: Add Replay Buffer to Curriculum
**File:** `insight/train.py`, `_build_stage_dataset()` (line ~276)

When building stage N's dataset, include 20% replay data from previous stage distributions:
```python
# Mix in peaking-only replay data in all multi-type stages
if 'peaking' not in stage_filter_types or len(stage_filter_types) > 1:
    replay_dataset = SyntheticEQDataset(
        filter_types=('peaking',), num_bands=num_bands,
        gain_range=(-18, 18), freq_range=(100, 12000),
        q_range=(0.3, 8.0), min_gain_db=3.0, size=10000, ...
    )
    current_dataset = ConcatDataset([current_dataset, replay_dataset])
```

**Impact:** Type accuracy 60% → ~80%, prevents catastrophic forgetting
**Cost:** 20% more data per stage

### Step 7: Soft Type Introduction
When a new stage introduces new filter types, start with weighted sampling (70% old types / 30% new types) and linearly shift to equal proportions over 5 epochs.

### Step 8: Classification Head Warmup
Freeze the type classification head weights for the first 3 epochs of each new stage, allowing regression parameters to adapt first.

**Phase 2 Success Criteria:** Type acc > 80%, Freq MAE < 0.5 oct, Gain MAE < 2.5 dB

---

## Phase 3: Model Capacity & Refinement

### Step 9: Increase Model Capacity
**File:** `insight/conf/config.yaml`

```yaml
model:
  encoder:
    channels: 192
    num_blocks: 6
    num_stacks: 3        # increase from 2
    kernel_size: 5
    embedding_dim: 192
```

Receptive field increases from ~505 to ~1500 frames. Keep `use_full_spectrum: false`.

**Impact:** All metrics improve further
**Cost:** ~2-3x parameters, ~2x training time

### Step 10: Re-enable Regularization Losses
Gradually enable `L_coverage` (0.05), `L_spread` (0.01) once the model can reliably estimate basic parameters.

### Step 11: Learning Rate Tuning
Phase 1-2: `lr=3e-4` (higher for simpler loss landscape)
Phase 3: `lr=1e-4` (lower for larger model)
Use OneCycleLR with `pct_start=0.5`, `div_factor=5`.

**Phase 3 Success Criteria:** Freq MAE < 0.25 oct, Gain MAE < 1.5 dB, Type acc > 90%

---

## Phase 4: Evaluation & Experiment Tracking

### Step 12: Add Per-Type and Per-Band Metrics
**File:** `insight/train.py`, `validate()` method

Log per-type accuracy (peaking, lowshelf, etc.) and per-band frequency/gain MAE. Add spectral cosine similarity between predicted and target H_mag.

### Step 13: Composite Checkpoint Selection
Save best checkpoint based on composite metric, not just val_loss:
```python
composite = 0.3 * gain_mae + 0.5 * freq_mae + 0.2 * (1 - type_acc)
```

### Step 14: TensorBoard Integration
**File:** `insight/train.py`

Add `SummaryWriter` logging all 10 loss components independently, attention entropy per band, parameter histograms (pred vs target gain/freq/q distributions), and gradient norms per module.

### Step 15: Real Audio Test Script
Already created as `insight/test_real_audio.py`. Run with multiple musdb18 tracks and EQ presets after each phase to track real-world performance.

---

## Hyperparameter Summary

| Parameter | Current | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Learning rate | 1e-4 | 3e-4 | 3e-4 | 1e-4 |
| Batch size | 1024 | 2048 | 2048 | 1024 |
| Grad accumulate | 4 | 4 | 4 | 8 |
| Channels | 128 | 128 | 128 | 192 |
| Num stacks | 2 | 2 | 2 | 3 |
| Embedding dim | 128 | 128 | 128 | 192 |
| freq_blend_weight init | 0.0 | -6.0 | -6.0 | -6.0 |
| Gain activation | tanh*24 | tanh(x/3)*24 | tanh(x/3)*24 | tanh(x/3)*24 |
| lambda_freq_anchor | 3.0 | 5.0 | 5.0 | 4.0 |
| lambda_type | 3.0 | 2.0 | 2.0 | 1.5 |
| lambda_freq_internal | 3.0 | 0.0 | 0.0 | 0.0 |
| lambda_hmag | 1.0 | 0.5 | 0.5 | 0.3 |

## Files to Modify

| File | Changes |
|---|---|
| `insight/differentiable_eq.py` | Fix freq_blend init (line ~760), gain activation (line ~783), prior centers |
| `insight/dsp_frontend.py` | Add normalization to `mel_spectrogram()` (line ~148) |
| `insight/loss_multitype.py` | Remove `lambda_freq_internal` from `PermutationInvariantParamLoss` (line ~369) |
| `insight/train.py` | Add replay buffer, per-type metrics, TensorBoard, composite checkpoint |
| `insight/conf/config.yaml` | New loss weights, simplified 2-stage curriculum, Phase 3 capacity |

## Verification

After each phase:
1. Run `python evaluate_model.py --checkpoint checkpoints/<new>/best.pt --n_samples 50` for synthetic metrics
2. Run `python test_real_audio.py --checkpoint checkpoints/<new>/best.pt --eq-preset complex` for real audio
3. Run `python test_real_audio.py --checkpoint checkpoints/<new>/best.pt --eq-preset bass_boost` for peaking-only
4. Compare to Phase 1/2/3 success criteria in the table above
