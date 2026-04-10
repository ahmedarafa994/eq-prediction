# Surgical Fixes for EQ Parameter Estimation — Design Spec

**Date:** 2026-04-10
**Status:** Draft
**Goal:** Double model accuracy across all metrics with 5 targeted fixes, minimal architecture changes, ~2-3 days.

## Current vs Target Metrics

| Metric | Current (epoch 73) | Target | Required Improvement |
|--------|-------------------|--------|---------------------|
| Gain MAE | 2.38 dB | < 1.0 dB | ~2.4x better |
| Frequency MAE | 1.51 octaves | < 0.5 octaves | ~3x better |
| Q MAE | 0.41 | < 0.2 | ~2x better |
| Type Accuracy | 53.5% | > 90% | +37pp |
| LowShelf Accuracy | ~24% | > 85% | +61pp |

## Root Cause Analysis

The training plateau at epoch 73 stems from five identified issues:

1. **H_mag_typed gradient blackout** — `train.py:696-700` wraps teacher-forced spectral in `torch.no_grad()` AND `.detach()`, completely blocking gradient flow from the typed spectral loss.
2. **16 competing loss terms** — Static weights cause gradient conflicts; many auxiliary terms (embed_var, contrastive, film_diversity, activity, spread) add noise without clear benefit.
3. **No proper render-domain loss** — The spectral loss compares soft-type renders but lacks a hard-render path that directly evaluates gain/freq/Q accuracy.
4. **Insufficient data diversity** — 50k synthetic-only samples lack the spectral complexity of real audio.
5. **Type classifier lacks spectral shape features** — Classification relies solely on learned embeddings without explicit DSP-informed features.

---

## Fix 1: Enable Gradient Flow on Teacher-Forced Spectral Loss

### Problem

At `train.py:694-700`:
```python
# Use torch.no_grad() to avoid OOM from extra differentiable biquad pass
with torch.no_grad():
    H_mag_typed = model_ref.dsp_cascade(
        pred_gain.detach(), pred_freq.detach(), pred_q.detach(),
        filter_type=target_ft,
    )
```

Both `torch.no_grad()` and `.detach()` block all gradients. The typed spectral loss (λ=1.0) contributes to total loss numerically but provides zero learning signal. This means ~6% of the loss function is dead weight.

### Fix

Remove `torch.no_grad()` context and `.detach()` calls. Use `torch.cuda.amp.autocast` for memory efficiency instead:

```python
# Teacher-forced spectral: render predicted params with GT types
# Gradients flow to gain/freq/Q predictions (NOT to type classifier)
with torch.amp.autocast('cuda', enabled=False):
    H_mag_typed = model_ref.dsp_cascade(
        pred_gain.float(), pred_freq.float(), pred_q.float(),
        filter_type=target_ft,
    )
```

Key: ground truth `target_ft` is used (not predicted types), so gradients flow only to `pred_gain`, `pred_freq`, `pred_q` — exactly the parameters that need improvement.

### OOM Mitigation

The original `no_grad` was added to prevent OOM. To manage memory:
- Cast to float32 only for the biquad pass (avoid bf16 precision issues in DSP math)
- The biquad computation is lightweight (matrix multiply + division) — the OOM was likely from accumulating the full backward graph alongside the main forward pass, which is fixed by gradient checkpointing if needed
- If OOM persists: `torch.utils.checkpoint.checkpoint()` on the biquad call

### Expected Impact

~15% improvement on gain MAE, moderate improvement on freq/Q MAE. The typed spectral loss directly compares "what your gain/freq/Q predictions produce with the correct filter type" to the target — it's the most direct learning signal for parameter accuracy.

### Files Modified

- `insight/train.py` — lines 694-700

---

## Fix 2: Simplify Loss to 6 Core Terms + Learned Weighting

### Problem

Current loss function (`loss_multitype.py:730-747`) sums 16 terms with static weights:

| Term | Lambda | Purpose | Keep? |
|------|--------|---------|-------|
| loss_gain | 2.0 | Huber on matched gain | Yes |
| loss_freq | 1.0 | Huber on matched freq | Yes |
| loss_q | 0.5 | Huber on matched Q | Yes |
| loss_type | 4.0 | Focal loss on type | Yes |
| loss_spectral | 1.5 | Soft-type spectral L1 | Yes |
| loss_typed_spectral | 1.0 | Teacher-forced spectral | Yes |
| loss_hmag | 0.25 | Raw H_mag L1 | Drop (redundant with spectral) |
| loss_activity | (internal) | Band activity reg | Drop (premature regularization) |
| loss_spread | (internal) | Freq spread reg | Drop (conflicts with freq loss) |
| loss_embed_var | 0.1 | Embedding variance | Drop (auxiliary) |
| loss_contrastive | 0.05 | Contrastive embedding | Drop (auxiliary) |
| loss_hdb | 0.5 | H_db prediction L1 | Drop (redundant with spectral) |
| loss_gain_zero | 0.25 | Zero-gain penalty | Drop (interferes with gain learning) |
| loss_type_entropy | 0.02 | Type distribution entropy | Drop (conflicts with focal) |
| loss_type_prior | 0.05 | Type prior matching | Drop (conflicts with focal) |
| loss_film_diversity | 0.1 | FiLM diversity | Drop (already 0.0 constant) |

### Fix

**Step A: Reduce to 6 core terms** with rebalanced static weights:

```python
total_loss = (
    lambda_spectral * loss_spectral       # λ=3.0 — primary signal
    + lambda_typed_spectral * loss_typed   # λ=2.0 — type-decoupled signal (Fix 1 enables this)
    + lambda_type * loss_type              # λ=3.0 — type classification
    + lambda_gain * loss_gain              # λ=1.5 — direct parameter supervision
    + lambda_freq * loss_freq              # λ=1.0 — direct parameter supervision
    + lambda_q * loss_q                    # λ=0.5 — direct parameter supervision
)
```

**Step B: Add learned uncertainty weighting** (Kendall et al. 2018):

Instead of fixed lambdas, learn a `log_sigma` per loss term:

```python
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_losses=6):
        super().__init__()
        # Initialize log_sigma so initial weights ≈ static lambdas above
        self.log_sigma = nn.Parameter(torch.tensor([
            -1.1,  # spectral: exp(-2*-1.1)/2 ≈ 4.5 → ~3.0 effective
            -0.7,  # typed_spectral: ~2.0 effective
            -1.1,  # type: ~3.0 effective
            -0.4,  # gain: ~1.5 effective
             0.0,  # freq: ~1.0 effective
             0.7,  # q: ~0.5 effective
        ]))

    def forward(self, losses):
        # losses: list of 6 scalar losses
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-2 * self.log_sigma[i])
            total += precision * loss + self.log_sigma[i]
        return total
```

This allows the model to automatically downweight noisy losses and upweight informative ones during training. The `log_sigma` regularization term prevents all weights from going to zero.

### Expected Impact

~10-15% improvement across all metrics. Removing conflicting auxiliary losses clears up gradient signal; learned weighting prevents any single loss from dominating.

### Files Modified

- `insight/loss_multitype.py` — Replace total_loss computation (lines 729-747), add UncertaintyWeightedLoss class
- `insight/train.py` — Add `log_sigma` parameters to optimizer
- `insight/conf/config.yaml` — Update loss section (remove dropped lambdas, add new weights)

---

## Fix 3: Multi-Scale Render-Domain Loss

### Problem

Current spectral losses compare frequency responses at a single FFT size (n_fft from config, typically 2048). This misses:
- Narrow Q peaks that need high frequency resolution to resolve
- Broad spectral tilts (shelves) that are better captured at lower resolution
- The actual perceptual effect of parameter errors

### Fix

Add a hard-render multi-scale spectral loss:

```python
def multi_scale_spectral_loss(pred_gain, pred_freq, pred_q, pred_type_logits,
                               target_H_mag, dsp_cascade, fft_sizes=[256, 512, 1024]):
    """
    Hard-render: use argmax types (no gradients to type classifier).
    Gradients flow to gain/freq/Q through the differentiable biquad.
    Multi-scale: compare at 3 resolutions for peaks + tilts.
    """
    hard_types = pred_type_logits.argmax(dim=-1)  # (B, num_bands)

    total = 0.0
    for n_fft in fft_sizes:
        # Render with hard types — gradients to gain/freq/Q only
        H_mag_pred = dsp_cascade(pred_gain, pred_freq, pred_q,
                                  filter_type=hard_types, n_fft=n_fft)

        # Resample target to match resolution
        target_resampled = F.interpolate(
            target_H_mag.unsqueeze(1), size=n_fft // 2 + 1, mode='linear'
        ).squeeze(1)

        # Log-domain L1 (perceptually meaningful dB comparison)
        pred_db = 20 * torch.log10(H_mag_pred.sum(dim=1).clamp(min=1e-6))
        tgt_db = 20 * torch.log10(target_resampled.clamp(min=1e-6))
        total += F.l1_loss(pred_db, tgt_db)

    return total / len(fft_sizes)
```

This replaces the current `loss_hmag` slot. Gradients from hard types flow through the biquad to gain/freq/Q via `torch.where` in `compute_biquad_coeffs_multitype` (which already handles integer type indices).

### Expected Impact

~10% improvement on freq/Q MAE. Multi-scale comparison catches both narrow peaks and broad tilts that single-scale misses. Log-domain comparison weights errors perceptually.

### Files Modified

- `insight/loss_multitype.py` — Add `multi_scale_spectral_loss` function, integrate into forward
- `insight/train.py` — Pass `dsp_cascade` reference to loss function

---

## Fix 4: Scale Dataset to 200k + MUSDB18 Real Audio

### Problem

50k synthetic samples (noise, sweep, harmonic, speech_like) lack spectral diversity. The model overfits to synthetic spectral patterns and fails to generalize. Real audio has transients, harmonics, noise floors, and spectral evolution that synthetic signals don't capture.

### Fix

**Step A: Increase synthetic dataset to 200k:**

In `conf/config.yaml`:
```yaml
data:
  dataset_size: 200000
  val_dataset_size: 5000
```

**Step B: Enable MUSDB18 real audio mixing:**

The `dataset_pipeline/` already supports MUSDB18. Configure mixed training:
```yaml
data:
  musdb_fraction: 0.3          # 30% real audio, 70% synthetic
  musdb_root: "data/musdb18"
```

Modify `dataset.py` to alternate between synthetic and MUSDB18 sources. For MUSDB18 samples:
1. Load a random 3-second segment from MUSDB18 stems
2. Apply random EQ with known parameters (same as synthetic)
3. Use the dry stem as reference for computing actual H_mag

**Step C: Harder parameter ranges in later curriculum stages:**

Current curriculum already ramps from ±6dB to ±12dB gain. Add more extreme Q values in the final stage:
```yaml
# Stage 3 (full_range)
q_bounds: [0.05, 15.0]  # was [0.1, 10.0]
```

### Expected Impact

~10% improvement across all metrics. Real audio diversity forces the model to learn generalizable features rather than synthetic-specific patterns.

### Files Modified

- `insight/conf/config.yaml` — Update dataset_size, add musdb_fraction
- `insight/dataset.py` — Add MUSDB18 source mixing logic
- `insight/train.py` — Handle mixed dataset creation

---

## Fix 5: Spectral Shape Features for Type Classifier

### Problem

The type classifier in `MultiTypeEQParameterHead` uses only learned embeddings (the encoder output). It lacks explicit DSP-informed features that distinguish filter types by their spectral shape. LowShelf accuracy is 24% because shelves and peaking filters look similar in the embedding space without spectral shape cues.

### Fix

**Key challenge:** The type classifier predicts types BEFORE H_mag is computed (H_mag depends on type selection via Gumbel-Softmax). To avoid this chicken-and-egg problem, we compute "what-if" frequency responses for ALL 5 filter types per band, then derive shape features from each.

For each band with predicted (gain, freq, Q), run the biquad for each of the 5 possible types. This gives 5 candidate frequency responses per band. Compute 4 shape features from each → 20 features per band total.

```python
def compute_per_type_shape_features(pred_gain, pred_freq, pred_q, dsp_cascade,
                                     sample_rate=44100, n_fft=512):
    """
    For each band, compute frequency response for ALL 5 filter types,
    then derive shape features. Gives the type classifier direct information:
    "what would the response look like if this were peaking vs lowshelf?"

    pred_gain/freq/q: (B, num_bands)
    Returns: (B, num_bands, 20) — 4 features × 5 types
    """
    B, N = pred_gain.shape
    all_features = []

    for type_idx in range(5):  # peaking, lowshelf, highshelf, hp, lp
        type_tensor = torch.full((B, N), type_idx, device=pred_gain.device, dtype=torch.long)

        # Compute per-band frequency response for this type
        H_mag = dsp_cascade(pred_gain, pred_freq, pred_q,
                            filter_type=type_tensor, n_fft=n_fft)  # (B, N, n_fft_bins)
        H_db = 20 * torch.log10(H_mag.clamp(min=1e-6))

        n_bins = H_mag.shape[-1]
        mid = n_bins // 2
        q25 = n_bins // 4
        q75 = 3 * n_bins // 4

        # 1. Energy asymmetry — low vs high frequency
        low_e = H_mag[..., :mid].pow(2).mean(-1)
        high_e = H_mag[..., mid:].pow(2).mean(-1)
        asymmetry = (low_e - high_e) / (low_e + high_e).clamp(min=1e-8)

        # 2. Roll-off slope
        slope = (H_db[..., q75:].mean(-1) - H_db[..., :q25].mean(-1)) / 40.0

        # 3. Peak-to-plateau ratio (peaking = high, shelf = low)
        peak_val = H_db.max(dim=-1).values
        mean_val = H_db.mean(dim=-1)
        peak_ratio = (peak_val - mean_val) / 12.0

        # 4. Edge energy ratio (HP/LP have near-zero energy on one side)
        edge_ratio = H_mag[..., :q25].pow(2).mean(-1) / H_mag[..., q75:].pow(2).mean(-1).clamp(min=1e-8)
        edge_ratio = torch.log10(edge_ratio.clamp(min=1e-4, max=1e4)) / 4.0  # normalized

        all_features.append(torch.stack([asymmetry, slope, peak_ratio, edge_ratio], dim=-1))

    return torch.cat(all_features, dim=-1)  # (B, N, 20)
```

Inject into `MultiTypeEQParameterHead` between gain/freq/Q prediction and type prediction:

```python
# After gain/freq/Q heads produce predictions:
shape_features = compute_per_type_shape_features(
    pred_gain, pred_freq, pred_q, self.dsp_cascade, n_fft=512)  # (B, N, 20)

# Concatenate with band embedding for type classifier
type_input = torch.cat([band_embedding, shape_features], dim=-1)
# Update type classifier first linear: input_dim += 20
```

The 5 biquad evaluations at n_fft=512 are cheap (~0.5ms total). All operations are differentiable — gradients flow from type loss through shape features back to gain/freq/Q predictions, creating a beneficial coupling.

### Feature Discrimination Table

Each row shows expected feature values when computed with the *correct* filter type. The classifier sees all 5 type-variants simultaneously and learns which pattern matches the observed audio.

| Feature | Peaking | LowShelf | HighShelf | HighPass | LowPass |
|---------|---------|----------|-----------|----------|---------|
| Asymmetry | ~0 | Negative | Positive | Positive | Negative |
| Slope | ~0 | Negative | Positive | Positive | Negative |
| Peak-to-plateau | High | Low | Low | Medium | Medium |
| Edge energy ratio | ~0 | Positive | Negative | Very negative | Very positive |

LowShelf vs Peaking: peak-to-plateau (low vs high) + edge ratio sign. LowShelf vs HighShelf: asymmetry + edge ratio sign flip. HP/LP: extreme edge ratios (near-zero energy on one side).

### Expected Impact

Type accuracy +10-15pp overall, LowShelf accuracy +20-30pp. Per-type biquad evaluation gives the classifier direct "what would this look like" information rather than relying on the encoder to discover spectral shape differences from raw spectrograms.

### Files Modified

- `insight/model_tcn.py` — Add `compute_per_type_shape_features`, modify `MultiTypeEQParameterHead` to accept shape features (input_dim += 20)
- `insight/differentiable_eq.py` — No changes needed (biquad already supports integer type indices)

---

## Implementation Order

1. **Fix 1** (gradient flow) — 30 min, immediate impact, zero risk
2. **Fix 2** (loss simplification) — 2 hours, clears gradient conflicts
3. **Fix 3** (multi-scale render loss) — 2 hours, improves freq/Q signal
4. **Fix 5** (spectral shape features) — 2 hours, improves type accuracy
5. **Fix 4** (dataset scaling) — 3 hours, requires data pipeline work

Fixes 1-3 should be implemented and validated together (one training run). Fix 5 can be added in the same run. Fix 4 requires dataset regeneration and a fresh training run.

## Validation Plan

After implementing all fixes, run a fresh training for 120 epochs. Checkpoints saved per the existing schedule. Track:

- Per-metric MAE (gain, freq, Q) at each validation step
- Per-type accuracy breakdown (especially LowShelf)
- Loss component magnitudes and learned sigma values
- Gradient norms per parameter group

**Success criteria:** Meeting all 5 target metrics by epoch 100.

**Fallback:** If surgical fixes plateau at 60-70% of target, escalate to architecture changes (encoder redesign, larger model). But based on the root cause analysis, the gradient blackout alone likely accounts for 15-20% of the gap.

## Risk Assessment

| Fix | Risk | Mitigation |
|-----|------|------------|
| Fix 1 (gradients) | OOM | autocast + checkpoint if needed |
| Fix 2 (loss simplification) | Removing useful signal | Monitor per-metric trends in first 10 epochs; re-add terms if regression |
| Fix 3 (multi-scale) | Minor compute increase | 3 FFT sizes is ~3x spectral compute; negligible vs encoder cost |
| Fix 4 (dataset) | Longer data loading | Precompute and cache mel spectrograms |
| Fix 5 (shape features) | Feature engineering bias | Features are differentiable and derived from model output, not hand-tuned thresholds |
