# Diagnosis: `freq_mae` Plateau in IDSP EQ Estimator Training

## Context

`freq_mae` (frequency error in octaves) plateaus at ~1.96 octaves across 15 epochs of Stage 1 ("easy_multitype") and shows no meaningful improvement after epoch 3. The training history shows:

| Epoch | freq_mae | freq_err_low | freq_err_mid | freq_err_high | train_loss |
|-------|----------|-------------|-------------|--------------|------------|
| 1     | 2.044    | 2.126       | 2.083       | 3.149        | 9.575      |
| 3     | 1.991    | 2.223       | 1.894       | 2.814        | 8.022      |
| 8     | 1.970    | 2.086       | 2.135       | 2.782        | 7.707      |
| 15    | 1.960    | 2.062       | 2.167       | 2.735        | 7.533      |

**Key observations:**
- `train_loss` IS improving (9.58 → 7.53) — so the optimizer works and gradients flow
- `gain_mae` improves slightly (4.60 → 4.04) — gain head learns somewhat
- `freq_mae` is essentially flat (~1.96) — frequency prediction is **stuck**
- `freq_err_high` is worst (2.7 octaves) — high-frequency estimation is weakest
- **~2 octaves of error means the model is barely better than random** over the [20, 20000] Hz range (~10 octaves)

## Root Cause Analysis (Ranked by Likelihood)

### Bug 1 (CRITICAL): Attention Position Bias Locks Bands to Fixed Positions

**File:** `differentiable_eq.py:665-668` (MultiTypeEQParameterHead)

```python
mel_pos = torch.linspace(0, 1, n_mels)
centers = torch.linspace(0.15, 0.85, num_bands)
gaussian_init = -1.5 * (mel_pos.unsqueeze(0) - centers.unsqueeze(1)) ** 2
self.attn_position_bias = nn.Parameter(gaussian_init)
```

The position bias is initialized as a **learned parameter** with 3 fixed Gaussian centers at 15%, 50%, 85% of the mel axis. In the attention computation (`differentiable_eq.py:720-721`):

```python
content = torch.einsum("bnf,bfm->bnm", queries, cnn_feat)
attn = F.softmax(content + self.attn_position_bias.unsqueeze(0), dim=-1)
```

**The problem:** With `n_mels=256` and only 3 bands, each Gaussian has a fairly wide receptive field. The `-1.5` scale means the peak bin gets weight `exp(0) = 1.0` and bins 0.5 mel-width away get `exp(-1.5 * 0.25) ≈ 0.69`. The attention is dominated by the position bias rather than the content signal, especially early in training when `content` (from random queries) is small. This causes all predicted frequencies to collapse to the 3 fixed centers regardless of input.

**Evidence from training history:** `freq_span` is consistently ~6.2 octaves (matches 15% to 85% of the log-frequency range = 0.15 × 10 to 0.85 × 10 = 1.5 to 8.5 octaves ≈ 7 octaves span). The model is just outputting the Gaussian centers.

**Confirmed by simulation:** Running the attention computation with the initial Gaussian bias shows:
- Attention entropy is **99%+ of maximum** (essentially uniform over all 256 bins)
- Band 0 weighted mean: 368 Hz instead of target ~56 Hz (collapses toward center)
- Band 2 weighted mean: 1087 Hz instead of target ~7096 Hz (collapses toward center)
- Predicted span: 7.0 octaves — exactly matching observed training value of ~6.2 octaves

**Fix:** Reduce initial position bias strength and add attention temperature sharpening.

### Bug 2 (HIGH): Frequency Attention Entropy Collapse

**File:** `differentiable_eq.py:718-726`

```python
cnn_in = torch.cat([mel_profile_normed.unsqueeze(1), pos], dim=1)
cnn_feat = self.mel_cnn(cnn_in)
queries = self.query_proj(trunk_out)
content = torch.einsum("bnf,bfm->bnm", queries, cnn_feat)
attn = F.softmax(content + self.attn_position_bias.unsqueeze(0), dim=-1)
mel_log_freqs = torch.linspace(self.log_f_min, self.log_f_max, n_mels, ...)
log_freq = (attn * mel_log_freqs.view(1, 1, n_mels)).sum(-1)
freq = torch.exp(log_freq)
```

Frequency is computed as a **weighted mean** over all 256 mel bins. This is inherently blurry — it can't produce sharp frequency estimates because the softmax over 256 bins distributes probability mass broadly. When the attention is diffuse (high entropy), the weighted mean regresses toward the center of the frequency range.

**Mechanical explanation:** If attention distributes uniformly, `log_freq = mean(log_freqs) ≈ log(sqrt(20*20000)) ≈ 2.83`, giving `freq ≈ 630 Hz`. With 3 bands each getting a broad Gaussian blob, the model can't resolve peaks narrower than ~1 octave.

**Fix:** Use a **top-k sparse attention** or sharpen the temperature so attention focuses on a few bins, making the weighted mean more peak-like.

### Bug 3 (HIGH): No Gradient Path from `freq_mae` Through Attention

**File:** `loss_multitype.py:231-241` (PermutationInvariantParamLoss)

```python
freq_elementwise = self.huber_elementwise(log_pred_f, log_match_f)
log_f_center = 0.5 * (log_pred_f + log_match_f)
octave_pos = (log_f_center - log_f_min) / (log_f_max - log_f_min)
freq_weight = 1.0 + octave_pos  # 1.0 at 20 Hz, 2.0 at 20 kHz
loss_freq = (freq_elementwise * freq_weight).mean()
```

The parameter loss for frequency applies Huber loss to `log(pred_freq)` vs `log(target_freq)`. The gradient flows through:
1. `log_freq = (attn * mel_log_freqs).sum(-1)` → `freq = exp(log_freq)`
2. `log(freq) = log_freq` → back to the attention weights

**The problem:** When attention is diffuse (near-uniform), `d(log_freq)/d(attn_i) = mel_log_freqs_i`. The gradient is proportional to the log-frequency coordinate. For a uniform attention, the gradient is:
```
d(loss)/d(attn_i) ∝ mel_log_freqs_i - target_log_freq
```
This gradient is the same sign for all bins on one side of the target, pushing all of them equally. With softmax normalization, this means the attention barely moves because all bins get pushed in similar directions.

**Fix:** Add a **direct frequency regression loss** that bypasses the attention mechanism, or use a sharper loss (L1 instead of Huber) for frequency.

### Bug 4 (MEDIUM): Huber Loss Delta Too Large for Frequency

**File:** `loss_multitype.py:195`

```python
self.huber = nn.HuberLoss(delta=1.0)
```

With `delta=1.0`, the Huber loss behaves as quadratic for errors < 1.0 (i.e., < 1 octave) and linear for errors > 1 octave. Since `freq_mae ≈ 2.0` octaves, the loss is in the linear regime. For a 2-octave error, the gradient is constant at `sign(error) * 1.0`. For a 0.5-octave error, the gradient is `0.5`. **The gradient magnitude is the same whether the error is 2 octaves or 10 octaves**, which means there's no extra pressure to fix large frequency errors vs small ones.

**Fix:** Reduce Huber delta to 0.25 or use pure L1 for the frequency term so the gradient scales with error magnitude more aggressively.

### Bug 5 (MEDIUM): `lambda_freq_anchor` Starts Too Low in Stage 1

**File:** `conf/config.yaml:79`

```yaml
lambda_freq_anchor: 0.5  # Stage 1
```

The `freq_anchor_loss` (line 443-448 in loss_multitype.py) is a direct L1 on log-frequency of matched bands, weighted by gain magnitude. This is the **sharpest** frequency loss because it uses raw L1 (not Huber) and gain-weighting. But at `lambda_freq_anchor=0.5`, it contributes minimally to the total loss compared to `lambda_param=1.0` (which includes Huber frequency loss) and `lambda_hmag=2.0`.

**Fix:** Increase `lambda_freq_anchor` in early curriculum stages to provide stronger frequency-specific gradient signal.

### Bug 6 (LOW): Dataset Only Uses Peaking Filters But Model Has 5-Way Type Head

**File:** `conf/config.yaml:69-70`

```yaml
filter_types:
- peaking
```

All curriculum stages use only peaking filters, so `type_acc` is trivially 100%. But the model still has a 5-way classification head, and the Gumbel-Softmax adds noise to the type predictions. With `type_weights = [1.0, 0, 0, 0, 0]`, only the peaking type has nonzero weight, but the other 4 type logits still participate in `forward_soft()` where all 5 frequency responses are computed and blended by `type_probs`. This adds noise and computation cost without benefit.

This is likely a minor contributor but worth noting.

## Proposed Fixes

### Fix 1: Break Attention Collapse with Entropy Regularization + Sharper Attention

**File:** `differentiable_eq.py` — MultiTypeEQParameterHead.forward()

Replace the soft attention with temperature-scaled attention and add entropy penalty:

```python
# BEFORE (line 720-721):
content = torch.einsum("bnf,bfm->bnm", queries, cnn_feat)
attn = F.softmax(content + self.attn_position_bias.unsqueeze(0), dim=-1)

# AFTER:
content = torch.einsum("bnf,bfm->bnm", queries, cnn_feat)
attn_logits = content + self.attn_position_bias.unsqueeze(0)
attn_logits = attn_logits / 0.1  # Sharpen attention (temperature < 1)
attn = F.softmax(attn_logits, dim=-1)
```

Also add a learnable temperature parameter instead of hardcoding:

```python
# In __init__:
self.register_buffer("attn_temperature", torch.tensor(0.1))

# In forward:
attn_logits = content + self.attn_position_bias.unsqueeze(0)
attn_logits = attn_logits / self.attn_temperature.clamp(min=0.01)
attn = F.softmax(attn_logits, dim=-1)
```

**Why:** Temperature < 1 sharpens the softmax, making attention focus on fewer bins. This makes the weighted mean behave more like an argmax, producing sharper frequency estimates.

### Fix 2: Reduce Position Bias Initial Strength

**File:** `differentiable_eq.py` — MultiTypeEQParameterHead.__init__()

```python
# BEFORE (line 667):
gaussian_init = -1.5 * (mel_pos.unsqueeze(0) - centers.unsqueeze(1)) ** 2

# AFTER:
gaussian_init = -0.5 * (mel_pos.unsqueeze(0) - centers.unsqueeze(1)) ** 2
```

**Why:** With `-0.5`, the peak bin gets weight `exp(0) = 1.0` and a bin 0.5-width away gets `exp(-0.125) ≈ 0.88`. The bias still provides a mild initialization prior but doesn't dominate the content signal. The content term can now override the position prior when the spectral evidence is strong.

### Fix 3: Add Attention Entropy Loss

**File:** `loss_multitype.py` — MultiTypeEQLoss.forward()

Add a new loss component that penalizes high-entropy (diffuse) attention:

```python
# After the spread loss computation (around line 438):
# 9. Attention entropy loss (if model provides attention weights)
# This encourages sharp, peaked attention for better frequency localization.
# (Implementation: pass attention from model output to loss)
```

This requires threading the attention weights from the parameter head through the model output to the loss. Add to `MultiTypeEQParameterHead.forward()`:

```python
# Store attention for entropy regularization
self._last_attn = attn  # (B, N, n_mels)
```

Then in the training loop, extract it and pass to loss.

### Fix 4: Increase Frequency-Specific Loss Weights in Curriculum

**File:** `conf/config.yaml`

```yaml
# Stage 1: easy_multitype
# BEFORE:
lambda_freq_anchor: 0.5

# AFTER:
lambda_freq_anchor: 2.0
```

And in `loss_multitype.py`, reduce Huber delta for frequency:

```python
# BEFORE (line 195-196):
self.huber = nn.HuberLoss(delta=1.0)
self.huber_elementwise = nn.HuberLoss(delta=1.0, reduction="none")

# Use separate deltas for different parameter types
self.huber = nn.HuberLoss(delta=1.0)
self.huber_freq = nn.HuberLoss(delta=0.25, reduction="none")
```

Then use `self.huber_freq` for the frequency elementwise loss (line 235).

**Why:** `delta=0.25` means the Huber loss stays in quadratic regime up to 0.25 octaves of error, providing stronger gradients for the 2-octave errors we're seeing.

### Fix 5: Add Direct Frequency Regression Head (Parallel to Attention)

**File:** `differentiable_eq.py` — MultiTypeEQParameterHead

Add a direct sigmoid regression path alongside the attention path, and blend them:

```python
# In __init__:
self.freq_direct = nn.Linear(hidden_dim, 1)  # Direct regression
self.freq_blend = nn.Parameter(torch.tensor(0.5))  # Learnable blend weight

# In forward, after the attention-based freq computation:
raw_freq_direct = self.freq_direct(trunk_out).squeeze(-1)
prior = self.freq_prior_scale * self.freq_prior_raw
blended_direct = raw_freq_direct + prior.unsqueeze(0)
log_freq_direct = (
    torch.sigmoid(blended_direct) * (self.log_f_max - self.log_f_min)
    + self.log_f_min
)
freq_direct = torch.exp(log_freq_direct)

# Blend attention-based and direct regression:
alpha = torch.sigmoid(self.freq_blend)
freq = alpha * freq_attn + (1 - alpha) * freq_direct
```

**Why:** The direct regression path provides a strong gradient signal that doesn't suffer from the attention collapse issue. During early training, the direct path can learn approximate frequency positions, while the attention path refines them later.

## Implementation Order

1. **Fix 1 + Fix 2** (attention temperature + position bias) — Highest impact, directly addresses the core collapse
2. **Fix 4** (Huber delta + lambda_freq_anchor) — Strengthens the gradient signal for frequency
3. **Fix 5** (direct regression path) — Provides a fallback learning path that bypasses attention
4. **Fix 3** (entropy loss) — Optional, reinforces sharp attention

## Files to Modify

1. `insight/differentiable_eq.py` — MultiTypeEQParameterHead.__init__() and forward()
2. `insight/loss_multitype.py` — PermutationInvariantParamLoss, MultiTypeEQLoss
3. `insight/conf/config.yaml` — lambda_freq_anchor in curriculum stages
4. `insight/train.py` — Thread attention weights from model to loss (if Fix 3 is implemented)

## Verification

1. **Smoke test:** `cd insight && python test_model.py` — verify model still produces valid output shapes
2. **Quick training run:** Train for 5 epochs on Stage 1 and check `freq_mae` is improving epoch-over-epoch (should drop below 1.0 within 5 epochs if fixes work)
3. **Attention visualization:** Log `attn.max(dim=-1).values` (peak attention weight) — should increase from ~0.01 to >0.3 as attention sharpens
4. **Full training:** Run complete curriculum and verify `freq_mae < 0.5` octaves by end of training
