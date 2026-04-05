# Plan: Optimize Blind EQ Parameter Estimation Across the ML Lifecycle

## Context

The IDSP blind parametric EQ estimation system predicts (gain, freq, Q, filter_type) from wet audio without the dry signal. The architecture is a causal TCN encoder + MultiTypeEQParameterHead + DifferentiableBiquadCascade, trained with curriculum learning and Hungarian-matched multi-component loss.

This plan implements the highest-impact optimizations from the ML lifecycle framework, ordered by expected gain. Each change is small, targeted, and independently verifiable.

---

## Step 1: Fix Q Parameterization — Log-Space Mapping

**File:** `insight/differentiable_eq.py` (lines 748-749 in `MultiTypeEQParameterHead.forward`)

**Current (line 749):**
```python
q = torch.sigmoid(gainq[:, :, 1]) * (10.0 - 0.1) + 0.1
```
This gives uniform resolution per unit Q, but Q perception is logarithmic. The loss measures Q error in log-space (log10), creating a mismatch between the parameterization and the loss landscape.

**Change:**
```python
q = torch.exp(torch.sigmoid(gainq[:, :, 1]) * (math.log(10.0) - math.log(0.1)) + math.log(0.1))
```

**Also in `EQParameterHead`** (line 597):
```python
q = torch.exp(torch.sigmoid(gainq[:, :, 1]) * (math.log(10.0) - math.log(0.1)) + math.log(0.1))
```

**Expected impact:** 5-10% improvement in Q MAE (decades), better gradient alignment with log-domain loss.

---

## Step 2: Normalize Hungarian Cost Matrix to Comparable Scales

**File:** `insight/loss_multitype.py` — `HungarianBandMatcher.compute_cost_matrix` (lines 76-91)

**Current:** Raw L1 distances in different units:
- gain: 0–48 dB
- freq (log): 0–~10 octaves (weighted by lambda_freq=1.0)
- Q (log): 0–~2 decades (weighted by lambda_q=0.5)

The gain cost dominates because it has the largest absolute range, making freq/Q matching less influential.

**Change:** Normalize each cost to [0, 1] before applying lambdas:
```python
# After computing cost_gain, cost_freq, cost_q:
max_gain_diff = 48.0   # max possible |dB| difference
max_octave_diff = math.log(20000.0 / 20.0) / math.log(2.0)  # ~6.9
max_decade_diff = math.log10(10.0) - math.log10(0.1)  # 2.0

cost_gain_norm = cost_gain / max_gain_diff
cost_freq_norm = cost_freq / (self.lambda_freq * max_octave_diff)
cost_q_norm = cost_q / (self.lambda_q * max_decade_diff)

cost = cost_gain_norm + cost_freq_norm + cost_q_norm
```

**Expected impact:** 10-20% improvement in matching quality, especially for bands where Q or freq differences matter more than gain.

---

## Step 3: Add Dropout to Parameter Head Trunk

**File:** `insight/differentiable_eq.py` — `MultiTypeEQParameterHead.__init__` (after line 673)

**Add:**
```python
self.dropout = nn.Dropout(p=0.1)
```

**In `forward` (after line 743):**
```python
trunk_out = self.dropout(trunk_out)  # Add before ReLU
```

**Why:** The parameter head is the narrowest bottleneck (embedding_dim → num_bands × 64). It's prone to overfitting on the training distribution. 0.1 is conservative for regression.

**Also add** to `EQParameterHead.__init__` and forward similarly.

**Expected impact:** 3-8% reduction in train-val gap, more robust type classification.

---

## Step 4: Improve Gain Distribution in Dataset

**File:** `insight/dataset.py` — `_beta_gain` method (lines 299-317)

**Current:** 70% beta(2,5) concentrated near 0 dB, 30% uniform. The model rarely sees strong EQ during training.

**Change to a three-part mixture:**
```python
def _beta_gain(self):
    max_gain = min(abs(self.gain_range[0]), abs(self.gain_range[1]))
    for _ in range(100):
        sign = random.choice([-1, 1])
        r = random.random()
        if r < 0.3:
            # Strong EQ — easy to detect
            magnitude = random.uniform(self.min_gain_db, max_gain)
        elif r < 0.6:
            # Moderate EQ — concentrated at 30-70% of range
            magnitude = self.min_gain_db + np.random.beta(2, 2) * (max_gain - self.min_gain_db)
        else:
            # Subtle EQ — tests precision
            magnitude = np.random.beta(2, 5) * max_gain
        if magnitude >= self.min_gain_db:
            return sign * magnitude
    return sign * max_gain
```

**Expected impact:** 10-15% improvement on strong EQ detection, better coverage of the parameter space.

---

## Step 5: Add Per-Type Frequency Weighting to Loss

**File:** `insight/loss_multitype.py` — in the `forward` method's freq_anchor_loss section (around line 480-487)

**Add type-aware weighting:** HP/LP cutoff frequency is THE critical parameter, while peaking frequency has similar importance. Weight accordingly:

```python
# After computing loss_freq_anchor:
if matched_type is not None:
    # HP/LP: cutoff is the primary parameter — weight more
    is_hplp = (matched_type == 3) | (matched_type == 4)  # HIGHPASS or LOWPASS
    type_weight = torch.where(is_hplp, 2.0, 1.0)
    loss_freq_anchor = (gain_weight * type_weight * (log_pred_f - log_match_f).abs()).mean()
```

**Expected impact:** 5-15% improvement in frequency accuracy for HP/LP filters specifically.

---

## Step 6: Add Band Detection F1 to Validation Metrics

**File:** `insight/train.py` — `validate` method (around line 608, in param_maes dict)

**Add:**
```python
param_maes["band_f1"] = [],
```

**After computing matched results (around line 683):**
```python
# Band detection: is the model correct about active vs inactive?
threshold_db = 1.5
pred_active = (pred_gain.cpu().abs() > threshold_db).float()
target_active = (m_gain.abs() > threshold_db).float()
tp = (pred_active * target_active).sum()
fp = (pred_active * (1 - target_active)).sum()
fn = ((1 - pred_active) * target_active).sum()
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
param_maes["band_f1"].append(f1.item())
```

**Update the print statement** (line 728) to include `band_f1`.

**Expected impact:** Diagnostic — no direct model improvement but enables tracking whether the model correctly identifies active bands.

---

## Step 7: Add Intermediate Curriculum Stage for Shelf Types

**File:** `insight/conf/config.yaml` — insert between stages 3 and 4

**Add a `shelf_types_only` stage** to avoid the jump from peaking-only to all-5-types:

```yaml
- name: shelf_types
  epochs: 15
  filter_types:
  - peaking
  - lowshelf
  - highshelf
  param_ranges:
    gain:
    - -20.0
    - 20.0
    q:
    - 0.2
    - 8.0
  min_gain_db: 1.0
  lambda_type: 2.5
  lambda_freq_anchor: 5.0
  gumbel_temperature: 0.15
```

**Why:** Currently the curriculum jumps from peaking-only (stages 1-3) to all 5 types (stage 4). This is a steep jump. Adding an intermediate stage with shelf types (which have gain != 0 like peaking) bridges the gap before introducing HP/LP (which have gain=0).

**Expected impact:** Smoother curriculum progression, 3-8% improvement in final type accuracy.

---

## Step 8: Add Label Smoothing for Type Classification

**File:** `insight/loss_multitype.py` — type classification section (around line 413-420)

**Current:**
```python
loss_type = F.cross_entropy(pred_type_logits.reshape(B * N, C), type_target.reshape(B * N), weight=class_weights)
```

**Change to:**
```python
loss_type = F.cross_entropy(
    pred_type_logits.reshape(B * N, C),
    type_target.reshape(B * N),
    weight=class_weights,
    label_smoothing=0.1,
)
```

PyTorch's `F.cross_entropy` natively supports `label_smoothing` parameter (since 1.10). This prevents the model from becoming overconfident in type predictions.

**Expected impact:** 2-5% improvement in type accuracy, better-calibrated type probabilities.

---

## Verification

After each step, run the existing tests to verify no regressions:

```bash
cd insight
python test_eq.py              # Biquad gradient flow — must pass after Step 1
python test_model.py           # Model forward/inverse — must pass after Steps 1, 3
python test_multitype_eq.py    # Multi-type filters — must pass after Steps 1, 2, 5
python test_streaming.py       # Streaming consistency — must pass after all steps
python train.py --config conf/config.yaml  # Smoke test: train for 2-3 epochs
```

**Specific verification per step:**

1. **Step 1 (Q log-space):** Verify Q predictions still cover [0.1, 10.0] range. Check test_model.py passes.
2. **Step 2 (Hungarian normalization):** Run one epoch of training, verify matching still works and loss decreases.
3. **Step 3 (Dropout):** Verify train-val gap is smaller than before at same epoch count.
4. **Step 4 (Gain distribution):** Plot histogram of generated gains — should show three modes.
5. **Step 5 (Type-aware freq):** Check freq MAE for HP/LP filters specifically improves.
6. **Step 6 (Band F1):** Verify the metric appears in validation logs.
7. **Step 7 (Curriculum stage):** Count total epochs = 150 (add 15, remove 15 from last stage).
8. **Step 8 (Label smoothing):** Verify type_probs are less extreme (not always 0.99+).

**Full integration test:** Train for ~20 epochs across stages 1-2 and verify:
- Loss decreases monotonically
- No NaN/Inf warnings
- Validation metrics are in expected ranges
- Streaming mode still produces consistent output
