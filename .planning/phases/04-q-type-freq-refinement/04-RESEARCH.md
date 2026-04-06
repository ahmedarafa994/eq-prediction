# Phase 4: Q, Type & Frequency Refinement - Research

**Researched:** 2026-04-06
**Domain:** PyTorch differentiable DSP, classification loss, curriculum learning
**Confidence:** HIGH

## Summary

Phase 4 addresses three interconnected parameter accuracy problems in the IDSP EQ estimator: (1) Q prediction is bottlenecked by a sigmoid-to-exp mapping that compresses gradients at extreme Q values, (2) filter type classification is imbalanced (peaking at 50% vs HP/LP at 10%) and uses standard cross-entropy which cannot focus on hard examples, and (3) Hungarian matching cost weights over-emphasize gain relative to frequency. Additionally, curriculum stage transitions are currently epoch-only and need metric gating to prevent advancing before mastery.

The Q parameterization fix follows the same pattern that successfully improved gain in Phase 2: replace a saturating nonlinear mapping with log-linear output plus STE clamp. The gain MLP (2-layer: Linear->ReLU->Linear) provides the exact template to follow for the Q MLP. Class-balanced focal loss is a well-established technique for imbalanced classification (Cui et al., CVPR 2019) that combines inverse-frequency class weighting with focal loss's hard-example focusing. Metric-gated curriculum requires threading validation metrics into the stage transition logic currently in `_apply_curriculum_stage`.

**Primary recommendation:** Replicate the gain MLP + STE clamp pattern for Q; implement class-balanced focal loss as a drop-in replacement for `nn.CrossEntropyLoss`; equalize Hungarian cost matrix lambda weights; add metric threshold checks to `_apply_curriculum_stage` with hard epoch caps.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Replace single Linear Q head with deep 3-layer MLP (matching gain head pattern: Linear->ReLU->Linear->ReLU->Linear). Output log(Q) directly, apply STE clamp to [log(0.1), log(10)].
- **D-02:** Q MLP hidden dimensions match gain MLP (hidden_dim -> hidden_dim -> hidden_dim -> 1). Same dropout (0.2) as gain head for consistency.
- **D-03:** Target: Q MAE < 0.2 decades (QP-02). Current ~0.49 decades.
- **D-04:** Equalize cost matrix weights to lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0 (currently gain=2.0, freq=0.5, q=0.5).
- **D-05:** Type match cost weight (lambda_type_match) stays at 0.5.
- **D-06:** Each curriculum stage defines a dict of metric thresholds that must ALL be met before advancing.
- **D-07:** Threshold values derived from Phase 1-3 baseline metrics. Thresholds stored in config.yaml.
- **D-08:** Hard epoch cap per stage prevents infinite stall: if metrics aren't met after N epochs, advance anyway. Default cap = 2x the stage's configured epoch count.
- **D-09:** Replace standard cross-entropy with class-balanced focal loss. Focal loss (gamma=2.0). Class weights inverse-proportional to data frequency.
- **D-10:** Class weights computed from type_weights in config [0.5, 0.15, 0.15, 0.1, 0.1] -> inverse normalization.
- **D-11:** Per-type accuracy breakdown reported at each validation step.
- **D-12:** Target: overall type accuracy > 95%, minimum per-type accuracy > 80%.

### Claude's Discretion
- Exact MLP layer initialization (Xavier vs default)
- Focal loss gamma value (start at 2.0, tunable)
- How to compute class weights from type_weights config
- Test structure and file organization

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QP-01 | Switch Q head from sigmoid-to-exp to log-linear parameterization with STE clamp | D-01/D-02: 3-layer MLP pattern from gain head (differentiable_eq.py:562-566); ste_clamp already used for gain (differentiable_eq.py:32-33) |
| QP-02 | Q MAE < 0.2 decades on validation set | Log-linear output gives identity gradients within [log(0.1), log(10)] bounds, removing sigmoid saturation at Q extremes |
| TYPE-01 | Filter type accuracy > 95% on validation set | Class-balanced focal loss (D-09/D-10) addresses 5:1 class imbalance; focal gamma=2.0 focuses on hard examples |
| TYPE-02 | Per-type accuracy breakdown (peaking, lowshelf, highshelf, highpass, lowpass) | D-11: Extend validate() to track per-type accuracy from matched_ft tensor; type_names from FILTER_NAMES |
| FREQ-01 | Frequency MAE < 0.25 octaves on validation set | D-04: Equalizing Hungarian cost weights gives frequency equal importance to gain in band assignment |
| FREQ-02 | Hungarian matching cost matrix balances gain and frequency weight equally | D-04: Change HungarianBandMatcher defaults from (lambda_gain=2.0, lambda_freq=0.5) to (1.0, 1.0) |
| DATA-03 | Metric-gated curriculum transitions | D-06/D-07/D-08: Extend config.yaml stages with metric_thresholds dict; modify _apply_curriculum_stage to check metrics with hard epoch caps |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.8.0+cu128 | Tensor ops, autograd, nn.Module | Core framework (verified) |
| scipy | 1.11.4 | Hungarian matching (linear_sum_assignment) | Already used in HungarianBandMatcher |
| numpy | 1.26.4 | Array ops for cost matrix | Already used in matcher.match() |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyyaml | - | Config loading for curriculum thresholds | config.yaml stage definitions |

### No New Dependencies Required
All changes are implemented in pure PyTorch/scipy/numpy already available. Focal loss is ~15 lines of code using `F.cross_entropy` and `torch.exp`. No external focal loss library needed.

## Architecture Patterns

### Pattern 1: Q MLP with STE Clamp (Replicating Gain Head)
**What:** Replace `nn.Linear(hidden_dim, 1)` + sigmoid-to-exp mapping with 3-layer MLP outputting log(Q) directly, bounded by STE clamp.
**When to use:** When gradient saturation from sigmoid mapping prevents learning at parameter extremes.
**Current code (lines 763-769 in differentiable_eq.py):**
```python
# CURRENT: Single Linear + sigmoid->exp (gradient saturation at extremes)
q_raw = self.q_head(trunk_out).squeeze(-1)  # nn.Linear(hidden_dim, 1)
q = torch.exp(
    torch.sigmoid(q_raw) * (math.log(10.0) - math.log(0.1)) + math.log(0.1)
)
q = torch.nan_to_num(q, nan=1.0, posinf=10.0, neginf=0.1)
q = torch.clamp(q, min=0.1, max=10.0)
```
**Target code pattern (from gain_mlp, lines 562-566, 743-745):**
```python
# Construction (matching gain head 2-layer pattern with hidden_dim):
self.q_mlp = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1),
)

# Forward (log-linear output + STE clamp):
q_log = self.q_mlp(trunk_out).squeeze(-1)
q_log = torch.nan_to_num(q_log, nan=0.0, posinf=math.log(10.0), neginf=math.log(0.1))
q_log = ste_clamp(q_log, math.log(0.1), math.log(10.0))
q = torch.exp(q_log)
```

**Key difference:** D-01 specifies 3-layer MLP (Linear->ReLU->Linear->ReLU->Linear) which is one more hidden layer than the current gain_mlp (2-layer: Linear->ReLU->Linear). The gain head is 2-layer; the Q head decision explicitly states "deep 3-layer MLP" with pattern "hidden_dim -> hidden_dim -> hidden_dim -> 1".

### Pattern 2: Class-Balanced Focal Loss
**What:** Replace `nn.CrossEntropyLoss(label_smoothing=0.05)` with focal loss + inverse-frequency class weights.
**When to use:** Imbalanced classification where minority classes are critical (HP/LP at 10% sampling rate).

```python
# Source: [ASSUMED] - standard focal loss formulation (Lin et al., 2017)
#         + class-balanced weighting (Cui et al., CVPR 2019)
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, type_weights, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Inverse-frequency weights from config type_weights
        # type_weights = [0.5, 0.15, 0.15, 0.1, 0.1] (sampling proportions)
        # Inverse: [1/0.5, 1/0.15, 1/0.15, 1/0.1, 1/0.1] = [2.0, 6.67, 6.67, 10.0, 10.0]
        # Normalize to sum=1 then scale by num_classes=5:
        inv_w = 1.0 / torch.tensor(type_weights, dtype=torch.float32)
        inv_w = inv_w / inv_w.sum() * len(type_weights)
        self.register_buffer('class_weights', inv_w)

    def forward(self, logits, targets):
        # logits: (B*N, 5), targets: (B*N,)
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.class_weights,
            reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()
```

**Why not use a library:** The implementation is ~15 lines. Focal loss is simple enough that adding a dependency provides no benefit and introduces version coupling risk.

### Pattern 3: Metric-Gated Curriculum Transitions
**What:** Extend curriculum stage definitions in config.yaml with metric thresholds; modify `_apply_curriculum_stage` to check validation metrics before advancing.
**When to use:** When epoch-only transitions cause the model to advance to harder stages before mastering easier ones.

```yaml
# config.yaml stage definition (extends current format)
curriculum:
  stages:
    - name: "warmup"
      epochs: 10
      metric_thresholds: {gain_mae: 3.5, type_acc: 0.5}  # ALL must be met
      epoch_cap: 20  # 2x configured epochs (D-08)
      lambda_type: 1.5
      gumbel_temperature: 1.5
    - name: "type_learning"
      epochs: 20
      metric_thresholds: {gain_mae: 2.5, type_acc: 0.7, q_mae: 0.4}
      epoch_cap: 40
      ...
```

```python
# In _apply_curriculum_stage (train.py):
# Current: purely epoch-based (line 913-925)
# Target: check metrics + epoch cap
def _apply_curriculum_stage(self, epoch):
    ...
    if stage_idx == self._current_stage_idx:
        return  # already in this stage

    # Metric gating: check if all thresholds are met
    thresholds = stage.get("metric_thresholds", {})
    epoch_cap = stage.get("epoch_cap", stage["epochs"] * 2)

    if thresholds and self._last_metrics:
        all_met = all(
            self._check_metric(k, v) for k, v in thresholds.items()
        )
        if not all_met and (epoch - stage_start_epoch) < epoch_cap:
            return  # stay in current stage, metrics not met yet

    # Advance to new stage
    self._current_stage_idx = stage_idx
    ...
```

### Anti-Patterns to Avoid
- **Do NOT change `q_raw` interpretation in the loss function:** `loss_multitype.py` already applies `torch.log(pred_q + 1e-8)` when computing Q loss in log-space. With the new log-linear Q head, `pred_q` will already be exp(q_log), so the loss computation stays correct without changes. [VERIFIED: loss_multitype.py lines 264, 484]
- **Do NOT change Hungarian matcher lambda_gain for the loss weighting:** D-04 equalizes the **matching cost** matrix weights (HungarianBandMatcher), NOT the loss function lambda weights (MultiTypeEQLoss.lambda_gain). These are separate: matching cost determines band assignment, loss weights determine gradient scaling. Both need updating but they serve different purposes.
- **Do NOT remove the existing `self.q_head` parameter name:** Old checkpoints will fail to load. Use `strict=False` (already in place at train.py:811) and let missing keys print their warning.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Focal loss modulation | Custom backward for `(1-pt)^gamma` | `F.cross_entropy(reduction='none')` + `torch.exp(-ce_loss)` | Standard PyTorch ops give numerically stable gradients; custom autograd is unnecessary |
| Hungarian matching (already exists) | New matching algorithm | `scipy.optimize.linear_sum_assignment` | Already used, handles edge cases |
| Class weight normalization | Manual weight tensor | `F.cross_entropy(weight=...)` | Handles device placement, dtype, and label smoothing in one call |

## Common Pitfalls

### Pitfall 1: Q output space mismatch with loss function
**What goes wrong:** The loss function computes `torch.log(pred_q + 1e-8)` (loss_multitype.py:264, 484). If Q head outputs log(Q) directly but `pred_q` is exp(q_log), the loss still works correctly because log(exp(q_log)) = q_log. But if someone changes the Q head output without verifying the loss still receives Q values (not log-Q), the loss will compute log(log(Q)).
**Why it happens:** The loss expects Q values (not log-Q) as input.
**How to avoid:** The forward() method must still return `q = torch.exp(q_log)` so the downstream code receives Q values. The STE clamp is applied to q_log, then exp() converts back to Q space. All existing code that consumes Q values (loss, validation metrics, DSP cascade) continues to work unchanged.
**Warning signs:** Q loss becomes NaN or near-zero; Q MAE spikes to >10 decades.

### Pitfall 2: Focal loss NaN with label smoothing + class weights
**What goes wrong:** When `pt` approaches 1.0 (easy examples), `(1-pt)^gamma` approaches 0 and the focal weight can underflow to 0 in bf16. Combined with label smoothing (which reduces pt slightly), this is usually fine, but if all logits are very confident and label_smoothing is 0, focal_weight can become exactly 0, making loss 0 and producing NaN gradients.
**Why it happens:** bf16 has less precision for small values near 0.
**How to avoid:** Keep `label_smoothing=0.05` (already in current code) which prevents pt from reaching 1.0. Optionally clamp focal_weight to min=1e-8.
**Warning signs:** type_loss shows 0.0 for many consecutive steps.

### Pitfall 3: Metric-gated curriculum infinite stall
**What goes wrong:** If metric thresholds are set too aggressively, the model stays in an early stage indefinitely, never advancing to harder curriculum stages where the model actually learns to meet those thresholds.
**Why it happens:** Some metrics can only improve when exposed to harder training data (e.g., type accuracy may need the "type_learning" stage to get above 0.6).
**How to avoid:** D-08 provides a hard epoch cap (2x configured epoch count) as a safety valve. The implementation MUST include this fallback.
**Warning signs:** Training log shows "stage X" for 50+ epochs with no stage change.

### Pitfall 4: Hungarian matcher weight equalization affects loss, not just assignment
**What goes wrong:** Equalizing the Hungarian cost matrix weights (D-04) changes which bands get matched to which targets. This changes the matched gain/freq/q used for loss computation, which can cause a temporary spike in all MAE metrics as the new matching settles.
**Why it happens:** Hungarian matching is a discrete optimization -- changing weights can produce different optimal permutations.
**How to avoid:** Expect a 1-2 epoch transition period after changing weights. Do not interpret the first 2 epochs post-change as regression.
**Warning signs:** All MAE metrics spike by 20-50% immediately after the change, then recover.

### Pitfall 5: Per-type accuracy collection requires matched targets
**What goes wrong:** Computing per-type accuracy from `output["filter_type"] == target_ft` (unmatched) gives wrong results because band ordering differs between prediction and target.
**Why it happens:** Hungarian matching reorders targets to align with predictions.
**How to avoid:** Use `matched_ft` from the validation loop's Hungarian matching (already computed at train.py:720-729). Extend the existing matched_ft comparison to accumulate per-type counts.
**Warning signs:** Per-type accuracy for low-count classes (HP/LP) shows very low numbers even when overall accuracy seems reasonable.

## Code Examples

### Q MLP Construction and Initialization
```python
# Source: [VERIFIED: differentiable_eq.py lines 562-566 (gain_mlp pattern)]
# D-01: 3-layer MLP for Q (hidden_dim -> hidden_dim -> hidden_dim -> 1)

# In MultiTypeEQParameterHead.__init__():
self.q_mlp = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),  # 64 -> 64
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),  # 64 -> 64
    nn.ReLU(),
    nn.Linear(hidden_dim, 1),           # 64 -> 1
)

# Initialization (matching gain_mlp pattern at lines 657-660):
nn.init.xavier_uniform_(self.q_mlp[0].weight)
nn.init.zeros_(self.q_mlp[0].bias)
nn.init.xavier_uniform_(self.q_mlp[2].weight)
nn.init.zeros_(self.q_mlp[2].bias)
nn.init.xavier_uniform_(self.q_mlp[4].weight)  # Third Linear layer
nn.init.zeros_(self.q_mlp[4].bias)
```

### Q Forward (Replacing lines 763-769)
```python
# Source: [VERIFIED: differentiable_eq.py lines 743-745 (gain forward pattern)]
# D-01: Log-linear Q with STE clamp

# REPLACE:
#   q_raw = self.q_head(trunk_out).squeeze(-1)
#   q = torch.exp(torch.sigmoid(q_raw) * (...) + math.log(0.1))
# WITH:
q_log = self.q_mlp(trunk_out).squeeze(-1)
q_log = torch.nan_to_num(q_log, nan=0.0, posinf=math.log(10.0), neginf=math.log(0.1))
q_log = ste_clamp(q_log, math.log(0.1), math.log(10.0))
q = torch.exp(q_log)
```

### Class-Balanced Focal Loss (Replacement for self.type_loss)
```python
# Source: [ASSUMED] - Focal loss (Lin et al., ICCV 2017)
#         Class-balanced weighting (Cui et al., CVPR 2019)
# D-09/D-10: Drop-in replacement for nn.CrossEntropyLoss

# In MultiTypeEQLoss.__init__():
# REPLACE: self.type_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
# WITH:
type_weights_cfg = [0.5, 0.15, 0.15, 0.1, 0.1]  # from config data.type_weights
inv_w = 1.0 / torch.tensor(type_weights_cfg, dtype=torch.float32)
inv_w = inv_w / inv_w.sum() * len(type_weights_cfg)
self.register_buffer('type_class_weights', inv_w)
self.focal_gamma = 2.0  # D-09
self.type_label_smoothing = 0.05

# In forward(), type loss computation section:
# REPLACE:
#   loss_type = self.type_loss(pred_type_logits.reshape(B*N, C), matched_filter_type.reshape(B*N))
# WITH:
type_logits_flat = pred_type_logits.reshape(B * N, C)
type_targets_flat = matched_filter_type.reshape(B * N)
ce_loss = F.cross_entropy(
    type_logits_flat, type_targets_flat,
    weight=self.type_class_weights, reduction='none',
    label_smoothing=self.type_label_smoothing,
)
pt = torch.exp(-ce_loss)
focal_weight = (1.0 - pt) ** self.focal_gamma
loss_type = (focal_weight * ce_loss).mean()
```

### Per-Type Accuracy in Validation
```python
# Source: [VERIFIED: train.py lines 720-741 (existing validation metrics)]
# D-11: Extend param_maes dict and accumulation

# In validate(), add to param_maes dict:
param_maes = {
    "gain": [], "gain_raw": [], "freq": [], "q": [], "type_acc": [],
    "type_peaking": [], "type_lowshelf": [], "type_highshelf": [],
    "type_highpass": [], "type_lowpass": [],
}

# After computing type_acc (line 740-741):
from differentiable_eq import FILTER_NAMES
for type_idx, type_name in enumerate(FILTER_NAMES):
    mask = (matched_ft == type_idx)
    if mask.sum() > 0:
        per_type_acc = (output["filter_type"][mask] == matched_ft[mask]).float().mean().item()
        param_maes[f"type_{type_name}"].append(per_type_acc)

# In logging section (after line 763):
per_type_str = " | ".join(
    f"{name}={sum(metrics.get(f'type_{name}', [0])) / max(len(metrics.get(f'type_{name}', [0])), 1):.1%}"
    for name in FILTER_NAMES
)
print(f"  [val] per-type: {per_type_str}")
```

### Hungarian Matcher Weight Equalization
```python
# Source: [VERIFIED: loss_multitype.py lines 67-77 (HungarianBandMatcher defaults)]
# D-04: Equalize cost matrix weights

# In train.py __init__(), line 231-233:
# REPLACE:
#   self.matcher = HungarianBandMatcher(
#       lambda_freq=1.0, lambda_q=0.5, lambda_type_match=0.5)
# WITH:
self.matcher = HungarianBandMatcher(
    lambda_gain=1.0,   # was 2.0 (default)
    lambda_freq=1.0,   # was 0.5 (default)
    lambda_q=1.0,      # was 0.5 (default)
    lambda_type_match=0.5,  # D-05: unchanged
)

# Also update the matcher inside MultiTypeEQLoss (loss_multitype.py:333-335):
# The inner matcher is created with default lambda_gain=2.0.
# Pass equalized weights to PermutationInvariantParamLoss:
self.param_loss = PermutationInvariantParamLoss(
    lambda_freq=1.0,      # was 1.0 (ok)
    lambda_q=1.0,         # was 0.5 (needs change)
    lambda_type_match=0.5,
)
# And HungarianBandMatcher.__init__ defaults need to be updated or overridden:
# In PermutationInvariantParamLoss.__init__():
self.matcher = HungarianBandMatcher(
    lambda_freq=lambda_freq,
    lambda_q=lambda_q,
    lambda_gain=1.0,        # equalized (was default 2.0)
    lambda_type_match=lambda_type_match,
)
```

### Metric-Gated Curriculum in train.py
```python
# Source: [VERIFIED: train.py lines 905-977 (_apply_curriculum_stage)]
# D-06/D-07/D-08: Metric gating with epoch cap

# Add to Trainer.__init__():
self._last_metrics = {}  # Updated after each validate() call

# In fit(), after validate() (line 1026):
self._last_metrics = metrics  # Store for curriculum gating

# Modified _apply_curriculum_stage():
def _apply_curriculum_stage(self, epoch):
    self._update_type_transition(epoch)
    if not self.curriculum_stages:
        return

    # Compute cumulative epoch boundaries
    cumulative = 0
    stage_idx = 0
    stage_start_epoch = 0
    for i, stage in enumerate(self.curriculum_stages):
        prev_cumulative = cumulative
        cumulative += stage["epochs"]
        if epoch <= cumulative:
            stage_idx = i
            stage_start_epoch = prev_cumulative + 1
            break
    else:
        stage_idx = len(self.curriculum_stages) - 1

    stage = self.curriculum_stages[stage_idx]
    stage_epochs = stage["epochs"]

    # Intra-stage Gumbel temperature annealing (unchanged)
    target_tau = stage.get("gumbel_temperature", 1.0)
    epoch_in_stage = epoch - stage_start_epoch
    if stage_idx > 0:
        prev_tau = self.curriculum_stages[stage_idx - 1].get("gumbel_temperature", 1.0)
    else:
        prev_tau = 1.0
    progress = epoch_in_stage / max(stage_epochs - 1, 1)
    current_tau = prev_tau + (target_tau - prev_tau) * progress
    current_tau = max(current_tau, 0.1)
    self.model.param_head.gumbel_temperature.fill_(current_tau)

    if stage_idx == self._current_stage_idx:
        return  # Already in this stage

    # D-06/D-07: Check metric thresholds before advancing
    thresholds = stage.get("metric_thresholds", {})
    if thresholds and self._last_metrics:
        all_met = True
        for metric_name, threshold in thresholds.items():
            current_val = self._last_metrics.get(metric_name)
            if current_val is None:
                all_met = False
                break
            # For "mae" metrics, lower is better; for "acc", higher is better
            if "mae" in metric_name:
                if current_val > threshold:
                    all_met = False
                    break
            elif "acc" in metric_name:
                if current_val < threshold:
                    all_met = False
                    break

        # D-08: Hard epoch cap prevents infinite stall
        epoch_cap = stage.get("epoch_cap", stage["epochs"] * 2)
        epochs_in_current = epoch - stage_start_epoch

        if not all_met and epochs_in_current < epoch_cap:
            print(f"  [curriculum] Stage transition to '{stage['name']}' blocked: "
                  f"metrics not met (staying {epochs_in_current}/{epoch_cap} epochs)")
            return

    # Advance to new stage (existing code continues...)
    self._current_stage_idx = stage_idx
    # ... rest unchanged
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sigmoid-to-exp Q mapping | Log-linear + STE clamp | Phase 2 (gain), Phase 4 (Q) | Identity gradients within bounds, no saturation |
| Standard cross-entropy | Class-balanced focal loss | Phase 4 | Handles 5:1 class imbalance, focuses on hard examples |
| Epoch-only curriculum | Metric-gated + epoch cap | Phase 4 | Prevents premature stage advancement |
| Unequal Hungarian matching (gain=2x freq) | Equal cost weights | Phase 4 | Fair band assignment between gain and frequency |

**Deprecated/outdated:**
- `self.q_head = nn.Linear(hidden_dim, 1)` — Single linear layer with sigmoid-to-exp causes gradient saturation. Replaced by 3-layer MLP with STE clamp. [VERIFIED: differentiable_eq.py:570]
- `nn.CrossEntropyLoss(label_smoothing=0.05)` for type classification — No class weighting, no hard-example focus. Replaced by class-balanced focal loss. [VERIFIED: loss_multitype.py:336]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Focal loss gamma=2.0 is a good starting point for 5-class imbalanced EQ type classification | Pattern 2 | May over-suppress easy examples; gamma=1.0 could be safer initial value |
| A2 | Inverse-frequency class weighting from type_weights is sufficient (no effective number of samples / beta parameter needed) | Pattern 2 | For small datasets, effective-number weighting (Cui et al.) might be better; for 200k samples with ~5:1 imbalance, simple inverse frequency is standard |
| A3 | Metric threshold values in config.yaml will be computed empirically from Phase 1-3 baselines | Pattern 3 | Thresholds set too aggressively cause curriculum stalls; too loose causes premature advancement |
| A4 | The gain_mlp 2-layer pattern should be extended to 3-layer for Q (D-01 explicitly states "deep 3-layer MLP") | Pattern 1 | Extra layer adds parameters but D-01 is a locked decision |

**If this table is empty:** All claims in this research were verified or cited -- no user confirmation needed.

## Open Questions

1. **Metric threshold values for curriculum stages**
   - What we know: D-07 says thresholds derived from Phase 1-3 baselines; D-08 provides epoch cap fallback
   - What's unclear: Exact threshold values for each stage. These must be computed from validation logs after Phase 3 training completes.
   - Recommendation: Start with conservative thresholds (easy to meet) and tune up based on observed metrics. The epoch cap (2x epochs) provides a safety net.

2. **Old checkpoint compatibility**
   - What we know: `q_head` parameter name changes to `q_mlp` (nn.Sequential). train.py:811 already uses `strict=False` loading.
   - What's unclear: Whether any Phase 3 checkpoints need to be used for Phase 4 warm start, or if training will be from scratch.
   - Recommendation: Use `strict=False` and let missing keys initialize randomly (same pattern as Phase 2/3 architecture changes).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | All operations | Yes | 2.8.0+cu128 | - |
| scipy | Hungarian matching | Yes | 1.11.4 | - |
| numpy | Cost matrix ops | Yes | 1.26.4 | - |
| CUDA GPU | Training | Yes | Available | - |
| pyyaml | Config loading | Yes | - | - |

**Missing dependencies with no fallback:** None

**Missing dependencies with fallback:** None

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Standalone Python scripts (no pytest) |
| Config file | None -- each test file is self-contained |
| Quick run command | `cd insight && python test_q_type_freq.py` |
| Full suite command | `cd insight && python test_q_type_freq.py && python test_eq.py && python test_model.py && python test_loss_architecture.py` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| QP-01 | Q MLP outputs log-linear, STE clamp bounds correct | unit | `python test_q_type_freq.py::test_q_log_linear_output` | Wave 0 |
| QP-01 | Q MLP gradient flows through STE clamp (no saturation) | unit | `python test_q_type_freq.py::test_q_gradient_flow` | Wave 0 |
| QP-02 | Q MAE < 0.2 decades | integration | `python test_q_type_freq.py::test_q_mae_synthetic` | Wave 0 |
| TYPE-01 | Type accuracy > 95% with focal loss | integration | `python test_q_type_freq.py::test_focal_loss_improves_type` | Wave 0 |
| TYPE-02 | Per-type accuracy breakdown reported | unit | `python test_q_type_freq.py::test_per_type_accuracy` | Wave 0 |
| FREQ-01 | Frequency MAE < 0.25 octaves | integration | `python test_q_type_freq.py::test_freq_mae_equalized_matching` | Wave 0 |
| FREQ-02 | Hungarian cost matrix has equal gain/freq weights | unit | `python test_q_type_freq.py::test_equalized_cost_matrix` | Wave 0 |
| DATA-03 | Metric-gated curriculum blocks premature advance | unit | `python test_q_type_freq.py::test_metric_gated_curriculum` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cd insight && python test_q_type_freq.py`
- **Per wave merge:** Full suite (all test files)
- **Phase gate:** Full suite green + training run achieving target metrics

### Wave 0 Gaps
- [ ] `insight/test_q_type_freq.py` -- covers QP-01, QP-02, TYPE-01, TYPE-02, FREQ-01, FREQ-02, DATA-03

## Security Domain

No security concerns for this phase. Changes are internal to the ML training pipeline -- no user input handling, no network communication, no data persistence beyond model checkpoints.

## Sources

### Primary (HIGH confidence)
- [VERIFIED: insight/differentiable_eq.py] -- Q head implementation (lines 570, 763-769), gain MLP pattern (lines 562-566, 743-745), STE clamp (lines 7-33), classification head (lines 573-597)
- [VERIFIED: insight/loss_multitype.py] -- HungarianBandMatcher (lines 58-217), MultiTypeEQLoss (lines 270-582), type loss cross-entropy (line 336)
- [VERIFIED: insight/train.py] -- Trainer class, curriculum stages (lines 905-977), validation metrics (lines 615-765), metric accumulation
- [VERIFIED: insight/conf/config.yaml] -- Current loss weights, curriculum stage definitions, type_weights

### Secondary (MEDIUM confidence)
- Focal loss formulation from Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017) -- standard technique, well-established [ASSUMED]
- Class-balanced loss formulation from Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019) -- standard technique [ASSUMED]

### Tertiary (LOW confidence)
- Exact metric threshold values for curriculum stages -- must be determined empirically post Phase 3 [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies; all changes in existing PyTorch/scipy
- Architecture: HIGH - exact code locations and patterns verified in codebase
- Pitfalls: HIGH - derived from code analysis of existing gradient paths and loss computations
- Focal loss implementation: MEDIUM - standard technique but not verified against PyTorch 2.8 specific behavior

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable - no fast-moving dependencies)
