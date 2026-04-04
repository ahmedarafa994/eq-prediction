# Fix: NaN in Hungarian Cost Matrix — Root Cause Diagnosis & Fix

## Context

Training fails because the Hungarian cost matrix produces NaN in every batch. The root cause is a **missing model instantiation** in `train.py` combined with insufficient NaN guards in the cost matrix and forward pass. Once the model is properly created, there are still latent NaN risks from Gumbel-Softmax edge cases and unguarded log operations.

## Root Cause Analysis

### Bug #1 (Showstopper): Missing model creation in `train.py`

`self.model` is referenced 15+ times (lines 115, 164, 165, 216, 241, 419–421, 474, 496, 499, 503, 544, 549, 589, 598, 638, 640, 645, 750, 784, 831, 836) but **never assigned**. The variables `enc_cfg` and `num_filter_types` (lines 80–86) are computed but unused. The model creation block is completely absent between lines 86 and 88.

The correct pattern exists in `training/lightning_module.py:68–80`.

### Bug #2 (NaN propagation): Cost matrix doesn't guard against NaN inputs

In `loss_multitype.py:71–74`:
```python
cost_freq = self.lambda_freq * (
    torch.log(pf.clamp(min=1e-5)) - torch.log(tf.clamp(min=1e-5))
).abs()
```

`torch.clamp` **passes NaN through unchanged**. If `pred_freq` is NaN (from upstream), the cost matrix becomes NaN. The same applies to `cost_q` and `cost_gain`. There is no NaN check on model outputs before cost computation.

### Bug #3 (Latent NaN): Gumbel-Softmax edge case

In `differentiable_eq.py:780–782`:
```python
type_probs = F.gumbel_softmax(type_logits, tau=self.gumbel_temperature, hard=False)
```

`F.gumbel_softmax` uses `-log(exponential_())` for Gumbel noise. When `exponential_()` returns a value near 0, `log(0)` = `-inf`, making `gumbels = inf`. Then `softmax(logits + inf)` produces `inf/inf = NaN` when multiple elements become inf simultaneously. With batch_size=4096, num_bands=3, num_filter_types=5, there are 61,440 Gumbel samples per batch — the tail event is inevitable at scale.

### Bug #4 (Latent NaN): `forward_soft` log-domain product can amplify NaN

In `differentiable_eq.py:433–434`:
```python
H_mag_total = torch.exp(torch.sum(torch.log(H_mag_blended_safe), dim=1))
```

If ANY band's blended magnitude is NaN (from NaN type_probs), the sum is NaN, and `exp(NaN)` = NaN. This NaN then enters `H_mag`, which enters the loss, which produces NaN gradients.

## Implementation Plan

### Step 1: Add missing model creation in `train.py`

**File**: `insight/train.py`, insert between lines 86 and 88.

Add the model instantiation using the same pattern as `training/lightning_module.py:64–80`:

```python
use_full_spectrum = model_cfg.get("use_full_spectrum", False)
input_bins = (
    (self.n_fft // 2 + 1) if use_full_spectrum else data_cfg.get("n_mels", 128)
)
self.model = StreamingTCNModel(
    n_mels=input_bins,
    embedding_dim=enc_cfg.get("embedding_dim", 128),
    num_bands=self.num_bands,
    channels=enc_cfg.get("channels", 128),
    num_blocks=enc_cfg.get("num_blocks", 4),
    num_stacks=enc_cfg.get("num_stacks", 2),
    sample_rate=self.sample_rate,
    n_fft=self.n_fft,
    num_filter_types=num_filter_types,
    kernel_size=enc_cfg.get("kernel_size", 3),
    use_full_spectrum=use_full_spectrum,
).to(self.device)
```

### Step 2: Add NaN guards in cost matrix computation

**File**: `insight/loss_multitype.py`, method `compute_cost_matrix` (lines 39–104).

After the pairwise computation and before the type matching, add input validation:

```python
# Guard against NaN in model outputs — clamp doesn't catch NaN
for name, tensor in [("pred_freq", pred_freq), ("pred_q", pred_q),
                      ("target_freq", target_freq), ("target_q", target_q)]:
    if not torch.isfinite(tensor).all():
        tensor = torch.nan_to_num(tensor, nan=1.0)  # Replace with safe default
```

Also add a NaN-safe clamp after the total cost:

```python
cost = cost.clamp(max=1e4)
cost = torch.nan_to_num(cost, nan=1e4)  # Replace any residual NaN
```

### Step 3: Fix Gumbel-Softmax NaN edge case

**File**: `insight/differentiable_eq.py`, method `MultiTypeEQParameterHead.forward` (lines 779–782).

Replace the bare `F.gumbel_softmax` with a clamped version:

```python
if self.training and not hard_types:
    # Clamp Gumbel noise to prevent inf from log(exp≈0)
    gumbels = -torch.empty_like(type_logits).exponential_().log()
    gumbels = gumbels.clamp(-10.0, 10.0)  # Prevent inf/-inf
    type_probs = ((type_logits + gumbels) / self.gumbel_temperature).softmax(dim=-1)
else:
    type_probs = F.softmax(type_logits, dim=-1)
```

### Step 4: Add NaN guard in `forward_soft`

**File**: `insight/differentiable_eq.py`, method `forward_soft` (lines 394–436).

After blending magnitude responses, add a NaN check:

```python
H_mag_blended = (all_H_mag * type_weights).sum(dim=2)
H_mag_blended = torch.nan_to_num(H_mag_blended, nan=1.0)  # Guard against NaN from type_probs
H_mag_blended_safe = H_mag_blended.clamp(min=1e-6)
```

### Step 5: Add diagnostic NaN checks at key boundaries

**File**: `insight/train.py`, in `train_one_epoch` (around line 504).

Add NaN detection BEFORE the loss computation to catch upstream issues:

```python
# Diagnostic: check model outputs before loss
pred_gain, pred_freq, pred_q = output["params"]
if not torch.isfinite(pred_freq).all():
    print(f"  WARNING: NaN in pred_freq at step {self.global_step}")
if not torch.isfinite(output["H_mag"]).all():
    print(f"  WARNING: NaN in H_mag at step {self.global_step}")
```

## Files to Modify

| File | Changes |
|------|---------|
| `insight/train.py` | Add model creation (Step 1), add NaN diagnostics (Step 5) |
| `insight/loss_multitype.py` | Add NaN guards in `compute_cost_matrix` (Step 2) |
| `insight/differentiable_eq.py` | Fix Gumbel-Softmax (Step 3), guard `forward_soft` (Step 4) |

## Verification

1. **Smoke test**: Run `python -c "from train import Trainer; t = Trainer(); print('Model created:', type(t.model))"` — should not crash
2. **NaN test**: Run 2 epochs of training with `torch_compile: false` and verify no NaN warnings
3. **Gradient flow**: Run `python test_streaming.py` to confirm gradients remain finite
4. **Full training**: Run `python train.py` for 1 full curriculum stage and verify stable loss descent
