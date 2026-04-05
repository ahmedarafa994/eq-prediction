# Fix NaN Values During Training

## Context

Training the IDSP EQ estimator produces NaN losses. Root cause analysis across 6 files identified **unprotected divisions** and **FP16-unsafe epsilons** as the primary culprits. The current config uses `precision: 16-mixed` (FP16 max ~65504), which amplifies numerical instability. With `num_filter_types=1` (peaking-only), the peaking-only code path in `differentiable_eq.py` is hit — and it has the worst protections.

## Root Causes (ordered by severity)

| # | Issue | File:Line | Severity |
|---|-------|-----------|----------|
| 1 | Division by `a0` without clamping (3 methods) | `differentiable_eq.py:49,146,229` | Critical |
| 2 | `freq_response` epsilon `1e-12` underflows in FP16 | `differentiable_eq.py:265` | Critical |
| 3 | Target `H_mag` not clamped (only predicted is clamped) | `train.py:321,350` | High |
| 4 | `STFTLoss` spectral convergence divides by zero | `loss.py:42` | High |
| 5 | Cost matrix log clamps and potential FP16 overflow | `loss_multitype.py:58-63` | Medium |

## Changes

### Fix 1: Clamp `a0` before division in all coefficient methods

**File: `differentiable_eq.py`**

**1a. `compute_biquad_coeffs` (peaking-only) — lines 49-53**
```python
# Replace:
b0 = b0 / a0
b1 = b1 / a0
b2 = b2 / a0
a1 = a1 / a0
a2 = a2 / a0

# With:
a0_safe = a0.clamp(min=1e-6)
b0 = b0 / a0_safe
b1 = b1 / a0_safe
b2 = b2 / a0_safe
a1 = a1 / a0_safe
a2 = a2 / a0_safe
```

**1b. `compute_biquad_coeffs_multitype` (hard) — lines 146-150**
Same pattern — add `a0_safe = a0.clamp(min=1e-6)` before the 5 divisions.

**1c. `compute_biquad_coeffs_multitype_soft` — line 229**
Change `a0_raw.clamp(min=1e-4)` → `a0_raw.clamp(min=1e-6)` for consistency.

### Fix 2: FP16-safe epsilon in `freq_response`

**File: `differentiable_eq.py`, line 265**
```python
# Replace:
H_mag = torch.sqrt(num_mag2 / (den_mag2 + 1e-12))
# With:
H_mag = torch.sqrt(num_mag2 / (den_mag2 + 1e-6))
```

### Fix 3: Clamp target `H_mag` in training loop

**File: `train.py`, lines 321-324 (AMP branch) and 350-353 (non-AMP)**
```python
# After computing target_H_mag, add clamp:
target_H_mag = self.model.dsp_cascade(
    target_gain, target_freq, target_q,
    n_fft=self.n_fft, filter_type=target_ft
).clamp(min=1e-4, max=1e4)
```

### Fix 4: Safe denominators in STFTLoss

**File: `loss.py`**

Line 42 — spectral convergence:
```python
# Replace:
sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
# With:
sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro").clamp(min=1e-6)
```

Line 45 — log magnitude:
```python
# Replace:
log_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
# With:
log_loss = F.l1_loss(torch.log(x_mag.clamp(min=1e-6)), torch.log(y_mag.clamp(min=1e-6)))
```

### Fix 5: FP16-safe cost matrix

**File: `loss_multitype.py`, lines 58-63**
```python
# Lines 58, 61: change clamp(min=1e-4) → clamp(min=1e-6) for log arguments
# After line 63, add:
cost = cost.clamp(max=1e4)
```

## What NOT to Change

- `precision: 16-mixed` in config.yaml — the fixes above make it safe
- Learning rate, loss weights, or model architecture
- Existing safeguards (NaN skip at train.py:334, gradient clipping at train.py:343)
- `1e-8` epsilons in alpha computation — secondary safeguard, not FP16-exposed

## Verification

1. Run `python train.py` for 5 epochs of the `easy_multitype` stage
2. Confirm zero NaN/Inf warnings in training logs
3. Confirm loss decreases steadily (no divergence or flat-lining)
4. Run existing test suite: `python test_eq.py`, `python test_model.py`
