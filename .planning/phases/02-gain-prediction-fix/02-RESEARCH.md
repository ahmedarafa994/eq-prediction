# Phase 2: Gain Prediction Fix - Research

**Researched:** 2026-04-06
**Domain:** Differentiable DSP gain prediction architecture in PyTorch
**Confidence:** HIGH

## Summary

Phase 2 targets the gain prediction mechanism in `MultiTypeEQParameterHead` (in `differentiable_eq.py`). The current implementation has a dual-path gain architecture: a primary MLP (`gain_mlp`) with Tanh activation, and an auxiliary mel-residual path (`gain_mel_aux`) that blends via a learned sigmoid gate. The auxiliary path injects noise from the mel-residual signal rather than providing useful gain information, degrading gain MAE from what the primary path alone would achieve.

The baseline gain MAE (matched) is **5.60 dB** (pre-fix, from Phase 1 baseline_metrics.md). The target is **< 3 dB**. Three interventions are needed: (1) remove the mel-residual auxiliary gain path entirely, (2) replace Tanh in `gain_mlp` with STE clamp for unattenuated gradient flow, and (3) verify all changes preserve streaming inference consistency.

**Primary recommendation:** Remove the mel-residual aux path, replace Tanh with STE clamp in `gain_mlp`, use `gain_trunk_head` pattern (already STE-clamped) as the reference implementation, and run streaming consistency tests before and after changes.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Full removal from BOTH EQParameterHead (line 509) and MultiTypeEQParameterHead (line 531). Delete all mel-residual gain readout code, attention weights, blending logic. Gain comes purely from the trunk embedding MLP.
- **D-02:** After removal, remove the related config keys (if any) for mel-residual blend weight.
- **D-03:** STE clamp is already in use for gain in both heads -- verify it's the ONLY activation (no tanh remnants). The requirement to "replace tanh with STE clamp" appears already satisfied but needs verification across all codepaths.
- **D-04:** Gain range stays +/-24 dB with ste_clamp bounds (current setting).
- **D-05:** All changes must preserve streaming inference. Mel-residual removal should not affect init_streaming() or process_frame() -- those operate on the TCN encoder output, not the gain head internals.
- **D-06:** Run baseline validation before changes (from Phase 1 metrics) and after changes to measure gain MAE improvement.
- **D-07:** Target: gain MAE < 3 dB with matched metrics (down from ~6 dB baseline).

### Claude's Discretion
- Exact MLP architecture for the primary gain path (layer sizes, activation)
- Test structure and file organization
- How to handle any config keys that become dead after mel-residual removal

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GAIN-01 | Replace Gaussian readout with direct MLP regression head from trunk embedding | `gain_mlp` already exists as a 2-layer MLP (64->64->1) in MultiTypeEQParameterHead; after mel-residual removal it becomes the sole gain path |
| GAIN-02 | Use STE clamp for gain activation instead of tanh | `gain_mlp` currently ends with `nn.Tanh()` (line 567); must be removed and replaced with STE clamp on the scaled output. `gain_trunk_head` already uses STE clamp correctly (line 797). |
| GAIN-03 | Remove mel-residual auxiliary gain path | 6 components to remove: `gain_smooth_pool`, `gain_mel_aux` MLP, `gain_aux_scale`, `gain_blend_gate`, blending logic (lines 767-782), and `use_mel_for_gain` flag. Also remove from `train.py` gradient monitoring (line 517). |
| GAIN-04 | Gain MAE < 1 dB on validation (phase target relaxed to < 3 dB per D-07) | Baseline is 5.60 dB matched. Removing noise injection and fixing gradients should provide significant improvement. |
| STRM-01 | All model changes preserve streaming inference | `process_frame()` in `model_tcn.py` calls `self.param_head(embedding, mel_profile=mel_profile, hard_types=True)`. After mel-residual removal, mel_profile is still passed but gain path ignores it. Streaming path does not use mel-residual internally. |
| STRM-02 | Streaming vs batch consistency verified (< 0.1 dB gain difference) | Existing `test_streaming.py` tests batch-vs-streaming consistency but checks embedding difference, not gain difference. Need new test comparing gain_db outputs. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.8.0+cu128 | Deep learning framework | Project dependency, already installed [VERIFIED: runtime check] |
| scipy | 1.11.4 | Hungarian matching (linear_sum_assignment) | Already used for permutation-invariant metrics [VERIFIED: runtime check] |
| pyyaml | installed | Config file loading | Already used for conf/config.yaml [VERIFIED: runtime check] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | installed | Array ops in Hungarian matching | Used inside HungarianBandMatcher |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| gain_mlp (2-layer MLP) | gain_trunk_head (single linear layer) | gain_trunk_head is simpler but may have less capacity; gain_mlp with STE clamp is the better choice per D-03 |

## Architecture Patterns

### Current Gain Architecture (BEFORE Phase 2)

The `MultiTypeEQParameterHead.__init__()` constructs these gain-related components:

1. `self.gain_mlp` -- Sequential(Linear(64,64), ReLU, Linear(64,1), **Tanh**) -- primary path
2. `self.gain_output_scale` -- Parameter(tensor(24.0)) -- scales Tanh output from [-1,1] to [-24,24]
3. `self.use_mel_for_gain` -- boolean flag (True when n_mels > 0)
4. `self.gain_smooth_pool` -- AvgPool1d for smoothing mel profile
5. `self.gain_mel_aux` -- Sequential(Linear(65,64), ReLU, Linear(64,1), Tanh) -- auxiliary path
6. `self.gain_aux_scale` -- Parameter(tensor(24.0)) -- scales aux output
7. `self.gain_blend_gate` -- Parameter(tensor(0.8473)) -- learned blend weight
8. `self.gain_trunk_head` -- Linear(64,1) -- fallback gain when no mel_profile

The `forward()` method (lines 697-831):
- When `use_mel_profile and mel_profile is not None`:
  - Computes mel_residual = mel_profile - smoothed(mel_profile)
  - Computes gain_readout via attention-weighted mel_residual
  - gain_mlp output (primary) blended with gain_mel_aux output (auxiliary)
  - Final ste_clamp on blended result
- When mel_profile is not available (else branch):
  - gain_trunk_head(trunk_out) -> ste_clamp (clean path, no Tanh)

### Target Gain Architecture (AFTER Phase 2)

1. `self.gain_mlp` -- Sequential(Linear(64,64), ReLU, Linear(64,1)) -- **no Tanh**
2. `self.gain_output_scale` -- Parameter(tensor(24.0)) -- scales raw output
3. **Removed:** use_mel_for_gain, gain_smooth_pool, gain_mel_aux, gain_aux_scale, gain_blend_gate
4. **Removed:** gain_trunk_head (dead code after unification)

The `forward()` simplified gain computation:
```python
gain_db = self.gain_mlp(trunk_out).squeeze(-1) * self.gain_output_scale
gain_db = ste_clamp(gain_db, -24.0, 24.0)
```

### Key Insight: Tanh in gain_mlp

The CONTEXT.md says "STE clamp is already in use for gain in both heads." This is **partially true**: the final `ste_clamp` call is present (line 784). However, `gain_mlp` itself contains `nn.Tanh()` as its last layer (line 567), which means the raw output is compressed to [-1,1] before scaling by 24.0. The gradients through Tanh are `1 - tanh(x)^2`, which attenuate for values near the bounds. The STE clamp afterward provides hard bounds but the Tanh gradient attenuation has already occurred upstream. Both Tanh removal and STE clamp are needed.

### Project Structure
```
insight/
  differentiable_eq.py    -- PRIMARY: MultiTypeEQParameterHead class (mel-residual removal + Tanh fix)
  model_tcn.py            -- SECONDARY: StreamingTCNModel passes mel_profile to param_head
  train.py                -- TERTIARY: gradient monitoring references gain_mel_aux (line 517)
  test_streaming.py       -- EXISTING: streaming consistency tests (need gain-specific additions)
  test_gain_fix.py        -- NEW: Wave 0 test for gain fix verification
  conf/config.yaml        -- NO CHANGES: no mel-residual config keys exist
```

### Anti-Patterns to Avoid

1. **Partial removal** -- Leaving `use_mel_for_gain` flag and dead code branches. Must fully remove the conditional path so the gain path is always a clean MLP to STE clamp.

2. **Breaking streaming** -- `process_frame()` calls `self.param_head(embedding, mel_profile=mel_profile, hard_types=True)`. After removal, mel_profile is still passed (for frequency and type heads) but the gain path ignores it.

3. **Forgetting gradient monitoring** -- `train.py` line 517 references `gain_mel_aux` in gradient monitoring. Must be removed after the parameter head cleanup.

4. **Checkpoint incompatibility** -- Old checkpoints contain keys for removed parameters. Need `strict=False` in `load_state_dict()` or documentation that retraining is required.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gradient through clamping | Custom backward pass | ste_clamp (line 32) | Already implemented with fp32 safety for bf16 AMP |
| Gain bounds | Custom scaling logic | ste_clamp(-24, 24) | Already used in gain_trunk_head fallback path |
| MLP gain head | New architecture | Modify existing gain_mlp | Already has correct structure except Tanh |

**Key insight:** All the building blocks already exist in the codebase. This phase is primarily a removal and simplification operation, not a construction operation.

## Common Pitfalls

### Pitfall 1: Tanh Gradient Attenuation
**What goes wrong:** The gain_mlp outputs through Tanh, which squashes gradients near the bounds. When the model needs to predict gains near +/-24 dB, the Tanh derivative approaches zero, making learning extremely slow or impossible for extreme gains.
**Why it happens:** Tanh(x) derivative = 1 - tanh(x)^2. For x = 2.0 (which would give tanh = 0.96, mapping to ~23 dB), the gradient is only 0.07 -- a 93% attenuation.
**How to avoid:** Remove Tanh from gain_mlp. Let the raw linear output be scaled by gain_output_scale and bounded by ste_clamp (identity gradient).
**Warning signs:** Gain predictions clustering near zero; gain MAE not improving during training; gradient norm for gain parameters near zero.

### Pitfall 2: Dead Code After Conditional Removal
**What goes wrong:** After removing the mel-residual gain path, the `else` branch (using `gain_trunk_head`) becomes unreachable if mel_profile is always provided.
**Why it happens:** The forward method has an `if self.use_mel_for_gain` check that gates the entire mel-residual path. After removal, there's no reason to keep `gain_trunk_head` and its branch.
**How to avoid:** Unify to a single gain path using `gain_mlp` (with Tanh removed). Remove `gain_trunk_head` entirely.
**Warning signs:** Unreachable code; test coverage gaps on the else branch.

### Pitfall 3: mel_profile Still Needed for Other Heads
**What goes wrong:** Removing mel_profile from the param head entirely would break frequency prediction (attention CNN) and type classification (mel features).
**Why it happens:** mel_profile is used by three subsystems: gain (being removed), frequency attention (staying), and type classification (staying).
**How to avoid:** Only remove gain-specific mel processing. Keep mel_profile as a parameter to forward() for the other heads.
**Warning signs:** Type accuracy or frequency MAE regressing after changes.

### Pitfall 4: Streaming Consistency After Architecture Change
**What goes wrong:** The model architecture change could subtly alter how process_frame() computes gain, causing batch vs streaming divergence.
**Why it happens:** process_frame() independently computes mel_profile from the streaming buffer and passes it to param_head. If the gain path change introduces any non-determinism, streaming results diverge.
**How to avoid:** Test streaming vs batch gain difference explicitly, not just embedding difference. The existing test checks embedding consistency but not gain-specific consistency.
**Warning signs:** Gain predictions differ between batch and streaming mode by more than 0.1 dB.

## Code Examples

### Existing STE clamp implementation (already in codebase)
```python
# Source: insight/differentiable_eq.py lines 7-33
class StraightThroughClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        ctx.input_dtype = x.dtype
        x_fp32 = x.float()
        return torch.clamp(x_fp32, min_val, max_val).to(ctx.input_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.float()
        return grad.to(ctx.input_dtype), None, None

def ste_clamp(x, min_val, max_val):
    return StraightThroughClamp.apply(x, min_val, max_val)
```

### Reference: gain_trunk_head (already uses STE clamp correctly)
```python
# Source: insight/differentiable_eq.py lines 796-797 (else branch)
gain_raw = self.gain_trunk_head(trunk_out).squeeze(-1)
gain_db = ste_clamp(gain_raw, -24.0, 24.0)
```

### Target: gain_mlp with Tanh removed
```python
# Modified gain_mlp (remove Tanh from Sequential)
self.gain_mlp = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1),
    # Tanh REMOVED -- raw output, bounded by ste_clamp
)

# Forward usage:
gain_db = self.gain_mlp(trunk_out).squeeze(-1) * self.gain_output_scale
gain_db = ste_clamp(gain_db, -24.0, 24.0)
```

### Mel-residual components to remove from __init__
```python
# ALL of these in MultiTypeEQParameterHead.__init__() become dead after removal:
# self.use_mel_for_gain (flag)
# self.gain_smooth_pool (AvgPool1d)
# self.gain_mel_aux (Sequential MLP)
# self.gain_aux_scale (Parameter)
# self.gain_blend_gate (Parameter)
# self.gain_trunk_head (Linear -- dead code in else branch)
```

### Forward code to remove (lines ~762-784)
```python
# REMOVE: mel-residual gain computation and blending
# mel_smooth = self.gain_smooth_pool(mel_2d).squeeze(1)
# mel_residual = mel_profile - mel_smooth
# gain_readout = ...
# gain_mel_in = torch.cat([trunk_out, gain_readout.unsqueeze(-1)], dim=-1)
# gain_aux = self.gain_mel_aux(gain_mel_in).squeeze(-1) * self.gain_aux_scale
# alpha = torch.sigmoid(self.gain_blend_gate)
# gain_db = alpha * gain_db + (1 - alpha) * gain_aux
```

### train.py gradient monitoring update needed (line 517)
```python
# BEFORE:
if "gain_mlp" in name or "gain_trunk_head" in name or "gain_mel_aux" in name:
# AFTER:
if "gain_mlp" in name:
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Gaussian readout for gain | MLP regression head (gain_mlp) | Pre-Phase 2 | Already in codebase but with Tanh |
| Tanh gain activation | STE clamp | Being changed Phase 2 | Full gradient flow within bounds |
| Dual-path gain (primary + mel-aux) | Single MLP path | Being changed Phase 2 | Noise removal from gain predictions |
| gain_trunk_head fallback | Unified gain_mlp | Being changed Phase 2 | Single codepath, simpler architecture |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | No config.yaml keys reference mel-residual blend weight | Code Examples | Searched all config files -- none found [VERIFIED: grep on conf/*.yaml] |
| A2 | gain_trunk_head can be removed alongside gain_mel_aux since gain_mlp replaces both | Architecture Patterns | gain_trunk_head is only used in the else branch when mel_profile is None -- after unification it becomes dead code |
| A3 | process_frame() will still pass mel_profile to param_head after changes (for freq/type heads) | Pitfalls | mel_profile is computed from streaming buffer independently -- gain path removal does not affect it |
| A4 | Existing checkpoints contain gain_mel_aux, gain_smooth_pool, gain_blend_gate keys that will cause load errors | Pitfalls | Old checkpoint load will fail with strict=True; needs strict=False or retraining |

## Open Questions

1. **Checkpoint compatibility**
   - What we know: Old checkpoints contain keys for gain_mel_aux, gain_smooth_pool, etc. that will not exist after removal.
   - What's unclear: Whether train.py should auto-handle this or require retraining from scratch.
   - Recommendation: Add `strict=False` to `_load_checkpoint` for backward compat, or document that Phase 2 requires retraining.

2. **gain_mlp vs gain_trunk_head unification**
   - What we know: Both produce gain predictions. gain_mlp has 2 layers + Tanh, gain_trunk_head has 1 layer + STE clamp.
   - What's unclear: Whether to keep the 2-layer MLP (more capacity) or simplify to single linear.
   - Recommendation: Keep gain_mlp (2 layers) with Tanh removed. The extra capacity helps with the full gain range. This falls under Claude's Discretion per CONTEXT.md.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | Model, training, STE clamp | Yes | 2.8.0+cu128 | -- |
| CUDA GPU | Training | Yes | NVIDIA RTX PRO 6000 Blackwell | -- |
| scipy | Hungarian matching | Yes | 1.11.4 | -- |
| pyyaml | Config loading | Yes | installed | -- |

**Missing dependencies with no fallback:** None

**Missing dependencies with fallback:** None

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Standalone Python scripts (no pytest) |
| Config file | none -- existing pattern |
| Quick run command | `cd insight && python test_gain_fix.py` |
| Full suite command | `cd insight && python test_metrics.py && python test_eq.py && python test_model.py && python test_streaming.py` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GAIN-01 | Gain predicted via MLP from trunk embedding only | unit | `cd insight && python test_gain_fix.py` | No -- Wave 0 |
| GAIN-02 | STE clamp used, no Tanh in gain path | unit | `cd insight && python test_gain_fix.py` | No -- Wave 0 |
| GAIN-03 | Mel-residual aux path fully removed | unit | `cd insight && python test_gain_fix.py` | No -- Wave 0 |
| GAIN-04 | Gain MAE < 3 dB after training | integration | `cd insight && python train.py` (manual check) | N/A |
| STRM-01 | Streaming inference preserved | unit | `cd insight && python test_gain_fix.py` | No -- Wave 0 |
| STRM-02 | Streaming vs batch gain diff < 0.1 dB | unit | `cd insight && python test_gain_fix.py` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `cd insight && python test_gain_fix.py`
- **Per wave merge:** `cd insight && python test_metrics.py && python test_eq.py && python test_streaming.py`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `insight/test_gain_fix.py` -- covers GAIN-01, GAIN-02, GAIN-03, STRM-01, STRM-02
  - Test: gain path uses only trunk embedding (no mel_residual parameters in model)
  - Test: gain_mlp has no Tanh activation
  - Test: STE clamp is the gain activation (gradient is identity within bounds)
  - Test: streaming vs batch gain consistency < 0.1 dB
  - Test: gradient flows through gain path (no zero gradients)

## Sources

### Primary (HIGH confidence)
- Code review: `insight/differentiable_eq.py` lines 509-831 (EQParameterHead + MultiTypeEQParameterHead)
- Code review: `insight/model_tcn.py` lines 497-755 (StreamingTCNModel forward + process_frame)
- Code review: `insight/train.py` lines 500-540 (gradient monitoring)
- Code review: `insight/loss_multitype.py` (loss computation)
- Code review: `insight/baseline_metrics.md` (pre-fix baseline: 5.60 dB gain MAE matched)
- Runtime verification: PyTorch 2.8.0+cu128, scipy 1.11.4, NVIDIA RTX PRO 6000 Blackwell

### Secondary (MEDIUM confidence)
- Phase 1 context and summaries -- metrics infrastructure established
- Config files reviewed -- no mel-residual config keys exist

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all verified by runtime check and code review
- Architecture: HIGH -- full code read of all affected files
- Pitfalls: HIGH -- identified from code analysis (Tanh in gain_mlp, dual gain paths, checkpoint compat)

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable -- core PyTorch API, no external dependencies)
