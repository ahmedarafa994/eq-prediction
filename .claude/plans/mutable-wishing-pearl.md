# Fix Frequency Compression in EQ Parameter Estimation

## Context

The model estimates EQ band parameters from audio. After 70 epochs, gain estimation is excellent (1.36 dB MAE), but **predicted frequencies cluster in the low-mid range** instead of spanning 20 Hz–20 kHz. Per-band results show 2+ octave errors for high-frequency bands (Bands 2 & 3), while the model accurately estimates gains. Five interacting root causes were identified.

## Root Causes

1. **Mel filterbank bottleneck** — 256 mel bins give ~4x more resolution below 3 kHz. `use_full_spectrum: false` means the model literally sees less information at high frequencies.
2. **No frequency-dependent loss weighting** — All octave errors weighted equally, but the model has better low-frequency info, so the optimizer exploits this.
3. **Missing coverage incentive** — Spread loss only enforces 0.5-octave minimum gap. Bands can cluster with no penalty.
4. **Asymmetric clamp values** — Hungarian matching uses `1e-6`, regression loss uses `1e-3`. Gradient inconsistency.
5. **No frequency prior in parameter head** — Random initialization means bands start with no frequency spread prior.

## Plan (4 phases, ordered by impact)

### Phase 1: Quick Fixes

**1a. Unify clamp values in `loss_multitype.py`**
- Lines 69, 75: change `clamp(min=1e-6)` → `clamp(min=1e-5)` in Hungarian matching
- Lines 228-229, 231-233: change `clamp(min=1e-3)` → `clamp(min=1e-5)` in regression loss
- Lines 352-354: change `clamp(min=1e-3)` → `clamp(min=1e-5)` in hmag loss
- Line 379: change `clamp(min=1e-3)` → `clamp(min=1e-5)` in spread loss
- `1e-5` is safe for FP16 (AMP autocast promotes log ops to FP32)

**1b. Enable full spectrum input in `conf/config.yaml`**
- Line 45: `use_full_spectrum: true`
- Consider reducing `num_blocks: 6` → `4` and `num_stacks: 3` → `2` to offset 4x larger input
- May need to reduce `batch_size: 32` → `24` if GPU OOM
- Verify dataset doesn't use stale precomputed mel cache

### Phase 2: Loss Improvements

**2a. Frequency-dependent weighting in `loss_multitype.py`**
- In `PermutationInvariantParamLoss.forward` (around line 227):
  - Change `self.huber` to `reduction='none'` for frequency loss only
  - Add octave-position weight: `1.0` at 20 Hz → `2.0` at 20 kHz (linear ramp in log-space)
  - Apply weight element-wise before `.mean()`
  - Keep gain/Q losses with default `reduction='mean'`

**2b. Coverage loss in `loss_multitype.py`**
- In `MultiTypeEQLoss.forward` (after spread loss, line 388):
  - Compute sorted log-frequencies, measure actual span vs target span (80% of log-range)
  - One-sided penalty: only penalize insufficient coverage, no penalty for wide spread
  - Add `lambda_coverage` weight (default `0.3`)
- Update `__init__` to accept `lambda_coverage` parameter

**2c. Frequency anchor loss in `loss_multitype.py`**
- Add direct L1 loss on log-frequency of matched bands, weighted by gain magnitude
- Stronger gradient signal than Huber for large octave errors
- Add `lambda_freq_anchor` weight (default `0.5`)
- Add to `total_loss` computation

### Phase 3: Architecture Improvements

**3a. Frequency prior in parameter head (`differentiable_eq.py`)**
- In `MultiTypeEQParameterHead.__init__`:
  - Add learnable per-band frequency bias initialized to uniform log-spacing
  - Bands start evenly spaced (e.g., 0.15, 0.50, 0.85 in sigmoid space → ~70 Hz, 634 Hz, 5.7 kHz)
  - Learnable scale decays during training as the model learns real positions
- This is a strong initialization prior, not a constraint

**3b. Frequency-aware input projection (`model_tcn.py`) — optional, low priority**
- If Phase 1-2 results are insufficient:
  - Replace 1x1 conv input projection with kernel_size=3 conv for slight frequency smoothing
  - Or add log-spaced subband grouping before the TCN

### Phase 4: Training Adjustments

**4a. Add per-octave validation metrics in `train.py`**
- Log frequency span (in octaves) of predicted bands
- Log per-octave error breakdown: low (<500 Hz), mid (500-4kHz), high (>4kHz)
- Makes frequency compression visible in training logs

**4b. Add frequency-focused curriculum stage in `conf/config.yaml`**
- New stage between `medium_multitype` and `full_difficulty`
- Moderate param ranges but higher `lambda_spread: 0.2`, `lambda_coverage: 0.3`, `lambda_freq_anchor: 0.5`
- Establishes good frequency spread behavior before tackling full difficulty

## Files to Modify

| File | Changes |
|------|---------|
| `insight/loss_multitype.py` | Clamp unification (1a), freq weighting (2a), coverage loss (2b), anchor loss (2c) |
| `insight/conf/config.yaml` | Full spectrum (1b), new loss weights (2b/2c), curriculum stage (4b) |
| `insight/differentiable_eq.py` | Frequency prior in `MultiTypeEQParameterHead` (3a) |
| `insight/train.py` | Validation metrics (4a), dataset precompute flag for full spectrum |
| `insight/model_tcn.py` | Optional: input projection change (3b) |

## Expected Impact

1. **Full spectrum (1b)** — Largest single impact. Eliminates root cause.
2. **Frequency-dependent weighting (2a)** — Compensates for residual STFT bin density differences.
3. **Frequency prior (3a)** — Strong initialization prevents clustering from the start.
4. **Coverage loss (2b)** — Directly penalizes the clustering failure mode.
5. **Clamp unification (1a)** — Eliminates systematic gradient inconsistency.
6. **Frequency anchor (2c)** — Sharper gradients for large octave errors.

## Verification

1. Run `python test_model.py` after each change to verify forward pass works
2. Train for 10-15 epochs on `easy_multitype` stage and check:
   - Per-octave frequency errors should be more balanced
   - Frequency span should be >5 octaves (vs current ~2-3)
3. Full training run (80 epochs) and compare to epoch_070 baseline:
   - Freq MAE target: <1.0 octave (vs 1.57 current)
   - Band 2 & 3 frequency errors: <1.0 octave (vs 2.27, 1.84)
   - Gain MAE should not regress (stay <1.5 dB)
4. Run `python export.py` to verify ONNX export still works
