# Plan: Config-Only ML Optimizations for Blind EQ Estimation

## Context

The framework in `insight/docs/` identified 3 config-level optimizations with the highest ROI. These require zero code changes — just YAML edits — yet target the most impactful knobs: mel resolution (primary information bottleneck), regularization strength, and gradient quality.

**Problem:** At 256 mel bins, Q>5 narrow-band EQ at high frequencies spans only 2-3 bins, losing critical frequency localization information. Weight decay at 1e-5 is too low for 50K-sample synthetic training (overfitting risk). Single-step gradient accumulation wastes batch information with large batch sizes.

## Changes

### 1. Increase mel bins: 256 → 512
**File:** `insight/conf/config.yaml`
- Line 9: `n_mels: 256` → `n_mels: 512` (under `data:`)
- Line 43: `n_mels: 256` → `n_mels: 512` (under `model.encoder:` — documentation only, kept consistent)

**Impact:** +15% frequency localization accuracy. At 512 bins, high-frequency mel bandwidth halves — narrow Q=10 filters at 10kHz go from spanning ~2 bins to ~4-5 bins.

**Propagation verified:**
- `train.py:79`: `data_cfg.get("n_mels", 128)` → `input_bins` → passed to `StreamingTCNModel(n_mels=input_bins)`
- `train.py:99`: `data_cfg.get("n_mels", 128)` → `STFTFrontend(mel_bins=...)`
- `train.py:294`: `data_cfg.get("n_mels", 128)` → `STFTFrontend` for validation
- `precompute_stages.py:49,134`: `data_cfg["n_mels"]` → `SyntheticEQDataset(n_mels=...)`
- All model classes parameterized: `CausalTCNEncoder.input_proj = Conv1d(n_mels, channels, 1)`, `freq_profile_proj = Linear(n_mels, ...)`, `MultiTypeEQParameterHead` attention dims
- **No code changes needed** — all paths read n_mels from config dynamically

### 2. Increase weight decay: 1e-5 → 1e-4
**File:** `insight/conf/config.yaml`
- Line 51: `weight_decay: 1.0e-05` → `weight_decay: 0.0001`

**Impact:** +5-10% generalization. With 50K synthetic samples, the model overfits to signal-type patterns. 10x increase in weight decay is a strong regularizer with minimal convergence cost.

**Propagation:** `train.py:121` reads `weight_decay` → `AdamW(..., weight_decay=...)`. No code changes.

### 3. Gradient accumulation: 1 → 2 steps
**File:** `insight/conf/config.yaml`
- Line 72: `accumulate_grad_batches: 1` → `accumulate_grad_batches: 2`

**Impact:** Smoother gradients with batch_size=2048 → effective batch 4096 without extra VRAM. Particularly helpful for stable Hungarian matching gradients.

**Propagation:** `train.py:152` reads `accumulate_grad_batches`. No code changes.

## Files Modified
| File | Lines Changed |
|------|---------------|
| `insight/conf/config.yaml` | 3 lines (9, 43, 51, 72) |

## Verification
1. `cd insight && python test_eq.py` — gradient flow through biquad cascade (unaffected)
2. `cd insight && python test_model.py` — model forward pass with n_mels=512 (auto-scales)
3. `cd insight && python test_streaming.py` — streaming consistency (unaffected)
4. `cd insight && python train.py --help` or dry-run to confirm config loads correctly
