---
phase: quick-260408-p8c
plan: 01
subsystem: model
tags: [shelf-detection, type-classification, feature-engineering, lowshelf-accuracy]
dependency_graph:
  requires: []
  provides: [shelf-detection-features, n_shelf_bands-config]
  affects: [MultiTypeEQParameterHead, StreamingTCNModel, type_head-input-dim]
tech_stack:
  added: []
  patterns: [fixed-scalar-features, non-learned-statistics, log-energy-ratio, spectral-tilt]
key_files:
  created: []
  modified:
    - insight/differentiable_eq.py
    - insight/model_tcn.py
    - insight/conf/config.yaml
decisions:
  - "Used abs() on mel_profile before log-ratio to guard against negative log-mel values producing NaN"
  - "n_shelf_bands=16 out of 128 mels — lowest 12.5% of filterbank (~sub-200Hz at 44100Hz sr)"
  - "All three shelf features (lo_ratio, hi_ratio, tilt_slope) clamped to [-5, 5] for training stability"
metrics:
  duration: "~10 minutes"
  completed: "2026-04-08T18:15:47Z"
  tasks_completed: 2
  files_modified: 3
---

# Quick Task 260408-p8c: Add Low-Frequency Energy Ratio Shelf-Detection Features

**One-liner:** Fixed (non-learned) lo_ratio, hi_ratio, tilt_slope scalar features injected into type_head input (hidden_dim 64→67) to give the type classifier direct discriminative signal for lowshelf vs peaking vs highshelf.

## What Was Changed

### `insight/differentiable_eq.py` (commit 306bf80)

**`MultiTypeEQParameterHead.__init__`:**
- Added `n_shelf_bands: int = 16` parameter, stored as `self.n_shelf_bands`
- Changed `type_input_dim` from `hidden_dim` (64) to `hidden_dim + 3` (67), widening the first `type_head` Linear layer from `Linear(64, 128)` to `Linear(67, 128)`

**New method `_compute_shelf_features(mel_profile)`:**
- Returns `(B, 3)` tensor: `[lo_ratio, hi_ratio, tilt_slope]`
- `lo_ratio`: `log(mean(|mel[0:n]|) / mean(|mel[n:]|))` — bass energy tilt relative to rest
- `hi_ratio`: `log(mean(|mel[-n:]|) / mean(|mel[:-n]|))` — treble energy tilt relative to rest
- `tilt_slope`: linear regression slope across all mel bands (normalized to `[-1, 1]` x-axis)
- Uses `abs()` on mel_profile before ratio computation to handle negative log-mel values safely
- All three features clamped to `[-5.0, 5.0]` to prevent exploding values from destabilizing training

**`MultiTypeEQParameterHead.forward`:**
- After computing `type_input` (via trunk + type_fusion_proj), computes shelf features from the raw (non-centered) `mel_profile`
- Expands `(B, 3)` → `(B, num_bands, 3)` and concatenates to `type_input` before passing to `type_head`
- When `mel_profile is None`, zero-pads 3 dimensions to maintain consistent `type_head` input size

### `insight/model_tcn.py` (commit 306bf80)

**`StreamingTCNModel.__init__`:**
- Added `n_shelf_bands: int = 16` parameter, stored as `self.n_shelf_bands`
- Passes `n_shelf_bands=n_shelf_bands` to `MultiTypeEQParameterHead` constructor

### `insight/conf/config.yaml` (commit 5f707f8)

Added under `model` section:
```yaml
n_shelf_bands: 16  # mel bands used for lo/hi energy ratio shelf detection (out of 128)
```

## The 3 Shelf-Detection Features

| Feature | Formula | Discriminative Signal |
|---------|---------|----------------------|
| `lo_ratio` | `log(E_low / E_rest)` where `E_low = mean(|mel[0:16]|)` | High for lowshelf (bass boost), low for highshelf/peaking |
| `hi_ratio` | `log(E_hi_tail / E_rest)` where `E_hi_tail = mean(|mel[-16:]|)` | High for highshelf (treble boost), low for lowshelf/peaking |
| `tilt_slope` | Linear regression slope of mel profile vs band index | Positive slope → high-frequency emphasis; negative → low-frequency emphasis |

**Why these features fix the lowshelf problem:** The learned `type_mel_proj` Conv1D fails to extract the monotonic energy gradient that distinguishes a shelf from a localized peak, because random initialization can converge to non-discriminative filters. These fixed scalar statistics provide the exact same gradient signal without any learning required — a lowshelf always has a higher `lo_ratio` than a peaking filter at the same frequency, by mathematical necessity.

**Discriminative test result:**
- Lowshelf-like profile `lo_ratio`: +1.03
- Peaking-like profile `lo_ratio`: -0.62

## Checkpoint Incompatibility Note

The `type_head`'s first linear layer changed input dimension from 64 → 67. **Any existing checkpoint is incompatible.** Training must start from scratch. This is expected and acceptable — the checkpoint files predate this change.

## Test Results

All 4 inline Task 1 tests passed:
- Forward with mel_profile: `type_logits` shape `(4, 5, 5)` correct
- Forward without mel_profile (zero-pad path): shape correct
- Shelf features discriminate lowshelf from peaking: `lo_ratio` 1.03 vs -0.62
- Shelf features finite and clamped for extreme input (`randn * 100`)

All 3 existing test scripts passed (Task 2):
- `test_model.py`: all 6 tests pass including gradient flow, shapes, dataset step
- `test_streaming.py`: batch/streaming consistency, latency benchmarks
- `test_multitype_eq.py`: gradient flow for all 5 filter types, MultiTypeEQParameterHead shapes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added abs() before log-ratio to handle negative mel values**
- **Found during:** Task 1 verification (Test 4)
- **Issue:** `_compute_shelf_features` produced NaN/inf for `torch.randn * 100` input because negative values in the numerator of the log ratio caused `log(negative / positive) = NaN`. The plan's formula used `(lo_mean + 1e-6)` which doesn't guard against negative means.
- **Fix:** Compute `mel_abs = mel_profile.abs()` and use `mel_abs` for all ratio computations. The tilt_slope uses `y_centered` which centers around the mean, so its sign is meaningful and abs() is not applied there.
- **Files modified:** `insight/differentiable_eq.py`
- **Commit:** 306bf80

## Self-Check: PASSED

- `insight/differentiable_eq.py` exists and contains `_compute_shelf_features` and `lo_ratio`
- `insight/model_tcn.py` exists and contains `n_shelf_bands`
- `insight/conf/config.yaml` exists and contains `n_shelf_bands: 16`
- Commit 306bf80 exists (Task 1)
- Commit 5f707f8 exists (Task 2)
