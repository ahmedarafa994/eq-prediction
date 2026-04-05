# Plan: Debug Training & Simplify EQ Estimation Model

## Context

The differentiable DSP EQ estimator is failing to learn. Training history shows the model stuck at random performance after 20+ epochs: gain MAE ~7 dB, freq MAE ~3 octaves, type accuracy ~50%. Additionally, `training_history.json` only has 20 entries while checkpoints exist up to epoch 70 — the history is being lost across runs/resumes.

Root causes identified:
1. **History bug**: Checkpoints saved without history key; stale disk file overwrites on resume
2. **Over-complex model**: 5-way filter type classification from the start, mel compression losing narrow EQ features, BatchNorm destroying magnitude info
3. **Weak training signal**: Cumulative mean pooling dilutes informative frames, loss weights may not be balanced

## Step 1: Fix training history persistence

**File:** `insight/train.py`

- In `save_checkpoint()`: write history JSON to a temp file first, then rename atomically (prevents corruption on crash)
- In `_load_checkpoint()`: validate loaded history length against checkpoint epoch. If mismatch, warn and reconstruct placeholder entries for skipped epochs
- Add `history_len` to checkpoint state dict for validation on resume

## Step 2: Create peaking-only baseline config

**New file:** `insight/conf/config_peaking_baseline.yaml`

Simplifications:
- `num_bands: 3` (from 5)
- `filter_types: ["peaking"]` — single type
- `gain_bounds: [-6, 6]`, `q_bounds: [0.5, 3.0]` — narrower ranges
- No curriculum — flat 50 epochs, `learning_rate: 0.001` (10x higher)
- `dataset_size: 20000`
- Loss: `lambda_hmag: 2.0`, `lambda_param: 1.0`, `lambda_type: 0.0`, `lambda_spectral: 0.0`, `lambda_spread: 0.0`
- `use_mel: false` — use full-band magnitude spectrum (1025 bins) to bypass mel compression as a test

## Step 3: Wire peaking-only mode through the codebase

**File:** `insight/train.py`
- Read `num_filter_types` from config (default to length of `filter_types` list)
- When dataset has only peaking bands, pass `filter_type=None` to `dsp_cascade.forward()` instead of using `forward_soft()` — eliminates the soft type blending overhead
- Skip type classification loss when `lambda_type=0`

**File:** `insight/model_tcn.py` — `StreamingTCNModel`
- Accept `input_bins` param (either 128 mel bins or 1025 full-spectrum bins) in place of hardcoded `n_mels` for the encoder
- When only peaking is used, call `dsp_cascade(gain, freq, q, n_fft)` directly (no soft blending)

**File:** `insight/dataset.py`
- Add `use_mel: false` option: when set, precompute full log-magnitude spectrum `(n_fft//2+1, T)` instead of mel-spectrogram
- Enforce minimum 1-octave frequency separation between bands after sampling
- Use RMS normalization for dry signals (instead of peak normalization)

## Step 4: Fix architecture issues preventing learning

**File:** `insight/model_tcn.py`

- Replace `BatchNorm1d` with `InstanceNorm1d(affine=True)` in `TCNBlock` (line 62) — preserves per-sample magnitude info needed for gain estimation
- Replace cumulative mean with learned attention pooling in `CausalTCNEncoder.forward()`:
  - Add `self.attn_proj = nn.Conv1d(channels, 1, 1)` 
  - Compute softmax attention over time, weighted sum → embedding
  - Allows model to focus on spectrally informative frames
- Add per-sample input normalization (subtract mean, divide by std) before input projection

**File:** `insight/differentiable_eq.py`
- Fix `EQParameterHead` to use log-frequency parameterization (same as `MultiTypeEQParameterHead` already does)

## Step 5: Update training loop for baseline

**File:** `insight/train.py`
- Add gradient norm logging per loss component (helps diagnose training dynamics)
- When `curriculum` section is absent from config, use flat training path (already exists at line 566)
- Support `config_path` argument so baseline config can be selected

## Verification

1. Run existing tests to confirm no regressions:
   ```bash
   cd insight && python test_eq.py && python test_model.py && python test_streaming.py
   ```
2. Run baseline training for 10 epochs:
   ```bash
   cd insight && python train.py --config conf/config_peaking_baseline.yaml
   ```
   - Verify train loss decreases consistently
   - Verify `training_history.json` has correct number of entries matching epochs
   - Verify checkpoints contain valid history
3. After 10 epochs, target: gain MAE < 3 dB, freq MAE < 1.5 oct (significant improvement over current random performance)
4. Run `python test_checkpoint.py` with the new checkpoint to inspect predictions

## Execution Order

Step 1 → Step 2 → Step 3 → Step 4 → Step 5 (each depends on the previous)
