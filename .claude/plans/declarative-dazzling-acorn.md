# Plan: Integrate ML Framework into IDSP EQ Estimator Training

## Context

The IDSP EQ estimator (`insight/`) trains a TCN model to predict multi-band parametric EQ parameters from audio. It has 11 loss lambda weights all manually tuned, hardcoded label smoothing, and no uncertainty quantification on predictions. The ML framework (`ml_framework/`) we built provides tools for hyperparameter tuning, label smoothing, and conformal prediction. We integrate 3 high-impact improvements.

---

## Change 1: Configurable Label Smoothing Schedule (4 lines, 3 files)

**Why**: `label_smoothing=0.1` is hardcoded in `loss_multitype.py:518`. Making it configurable with a curriculum schedule (lower early, higher during calibration) improves filter type classification generalization.

**Files**:
- `insight/loss_multitype.py` — Add `label_smoothing` param to `__init__`, use `self.label_smoothing` instead of hardcoded `0.1` at line 518
- `insight/conf/config.yaml` — Add `label_smoothing` field per curriculum stage + top-level default
- `insight/train.py` — Add 2-line patch in `_setup_stage()` to apply per-stage `label_smoothing` to `self.criterion`

---

## Change 2: Loss Weight Tuning via Optuna (new script)

**Why**: 11 loss lambda weights are manually set. Bayesian optimization can find better combinations, potentially reducing val_loss significantly.

**File to create**: `insight/tune_loss_weights.py`

**Design**:
- Custom Optuna objective wrapping existing `Trainer` class (framework's `bayesian_optimization_tpe` is sklearn-only)
- Per-stage search spaces (only tune active lambdas per stage: 3-8 params instead of 11)
- Short trial runs (5-10 epochs) with optional warm-start from checkpoint
- Composite objective: normalized gain_mae + freq_mae + q_mae + (1-type_acc) + (1-band_f1)
- Results saved as JSON + config snippet for paste into config.yaml

**Usage**: `python tune_loss_weights.py --stage 2 --n-trials 50 --trial-epochs 8 --resume checkpoints/idsp-eq-v3-improved/epoch_029.pt`

---

## Change 3: Conformal Prediction Intervals (new script)

**Why**: No uncertainty quantification on EQ parameter predictions. Conformal prediction gives guaranteed coverage intervals ("gain is 6.2 ± 1.5 dB with 90% confidence").

**File to create**: `insight/run_conformal_evaluation.py`

**Design**:
- Wraps existing `ConformalEQPredictor` from `calibration.py` with evaluation harness
- Split conformal: calibration set → quantile → test set intervals
- Reports per-parameter (gain/freq/Q) coverage and interval width
- Per-filter-type breakdown (peaking, shelf, HP, LP)
- Conformal prediction sets for filter type classification
- Uses `evaluate_model.py`'s `compute_metrics_batch()` pattern for prediction collection with Hungarian matching

**Usage**: `python run_conformal_evaluation.py --checkpoint best.pt --alpha 0.1`

---

## Implementation Order

1. **Change 1** (label smoothing) — smallest, safest, no new dependencies
2. **Change 3** (conformal evaluation) — standalone script, reads checkpoints only
3. **Change 2** (loss tuning) — requires `pip install optuna`, most complex

## Verification

- Change 1: Run existing tests (`python test_model.py`), run 1 training epoch to verify label_smoothing is applied
- Change 2: Run `python tune_loss_weights.py --stage 1 --n-trials 3 --trial-epochs 2` as smoke test
- Change 3: Run `python run_conformal_evaluation.py --checkpoint best.pt --n-calibration 100 --n-test 100` as smoke test
