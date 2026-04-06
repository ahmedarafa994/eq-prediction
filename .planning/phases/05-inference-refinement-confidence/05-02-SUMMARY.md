---
phase: 05-inference-refinement-confidence
plan: 02
status: complete
requirements: [INFR-01, INFR-02]
---

# Plan 05-02: MC-Dropout Confidence Estimation

## What was built

### Task 1: MC-Dropout tests added to test suite
- `insight/test_inference_refinement.py` — 2 new test functions added (8 total):
  - `test_mc_dropout_selective_mode` — verifies BatchNorm stays eval, Dropout goes train
  - `test_confidence_output_shape` — validates forward(refine=True) returns correct confidence structure

### Task 2: MC-Dropout methods added to StreamingTCNModel
- `insight/model_tcn.py` — three new methods added:
  - `_refinement_config()` — reads config.yaml refinement section, falls back to defaults
  - `_run_mc_dropout_passes()` — runs N stochastic forward passes with selective Dropout enable, BatchNorm stays eval
  - `_compute_confidence()` — computes per-band confidence from MC-Dropout statistics:
    - type_entropy: normalized Shannon entropy of mean type_probs
    - gain_variance: variance across passes (dB^2)
    - freq_variance: variance in log-space (log-octave^2)
    - q_variance: variance in log-space (log-Q^2)
    - overall_confidence: weighted combination (0.4*type_conf + 0.2*gain_conf + 0.2*freq_conf + 0.2*q_conf)

- `forward()` signature updated: `def forward(self, mel_frames, refine: bool = False):`
  - When `refine=True`: runs MC-Dropout confidence then gradient refinement
  - When `refine=False` (default): identical behavior to before

### Task 3: evaluate_with_refinement.py created
- `insight/evaluate_with_refinement.py` — evaluation script:
  - Loads trained checkpoint (or random model for structural test)
  - Runs single-pass and refine=True evaluation on synthetic validation set
  - Computes Hungarian-matched gain MAE for both modes
  - Reports improvement percentage against 30% INFR-01 gate
  - Reports mean type entropy and confidence

## Key Files
- `insight/model_tcn.py` — _run_mc_dropout_passes(), _compute_confidence(), _refinement_config(), forward(refine=False)
- `insight/test_inference_refinement.py` — 8 tests total
- `insight/evaluate_with_refinement.py` — evaluation comparison script (new file)

## Test Results
- All 8 tests designed; syntax verified via py_compile
- Note: Full execution requires torch runtime (not available in current environment)
