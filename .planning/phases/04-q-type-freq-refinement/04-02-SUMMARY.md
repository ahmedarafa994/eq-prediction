---
phase: 04-q-type-freq-refinement
plan: 02
status: complete
requirements: [QP-02, FREQ-02, TYPE-01, TYPE-02, DATA-03]
---

# Plan 04-02: Focal Loss + Metric-Gated Curriculum + Test Suite

## What was built

### Task 1: Equalized Hungarian cost weights + class-balanced focal loss
- `loss_multitype.py`: HungarianBandMatcher defaults equalized (lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0)
- Replaced `nn.CrossEntropyLoss` with class-balanced focal loss (gamma=2.0, inverse-frequency weights)
- Per-type weights: HP/LP get ~10x weight vs peaking

### Task 2: Metric-gated curriculum + per-type accuracy
- `train.py`: Validation matcher uses equalized weights
- `train.py`: `_last_metrics` tracking for metric-gated curriculum
- `train.py`: Metric gating in `_apply_curriculum_stage()` with epoch_cap fallback
- `train.py`: Per-type accuracy breakdown for all 5 filter types in validate()
- `config.yaml`: All 4 curriculum stages have metric_thresholds and epoch_cap

### Task 3: Phase 4 test suite
- `test_q_type_freq.py`: 7 tests covering QP-01, FREQ-02, TYPE-01, TYPE-02, DATA-03

## Test Results
- All 7 Phase 4 tests PASS
- test_eq.py PASS, test_model.py PASS (no regression)

## Key Files
- `insight/loss_multitype.py` — focal loss + equalized weights
- `insight/train.py` — metric gating + per-type accuracy
- `insight/conf/config.yaml` — curriculum metric thresholds
- `insight/test_q_type_freq.py` — Phase 4 test suite
