# Roadmap: IDSP EQ Estimator — Accuracy Improvement

## Overview

Systematic bug-fixing of a differentiable parametric EQ estimation pipeline that has plateaued at ~6 dB gain MAE. Five phases address eight compounding issues in strict dependency order: fix measurement first, then gain mechanism, then loss architecture, then remaining parameters, then polish for professional quality. Each phase delivers a verifiable improvement in model accuracy.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Metrics & Data Foundation** - Fix validation measurement and balance training data to establish trustworthy baseline
- [ ] **Phase 2: Gain Prediction Fix** - Replace broken gain head with direct MLP + STE clamp, remove mel-residual path, verify streaming
- [ ] **Phase 3: Loss Architecture Restructuring** - Phase loss activation, dual forward path, log-cosh for gain, independent loss weights
- [ ] **Phase 4: Q, Type & Frequency Refinement** - Fix Q parameterization, metric-gated curriculum, push all parameters to target accuracy
- [ ] **Phase 5: Inference Refinement & Confidence** - Inference-time parameter optimization and per-band confidence estimation

## Phase Details

### Phase 1: Metrics & Data Foundation
**Goal**: The model's actual performance can be measured accurately, and training data represents the full gain range uniformly
**Depends on**: Nothing (first phase)
**Requirements**: METR-01, METR-02, METR-03, METR-04, DATA-01
**Success Criteria** (what must be TRUE):
  1. Validation MAE is computed with Hungarian-matched targets, producing a lower and trustworthy baseline measurement
  2. Per-parameter MAE (gain dB, freq octaves, Q decades, type accuracy) is reported separately at each validation step
  3. All loss components are logged during both training and validation runs
  4. Gradient norms are monitored per parameter group (gain, freq, Q, type) to diagnose training dynamics
  5. Training data uses uniform gain distribution across the full range instead of Beta-concentrated near zero
**Plans**: 4 plans

Plans:
- [ ] 01-03-PLAN.md — Pre-fix baseline: run validation with current buggy code BEFORE changes (D-01)
- [x] 01-01-PLAN.md — Validation metrics instrumentation: test_metrics.py, component logging, gradient norm fix
- [ ] 01-02-PLAN.md — Data distribution fix: uniform gain, HP/LP gain range, cache regeneration
- [x] 01-04-PLAN.md — Post-fix baseline: run validation after fixes, compute delta against pre-fix (D-09)

### Phase 2: Gain Prediction Fix
**Goal**: The gain prediction mechanism produces accurate gain estimates with full gradient flow, without noise injection from auxiliary paths
**Depends on**: Phase 1
**Requirements**: GAIN-01, GAIN-02, GAIN-03, GAIN-04, STRM-01, STRM-02
**Success Criteria** (what must be TRUE):
  1. Gain is predicted via direct MLP regression head from trunk embedding (no Gaussian readout, no mel-residual path)
  2. Gain activation uses STE clamp so gradients pass through with identity within bounds (no tanh attenuation)
  3. Mel-residual auxiliary gain path is fully removed from the model code
  4. Gain MAE on validation set drops below 3 dB with matched metrics (down from ~6 dB)
  5. All changes preserve streaming inference — streaming vs batch consistency within 0.1 dB gain difference
**Plans**: TBD

### Phase 3: Loss Architecture Restructuring
**Goal**: The loss function directs gradient signal to gain regression first, then progressively activates spectral and other losses, preventing loss component competition
**Depends on**: Phase 2
**Requirements**: LOSS-01, LOSS-02, LOSS-03, LOSS-04, LOSS-05, LOSS-06, DATA-02
**Success Criteria** (what must be TRUE):
  1. Loss weights for gain, freq, and Q are independently tunable (not a single combined param loss)
  2. Training begins with a gain-only warmup period before spectral losses activate
  3. Gain regression uses log-cosh loss instead of Huber
  4. Dual forward path exists — hard argmax types for param regression loss, soft Gumbel for spectral loss
  5. Audio-domain reconstruction loss provides additional training signal
  6. Gumbel-Softmax type probs are detached from gain gradient path during warmup
**Plans**: TBD

### Phase 4: Q, Type & Frequency Refinement
**Goal**: All four parameter types (gain, freq, Q, type) reach target accuracy thresholds with metric-gated curriculum progression
**Depends on**: Phase 3
**Requirements**: QP-01, QP-02, TYPE-01, TYPE-02, FREQ-01, FREQ-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. Q head uses log-linear parameterization with STE clamp instead of sigmoid-to-exp
  2. Q MAE < 0.2 decades on validation set
  3. Filter type accuracy > 95% on validation set, with per-type breakdown reported
  4. Frequency MAE < 0.25 octaves on validation set
  5. Hungarian matching cost matrix balances gain and frequency weight equally
  6. Curriculum stage transitions are gated by metric thresholds (not epoch count alone)
**Plans**: TBD

### Phase 5: Inference Refinement & Confidence
**Goal**: Inference-time optimization pushes accuracy beyond single-pass results, and each band prediction carries a calibrated confidence estimate
**Depends on**: Phase 4
**Requirements**: INFR-01, INFR-02
**Success Criteria** (what must be TRUE):
  1. Inference-time refinement improves gain MAE by at least 30% over single-pass prediction
  2. Each predicted band has a calibrated confidence estimate reflecting type certainty and parameter uncertainty
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Metrics & Data Foundation | 0/3 | Planning complete | - |
| 2. Gain Prediction Fix | 0/? | Not started | - |
| 3. Loss Architecture Restructuring | 0/? | Not started | - |
| 4. Q, Type & Frequency Refinement | 0/? | Not started | - |
| 5. Inference Refinement & Confidence | 0/? | Not started | - |
