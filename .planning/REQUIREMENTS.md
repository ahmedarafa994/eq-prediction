# Requirements: IDSP EQ Estimator — Accuracy Improvement

**Defined:** 2026-04-05
**Core Value:** The model must accurately estimate EQ parameters from wet audio alone — gain MAE < 1 dB

## v1 Requirements

### Metrics & Measurement

- [ ] **METR-01**: Validation MAE computed with Hungarian-matched targets (not unmatched)
- [ ] **METR-02**: Per-parameter MAE reported separately (gain dB, freq octaves, Q decades, type accuracy)
- [ ] **METR-03**: All loss components logged during training and validation
- [ ] **METR-04**: Gradient norm monitoring per parameter group (gain, freq, Q, type)

### Gain Prediction

- [ ] **GAIN-01**: Replace Gaussian readout with direct MLP regression head from trunk embedding
- [ ] **GAIN-02**: Use STE clamp for gain activation instead of tanh (identity gradient within bounds)
- [ ] **GAIN-03**: Remove mel-residual auxiliary gain path (it injects noise, not signal)
- [ ] **GAIN-04**: Gain MAE < 1 dB on validation set with matched metrics

### Loss Architecture

- [ ] **LOSS-01**: Separate loss weights for gain, freq, Q (independent tuning, not combined param loss)
- [ ] **LOSS-02**: Loss phasing — gain-only warmup period before enabling spectral losses
- [ ] **LOSS-03**: Log-cosh loss for gain regression (smoother gradients than Huber)
- [ ] **LOSS-04**: Dual forward path — hard argmax types for param regression loss, soft Gumbel for spectral loss
- [ ] **LOSS-05**: Audio-domain reconstruction loss as additional training signal
- [ ] **LOSS-06**: Per-band loss weighting based on band activity

### Data & Training

- [ ] **DATA-01**: Uniform gain distribution instead of Beta(2,5) concentration near zero
- [ ] **DATA-02**: Gumbel-Softmax gradient protection — detach type probs from gain gradient path during warmup
- [ ] **DATA-03**: Metric-gated curriculum transitions (advance stage only when metrics meet threshold)

### Q Parameterization

- [ ] **QP-01**: Switch Q head from sigmoid-to-exp to log-linear parameterization with STE clamp
- [ ] **QP-02**: Q MAE < 0.2 decades on validation set

### Type Classification

- [ ] **TYPE-01**: Filter type accuracy > 95% on validation set
- [ ] **TYPE-02**: Per-type accuracy breakdown (peaking, lowshelf, highshelf, highpass, lowpass)

### Frequency Prediction

- [ ] **FREQ-01**: Frequency MAE < 0.25 octaves on validation set
- [ ] **FREQ-02**: Hungarian matching cost matrix balances gain and frequency weight equally

### Inference & Confidence

- [ ] **INFR-01**: Inference-time refinement improves gain MAE by ≥30% over single-pass
- [ ] **INFR-02**: Per-band confidence estimation (calibrated probability for type + parameter uncertainty)

### Streaming Compatibility

- [ ] **STRM-01**: All model changes preserve streaming inference (causal convolutions, frame-by-frame processing)
- [ ] **STRM-02**: Streaming vs batch consistency verified (< 0.1 dB gain difference)

## v2 Requirements

### Advanced Training

- **TRN-01**: Snapshot ensemble for inference-time accuracy boost
- **TRN-02**: Feature importance analysis (which mel bins drive gain predictions)
- **TRN-03**: SpecAugment integration for training robustness

### Production

- **PROD-01**: Temperature scaling calibration for type confidence
- **PROD-02**: Conformal prediction intervals for parameter estimates
- **PROD-03**: ONNX export validation with accuracy parity

## Out of Scope

| Feature | Reason |
|---------|--------|
| Dynamic EQ / multi-band compression | Different problem, parametric EQ only for v1 |
| New model architectures (transformers) | Fix current pipeline first |
| Real-time API / DAW plugin deployment | Accuracy first, deploy later |
| Mobile/edge optimization | Desktop DAW plugin is target |
| Real-time chat / collaboration | Not relevant to this system |
| Spectral model as primary approach | Slow parametric extraction, not product-viable |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| METR-01 | Phase 1 | Pending |
| METR-02 | Phase 1 | Pending |
| METR-03 | Phase 1 | Pending |
| METR-04 | Phase 1 | Pending |
| DATA-01 | Phase 1 | Pending |
| GAIN-01 | Phase 2 | Pending |
| GAIN-02 | Phase 2 | Pending |
| GAIN-03 | Phase 2 | Pending |
| GAIN-04 | Phase 2 | Pending |
| LOSS-01 | Phase 3 | Pending |
| LOSS-02 | Phase 3 | Pending |
| LOSS-03 | Phase 3 | Pending |
| LOSS-04 | Phase 3 | Pending |
| LOSS-05 | Phase 3 | Pending |
| LOSS-06 | Phase 3 | Pending |
| DATA-02 | Phase 3 | Pending |
| DATA-03 | Phase 4 | Pending |
| QP-01 | Phase 4 | Pending |
| QP-02 | Phase 4 | Pending |
| TYPE-01 | Phase 4 | Pending |
| TYPE-02 | Phase 4 | Pending |
| FREQ-01 | Phase 4 | Pending |
| FREQ-02 | Phase 4 | Pending |
| INFR-01 | Phase 5 | Pending |
| INFR-02 | Phase 5 | Pending |
| STRM-01 | Phase 2 | Pending |
| STRM-02 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-05*
*Last updated: 2026-04-05 after initial definition*
