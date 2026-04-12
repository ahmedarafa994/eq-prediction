# IDSP EQ Estimator — Accuracy Improvement

## What This Is

A differentiable DSP system for blind parametric EQ parameter estimation. Given an audio signal processed through a multi-band parametric EQ (up to 5 bands, 5 filter types), the model estimates the EQ parameters (gain, frequency, Q, filter type) without access to the original dry signal. v1.0 shipped architectural improvements (gain head, loss restructuring, inference refinement). v1.1 focuses on breaking the accuracy plateau through wav2vec2 backbone fine-tuning and training strategy optimization.

## Current Milestone: v1.1 Backbone Fine-tuning & Accuracy Push

**Goal:** Break through the 2.68 dB gain MAE plateau by unfreezing the wav2vec2 backbone for end-to-end fine-tuning, closing the gap toward <1 dB gain MAE.

**Target features:**
- Wav2vec2 backbone unfreezing with gradient checkpointing
- 3-group optimizer with differentiated LRs (backbone/encoder/head)
- Freeze-then-unfreeze curriculum (warmup → full fine-tuning)
- Training strategy iteration (data augmentation, loss tuning, LR schedules)

## Core Value

The model must accurately estimate EQ parameters from wet audio alone. If gain MAE stays above 1 dB, the product has no value for audio professionals.

## Requirements

### Validated

- ✓ Differentiable biquad filter cascade with gradient flow — existing
- ✓ Hungarian matching for permutation-invariant band assignment — existing
- ✓ Multi-type filter support (peaking, lowshelf, highshelf, highpass, lowpass) — existing
- ✓ TCN encoder with 2D spectral front-end and attention pooling — existing
- ✓ Streaming inference (frame-by-frame) — existing
- ✓ Curriculum learning (5-stage) — existing
- ✓ Synthetic + MUSDB18 data pipeline with precompute — existing
- ✓ Spectral model alternative (0.20 dB MAE for H_db prediction) — existing
- ✓ ONNX export — existing
- ✓ METR-01: Hungarian-matched validation MAE — v1.0 (Phase 1)
- ✓ METR-02: Per-parameter MAE reported separately — v1.0 (Phase 1)
- ✓ METR-03: All loss components logged — v1.0 (Phase 1)
- ✓ METR-04: Gradient norm monitoring per parameter group — v1.0 (Phase 1)
- ✓ DATA-01: Uniform gain distribution — v1.0 (Phase 1)
- ✓ GAIN-01: Direct MLP regression head from trunk embedding — v1.0 (Phase 2)
- ✓ GAIN-02: STE clamp for gain activation — v1.0 (Phase 2)
- ✓ GAIN-03: Mel-residual auxiliary gain path removed — v1.0 (Phase 2)
- ✓ STRM-01: Streaming inference preserved — v1.0 (Phase 2)
- ✓ STRM-02: Streaming vs batch consistency verified — v1.0 (Phase 2)
- ✓ LOSS-01: Independent loss weights for gain/freq/Q — v1.0 (Phase 3)
- ✓ LOSS-02: Gain-only warmup period — v1.0 (Phase 3)
- ✓ LOSS-03: Log-cosh loss for gain regression — v1.0 (Phase 3)
- ✓ LOSS-04: Dual forward path (hard argmax + soft Gumbel) — v1.0 (Phase 3)
- ✓ LOSS-05: Spectral reconstruction loss — v1.0 (Phase 3)
- ✓ LOSS-06: Per-band activity mask — v1.0 (Phase 3)
- ✓ DATA-02: Gumbel-Softmax detach during warmup — v1.0 (Phase 3)
- ✓ QP-01: Q log-linear parameterization with STE clamp — v1.0 (Phase 4)
- ✓ FREQ-02: Equalized Hungarian cost weights — v1.0 (Phase 4)
- ✓ TYPE-02: Per-type accuracy breakdown — v1.0 (Phase 4)
- ✓ DATA-03: Metric-gated curriculum transitions — v1.0 (Phase 4)
- ✓ INFR-01: Gradient-based parameter refinement — v1.0 (Phase 5)
- ✓ INFR-02: MC-Dropout confidence estimation — v1.0 (Phase 5)

### Active

- [ ] GAIN-04: Gain MAE < 1 dB — v1.0 plateau at 2.68 dB, backbone fine-tuning needed
- [ ] QP-02: Q MAE < 0.2 decades — v1.0 plateau at 0.52 dec
- [ ] TYPE-01: Filter type accuracy > 95% — v1.0 plateau at 46.9%
- [ ] FREQ-01: Frequency MAE < 0.25 octaves — v1.0 plateau at 1.79 oct
- [ ] BACK-01: Wav2vec2 backbone unfreezing with gradient checkpointing
- [ ] BACK-02: 3-group optimizer (backbone/encoder/head) with differentiated LRs
- [ ] BACK-03: Freeze-then-unfreeze curriculum (warmup → fine-tuning)
- [ ] TRAIN-01: Training strategy iteration to close accuracy gap

### Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time deployment / API serving | Accuracy first, deploy later |
| Spectral model as primary approach | Parametric extraction via optimization too slow for product |
| New model architectures (transformers) | Fix current pipeline first |
| Mobile/edge optimization | Product target is desktop DAW plugin |
| Multi-band compression / dynamic EQ | Parametric EQ only for v1 |

## Context

**Shipped v1.0** with ~15,000 LOC Python across 120 files. All 5 phases implemented: measurement fixes, gain head rebuild, loss restructuring, parameter refinement, and inference optimization.

**Tech stack:** PyTorch, scipy (Hungarian matching), custom STFT frontend, no torchaudio in training loop.

**Pre-fix baseline:** gain MAE 5.60 dB, freq MAE 2.38 oct, Q MAE 0.47 decades, type accuracy 48.1%.

**Known issues:**
- Live training verification deferred — all implementations are code-verified but convergence metrics require multi-epoch training run
- Loss weight schedules and curriculum thresholds are analytical starting points needing empirical tuning
- Inference refinement adds 50-200ms latency (acceptable for batch/evaluation use)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix TCN parametric pipeline, don't switch to spectral | Spectral model predicts H_db well but param extraction too slow | ✓ Good |
| Professional quality target (gain < 1 dB) | Commercial product for audio professionals | — Pending training |
| Preserve streaming mode | Product requires real-time frame-by-frame inference | ✓ Good — preserved through all phases |
| Direct MLP + STE clamp for gain | Gaussian readout produced uncalibrated estimates | ✓ Good |
| Dual forward path (hard + soft types) | Prevents type uncertainty from contaminating gain regression | ✓ Good |
| Class-balanced focal loss | Rare types (HP/LP) underrepresented in data | — Pending training |
| Gradient-based refinement + MC-Dropout | Batch-mode accuracy boost without streaming impact | — Pending training |

## Constraints

- **Tech stack:** PyTorch, scipy (Hungarian matching), no torchaudio in training loop — must stay differentiable
- **Compute:** Single GPU on Lightning AI — model must fit in available VRAM
- **Data:** 200k precomputed synthetic + MUSDB18 dataset — can regenerate if needed
- **Architecture:** Must preserve streaming inference capability (causal convolutions)
- **Evaluation:** Must have proper Hungarian-matched validation metrics to track real progress

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-12 after v1.1 milestone start*
