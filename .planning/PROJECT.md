# IDSP EQ Estimator — Accuracy Improvement

## What This Is

A differentiable DSP system for blind parametric EQ parameter estimation. Given an audio signal processed through a multi-band parametric EQ (up to 5 bands, 5 filter types), the model estimates the EQ parameters (gain, frequency, Q, filter type) without access to the original dry signal. The system exists and trains but plateaus at poor accuracy — this project fixes it to professional quality for commercial deployment.

## Core Value

The model must accurately estimate EQ parameters from wet audio alone. If gain MAE stays above 1 dB, the product has no value for audio professionals.

## Requirements

### Validated

<!-- Inferred from existing codebase -->

- ✓ Differentiable biquad filter cascade with gradient flow — existing
- ✓ Hungarian matching for permutation-invariant band assignment — existing
- ✓ Multi-type filter support (peaking, lowshelf, highshelf, highpass, lowpass) — existing
- ✓ TCN encoder with 2D spectral front-end and attention pooling — existing
- ✓ Streaming inference (frame-by-frame) — existing
- ✓ Curriculum learning (5-stage) — existing
- ✓ Synthetic + MUSDB18 data pipeline with precompute — existing
- ✓ Spectral model alternative (0.20 dB MAE for H_db prediction) — existing
- ✓ ONNX export — existing

### Validated in Phase 3

- ✓ LOSS-01: Independent loss weights (lambda_param=0.0, independent lambda_gain/freq/q)
- ✓ LOSS-02: Gain-only warmup with hybrid gate (epoch count + gain_mae_ema threshold, 15-epoch hard cap)
- ✓ LOSS-03: Log-cosh loss for gain regression (numerically stable formulation)
- ✓ LOSS-04: Dual forward path (hard argmax for hmag_loss.detach(), soft Gumbel for spectral_loss)
- ✓ LOSS-05: Spectral L1 reconstruction (H_mag_soft vs target_H_mag, lambda_spectral=0.1)
- ✓ LOSS-06: Active band mask from dataset through train.py to loss
- ✓ DATA-02: Gumbel-Softmax detach during warmup (pred_type_logits.detach())

### Active

<!-- Current scope. Building toward these. -->

- [ ] Gain MAE < 1 dB (currently ~6 dB)
- [ ] Frequency MAE < 0.25 octaves (currently ~2.4 oct)
- [ ] Filter type accuracy > 95% (currently ~48%)
- [ ] Q MAE < 0.2 decades (currently ~0.49)
- [ ] Validation metrics computed with Hungarian matching (currently unmatched — metric bug)
- [ ] Gain prediction mechanism that doesn't rely on Gaussian readout from mel residual
- [ ] Loss weights that give gain equal or greater priority than frequency
- [ ] Gain activation without tanh gradient attenuation at moderate gains
- [ ] Training data with balanced gain distribution (not Beta-concentrated near zero)
- [ ] Stable gradient flow through Gumbel-Softmax during early training

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Real-time deployment / API serving — accuracy first, deploy later
- Spectral model as primary approach — spectral prediction works but parametric extraction via optimization is too slow for product
- New model architectures (transformers, etc.) — fix what's there first
- Mobile/edge optimization — product target is desktop DAW plugin
- Multi-band compression or dynamic EQ — parametric EQ only for v1

## Context

**Technical environment:** PyTorch training pipeline on Lightning AI (GPU-equipped). Main code in `insight/`. Models: `model_tcn.py` (primary), `model_cnn.py` (legacy). Loss: `loss_multitype.py`. Data: `dataset.py` + `dataset_pipeline/`. Config: `conf/config.yaml`.

**Prior work and iteration:** 14+ training runs (v2–v14) all plateau at similar metrics. Root cause analysis (`diagnostics/root_cause_analysis.md`) identifies 8 specific issues. Some fixes attempted in `fixes/` directory. The spectral model was developed as a workaround for encoder collapse but doesn't solve the parametric extraction problem.

**Key known issues (from root cause analysis):**
1. Gaussian readout produces uncalibrated gain estimates — mel residual amplitude doesn't map to dB gain
2. Validation metric bug — MAE computed on unmatched targets inflates reported error
3. Loss weight imbalance — gain signal drowned by spectral and competing losses
4. Tanh soft-clamp causes 20-40% gradient attenuation at moderate gains
5. Gumbel-Softmax dilutes gain gradients during early training (uncertain type probs)
6. Beta(2,5) gain distribution concentrates samples near zero
7. Hungarian matching underweights gain relative to frequency in assignment cost
8. Product-of-band gradient scaling creates unstable gain optimization

**What's already been tried:** Loss weight tuning (many iterations), embedding variance regularization, contrastive loss, spectral residual skip connection, 2D front-end redesign, Gumbel temperature annealing, curriculum learning. All helped marginally but none broke the plateau.

## Constraints

- **Tech stack:** PyTorch, scipy (Hungarian matching), no torchaudio in training loop — must stay differentiable
- **Compute:** Single GPU on Lightning AI — model must fit in available VRAM
- **Data:** 200k precomputed synthetic + MUSDB18 dataset — can regenerate if needed
- **Architecture:** Must preserve streaming inference capability (causal convolutions)
- **Evaluation:** Must have proper Hungarian-matched validation metrics to track real progress

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix TCN parametric pipeline, don't switch to spectral | Spectral model predicts H_db well (0.20 dB) but param extraction via scipy optimization is too slow for product; parametric approach gives direct parameter output | — Pending |
| Professional quality target (gain < 1 dB) | Commercial product for audio professionals — anything less isn't viable | — Pending |
| Preserve streaming mode | Product requires real-time frame-by-frame inference | — Pending |

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
*Last updated: 2026-04-06 after phase 3 completion*
