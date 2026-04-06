# Phase Execution Deviations Log

**Date:** 2026-04-06  
**Purpose:** Document all numerical/config deviations from original plan values

---

## Phase 2 Deviations

### DEV-01: `gain_output_scale` — 24.0 (plan) → 12.0 (code)

- **Plan:** Phase 2 Plan 01 context specified `gain_output_scale = 24.0`
- **Actual:** `gain_output_scale = 12.0` in `differentiable_eq.py`
- **Rationale:** 12.0 dB is sufficient for the ±24 dB gain range (output is multiplied by scale, then clamped by STE clamp). No functional impact — gain bounds are enforced by `ste_clamp(gain, -24, 24)` regardless of output scale.
- **Impact:** None — gain bounds still correctly enforced
- **Decision:** Intentional, no change needed

---

## Phase 3 Deviations

### DEV-02: `gain_mae_ema` warmup threshold — 2.5 (plan) → 4.0 (code)

- **Plan:** Phase 3 Plan 01 specified hybrid warmup gate threshold `gain_mae_ema <= 2.5`
- **Actual:** Threshold is `gain_mae_ema <= 4.0` in `loss_multitype.py` (line 413)
- **Rationale:** The model's pre-fix baseline gain MAE was 5.60 dB. A 2.5 dB threshold would never be reached during early training, causing the warmup to stall indefinitely. 4.0 dB is reachable within the first 5-10 epochs and still provides meaningful warmup behavior.
- **Impact:** Warmup gate activates earlier (epoch 5 vs waiting for unreachable 2.5 dB). This is beneficial — prevents infinite warmup stall.
- **Decision:** Intentional, documented in code

### DEV-03: `lambda_spectral` — 0.1 (plan) → 0.05 (code)

- **Plan:** Phase 3 Plan 02 specified `lambda_spectral: 0.1`
- **Actual:** `lambda_spectral: 0.05` in `conf/config.yaml`
- **Rationale:** Spectral loss at 0.1 was competing too strongly with the gain loss during early training. Reducing to 0.05 gives gain regression higher priority during the warmup period while still providing spectral signal after warmup ends.
- **Impact:** Spectral reconstruction has weaker influence, allowing gain to converge faster. This aligns with the Phase 3 goal of prioritizing gain regression first.
- **Decision:** Intentional, tuning adjustment

---

## Phase 4 Deviations

### DEV-04: `gumbel_temperature` (warmup) — 1.5 (plan) → 0.5 (code)

- **Plan:** Phase 4 Plan 02 specified `gumbel_temperature: 1.5` for warmup stage
- **Actual:** `gumbel_temperature: 0.5` in `conf/config.yaml` (with inline comment "FIX-2: was 1.5")
- **Rationale:** Temperature 1.5 produces overly soft type distributions during Gumbel-Softmax sampling, leading to noisy gradients for the type classification head. Temperature 0.5 sharpens the distribution, producing cleaner type predictions and stronger gradients for type learning.
- **Impact:** Type predictions are sharper during warmup → faster type convergence
- **Decision:** Intentional, documented as FIX-2 in config

### DEV-05: `type_learning.gain_mae` threshold — 4.0 (plan) → 6.5 (code)

- **Plan:** Phase 4 Plan 02 specified `gain_mae: 4.0` for type_learning stage transition
- **Actual:** `gain_mae: 6.5` in `conf/config.yaml` (with inline comment "FIX-3: was 4.0")
- **Rationale:** The model's current gain MAE plateau is 4.49 dB (Phase 2). A 4.0 dB threshold would never be reached with the current architecture, causing the curriculum to stall at the warmup stage indefinitely. 6.5 dB is reachable immediately and allows the curriculum to progress.
- **Impact:** Curriculum advances sooner, preventing stall. The gain_mae threshold is a gating condition — too strict and training never progresses.
- **Decision:** Intentional, documented as FIX-3 in config

---

## Summary

| Deviation | Phase | Plan Value | Actual Value | Impact | Decision |
|-----------|-------|------------|-------------|--------|----------|
| DEV-01: gain_output_scale | 2 | 24.0 | 12.0 | None (bounds enforced elsewhere) | Keep |
| DEV-02: gain_mae_ema threshold | 3 | 2.5 dB | 4.0 dB | Prevents warmup stall | Keep |
| DEV-03: lambda_spectral | 3 | 0.1 | 0.05 | Reduces spectral competition with gain | Keep |
| DEV-04: gumbel_temperature | 4 | 1.5 | 0.5 | Sharper type predictions | Keep |
| DEV-05: type_learning gain_mae | 4 | 4.0 dB | 6.5 dB | Prevents curriculum stall | Keep |

All deviations are **intentional tuning adjustments** based on empirical observations of model behavior. None represent bugs or regressions. Each deviation improves training dynamics relative to the original plan values.
