# Phase 3: Loss Architecture Restructuring - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-04-06
**Phase:** 03-loss-architecture-restructuring
**Areas discussed:** Dual forward path, Gain-only warmup, Audio reconstruction loss, Loss weights tuning

## Dual Forward Path

| Option | Description | Selected |
|--------|-------------|----------|
| hmag uses hard, spectral uses soft | hmag_loss uses H_mag_hard (argmax types), for clean param comparison. spectral_loss uses H_mag_soft (Gumbel) for differentiable type gradients. | ✓ |
| Both use soft path | Simpler but spectral blurring contaminates magnitude comparison | |
| Both use hard path | Clean but loses differentiable type gradient | |

**User's choice:** hmag uses hard, spectral uses soft (Recommended)
**Notes:** Strict LOSS-04 interpretation — param losses need committed types for meaningful comparison, spectral benefits from smooth gradients.

## Gain-only Warmup

| Option | Description | Selected |
|--------|-------------|----------|
| 5 epochs gain-only | Current default, Predictable, already partially implemented | ✓ |
| 10 epochs | More thorough convergence | |
| 3 epochs | Quick warmup | |

**User's choice:** 5 epochs (current default)
**Notes:** Matches existing warmup_epochs config. Metric-gated transitions deferred to Phase 4.

## Gumbel Detach

| Option | Description | Selected |
|--------|-------------|----------|
| Detach during warmup only | DATA-02 requirement. After warmup, allow joint gradients. | ✓ |
| Never detach | Simpler but Gumbel noise dilutes gain signal throughout | |
| Always detach | Prevents any type gradient from diluting gain. May hurt joint optimization. | |

**User's choice:** Detach during warmup only (Recommended)
**Notes:** type_probs.detach() applied to gain path during warmup period only.

## Audio Reconstruction Loss

| Option | Description | Selected |
|--------|-------------|----------|
| Spectral-domain MR-STFT | Compare H_mag * wet_spectrum vs target_spectrum. No time-domain reconstruction. | ✓ |
| Time-domain waveform loss | More expensive, requires phase handling. MR-STFT already captures spectral similarity. | |

**User's choice:** Spectral-domain MR-STFT (Recommended)
**Notes:** Operates in frequency domain. Uses existing MR-STFT infrastructure.

## Loss Weight Tuning

| Option | Description | Selected |
|--------|-------------|----------|
| Use current weights | Keep config.yaml values as starting point. Warmup handles phasing. | ✓ |
| Increase gain weight | Boost gain relative to others | |
| Grid search | Short grid search for find better weights | |

**User's choice:** Use current weights (Recommended)
**Notes:** Current weights tuned through Phases 1-2. Warmup schedule handles the activation order.

## Claude's Discretion

- Exact implementation of Gumbel detach (where in forward() to insert .detach())
- Whether to pass pred_H_mag_hard separately or compute inline
- Warmup epoch staging thresholds

## Deferred Ideas

None — discussion stayed within phase scope
