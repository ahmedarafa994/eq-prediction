# Phase 3: Loss Architecture Restructuring - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Restructure the loss function so gradient signal flows to gain regression first, then progressively activates spectral and other parameter losses. Includes: dual forward path (hard/soft types), log-cosh for gain, Gumbel detach during warmup, audio reconstruction loss wiring, and independent loss weights. No changes to model encoder or parameter head architecture — those are Phase 2 and Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Dual forward path
- **D-01:** hmag_loss uses H_mag_hard (argmax types) for clean param comparison. spectral_loss uses H_mag_soft (Gumbel-Softmax) for differentiable type gradients.
- **D-02:** model_tcn.py already has partial split (lines 637-643): `H_mag_hard` computed with argmax, `H_mag` soft during training. Verify loss_multitype.py uses `H_mag_hard` for hmag_loss component.

### Gain-only warmup
- **D-03:** Epoch-based warmup: 5 epochs gain-only, then activate freq/Q, then type, then spectral. Keep current phased activation schedule in MultiTypeEQLoss.
- **D-04:** Metric-gated transitions deferred to Phase 4 (DATA-03).
- **D-05:** Gumbel-Softmax type probabilities detached from gain gradient path during warmup only. After warmup, allow joint gradients.

### Audio reconstruction loss
- **D-06:** Spectral-domain MR-STFT: compare H_mag * wet_spectrum vs target_spectrum. No time-domain waveform reconstruction.
- **D-07:** Verify train.py actually passes pred_audio/target_audio to loss. Current code may have the wiring missing — loss_multitype.py checks for None and zeros out if not provided.

### Loss weights
- **D-08:** Use current config.yaml weights as starting point (lambda_gain: 2.5, lambda_freq: 1.5, lambda_q: 1.5, etc.). No grid search needed — warmup phasing handles the priority.

### Already-implemented items to verify
- **D-09:** log_cosh_loss() for gain — already implemented (loss_multitype.py:22-34, used at line 413). Verify it's correctly wired.
- **D-10:** Independent per-parameter weights — already in total loss computation (lines 498-509). Verify no combined lambda_param wrapper leaks.
- **D-11:** Per-band activity weighting (LOSS-06) — activity_loss exists. Verify it uses the correct active_band_mask.

### Claude's Discretion
- Exact implementation of Gumbel detach (detach type_probs tensor vs zero gradient scaling)
- How to wire audio reconstruction in train.py if currently missing
- Test structure and file organization

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Loss architecture
- `insight/loss_multitype.py` — MultiTypeEQLoss (main loss), log_cosh_loss (line 22), warmup gating (lines 348-353), dual H_mag consumption, total loss computation
- `insight/loss.py` — MultiResolutionSTFTLoss (MR-STFT used for spectral loss)

### Model forward path
- `insight/model_tcn.py` lines 606-655 — StreamingTCNModel.forward(): dual H_mag computation (H_mag_soft for training, H_mag_hard always)
- `insight/differentiable_eq.py` — DifferentiableBiquadCascade.forward() and forward_soft()

### Training loop
- `insight/train.py` — Trainer class, loss computation, validate() method, curriculum stages

### Configuration
- `insight/conf/config.yaml` — Loss weights, warmup epochs, curriculum stages

### Prior phase context
- `.planning/phases/02-gain-prediction-fix/02-CONTEXT.md` — Phase 2 decisions (gain head cleanup)
- `.planning/phases/01-metrics-data-foundation/01-CONTEXT.md` — Phase 1 decisions (metrics, data)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `log_cosh_loss()` already implemented and wired for gain — no new code needed
- Warmup gating infrastructure in MultiTypeEQLoss (is_warmup, is_freq_q_active, is_type_active, is_spectral_active)
- H_mag_hard already computed in model_tcn.py:637 and returned in output dict
- Independent per-parameter lambda weights already in config and loss computation

### Established Patterns
- Loss components logged in dict returned by MultiTypeEQLoss.forward()
- Hungarian matching applied once in loss, used for all param regression
- Config-driven loss weights via YAML

### Integration Points
- `model_tcn.py` returns both `H_mag` and `H_mag_hard` in forward() output dict
- `train.py` calls `self.loss_fn()` with the model output — need to verify H_mag_hard is passed to hmag_loss
- `loss_multitype.py` forward() signature accepts pred_H_mag but not pred_H_mag_hard separately

</code_context>

<specifics>
## Specific Ideas

- The dual forward path is partially implemented — model outputs both H_mag and H_mag_hard, but loss_multitype.py only receives pred_H_mag. Need to thread H_mag_hard through to hmag_loss.
- Gumbel detach: during warmup, wrap type_probs with `.detach()` before computing gain-related losses. After warmup, remove detach.
- Audio reconstruction loss may not be wired in train.py — loss_multitype.py has the code (MR-STFT) but zeros it out when pred_audio is None.

</specifics>

<deferred>
## Deferred Ideas

- Metric-gated curriculum transitions (gain threshold, freq threshold) — Phase 4 (DATA-03)
- Loss weight grid search or adaptive weight balancing — can revisit if warmup alone doesn't break the plateau

</deferred>

---

*Phase: 03-loss-architecture-restructuring*
*Context gathered: 2026-04-06*
