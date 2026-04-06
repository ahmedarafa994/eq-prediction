---
phase: 03-loss-architecture-restructuring
plan: 01
status: complete
wave: 1
---

## Summary

Wired the dual forward H_mag path and Gumbel detach during warmup. All other
plan items (warmup_epochs config, hybrid warmup gate, log-cosh, independent
weights, dual H_mag) were already implemented in prior sessions.

## Changes

### insight/loss_multitype.py
- **Gumbel detach (DATA-02):** Added conditional `pred_type_logits_for_match`
  that detaches type logits from the Hungarian matcher during warmup (lines
  434-439). Prevents noisy random-init type gradients from contaminating gain
  regression during gain-only warmup epochs.

### Already implemented (verified passing):
- `warmup_epochs: 5` in conf/config.yaml (loss section)
- `pred_H_mag_soft` / `pred_H_mag_hard` dual path in forward signature
- `pred_H_mag_hard.detach()` in hmag_loss computation
- Hybrid warmup gate (epoch + gain_mae_ema, 15-epoch hard cap)
- `log_cosh_loss` for gain regression
- Independent per-parameter weights (lambda_param=0.0)
- `update_gain_mae()` called per batch in train.py

## Test Results

```
9 tests: 9 passed, 0 failed
  PASS: log_cosh_wired (LOSS-03)
  PASS: independent_weights (LOSS-01)
  PASS: warmup_gating (LOSS-02)
  PASS: dual_hmag_signature (LOSS-04)
  PASS: hybrid_warmup_gate (DATA-02)
  PASS: spectral_reconstruction (LOSS-05)
  PASS: activity_mask (LOSS-06)
  PASS: warmup_config (config.yaml)
  PASS: update_gain_mae
```

## Key Decisions
- Detach is placed at matcher call site (not inside matcher) — keeps matcher
  generic and lets the loss function control gradient flow.
- No changes to train.py needed — dual H_mag path already wired there.
