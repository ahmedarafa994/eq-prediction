---
phase: 03-loss-architecture-restructuring
plan: 01
subsystem: loss
tags: [loss-architecture, warmup-gating, gumbel-detach, dual-hmag]
dependency_graph:
  requires: []
  provides: [LOSS-01, LOSS-02, LOSS-03, LOSS-04, DATA-02]
  affects: [loss_multitype.py, train.py, conf/config.yaml]
tech_stack:
  added: []
  patterns: [gumbel-detach-during-warmup, hybrid-warmup-gate, dual-hmag-path]
key_files:
  created: [insight/test_loss_architecture.py]
  modified: [insight/loss_multitype.py, insight/conf/config.yaml, insight/train.py]
decisions:
  - Gumbel detach applied only during warmup (is_warmup flag)
  - pred_type_logits_for_match pattern used for clean conditional detach
metrics:
  duration: 15min
  completed: 2026-04-06
---

# Phase 3 Plan 01: Loss Architecture Wiring Summary

Implemented Gumbel-Softmax detach from gain gradient path during warmup (DATA-02). Most plan items (dual H_mag path, hybrid warmup gate, config-driven warmup_epochs, log-cosh, independent weights) were already implemented in prior commits -- this plan verified them via the test suite and added the one missing piece.

## Changes Made

### insight/loss_multitype.py
- Added `pred_type_logits_for_match` conditional detach: during warmup, `pred_type_logits.detach()` prevents noisy type gradients from contaminating gain regression; after warmup, raw logits flow freely for joint learning.

### insight/test_loss_architecture.py
- Fixed `test_gumbel_detach_warmup`: added `requires_grad=True` to `pred_gain` so `loss_gain.backward()` has a valid computation graph.
- 7 tests all pass: log_cosh_wired, independent_weights, warmup_gating, dual_hmag_signature, gumbel_detach_warmup, hybrid_warmup_gate, warmup_config.

### Already Implemented (verified, no changes needed)
- Dual H_mag path: `pred_H_mag_soft` and `pred_H_mag_hard` in forward() signature
- H_mag_hard detach: `.detach()` in hmag_loss computation
- Hybrid warmup gate: epoch AND gain_mae_ema threshold with 15-epoch hard cap
- Config: `warmup_epochs: 5` in conf/config.yaml
- train.py: passes both H_mag outputs to criterion, calls `update_gain_mae` per batch

## Deviations from Plan

None - plan executed as written. Tasks 0-2 were pre-implemented in prior commits; Task 3 (Gumbel detach) was the only new code change.

## Test Results

```
7 tests: 7 passed, 0 failed
python test_eq.py - PASSED
python test_multitype_eq.py - PASSED
```
