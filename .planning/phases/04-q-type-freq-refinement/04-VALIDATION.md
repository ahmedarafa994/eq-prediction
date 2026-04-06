---
phase: 4
slug: q-type-freq-refinement
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-06
updated: 2026-04-06
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Standalone Python scripts (no pytest) |
| **Config file** | none — existing pattern |
| **Quick run command** | `cd insight && python test_multitype_eq.py` |
| **Full suite command** | `cd insight && python test_eq.py && python test_model.py && python test_multitype_eq.py && python test_streaming.py` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd insight && python test_multitype_eq.py`
- **After every plan wave:** Run full suite
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | QP-01 | — | N/A | unit | `cd insight && python -c "from differentiable_eq import MultiTypeEQParameterHead; h=MultiTypeEQParameterHead(64,5); import torch; x=torch.randn(2,64); p=h(x); assert p['q'].shape==(2,5); print('OK')"` | ✅ | ✅ green |
| 04-02-01 | 02 | 1 | FREQ-02 | — | N/A | unit | `cd insight && python -c "from loss_multitype import HungarianBandMatcher; m=HungarianBandMatcher(); assert m.lambda_gain==1.0 and m.lambda_freq==1.0; print('OK')"` | ✅ | ✅ green |
| 04-03-01 | 03 | 1 | TYPE-01 | — | N/A | unit | `cd insight && python test_multitype_eq.py` | ✅ | ✅ green |
| 04-04-01 | 04 | 2 | QP-02, TYPE-02, FREQ-01 | — | N/A | integration | `cd insight && python test_model.py` | ✅ | ✅ green |
| 04-05-01 | 05 | 2 | DATA-03 | — | N/A | integration | training dry-run with metric thresholds | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. Test files already exist:
- `insight/test_multitype_eq.py` — multi-type EQ + parameter head tests
- `insight/test_model.py` — CNN/TCN model forward/inverse/cycle/gradient
- `insight/test_eq.py` — biquad gradient flow
- `insight/test_streaming.py` — streaming consistency
- `insight/test_q_type_freq.py` — Phase 4 test suite (7 tests)
- `insight/test_q_type_freq.py` — Phase 4 test suite (7 tests)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions | Status |
|----------|-------------|------------|-------------------|--------|
| Training convergence to target metrics | QP-02, TYPE-02, FREQ-01 | Requires multi-epoch training run | Run `python train.py` and verify Q MAE < 0.2 decades, type acc > 95%, freq MAE < 0.25 octaves | ⬜ pending (needs training) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (pending live training verification for convergence metrics)
