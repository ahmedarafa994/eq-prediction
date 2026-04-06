---
phase: 2
slug: gain-prediction-fix
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-06
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Standalone Python scripts (no pytest) |
| **Config file** | none — existing pattern |
| **Quick run command** | `cd insight && python test_gain_fix.py` |
| **Full suite command** | `cd insight && python test_metrics.py && python test_eq.py && python test_model.py && python test_streaming.py` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd insight && python test_gain_fix.py`
- **After every plan wave:** Run `cd insight && python test_metrics.py && python test_eq.py && python test_model.py && python test_streaming.py`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | GAIN-01 | T-02-01 | Gain predicted via MLP from trunk embedding only | unit | `cd insight && python test_gain_fix.py` | No -- W0 | pending |
| 02-01-01 | 01 | 1 | GAIN-02 | T-02-02 | STE clamp used, no Tanh in gain path | unit | `cd insight && python test_gain_fix.py` | No -- W0 | pending |
| 02-01-01 | 01 | 1 | GAIN-03 | T-02-03 | Mel-residual aux path fully removed | unit | `cd insight && python test_gain_fix.py` | No -- W0 | pending |
| 02-01-01 | 01 | 1 | STRM-01 | T-02-04 | Streaming inference preserved | unit | `cd insight && python test_gain_fix.py` | No -- W0 | pending |
| 02-01-01 | 01 | 1 | STRM-02 | T-02-05 | Streaming vs batch gain diff < 0.1 dB | unit | `cd insight && python test_gain_fix.py` | No -- W0 | pending |
| 02-02-01 | 02 | 2 | GAIN-04 | — | Gain MAE < 3 dB after training | integration | `cd insight && python train.py` (manual) | N/A | pending |

*Status: pending · green · red · flaky*

---

## Wave 0 Requirements

- [ ] `insight/test_gain_fix.py` — covers GAIN-01, GAIN-02, GAIN-03, STRM-01, STRM-02
  - Test: gain path uses only trunk embedding (no mel_residual parameters in model)
  - Test: gain_mlp has no Tanh activation
  - Test: STE clamp is the gain activation (gradient is identity within bounds)
  - Test: streaming vs batch gain consistency < 0.1 dB
  - Test: gradient flows through gain path (no zero gradients)

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Gain MAE < 3 dB after training | GAIN-04 | Requires full training run (~hours) | Run `cd insight && python train.py`, check validation metrics for gain MAE |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (gain MAE target deferred to Phase 3 training)
