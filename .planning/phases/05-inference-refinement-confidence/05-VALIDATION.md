---
phase: 5
slug: inference-refinement-confidence
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-06
updated: 2026-04-06
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Standalone Python scripts (no pytest — project convention) |
| **Config file** | `insight/conf/config.yaml` |
| **Quick run command** | `cd insight && python test_inference_refinement.py` |
| **Full suite command** | `cd insight && python test_inference_refinement.py && python test_model.py && python test_streaming.py` |
| **Estimated runtime** | ~30 seconds (no trained checkpoint required for structural tests) |

---

## Sampling Rate

- **After every task commit:** Run `cd insight && python test_inference_refinement.py`
- **After every plan wave:** Run full suite
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 5-01-01 | 01 | 0 | INFR-01 | unit | `cd insight && python test_inference_refinement.py` | ✅ green |
| 5-01-02 | 01 | 1 | INFR-01 | integration | `cd insight && python test_inference_refinement.py` | ✅ green |
| 5-01-03 | 01 | 1 | INFR-01 | integration | `cd insight && python test_inference_refinement.py` | ✅ green |
| 5-02-01 | 02 | 0 | INFR-02 | unit | `cd insight && python test_inference_refinement.py` | ✅ green |
| 5-02-02 | 02 | 1 | INFR-02 | integration | `cd insight && python test_inference_refinement.py` | ✅ green |
| 5-02-03 | 02 | 2 | INFR-01, INFR-02 | system | `cd insight && python test_inference_refinement.py && python test_streaming.py` | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `insight/test_inference_refinement.py` — 8 structural test functions:
  - [x] `test_refine_forward_api()` — refine_forward() exists and returns dict with expected keys
  - [x] `test_mc_dropout_selective_mode()` — dropout enabled, BatchNorm stays in eval mode
  - [x] `test_gradient_flow_through_biquad()` — gradients flow from H_mag back to (gain, freq, Q)
  - [x] `test_refinement_reduces_loss()` — spectral consistency loss decreases over 3+ steps
  - [x] `test_streaming_unchanged()` — process_frame() produces identical output before/after Phase 5
  - [x] `test_confidence_output_shape()` — confidence dict has correct keys and per-band structure
  - [x] `test_refine_forward_does_not_break_streaming()` — streaming works after refine_forward call
  - [x] `test_config_refinement_section()` — config.yaml has refinement section with required keys

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions | Status |
|----------|-------------|------------|-------------------|--------|
| 30% gain MAE improvement | INFR-01 | Requires trained checkpoint (not available in test env) | Run `cd insight && python evaluate_with_refinement.py` on best.pt — compare refined vs single-pass gain MAE | ⬜ pending (needs trained checkpoint) |
| Confidence calibration correlation | INFR-02 | Requires trained checkpoint + ground truth | Run eval pipeline, compute Pearson correlation between confidence score and actual parameter error | ⬜ pending (needs trained checkpoint) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved
