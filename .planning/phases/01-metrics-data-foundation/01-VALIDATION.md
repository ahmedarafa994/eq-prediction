---
phase: 1
slug: metrics-data-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Standalone Python scripts (no pytest) |
| **Config file** | none — existing pattern |
| **Quick run command** | `cd insight && python test_metrics.py` |
| **Full suite command** | `cd insight && python test_metrics.py && python test_eq.py && python test_model.py` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd insight && python test_metrics.py`
- **After every plan wave:** Run `cd insight && python test_metrics.py && python test_eq.py`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | METR-01 | — | Hungarian matching produces correct assignments on known inputs | unit | `python test_metrics.py` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | METR-02 | — | Per-parameter MAE (gain, freq, Q, type) reported accurately | unit | `python test_metrics.py` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | METR-03 | — | All loss components logged during validation | unit | `python test_metrics.py` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | METR-04 | — | Gradient norms captured per parameter group | unit | `python test_metrics.py` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 1 | DATA-01 | — | Uniform gain distribution across full range | unit | `python test_metrics.py` | ❌ W0 | ⬜ pending |
| 1-03-02 | 03 | 1 | DATA-01 | — | HP/LP gain uses full gain_range | unit | `python test_metrics.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `insight/test_metrics.py` — stubs for METR-01 through DATA-01

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Baseline vs post-fix MAE comparison | METR-01 | Requires running a full validation epoch with real checkpoint | Run baseline validation, apply fixes, re-run validation, compare matched MAE delta |
| Precomputed cache regeneration | DATA-01 | Requires regenerating 200k sample dataset file | Delete old cache, run data generation, verify new cache has uniform distribution |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
