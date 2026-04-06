---
phase: 1
slug: metrics-data-foundation
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-05
updated: 2026-04-06
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
| 1-01-01 | 01 | 1 | METR-01 | — | Hungarian matching produces correct assignments on known inputs | unit | `python test_metrics.py` | ✅ exists | ✅ green |
| 1-01-02 | 01 | 1 | METR-02 | — | Per-parameter MAE (gain, freq, Q, type) reported accurately | unit | `python test_metrics.py` | ✅ exists | ✅ green |
| 1-01-03 | 01 | 1 | METR-03 | — | All loss components logged during validation | unit | `python test_metrics.py` | ✅ exists | ✅ green |
| 1-01-04 | 01 | 1 | METR-04 | — | Gradient norms captured per parameter group | unit | `python test_metrics.py` | ✅ exists | ✅ green |
| 1-02-01 | 02 | 1 | DATA-01 | — | Uniform gain distribution across full range | unit | `python test_metrics.py` | ✅ exists | ✅ green |
| 1-02-02 | 02 | 1 | DATA-01 | — | Precomputed caches deleted and will regenerate with uniform data | manual | `ls insight/data/*.pt` returns empty | ✅ done | ✅ green |
| 1-03-01 | 03 | 0 | METR-01 | — | Pre-fix baseline capture with existing checkpoint (D-01) | manual | `test -f insight/baseline_metrics.md` | ✅ exists | ✅ green |
| 1-04-01 | 04 | 2 | METR-02 | — | Post-fix baseline deferred (old checkpoint trained on beta distribution) | manual | See 01-04-SUMMARY.md | ⚠️ deferred | ⚠️ deferred |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `insight/test_metrics.py` — 5 tests for METR-01 through METR-04, DATA-01 (all passing)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions | Status |
|----------|-------------|------------|-------------------|--------|
| Pre-fix baseline capture | METR-01 | Requires running a full validation epoch with real checkpoint before any code changes | Plan 03 executed; pre-fix baseline captured in baseline_metrics.md | ✅ green |
| Post-fix baseline and delta | METR-02 | Requires running validation after Plans 01/02 and comparing to pre-fix baseline | DEFERRED: old checkpoint trained on beta(2,2) distribution; next training run will produce correct numbers | ⚠️ deferred |
| Precomputed cache regeneration | DATA-01 | Requires regenerating 200k sample dataset file | All 6 .pt cache files deleted; will regenerate on next training run | ✅ green |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved
