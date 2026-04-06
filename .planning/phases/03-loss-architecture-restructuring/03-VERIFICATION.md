---
phase: 03-loss-architecture-restructuring
verified: 2026-04-06T00:00:00Z
status: human_needed
score: 6/6 must-haves verified
human_verification:
  - test: "Run a short training session (3-5 epochs) and verify warmup gate activates correctly"
    expected: "Epochs 1-4 show gain-only warmup in logs; epoch 5+ shows freq/Q active; spectral activates at epoch 7+; gain_mae_ema threshold gates observed"
    why_human: "Dynamic training behavior with GPU execution cannot be verified by static code analysis"
  - test: "Monitor training logs for loss component competition"
    expected: "During warmup, only loss_gain is non-zero. After warmup, loss_freq and loss_q activate. Type loss activates 1 epoch later. Spectral loss activates 2 epochs later."
    why_human: "Requires running multi-epoch training and observing dynamic log output"
---

# Phase 3: Loss Architecture Restructuring Verification Report

**Phase Goal:** The loss function directs gradient signal to gain regression first, then progressively activates spectral and other losses, preventing loss component competition
**Verified:** 2026-04-06
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Loss weights for gain, freq, and Q are independently tunable (not a single combined param loss) | VERIFIED | `loss_multitype.py`: `lambda_param=0.0` in config, independent `lambda_gain=2.5`, `lambda_freq=1.5`, `lambda_q=1.5` in both `__init__` and `forward()` total_loss computation (lines 546-557). Test `test_independent_weights` passes. |
| 2 | Training begins with a gain-only warmup period before spectral losses activate | VERIFIED | `loss_multitype.py` lines 372-386: hybrid warmup gate (`past_epoch_threshold and gain_converged or past_hard_cap`). `is_freq_q_active = not is_warmup`. `config.yaml`: `warmup_epochs: 5`. Test `test_warmup_gating` passes. |
| 3 | Gain regression uses log-cosh loss instead of Huber | VERIFIED | `loss_multitype.py` line 458: `loss_gain = log_cosh_loss(pred_gain, matched_gain).mean()`. `log_cosh_loss()` defined at line 23. Test `test_log_cosh_wired` verifies numerical correctness (1.325, 2.308). |
| 4 | Dual forward path exists -- hard argmax types for param regression loss, soft Gumbel for spectral loss | VERIFIED | `loss_multitype.py` forward signature: `pred_H_mag_soft`, `pred_H_mag_hard` (separate params). `hmag_loss` uses `pred_H_mag_hard.float().detach()` (line 504). `spectral_loss` uses `pred_H_mag_soft` (line 514). Test `test_dual_hmag_signature` passes. |
| 5 | Audio-domain reconstruction loss provides additional training signal | VERIFIED | `loss_multitype.py` lines 509-519: `spectral_loss` uses `F.l1_loss(torch.log(pred_spec_safe), torch.log(target_spec_safe))` from `pred_H_mag_soft`. Config `lambda_spectral: 0.1`. Test `test_spectral_reconstruction_fires` verifies numerical correctness (log(2.0) ~ 0.693). |
| 6 | Gumbel-Softmax type probs are detached from gain gradient path during warmup | VERIFIED | `loss_multitype.py` lines 436-439: `pred_type_logits_for_match = pred_type_logits.detach()` when `is_warmup`. Used in matcher call at line 454. Test `test_hybrid_warmup_gate` verifies epoch+gain_mae_ema hybrid gate with 15-epoch hard cap. |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `insight/loss_multitype.py` | MultiTypeEQLoss with dual H_mag, log-cosh, hybrid warmup gate, spectral L1 | VERIFIED | 562 lines. All features implemented and wired. `pred_H_mag_hard.detach()` at line 504. Spectral L1 at lines 513-516. Gumbel detach at lines 436-439. |
| `insight/test_loss_architecture.py` | 9 automated tests for LOSS-01 through LOSS-06 and DATA-02 | VERIFIED | 269 lines. All 9 tests pass. |
| `insight/conf/config.yaml` | `warmup_epochs: 5` in loss section | VERIFIED | Line 57. Parsed correctly by YAML. |
| `insight/train.py` | Passes dual H_mag, active_band_mask, calls update_gain_mae | VERIFIED | Lines 428-429: `pred_H_mag_soft`, `pred_H_mag_hard` from model output. Lines 409-411: `active_band_mask` extraction. Lines 549-552: `update_gain_mae` call. |
| `insight/dataset.py` | Returns active_band_mask in samples and collate_fn | VERIFIED | Lines 292, 326: `active_band_mask` in sample dicts. Lines 378-388: stacking in `collate_fn`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `model_tcn.py` output | `train.py` | `output["H_mag"]` as `pred_H_mag_soft`, `output["H_mag_hard"]` as `pred_H_mag_hard` | WIRED | train.py lines 428-429 |
| `train.py` | `loss_multitype.py` | `criterion(pred_gain, ..., pred_H_mag_soft, pred_H_mag_hard, ...)` | WIRED | train.py lines 441-457 |
| `loss_multitype.py` | `pred_H_mag_hard.detach()` | hmag_loss computation | WIRED | loss_multitype.py line 504 |
| `loss_multitype.py` | `pred_H_mag_soft` | spectral_loss computation | WIRED | loss_multitype.py line 514 |
| `loss_multitype.py` | `pred_type_logits.detach()` | Hungarian matcher during warmup | WIRED | loss_multitype.py lines 436-439, 454 |
| `train.py` | `loss_multitype.py` | `criterion.update_gain_mae(batch_gain_mae)` | WIRED | train.py lines 549-552 |
| `dataset.py` | `train.py` | `batch["active_band_mask"]` | WIRED | dataset.py line 388, train.py lines 409-411 |
| `train.py` | `loss_multitype.py` | `active_band_mask=active_band_mask` in criterion call | WIRED | train.py line 455 (training), line 691 (validation) |
| `conf/config.yaml` | `train.py` | `loss_cfg.get("warmup_epochs", 5)` | WIRED | config.yaml line 57, train.py line 227 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `loss_multitype.py` forward() | `loss_gain` | `log_cosh_loss(pred_gain, matched_gain)` | Yes -- computed from actual predictions via Hungarian matching | FLOWING |
| `loss_multitype.py` forward() | `loss_freq`, `loss_q` | Huber on log-space predictions | Yes -- zeroed during warmup, active after | FLOWING |
| `loss_multitype.py` forward() | `hmag_loss` | `F.l1_loss(log(pred_H_mag_hard.detach()), log(target_H_mag))` | Yes -- detached hard path vs ground truth | FLOWING |
| `loss_multitype.py` forward() | `spectral_loss` | `F.l1_loss(log(pred_H_mag_soft), log(target_H_mag))` | Yes -- soft path vs ground truth, gated by `is_spectral_active` | FLOWING |
| `loss_multitype.py` forward() | `activity_loss` | `(pred_gain * inactive_mask.float()).abs().mean()` | Yes -- computed from real mask and predictions | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 9 loss architecture tests pass | `cd insight && python test_loss_architecture.py` | 9 tests: 9 passed, 0 failed | PASS |
| Phase 2 regression: gain head tests pass | `cd insight && python test_gain_head.py` | All 9 gain head tests passed | PASS |
| Phase 2 regression: streaming tests pass | `cd insight && python test_streaming.py` | All streaming TCN tests passed | PASS |
| Config YAML parses without error | `python -c "import yaml; yaml.safe_load(open('conf/config.yaml'))"` | No error, `warmup_epochs=5`, `lambda_spectral=0.1` | PASS |
| Loss forward() signature has dual H_mag params | `python -c "from loss_multitype import ...; inspect.signature(MultiTypeEQLoss.forward)"` | `pred_H_mag_soft` and `pred_H_mag_hard` in signature | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| LOSS-01 | 03-01 | Separate loss weights for gain, freq, Q | SATISFIED | `lambda_param=0.0`, independent `lambda_gain/freq/q` in config and loss computation. Test passes. |
| LOSS-02 | 03-01 | Loss phasing -- gain-only warmup | SATISFIED | Hybrid warmup gate with `warmup_epochs: 5`. `loss_freq` and `loss_q` zeroed during warmup. Test passes. |
| LOSS-03 | 03-01 | Log-cosh for gain regression | SATISFIED | `log_cosh_loss()` function defined and called at line 458. Test verifies numerical correctness. |
| LOSS-04 | 03-01 | Dual forward path (hard/soft H_mag) | SATISFIED | `pred_H_mag_soft` and `pred_H_mag_hard` in forward signature. `detach()` on hard path for hmag_loss. Test passes. |
| LOSS-05 | 03-02 | Audio-domain reconstruction loss | SATISFIED | Spectral L1 between `pred_H_mag_soft` and `target_H_mag`. `lambda_spectral: 0.1` in config. Test passes. |
| LOSS-06 | 03-02 | Per-band activity weighting | SATISFIED | `active_band_mask` wired from dataset through train.py to loss. Activity loss penalizes inactive band gains. Test passes. |
| DATA-02 | 03-01 | Gumbel-Softmax detach during warmup | SATISFIED | `pred_type_logits_for_match = pred_type_logits.detach()` when `is_warmup`. Hybrid warmup gate with 15-epoch hard cap. Test passes. |

No orphaned requirements found -- all 7 requirement IDs mapped to Phase 3 in REQUIREMENTS.md are addressed across the two plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `conf/config.yaml` | 46 + 58 | Duplicate `lambda_spectral` key (0.0 and 0.1) | WARNING | YAML takes last value (0.1), so functionally correct. But the stale `lambda_spectral: 0.0` at line 46 is confusing and should be removed. |

No TODO/FIXME/PLACEHOLDER comments found in `loss_multitype.py`, `train.py`, or `dataset.py`. No empty implementations or stub patterns detected.

### Human Verification Required

### 1. Training Warmup Gate Activation

**Test:** Run a short training session (3-5 epochs) with `conf/config.yaml` and observe epoch logs.
**Expected:** Epochs 1-4 show "GAIN-ONLY WARMUP" in console output. Epoch 5+ shows "Gain + Freq + Q active". Epoch 6+ shows "+ Type loss activated". Epoch 7+ shows "All losses active". The hybrid gate means if `gain_mae_ema` stays above 2.5, warmup continues past epoch 5 until either gain converges or hard cap at epoch 15.
**Why human:** Dynamic training behavior requires running the GPU training loop and observing real-time log output. Static code analysis confirms the logic is correct but cannot verify it triggers at the right epochs.

### 2. Loss Component Competition Absence

**Test:** During training, monitor per-component loss values in training logs.
**Expected:** During warmup epochs, `loss_freq`, `loss_q`, `type_loss` should be exactly 0.0 (not just small). After warmup, they should have non-zero values that decrease over time. `spectral_loss` should remain 0.0 until 2 epochs after warmup ends.
**Why human:** Requires multi-epoch training execution with live GPU computation.

---

_Verified: 2026-04-06T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
