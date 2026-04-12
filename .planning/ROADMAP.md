# Roadmap: IDSP EQ Estimator — Accuracy Improvement

## Milestones

- ✅ **v1.0 Accuracy Improvement** — Phases 1-5 (shipped 2026-04-06)
- ◆ **v1.1 Backbone Fine-tuning & Accuracy Push** — Phases 6-7 (active)

## Phases

<details>
<summary>✅ v1.0 Accuracy Improvement (Phases 1-5) — SHIPPED 2026-04-06</summary>

- [x] Phase 1: Metrics & Data Foundation (4/4 plans) — completed 2026-04-06
- [x] Phase 2: Gain Prediction Fix (2/2 plans) — completed 2026-04-06
- [x] Phase 3: Loss Architecture Restructuring (2/2 plans) — completed 2026-04-06
- [x] Phase 4: Q, Type & Frequency Refinement (2/2 plans) — completed 2026-04-06
- [x] Phase 5: Inference Refinement & Confidence (2/2 plans) — completed 2026-04-06

</details>

### v1.1 Backbone Fine-tuning & Accuracy Push

- [ ] **Phase 6: Backbone Unfreezing** — Launch end-to-end wav2vec2 fine-tuning
      **Plans:** 2 plans
      - [ ] 06-01-PLAN.md — Pre-flight verification and config calibration for resume
      - [ ] 06-02-PLAN.md — Launch fine-tuning and monitor backbone unfreeze transition
- [ ] **Phase 7: Training Strategy & Accuracy Push** — LLRD, EMA, loss rebalancing to hit targets

---

## Phase 6: Backbone Unfreezing

**Goal:** Launch wav2vec2 backbone fine-tuning with existing code, verify gradient flow and VRAM stability

**Requirements:** BACK-01, BACK-02, BACK-03

**Success Criteria:**
1. Training runs stably for 10+ epochs with unfrozen backbone (no OOM, no NaN)
2. Backbone gradients flow to all 211 parameter groups
3. Gain MAE improves from 2.68 dB baseline within first 10 epochs
4. Freeze→unfreeze transition works cleanly mid-training

**Context:**
- Code already implemented in model_tcn.py (freeze/unfreeze methods, HF gradient checkpointing)
- train.py updated with 3-group optimizer and _rebuild_optimizer_if_needed()
- Config ready: conf/config_wav2vec2_unfreeze.yaml (freeze_epochs=2, backbone_lr=1e-5, batch_size=512, grad_accum=6)
- Current run (frozen backbone) completing epoch 37-40, will resume from best checkpoint

---

## Phase 7: Training Strategy & Accuracy Push

**Goal:** Add advanced training techniques and hit intermediate accuracy targets

**Requirements:** TRNS-01, TRNS-02, TRNS-03, ACCU-01, ACCU-02, ACCU-03

**Success Criteria:**
1. Layer-wise LR decay applied across wav2vec2 transformer layers
2. EMA model produces stable, improved validation metrics
3. Loss weights rebalanced based on gradient magnitude analysis after unfreezing
4. Gain MAE < 2.0 dB (from 2.68 dB plateau)
5. Type classification accuracy > 60% (from 46.9%)
6. Primary validation score < 4.5 (from 5.26)

**Context:**
- Depends on Phase 6 establishing stable backbone fine-tuning
- LLRD: lower layers get smaller LR (e.g., layer 0 at 0.1× base, layer 11 at 1.0× base)
- EMA: exponential moving average of weights for evaluation stability
- Loss rebalancing: gradient magnitudes will shift when backbone unfreezes, requiring weight adjustment

---

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Metrics & Data Foundation | v1.0 | 4/4 | Complete | 2026-04-06 |
| 2. Gain Prediction Fix | v1.0 | 2/2 | Complete | 2026-04-06 |
| 3. Loss Architecture Restructuring | v1.0 | 2/2 | Complete | 2026-04-06 |
| 4. Q, Type & Frequency Refinement | v1.0 | 2/2 | Complete | 2026-04-06 |
| 5. Inference Refinement & Confidence | v1.0 | 2/2 | Complete | 2026-04-06 |
| 6. Backbone Unfreezing | v1.1 | 0/2 | Pending | — |
| 7. Training Strategy & Accuracy Push | v1.1 | 0/0 | Pending | — |
