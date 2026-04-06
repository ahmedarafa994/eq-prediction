# Phase 4: Q, Type & Frequency Refinement - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Push all four parameter types (gain, freq, Q, type) to their target accuracy thresholds. Fix Q parameterization from sigmoid-to-exp to log-linear STE, balance Hungarian matching cost weights equally, implement metric-gated curriculum transitions, and push filter type accuracy above 95% with class-balanced focal loss. No changes to model encoder architecture or gain prediction mechanism (those are Phase 2/3).

</domain>

<decisions>
## Implementation Decisions

### Q parameterization
- **D-01:** Replace single Linear Q head with deep 3-layer MLP (matching gain head pattern: Linear→ReLU→Linear→ReLU→Linear). Output log(Q) directly, apply STE clamp to [log(0.1), log(10)]. This fixes gradient saturation at extreme Q values caused by sigmoid→exp mapping.
- **D-02:** Q MLP hidden dimensions match gain MLP (hidden_dim → hidden_dim → hidden_dim → 1). Same dropout (0.2) as gain head for consistency.
- **D-03:** Target: Q MAE < 0.2 decades (QP-02). Current ~0.49 decades.

### Hungarian cost balance
- **D-04:** Equalize cost matrix weights to lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0 (currently gain=2.0, freq=0.5, q=0.5). Satisfies FREQ-02 requirement for balanced gain/frequency weighting.
- **D-05:** Type match cost weight (lambda_type_match) stays at 0.5 — type classification doesn't need equal weight in assignment, it benefits more from improved param matching.

### Metric-gated curriculum
- **D-06:** Each curriculum stage defines a dict of metric thresholds that must ALL be met before advancing. Example: stage 2 requires {gain_mae: 3.0, type_acc: 0.6}.
- **D-07:** Threshold values derived from Phase 1-3 baseline metrics (researcher/planner will compute from validation logs). Thresholds stored in config.yaml for tuning without code changes.
- **D-08:** Hard epoch cap per stage prevents infinite stall: if metrics aren't met after N epochs, advance anyway. Default cap = 2x the stage's configured epoch count.

### Type accuracy strategy
- **D-09:** Replace standard cross-entropy with class-balanced focal loss for type classification. Focal loss (gamma=2.0) focuses on hard-to-classify samples. Class weights inverse-proportional to data frequency (peaking 50% gets lower weight, HP/LP 10% get higher).
- **D-10:** Class weights computed from type_weights in config [0.5, 0.15, 0.15, 0.1, 0.1] → inverse normalization. No manual tuning needed.
- **D-11:** Per-type accuracy breakdown reported at each validation step (TYPE-02). Report accuracy for each of the 5 filter types separately.
- **D-12:** Target: overall type accuracy > 95%, minimum per-type accuracy > 80% (TYPE-01).

### Claude's Discretion
- Exact MLP layer initialization (Xavier vs default)
- Focal loss gamma value (start at 2.0, tunable)
- How to compute class weights from type_weights config
- Test structure and file organization

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Q parameterization
- `insight/differentiable_eq.py` lines 570, 763-768 — Current Q head (single Linear), sigmoid→exp mapping, clamp logic
- `insight/differentiable_eq.py` lines 562-567 — Gain MLP pattern (3-layer) to replicate for Q head
- `insight/differentiable_eq.py` line 7 — StraightThroughClamp (STE clamp function)

### Hungarian matching cost
- `insight/loss_multitype.py` lines 47-57 — HungarianBandMatcher.__init__ (lambda_gain, lambda_freq, lambda_q defaults)
- `insight/loss_multitype.py` lines 59-99 — compute_cost_matrix (cost computation with weights)

### Curriculum gating
- `insight/train.py` lines 905-975 — _apply_curriculum_stage (current epoch-based transitions)
- `insight/train.py` lines 310, 548 — Curriculum stage loading, gain_mae_ema (hybrid warmup from Phase 3)
- `insight/conf/config.yaml` lines 79-100 — Curriculum stage definitions

### Type classification
- `insight/differentiable_eq.py` lines 573-597 — Type classification head (type_mel_proj, classification_head)
- `insight/differentiable_eq.py` lines 780-788 — Gumbel-Softmax forward path
- `insight/loss_multitype.py` — Type classification loss (cross-entropy, needs focal replacement)

### Configuration
- `insight/conf/config.yaml` — Loss weights, curriculum stages, data config (type_weights at line 17)

### Prior phase context
- `.planning/phases/01-metrics-data-foundation/01-CONTEXT.md` — Phase 1 decisions (metrics, data)
- `.planning/phases/02-gain-prediction-fix/02-CONTEXT.md` — Phase 2 decisions (gain head)
- `.planning/phases/03-loss-architecture-restructuring/03-CONTEXT.md` — Phase 3 decisions (loss, warmup)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ste_clamp()` (differentiable_eq.py:32) — Already used for gain, can be directly reused for Q log-linear output
- Gain MLP pattern (differentiable_eq.py:562-567) — 3-layer MLP with dropout, template for Q head
- HungarianBandMatcher (loss_multitype.py) — Already parameterized with lambda weights, just need to change defaults
- Gumbel-Softmax infrastructure (differentiable_eq.py:780-788) — Type sampling already works, focal loss only changes the loss function
- gain_mae_ema tracking in MultiTypeEQLoss — Pattern for per-parameter EMA monitoring

### Established Patterns
- Config-driven weights: lambda_* values in config.yaml, passed to loss constructor
- Per-parameter loss logging: dict returned by MultiTypeEQLoss.forward()
- Standalone test scripts: test_*.py with if __name__ == "__main__" blocks

### Integration Points
- MultiTypeEQParameterHead.forward() returns q_raw → needs to return log(Q) instead
- HungarianBandMatcher.compute_cost_matrix() lambda_* params set at construction from config
- train.py _apply_curriculum_stage() needs metric input (currently epoch-only)
- validate() already computes per-param MAEs — extend for per-type accuracy

</code_context>

<specifics>
## Specific Ideas

- The sigmoid→exp Q mapping compresses gradients at extremes. Log-linear output with STE clamp gives identity gradients within bounds, same benefit that helped gain in Phase 2.
- Class-balanced focal loss is standard for imbalanced classification — peaking at 50% sampling rate vs HP/LP at 10% is a 5:1 imbalance that focal loss handles well.
- Metric-gated curriculum prevents the common failure mode where the model advances to harder stages before mastering easier ones.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-q-type-freq-refinement*
*Context gathered: 2026-04-06*
