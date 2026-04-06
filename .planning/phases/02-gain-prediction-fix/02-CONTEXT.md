# Phase 2: Gain Prediction Fix - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the gain prediction mechanism to produce accurate estimates with full gradient flow. Remove the mel-residual auxiliary gain path that injects noise, ensure STE clamp is used throughout, and verify gain MAE drops below 3 dB. No changes to loss architecture or other parameter heads — those belong to Phase 3 and Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Mel-residual removal
- **D-01:** Full removal from BOTH EQParameterHead (line 509) and MultiTypeEQParameterHead (line 531). Delete all mel-residual gain readout code, attention weights, blending logic. Gain comes purely from the trunk embedding MLP.
- **D-02:** After removal, remove the related config keys (if any) for mel-residual blend weight.

### Gain activation
- **D-03:** STE clamp is already in use for gain in both heads — verify it's the ONLY activation (no tanh remnants). The requirement to "replace tanh with STE clamp" appears already satisfied but needs verification across all codepaths.
- **D-04:** Gain range stays ±24 dB with ste_clamp bounds (current setting).

### Streaming compatibility
- **D-05:** All changes must preserve streaming inference. Mel-residual removal should not affect init_streaming() or process_frame() — those operate on the TCN encoder output, not the gain head internals.

### Verification
- **D-06:** Run baseline validation before changes (from Phase 1 metrics) and after changes to measure gain MAE improvement.
- **D-07:** Target: gain MAE < 3 dB with matched metrics (down from ~6 dB baseline).

### Claude's Discretion
- Exact MLP architecture for the primary gain path (layer sizes, activation)
- Test structure and file organization
- How to handle any config keys that become dead after mel-residual removal

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Gain prediction code
- `insight/differentiable_eq.py` — StraightThroughClamp (line 7), ste_clamp (line 32), EQParameterHead (line 509), MultiTypeEQParameterHead (line 531). Both heads contain mel-residual auxiliary gain paths to remove.
- `insight/model_tcn.py` — StreamingTCNModel that uses MultiTypeEQParameterHead. init_streaming() and process_frame() methods must remain compatible.

### Loss and metrics
- `insight/loss_multitype.py` — HungarianBandMatcher and MultiTypeEQLoss. Gain regression loss components.
- `insight/train.py` — validate() method for gain MAE measurement. Trains the model.

### Data
- `insight/dataset.py` — SyntheticEQDataset. Phase 1 fixed gain distribution to uniform.

### Testing patterns
- `insight/test_model.py` — Test pattern to follow
- `insight/test_multitype_eq.py` — Multi-type filter tests

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ste_clamp()` already exists and is used for gain clamping in both heads
- `HungarianBandMatcher` for matched gain MAE evaluation
- Phase 1 metrics infrastructure (per-parameter MAE, component logging)

### Established Patterns
- Standalone test scripts with `if __name__ == "__main__":` blocks
- Config via `conf/config.yaml`
- Two parameter heads: EQParameterHead (legacy CNN) and MultiTypeEQParameterHead (primary TCN)

### Integration Points
- MultiTypeEQParameterHead.forward() returns gain_db, freq, q, filter_type_probs, H_mag
- train.py consumes these outputs for loss computation and validation metrics
- process_frame() in StreamingTCNModel calls the head on single-frame embeddings

</code_context>

<specifics>
## Specific Ideas

- The mel-residual path was originally intended to help the model "see" the spectral shape for gain prediction, but diagnostics showed it injects noise rather than signal (gain MAE got worse, not better)
- Full removal is the clean approach — don't leave dead code paths

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-gain-prediction-fix*
*Context gathered: 2026-04-06*
