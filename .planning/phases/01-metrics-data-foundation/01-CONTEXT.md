# Phase 1: Metrics & Data Foundation - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix validation measurement to produce trustworthy Hungarian-matched baseline metrics (per-parameter MAE, loss component logging, gradient norm monitoring), and balance training data to use uniform gain distribution across the full range. No model architecture changes — measurement and data only.

</domain>

<decisions>
## Implementation Decisions

### Validation metric baseline
- **D-01:** Run a baseline validation epoch with the current (buggy) code BEFORE making changes. Capture matched vs raw MAE for gain, freq, Q, and type accuracy. This becomes the "before" measurement.
- **D-02:** Log all loss components (param regression, spectral, band activity, frequency spread, type classification) during validation, not just training. Currently validation only logs total loss.
- **D-03:** Add per-parameter-group gradient norm monitoring (gain, freq, Q, type) during training. Current code tracks overall gradient norm but not broken down by parameter group.
- **D-04:** Hungarian matching already exists in validate() and is used for MAE computation — the matching logic itself is correct. The issue is the unmatched "raw" MAE was historically reported alongside it, causing confusion about the true baseline.

### Gain distribution
- **D-05:** Replace `np.random.beta(2, 2)` gain sampling with true uniform distribution across `gain_range`. The beta(2,2) concentrates near the center and undersamples extreme gains.
- **D-06:** Fix HP/LP filter gain sampling — currently uses `random.uniform(-1, 1)` (lines 183, 187 in dataset.py) which gives tiny gains. Should use the full `gain_range` for all filter types that have meaningful gain.
- **D-07:** Regenerate precomputed dataset cache after distribution fix. Old cache (`dataset_musdb_200k.pt`) will have the biased distribution.

### Test/regression strategy
- **D-08:** Write `test_metrics.py` as a standalone test (following existing `test_*.py` pattern) that verifies: Hungarian matching produces correct assignments on synthetic known inputs, per-parameter MAE computation is accurate, loss component logging outputs all expected keys.
- **D-09:** After fixes, re-run the baseline validation and compare matched MAE to the pre-fix baseline. Document the delta.

### Claude's Discretion
- Exact format of validation log output (print formatting)
- How to structure gradient norm monitoring code (inline vs helper function)
- Whether to keep or remove the "raw" (unmatched) MAE metric alongside the matched one

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Metrics and validation
- `insight/loss_multitype.py` — HungarianBandMatcher class (cost matrix, matching logic), MultiTypeEQLoss (loss components)
- `insight/train.py` lines 589-710 — validate() method, metric computation, logging
- `insight/train.py` lines 480-500 — gradient norm monitoring (current implementation to extend)

### Data distribution
- `insight/dataset.py` lines 214-220 — _sample_gain() method (beta distribution to replace)
- `insight/dataset.py` lines 154-212 — _sample_multitype_params() method (HP/LP gain fix at lines 183, 187)

### Testing patterns
- `insight/test_model.py` — existing test pattern to follow for test_metrics.py
- `insight/test_multitype_eq.py` — test pattern for multi-type filter validation

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `HungarianBandMatcher` in loss_multitype.py: Already computes cost matrix and solves assignment. Used in validate() but NOT in training loss (training uses a separate matcher instance).
- `validate()` in train.py: Already has the metric computation structure — gain_mae, freq_mae, q_mae, type_acc with Hungarian matching. Needs component logging added.
- Gradient clipping infrastructure in train.py: Already clips and checks for NaN gradients. Per-parameter-group monitoring extends this.

### Established Patterns
- Standalone test scripts: `test_*.py` files with `if __name__ == "__main__":` blocks, no pytest
- Config via YAML: All hyperparameters in `conf/config.yaml`
- Print-based logging: No logging framework, just formatted print statements

### Integration Points
- `dataset.py` __getitem__ returns gain/freq/q/filter_type tensors — distribution changes happen in _sample_gain() and _sample_multitype_params()
- `train.py` Trainer.__init__ creates val_loader from dataset — precomputed cache must be regenerated
- Validation metric dict returned by validate() is used for early stopping and checkpoint selection

</code_context>

<specifics>
## Specific Ideas

- The "6 dB gain MAE" from prior runs was computed without Hungarian matching. The matched baseline will be lower (matching removes the permutation penalty). This is expected — the new number is the trustworthy one.
- The beta(2,2) distribution was described as "near-uniform" in the docstring but actually concentrates ~40% of samples in the middle third of the gain range, starving the model of extreme gain examples it needs to learn.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---
*Phase: 01-metrics-data-foundation*
*Context gathered: 2026-04-05*
