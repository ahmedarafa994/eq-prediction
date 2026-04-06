# Phase 2: Gain Prediction Fix - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 02-gain-prediction-fix
**Areas discussed:** mel-residual removal scope

---

## Mel-residual removal scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full removal (Recommended) | Remove mel-residual from BOTH codepaths (EQParameterHead + MultiTypeEQParameterHead). Cleanest approach. | ✓ |
| Config-based disable | Zero out blending weight via config. Reversible but leaves dead code. | |
| Primary model only | Remove only from MultiTypeEQParameterHead. Leave legacy head unchanged. | |

**User's choice:** Full removal — delete all mel-residual gain readout code from both heads.
**Notes:** User confirmed full cleanup is the right approach. No reason to keep dead code paths.

---

## Claude's Discretion

- Gain activation strategy (verify STE clamp usage, gain range bounds)
- Whether to add gain-only diagnostic mode to train.py
- Exact test structure and file organization
- MLP architecture details for primary gain path

## Deferred Ideas

None — discussion stayed within phase scope.
