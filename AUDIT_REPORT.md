# IDSP EQ Estimator — Pipeline Audit Report

**Date:** 2026-04-12  
**Auditor:** Senior ML Infrastructure Engineer & Data Systems Auditor  
**Scope:** Training pipelines, dataset generation pipelines, and associated infrastructure  
**Repository:** `/teamspace/studios/this_studio` (branch: `feat/hierarchical-masking-curriculum`)

---

## Executive Summary

This audit covers a blind parametric EQ estimation ML system comprising:
- **Data pipeline:** On-the-fly synthetic dataset generation (`dataset.py`) + offline MUSDB18 dataset generation (`dataset_pipeline/`)
- **Model pipeline:** Hybrid TCN encoder with multiple backends (`model_tcn.py`), differentiable DSP layer (`differentiable_eq.py`), STFT frontend (`dsp_frontend.py`)
- **Training pipeline:** Custom trainer (`train.py`, 1959 lines) + PyTorch Lightning alternative (`training/lightning_module.py`)
- **Loss pipeline:** Multi-component Hungarian-matched loss (`loss_multitype.py`, 936 lines)

**Overall Assessment:** The codebase demonstrates strong research-grade engineering with sophisticated techniques (Hungarian matching, Gumbel-Softmax, curriculum learning, gradient checkpointing). However, several critical gaps exist in data validation, error recovery, test coverage, security, and production readiness that pose risks to model quality and system stability.

---

## 1. Data Integrity & Quality

### CRITICAL-01: No Input Audio Validation in Synthetic Dataset

**File:** `insight/dataset.py` — `SyntheticEQDataset.__getitem__()`

**Issue:** The synthetic dataset generates audio signals procedurally (noise, sweeps, harmonics, speech-like, percussive) with no validation that the generated signals are:
- Within valid amplitude ranges (could clip)
- Free of NaN/Inf after DSP processing
- Representative of real-world audio distributions

The `__getitem__` method applies random augmentations (volume, time shift, noise) without bounds checking on the final wet_audio output.

**Impact:** Corrupted samples can silently poison training, causing gradient instability or NaN weights. The model may learn from pathological examples.

**Remediation:**
```python
# In __getitem__, after generating wet_audio:
if not torch.isfinite(wet_audio).all():
    # Regenerate or skip this sample
    return self.__getitem__((index + 1) % len(self))
# Clip-safe normalization
wet_audio = torch.clamp(wet_audio, -1.0, 1.0)
```

---

### CRITICAL-02: No Ground Truth Verification for DSP Pipeline Consistency

**File:** `insight/differentiable_eq.py` — `DifferentiableBiquadCascade`

**Issue:** The training loop generates ground-truth frequency responses by calling `self.dsp_cascade(target_gain, target_freq, target_q, filter_type=target_ft)`. However, there is no periodic verification that the DSP cascade's forward path is consistent with the torchaudio `lfilter`-based path used in the offline data generator (`dataset_pipeline/generate_data.py`). These are **two independent implementations** of biquad filtering.

**Impact:** If the two implementations diverge (e.g., due to numerical precision differences), the model trains on inconsistent targets — the loss function's "ground truth" H_mag may not match what the offline generator's biquad cascade would produce.

**Remediation:**
- Add a periodic consistency test: run both implementations on identical inputs and assert outputs match within tolerance (e.g., `atol=1e-5`).
- Unify the biquad coefficient computation: have `generate_data.py` import from `differentiable_eq.py` rather than reimplementing `compute_biquad_coeffs_*`.

---

### HIGH-03: Precompute Cache Has No Staleness Detection

**File:** `insight/train.py` — Trainer initialization; `insight/dataset.py` — `load_precomputed()`, `save_precomputed()`

**Issue:** When `precompute_cache_path` is set, the trainer loads a cached `.pt` file. However:
1. No hash/signature of the dataset generation parameters is stored alongside the cache
2. If `config.yaml` changes (e.g., gain bounds, type weights, signal types), the stale cache is loaded silently
3. The cache includes `PIPELINE_SCHEMA_VERSION` but it is never checked on load

**Impact:** Training proceeds with outdated data distributions, causing silent degradation. The model learns from parameters no longer matching the current config.

**Remediation:**
```python
# In save_precomputed():
cache_metadata = {
    "schema_version": PIPELINE_SCHEMA_VERSION,
    "metadata_hash": compute_metadata_signature(self._generation_params),
    "created_at": utc_now_iso(),
}
# In load_precomputed():
loaded_hash = cache.get("metadata_hash")
expected_hash = compute_metadata_signature(self._generation_params)
if loaded_hash != expected_hash:
    raise RuntimeError("Cache staleness detected — regenerate or clear cache")
```

---

### HIGH-04: MUSDB18 Dataset Has No Schema Validation on Load

**File:** `dataset_musdb.py` (referenced but not fully reviewed; pattern inferred from `generate_data.py`)

**Issue:** The offline data generator (`dataset_pipeline/generate_data.py`) produces `manifest.json` with schema version, but the MUSDB18 dataloader has no validation that loaded samples conform to the expected schema (`schema_version`, required fields in `bands`, etc.).

**Impact:** If the manifest format changes or files are corrupted, the dataloader may silently produce malformed batches.

**Remediation:** Use `pipeline_utils.validate_band_list()` at dataloader initialization to validate a sample of loaded data. Add schema version checking on manifest load.

---

### MEDIUM-05: No Data Lineage Tracking for Generated Samples

**Issue:** While `pipeline_utils.py` has `build_sample_id()` and `compute_metadata_signature()`, these are not consistently used. The synthetic dataset generates samples on-the-fly with no persistent record of which random seeds produced which samples, making debugging and reproducibility difficult.

**Impact:** Impossible to reproduce a specific training run's exact data sequence. Debugging bad batches requires re-generating from scratch.

**Remediation:**
- Log dataset generation seed and config hash at training start
- Add `sample_index` to batch metadata for traceability
- Consider writing a per-epoch data manifest for precomputed datasets

---

### MEDIUM-06: Duration Randomization Can Produce Non-Integer Sample Counts

**File:** `insight/dataset.py` — `__init__` with `duration_range`

**Issue:** When `duration_range` is set, `self.num_samples = int(duration * sample_rate)` uses the base `duration` field, not the randomized duration. This means actual audio length may not match the declared duration, causing shape mismatches in downstream components that rely on `num_samples`.

**Impact:** Silent shape mismatches that could cause crashes or incorrect batching when duration varies.

**Remediation:** Either compute `num_samples` from `duration_range` midpoint, or store the actual per-sample duration and use it consistently.

---

## 2. Pipeline Architecture & Design

### CRITICAL-07: Massive Monolithic Trainer (1959 Lines) Violates Single Responsibility

**File:** `insight/train.py`

**Issue:** The `Trainer` class handles: config loading, dataset creation, precompute caching, model instantiation, optimizer setup (with 3 different optimizer types), curriculum management, training loop, validation loop, metric computation, checkpointing, signal handling, event logging, ONNX export, and NaN recovery. This is a god object.

**Impact:**
- Any change risks unintended side effects across unrelated concerns
- Testing individual components in isolation is nearly impossible
- New team members cannot understand the training flow without reading 2000 lines

**Remediation:** Decompose into:
- `DatasetManager` — dataset creation, precompute, caching
- `OptimizerFactory` — optimizer construction (AdamW, 8-bit, DeepSpeed)
- `CurriculumManager` — stage transitions, parameter injection
- `CheckpointManager` — save/load/recovery
- `TrainingLoop` — epoch iteration, logging
- `ValidationLoop` — metric computation, rendering

---

### HIGH-08: No Graceful Degradation for Missing Optional Dependencies

**File:** `insight/train.py` — imports at top

**Issue:** `bitsandbytes`, `deepspeed`, `transformers`, `timm` are imported with try/except, but failure modes are inconsistent:
- `HAS_BITSANDBYTES` and `HAS_DEEPSPEED` are checked before use
- But `MUSDB18EQDataset` and `LitdataEQDataset` are set to `None` on import failure, then checked only at runtime when `dataset_type` is set — causing a late, confusing `RuntimeError` rather than early failure

**Impact:** Training starts successfully but crashes mid-initialization when a missing dependency is discovered, wasting time.

**Remediation:** Add a `validate_dependencies()` function called in `Trainer.__init__()` that checks all needed dependencies against the config and fails fast with a clear message.

---

### HIGH-09: Dual Training Pipelines Create Confusion and Drift

**Files:** `insight/train.py` (custom Trainer) vs. `training/lightning_module.py` (PyTorch Lightning)

**Issue:** Two complete training implementations exist:
1. Custom `Trainer` class in `insight/train.py` — actively used (referenced by `resume_training.sh`, `launch_wav2vec2.sh`)
2. `EQEstimatorLightning` in `training/lightning_module.py` and `insight/training/lightning_module.py` — exists but unclear if actively maintained

These two implementations have **divergent loss computation, validation logic, and checkpointing behavior**. Bug fixes in one may not propagate to the other.

**Impact:** Maintenance burden; risk of training two different models with different behavior; unclear which is the source of truth.

**Remediation:**
- Designate one as primary (likely the Lightning module for better ecosystem support)
- Migrate unique features from the custom trainer into Lightning callbacks
- Deprecate the custom trainer with a migration guide

---

### MEDIUM-10: Signal Handling for Graceful Shutdown Is Incomplete

**File:** `insight/train.py` — `_register_signal_handlers()`, `_handle_signal()`

**Issue:** Signal handlers are registered for SIGINT and SIGTERM, but:
- Only `self._received_signal` is set — no checkpoint is automatically saved on signal receipt
- The training loop checks `self._received_signal` at epoch boundaries only
- If a signal arrives mid-epoch, the epoch completes (potentially with corrupted state) before the loop exits

**Impact:** Training can lose an entire epoch of work on SIGINT. No guarantee of clean state on termination.

**Remediation:** On signal receipt, save an emergency checkpoint before breaking the loop. Consider saving mid-epoch if the signal is received during validation.

---

### MEDIUM-11: Configuration Has No Schema Validation

**File:** `insight/train.py` — `load_config()`; `conf/config.yaml`

**Issue:** Config is loaded with `yaml.safe_load()` and accessed via `dict.get()` with defaults scattered throughout. No validation ensures:
- Required keys are present
- Values are in valid ranges (e.g., learning rate > 0, epochs > 0)
- Nested structures have expected keys (e.g., `model.encoder.backend` must be one of known values)

**Impact:** Typos in config keys silently fall back to defaults. A misspelled `lambda_spectra` instead of `lambda_spectral` would silently use the default, changing training behavior.

**Remediation:** Use a config validation library (e.g., `pydantic`, `omegaconf` with structured configs) to validate config at load time with clear error messages.

---

### LOW-12: Shell Scripts Lack Proper Error Handling

**Files:** `resume_training.sh`, `launch_wav2vec2.sh`

**Issue:**
- `resume_training.sh`: No `set -euo pipefail`, no error handling, output not redirected to log
- `launch_wav2vec2.sh`: Uses `nohup` with output to log, but no PID file, no health check, no restart logic
- Neither script verifies the working directory exists or that dependencies are available

**Remediation:**
```bash
#!/bin/bash
set -euo pipefail
cd /teamspace/studios/this_studio/insight || { echo "cd failed"; exit 1; }
if [ ! -f train.py ]; then echo "train.py not found"; exit 1; fi
PYTHONUNBUFFERED=1 python train.py 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

---

## 3. Performance & Efficiency

### HIGH-13: Precompute Mode Loads Entire Dataset into RAM

**File:** `insight/dataset.py` — `precompute()`, `load_precomputed()`

**Issue:** When `precompute_mels=True`, the entire dataset is precomputed and stored in memory as lists of tensors. For 200k samples with 128-dim mel spectrograms, this is ~tens of GB of RAM. The `load_precomputed()` call in `Trainer.__init__()` loads everything upfront.

**Impact:** OOM on memory-constrained machines. Limits maximum dataset size. The `config_wav2vec2_unfreeze.yaml` sets `precompute_mels: false` but the default `config.yaml` (AST config) sets `precompute_mels: false` too — so this may not be currently triggered, but the risk remains.

**Remediation:**
- Implement chunked/lazy loading from disk cache (e.g., HDF5, memory-mapped tensors)
- Or use on-the-fly generation with DataLoader `num_workers > 0` (currently set to 0 in configs)

---

### HIGH-14: DataLoader num_workers=0 Is a Severe Bottleneck

**Files:** `conf/config.yaml` (`num_workers: 0`), `insight/conf/config.yaml` (`num_workers: 0`)

**Issue:** Both active configs set `num_workers: 0`, meaning data generation happens on the main process, blocking the GPU. For on-the-fly synthetic data generation (which involves biquad cascade computation), this is a significant bottleneck.

**Impact:** GPU sits idle while CPU generates the next batch. Estimated 30-50% training time wasted.

**Remediation:**
- Set `num_workers: 4` (or `min(cpu_count() - 1, 8)`)
- Ensure `SyntheticEQDataset` is pickle-safe for multiprocessing
- Use the existing `seed_worker` function with `worker_init_fn`

---

### MEDIUM-15: torch.compile Disabled in Production Configs

**Files:** Both configs have `use_torch_compile: false`

**Issue:** `torch.compile()` can provide 20-40% speedup for training loops but is disabled in all active configs. The comment in `model_tcn.py` notes Triton kernel compilation failures, but `torch.compile` is separate from custom Triton kernels.

**Impact:** Unnecessary training time. Each epoch takes longer than needed.

**Remediation:** Enable `use_torch_compile: true` and test. If it fails, investigate the specific compatibility issue (likely a specific op, not torch.compile itself).

---

### MEDIUM-16: Validation Loop Recomputes Expensive Hungarian Matching Every N Steps

**File:** `insight/train.py` — validation loop; `insight/metrics.py` — `compute_eq_metrics()`

**Issue:** Hungarian matching (O(n³) via `scipy.optimize.linear_sum_assignment`) is called on every validation batch. The `val_compute_soft_every_n` parameter controls soft type computation frequency, but Hungarian matching runs every time.

**Impact:** Validation can be slower than necessary, especially with large validation sets.

**Remediation:** Cache Hungarian assignments per validation batch or reduce validation frequency. Consider approximate matching for validation.

---

### LOW-17: Fused Triton Kernels Disabled but Import Overhead Remains

**File:** `insight/model_tcn.py` — top-level imports

**Issue:** The Triton fused kernel module is imported (and fails), but the import attempt and error handling still execute at every module load. The fallback to native PyTorch is correct, but the import block could be cleaner.

**Impact:** Minor startup delay; noisy error logs.

**Remediation:** Remove the Triton import entirely since it's confirmed non-functional, or wrap in a clean feature flag.

---

## 4. Testing & Validation

### CRITICAL-18: No Integration Tests Across Pipeline Stages

**Issue:** Existing tests (`test_multitype_eq.py`, `test_streaming.py`, `test_model.py`, etc.) are all **unit tests** that test individual components in isolation:
- Biquad gradient flow ✓
- Filter frequency response shapes ✓
- Parameter head output shapes ✓
- Streaming consistency ✓

But there are **zero integration tests** that verify:
- A full forward pass: audio → mel → encoder → head → DSP cascade → loss
- That training one step produces a finite loss and gradients
- That the loss function correctly combines all sub-components
- That checkpoint save/load roundtrips correctly

**Impact:** Components can pass all unit tests but fail when assembled. The "catastrophic TCN encoder collapse" mentioned in `model_tcn.py` is exactly this type of integration failure.

**Remediation:** Add `test_integration.py`:
```python
def test_full_forward_pass():
    model = StreamingTCNModel(...)
    loss_fn = MultiTypeEQLoss(...)
    batch = create_test_batch()
    output = model(batch["mel"])
    loss = loss_fn(output, batch)
    assert torch.isfinite(loss)
    loss.backward()
    # Verify all params have gradients
```

---

### HIGH-19: No Tests for Edge Cases in Data Generation

**Issue:** The synthetic dataset generates well-behaved random signals. No tests verify behavior with:
- Extremely short audio (< 100ms)
- Silent audio (all zeros)
- Clipped audio (values > 1.0)
- Audio with NaN/Inf injection
- Batch size of 1
- Number of bands != 5

**Impact:** Edge cases in production (real audio) may cause crashes or silent failures.

**Remediation:** Add parameterized tests for each edge case:
```python
@pytest.mark.parametrize("duration", [0.01, 0.1, 1.0, 10.0])
def test_various_durations(duration):
    ds = SyntheticEQDataset(duration=duration)
    sample = ds[0]
    assert torch.isfinite(sample["wet_audio"]).all()
```

---

### HIGH-20: No Tests for Loss Function Correctness

**File:** `test_new_losses.py` exists but is limited

**Issue:** The 936-line `loss_multitype.py` has complex interactions between sub-losses (param regression, type classification, spectral, typed spectral, group delay, phase, type diversity, perceptual, anti-collapse). There is no test that verifies:
- The combined loss produces finite gradients for all components
- Hungarian matching correctly assigns predictions to targets
- Loss weights are applied as expected
- Edge cases (all-zero predictions, all-same-type) don't produce NaN

**Impact:** Loss function bugs can silently corrupt training. The 2.68 dB gain MAE plateau may be partially caused by loss function issues that tests would have caught.

**Remediation:** Add comprehensive loss tests:
```python
def test_loss_components_finite():
    loss_fn = MultiTypeEQLoss(...)
    pred, target = create_edge_case_batch()
    loss_dict = loss_fn(pred, target)
    for name, val in loss_dict.items():
        assert torch.isfinite(val), f"{name} is not finite"

def test_hungarian_assignment_correctness():
    matcher = HungarianBandMatcher(...)
    # Known input → verify known output
```

---

### MEDIUM-21: No Regression Tests for Training Reproducibility

**Issue:** With `seed: 42` set in configs, training runs should be deterministic. But there is no test that verifies two runs with the same seed produce identical results (at least for a few steps). The `pipeline_utils.set_global_seed()` has a `deterministic` flag, but it's `False` by default.

**Impact:** Non-determinism makes it impossible to verify that code changes don't alter training behavior.

**Remediation:** Add a test that runs 10 training steps twice with the same seed and compares all outputs within floating-point tolerance.

---

### LOW-22: Test Files Scattered Across Root and insight/ Directory

**Issue:** Test files exist at both `/teamspace/studios/this_studio/test_*.py` and `/teamspace/studios/this_studio/insight/test_*.py`. Some are duplicates, some are different. No test runner configuration (pytest.ini, setup.cfg) is present.

**Impact:** Unclear which tests to run. Easy to miss tests during CI.

**Remediation:** Consolidate all tests under `insight/tests/`, add `pytest.ini`, and document how to run the full test suite.

---

## 5. Security & Access Control

### MEDIUM-23: No Input Sanitization for File Paths in Data Pipeline

**File:** `insight/dataset_pipeline/generate_data.py` — `process_file()`

**Issue:** `torchaudio.load(file_path)` loads arbitrary paths without validation. If `input_dir` is set to a user-controlled path, path traversal is possible.

**Impact:** Low risk in current usage (paths are hardcoded in configs), but a concern if the pipeline is exposed via API or shared service.

**Remediation:** Validate that `file_path` is under `input_dir` using `Path.resolve().relative_to()`.

---

### MEDIUM-24: Checkpoint Files Contain Arbitrary Code via pickle

**File:** `insight/train.py` — `torch.load(ckpt_path, weights_only=False)`

**Issue:** Checkpoints are loaded with `weights_only=False`, which allows arbitrary Python object deserialization. A malicious checkpoint file could execute arbitrary code.

**Impact:** If checkpoints are shared from untrusted sources (e.g., downloaded from the internet, shared between collaborators), this is a code execution vulnerability.

**Remediation:**
- Use `weights_only=True` where possible (PyTorch 2.0+)
- Or implement checkpoint signature verification (e.g., SHA-256 hash of checkpoint stored separately)
- Document that checkpoints from untrusted sources should not be loaded

---

### LOW-25: No Rate Limiting or Quota on Data Generation

**File:** `insight/dataset_pipeline/generate_data.py` — multiprocessing pool

**Issue:** The offline data generator spawns `cpu_count() - 1` processes by default, which could exhaust system resources on shared machines.

**Impact:** Could disrupt other workloads on shared compute infrastructure.

**Remediation:** Add `--max_processes` CLI argument and respect cgroup/cpu quota limits.

---

## 6. Documentation & Maintainability

### HIGH-26: Critical Design Decisions Not Documented in Code

**Issue:** Many critical design decisions are only documented in external files (`.planning/PROJECT.md`, `CLAUDE.md`, wiki articles) but not in the code itself:
- Why `hp_lp_gain_target="zero"` is required
- Why the curriculum starts with all types (no peaking-only stage)
- Why `detach_type_for_params` and `detach_type_for_render` exist
- The significance of the 2.68 dB gain MAE plateau

**Impact:** New contributors cannot understand the "why" behind code decisions without reading dozens of external documents.

**Remediation:** Add docstrings to key classes and methods explaining design rationale:
```python
class SyntheticEQDataset:
    """
    Design note: hp_lp_gain_target must be 'zero' because HP/LP filters
    have no meaningful gain parameter — forcing non-zero gains creates
    a degenerate learning signal. See .planning/phases/04/ for analysis.
    """
```

---

### HIGH-27: No Runbook for Common Training Operations

**Issue:** Operations like "resume training from checkpoint", "change curriculum mid-run", "debug NaN loss", "switch encoder backend" are not documented as runbooks. The `resume_training.sh` script is minimal and doesn't handle common scenarios.

**Impact:** Each training interruption requires manual investigation. Knowledge is tribal.

**Remediation:** Create `docs/runbooks/`:
- `resume-training.md`
- `debug-nan-loss.md`
- `switch-encoder.md`
- `interpret-metrics.md`
- `checkpoint-management.md`

---

### MEDIUM-28: Dependency Versions Not Pinned

**File:** `insight/requirements_optimized.txt`

**Issue:** Dependencies use `>=` constraints (e.g., `torch>=2.0.0`, `scipy>=1.10.0`). No `requirements.lock` or `pip freeze` output is committed.

**Impact:** Different environments may install different versions, causing subtle behavioral differences. The Triton kernel failure is likely version-dependent.

**Remediation:** Commit a `requirements.lock` (from `pip freeze`) alongside `requirements_optimized.txt`. Consider using `pip-tools` or `uv` for lockfile management.

---

### LOW-29: Inline Comments Are Sparse in Complex Code Sections

**Issue:** `loss_multitype.py` (936 lines) and `differentiable_eq.py` (771 lines) have minimal inline comments explaining the math behind the Bristow-Johnson formulas, the Hungarian matching cost function, or the Gumbel-Softmax temperature schedule.

**Impact:** Code reviews and debugging require external reference materials.

**Remediation:** Add math-formula comments for key equations, with references to source papers (e.g., "Bristow-Johnson EQ Cookbook, eq. 7").

---

## 7. Monitoring & Alerting

### CRITICAL-30: No Real-Time Pipeline Health Monitoring

**Issue:** The training loop prints metrics to stdout but has no:
- Structured logging (JSON logs for machine parsing)
- Metrics export to monitoring systems (Prometheus, WandB, TensorBoard)
- Alerting on anomalous metrics (sudden loss spike, gradient explosion)
- SLA tracking (epochs per hour, training completion time)

The `WandB` dependency is listed in `requirements_optimized.txt` but is not integrated into the training loop.

**Impact:** Training failures are only discovered post-hoc by checking logs. The 2.68 dB plateau was likely discovered late.

**Remediation:**
- Integrate WandB or TensorBoard logging into the training loop
- Add structured JSON logging with timestamps
- Set up alerts for: loss > threshold, NaN detection, validation metric regression

---

### HIGH-31: NaN Detection Is Reactive, Not Preventive

**File:** `insight/train.py` — `_has_nan_weights()`, `_recover_from_nan()`

**Issue:** NaN detection only checks model weights periodically. There is no:
- Per-batch loss monitoring (a single NaN batch can corrupt weights)
- Gradient norm monitoring (exploding gradients precede NaN weights)
- Early warning system (loss trending upward for N consecutive batches)

**Impact:** By the time NaN weights are detected, many epochs of training may be corrupted. Recovery reloads an old checkpoint, losing significant work.

**Remediation:**
```python
# In train_one_epoch(), after loss.backward():
grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
if not torch.isfinite(grad_norm):
    # Skip this batch, log warning, don't update
    continue
```

---

### HIGH-32: No Validation Metric Trending or Early Stopping Reliability

**File:** `insight/train.py` — early stopping logic

**Issue:** Early stopping uses `early_stopping_patience: 30` epochs, but:
- No minimum epochs before early stopping can trigger
- No smoothing/averaging of validation metrics (single-batch noise can trigger premature stop)
- The `primary_val_score` composite metric weights type_error at 4.0× — a single bad batch can dominate

**Impact:** Training may stop prematurely due to noisy validation, or continue for 30 wasted epochs after convergence.

**Remediation:**
- Add `min_epochs_before_early_stop: 20` to config
- Use exponential moving average of validation metrics
- Log the individual components of `primary_val_score` separately

---

### MEDIUM-33: Event Logging Exists But Is Not Machine-Readable

**File:** `insight/train.py` — `_append_event()`, `events.jsonl`

**Issue:** The training loop writes events to `events.jsonl`, which is good. But:
- Not all significant events are logged (e.g., curriculum stage transitions, NaN recovery attempts)
- No schema for event types
- No tooling to parse or visualize events

**Impact:** The event log is underutilized for debugging and analysis.

**Remediation:**
- Define an event schema (JSON Schema) for all event types
- Add missing events: `curriculum_stage_changed`, `nan_detected`, `nan_recovery_attempted`, `checkpoint_saved`, `gradient_clipped`
- Provide a simple CLI tool or Jupyter notebook to analyze `events.jsonl`

---

### LOW-34: No Performance Profiling Infrastructure

**Issue:** No integration of PyTorch Profiler, Nsight, or other profiling tools. Bottleneck analysis is ad-hoc.

**Impact:** Performance optimizations are guesswork rather than data-driven.

**Remediation:** Add an optional `--profile` flag that runs PyTorch Profiler for the first 10 batches and outputs a trace file.

---

## Summary Matrix

| Priority | Count | Key Risks |
|----------|-------|-----------|
| **Critical** | 6 | Data corruption, no integration tests, no monitoring, monolithic trainer |
| **High** | 10 | Cache staleness, dual pipeline drift, no workers, no loss tests, no runbooks |
| **Medium** | 9 | No lineage, signal handling, config validation, dependency pinning, NaN detection |
| **Low** | 9 | Shell scripts, Triton cleanup, test organization, comments, profiling |

---

## Recommended Action Plan

### Phase 1: Critical Fixes (Week 1-2)
1. **CRITICAL-01**: Add output validation in `SyntheticEQDataset.__getitem__`
2. **CRITICAL-02**: Unify biquad implementations (single source of truth)
3. **CRITICAL-07**: Begin trainer decomposition (start with `CheckpointManager`)
4. **CRITICAL-18**: Add integration tests for full forward pass
5. **CRITICAL-30**: Integrate WandB/TensorBoard logging
6. **CRITICAL-31**: Add per-batch loss/gradient monitoring

### Phase 2: High Priority (Week 3-4)
7. **HIGH-03**: Add cache staleness detection
8. **HIGH-08**: Add dependency validation at startup
9. **HIGH-09**: Decide on single training pipeline; deprecate other
10. **HIGH-14**: Enable `num_workers > 0`
11. **HIGH-19**: Add edge case tests for data generation
12. **HIGH-20**: Add comprehensive loss function tests
13. **HIGH-26**: Add inline documentation for critical decisions
14. **HIGH-27**: Create runbooks for common operations
15. **HIGH-31**: Add gradient norm monitoring and clipping
16. **HIGH-32**: Improve early stopping reliability

### Phase 3: Medium Priority (Week 5-6)
17. **MEDIUM-05**: Add data lineage tracking
18. **MEDIUM-06**: Fix duration/sample count mismatch
19. **MEDIUM-10**: Improve signal handling with emergency checkpoint
20. **MEDIUM-11**: Add config schema validation
21. **MEDIUM-15**: Test and enable `torch.compile`
22. **MEDIUM-21**: Add reproducibility regression tests
23. **MEDIUM-23/24**: Address security concerns (path sanitization, checkpoint loading)
24. **MEDIUM-28**: Pin dependency versions
25. **MEDIUM-33**: Enhance event logging

### Phase 4: Low Priority (Ongoing)
26. **LOW-12**: Harden shell scripts
27. **LOW-17**: Remove dead Triton import code
28. **LOW-22**: Consolidate test files
29. **LOW-25**: Add process limits to data generator
30. **LOW-29**: Add math comments to complex code
31. **LOW-34**: Add profiling infrastructure

---

*End of Audit Report.*
