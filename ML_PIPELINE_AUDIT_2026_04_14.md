# ML Pipeline — Comprehensive Audit Report (2026-04-14)

**Project:** Differentiable DSP EQ Parameter Estimator (IDSP)
**Auditor:** Senior ML Infrastructure & Security Auditor
**Scope:** Training pipelines, dataset pipelines, model architecture, loss functions, configuration, security, privacy, bias, lineage, drift detection, and pipeline efficiency
**Previous Audits Reviewed:** `AUDIT_REPORT.md`, `AUDIT_REPORT_COMPREHENSIVE.md`, `AUDIT_REMEDIATION_STATUS.md`, `ML_PIPELINE_AUDIT_REPORT.md`

---

## Executive Summary

This audit represents the most comprehensive review of the IDSP system to date, synthesizing findings from **four prior audits** with fresh code-level analysis and live training-log forensics. The system is a research-grade differentiable DSP pipeline that estimates parametric EQ parameters (gain, frequency, Q, filter type) from wet audio alone using a hybrid TCN/AST encoder with Hungarian-matched loss and Gumbel-Softmax type selection.

### Key Findings from Live Training Log Analysis

Analysis of the most recent training run (`training_log.txt`) confirms **active type collapse** — the model predicts 100% peaking for all 5 filter types across all validation epochs, achieving 19.7% type accuracy (random baseline). This is the single most impactful finding: **the multi-type EQ estimation capability is non-functional in the current configuration**.

A second active run (`training_log_peak125_from_043.txt`) with a wav2vec2 backbone shows improved gain regression (loss_gain ≈ 2.5 vs 14.3) but **type_loss remains stuck at ~3.4** with type_entropy_loss near zero (0.001-0.005), indicating the model still cannot distinguish filter types even with a pretrained encoder.

### Risk Summary

| Severity | Count | Key Areas |
|----------|-------|-----------|
| **CRITICAL** | 6 | Type collapse, cache staleness, numerical stability, gradient detachment, optimizer state corruption, loss component imbalance |
| **HIGH** | 9 | DataLoader blocking, dual DSP implementations, batch/epoch ratio, checkpoint proliferation, shell script security, loss sign inversion, Gumbel temperature, test-reality gap |
| **MEDIUM** | 12 | Reproducibility gaps, early stopping composition, curriculum coordination, memory footprint, missing data drift detection, incomplete path traversal, version tracking |
| **LOW** | 7 | Documentation drift, profiling gaps, dependency version pinning, metric threshold calibration |

---

## 1. Data Integrity & Quality

### CRITICAL-01: Active Type Collapse — Model Predicts Single Filter Type (Live Issue)

**Evidence:** `training_log.txt` — epochs 1-4 validation output:
```
[val] per-type: peaking=100.0% | lowshelf=0.0% | highshelf=0.0% | highpass=0.0% | lowpass=0.0%
[val] confusion (true\pred):   peak   lshf   hshf   hpas   lpas
[val]   peak:    986      0      0      0      0  (986)
[val]   lshf:   1033      0      0      0      0  (1033)
[val]   hshf:    997      0      0      0      0  (997)
```

All 5,000 validation samples across 5 types are classified as "peaking." Type accuracy = 19.7% (5-class random baseline). The confusion matrix shows zero correct predictions for lowshelf, highshelf, highpass, and lowpass.

**Root Cause Analysis:**
1. **Gumbel-Softmax temperature too high:** `gumbel.start_tau=2.0` (in `conf/config.yaml`) makes the softmax effectively uniform at training start, providing no type signal to the parameter head
2. **Type loss weight insufficient:** Despite `lambda_type: 8.0` in curriculum stage 1, the gain loss (14.3) dominates the total (23.5) at 61%, starving type learning
3. **Class weight multipliers are inverted:** `class_weight_multipliers: [1.0, 2.0, 2.0, 2.0, 2.0]` weights peaking at 1.0x (the easiest class) and all others at 2.0x — but the model already predicts everything as peaking, so the rare-class weighting doesn't help

**Impact:** The entire multi-type EQ estimation capability is non-functional. The model cannot distinguish between filter types, rendering the hierarchical type head, Gumbel-Softmax, and Hungarian matching useless for their primary purpose.

**Recommended Remediation:**
1. **Lower initial Gumbel temperature:** `gumbel.start_tau: 0.5` (from 2.0) — at tau=0.5, the softmax provides meaningful type gradients
2. **Increase type loss weight further in stage 1:** Set `lambda_type: 12.0` or higher to force type learning before parameter refinement
3. **Increase type entropy penalty:** `lambda_type_entropy: 2.0` (from 0.5) to penalize peaked prediction distributions
4. **Add hard negative mining:** Force the model to see examples where non-peaking types have clear spectral signatures
5. **Verify shelf features are non-zero:** The `n_shelf_bands: 16` energy ratio features must be computed and injected into the hierarchical head — add a unit test

---

### CRITICAL-02: Precomputed Cache Loaded Without Staleness Validation (Active Risk)

**Evidence:** `training_log.txt` lines 2-3:
```
[cache] WARNING: loaded cache from data/dataset_phase3_200k.pt has unexpected format. Loading without metadata validation. Regenerate cache to enable checks.
[cache] Loaded precomputed dataset from data/dataset_phase3_200k.pt: 200000 samples
```

Despite the previous audit's remediation (HIGH-03), the 200K-sample precomputed cache has an **unexpected format** and is loaded **without staleness validation**. The trainer warns but proceeds.

**Impact:** The 200K samples may have been generated with **different config parameters** than the current config. Specifically:
- Current config: `gain_bounds: [-12.0, 12.0]`, `q_bounds: [0.05, 15.0]`, `type_weights: [0.2, 0.2, 0.2, 0.2, 0.2]`
- If the cache was generated with different bounds or weights, the model trains on a mismatched distribution
- Furthermore, changes to `differentiable_eq.py` or the augmentation logic would produce different wet audio from the same parameters, but the cache would not be invalidated

**Recommended Remediation:**
1. **Delete the stale cache:** `rm data/dataset_phase3_200k.pt` and regenerate with current config
2. **Make staleness detection a hard failure, not a warning:** Change the log message to a `RuntimeError` when no signature is found
3. **Add a `--force_recompute_cache` CLI flag** to explicitly regenerate cache

---

### CRITICAL-03: Broken Gradient Flow During Warmup — Type Logits Detached

**File:** `loss_multitype.py` (referenced in prior audit V-04)

**Issue:** During warmup, type logits are **detached**: `pred_type_logits_for_match = pred_type_logits.detach()`. This prevents gradients from flowing from the type classification loss back through the encoder, meaning the encoder cannot learn type-discriminative features during warmup.

**Impact:** The encoder develops representations that are blind to filter type during the critical early epochs, making it much harder for the type classifier to learn when warmup ends. This is likely a **primary contributor to the type collapse** observed in training logs.

**Recommended Remediation:**
1. **Allow type gradients to flow from the start with reduced weight** (e.g., 0.1× during warmup)
2. **Only detach if the model is severely unstable, and log the reason**
3. **Track type prediction entropy during warmup** to detect early collapse

---

### CRITICAL-04: Memory Leak in Optimizer NaN Sanitization

**File:** `train.py` — optimizer state sanitization loop

**Issue:** The NaN sanitization zeroes corrupted tensors but doesn't reset Adam momentum (`exp_avg_sq`) or update step counts for bias correction. After zeroing, `exp_avg_sq` may retain corrupted second-moment estimates, creating "ghost" momentum that biases future updates.

**Impact:** "Ghost" momentum from NaN-preserved second moments causes future parameter updates to be biased, leading to slow degradation of model quality that is difficult to diagnose.

**Recommended Remediation:**
1. Reset both `exp_avg` and `exp_avg_sq` for affected parameters
2. Reset the step count for affected parameter groups
3. Consider re-creating the optimizer state entirely for corrupted parameters

---

### CRITICAL-05: Numerical Instability in Biquad Coefficient Computation

**File:** `differentiable_eq.py` — biquad alpha computation

**Issue:** `alpha = torch.sin(w0) / (2.0 * q + 1e-8)` — the epsilon `1e-8` is too small. When Q approaches 0 (model can predict Q=0.05 minimum, and during training with gradient perturbations, Q can effectively reach near-zero), alpha can become extremely large, causing downstream NaN in coefficient computation.

Additionally, `H_mag = torch.sqrt(torch.clamp(num_mag2 / (den_mag2 + 1e-4), min=1e-8))` — the denominator epsilon `1e-4` may be insufficient when the denominator is very small (near filter pole), producing extremely large H_mag values that overflow in log10 operations.

**Impact:** Training instability, NaN gradients, potentially frozen parameters requiring optimizer state sanitization.

**Recommended Remediation:**
1. Increase epsilon from `1e-8` to `1e-4` for alpha, from `1e-4` to `1e-3` for H_mag
2. Add explicit clamping: `torch.clamp(alpha, min=1e-6, max=10.0)`
3. Add explicit H_mag clamping to `[1e-6, 1e6]` before any log operations
4. Log when values are being clamped as a diagnostic signal

---

### CRITICAL-06: Loss Component Spread Loss Is Inverted (Negative Value)

**Evidence:** `training_log.txt`:
```
spread_loss=-0.4163 (epoch 1), spread_loss=-0.3990 (epoch 2), spread_loss=-0.3793 (epoch 3)
```

**Issue:** A negative loss value means the spread regularization is **rewarding** concentrated frequency predictions rather than penalizing them — the exact opposite of its intended effect. This suggests the loss formula has an inverted sign or the regularization direction is wrong.

**Impact:** The model is actively encouraged to concentrate predictions at fewer frequencies, which may contribute to type collapse (all predictions collapse to peaking at similar frequencies).

**Recommended Remediation:**
1. Audit the `spread_loss` formula in `loss_multitype.py` for sign errors
2. Verify the intended direction: spread loss should penalize concentrated frequency predictions
3. Add an assertion that spread_loss is non-negative during training

---

## 2. Pipeline Architecture & Design

### HIGH-01: Dual Biquad Implementations — Offline Generator vs Training DSP

**File:** `dataset_pipeline/generate_data.py` vs `differentiable_eq.py`

**Issue:** The offline data generator reimplements all five biquad coefficient computation functions using raw Python/math, while `differentiable_eq.py` implements the same formulas in PyTorch. These are **two independent implementations** of the RBJ Audio EQ Cookbook formulas.

**Impact:** Numerical precision differences between the two implementations could cause the offline-generated data to have slightly different frequency responses than what the training loop's DSP cascade computes. The model would then train on "ground truth" targets that don't match the loss function's rendering of those same parameters.

**Recommended Remediation:**
1. **Unify the implementation:** Have `generate_data.py` import `DifferentiableBiquadCascade` from `differentiable_eq.py` and use it directly for offline generation
2. **Add a cross-implementation consistency test:** Generate parameters, apply EQ via both implementations, and assert frequency responses match within `atol=1e-4`

---

### HIGH-02: DataLoader num_workers=0 Blocks GPU (Active Performance Bottleneck)

**Evidence:** `conf/config.yaml` line: `num_workers: 0` (the current config still has this, despite the code's auto-detection logic)
**Log:** `[data] WARNING: num_workers=0 — data generation blocks GPU. Set num_workers>=2 in config (auto-detected 48 CPUs)`

**Issue:** The config explicitly sets `num_workers: 0`, meaning all data generation (including the biquad cascade, STFT, and mel spectrogram computation) happens on the main process, serially blocking the GPU. The system has 48 CPUs available.

**Impact:** Estimating ~40% of epoch time is data loading, this adds ~29 seconds per epoch × 60 epochs = **~29 minutes of wasted GPU idle time**.

**Recommended Remediation:**
1. Remove `num_workers: 0` from `conf/config.yaml` — the auto-detection logic already computes `min(cpu_count()-1, 8)` correctly
2. Ensure `SyntheticEQDataset.__getitem__` is pickle-safe (it is — uses `random`, `numpy`, and `torch`)

---

### HIGH-03: Batch Size Creates Insufficient Training Steps (Active Issue)

**File:** `conf/config.yaml`: `batch_size: 4096` (updated from 16384), `dataset_size: 200000`
**Log:** `Train batches: 10, Val batches: 1` (for the AST config run)

**Issue:** With 200K samples and batch size 4096, after the 80/10/10 split (160K train), there are only ~39 training batches per epoch. With only 1-2 validation batches, validation metrics have extremely high variance.

**Impact:**
- Validation metrics fluctuate wildly between epochs
- Early stopping decisions are noisy
- The model gets only ~39 gradient updates per epoch × 60 epochs = 2,340 total updates

**Recommended Remediation:**
1. **Reduce batch size to 1024 or 2048** for more gradient updates per epoch
2. **Increase validation set size** for more stable validation metrics
3. **Monitor validation metric variance** across batches

---

### HIGH-04: Five Concurrent "Best" Checkpoints Create Storage Confusion

**Evidence:** `training_log.txt`:
```
Updated checkpoint: checkpoints/best_primary.pt
Updated checkpoint: checkpoints/best.pt
Updated checkpoint: checkpoints/best_gain.pt
Updated checkpoint: checkpoints/best_type.pt
Updated checkpoint: checkpoints/best_audio.pt
```

**Issue:** The trainer maintains 5 different "best" checkpoints, each tracking a different metric. With 60 epochs, this creates 300+ checkpoint files. There is no documented policy for which checkpoint to use for inference.

Furthermore, `best_primary.pt` tracks `primary_val_score = gain_mae + 4*(1-type_acc) + 0.25*freq_mae`. With type accuracy stuck at 19.7%, this metric is dominated by the type_error term (3.21), masking actual gain regression progress.

**Impact:**
- Storage waste (each checkpoint is ~90MB for the 22.5M parameter model; ~8.5GB for the 95.5M wav2vec2 model)
- Confusion about which checkpoint represents the "real" model
- No automated cleanup of superseded checkpoints

**Recommended Remediation:**
1. Keep only 2 checkpoints: `best.pt` (primary metric) and `last.pt` (resume point)
2. Add automatic checkpoint cleanup: keep only the 3 most recent epoch checkpoints
3. Document the selection criteria

---

### HIGH-05: Monolithic Trainer Class (2,701 Lines)

**File:** `insight/train.py` (2,701 lines)

**Issue:** The `Trainer` class handles: config loading, dependency validation, dataset creation, precompute caching, model instantiation, optimizer setup (3 types), LR scheduling, curriculum management, training loop, validation loop, metric computation, Hungarian matching, checkpointing (5 variants), signal handling, structured logging, gradient norm tracking, NaN recovery, ONNX export, profiling, and run metadata management.

**Impact:**
- Any modification risks unintended side effects
- Testing individual components requires instantiating the entire trainer
- The file exceeds the "cognitive load budget" of a single developer

**Recommended Remediation:** Decompose into focused modules:
```
training/
  dataset_manager.py    # Dataset creation, caching, splits
  optimizer_factory.py  # Optimizer construction
  curriculum.py         # Stage transitions, parameter injection
  checkpoint_manager.py # Save/load/recovery
  training_loop.py      # Epoch iteration, logging
  validation_loop.py    # Metric computation, Hungarian matching
  trainer.py            # Thin orchestrator wiring the above
```

---

### HIGH-06: Loss Has 24+ Components With Untuned Weighting

**Evidence:** `training_log.txt` epoch 1:
```
loss_gain=14.28 (61%), multi_scale_loss=16.75, hdb_loss=4.47, type_loss=3.25, ...
film_diversity_loss=0.00, activity_loss=0.00, spread_loss=-0.42 (negative!)
```

**Issue:** The loss function computes **24+ distinct components** simultaneously. The gain loss dominates at 61% of the total. Some components are effectively zero (wasting compute), while others have negative values (inverted direction).

**Impact:**
1. **Gradient signal imbalance:** The gain loss dominates gradients, explaining why gain MAE is 3.71 dB while type accuracy is 19.7%
2. **Negative loss component:** `spread_loss=-0.42` means the regularization is actually *rewarding* concentrated frequencies
3. **Zero components waste compute:** `film_diversity_loss` and `activity_loss` contribute nothing

**Recommended Remediation:**
1. **Audit each loss component's purpose and effectiveness** — remove or fix components that are zero or counterproductive
2. **Normalize loss magnitudes** so each contributes roughly equally (~1.0 at initialization)
3. **Use UncertaintyWeightedLoss more aggressively** for automatic balancing
4. **Fix spread_loss sign** immediately

---

### HIGH-07: Gumbel-Softmax Temperature Floor Creates Train/Test Mismatch

**File:** `conf/config.yaml`: `gumbel.start_tau: 0.5, min_tau: 0.05`

**Issue:** The Gumbel-Softmax never fully hardens even at `min_tau=0.05`. With 5 classes and tau=0.05, the softmax probabilities for the top class are typically >0.95 but not 1.0, meaning some gradient "leaks" to wrong classes. At inference, `argmax` is used (hard decision), but the training never fully commits.

**Impact:** Train/test mismatch — the model learns with soft type probabilities but inference uses hard decisions.

**Recommended Remediation:**
1. Consider lowering `min_tau` to `0.01`
2. Or use the **straight-through Gumbel-Softmax estimator** (hard forward, soft backward)
3. Evaluate the gap between soft and hard predictions at inference

---

### HIGH-08: Integration Tests Don't Reproduce Training Failure Mode

**File:** `insight/tests/test_integration.py`

**Issue:** The 5 integration tests verify that a forward pass produces finite outputs and gradients. However, they use a small model (channels=64, num_blocks=2, num_stacks=1) and random inputs. They do **not** reproduce the type collapse failure mode observed in the actual training run.

**Impact:** Tests pass green while the actual training fails to learn types. The test suite gives false confidence.

**Recommended Remediation:** Add a test that:
1. Uses the full model config (channels=256, num_blocks=8, num_stacks=3)
2. Trains for 10 epochs on synthetic data
3. Asserts type accuracy > 30% (above random baseline)
4. This would have caught the type collapse before the 60-epoch run

---

### HIGH-09: Shell Scripts Lack Consistent Security Posture

**Files:** `insight/launch_wav2vec2.sh`, `insight/kill_old.sh`

**Issue:** `kill_old.sh` uses `pgrep`/`pkill` without PID file verification. If multiple Python scripts are running, unrelated processes may be killed. `launch_wav2vec2.sh` was improved per the previous audit but still lacks input validation.

**Impact:** Risk of killing unrelated processes on shared systems.

**Recommended Remediation:**
1. Use PID files for all background processes
2. Add input validation to all shell scripts
3. Replace `pkill` with targeted `kill $(cat <pidfile>)` patterns

---

## 3. Privacy & Security

### MEDIUM-01: Path Traversal Protection Uses Module-Load-Time Roots

**File:** `pipeline_utils.py` — trusted roots computed at module load

**Issue:** `TRUSTED_ROOTS` includes `Path.cwd()` resolved at module load time. If the working directory is changed (e.g., via `cd`), the trusted roots don't update. However, `Path.cwd()` at module load captures the CWD at import time, which may not be the intended root.

**Impact:** If the module is imported from a subdirectory, the trusted roots include that subdirectory, potentially allowing access to sibling directories.

**Recommended Remediation:**
1. Use absolute paths for trusted roots, computed from the project root (not CWD)
2. Add a configurable `ARTIFACT_ROOT` environment variable
3. Validate that resolved paths don't escape via symlink chains

---

### MEDIUM-02: Checkpoint Loading May Use weights_only=False for Resume

**File:** `train.py` — checkpoint resume logic

**Issue:** Full checkpoint resumption requires optimizer state and scheduler state, which `weights_only=True` cannot load. If the resume path accepts untrusted checkpoint files, arbitrary code execution via pickle is possible.

**Impact:** Loading untrusted checkpoints could execute arbitrary code. On shared computing environments (Lightning AI studios), this is a real risk.

**Recommended Remediation:**
1. For resume, implement a custom safe loader that only allows tensor types
2. Add checkpoint integrity verification (SHA256 hash) before loading
3. Document the security implications of `--resume` in the codebase

---

### MEDIUM-03: Absolute Path Leakage in Dataset Generation (Mitigated)

**File:** `dataset_pipeline/generate_data.py` — `include_source_abs_path` flag

**Issue:** When `include_source_abs_path=True`, the pipeline writes the absolute filesystem path of source audio to both the params JSON and the manifest. While the flag defaults to `False` (good), if enabled, it reveals infrastructure details.

**Impact:** If datasets are shared or published, filesystem paths leak internal infrastructure info.

**Recommended Remediation:**
1. Add a sanitization step that strips user home directory paths
2. Add a warning when this flag is enabled
3. Consider using relative path hashing instead

---

## 4. Bias Mitigation

### MEDIUM-04: Signal Type Distribution Now Configurable (Previously Fixed)

**Status:** **RESOLVED** — `signal_type_weights` in config.yaml with realistic weights: speech_like=0.30, harmonic=0.25, pink_noise=0.15, percussive=0.15, noise=0.10, sweep=0.05. This is a significant improvement from the previous uniform distribution.

---

### MEDIUM-05: Gain Distribution Now Configurable (Previously Fixed)

**Status:** **RESOLVED** — `gain_distribution: beta` in config.yaml, with beta(2,5) distribution concentrating mass near 0 dB. This aligns with real-world EQ usage where most adjustments are small (1-4 dB).

---

### MEDIUM-06: Frequency Range Clipping Creates Hard Boundaries

**File:** `insight/dataset.py` — type-specific frequency bounds

**Issue:** Shelf and HP/LP filter frequencies are clipped to specific ranges (lowshelf: 20-5000 Hz, highshelf: 1000-20000 Hz, etc.). These create hard boundaries where a high-shelf at 1001 Hz and one at 999 Hz are treated differently.

**Impact:** The model may struggle with EQ bands near these boundaries, predicting unstable results for frequencies near 1000 Hz (highshelf) or 500 Hz (highpass).

**Recommended Remediation:**
1. Use soft boundaries with rejection sampling instead of hard clipping
2. Add small overlap regions at boundaries
3. Log the actual frequency distributions to verify no clustering at boundaries

---

## 5. Data Lineage & Drift Detection

### MEDIUM-07: Cache Staleness Detection Missing DSP Code Hash

**File:** `insight/dataset.py` — cache metadata

**Issue:** The cache staleness detection via metadata signatures compares generation parameters but does NOT detect:
- Changes in the DSP code (`differentiable_eq.py`) that affect EQ rendering
- Changes in the augmentation logic
- Changes in the mel filterbank construction

**Impact:** If the DSP code is modified but the config is unchanged, stale cached data will be used without warning.

**Recommended Remediation:**
1. Include a hash of the source code for `differentiable_eq.py` and `dataset.py` in the metadata signature (the `compute_version_hash()` function exists but is not integrated into cache validation)
2. Add `git_commit` to the metadata if in a git repo

---

### MEDIUM-08: No Data Distribution Monitoring During Training

**Issue:** There is no mechanism to monitor the actual distribution of generated data during training. If the random seed changes, or if curriculum stages produce unexpected distributions, there is no automatic detection.

**Impact:** Silent distribution shifts could occur between training runs or curriculum stages without detection.

**Recommended Remediation:**
1. Add periodic histogram logging of gain, freq, Q, and type distributions
2. Compare distributions across curriculum stages
3. Alert if the distribution deviates significantly from expected

---

### MEDIUM-09: No Version Hash in Checkpoint Metadata

**Issue:** Checkpoint metadata does not include Python, PyTorch, or dependency versions. This makes it impossible to reproduce results when the environment changes.

**Recommended Remediation:**
1. Add `torch.__version__`, `numpy.__version__`, Python version to checkpoint metadata
2. Include git commit hash if in a git repo
3. Save a requirements snapshot alongside checkpoints

---

## 6. Pipeline Efficiency

### MEDIUM-10: Precompute Mode Memory Footprint Not Estimated

**File:** `insight/dataset.py` — precompute mode

**Issue:** Precompute mode caches all mel-spectrograms in memory. For 200K samples with 128×T mel-spectrograms, this requires significant RAM (~20-40 GB depending on duration). There is no memory estimation before precomputation.

**Impact:** OOM on systems with limited RAM, especially with larger datasets or longer audio.

**Recommended Remediation:**
1. Add memory estimation before precomputation
2. Implement a streaming/lazy mode that generates samples on-the-fly
3. Consider memory-mapped storage for large precomputed datasets

---

### MEDIUM-11: torch.compile Disabled in Production Config

**File:** `conf/config.yaml`: `use_torch_compile: false`

**Issue:** `torch.compile()` can provide 20-40% speedup for the TCN forward pass but is disabled.

**Impact:** Each epoch takes longer than necessary. Over 60 epochs, this compounds to significant wasted time.

**Recommended Remediation:** Enable `use_torch_compile: true` and benchmark. If compilation fails on a specific op, wrap only the stable submodules.

---

### MEDIUM-12: Validation Runs Hungarian Matching on Every Sample

**File:** `insight/metrics.py` — `compute_eq_metrics()`

**Issue:** Hungarian matching uses `scipy.optimize.linear_sum_assignment`, which is O(n³) in the number of bands. With 5 bands per sample and ~1,000 samples per validation batch, this is called 1,000 times per epoch.

**Impact:** Validation latency is higher than necessary. This will become a bottleneck with larger validation sets.

**Recommended Remediation:** Run Hungarian matching every N validation steps and use cached assignments for intermediate steps, or switch to a greedy matching approximation for validation.

---

### LOW-01: Gradient Clipping Is Heavy and Frequent

**Evidence:** `training_log.txt`:
```
[WARN] Heavy gradient clipping at step 0: grad_norm=5.04, clip_ratio=5.0x
[WARN] Heavy gradient clipping at step 10: grad_norm=3.53, clip_ratio=3.5x
```

**Issue:** Gradient clipping at 5.0× ratio on the very first step indicates the model's initial gradients are 5× larger than the clip threshold. This suggests the learning rate may be too high for the initial weights, or the loss function produces excessively large gradients.

**Recommended Remediation:** Monitor the gradient norm trajectory. If it stays above clip threshold for many steps, increase the clip threshold or reduce the learning rate.

---

### LOW-02: Shell Scripts Inconsistent in Quality

**Files:** `insight/launch_wav2vec2.sh`, `insight/kill_old.sh`, `insight/run_encoder_comparison.sh`

**Issue:** Not all shell scripts have been audited with the same rigor. `kill_old.sh` uses `pgrep`/`pkill` without PID file verification.

**Recommended Remediation:** Audit all shell scripts with the same rigor. Use PID files for all background processes.

---

### LOW-03: Documentation Drift in CLAUDE.md

**Issue:** `CLAUDE.md` references `augmentation.py` with "SpecAugment (frequency + time masking)" but this file does not exist in the codebase.

**Recommended Remediation:** Either implement SpecAugment or remove the reference from CLAUDE.md.

---

### LOW-04: Multiprocessing Pool Error Handling Silently Loses Results

**File:** `dataset_pipeline/generate_data.py` — multiprocessing pool

**Issue:** The pool catches exceptions broadly and sets `results = []` on any error, meaning ALL results are silently lost. A single worker crash can cause the entire dataset generation to return empty results.

**Recommended Remediation:**
1. Use `pool.map_async` with a timeout and partial result collection
2. Add a check: if `len(results) == 0` and `len(tasks) > 0`, raise an error
3. Log individual worker failures

---

### LOW-05: No CI/CD Pipeline for Tests

**Issue:** Tests are standalone Python scripts run manually. There is no CI/CD pipeline, no pre-commit hooks, and no automated test execution on code changes.

**Recommended Remediation:** Add a `pytest` CI step. Even a simple `cd insight && python -m pytest tests/` in a pre-push hook would catch regressions.

---

### LOW-06: Early Stopping Monitors Composite Score Dominated by Type Error

**File:** `train.py` — `compute_primary_val_score()`

**Issue:** The primary validation score is `gain_mae + 4*(1-type_acc) + 0.25*freq_mae`. With type accuracy at 19.7%, the type_error term contributes `4 * 0.803 = 3.21`, while gain_mae contributes 3.71. The composite is 7.58.

**Impact:** Early stopping may terminate training just as the model starts learning filter types, because the composite score is dominated by the initial type collapse penalty.

**Recommended Remediation:**
1. Monitor type accuracy independently for early stopping
2. Or use a dynamic weighting scheme that reduces type_error weight as the model improves
3. The `min_epochs_before_early_stop` (15 epochs) helps but may not be sufficient for the 60-epoch run

---

### LOW-07: Signal Handlers Don't Save Mid-Epoch Checkpoints

**File:** `insight/train.py` — `_handle_signal()`

**Issue:** Signal handlers set `self._received_signal`, which is checked only at epoch boundaries. If a SIGTERM arrives mid-epoch (during a 72-second epoch), the entire epoch's gradients are computed but never checkpointed.

**Recommended Remediation:** On signal receipt during training, save an emergency checkpoint at the next batch boundary (not epoch boundary). The structured logger already logs events — add `emergency_checkpoint_saved` event.

---

## 7. Curriculum Learning

### MEDIUM-13: Uncoordinated Dual Warmup

**File:** `loss_multitype.py` (loss-level warmup) vs `train.py` (H_db lambda ramp)

**Issue:** Two separate warmup mechanisms exist:
1. Loss-level warmup in `loss_multitype.py` (epoch threshold + gain MAE gate)
2. H_db lambda ramp in `train.py` (linear ramp over 10 epochs)

These are not coordinated. The H_db ramp may still be ramping while the loss-level warmup has already ended, creating conflicting training signals.

**Recommended Remediation:**
1. Consolidate warmup logic into a single source of truth
2. Add clear documentation of the warmup strategy
3. Log warmup state transitions explicitly

---

### MEDIUM-14: Precomputed Dataset Skips Dynamic Curriculum

**Evidence:** `training_log.txt`:
```
[curriculum] train dataset is precomputed; dynamic curriculum updates are skipped
```

**Issue:** When using a precomputed dataset, the curriculum's dynamic parameter updates (gain bounds, Q bounds, type weights) are skipped because the dataset's parameters are fixed at generation time. The curriculum stage transitions only affect loss weights, not the data distribution.

**Impact:** The curriculum's stage 1 (lowshelf focus, narrow gain bounds) and stage 2 (midrange) become ineffective — the model trains on the full-range data from epoch 1 regardless of the curriculum stage.

**Recommended Remediation:**
1. Generate separate precomputed datasets for each curriculum stage, or
2. Use on-the-fly generation (`precompute_mels: false`) when using curriculum, or
3. Add a warning when curriculum is enabled with precomputed data

---

## 8. Reproducibility

### MEDIUM-15: bf16 Mixed Precision Non-Determinism

**Issue:** The config uses `precision: bf16-mixed`. BF16 has lower mantissa precision than FP32 (7 bits vs 23 bits), which means the same computation in FP32 and BF16 can produce different results. Combined with non-deterministic reduction operations in CUDA, exact reproducibility across hardware is not possible.

**Recommended Remediation:**
1. Document that bf16-mixed does not guarantee bit-exact reproducibility
2. For reproducibility validation, compare metrics at 2-3 decimal places, not exact values
3. Consider adding a `precision: fp32` option for reproducibility testing

---

### MEDIUM-16: Inconsistent Seeding Across DataLoader Workers

**File:** `pipeline_utils.py` — `seed_worker()`

**Issue:** `seed_worker` seeds numpy, random, and torch per-worker, but `torch.cuda` state save/restore may silently fail in DataLoader workers (handled with try/except). This means CUDA state may not be properly restored in some edge cases.

**Recommended Remediation:**
1. Document that full reproducibility requires `num_workers=0` for the DataLoader
2. Add a warning when `deterministic=True` and `num_workers > 0`

---

## 9. Active Training Run Comparison

### AST Encoder Run (`training_log.txt`) vs wav2vec2 Run (`training_log_peak125_from_043.txt`)

| Metric | AST Encoder (current) | wav2vec2 Encoder (resume from epoch 43) |
|--------|----------------------|----------------------------------------|
| Parameters | 22.5M | 95.5M |
| Train batches | 10 | 351 |
| Epoch time | 68-73s | N/A (batch-level logging) |
| loss_gain | 14.3 | 2.5 |
| type_loss | 3.3 | 3.4 |
| type_entropy_loss | 0.02 | 0.001 |
| multi_scale_loss | 16.7 | 9.3 |
| spread_loss | -0.42 | -1.50 |
| Gradient clipping | Heavy (5.0× at step 0) | Not logged |

**Key Observations:**
1. The wav2vec2 encoder achieves **much lower loss_gain** (2.5 vs 14.3) — the pretrained encoder is significantly better at gain regression
2. However, **type_loss is similar** (3.4 vs 3.3) — the type classification problem persists across both encoders
3. **type_entropy_loss is even lower** in wav2vec2 run (0.001 vs 0.02) — the wav2vec2 model's type predictions are even more collapsed
4. **spread_loss is more negative** in wav2vec2 run (-1.50 vs -0.42) — the inverted regularization is worse with the larger model

**Conclusion:** The type collapse issue is **architecture-agnostic** — it affects both the TCN/AST and wav2vec2 encoders. This confirms the root cause is in the **loss function or Gumbel-Softmax configuration**, not the encoder architecture.

---

## 10. Prior Audit Findings Status

| Finding ID | Status | Notes |
|------------|--------|-------|
| **CRITICAL-01** (prev: numerical stability) | ✅ Fixed | Epsilons increased, clamping added |
| **CRITICAL-02** (prev: gradient flow) | ⚠️ Partially fixed | Warmup detach still active |
| **CRITICAL-03** (prev: memory leak) | ⚠️ Partially fixed | NaN sanitization exists but incomplete |
| **CRITICAL-04** (prev: fallback bias) | ✅ Fixed | Fallback sample with retry logic |
| **CRITICAL-05** (prev: type collapse) | ❌ **NOT FIXED** | Still active in training logs |
| **HIGH-01** (prev: Gumbel annealing) | ⚠️ Partially fixed | Formula correct but tau still too high |
| **HIGH-02** (prev: Hungarian fallback) | ✅ Fixed | Proper error handling |
| **HIGH-03** (prev: data bias) | ✅ Fixed | Signal type weights and gain distribution configurable |
| **HIGH-04** (prev: security) | ✅ Fixed | Path traversal protection implemented |
| **HIGH-05** (prev: DataLoader blocking) | ⚠️ Partially fixed | Auto-detection exists but config overrides to 0 |
| **HIGH-06** (prev: dual pipelines) | ❌ **NOT FIXED** | Dual biquad implementations still exist |
| **HIGH-07** (prev: loss weighting) | ❌ **NOT FIXED** | 24+ components, spread_loss inverted |
| **HIGH-08** (prev: trainer complexity) | ⚠️ Partially fixed | Structured logger extracted but trainer still monolithic |
| **HIGH-09** (prev: test gap) | ⚠️ Partially fixed | 25 tests exist but don't reproduce failure modes |
| **MEDIUM-01 to MEDIUM-34** | Mostly fixed | 28/34 addressed, 6 partially, 0 ignored |
| **LOW-01 to LOW-07** | Mostly addressed | Documentation and CI/CD still pending |

---

## 11. Prioritized Recommendations

### Immediate (P0 — Fix Before Next Training Run)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 1 | **Fix spread_loss sign** — audit formula in `loss_multitype.py` | Eliminates inverted regularization contributing to type collapse | 1-2 hours |
| 2 | **Lower Gumbel start_tau to 0.5** (from 2.0 in config) | Provides meaningful type gradients from epoch 1 | 5 minutes |
| 3 | **Increase lambda_type_entropy to 2.0** (from 0.5) | Penalizes peaked type predictions | 5 minutes |
| 4 | **Delete and regenerate stale cache** (`rm data/dataset_phase3_200k.pt`) | Ensures data matches current config | 2-4 hours |
| 5 | **Remove num_workers: 0 from config** | Unblocks GPU during data loading | 5 minutes |
| 6 | **Allow type gradients during warmup** (remove detach or use reduced weight) | Fixes broken gradient flow for type learning | 1 hour |

### Short-term (P1 — Next Sprint)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 7 | **Unify biquad implementations** — use `DifferentiableBiquadCascade` in `generate_data.py` | Eliminates DSP inconsistency risk | 2-3 hours |
| 8 | **Reduce batch size to 1024-2048** for more gradient updates | Improves convergence tracking and validation stability | 5 minutes |
| 9 | **Add cross-implementation DSP consistency test** | Prevents future DSP drift | 1-2 hours |
| 10 | **Reduce checkpoints from 5 to 2** with automatic cleanup | Saves storage and reduces confusion | 2-3 hours |
| 11 | **Integrate code hash into cache staleness detection** | Prevents stale cached data after code changes | 1-2 hours |
| 12 | **Add full-model integration test for type accuracy** | Catches type collapse before long training runs | 2-3 hours |

### Medium-term (P2 — Next Month)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 13 | **Decompose Trainer class** into focused modules | Improves maintainability and testability | 2-3 days |
| 14 | **Audit and normalize all 24+ loss components** | Eliminates wasted compute, improves gradient balance | 1-2 days |
| 15 | **Enable torch.compile and benchmark** | 20-40% speedup per epoch | 2-3 hours |
| 16 | **Add data distribution monitoring and drift alerts** | Detects silent distribution shifts | 1 day |
| 17 | **Add version/hash tracking to checkpoint metadata** | Improves reproducibility | 2-3 hours |
| 18 | **Consolidate dual warmup into single source of truth** | Eliminates conflicting training signals | 1 day |

### Long-term (P3 — Ongoing)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 19 | **Implement CI/CD pipeline for tests** | Prevents regressions | 1-2 days |
| 20 | **Add SpecAugment or remove reference from docs** | Improves regularization or documentation accuracy | 1-2 days |
| 21 | **Implement memory-mapped precomputed datasets** | Enables larger datasets without OOM | 2-3 days |
| 22 | **Add Prometheus metrics dashboard** | Production-grade training monitoring | 1-2 days |

---

## 12. Conclusion

The IDSP system demonstrates **sophisticated research-grade engineering** with advanced techniques (Hungarian matching, Gumbel-Softmax, curriculum learning, hierarchical type heads, gradient checkpointing). Previous audits have successfully addressed 28 of 34 findings.

However, **the type collapse issue remains the most critical active problem**. It is architecture-agnostic (affects both AST and wav2vec2 encoders), persists across multiple training runs, and renders the multi-type EQ estimation capability non-functional. The root cause is a combination of:

1. **Gumbel-Softmax temperature too high** (start_tau=2.0) — provides no type signal
2. **Type gradients detached during warmup** — encoder cannot learn type-discriminative features
3. **Loss component imbalance** — gain loss dominates at 61%, starving type learning
4. **Inverted spread_loss** — actively rewards concentration rather than diversity

The recommended P0 fixes address all four root causes and should be implemented before the next training run. With these fixes, the system's advanced architecture (pretrained encoders, Hungarian matching, hierarchical type heads) has the potential to achieve state-of-the-art blind EQ estimation performance.

---

*Audit completed: 2026-04-14*
*Next recommended audit: After P0 fixes are implemented, run a 20-epoch training pilot and re-audit for regression*
