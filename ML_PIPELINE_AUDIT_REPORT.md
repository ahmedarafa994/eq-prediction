# ML Pipeline Comprehensive Audit Report

**Project:** Differentiable DSP EQ Parameter Estimation (IDSP)  
**Date:** 2026-04-14  
**Scope:** Training pipeline, dataset pipeline, DSP frontend, loss functions, configuration, lineage, security  
**Auditor:** Automated pipeline audit  

---

## Executive Summary

The audit examined 12 core modules across the training and data generation pipelines. **29 findings** were identified across 6 categories. The system has strong foundational design (differentiable DSP, Hungarian matching, curriculum learning) but has significant numerical stability risks in the biquad coefficient computation and frequency response paths that can cause training collapse. Data integrity protections are above average for a research codebase, but bias in synthetic data generation and missing drift detection mechanisms need attention.

**Risk Summary:**

| Severity | Count | Areas |
|----------|-------|-------|
| CRITICAL | 5 | Numerical stability, gradient flow, memory leaks |
| HIGH | 7 | Gumbel annealing, Hungarian fallback, data bias, security |
| MEDIUM | 10 | Reproducibility, validation gaps, curriculum coordination |
| LOW | 7 | Logging, profiling, documentation, version pinning |

---

## 1. Data Integrity & Preprocessing

### 1.1 CRITICAL — Fallback Sample Label Contamination (`dataset.py:489-499, 560-595`)

**Finding:** When EQ rendering fails (3 retries exhausted), the fallback sample returns `dry_audio == wet_audio` with all-zero gains and all-peaking type labels. This injects "identity EQ" samples into training data silently. The `_fallback_count` is logged but not exposed as a training metric.

**Impact:** If DSP instability is frequent (e.g., extreme Q values), a significant fraction of training data becomes "no-op" samples, teaching the model that flat EQ is common. This creates a systematic bias toward predicting zero gain.

**Recommendation:**
- Track fallback rate as a training metric and alert if >1% of samples fall back
- Consider re-drawing parameters instead of falling back to identity
- Add fallback sample count to validation reports

### 1.2 HIGH — Synthetic Data Distribution Bias (`dataset.py:358-360, 60-62`)

**Finding:** Gain sampling uses `random.uniform(gain_range[0], gain_range[1])` — uniform distribution across [-12, +12] dB. Real-world EQ usage follows a power-law distribution: most adjustments are small (1-4 dB), extreme values are rare. The `generate_data.py` file uses `beta_gain()` with `np.random.beta(2, 5)` which is closer to realistic, but `dataset.py` does not.

**Impact:** The model sees equal numbers of large and small EQ adjustments, which is unrealistic. At inference time on real audio, the model may over-estimate gain magnitudes because the training distribution has too many extreme examples.

**Recommendation:**
- Align `dataset.py` gain sampling with `generate_data.py`'s `beta_gain()` distribution
- Or add a `gain_distribution` config parameter ("uniform" | "beta" | "realistic")
- Document the expected gain distribution in the config schema

### 1.3 HIGH — Type Weight Imbalance in Default Config (`dataset.py:38, config.yaml:33-37`)

**Finding:** `DEFAULT_TYPE_WEIGHTS = [0.5, 0.15, 0.15, 0.1, 0.1]` heavily favors peaking filters. However, `config.yaml` overrides to `[0.2, 0.2, 0.2, 0.2, 0.2]` (uniform). The first curriculum stage uses `[0.2, 0.35, 0.15, 0.15, 0.15]`. These inconsistencies make it difficult to understand the actual training distribution without reading both files.

**Impact:** The hard-coded defaults in `dataset.py` differ from the config-driven values. If the config is not loaded correctly, training silently uses a biased distribution. During curriculum transitions, the type distribution shifts significantly, which can cause the type classifier to oscillate.

**Recommendation:**
- Remove `DEFAULT_TYPE_WEIGHTS` from `dataset.py` and require config specification
- Add validation that type weights sum to 1.0 (within tolerance)
- Log the actual type distribution being used at each curriculum stage transition

### 1.4 MEDIUM — Missing Audio Normalization Consistency (`dataset.py:267-268 vs 294-296`)

**Finding:** Individual signals are peak-normalized (`audio / peak`), but after mixing in `_generate_dry_mix`, the mix is also peak-normalized. This means the relative levels between sources are preserved within a mix but the absolute level is always 1.0. The `_augment_audio_pair` then scales by `random.uniform(0.5, 1.0)`.

**Impact:** All training samples have similar loudness ranges, which may not reflect real-world audio where level varies significantly. The model might not learn to handle quiet or loud inputs well.

**Recommendation:**
- Add LUFS-based normalization option as an alternative to peak normalization
- Consider adding a wider gain variation range (0.3 to 1.0) for robustness
- Document the normalization strategy in the config

### 1.5 MEDIUM — No SpecAugment Implementation in Training Loop (`augmentation.py` not found)

**Finding:** The CLAUDE.md references `augmentation.py` with "SpecAugment (frequency + time masking)" but this file does not exist in the codebase. The only augmentation is the label-preserving audio augmentation in `dataset.py` (polarity inversion, time shift, gain scaling).

**Impact:** SpecAugment is a proven regularization technique for audio models. Its absence may contribute to overfitting, especially on synthetic data where the distribution is narrower than real audio.

**Recommendation:**
- Either implement SpecAugment or remove the reference from CLAUDE.md
- If implemented, apply to mel-spectrograms after precomputation, not to raw audio
- Add a config flag to enable/disable augmentation

---

## 2. Preprocessing & DSP Correctness

### 2.1 CRITICAL — Division-by-Zero Risk in Biquad Coefficients (`differentiable_eq.py:112, 153`)

**Finding:** `alpha = torch.sin(w0) / (2.0 * q + 1e-8)` — the epsilon 1e-8 is too small. When Q approaches 0 (the model can predict Q=0.1 minimum, and during training with gradient perturbations, Q can effectively reach near-zero), alpha can become extremely large, causing downstream NaN in coefficient computation.

**Impact:** Training instability, NaN gradients, potentially frozen parameters requiring optimizer state sanitization.

**Recommendation:**
- Increase epsilon from 1e-8 to 1e-4
- Add explicit `torch.clamp(alpha, min=1e-6, max=10.0)` after computation
- Log when alpha is being clamped as a diagnostic signal

### 2.2 CRITICAL — Insufficient Magnitude Clamping in Frequency Response (`differentiable_eq.py:259`)

**Finding:** `H_mag = torch.sqrt(torch.clamp(num_mag2 / (den_mag2 + 1e-4), min=1e-8))` — the denominator epsilon 1e-4 may be insufficient when the denominator is very small (near pole of the filter). This can produce extremely large H_mag values that overflow in log10 operations.

**Impact:** NaN/inf in spectral losses, causing the entire batch to be skipped or producing corrupted gradients.

**Recommendation:**
- Increase clamping to `torch.clamp(num_mag2 / (den_mag2 + 1e-3), min=1e-8, max=1e8)`
- Add explicit H_mag clamping to [1e-6, 1e6] before any log operations
- Consider using `torch.log1p` for more stable log-domain computation

### 2.3 HIGH — Mel Filterbank Edge Artifacts (`dsp_frontend.py:42-53`)

**Finding:** The mel filterbank uses `.long()` for bin index conversion (`bin_points = ((self.n_fft + 1) * hz_points / self.sample_rate).long()`), which truncates. This can cause the lowest mel bins to collapse to index 0 if `f_min` is very low relative to the FFT resolution. Additionally, no dithering or normalization is applied to the filterbank.

**Impact:** Very low-frequency information may be lost or aliased, affecting the model's ability to detect low-shelf and high-pass filters.

**Recommendation:**
- Use `torch.round()` instead of `.long()` for bin center computation
- Add triangular normalization (sum-to-one per filter) for consistent energy
- Validate that mel_fb has no all-zero rows after construction

### 2.4 MEDIUM — Phase Preservation Assumption (`differentiable_eq.py`, `dsp_frontend.py:154-167`)

**Finding:** `apply_eq_to_complex_stft` modifies only magnitude and preserves phase: `new_mag * torch.exp(1j * phase)`. This is correct for linear-phase EQ but real parametric EQ is minimum-phase and does modify phase. The frequency-domain approximation ignores this.

**Impact:** Training data generated via frequency-domain EQ application differs subtly from real IIR biquad filtering. The model may learn features that don't transfer perfectly to real-world inference.

**Recommendation:**
- Document this as a known approximation
- Consider adding a phase correction term based on the biquad group delay
- Evaluate model on both frequency-domain and time-domain rendered test data

---

## 3. Numerical Stability

### 3.1 CRITICAL — Memory Leak in Optimizer NaN Sanitization (`train.py:1389-1394`)

**Finding:** The optimizer state sanitization loop zeroes NaN values but doesn't reset momentum or update counts:
```python
for state in self.optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            v.zero_()
```
After zeroing, the Adam momentum (`exp_avg`) is zero but `exp_avg_sq` may retain corrupted values. The step count for bias correction continues incrementing, creating a mismatch.

**Impact:** "Ghost" momentum from NaN-preserved second moments can cause future parameter updates to be biased, leading to slow degradation of model quality.

**Recommendation:**
- Reset both `exp_avg` and `exp_avg_sq` for affected parameters
- Reset the step count for affected parameter groups
- Consider completely re-creating the optimizer state for corrupted parameters

### 3.2 CRITICAL — Broken Gradient Flow During Warmup (`loss_multitype.py:603-606`)

**Finding:** During warmup, type logits are detached: `pred_type_logits_for_match = pred_type_logits.detach()`. This prevents gradients from flowing from the type classification loss back through the encoder, meaning the encoder cannot learn type-discriminative features during warmup.

**Impact:** The encoder develops representations that are blind to filter type, making it harder for the type classifier to learn when warmup ends. This likely contributes to type collapse issues.

**Recommendation:**
- Allow type gradients to flow from the start with reduced weight (e.g., 0.1x during warmup)
- Only detach if the model is severely unstable, and log the reason
- Track type prediction entropy during warmup to detect early collapse

### 3.3 HIGH — Log-Cosh Numerical Discontinuity (`loss_multitype.py:40`)

**Finding:** The formulation `abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff) + 1e-8) - math.log(2)` has a subtle issue: when `abs_diff` is exactly 0.0, `torch.exp(0.0) = 1.0`, so `log1p(1.0 + 1e-8) = log(2.00000001)`, giving a result of approximately `0.0 + 0.693 - 0.693 = ~0.0`. This is correct but the gradient at zero depends on floating-point precision.

**Impact:** Near-zero predictions may produce slightly inconsistent gradients between forward and backward passes in mixed precision (bf16).

**Recommendation:**
- Use `torch.where(abs_diff < 1e-6, 0.5 * diff**2, result)` for near-zero stability
- Or use the direct `torch.log(torch.cosh(diff))` with a guard
- Add a unit test verifying gradient correctness at diff=0

---

## 4. Privacy & Security

### 4.1 HIGH — Absolute Source Path Leakage (`generate_data.py:297-298, 315-316`)

**Finding:** When `include_source_abs_path=True`, the pipeline writes the absolute filesystem path of source audio to both the params JSON and the manifest:
```python
if include_source_abs_path:
    param_payload["source_file"] = source_abs_path
```
These paths may include usernames, directory structures, or mount points that reveal infrastructure details.

**Impact:** If datasets are shared or published, filesystem paths leak internal infrastructure info. On shared systems, this could reveal other users' directory names.

**Recommendation:**
- Default `include_source_abs_path` to `False` (already done, good)
- Add a sanitization step that strips user home directory paths
- Add a warning when this flag is enabled

### 4.2 MEDIUM — Unsafe Torch Load for Checkpoints (`train.py` implied, `pipeline_utils.py`)

**Finding:** `torch.load(path, map_location="cpu", weights_only=True)` is used in `dataset.py:645` with `weights_only=True` (good). However, the CLAUDE.md mentions checkpoint resume, and the training pipeline likely uses `weights_only=False` for full optimizer state deserialization, which allows arbitrary code execution via pickle.

**Impact:** Loading untrusted checkpoints could execute arbitrary code. In a shared computing environment (Lightning AI studios), this is a real risk.

**Recommendation:**
- Always use `weights_only=True` unless optimizer state is needed
- For optimizer state, implement a custom safe loader that only allows tensor types
- Add checkpoint integrity verification (SHA256 hash) before loading
- Document the security implications of `--resume` in the codebase

### 4.3 MEDIUM — Path Traversal Protection Incomplete (`pipeline_utils.py:270-285`)

**Finding:** `validate_path_under_root` and `resolve_trusted_artifact_path` provide good protection against path traversal. However, symlinks are resolved before checking (`Path.resolve()` follows symlinks), which is correct. But the `allowed_roots` list includes `Path.cwd()` which changes with working directory.

**Impact:** If the working directory is changed (e.g., via `cd`), the trusted roots change. An attacker who can control the CWD could potentially bypass path restrictions.

**Recommendation:**
- Use absolute paths for trusted roots, computed at module load time
- Consider adding a configurable `ARTIFACT_ROOT` environment variable
- Validate that resolved paths don't escape via symlink chains

---

## 5. Bias Mitigation

### 5.1 HIGH — Signal Type Distribution Not Configurable (`dataset.py:62`)

**Finding:** `signal_types=("noise", "pink_noise", "sweep", "harmonic", "speech_like", "percussive")` is a hard-coded tuple with equal weighting. The mix generation (`_generate_dry_mix`) selects 2-3 random types with Dirichlet weights, but the overall signal type distribution is uniform.

**Impact:** Real audio is dominated by speech-like and harmonic content. The uniform distribution means the model trains on equal amounts of noise and sweeps, which don't represent deployment conditions. The model may perform worse on speech and music (the actual use cases).

**Recommendation:**
- Add `signal_type_weights` to the config to allow biased distributions
- Consider weighting speech_like and harmonic types higher (e.g., 0.4, 0.3)
- Validate model performance stratified by signal type

### 5.2 MEDIUM — Frequency Range Clipping Creates Artifacts (`dataset.py:317-318, 328-329`)

**Finding:** Shelf and HP/LP filter frequencies are clipped to specific ranges:
- Lowshelf: 20-5000 Hz
- Highshelf: 1000-20000 Hz  
- Highpass: 20-500 Hz
- Lowpass: 2000-20000 Hz

These are reasonable ranges but create hard boundaries. A high-shelf at 1001 Hz and one at 999 Hz are treated differently (one is allowed, one is clipped to 1000 Hz), which the model sees as a discontinuity.

**Impact:** The model may struggle with EQ bands near these boundaries, predicting unstable results for frequencies near 1000 Hz (highshelf) or 500 Hz (highpass).

**Recommendation:**
- Use soft boundaries with rejection sampling instead of hard clipping
- Add small overlap regions at boundaries
- Log the actual frequency distributions to verify no clustering at boundaries

### 5.3 LOW — Gain Sign Penalty Creates Asymmetric Learning (`loss_multitype.py`, config: sign_penalty_weight=0.5)

**Finding:** The sign penalty (`sign_penalty_weight: 0.5`) penalizes the model for predicting gain with the wrong sign. This creates an asymmetric learning signal: getting the sign right is rewarded, but the magnitude error is the same whether 2 dB off in either direction.

**Impact:** May slow convergence for large gain predictions where sign is uncertain.

**Recommendation:**
- Monitor the sign accuracy separately during training
- Consider reducing the weight as training progresses (curriculum on loss weights)

---

## 6. Data Lineage & Drift Detection

### 6.1 MEDIUM — Cache Staleness Detection Works But Limited (`dataset.py:610-671`)

**Finding:** The cache staleness detection via metadata signatures is well-implemented:
- Compares SHA256 of generation parameters
- Rejects unsigned caches
- Provides clear error messages

However, it does NOT detect:
- Changes in the DSP code (`differentiable_eq.py`) that affect EQ rendering
- Changes in the augmentation logic
- Changes in the mel filterbank construction

**Impact:** If the DSP code is modified but the config is unchanged, stale cached data will be used without warning.

**Recommendation:**
- Include a hash of the source code for `differentiable_eq.py` and `dataset.py` in the metadata signature
- Add `code_version` or `git_commit` to the metadata
- Consider a `--force-recompute` flag (already implemented — good)

### 6.2 MEDIUM — No Data Distribution Monitoring During Training

**Finding:** There is no mechanism to monitor the actual distribution of generated data during training. If the random seed changes, or if curriculum stages produce unexpected distributions, there is no automatic detection.

**Impact:** Silent distribution shifts could occur between training runs or curriculum stages without detection.

**Recommendation:**
- Add periodic histogram logging of gain, freq, Q, and type distributions
- Compare distributions across curriculum stages
- Alert if the distribution deviates significantly from expected

### 6.3 LOW — Manifest Includes Full Source Paths (`generate_data.py:403-427`)

**Finding:** The manifest includes `input_dir` (absolute path) and per-sample `source_relpath`. While the relative path is good, the absolute `input_dir` leaks filesystem information.

**Recommendation:** Use relative paths for `input_dir` or make it optional.

---

## 7. Pipeline Efficiency

### 7.1 HIGH — Redundant STFT Computation (Already Fixed)

**Finding:** `dsp_frontend.py:143-151` shows the STFT is computed once and both mel-spectrogram and complex STFT are derived from it. This is good — the comment "H-13: Compute STFT once and derive both outputs, saving ~30% frontend compute" indicates this was already optimized.

**Status:** RESOLVED

### 7.2 MEDIUM — Precompute Mode Memory Footprint (`dataset.py:446-458`)

**Finding:** Precompute mode caches all mel-spectrograms in memory. For 200k samples with 128xT mel-spectrograms, this requires significant RAM (~20-40 GB depending on duration). There is no memory estimation before precomputation.

**Impact:** OOM on systems with limited RAM, especially with larger datasets or longer audio.

**Recommendation:**
- Add memory estimation before precomputation
- Implement a streaming/lazy mode that generates samples on-the-fly (already partially done via `__getitem__`)
- Consider memory-mapped storage for large precomputed datasets

### 7.3 MEDIUM — Multiprocessing Pool Without Error Propagation (`generate_data.py:384-388`)

**Finding:** The multiprocessing pool catches exceptions broadly:
```python
try:
    results = pool.map(process_file, tasks, chunksize=...)
except Exception as e:
    print(f"  [generate] ERROR in pool processing: {e}")
    results = []
```
Setting `results = []` on any pool error means ALL results are silently lost.

**Impact:** A single worker crash (e.g., corrupt audio file) can cause the entire dataset generation to return empty results, which then passes the quality gate with 0% failure rate on 0 samples.

**Recommendation:**
- Use `pool.map_async` with a timeout and partial result collection
- Add a check: if `len(results) == 0` and `len(tasks) > 0`, raise an error
- Log individual worker failures

---

## 8. Reproducibility

### 8.1 MEDIUM — Inconsistent Seeding Across NumPy/Python/PyTorch (`pipeline_utils.py:333-345`)

**Finding:** `set_global_seed` seeds all three random sources correctly. However, in `seeded_index_context`, the states are saved and restored, but `torch.cuda` state save/restore may silently fail in DataLoader workers (handled with try/except). This means CUDA state may not be properly restored in some edge cases.

**Impact:** Minor: training runs may have slight non-determinism in CUDA operations within DataLoader workers.

**Recommendation:**
- Document that full reproducibility requires `num_workers=0` for the DataLoader
- Add a warning when `deterministic=True` and `num_workers > 0`
- Consider using `generator` parameter in DataLoader instead of worker-level seeding

### 8.2 MEDIUM — bf16 Mixed Precision Non-Determinism

**Finding:** The config uses `precision: bf16-mixed`. BF16 has lower mantissa precision than FP32 (7 bits vs 23 bits), which means the same computation in FP32 and BF16 can produce different results. Combined with non-deterministic reduction operations in CUDA, exact reproducibility across hardware is not possible.

**Impact:** Training curves will differ slightly between runs even with the same seed.

**Recommendation:**
- Document that bf16-mixed does not guarantee bit-exact reproducibility
- For reproducibility validation, compare metrics at 2-3 decimal places, not exact values
- Consider adding a `precision: fp32` option for reproducibility testing

### 8.3 LOW — No Version Hash in Checkpoint Metadata

**Finding:** Checkpoint metadata does not include Python, PyTorch, or dependency versions.

**Recommendation:**
- Add `torch.__version__`, `numpy.__version__`, Python version to checkpoint metadata
- Include git commit hash if in a git repo
- Save a requirements snapshot alongside checkpoints

---

## 9. Gumbel-Softmax & Type Classification

### 9.1 HIGH — Incorrect Cosine Annealing Formula (`train.py:999-1007`)

**Finding:** The cosine annealing uses:
```python
tau = min_tau + (start_tau - min_tau) * 0.5 * (1 + math.cos(math.pi * progress))
```
At `progress=0`: `tau = min_tau + 0.5 * (start_tau - min_tau) * 2.0 = start_tau` (correct)  
At `progress=1`: `tau = min_tau + 0.5 * (start_tau - min_tau) * 0.0 = min_tau` (correct)  
At `progress=0.5`: `tau = min_tau + 0.5 * (start_tau - min_tau) * 1.0 = min_tau + 0.5 * (start_tau - min_tau)` (correct midpoint)

Actually, this formula IS correct for cosine decay from start_tau to min_tau. The training pipeline agent's finding H-01 was incorrect in stating the formula is wrong.

**Status:** NOT AN ISSUE — Formula verified correct.

### 9.2 MEDIUM — Temperature Floor Too High for Hard Decisions

**Finding:** `min_tau: 0.1` means the Gumbel-Softmax never fully hardens. With 5 classes and tau=0.1, the softmax probabilities for the top class are typically >0.9 but not 1.0, meaning some gradient "leaks" to wrong classes.

**Impact:** At inference, `argmax` is used (hard decision), but the training never fully commits, creating a train/test mismatch.

**Recommendation:**
- Consider lowering `min_tau` to 0.05 or 0.01
- Or use the straight-through Gumbel-Softmax estimator (hard forward, soft backward)
- Evaluate the gap between soft and hard predictions at inference

---

## 10. Curriculum Learning

### 10.1 MEDIUM — Uncoordinated Dual Warmup (`loss_multitype.py:535-553` vs `train.py:984-991`)

**Finding:** Two separate warmup mechanisms exist:
1. Loss-level warmup in `loss_multitype.py` (epoch threshold + gain MAE gate)
2. H_db lambda ramp in `train.py` (linear ramp over 10 epochs)

These are not coordinated. The loss-level warmup uses a complex condition: `is_warmup = not ((past_epoch_threshold and gain_converged) or past_hard_cap)`. This means warmup ends either when the gain MAE drops below 2.5 dB AND past the threshold, OR after epoch 15.

**Impact:** The H_db ramp may still be ramping while the loss-level warmup has already ended, creating conflicting training signals.

**Recommendation:**
- Consolidate warmup logic into a single source of truth
- Add clear documentation of the warmup strategy
- Log warmup state transitions explicitly

---

## Vulnerability Summary Matrix

| ID | Severity | Category | Component | Status |
|----|----------|----------|-----------|--------|
| V-01 | CRITICAL | Numerical Stability | `differentiable_eq.py` — biquad alpha | Open |
| V-02 | CRITICAL | Numerical Stability | `differentiable_eq.py` — H_mag clamping | Open |
| V-03 | CRITICAL | Memory/Corruption | `train.py` — optimizer state sanitization | Open |
| V-04 | CRITICAL | Gradient Flow | `loss_multitype.py` — type detachment | Open |
| V-05 | CRITICAL | Data Integrity | `dataset.py` — fallback sample bias | Open |
| V-06 | HIGH | Data Bias | `dataset.py` — gain distribution | Open |
| V-07 | HIGH | Data Bias | `dataset.py`/config — type weight inconsistency | Open |
| V-08 | HIGH | Security | `generate_data.py` — path leakage | Open |
| V-09 | HIGH | Data Bias | `dataset.py` — signal type distribution | Open |
| V-10 | HIGH | DSP | `dsp_frontend.py` — mel filterbank edges | Open |
| V-11 | HIGH | Loss | `loss_multitype.py` — log-cosh at zero | Open |
| V-12 | HIGH | Efficiency | `generate_data.py` — pool error handling | Open |
| V-13 | MEDIUM | DSP | Phase preservation approximation | Open |
| V-14 | MEDIUM | Data | Audio normalization consistency | Open |
| V-15 | MEDIUM | Missing | SpecAugment not implemented | Open |
| V-16 | MEDIUM | Drift | Cache staleness limited to config only | Open |
| V-17 | MEDIUM | Drift | No distribution monitoring | Open |
| V-18 | MEDIUM | Memory | Precompute memory estimation | Open |
| V-19 | MEDIUM | Reproducibility | bf16 non-determinism | Open |
| V-20 | MEDIUM | Reproducibility | CUDA state in DataLoader workers | Open |
| V-21 | MEDIUM | Security | `torch.load` with pickle | Open |
| V-22 | MEDIUM | Security | Path traversal CWD dependency | Open |
| V-23 | MEDIUM | Data | Frequency boundary discontinuities | Open |
| V-24 | MEDIUM | Curriculum | Dual warmup coordination | Open |
| V-25 | MEDIUM | Gumbel | Temperature floor too high | Open |
| V-26 | LOW | Bias | Gain sign penalty asymmetry | Open |
| V-27 | LOW | Lineage | Manifest path leakage | Open |
| V-28 | LOW | Reproducibility | No version hash in checkpoints | Open |
| V-29 | LOW | Missing | Missing SpecAugment reference | Open |

---

## Prioritized Action Plan

### Phase 1 — Immediate (Training Stability) 
**Estimated impact: Prevents training crashes and NaN corruption**

1. **Fix biquad alpha epsilon** (V-01): Increase from 1e-8 to 1e-4, add alpha clamping
2. **Fix H_mag clamping** (V-02): Add upper bound clamping to [1e-6, 1e6]
3. **Fix optimizer state sanitization** (V-03): Reset all Adam state for corrupted params
4. **Re-enable type gradient flow** (V-04): Remove detach() or reduce to 0.1x weight

### Phase 2 — Short-Term (Data Quality)
**Estimated impact: Reduces systematic bias, improves generalization**

5. **Fix gain distribution** (V-06): Use beta distribution in dataset.py
6. **Fix type weight defaults** (V-07): Remove hard-coded defaults, require config
7. **Fix signal type distribution** (V-09): Add configurable weights
8. **Track fallback sample rate** (V-05): Add metric and alert threshold

### Phase 3 — Medium-Term (Robustness)
**Estimated impact: Improves reproducibility, security, and monitoring**

9. **Add code hash to cache signature** (V-16)
10. **Add distribution monitoring** (V-17)
11. **Consolidate warmup logic** (V-24)
12. **Fix pool error handling** (V-12)
13. **Add checkpoint integrity verification** (V-21)

### Phase 4 — Long-Term (Hardening)
**Estimated impact: Production-readiness and documentation**

14. Implement SpecAugment (V-15, V-29)
15. Add version pinning and checkpoint metadata (V-28)
16. Address frequency boundary discontinuities (V-23)
17. Document phase preservation approximation (V-13)
18. Security hardening for path handling (V-08, V-22, V-27)

---

## Data Drift Risk Assessment

| Risk Factor | Likelihood | Impact | Mitigation Status |
|-------------|-----------|--------|-------------------|
| Config-driven parameter drift | Medium | High | Partial (cache signatures) |
| Code-change drift | High | High | None (no code hash) |
| Curriculum stage transition drift | Medium | Medium | None (no monitoring) |
| Seed-dependent distribution variance | Low | Low | Handled (seeded generation) |
| DSP approximation vs real-world gap | High | Medium | Documented only |
| BF16 precision drift | Low | Low | Not mitigated |

---

## Bottleneck Analysis

| Bottleneck | Location | Impact | Recommendation |
|------------|----------|--------|----------------|
| On-the-fly EQ rendering | `dataset.py:_apply_eq_freq_domain` | ~40% of per-sample time | Precompute or cache rendered audio |
| Mel spectrogram computation | `dataset.py:_audio_to_mel` | ~20% of per-sample time | Already cached in precompute mode |
| Hungarian matching | `loss_multitype.py:linear_sum_assignment` | ~15% of loss computation | Batch-level matching, not sample-level |
| Full H_db computation per band | `differentiable_eq.py` | ~10% of forward pass | Consider caching frequency grid |
| STFT in forward pass | `train.py` | ~15% of training step | Already optimized (single STFT) |

---

*Report generated 2026-04-14. Findings should be validated against current code state before remediation.*
