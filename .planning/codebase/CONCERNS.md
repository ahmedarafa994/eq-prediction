# Codebase Concerns

**Analysis Date:** 2026-04-05

## Tech Debt

**Chronic NaN/Inf instability in training pipeline:**
- Issue: The training pipeline has pervasive NaN/Inf issues that have been addressed through symptom patches rather than root-cause fixes. There are 7 separate `patch_*.py` scripts (`patch_loss.py`, `patch_loss2.py`, `patch_loss3.py`, `patch_train.py`, `patch_train2.py`, `patch_train_val.py`, `patch_train_val_loss.py`) that perform string-replacement surgery on `train.py` and `loss_multitype.py` to inject `torch.nan_to_num()` guards. These patches modify source files at runtime via regex, which is fragile and unversioned.
- Files: `insight/patch_loss.py`, `insight/patch_loss2.py`, `insight/patch_loss3.py`, `insight/patch_train.py`, `insight/patch_train2.py`, `insight/patch_train_val.py`, `insight/patch_train_val_loss.py`
- Impact: Training produces NaN batches frequently enough that the trainer has dedicated NaN-counting, NaN-recovery, and NaN-batch-skipping logic throughout `train.py` (lines 390-469, 528-532, 656-668, 738-839). The `MultiTypeEQParameterHead` also has `nan_to_num` guards on its outputs (`differentiable_eq.py` lines 757, 783, 805), indicating the model itself generates NaN during forward passes. This masks real numerical bugs.
- Fix approach: (1) Root-cause the NaN by adding per-layer fp32 casting in the biquad coefficient computation under bf16 mixed precision. (2) Replace the patch scripts with a single `GradientHealthMonitor` callback that logs which layer first produces NaN. (3) Remove `nan_to_num` guards from the model forward pass and instead fix the underlying instability in `forward_soft` where blended biquad coefficients under bf16 overflow.

**Duplicate code between `dataset.py` and `dataset_musdb.py`:**
- Issue: `SyntheticEQDataset` (`dataset.py`) and `MUSDB18EQDataset` (`dataset_musdb.py`) share ~250 lines of identical code: `_sample_multitype_params`, `_sample_gain`, `_log_uniform`, `_apply_eq_freq_domain`, `_audio_to_mel`, `_build_mel_filterbank`, `precompute`, `save_precomputed`, `load_precomputed`. Only the source of dry audio differs (synthetic signals vs. MUSDB18 crops).
- Files: `insight/dataset.py` (405 lines), `insight/dataset_musdb.py` (342 lines)
- Impact: Bug fixes or parameter sampling changes must be applied twice. The type-specific frequency ranges (lowshelf: 20-5000 Hz, highshelf: 1000-20000 Hz, etc.) are duplicated.
- Fix approach: Extract a `MultiTypeEQMixin` base class or composition with the shared methods, and have both datasets inherit or delegate to it.

**Abandoned/unused modules left in tree:**
- Issue: Several files represent abandoned approaches that are not used by the primary training pipeline but still import from core modules: `model_cnn.py` (peaking-only CNN, superseded by TCN), `loss.py` (`EQLoss`, `CycleConsistencyLoss`, `CombinedIDSPLoss` -- not used by `train.py` which uses `loss_multitype.py`), `fixes/gain_fixes.py`, `fixes/modified_head.py`, `fixes/modified_loss.py` (alternative implementations never integrated).
- Files: `insight/model_cnn.py`, `insight/loss.py`, `insight/fixes/gain_fixes.py`, `insight/fixes/modified_head.py`, `insight/fixes/modified_loss.py`
- Impact: Confuses contributors about which code is active. Import changes in `differentiable_eq.py` or `model_tcn.py` may break these dormant files without anyone noticing.
- Fix approach: Move inactive modules to `insight/legacy/` or remove them. Keep `loss.py` only if `MultiResolutionSTFTLoss` is still imported by `loss_multitype.py` (it is -- line 19 -- so extract `MultiResolutionSTFTLoss` into `loss_multitype.py` or a shared module first).

**Duplicate attention pool check in `model_tcn.py`:**
- Issue: `AttentionTemporalPool.forward()` has the same fused-kernel check repeated twice (lines 315-316 and 318-319). The second block is dead code since `HAS_FUSED_KERNELS` is hardcoded to `False`.
- Files: `insight/model_tcn.py` lines 315-319
- Impact: Minor, but indicative of incomplete cleanup after disabling Triton kernels.
- Fix approach: Remove the duplicate check and the dead `HAS_FUSED_KERNELS` branching entirely since the flag is `False`.

**Large diagnostic scripts committed to main codebase:**
- Issue: `diagnose_gain.py` (904 lines), `diagnose_gradients.py` (840 lines), `audit_loss.py` (305 lines), `debug_attn.py`, `debug_peaks.py` are one-off diagnostic scripts that contain hardcoded checkpoint paths and ad-hoc analysis logic. They are mixed into the main source directory alongside production code.
- Files: `insight/diagnose_gain.py`, `insight/diagnose_gradients.py`, `insight/audit_loss.py`, `insight/debug_attn.py`, `insight/debug_peaks.py`
- Impact: Clutters the source tree. Some scripts may be broken if model APIs have changed since they were written.
- Fix approach: Move to `insight/tools/` or `insight/diagnostics/` directory.

## Known Bugs

**`training/evaluate.py` uses nonexistent classmethod:**
- Symptoms: `training/evaluate.py` line 179 calls `StreamingTCNModel.load_from_checkpoint(...)` which does not exist. `StreamingTCNModel` extends `nn.Module`, not `LightningModule`. This script will crash at runtime.
- Files: `insight/training/evaluate.py` line 179
- Trigger: Running `python training/evaluate.py --checkpoint ... --data_dir ...`
- Workaround: Load the checkpoint manually via `torch.load` and `model.load_state_dict()`.

**Gumbel temperature floor mismatch:**
- Symptoms: The curriculum config in `conf/config.yaml` line 98 sets `gumbel_temperature: 0.05` for the "finetune" stage, but `MultiTypeEQParameterHead` has `self.min_tau = 0.1` (line 618) and clamps via `torch.clamp(self.gumbel_temperature, min=self.min_tau)` (line 824). The trainer also applies `current_tau = max(current_tau, 0.1)` (train.py line 884). So the config value of 0.05 is never actually used.
- Files: `insight/conf/config.yaml` line 98, `insight/differentiable_eq.py` line 618, `insight/train.py` line 884
- Trigger: Any training run that reaches the finetune curriculum stage.
- Workaround: The clamp prevents harm but the config is misleading. Anyone reading the config would think tau goes to 0.05.

**`loss.py` `CombinedIDSPLoss` device mismatch:**
- Symptoms: `CombinedIDSPLoss.forward()` at line 223 creates `total_loss = torch.tensor(0.0, device=device)` where `device` is determined from the first non-None input tensor. If all inputs are None, `device` remains None, producing a CPU tensor. More critically, if only `gain_db` is passed (line 221 checks `gain_db is not None` first), but other inputs are on different devices, this can fail.
- Files: `insight/loss.py` lines 217-223
- Trigger: Calling `CombinedIDSPLoss` with only partial inputs on different devices.
- Workaround: Always pass at least one tensor on the correct device.

## Security Considerations

**Unsafe `torch.load` with `weights_only=False`:**
- Risk: All checkpoint and dataset loading calls use `weights_only=False` (`train.py` lines 751, 794; `dataset.py` line 355; `dataset_musdb.py` line 339; `test_checkpoint.py` line 181; `test_checkpoint_multi.py` line 128; `diagnose_gain.py` line 733). This allows arbitrary code execution via pickle deserialization if a checkpoint file is tampered with.
- Files: `insight/train.py`, `insight/dataset.py`, `insight/dataset_musdb.py`, `insight/test_checkpoint.py`, `insight/test_checkpoint_multi.py`, `insight/diagnose_gain.py`
- Current mitigation: No external checkpoint downloads; all files are locally generated.
- Recommendations: For dataset caches (`.pt` files), migrate to `weights_only=True` with an allowlist of safe types. For model checkpoints, use `safetensors` format instead of pickle-based `.pt` files.

**Patch scripts modify source files at runtime:**
- Risk: The `patch_*.py` scripts perform unvalidated string replacement on production source files (`train.py`, `loss_multitype.py`). Running them could corrupt source code silently if the target strings have changed since the patches were written.
- Files: `insight/patch_loss.py`, `insight/patch_loss2.py`, `insight/patch_loss3.py`, `insight/patch_train.py`, `insight/patch_train2.py`, `insight/patch_train_val.py`, `insight/patch_train_val_loss.py`
- Current mitigation: Scripts check if the target string exists before patching and print a message if not found.
- Recommendations: Delete all patch scripts. Integrate their NaN guards (if still needed) directly into the source files with proper version control.

**Hardcoded filesystem paths in config:**
- Risk: `conf/config_musdb_200k.yaml` line 9 contains an absolute path `/teamspace/lightning_storage/SNAP_DATA_SET/musdb18_hq` specific to one developer's environment. The precompute cache path `data/dataset_musdb_200k.pt` in config.yaml line 73 is relative but assumes the working directory is always `insight/`.
- Files: `insight/conf/config_musdb_200k.yaml` line 9, `insight/conf/config.yaml` line 73
- Current mitigation: The primary `config.yaml` uses synthetic data by default and does not require MUSDB18 paths.
- Recommendations: Use environment variable expansion (e.g., `${MUSDB_ROOT}`) or command-line arguments for filesystem paths.

## Performance Bottlenecks

**Hungarian matching on CPU in training hot loop:**
- Problem: `HungarianBandMatcher.match()` iterates over each batch element and calls `scipy.optimize.linear_sum_assignment` on CPU with `.detach().cpu().numpy()` transfer. With batch_size=1024, this means 1024 separate scipy calls per training step.
- Files: `insight/loss_multitype.py` lines 106-114
- Cause: scipy has no GPU implementation. The per-element loop with CPU transfer is inherently slow.
- Improvement path: Implement a batched PyTorch-native assignment solver, or use the Sinkhorn-Knopp approximation from the DETR paper. The current implementation accounts for a significant fraction of per-step time at large batch sizes.

**`torch.where` nested 4-deep for coefficient selection:**
- Problem: `compute_biquad_coeffs_multitype` in `differentiable_eq.py` uses 5 coefficients (b0, b1, b2, a0, a1) each computed via 4-level nested `torch.where` (lines 148-226). This creates 5 separate branching trees, each computing all 5 filter types and selecting one. Total: 25 filter-type computations, 20 of which are discarded.
- Files: `insight/differentiable_eq.py` lines 86-236
- Cause: `torch.where` evaluates both branches (no short-circuiting). All branches must exist for gradient flow.
- Improvement path: The `forward_soft` variant (line 238) already uses the more efficient weighted-sum approach. Consider making `forward_soft` the only path during training and removing the hard-typed path except for final inference.

**Massive precomputed dataset files on disk:**
- Problem: `data/dataset_musdb_200k.pt` is 26.8 GB. The total data directory is ~36 GB. These are stored as uncompressed pickle tensors.
- Files: `insight/data/dataset_musdb_200k.pt` (26.8 GB), `insight/data/dataset_musdb_50k.pt` (6.7 GB), `insight/data/dataset_musdb_10k.pt` (1.3 GB)
- Cause: Precomputed mel-spectrograms for 200K samples with 128 mel bins at full time resolution.
- Improvement path: Use memory-mapped tensors or on-the-fly computation with a fast DataLoader. Delete intermediate dataset sizes that are no longer needed (10k, 50k).

**SpecAugment implemented with Python-level per-batch-element loop:**
- Problem: `_spec_augment` in `train.py` lines 349-381 uses a Python `for i in range(B)` loop to apply masking per sample, creating B sequential operations instead of a vectorized batch operation.
- Files: `insight/train.py` lines 349-381
- Cause: `torch.randint` generates different mask parameters per sample, and the loop applies them individually.
- Improvement path: Use vectorized masking with `torch.arange` broadcasting, or use `torchaudio.transforms.FrequencyMasking` and `TimeMasking` which handle batching internally.

## Fragile Areas

**Encoder collapse detection and recovery:**
- Files: `insight/model_tcn.py` (anti-collapse hooks lines 27-29), `insight/train.py` (collapse detection lines 570-573, NaN recovery lines 784-839), `insight/loss_multitype.py` (embed_var_loss lines 332-344, contrastive_loss lines 354-371)
- Why fragile: The entire hybrid 2D+TCN architecture was built to fix a catastrophic encoder collapse in the original pure-1D TCN (cosine distance 0.006 between embeddings). Multiple anti-collapse mechanisms are stacked: embedding variance regularization, contrastive loss, spectral residual bypass, attention pooling. If any one of these is weakened, collapse can recur. The spectral bypass in `model_tcn.py` line 464 (`mel_profile = mel_frames.mean(dim=-1)`) is the ultimate safety net -- the param head can fall back to raw spectral features.
- Safe modification: Do not remove any of the three anti-collapse losses simultaneously. Reduce weights gradually while monitoring embedding variance. Never remove the `mel_profile` bypass.
- Test coverage: No automated test verifies that encoder collapse does not recur. The `embedding_variance()` method exists for diagnostics but is not called in any test.

**Gumbel-Softmax temperature annealing across curriculum stages:**
- Files: `insight/differentiable_eq.py` (Gumbel sampling lines 817-828), `insight/train.py` (annealing lines 870-884)
- Why fragile: The Gumbel temperature is annealed from 1.5 to 0.05 across 4 curriculum stages via `self.model.param_head.gumbel_temperature.fill_(current_tau)`. At low temperatures, Gumbel-Softmax approaches argmax and gradients can vanish. Also, the annealing happens mid-epoch when the stage changes, which can cause a sudden gradient shift.
- Safe modification: Keep the minimum tau at 0.1 or above. Use smooth exponential annealing instead of stage-boundary jumps. Test gradient magnitudes at each temperature.
- Test coverage: `test_multitype_eq.py` tests Gumbel-Softmax output shapes but not gradient flow at different temperatures.

**`MultiTypeEQParameterHead` parameter initialization coupling:**
- Files: `insight/differentiable_eq.py` lines 531-696
- Why fragile: The parameter head has ~15 sub-modules with carefully tuned initializations: `gain_output_scale` initialized to 24.0, `gain_blend_gate` at 0.8473 (matching a previous fixed 0.7/0.3 ratio), `freq_prior_raw` as linear spacing from 0.1 to 0.9, `attn_position_bias` as Gaussian centers. Changing any of these can destabilize training because the parameter head must produce outputs in very specific ranges (gain: +/-24 dB, freq: 20-20000 Hz, Q: 0.1-10.0).
- Safe modification: Initialization changes should be tested with a short training run (5-10 epochs) before committing. Monitor `grad_gain`, `grad_freq`, and `grad_q` in logs to verify gradient flow is not attenuated.

## Scaling Limits

**Precomputed dataset in-memory storage:**
- Current capacity: The `precompute()` method stores all samples in a Python list (`self._cache`). For 200K samples with mel-spectrograms, this requires ~12-15 GB of RAM during training.
- Limit: Machines with less than 32 GB RAM cannot train with the full 200K dataset. The `.pt` files on disk are also very large (26.8 GB for 200K).
- Scaling path: Use memory-mapped loading (`torch.load(..., mmap=True)` available in PyTorch 2.1+) or switch to streaming data generation that does not cache.

**Checkpoint storage grows linearly:**
- Current capacity: Each epoch checkpoint is ~22 MB. With 120 epochs and no cleanup, the `checkpoints/` directory would accumulate ~2.6 GB.
- Limit: The current run shows 25 epoch checkpoints (550 MB) after 25 epochs. No automatic cleanup exists in the primary trainer.
- Scaling path: Add `max_keep` logic to `save_checkpoint()` to retain only the best and last N checkpoints.

**Single-GPU training only:**
- Current capacity: The trainer uses `devices: 1` and has optional DeepSpeed ZeRO-2 integration that is disabled by default.
- Limit: Training on a single GPU with batch_size=1024 limits the effective model size and data throughput.
- Scaling path: The DeepSpeed integration exists (`train.py` lines 314-332) but is untested end-to-end. Multi-GPU would require verifying the Hungarian matching under distributed gradient synchronization.

## Dependencies at Risk

**Triton kernels permanently disabled:**
- Risk: The fused Triton kernels in `fused_kernels.py` are disabled (`HAS_TRITON = False` on line 19) due to API incompatibilities with Triton 3.4.0 (`tl.math.tanh` removed). The code is maintained but dead.
- Impact: No current runtime impact since PyTorch fallbacks work. But `FusedConvBNGELU`, `QuantizedTCNStack`, and `QuantizedActivation` classes are unused and confusing.
- Migration plan: Either fix the Triton kernels for the current Triton version or remove them entirely.

**PyTorch Lightning integration is out of sync:**
- Risk: The `training/` directory contains a full Lightning alternative (`lightning_module.py`, `train.py`, `curriculum.py`, `evaluate.py`) but it is out of sync with the primary trainer. The Lightning module does not have the encoder-collapse fixes, uses an older loss configuration, and `evaluate.py` is broken (wrong classmethod call).
- Impact: The Lightning training path does not work without fixes.
- Migration plan: Decide whether to maintain the Lightning path or remove it. If removing, delete `insight/training/` and update CLAUDE.md. If keeping, synchronize with the primary trainer's architecture and loss configuration.

**scipy as heavy dependency for one function:**
- Risk: `scipy.optimize.linear_sum_assignment` is the only scipy usage in the training hot path. scipy is a ~30 MB dependency used for a single function that could be replaced.
- Impact: Increases Docker image size and install time. scipy version conflicts can block installation.
- Migration plan: Implement a batched PyTorch-native assignment solver or use the Sinkhorn-Knopp approximation.

## Missing Critical Features

**No standalone real-audio evaluation pipeline:**
- Problem: There is no `evaluate_model.py` or `test_real_audio.py` in the repository (referenced in CLAUDE.md but not present). The only evaluation is embedded in `train.py`'s `validate()` method which uses synthetic data. The `training/evaluate.py` script exists but is broken.
- What's missing: A standalone evaluation script that loads a checkpoint, runs inference on real audio with known EQ settings, and produces metrics comparable to published results.
- Blocks: Production validation and benchmarking against real-world audio.

**No ONNX export validation for streaming mode:**
- Problem: `export.py` only exports the batch-mode forward pass. The streaming `process_frame()` path is not tested for ONNX compatibility. The model uses dynamic buffers (`_streaming_buffer`) and stateful operations that may not export cleanly.
- What's missing: Streaming ONNX export and validation that batch and streaming inference produce matching outputs.
- Blocks: DAW plugin deployment.

## Test Coverage Gaps

**No test for encoder collapse prevention:**
- What's not tested: The anti-collapse mechanisms (embedding variance loss, contrastive loss, spectral bypass) have no regression test. There is no test verifying embeddings remain diverse across a batch of different inputs.
- Files: `insight/test_model.py`, `insight/test_streaming.py`
- Risk: A refactoring change could silently reintroduce encoder collapse without any test catching it.
- Priority: High -- this was the primary failure mode of the original architecture.

**No test for the primary training pipeline:**
- What's not tested: The `Trainer` class in `train.py` (1019 lines) has no unit test. The existing `test_lightning_dummy.py` tests the Lightning variant only. There is no test for curriculum stage transitions, NaN recovery, checkpoint save/load, or optimizer state restoration.
- Files: `insight/train.py`
- Risk: Refactoring the trainer could break training without detection. The NaN recovery path (`_recover_from_nan`) is particularly risky since it involves loading checkpoints and resetting optimizer state.
- Priority: High -- the trainer is the largest and most complex file.

**No test for `loss_multitype.py` in isolation:**
- What's not tested: `MultiTypeEQLoss` and `HungarianBandMatcher` have no dedicated test. The loss has 10+ components with complex interactions (permutation matching, contrastive loss, spread regularization). Edge cases like all-identical predictions or all-zero predictions are not tested.
- Files: `insight/loss_multitype.py`
- Risk: Loss function changes could introduce subtle bugs that manifest as training degradation rather than crashes.
- Priority: Medium.

**No test for MUSDB18 dataset loading:**
- What's not tested: `MUSDB18EQDataset` has no test. It depends on `torchaudio` and specific MUSDB18 file structure. The dataset's precompute/load cycle is untested.
- Files: `insight/dataset_musdb.py`
- Risk: Changes to the dataset pipeline could corrupt precomputed caches silently.
- Priority: Low -- the dataset is only used for optional real-audio training.

---

*Concerns audit: 2026-04-05*
