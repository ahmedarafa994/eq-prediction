# Phase 3: Loss Architecture Restructuring - Research

**Researched:** 2026-04-06
**Domain:** Loss function design, gradient routing, curriculum warmup in differentiable DSP training
**Confidence:** HIGH

## Summary

Phase 3 restructures the loss function to direct gradient signal to gain regression first, then progressively activate spectral and other losses. The investigation reveals that many of the required features are already partially or fully implemented in the codebase, but several critical wiring gaps exist. The model already outputs both `H_mag_hard` (argmax types) and `H_mag` (soft Gumbel) from `model_tcn.py:forward()`, but `train.py` ignores `H_mag_hard` and passes only the soft version (with gain detached) to the loss. The loss function `loss_multitype.py` has warmup gating, log-cosh for gain, and independent per-parameter weights -- all implemented correctly. The two main gaps are: (1) threading `H_mag_hard` through to `hmag_loss` while using `H_mag` soft for `spectral_loss`, and (2) wiring audio reconstruction loss in `train.py` (currently `pred_audio`/`target_audio` are never passed to the criterion). Gumbel detach during warmup is not implemented -- `type_probs` flows freely to all loss terms during warmup.

**Primary recommendation:** Implement the dual forward path wiring first (highest impact -- clean param regression gradient), then wire audio reconstruction, then add Gumbel detach. Much of this phase is wiring existing code, not building new components.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** hmag_loss uses H_mag_hard (argmax types) for clean param comparison. spectral_loss uses H_mag_soft (Gumbel-Softmax) for differentiable type gradients.
- **D-02:** model_tcn.py already has partial split (lines 637-643): `H_mag_hard` computed with argmax, `H_mag` soft during training. Verify loss_multitype.py uses `H_mag_hard` for hmag_loss component.
- **D-03:** Epoch-based warmup: 5 epochs gain-only, then activate freq/Q, then type, then spectral. Keep current phased activation schedule in MultiTypeEQLoss.
- **D-04:** Metric-gated transitions deferred to Phase 4 (DATA-03).
- **D-05:** Gumbel-Softmax type probabilities detached from gain gradient path during warmup only. After warmup, allow joint gradients.
- **D-06:** Spectral-domain MR-STFT: compare H_mag * wet_spectrum vs target_spectrum. No time-domain waveform reconstruction.
- **D-07:** Verify train.py actually passes pred_audio/target_audio to loss. Current code may have the wiring missing -- loss_multitype.py checks for None and zeros out if not provided.
- **D-08:** Use current config.yaml weights as starting point. No grid search needed.
- **D-09:** log_cosh_loss() for gain -- already implemented. Verify it's correctly wired.
- **D-10:** Independent per-parameter weights -- already in total loss computation. Verify no combined lambda_param wrapper leaks.
- **D-11:** Per-band activity weighting (LOSS-06) -- activity_loss exists. Verify it uses correct active_band_mask.

### Claude's Discretion
- Exact implementation of Gumbel detach (detach type_probs tensor vs zero gradient scaling)
- How to wire audio reconstruction in train.py if currently missing
- Test structure and file organization

### Deferred Ideas (OUT OF SCOPE)
- Metric-gated curriculum transitions (gain threshold, freq threshold) -- Phase 4 (DATA-03)
- Loss weight grid search or adaptive weight balancing -- can revisit if warmup alone doesn't break the plateau
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LOSS-01 | Separate loss weights for gain, freq, Q | Already implemented -- `lambda_gain`, `lambda_freq`, `lambda_q` are independent in `loss_multitype.py:498-509`. `lambda_param` set to 0.0 in config disables the combined path. Verified no wrapper leakage. |
| LOSS-02 | Loss phasing -- gain-only warmup before spectral | Already implemented -- `warmup_epochs=5` default, phased activation in `loss_multitype.py:348-351`. Train.py sets `criterion.current_epoch` each epoch at line 976. |
| LOSS-03 | Log-cosh loss for gain regression | Already implemented -- `log_cosh_loss()` at `loss_multitype.py:22-34`, used at line 413. Numerically stable formulation verified. |
| LOSS-04 | Dual forward path -- hard argmax for param loss, soft Gumbel for spectral | **NOT YET WIRED.** Model outputs both `H_mag` and `H_mag_hard` (line 645-651), but train.py only passes soft version. Loss function does not accept a separate `pred_H_mag_hard` parameter. Requires: (1) new parameter in loss forward(), (2) train.py wiring. |
| LOSS-05 | Audio-domain reconstruction loss | **NOT WIRED.** Loss function has MR-STFT code (line 465-471) that zeros out when `pred_audio`/`target_audio` are None. Train.py never passes audio tensors. Need to compute `pred_audio = H_mag * wet_stft_magnitude` and wire through. |
| LOSS-06 | Per-band loss weighting based on band activity | Partially implemented. `activity_loss` exists (line 474-479) and uses `active_band_mask`. However, `active_band_mask` is never passed from `train.py` to the criterion. Dataset provides it in batch but train.py doesn't extract it. |
| DATA-02 | Gumbel-Softmax gradient protection -- detach type probs from gain path during warmup | **NOT IMPLEMENTED.** Currently `type_probs` flows freely to all loss components during warmup. Need to detach `type_probs` when `is_warmup=True` in loss computation, or detach in model forward during warmup epochs. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.8.0+cu128 | Differentiable DSP, loss computation, autograd | Project foundation |
| scipy | 1.11.4 | Hungarian matching (`linear_sum_assignment`) | Only reliable bipartite matching in Python |
| numpy | (bundled with scipy) | Cost matrix conversion for Hungarian matching | scipy dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyyaml | (present) | Config loading | Loss weight config, warmup settings |
| math | (stdlib) | `log(2)` constant for log-cosh | log_cosh_loss implementation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual epoch-based warmup | Learnable loss weights (e.g., Kendall et al. 2018) | Manual is simpler and the project already has curriculum infrastructure |
| Gumbel detach | StopGradient layer | `.detach()` is idiomatic PyTorch, no new abstraction needed |

**Installation:**
No new packages needed for this phase. All dependencies already installed.

**Version verification:**
```
Python 3.12.11
torch=2.8.0+cu128, cuda=True
scipy=1.11.4
```

## Architecture Patterns

### Recommended Project Structure (no changes needed)
```
insight/
├── loss_multitype.py    # Main loss — add pred_H_mag_hard param, Gumbel detach logic
├── loss.py              # MR-STFT loss — already complete, no changes
├── model_tcn.py         # Model — already outputs H_mag_hard, no changes
├── train.py             # Training loop — wire H_mag_hard, audio reconstruction, active_band_mask
├── differentiable_eq.py # DSP cascade — already has forward/forward_soft, no changes
└── conf/config.yaml     # Config — may need warmup_epochs added explicitly
```

### Pattern 1: Dual Forward Path (D-01, LOSS-04)
**What:** Model computes two frequency responses -- `H_mag_hard` (argmax types, clean param comparison) and `H_mag` (soft Gumbel, differentiable type gradients). Loss uses `H_mag_hard` for `hmag_loss` (frequency response L1) and `H_mag` for `spectral_loss` (MR-STFT).
**When to use:** Always during training. At inference, both converge to the same value.
**Current state in model_tcn.py (lines 636-643):**
```python
# Already computed in model forward():
H_mag_hard = self.dsp_cascade(gain_db, freq, q, self.n_fft, filter_type)  # argmax types
if self.training:
    H_mag = self.dsp_cascade.forward_soft(gain_db, freq, q, type_probs, self.n_fft)  # soft
else:
    H_mag = H_mag_hard
```
**Wiring gap:** `loss_multitype.py:forward()` accepts only `pred_H_mag` (single parameter). Needs a new `pred_H_mag_hard` parameter for hmag_loss, while existing `pred_H_mag` becomes the soft version for spectral_loss.

### Pattern 2: Epoch-based Warmup Gating (D-03, LOSS-02)
**What:** Loss components activate progressively based on `current_epoch`.
**When to use:** Controlled by `warmup_epochs` parameter (default 5).
**Current state in loss_multitype.py (lines 348-351):**
```python
is_warmup = self.current_epoch < self.warmup_epochs        # epochs 1-5: gain only
is_freq_q_active = self.current_epoch >= self.warmup_epochs  # epoch 6+: freq+Q
is_type_active = self.current_epoch >= self.warmup_epochs + 1  # epoch 7+: type
is_spectral_active = self.current_epoch >= self.warmup_epochs + 2  # epoch 8+: spectral
```
**Config gap:** `warmup_epochs` is not in `conf/config.yaml`. It defaults to 5 in the loss class `__init__` (line 286). Should be explicitly set in config for visibility.

### Pattern 3: Audio Reconstruction Loss (D-06, LOSS-05)
**What:** Spectral-domain MR-STFT between predicted audio (H_mag * wet_spectrum) and target audio (dry_spectrum or target_spectrum).
**When to use:** After warmup period (epoch 8+), provides additional gradient signal for spectral shape.
**Current state:** Loss function has the MR-STFT code (line 465-471) but it never fires because train.py passes `pred_audio=None, target_audio=None`.
**Implementation approach:** Compute reconstructed spectrum in train.py as `pred_audio = pred_H_mag * wet_stft_magnitude`, `target_audio = target_H_mag * wet_stft_magnitude`. The wet STFT is available from the input processing pipeline.

### Anti-Patterns to Avoid
- **Using H_mag_soft for hmag_loss:** The soft Gumbel coefficients produce blended filters that don't correspond to any real filter. Using this for hmag_loss creates a confusing gradient signal. Always use H_mag_hard for hmag_loss.
- **Detaching gain from all spectral paths:** Currently train.py detaches gain for the `pred_H_mag_for_loss` computation. With the dual path, this needs careful re-examination: gain should have gradient from the log_cosh regression term, but the spectral/hmag losses should focus on freq/Q/type.
- **Passing None for active_band_mask:** The activity loss silently zeros out when mask is None, making it dead weight. Always pass the mask from the dataset batch.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Log-cosh loss | Custom loss function | `log_cosh_loss()` at `loss_multitype.py:22-34` | Already implemented with numerically stable formulation |
| Warmup gating | Custom scheduling logic | `is_warmup`, `is_freq_q_active` etc. in `loss_multitype.py:348-351` | Already implemented with epoch-based phasing |
| MR-STFT spectral loss | Custom spectral distance | `MultiResolutionSTFTLoss` in `loss.py` | Already implemented with multi-resolution FFT sizes |
| Hungarian matching | Custom assignment solver | `HungarianBandMatcher` in `loss_multitype.py` | scipy's `linear_sum_assignment` is O(n^3) optimal |
| Per-parameter loss weights | Single combined lambda | `lambda_gain`, `lambda_freq`, `lambda_q` independently | Already in config and loss computation |

**Key insight:** This phase is primarily a wiring phase, not a building phase. Most components exist; the gaps are in connecting them correctly.

## Common Pitfalls

### Pitfall 1: NaN from H_mag_hard backward pass
**What goes wrong:** The model comment at train.py:423-427 explicitly warns that `H_mag_hard` (using `torch.where` with argmax types) produces NaN gradients: "non-selected branches still propagate NaN through the backward pass (0 * NaN = NaN in gradient accumulation)."
**Why it happens:** `torch.where` computes both branches and selects one. The non-selected branch can produce NaN in coefficients for invalid filter type/frequency combinations, and `0 * NaN = NaN` in gradient accumulation.
**How to avoid:** H_mag_hard must be used with `torch.no_grad()` or its gradient must be detached. For hmag_loss, we need `H_mag_hard.detach()` for the L1 comparison target (since we want the loss signal but not NaN gradients through the hard path). Alternatively, compute H_mag_hard only for the loss comparison and ensure no backward pass flows through it.
**Warning signs:** NaN loss at training step 1 after wiring H_mag_hard.

### Pitfall 2: Gumbel detach timing
**What goes wrong:** If type_probs are detached too aggressively or for too long, the type classification head never receives gradient signal and remains random (~20% accuracy for 5 types).
**Why it happens:** The gain regression loss doesn't need type information, but the freq/Q losses do need correct type information to match bands via Hungarian matching. Detaching type_probs during warmup means the matcher still uses type logits (not probs) for matching, so the matcher is unaffected.
**How to avoid:** Only detach during warmup (epochs 1-5). After warmup, allow joint gradients. The detach point should be in the loss function, not the model, to keep the model architecture clean.
**Warning signs:** Type accuracy stuck at ~20% after 20 epochs.

### Pitfall 3: Audio reconstruction wiring with STFT availability
**What goes wrong:** Computing `pred_audio = H_mag * wet_stft_magnitude` requires access to the wet audio STFT, but the training loop uses precomputed mel-spectrograms (`wet_mel` from batch). The raw audio may not be available if data was precomputed.
**Why it happens:** `dataset.precompute()` caches mel-spectrograms and discards raw audio. The `SyntheticEQDataset` stores `wet_mel` but may not store `wet_audio` after precomputation.
**How to avoid:** Either (a) compute the reconstruction in the spectral domain using the mel-spectrogram representation (simpler), or (b) ensure raw wet audio is available in the batch for STFT computation. Option (a) is safer and avoids STFT overhead.
**Warning signs:** AttributeError when accessing `batch["wet_audio"]` in train.py.

### Pitfall 4: Activity mask not passed through
**What goes wrong:** The `activity_loss` silently zeros out when `active_band_mask` is None, making the LOSS-06 requirement unsatisfied.
**Why it happens:** `train.py` extracts `gain`, `freq`, `q`, `filter_type` from the batch but never extracts `active_band_mask`.
**How to avoid:** Add `active_band_mask = batch.get("active_band_mask", None)` to train.py loss call. Verify the dataset actually provides this field.
**Warning signs:** `activity_loss=0.0000` in all training logs, even for epochs with inactive bands.

### Pitfall 5: warmup_epochs not in config
**What goes wrong:** The warmup period is hardcoded as default parameter `warmup_epochs=5` in `MultiTypeEQLoss.__init__`. If someone changes the config, the warmup doesn't change.
**Why it happens:** Config YAML doesn't have a `warmup_epochs` key; it relies on the Python default.
**How to avoid:** Add `warmup_epochs: 5` to `conf/config.yaml` under `loss:` section and wire it in `train.py:Trainer.__init__` at line 227.
**Warning signs:** Config changes have no effect on warmup behavior.

## Code Examples

### Dual forward path wiring in loss_multitype.py
The loss forward() needs a new parameter for H_mag_hard:
```python
# Current signature (line 310):
def forward(self, pred_gain, pred_freq, pred_q, pred_type_logits,
            pred_H_mag, target_gain, ...):

# Needed signature:
def forward(self, pred_gain, pred_freq, pred_q, pred_type_logits,
            pred_H_mag_soft, pred_H_mag_hard, target_gain, ...):
```

Then in the loss body, use `pred_H_mag_hard` for hmag_loss and `pred_H_mag_soft` for spectral:
```python
# hmag_loss uses hard types (clean comparison, no Gumbel noise):
pred_H_mag_hard_safe = torch.clamp(pred_H_mag_hard.float().detach(), min=1e-6, max=1e6)
target_H_mag_safe = torch.clamp(target_H_mag.float(), min=1e-6, max=1e6)
loss_hmag = F.l1_loss(torch.log(pred_H_mag_hard_safe), torch.log(target_H_mag_safe))
# Note: .detach() prevents NaN backward through torch.where argmax path
```

### Dual forward path wiring in train.py
```python
# In train_one_epoch (around line 430):
total_loss, components = self.criterion(
    pred_gain,
    pred_freq,
    pred_q,
    output["type_logits"],
    output["H_mag"],           # soft for spectral loss
    output["H_mag_hard"],      # hard for hmag_loss
    target_gain,
    target_freq,
    target_q,
    target_ft,
    target_H_mag,
    embedding=output["embedding"],
)
```

### Gumbel detach during warmup in loss function
```python
# In MultiTypeEQLoss.forward(), after warmup check:
# D-05: Detach type_probs from gain gradient path during warmup
if is_warmup:
    # During warmup, type classification head gets no gradient from param regression.
    # This prevents noisy type gradients from interfering with gain learning.
    pred_type_logits_for_match = pred_type_logits.detach()
else:
    pred_type_logits_for_match = pred_type_logits

# Use pred_type_logits_for_match in Hungarian matching (line 400-410)
```

### Audio reconstruction wiring (spectral domain)
```python
# In train.py train_one_epoch, after computing target_H_mag:
# D-06: Spectral-domain reconstruction for MR-STFT loss
# Compare predicted spectrum (H_mag * wet) vs target spectrum (H_target * wet)
if is_spectral_active:
    # Use soft H_mag for differentiable gradients
    pred_audio = output["H_mag"]  # Already soft during training
    target_audio = target_H_mag   # Ground truth EQ response
    # Both are (B, n_fft//2+1) frequency responses
else:
    pred_audio = None
    target_audio = None
```
Note: The actual MR-STFT loss in `loss.py` expects time-domain audio (Batch, Time). For spectral-domain, we may need to adapt the loss or use a simpler spectral L1/L2 instead. This needs investigation during planning.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Combined lambda_param weight | Independent lambda_gain/lambda_freq/lambda_q | Pre-this-phase | Already in codebase |
| Huber loss for gain | Log-cosh loss for gain | Pre-this-phase | Already in codebase |
| Single H_mag path | Dual H_mag_hard + H_mag_soft | This phase | Need to wire |
| No audio reconstruction | Spectral MR-STFT reconstruction | This phase | Need to wire |
| No Gumbel detach | Detach during warmup | This phase | Need to implement |

**Deprecated/outdated:**
- `lambda_param` combined weight: Set to 0.0 in config, disabled path in loss code (line 432-441). The independent weights path is active.
- `CombinedIDSPLoss` in `loss.py`: Legacy class, not used by current training. `MultiTypeEQLoss` in `loss_multitype.py` is the active loss.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | H_mag_hard backward produces NaN through torch.where | Pitfall 1 | If wrong, .detach() on hmag_loss input is unnecessary (but harmless) |
| A2 | Dataset batch provides `active_band_mask` field | LOSS-06 | If wrong, need to verify/generate the mask in dataset |
| A3 | Spectral MR-STFT loss expects time-domain audio, not frequency response | Pattern 3 | If wrong, wiring is simpler than expected |
| A4 | warmup_epochs=5 default is correct for the current config.yaml curriculum stages (10 epoch warmup stage) | LOSS-02 | If wrong, warmup period may conflict with curriculum stages |

## Open Questions (RESOLVED)

1. **MR-STFT input format**
   - What we know: `MultiResolutionSTFTLoss.forward()` expects `(x, y)` time-domain audio (Batch, Time) based on `loss.py:19-20` which calls `torch.stft()` on the inputs.
   - What's unclear: CONTEXT.md says "spectral-domain MR-STFT: compare H_mag * wet_spectrum vs target_spectrum" -- but MR-STFT applies its own STFT to time-domain signals. Feeding it frequency responses directly would be incorrect.
   - Recommendation: Either (a) adapt MR-STFT to accept magnitude spectra directly, or (b) use a simpler spectral L1 loss for the reconstruction term, or (c) ensure wet audio is available for time-domain reconstruction. Claude's discretion to determine the best approach.
   - **RESOLVED:** Using spectral L1 on log H_mag (no time-domain audio needed). MR-STFT call in forward() is replaced with `F.l1_loss(torch.log(pred_H_mag_soft), torch.log(target_H_mag))`. The `self.mr_stft` module stays in `__init__` but is no longer called in `forward()`.

2. **Dataset active_band_mask availability**
   - What we know: The loss function accepts `active_band_mask` but train.py never extracts it from the batch.
   - What's unclear: Whether `SyntheticEQDataset` or `MUSDB18EQDataset` actually include `active_band_mask` in the batch dict.
   - Recommendation: Verify during implementation. If not present, add it to the dataset's `__getitem__`.
   - **RESOLVED:** `active_band_mask` is not present in current `dataset.py`. Plan 02 Task 1 adds it to `__getitem__`, `precompute()`, `_generate_sample()`, and `collate_fn`. All synthetic samples use all-True mask (all 5 bands active).

3. **Interaction between loss warmup and curriculum stages**
   - What we know: Config has curriculum stages starting at epoch 1 (10-epoch "warmup" stage). Loss warmup_epochs=5 is a separate mechanism.
   - What's unclear: Whether these two systems conflict (curriculum "warmup" stage lasts 10 epochs, loss warmup ends at epoch 5).
   - Recommendation: These are orthogonal -- curriculum controls Gumbel temperature and LR, loss warmup controls which loss components are active. No conflict, but planner should document the interaction clearly.
   - **RESOLVED:** These are orthogonal systems. Curriculum stage config (Gumbel temperature, LR scale) is unchanged. The loss hybrid gate (epoch + gain_mae_ema) is a separate mechanism in `MultiTypeEQLoss`. The curriculum "warmup" stage name is coincidental -- it controls Gumbel temperature only, not loss component activation.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.x | All scripts | Yes | 3.12.11 | -- |
| PyTorch | Loss computation, autograd | Yes | 2.8.0+cu128 | -- |
| CUDA GPU | Training | Yes | Available | CPU fallback (slow) |
| scipy | Hungarian matching | Yes | 1.11.4 | -- |
| pyyaml | Config loading | Yes | (present) | -- |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Standalone Python scripts (no pytest) |
| Config file | None -- tests are self-contained |
| Quick run command | `cd insight && python test_eq.py` |
| Full suite command | `cd insight && python test_eq.py && python test_model.py && python test_streaming.py && python test_multitype_eq.py` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LOSS-01 | Independent gain/freq/Q weights in loss computation | unit | `python -c "from loss_multitype import MultiTypeEQLoss; ..."` | Needs creation |
| LOSS-02 | Warmup gating zeros out non-gain losses during warmup | unit | `python test_loss_warmup.py` | Needs creation (Wave 0) |
| LOSS-03 | Log-cosh loss produces correct gradients | unit | `python -c "from loss_multitype import log_cosh_loss; ..."` | Needs creation |
| LOSS-04 | Dual H_mag_hard/H_mag_soft used correctly in loss | unit | `python test_dual_hmag.py` | Needs creation (Wave 0) |
| LOSS-05 | Audio reconstruction loss fires when wired | unit | `python test_audio_recon.py` | Needs creation (Wave 0) |
| LOSS-06 | Activity mask weighting works | unit | `python test_activity_weight.py` | Needs creation (Wave 0) |
| DATA-02 | Gumbel detach prevents type gradients during warmup | unit | `python test_gumbel_detach.py` | Needs creation (Wave 0) |

### Sampling Rate
- **Per task commit:** `cd insight && python test_loss_warmup.py && python test_dual_hmag.py`
- **Per wave merge:** Full test suite: `test_eq.py && test_model.py && test_streaming.py && test_multitype_eq.py`
- **Phase gate:** All existing + new tests green before phase completion

### Wave 0 Gaps
- [ ] `insight/test_loss_architecture.py` -- covers LOSS-01 through LOSS-06, DATA-02 (combined test file following project pattern)
- [ ] No framework install needed -- standalone scripts are the project convention

## Security Domain

Not applicable for this phase. The changes are internal loss function restructuring with no external interfaces, no user input, no authentication, and no data persistence changes.

## Sources

### Primary (HIGH confidence)
- Direct code reading of `insight/loss_multitype.py` (all 514 lines) -- verified warmup gating, log-cosh, independent weights, dual H_mag handling, activity loss
- Direct code reading of `insight/model_tcn.py` lines 606-655 -- verified dual H_mag_hard/H_mag output in forward()
- Direct code reading of `insight/train.py` (all 1073 lines) -- verified loss wiring, epoch gating, audio reconstruction gap
- Direct code reading of `insight/loss.py` -- verified MR-STFT expects time-domain input
- Direct code reading of `insight/differentiable_eq.py` -- verified forward/forward_soft implementations
- Direct code reading of `insight/conf/config.yaml` -- verified loss weights, curriculum stages, no warmup_epochs key
- Phase 2 execution summary (`02-01-SUMMARY.md`) -- verified completed changes, streaming compatibility

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions D-01 through D-11 -- user decisions from discuss phase, cross-referenced against code

### Tertiary (LOW confidence)
- None -- all findings verified against source code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified present and version-checked
- Architecture: HIGH -- all wiring gaps identified by reading source code directly
- Pitfalls: HIGH -- NaN from torch.where documented in existing code comments (train.py:423-427)

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable domain, no external API dependencies)
