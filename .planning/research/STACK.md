# Technology Stack

**Project:** IDSP EQ Estimator -- Accuracy Improvement
**Researched:** 2026-04-05
**Overall Confidence:** HIGH

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch | 2.8.0+cu128 (current) | Differentiable training pipeline | Already in use. Provides autograd through custom biquad coefficient computation, bf16-mixed precision via AMP, `torch.compile` for kernel fusion. v2.8 supports `torch.compile` with custom autograd functions (required for STE clamp). No reason to change. | HIGH |
| Python | 3.10+ | Runtime | Already in use. PyTorch 2.8 requires 3.9+. | HIGH |

### DSP and Audio Processing

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Custom DifferentiableBiquadCascade | Existing (insight/differentiable_eq.py) | Differentiable biquad filter computation | Already implements the Robert Bristow-Johnson Audio EQ Cookbook formulas entirely in PyTorch. Supports 5 filter types with soft (Gumbel-Softmax) and hard (torch.where) type selection. Verified gradient flow through `test_eq.py`. Do NOT replace with external library -- this is already correct and well-tested. | HIGH |
| STFTFrontend (dsp_frontend.py) | Existing | Differentiable STFT/mel spectrogram | Custom implementation avoids torchaudio dependency in training loop (project constraint). Supports causal mode for streaming. Keep. | HIGH |
| scipy.optimize.linear_sum_assignment | 1.11.4 (current) | Hungarian matching for band permutation | Already in use. This is the standard algorithm for bipartite matching -- identical to DETR's approach. Runs on CPU per batch element, which is fine for batch_size=1024 (negligible overhead). | HIGH |

### Loss Functions

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| auraloss | 0.3.1+ | Multi-resolution STFT loss, mel-band STFT loss | Steinmetz & Reiss (DMRN+15, 2020). Provides `MultiResolutionSTFTLoss`, `MelSTFTLoss`, `RandomResolutionSTFTLoss` in pure PyTorch. The project already implements its own MR-STFT loss (in `loss.py`) but auraloss provides additional perceptually-motivated losses that could improve spectral matching. Specifically: `MelSTFTLoss` applies mel-frequency weighting that emphasizes perceptually important bands, which could help the hmag_loss focus on audible spectral differences rather than treating all frequency bins equally. | MEDIUM |
| Log-cosh loss (custom) | N/A -- implement from `fixes/modified_loss.py` | Gain parameter regression loss | Already prototyped in `insight/fixes/modified_loss.py`. Properties: ~x^2/2 near zero (MSE-like strong gradients), ~|x| for large errors (L1-like robustness), smooth everywhere (no Huber kink at delta). Superior to Huber(delta=5.0) for gain regression because small errors (1-5 dB) get MSE-like gradients rather than the flat L1 regime of Huber. | HIGH |
| Huber loss (torch.nn.HuberLoss) | Built-in | Frequency and Q regression loss | Already in use. Appropriate for these parameters because log-frequency errors span a wider range (~6.9 octaves) where the L1 regime of Huber prevents outlier domination. Keep for freq/Q; replace only for gain. | HIGH |

### Training Infrastructure

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| AdamW + CosineAnnealingLR | Built-in (torch.optim) | Optimizer and scheduler | Already in use. AdamW with weight_decay=0.02 provides decoupled weight decay. CosineAnnealing provides smooth LR decay. These are standard choices for audio ML. Keep. | HIGH |
| bf16-mixed precision (AMP) | Built-in (torch.amp) | Training speed and memory | Already in use via `precision: "bf16-mixed"`. Works correctly with the custom STE clamp autograd function (which internally casts to fp32). Keep. | HIGH |
| torch.compile | Built-in (PyTorch 2.x) | Kernel fusion for forward+backward | Already enabled in config (`use_torch_compile: true`). Provides 10-30% speedup on the biquad coefficient computation and frequency response evaluation. Keep. | HIGH |
| Gradient checkpointing | Built-in (torch.utils.checkpoint) | Memory reduction for longer sequences | Available but currently disabled. Enable if extending to longer audio or larger models. The FrequencyPreservingTCN already supports it via `use_gradient_checkpointing` flag. | HIGH |

### Supporting Libraries

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| nnAudio | 0.3.1 | Differentiable spectrogram alternatives (CQT, VQT) | If mel-spectrogram resolution proves insufficient for detecting narrow EQ bands. CQT provides constant-Q frequency resolution that better matches the logarithmic spacing of EQ parameters. Cheuk & Ma (IEEE Access, 2020). Pure PyTorch via 1D convolutions. | LOW -- only if mel resolution is a bottleneck |
| numpy | 1.26.4 (current) | Data generation, array manipulation | Already in use for dataset generation and augmentation. Keep. | HIGH |
| pyyaml | Current | Config parsing | Already in use for `conf/config.yaml`. Keep. | HIGH |
| onnxruntime | Optional | ONNX export and inference validation | Already used for export (`export.py`). Keep for deployment testing. | HIGH |

### Development and Testing

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| wandb | Optional | Training experiment tracking | Useful for comparing training runs across the 14+ iterations. Recommended for the accuracy improvement project to track per-parameter MAE curves. | HIGH |
| pytorch-lightning | Optional | Alternative training loop | Exists in `training/` directory but the primary trainer (`train.py`) is custom. Keep as backup but do not switch -- the custom trainer gives more control over curriculum stages and loss phasing. | MEDIUM |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not | Confidence |
|----------|-------------|-------------|---------|------------|
| Core framework | PyTorch 2.8 (keep) | JAX with jax.numpy | Complete rewrite required. PyTorch autograd through custom biquad functions is well-tested. JAX offers no advantage for this single-GPU, forward-mode-differentiation task. | HIGH |
| DSP layer | Custom DifferentiableBiquadCascade (keep) | DDSP library (TensorFlow) | DDSP is TensorFlow-based, not PyTorch. Provides differentiable filters but would require rewriting the entire pipeline. Also, DDSP's filter implementations are designed for synthesis (oscillator + filter), not blind parameter estimation. Our custom implementation is purpose-built for biquad EQ coefficient computation with gradient flow. | HIGH |
| DSP layer | Custom DifferentiableBiquadCascade (keep) | torchsignal / torchfilter | These repos (csteinmetz1) no longer exist or are private (404 errors). Cannot use. | HIGH |
| Audio features | Custom STFTFrontend (keep) | torchaudio transforms | Project constraint: "no torchaudio in training loop -- must stay differentiable." The custom frontend avoids the torchaudio dependency and supports causal mode for streaming inference. Keep. | HIGH |
| Loss functions | Log-cosh for gain, Huber for freq/Q | Focal loss for all parameters | Focal loss upweights hard examples but can be unstable early in training. Log-cosh provides a similar adaptive weighting (MSE-like for small errors, L1-like for large) without the instability. The focal weighting variant in `fixes/modified_loss.py` is available as an optional enhancement. | MEDIUM |
| Hungarian matching | scipy linear_sum_assignment (keep) | PyTorch differentiable Sinkhorn | Sinkhorn provides GPU-native differentiable matching but adds complexity. scipy's Hungarian is exact (not approximate), runs on CPU with negligible overhead (batch_size=1024, num_bands=5), and is battle-tested. Not worth changing. | HIGH |
| Encoder architecture | FrequencyAwareEncoder (keep) | Wave-U-Net, Demucs-style U-Net | The current 2D spectral front-end + grouped TCN + attention pooling was specifically designed to fix encoder collapse (cosine distance 0.006). It works. Switching to U-Net would break streaming capability (requires bidirectional processing). | HIGH |
| Encoder architecture | FrequencyAwareEncoder (keep) | Transformer encoder | Out of scope per PROJECT.md ("fix what's there first"). Transformers require O(n^2) attention, incompatible with streaming. The current TCN's causal convolutions are the right choice for real-time inference. | HIGH |
| Gumbel-Softmax | Keep with adjusted schedule | Straight-through estimator (STE) for type | STE would give hard type selection during training but provides no gradient through the selection, requiring REINFORCE or similar policy gradient methods. Gumbel-Softmax provides continuous relaxation with gradient flow. The issue is not the mechanism but the gradient interaction with gain -- fix via hard-type dual path (see ARCHITECTURE.md). | HIGH |
| Gain activation | STE clamp (recommended) | tanh soft clamp (current) | tanh attenuates gradients by 20-40% at moderate gains (12-18 dB). STE clamp provides identity gradient within bounds. Already implemented in codebase. | HIGH |
| Gain activation | STE clamp (recommended) | Asymmetric softplus clamp | Proposed in `fixes/gain_fixes.py` AlternativeGainHead. Still uses tanh at the final stage (`max_gain * tanh(raw / (max_gain * 1.5))`). Better than pure tanh but still attenuates. STE clamp is strictly superior. | HIGH |

## Key Research Papers Informing Stack Decisions

| Paper | Year | Relevance | Confidence |
|-------|------|-----------|------------|
| Engel et al., "DDSP: Differentiable Digital Signal Processing" | ICLR 2020 | Foundational paper establishing that traditional DSP operations (oscillators, filters, reverbs) can be embedded as layers in neural networks with exact gradient flow. Validates the project's approach of computing biquad coefficients in PyTorch. Note: the DDSP library itself is TensorFlow and should NOT be used -- take the concept, not the code. | HIGH |
| Carion et al., "End-to-End Object Detection with Transformers" (DETR) | ECCV 2020 | Established Hungarian matching for set prediction problems. The project's use of Hungarian matching for permutation-invariant band assignment directly follows this pattern. The matching cost weights (lambda_gain, lambda_freq, lambda_q, lambda_type_match) parallel DETR's classification + regression cost weighting. | HIGH |
| Steinmetz et al., "micro-tcn: Efficient Temporal Convolutional Networks for Analog Audio Effect Modeling" | AES 2022 | Validates TCN architecture with auraloss for audio effect modeling. Shows TCN with gated activations and dilated convolutions is effective for audio effects. The project's FrequencyAwareEncoder uses the same gated activation pattern (tanh * sigmoid) with grouped convolutions for frequency preservation. | MEDIUM |
| Steinmetz & Reiss, "auraloss: Audio-focused loss functions in PyTorch" | DMRN+15, 2020 | Provides perceptually-motivated audio losses. The MelSTFTLoss applies mel-frequency weighting that could improve hmag_loss. The project's custom MR-STFT loss follows the same multi-resolution pattern but lacks the mel weighting. | MEDIUM |
| Cheuk et al., "nnAudio: An on-the-fly GPU Audio to Spectrogram Conversion Toolbox" | IEEE Access, 2020 | Provides differentiable CQT, VQT, and other spectrogram transforms via PyTorch 1D convolutions. Useful if mel-spectrogram resolution proves insufficient for narrow EQ bands. Not needed now. | LOW |

## Installation

### Current Stack (already installed)

The project runs on Lightning AI with the following pre-installed:

```bash
# Core (already available)
torch==2.8.0+cu128
scipy==1.11.4
numpy==1.26.4
pyyaml

# Optional (already available)
onnxruntime  # for export
pytorch-lightning  # alternative trainer
```

### Recommended Additions

```bash
# Audio loss functions -- provides perceptually-motivated spectral losses
pip install auraloss

# Experiment tracking -- for comparing training runs
pip install wandb

# Differentiable spectrogram alternatives -- ONLY IF mel resolution proves insufficient
pip install nnAudio==0.3.1
```

### What NOT to Install

```bash
# Do NOT install -- TensorFlow-based, incompatible with PyTorch pipeline
pip install ddsp  # AVOID

# Do NOT use in training loop -- project constraint
# torchaudio is available for evaluation/export but NOT for training
```

## Version Pinning Rationale

| Package | Pin Policy | Reason |
|---------|-----------|--------|
| torch | >=2.2 | Required for `torch.compile` and bf16-mixed AMP. v2.8 is current. |
| scipy | >=1.6 | Required for `linear_sum_assignment`. API stable since 1.6. |
| numpy | >=1.21 | Required for array operations in dataset generation. Compatible with PyTorch 2.8. |
| auraloss | >=0.3 | Latest stable. API unlikely to change. |
| nnAudio | ==0.3.1 | Specific version tested with PyTorch 2.x. |

## Sources

- PyTorch 2.8: Verified from runtime (`python3 -c "import torch; print(torch.__version__)"`)
- scipy 1.11.4: Verified from runtime
- auraloss: github.com/csteinmetz1/auraloss -- verified README, PyPI listing, citation info
- nnAudio: github.com/KinWaiCheuk/nnAudio -- verified README, v0.3.1 tag
- DDSP (concept): github.com/magenta/ddsp -- verified as TensorFlow library, ICLR 2020 paper
- micro-TCN: github.com/csteinmetz1/micro-tcn -- verified README, AES 2022 citation
- DETR (concept): Carion et al., ECCV 2020 -- training data, widely cited
- Project config: insight/conf/config.yaml -- verified current loss weights, curriculum stages
- Root cause analysis: insight/diagnostics/root_cause_analysis.md -- verified 8 issues
- Existing fix attempts: insight/fixes/ -- verified gain_fixes.py, modified_head.py, modified_loss.py

---

*Stack research for: IDSP EQ Estimator accuracy improvement*
*Researched: 2026-04-05*
