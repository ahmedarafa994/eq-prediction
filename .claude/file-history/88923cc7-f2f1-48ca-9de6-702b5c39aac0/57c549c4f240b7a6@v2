# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **differentiable DSP (IDSP) system for blind parametric EQ parameter estimation**. Given an audio signal that has been processed through a multi-band parametric EQ, the model estimates the EQ parameters (gain, frequency, Q, filter type) without access to the original dry signal.

The main project lives in `insight/`. The rest of this studio directory contains Lightning AI environment files, plugin caches, and template apps unrelated to the core work.

## Commands

### Running Tests

Tests are standalone Python scripts (no pytest). Run from `insight/`:

```bash
cd insight
python test_eq.py              # Differentiable biquad gradient flow
python test_model.py           # CNN model forward/inverse/cycle/gradient
python test_streaming.py       # TCN streaming mode, batch-vs-streaming consistency
python test_multitype_eq.py    # Multi-type filters (HP/LP/shelf) + parameter head
python test_lightning_dummy.py # PyTorch Lightning fast_dev_run smoke test
python test_real_audio.py --audio path/to/audio.wav  # Against real audio with known EQ
```

### Training

```bash
cd insight
python train.py                                   # Primary: TCN model, config from conf/config.yaml
python train.py --config conf/config_simple.yaml   # Simplified peaking-only for verification
python train.py --resume checkpoints/best.pt       # Resume from checkpoint
python train_spectral.py                           # Spectral prediction model (H_db → params)
```

### Evaluation & Export

```bash
cd insight
python evaluate_model.py                           # Full evaluation with Hungarian-matched metrics
python export.py --checkpoint checkpoints/best.pt --output eq_estimator.onnx
```

### Data Generation

```bash
cd insight
python generate_dataset.py                         # Precompute 10k synthetic dataset to data/
python dataset_pipeline/generate_data.py            # Multi-type offline data generation with MUSDB18
```

## Architecture

### Core DSP Layer

- **`differentiable_eq.py`** — The foundation. `DifferentiableBiquadCascade` computes biquad filter coefficients from (gain, freq, Q, filter_type) tensors and evaluates frequency responses entirely in PyTorch for gradient flow. Coefficient formulas from the Robert Bristow-Johnson Audio EQ Cookbook. Supports 5 filter types: peaking, lowshelf, highshelf, highpass, lowpass. Uses `torch.where` for differentiable type selection and Gumbel-Softmax for soft type probabilities during training.

- **`dsp_frontend.py`** — `STFTFrontend`: differentiable STFT/iSTFT with a manually-built mel filterbank (no torchaudio dependency in training). Supports causal mode for streaming.

### Models (two approaches)

**Parametric approach (primary):**
- **`model_tcn.py`** — `StreamingTCNModel`: causal 1D TCN encoder (WaveNet-style gated activations with dilated convolutions) → `MultiTypeEQParameterHead` → `DifferentiableBiquadCascade`. Supports both batch training and frame-by-frame streaming inference via `init_streaming()` / `process_frame()`. Cumulative mean of skip connections produces a stable embedding.

**Spectral approach (alternative):**
- **`model_spectral.py`** — `SpectralEQModel`: bypasses encoder collapse by predicting H_db directly from spectral profile via MLP. Parameter extraction is then a post-hoc optimization (not a learning problem). Diagnostics showed 0.20 dB MAE for spectral prediction vs 5.77 dB for the TCN parametric approach. Uses `param_extractor.py` at inference.

**Earlier variant (kept for backward compatibility):**
- **`model_cnn.py`** — `EQEstimatorCNN`: 2D CNN on mel-spectrograms with `EQParameterHead` (peaking-only).

### Parameter Heads

- `EQParameterHead` — peaking-only: tanh/sigmoid activation for constrained gain/freq/Q output.
- `MultiTypeEQParameterHead` — adds filter type classification via Gumbel-Softmax, log-frequency parameterization for perceptually uniform resolution.

### Loss Functions

- **`loss.py`** — `MultiResolutionSTFTLoss`, `FreqResponseLoss`, `CycleConsistencyLoss`, `EQParameterPriorLoss`, `CombinedIDSPLoss` (for the CNN variant).
- **`loss_multitype.py`** (primary) — `HungarianBandMatcher` solves the band permutation problem (DETR-style). `MultiTypeEQLoss` combines: parameter regression (Huber), type classification (cross-entropy), frequency response L1, band activity regularization, and frequency spread regularization.

### Data

- **`dataset.py`** — `SyntheticEQDataset`: generates (dry, wet, params) tuples on-the-fly using synthetic signals (noise, sweep, harmonic, speech_like) with random multi-type EQ applied in frequency domain. Supports `precompute()` mode that caches mel-spectrograms in memory.
- **`dataset_pipeline/`** — Offline data generation with MUSDB18 real audio support.
- **`augmentation.py`** — SpecAugment (frequency + time masking) for training.
- **`data_validation.py`** — `DataQualityValidator` for detecting NaN/inf and distribution anomalies in training data.

### Training

- **`train.py`** — Custom `Trainer` class (no Lightning dependency). Uses `StreamingTCNModel` with `MultiTypeEQLoss`. AdamW + CosineAnnealing. Precomputes mel-spectrograms for speed. Implements curriculum learning from config.
- **`train_spectral.py`** — `SpectralTrainer` for the spectral prediction model.
- **`training/`** — PyTorch Lightning alternative (`lightning_module.py`, `train.py`) with curriculum learning (`curriculum.py`) and evaluation (`evaluate.py`).
- **`conf/config.yaml`** — All hyperparameters, model architecture, loss weights, and 5-stage curriculum (peaking foundation → shelf introduction → HP/LP addition → full finetuning → validation calibration).
- **`conf/config_simple.yaml`** — Simplified peaking-only config for initial verification.
- **`conf/config_spectral.yaml`** — Config for the spectral model.
- **`conf/config_full_spectrum.yaml`** — Full-band magnitude spectrum variant (no mel compression).

### Supporting Modules

- **`calibration.py`** — `TemperatureScaling` for calibrating Gumbel-Softmax confidence estimates post-training.
- **`gumbel_annealing.py`** — Intra-stage smooth Gumbel temperature annealing (addresses hardcoded min_tau floor in parameter head).
- **`optimizer_setup.py`** — Optimizer and LR scheduler configuration.
- **`param_extractor.py`** — Post-hoc parameter extraction from predicted H_db via scipy optimization (for spectral model).
- **`analyze_learning_curves.py`** — Training log analysis and visualization.
- **`profile_gpu.py`** — GPU profiling utilities.

### Data Directory

Precomputed datasets and MUSDB18 audio live under `insight/data/`. MUSDB18 path is configured via `data.musdb_root` in config.

## Key Design Decisions

- All DSP is differentiable in PyTorch (no external audio processing in the training loop).
- Hungarian matching makes the loss permutation-invariant to band ordering.
- Log-frequency parameterization gives equal resolution per octave.
- Gumbel-Softmax enables differentiable filter type selection; hard argmax at inference. Temperature is annealed across curriculum stages (1.0 → 0.05).
- The streaming mode uses a rolling buffer and cumulative skip-connection mean for frame-by-frame inference.
- No external audio files needed for synthetic training — all training data is generated on-the-fly. MUSDB18 is optional for real-audio training.
- The spectral model was developed after diagnosing TCN encoder collapse (cosine distance 0.006 between embeddings) — it avoids the multi-band decomposition problem by predicting H_db directly and extracting params via optimization at inference.
- Training uses bf16-mixed precision with gradient accumulation (4 batches) for effective batch size of 1024.

## Dependencies

PyTorch, torchaudio, scipy (for Hungarian matching and param extraction), pyyaml. Optional: pytorch-lightning, onnxruntime, wandb.
