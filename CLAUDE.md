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

<!-- GSD:project-start source:PROJECT.md -->
## Project

**IDSP EQ Estimator — Accuracy Improvement**

A differentiable DSP system for blind parametric EQ parameter estimation. Given an audio signal processed through a multi-band parametric EQ (up to 5 bands, 5 filter types), the model estimates the EQ parameters (gain, frequency, Q, filter type) without access to the original dry signal. The system exists and trains but plateaus at poor accuracy — this project fixes it to professional quality for commercial deployment.

**Core Value:** The model must accurately estimate EQ parameters from wet audio alone. If gain MAE stays above 1 dB, the product has no value for audio professionals.

### Constraints

- **Tech stack:** PyTorch, scipy (Hungarian matching), no torchaudio in training loop — must stay differentiable
- **Compute:** Single GPU on Lightning AI — model must fit in available VRAM
- **Data:** 200k precomputed synthetic + MUSDB18 dataset — can regenerate if needed
- **Architecture:** Must preserve streaming inference capability (causal convolutions)
- **Evaluation:** Must have proper Hungarian-matched validation metrics to track real progress
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.x - Core language (indicated by pyproject.toml, pip-style requirements)
- TypeScript/JavaScript - Node.js development environment (node-v25.9.0)
- YAML - Configuration files
- Bash - Shell scripting and development automation
## Runtime
- Python 3.x - Primary runtime for all training, evaluation, and data processing scripts
- Node.js v25.9.0 - Development environment (dev-only, not used in core project)
- pip - Python package management (requirements_optimized.txt, ml_framework/requirements.txt)
- npm - Node.js package management (dev-only, node_modules/)
- Lockfile: No lockfile detected (pip requirements files present, but no pipfile.lock or poetry.lock)
## Frameworks
- PyTorch >=2.0.0 - Deep learning framework
- PyTorch Lightning >=1.6.0 (optional) - Alternative training framework (training/ directory)
- numpy >=1.24.0 - Numerical computing
- scipy >=1.10.0 - Signal processing (biquad filter coefficients, Hungarian matching)
- torchaudio >=2.0.0 - Audio I/O and transformations
- Custom STFT implementation (dsp_frontend.py) - No external audio framework dependency for training
- matplotlib - Plotting and visualization (test_checkpoint.py, plot_musdb_training.py, diagnose_gain.py, analyze_learning_curves.py)
- Standalone pytest-style scripts (no framework detected)
- pyyaml >=6.0 - Configuration file parsing (config.yaml, conf/*.yaml)
- tqdm - Progress bars (train.py, training/evaluate.py)
- argparse - Command-line argument parsing (test scripts, utilities)
- PyTorch Lightning - Used in training/ directory (lightning_module.py, curriculum.py)
## Key Dependencies
- torch >=2.0.0 - Core deep learning framework
- torchaudio >=2.0.0 - Audio signal processing
- numpy >=1.24.0 - Numerical arrays and DSP operations
- scipy >=1.10.0 - Signal processing (biquad coefficient computation, Hungarian matching)
- bitsandbytes >=0.41.0 - 8-bit optimizer for memory efficiency (optional)
- triton >=2.1.0 - GPU kernel optimization (Linux only)
- deepspeed >=0.12.0 - Distributed training (optional)
- onnxruntime >=1.16.0 - ONNX model inference (optional)
- pyyaml >=6.0 - Configuration management
- argparse - CLI argument handling
## Configuration
- Config files: YAML format in `insight/conf/` directory
- No build process detected (Python scripts are directly runnable)
- Optional torch.compile for graph-level optimization (train.py)
- No .env files detected in repository (checked .gitignore patterns)
## Platform Requirements
- Python 3.x (version not specified in requirements)
- Linux (required for triton kernel optimization)
- GPU with CUDA support (for PyTorch operations)
- ONNX Runtime for deployment (optional)
- DAW plugin host for real-time inference (post-export)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Snake_case for Python files: `differentiable_eq.py`, `train.py`, `test_eq.py`
- Test files: `test_<module>.py` pattern, e.g., `test_model.py`, `test_streaming.py`
- Fix directory: `fixes/` contains experimental implementations and their tests
- Config files: `conf/*.yaml` for YAML-based configuration
- Snake_case for functions: `compute_biquad_coeffs`, `forward_pass`, `apply_eq_cascade`
- Test functions: `test_<function_name>()` pattern for unit tests
- Diagnostic functions: `diagnose_<name>()`, `debug_<name>()` pattern
- Snake_case for variables: `n_fft`, `batch_size`, `gain_db`, `mel_frames`
- Short descriptive names for loop variables: `i`, `t`, `b`, `n`
- PascalCase for classes: `StreamingTCNModel`, `DifferentiableBiquadCascade`, `HungarianBandMatcher`
- UPPER_SNAKE_CASE for constants: `FILTER_PEAKING`, `FILTER_LOWSHELF`, `NUM_FILTER_TYPES`
## Code Style
- 2-space indentation
- Blank lines between logical sections (docstrings, class definitions, function definitions)
- Maximum line length: ~100-120 characters (examples wrap for readability)
## Error Handling
- Assert for debug checks with descriptive messages
- Type hints for function signatures
- Try-except for optional dependencies (e.g., bitsandbytes, deepspeed)
## Logging
- Test output includes success/failure messages with details
- Diagnostic tools print summary statistics and visualizations
- Verbose output includes intermediate values for debugging
## Comments
- Module-level docstrings describing purpose and usage
- Complex algorithm explanations (DSP coefficient formulas)
- Parameter ranges and constraints
- Workarounds for specific issues
- Multi-line docstrings use `"""` triple quotes
- Short comments use `#` single quotes
- Algorithm rationale documented in module-level comments
## Function Design
- Functions are generally < 50 lines
- Complex logic extracted to helper functions
- Test functions each cover one specific aspect
- Type hints for all parameters and return values
- Descriptive parameter names
- Optional parameters with default values
- Functions returning multiple values use tuple: `return (gain_db, freq, q)`
- Dict returns for complex output: `return {"params": (..., ...), "H_mag": ...}`
- Test functions typically return nothing (print success/failure)
## Module Design
- Classes exported from modules: `DifferentiableBiquadCascade`, `StreamingTCNModel`
- Functions exported when useful: `ste_clamp`, `load_config`
- Constants at module level: `FILTER_NAMES`, `FILTER_PEAKING`
## Configuration
- All configs in `conf/` directory
- Top-level sections: `data`, `model`, `loss`, `trainer`, `curriculum`
- Hierarchical structure with nested dictionaries
- Comments explain each parameter's purpose
- `conf/config.yaml` — Main training configuration
- `conf/config_simple.yaml` — Simplified peaking-only config
- `conf/config_spectral.yaml` — Spectral model config
- `conf/config_musdb_200k.yaml` — MUSDB18 200k dataset config
- `conf/deepspeed_config.json` — DeepSpeed optimization settings
- `fixes/recommended_config.yaml` — Recommended config for gain fixes
## Type Hints
- Type hints for all function signatures
- Optional types for optional parameters
- Tuple unpacking for multiple returns
## Documentation
- Module docstrings describe purpose, key components, usage
- Class docstrings describe purpose, main methods, parameters
- Function docstrings describe purpose, parameters, returns, side effects
- Use reST-style format with Args:, Returns:, Raises: sections
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Blind estimation: Model receives only EQ'd audio (wet signal), must predict EQ parameters (gain, frequency, Q, filter type)
- Two parallel estimation approaches: parametric (TCN-based) and spectral (direct H_db prediction)
- Permutation-invariant training via Hungarian band matching (DETR-style)
- Streaming inference support for real-time DAW plugin deployment
- Multi-stage curriculum learning for stable convergence
- Anti-collapse mechanisms to prevent encoder collapse
## Layers
### DSP Layer
- `StraightThroughClamp`: Custom autograd function for gradient flow
- `DifferentiableBiquadCascade`: Multi-type biquad filter support (5 filter types)
### Frontend Layer
- `STFTFrontend`: Manually-built mel filterbank (no torchaudio dependency in training)
- `apply_eq_to_complex_stft`: Helper for frequency-domain EQ application
### Model Layer (Parametric Approach)
- `FrequencyAwareEncoder`: Hybrid 2D + 1D TCN architecture
- `MultiTypeEQParameterHead`: Parameter prediction from embeddings
- `StreamingTCNModel`: Complete model with streaming support
### Model Layer (Spectral Approach)
- `SpectralEQModel`: Predicts H_db from mel profile via MLP
- `param_extractor.py`: Post-hoc parameter extraction via scipy optimization
### Data Layer
- `SyntheticEQDataset`: On-the-fly synthetic signal generation
- `MUSDB18EQDataset`: Real audio from MUSDB18
- `collate_fn`: Padding and batching
### Loss Layer
- `HungarianBandMatcher`: DETR-style bipartite matching for band ordering
- `PermutationInvariantParamLoss`: Huber loss on matched parameters
- `MultiTypeEQLoss`: Combined loss components
### Training Layer
- `Trainer`: Custom training class (no Lightning dependency)
- `SpectralTrainer`: Training for spectral model
## Data Flow
### Training Flow
### Inference Flow
- Mel spectrogram (B, n_mels, T) → model.forward() → parameters + H_mag
- mel_frame (B, n_mels) → model.process_frame() → parameters
- Buffer accumulation over receptive field frames
- Cumulative skip connection mean for stable embeddings
- ONNX export: model → eq_estimator.onnx
- Plugin host: user inputs audio → mel_spectrogram → ONNX inference → biquad coefficients
- DSP layer stays in host (trivial computation, no ONNX benefit)
### Spectral Model Flow (Alternative)
## Key Abstractions
### HungarianBandMatcher
### MultiTypeEQParameterHead
### FrequencyAwareEncoder
## Entry Points
### Training Script
- Initialize Trainer with config from `conf/config.yaml`
- Load dataset (synthetic or MUSDB18) with optional precomputed cache
- Create StreamingTCNModel with FrequencyAwareEncoder
- Run curriculum-based training loop
- Save checkpoints, log metrics, implement early stopping
### Evaluation Script
- Load best checkpoint
- Run validation with Hungarian-matched metrics
- Export predictions or audio reconstruction results
### Export Script
- Export TCN encoder + parameter head to ONNX
- Supports dynamic batch and time frames for streaming
- Biquad coefficient computation stays in host code (plugin integration)
### Data Generation
- Generate synthetic multi-type EQ dataset with configurable parameters
- Support precomputation and caching for faster training
- Generate 200k samples for large-scale training
## Error Handling
- Runtime NaN detection in weights and buffers (BatchNorm stats)
- Checkpoint recovery: find clean checkpoint, reset optimizer state
- Nan batch skipping during training
- Per-parameter gradient norm monitoring
- NaN gradient clipping before optimizer step
- Gumbel-Softmax temperature annealing to prevent vanishing gradients
- BF16 mixed precision with gradient-safe operations
- Straight-through estimator for gradient flow
- Numerical clamping in biquad coefficient computation
## Cross-Cutting Concerns
- Per-component loss logging every N steps
- Gradient norm monitoring per parameter group
- Embedding variance diagnostic (anti-collapse health)
- Per-stage curriculum announcements
- Hungarian matching for fair parameter metrics
- Separate validation dataset for model selection
- Early stopping based on validation loss
- YAML-based hyperparameter configuration
- Curriculum stages with configurable lambda weights, gumbel temperature, LR scale
- Per-stage warmup for smooth LR transitions
- Standalone test scripts (no pytest): `test_eq.py`, `test_model.py`, `test_streaming.py`
- Gradient flow verification, model forward/inverse/cycle tests
- Streaming consistency validation
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
