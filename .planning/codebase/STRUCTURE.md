# Codebase Structure

**Analysis Date:** 2026-04-05

## Directory Layout

```
this_studio/
├── insight/                      # Core IDSP system
│   ├── checkpoints/             # Model checkpoints (not tracked in git)
│   │   ├── best.pt
│   │   ├── epoch_001.pt
│   │   └── ...
│   ├── conf/                    # Configuration files
│   │   ├── config.yaml         # Main training config (120 epochs, multi-stage curriculum)
│   │   ├── config_simple.yaml  # Simplified peaking-only for verification
│   │   └── deepspeed_config.json
│   ├── data/                    # Data storage
│   │   ├── musdb18_hq/         # MUSDB18 dataset (real audio)
│   │   └── dataset_musdb_200k.pt  # Precomputed dataset cache
│   ├── data_cache/              # Cache directory (git-ignored)
│   ├── dataset_pipeline/        # Offline data generation with MUSDB18
│   │   └── generate_data.py    # Multi-type data generation script
│   ├── diagnostics/             # Diagnostic scripts (git-ignored output)
│   ├── docs/                    # Documentation
│   ├── fixes/                   # Patch files for experiments
│   ├── mlops/                   # MLOps utilities
│   ├── training/                # Alternative training implementation
│   │   ├── curriculum.py       # Curriculum learning logic
│   │   ├── evaluate.py         # Evaluation script
│   │   ├── lightning_module.py # PyTorch Lightning module
│   │   └── train.py            # Lightning training script
│   ├── .github/                 # GitHub workflows
│   ├── .serena/                 # Serena config
│   ├── augment.py               # SpecAugment for data augmentation
│   ├── calibrate.py             # Temperature scaling for Gumbel-Softmax
│   ├── diagnose_gain.py         # Gain prediction diagnostics
│   ├── diagnose_gradients.py    # Gradient flow diagnostics
│   ├── differentiable_eq.py     # DSP layer: biquad coefficients & frequency response
│   ├── dsp_frontend.py          # STFT/iSTFT & mel-spectrogram extraction
│   ├── dataset.py               # Synthetic dataset generation
│   ├── dataset_musdb.py         # MUSDB18 dataset loader
│   ├── export.py                # ONNX model export
│   ├── fused_kernels.py         # Optimized kernel code
│   ├── generate_dataset.py      # Synthetic dataset generation (deprecated)
│   ├── generate_dataset_200k.py # Batch dataset generation
│   ├── gumbel_annealing.py      # Gumbel-Softmax temperature annealing
│   ├── loss.py                  # Single-resolution STFT loss (legacy)
│   ├── loss_multitype.py        # Multi-type EQ loss with Hungarian matching
│   ├── model_cnn.py             # 2D CNN model (peaking-only, legacy)
│   ├── model_spectral.py        # Spectral prediction model (alternative approach)
│   ├── model_tcn.py             # Hybrid TCN model with streaming support
│   ├── optimizer_setup.py       # Optimizer & LR scheduler config
│   ├── param_extractor.py       # Post-hoc parameter extraction (spectral model)
│   ├── plot_musdb_training.py   # MUSDB18 training analysis
│   ├── precompute_stages.py     # Stage-wise precomputation
│   ├── profile_gpu.py           # GPU profiling utilities
│   ├── train.py                 # Main training script (custom Trainer)
│   ├── train_spectral.py        # Training script for spectral model
│   ├── augment.py               # SpecAugment implementation
│   ├── analyze_learning_curves.py  # Training log analysis
│   ├── analyze_run.py           # Run analysis
│   ├── audit_loss.py            # Loss component auditing
│   ├── create_dummy.py          # Dummy data creation
│   ├── debug_attn.py            # Attention visualization
│   ├── debug_peaks.py           # EQ peak visualization
│   ├── test_eq.py               # Gradient flow tests
│   ├── test_model.py            # Model forward/inverse tests
│   ├── test_streaming.py        # Streaming consistency tests
│   ├── test_multitype_eq.py     # Multi-type filter tests
│   ├── test_checkpoint.py       # Checkpoint loading tests
│   ├── test_checkpoint_multi.py # Multi-checkpoint tests
│   └── requirements_optimized.txt  # Python dependencies
└── [other directories]          # Lightning AI environment, caches, etc.
```

## Directory Purposes

### insight/checkpoints/
**Purpose:** Model checkpoint storage
**Contains:**
- `best.pt`: Best validation model (determined by val_loss)
- `epoch_001.pt` through `epoch_NNN.pt`: Periodic checkpoints
- Checkpoint format: model_state_dict, optimizer_state_dict, scheduler_state_dict, val_loss, global_step
**Generated:** Training loop, auto-save on each epoch and improvement

### insight/conf/
**Purpose:** Configuration management
**Contains:**
- `config.yaml`: Main training configuration (120 epochs, 5-stage curriculum, 1024 batch size)
- `config_simple.yaml`: Simplified peaking-only config for verification
- `deepspeed_config.json`: DeepSpeed optimization settings
**Pattern:** YAML with sections for data, model, loss, trainer, curriculum

### insight/data/
**Purpose:** Dataset storage
**Contains:**
- `musdb18_hq/`: MUSDB18 dataset (train/test splits)
- `dataset_musdb_200k.pt`: Precomputed dataset cache (large file, 200k samples)
**Usage:** MUSDB18 for real audio training, cached dataset for faster iteration

### insight/dataset_pipeline/
**Purpose:** Offline multi-type data generation with MUSDB18
**Contains:**
- `generate_data.py`: Multi-process data generation script
**Usage:** Generate large-scale synthetic datasets with configurable signal types and filter type weights

### insight/diagnostics/
**Purpose:** Diagnostic outputs (git-ignored)
**Contains:**
- Visualizations of attention weights, EQ peaks, training curves
- Debug outputs from diagnostic scripts

### insight/docs/
**Purpose:** Documentation
**Contains:**
- Model documentation, training notes, architecture diagrams

### insight/training/
**Purpose:** Alternative training implementation (PyTorch Lightning)
**Contains:**
- `curriculum.py`: Curriculum learning logic
- `evaluate.py`: Evaluation script
- `lightning_module.py`: PyTorch Lightning module
- `train.py`: Lightning training script
**Usage:** Alternative training approach, maintained for comparison

### insight/.github/
**Purpose:** GitHub CI/CD workflows
**Contains:**
- Workflows for automated testing, training, and deployment

### insight/.serena/
**Purpose:** Serena configuration (semantic code analysis)

## Key File Locations

### Core Models
**Primary Model:** `model_tcn.py` (754 lines)
- `FrequencyAwareEncoder`: Hybrid TCN encoder
- `MultiTypeEQParameterHead`: Parameter prediction
- `StreamingTCNModel`: Complete model with streaming

**Legacy Model:** `model_cnn.py` (3488 bytes)
- `EQEstimatorCNN`: 2D CNN on mel-spectrograms (peaking-only)

**Spectral Model:** `model_spectral.py` (if exists)
- `SpectralEQModel`: Direct H_db prediction

### DSP Layer
**Differentiable EQ:** `differentiable_eq.py` (830 lines)
- `DifferentiableBiquadCascade`: Biquad filter coefficients
- `EQParameterHead`/`MultiTypeEQParameterHead`: Parameter heads

### Frontend
**STFT/iSTFT:** `dsp_frontend.py` (5176 bytes)
- `STFTFrontend`: Mel-spectrogram extraction

### Loss Functions
**Multi-type Loss:** `loss_multitype.py` (482 lines)
- `HungarianBandMatcher`: Band permutation solving
- `MultiTypeEQLoss`: Combined training objectives

**Legacy Loss:** `loss.py` (246 lines)
- `MultiResolutionSTFTLoss`: Spectral consistency

### Training
**Main Trainer:** `train.py` (1019 lines)
- `Trainer`: Custom training class
- Curriculum learning, NaN recovery, gradient accumulation

**Alternative:** `training/train.py`
- PyTorch Lightning training script

### Data
**Synthetic Dataset:** `dataset.py` (405 lines)
- `SyntheticEQDataset`: On-the-fly generation

**MUSDB18 Dataset:** `dataset_musdb.py` (342 lines)
- `MUSDB18EQDataset`: Real audio loader

### Utilities
**ONNX Export:** `export.py` (252 lines)
- Model export for DAW plugins

**GPU Profiling:** `profile_gpu.py` (303 lines)
- Performance optimization tools

## Naming Conventions

### Files
**Core Models:**
- `model_tcn.py`: TCN-based model
- `model_cnn.py`: CNN-based model (legacy)
- `model_spectral.py`: Spectral prediction model

**DSP Layer:**
- `differentiable_eq.py`: Differentiable biquad filters
- `dsp_frontend.py`: STFT/iSTFT processing

**Loss Functions:**
- `loss_multitype.py`: Multi-type EQ loss
- `loss.py`: Legacy STFT loss

**Datasets:**
- `dataset.py`: Synthetic dataset
- `dataset_musdb.py`: MUSDB18 dataset

**Training:**
- `train.py`: Main custom trainer
- `train_spectral.py`: Spectral model trainer

**Tests:**
- `test_*.py`: Standalone test scripts (no pytest)

**Diagnostic/Debug:**
- `diagnose_*.py`: Gain, gradient diagnostics
- `debug_*.py`: Attention, peak debugging
- `analyze_*.py`: Learning curve analysis

**Generation:**
- `generate_dataset*.py`: Batch dataset generation

**Utilities:**
- `export.py`: ONNX export
- `profile_gpu.py`: GPU profiling
- `calibrate.py`: Gumbel-Softmax calibration
- `gumbel_annealing.py`: Temperature annealing

### Classes
**Models:**
- `StreamingTCNModel`: Complete model with streaming
- `FrequencyAwareEncoder`: Hybrid TCN encoder
- `MultiTypeEQParameterHead`: Parameter prediction head
- `SpectralEQModel`: Spectral prediction model

**DSP:**
- `DifferentiableBiquadCascade`: Biquad filter coefficients
- `STFTFrontend`: STFT/iSTFT processing

**Loss:**
- `MultiTypeEQLoss`: Combined loss with Hungarian matching
- `HungarianBandMatcher`: Band permutation solver

**Datasets:**
- `SyntheticEQDataset`: Synthetic data generation
- `MUSDB18EQDataset`: Real audio loader

### Scripts
- `train.py`: Training entry point
- `evaluate_model.py`: Evaluation entry point
- `export.py`: ONNX export entry point
- `generate_dataset.py`: Dataset generation entry point

## Where to Add New Code

### New Model Architecture
**Primary location:** `model_tcn.py` (or new file)
- Add new encoder classes inheriting from `nn.Module`
- Implement `forward()` method returning dict with params and H_mag
- Support batch and streaming modes via `init_streaming()`/`process_frame()`
- Add to `StreamingTCNModel` if it's a variant

### New Loss Function
**Primary location:** `loss_multitype.py` (or new file)
- Create new loss class inheriting from `nn.Module`
- Implement `forward()` returning (total_loss, components_dict)
- Add to `MultiTypeEQLoss` components if it's a combined loss
- Update `train.py` Trainer class to instantiate and use

### New Dataset
**Primary location:** New file (e.g., `dataset_custom.py`)
- Create `dataset.CustomDataset` inheriting from `data.Dataset`
- Implement `__len__()` and `__getitem__()`
- Add to Trainer class `__init__()` if used for training
- Update collate_fn if custom tensor format needed

### New Training Configuration
**Primary location:** `conf/config.yaml` (or new file)
- Add sections for new hyperparameters
- Update Trainer class to load from config
- Add curriculum stages if curriculum learning needed

### New Utility
**Primary location:** New file in `insight/` root
- Follow naming conventions: `diagnose_*.py`, `analyze_*.py`, `profile_*.py`
- Import from main module, export if needed
- Add to tests or documentation

### New Model Export Format
**Primary location:** `export.py`
- Add export function for new format (e.g., TorchScript, ONNX)
- Follow pattern of `export_onnx()`
- Update command-line interface if needed

### New Diagnostic Tool
**Primary location:** New file (e.g., `diagnose_new_metric.py`)
- Follow naming conventions: `diagnose_*.py`
- Load checkpoint, run model on test data
- Visualize or log new metrics
- Add to test suite if used for validation

### New Curriculum Stage
**Primary location:** `conf/config.yaml` (curriculum section)
- Add stage to `stages` list with `name`, `epochs`, `lambda_type`, `gumbel_temperature`, `learning_rate_scale`
- Implement `_update_type_transition()` in Trainer if data distribution changes
- Add to `train.py` if stage-specific initialization needed

### New Spectral Model
**Primary location:** `model_spectral.py` (or new file)
- Create new model class inheriting from `nn.Module`
- Implement forward pass predicting parameters or H_db
- Create new trainer in `train_spectral.py` if needed
- Add to `evaluate_model.py` evaluation script

### New Test
**Primary location:** New file (e.g., `test_new_feature.py`)
- Follow naming convention: `test_*.py`
- Standalone script (no pytest) testing specific functionality
- Add to `CLAUDE.md` commands section if used in workflow

## Special Directories

### .github/
**Purpose:** GitHub CI/CD workflows
**Contains:** Automated testing, training, deployment pipelines
**Generated:** No, manually configured in workflow YAML files

### .serena/
**Purpose:** Serena semantic code analysis configuration
**Contains:** Configuration for code understanding tools
**Generated:** No, tool-specific configuration

### data_cache/
**Purpose:** Temporary data cache
**Contains:** Cached files, intermediate results
**Generated:** No, managed by scripts

### diagnostics/
**Purpose:** Diagnostic outputs and visualizations
**Contains:** Plots, logs from diagnostic scripts
**Generated:** Yes, by `diagnose_*.py` scripts

### fixes/
**Purpose:** Patch files for experiments
**Contains:** Modified versions of files for testing ideas
**Generated:** Yes, by developers

### training/
**Purpose:** Alternative training implementation
**Contains:** PyTorch Lightning module and scripts
**Generated:** No, code maintained in parallel

### checkpoints/
**Purpose:** Model checkpoint storage
**Contains:** Saved model states, optimizer states, training history
**Generated:** Yes, by training loop

---

*Structure analysis: 2026-04-05*
