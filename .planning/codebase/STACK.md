# Technology Stack

**Analysis Date:** 2026-04-05

## Languages

**Primary:**
- Python 3.x - Core language (indicated by pyproject.toml, pip-style requirements)
- TypeScript/JavaScript - Node.js development environment (node-v25.9.0)

**Secondary:**
- YAML - Configuration files
- Bash - Shell scripting and development automation

## Runtime

**Environment:**
- Python 3.x - Primary runtime for all training, evaluation, and data processing scripts
- Node.js v25.9.0 - Development environment (dev-only, not used in core project)

**Package Manager:**
- pip - Python package management (requirements_optimized.txt, ml_framework/requirements.txt)
- npm - Node.js package management (dev-only, node_modules/)
- Lockfile: No lockfile detected (pip requirements files present, but no pipfile.lock or poetry.lock)

## Frameworks

**Core:**
- PyTorch >=2.0.0 - Deep learning framework
  - torchaudio >=2.0.0 - Audio processing utilities
  - torch.compile - Graph-level optimization
- PyTorch Lightning >=1.6.0 (optional) - Alternative training framework (training/ directory)

**Data Processing:**
- numpy >=1.24.0 - Numerical computing
- scipy >=1.10.0 - Signal processing (biquad filter coefficients, Hungarian matching)
- torchaudio >=2.0.0 - Audio I/O and transformations

**Audio-Specific:**
- Custom STFT implementation (dsp_frontend.py) - No external audio framework dependency for training

**Visualization:**
- matplotlib - Plotting and visualization (test_checkpoint.py, plot_musdb_training.py, diagnose_gain.py, analyze_learning_curves.py)

**Testing:**
- Standalone pytest-style scripts (no framework detected)
  - test_eq.py
  - test_model.py
  - test_streaming.py
  - test_multitype_eq.py
  - test_lightning_dummy.py
  - test_checkpoint*.py
  - test_collapse_fix.py
  - test_ste_clamp.py

**ML Utilities:**
- pyyaml >=6.0 - Configuration file parsing (config.yaml, conf/*.yaml)
- tqdm - Progress bars (train.py, training/evaluate.py)
- argparse - Command-line argument parsing (test scripts, utilities)

**Optional Frameworks:**
- PyTorch Lightning - Used in training/ directory (lightning_module.py, curriculum.py)

## Key Dependencies

**Critical (Deep Learning):**
- torch >=2.0.0 - Core deep learning framework
- torchaudio >=2.0.0 - Audio signal processing
- numpy >=1.24.0 - Numerical arrays and DSP operations
- scipy >=1.10.0 - Signal processing (biquad coefficient computation, Hungarian matching)

**Training Optimization:**
- bitsandbytes >=0.41.0 - 8-bit optimizer for memory efficiency (optional)
- triton >=2.1.0 - GPU kernel optimization (Linux only)
- deepspeed >=0.12.0 - Distributed training (optional)
- onnxruntime >=1.16.0 - ONNX model inference (optional)

**Infrastructure:**
- pyyaml >=6.0 - Configuration management
- argparse - CLI argument handling

## Configuration

**Environment:**
- Config files: YAML format in `insight/conf/` directory
  - config.yaml - Primary training configuration
  - config_simple.yaml - Simplified peaking-only configuration
  - config_spectral.yaml - Spectral prediction model configuration
  - config_full_spectrum.yaml - Full-band magnitude spectrum variant
  - config_musdb_50k.yaml - MUSDB18 50k dataset configuration
  - config_musdb_200k.yaml - MUSDB18 200k dataset configuration
  - deepspeed_config.json - DeepSpeed configuration

**Build:**
- No build process detected (Python scripts are directly runnable)
- Optional torch.compile for graph-level optimization (train.py)

**Environment Variables:**
- No .env files detected in repository (checked .gitignore patterns)

## Platform Requirements

**Development:**
- Python 3.x (version not specified in requirements)
- Linux (required for triton kernel optimization)
- GPU with CUDA support (for PyTorch operations)

**Production:**
- ONNX Runtime for deployment (optional)
- DAW plugin host for real-time inference (post-export)

---

*Stack analysis: 2026-04-05*
