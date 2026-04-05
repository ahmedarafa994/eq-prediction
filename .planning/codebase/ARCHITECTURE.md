# Architecture

**Analysis Date:** 2026-04-05

## Pattern Overview

**Overall:** End-to-end differentiable blind EQ estimation with hybrid TCN-spectral architecture

**Key Characteristics:**
- Blind estimation: Model receives only EQ'd audio (wet signal), must predict EQ parameters (gain, frequency, Q, filter type)
- Two parallel estimation approaches: parametric (TCN-based) and spectral (direct H_db prediction)
- Permutation-invariant training via Hungarian band matching (DETR-style)
- Streaming inference support for real-time DAW plugin deployment
- Multi-stage curriculum learning for stable convergence
- Anti-collapse mechanisms to prevent encoder collapse

## Layers

### DSP Layer
**Purpose:** Differentiable biquad filter coefficient computation and frequency response evaluation
**Location:** `differentiable_eq.py`
**Contains:**
- `StraightThroughClamp`: Custom autograd function for gradient flow
- `DifferentiableBiquadCascade`: Multi-type biquad filter support (5 filter types)
  - `compute_biquad_coeffs_multitype`: Coefficient formulas from Bristow-Johnson audio EQ cookbook
  - `forward_soft`: Training path using Gumbel-Softmax blended coefficients
  - `freq_response`: Magnitude response computation via FFT
  - `process_audio`: Time-domain filtering via STFT/iSTFT
  - `apply_to_spectrum`: Frequency-domain EQ application
**Depends on:** PyTorch tensors, scipy for coefficient computation
**Used by:** All models (parametric and spectral paths)

### Frontend Layer
**Purpose:** Differentiable STFT/iSTFT and mel-spectrogram extraction
**Location:** `dsp_frontend.py`
**Contains:**
- `STFTFrontend`: Manually-built mel filterbank (no torchaudio dependency in training)
  - `stft`: Complex STFT computation
  - `istft`: Inverse STFT for audio reconstruction
  - `mel_spectrogram`: Log-mel spectrogram extraction
  - `get_magnitude`/`get_complex`: Alternative accessors
- `apply_eq_to_complex_stft`: Helper for frequency-domain EQ application
**Depends on:** PyTorch, numpy
**Used by:** Training pipeline (STFT for loss), models (mel spectrograms)

### Model Layer (Parametric Approach)
**Purpose:** Main blind EQ estimation model using hybrid TCN encoder
**Location:** `model_tcn.py`
**Contains:**
- `FrequencyAwareEncoder`: Hybrid 2D + 1D TCN architecture
  - `SpectralFrontend2D`: 2D conv blocks preserving frequency locality
  - `FrequencyPreservingTCN`: Grouped 1D convs keeping frequency sub-bands separate
  - `AttentionTemporalPool`: Learned temporal attention replacing cumulative mean
  - Anti-collapse hooks: embedding_variance()
  - Spectral residual bypass: mean mel profile passed to parameter head
- `MultiTypeEQParameterHead`: Parameter prediction from embeddings
  - Gain: MLP regression from trunk embedding
  - Frequency: Attention over mel profile + direct regression
  - Q: Trunk features → log scale
  - Filter type: Gumbel-Softmax classification (5 types)
- `StreamingTCNModel`: Complete model with streaming support
  - `forward`: Batch mode training
  - `init_streaming`/`process_frame`: Streaming inference
**Depends on:** `differentiable_eq.py`, `dsp_frontend.py`
**Used by:** `train.py` (primary training), evaluation, inference

### Model Layer (Spectral Approach)
**Purpose:** Alternative blind EQ estimation bypassing encoder collapse
**Location:** `model_spectral.py` (if exists)
**Contains:**
- `SpectralEQModel`: Predicts H_db from mel profile via MLP
- `param_extractor.py`: Post-hoc parameter extraction via scipy optimization
**Depends on:** PyTorch, scipy
**Used by:** `train_spectral.py` (alternative training script)

### Data Layer
**Purpose:** Training data generation and loading
**Location:** `dataset.py`, `dataset_musdb.py`, `dataset_pipeline/`
**Contains:**
- `SyntheticEQDataset`: On-the-fly synthetic signal generation
  - Signal types: noise, sweep, harmonic, speech_like
  - Multi-type parameter sampling with weighted filter types
  - Augmentation: volume scaling
  - Precompute mode: cache mel-spectrograms in memory
- `MUSDB18EQDataset`: Real audio from MUSDB18
- `collate_fn`: Padding and batching
**Depends on:** numpy, torchaudio, scipy
**Used by:** `train.py` (DataLoader)

### Loss Layer
**Purpose:** Training objectives with permutation-invariant matching
**Location:** `loss_multitype.py`, `loss.py`
**Contains:**
- `HungarianBandMatcher`: DETR-style bipartite matching for band ordering
- `PermutationInvariantParamLoss`: Huber loss on matched parameters
- `MultiTypeEQLoss`: Combined loss components
  - Parameter regression (matched)
  - Filter type classification
  - Frequency response magnitude L1
  - Spectral consistency (MR-STFT if audio provided)
  - Band activity regularization
  - Frequency spread regularization
  - Embedding variance regularization (anti-collapse)
  - Contrastive loss (anti-collapse)
**Depends on:** torch, scipy (Hungarian matching)
**Used by:** `train.py` training loop

### Training Layer
**Purpose:** Model training orchestration
**Location:** `train.py`, `training/`
**Contains:**
- `Trainer`: Custom training class (no Lightning dependency)
  - Curriculum learning: 5-stage schedule
  - Optimizer: AdamW with per-group LR (encoder vs head)
  - LR schedule: CosineAnnealing + warmup
  - Gradient accumulation, mixed precision (bf16)
  - NaN recovery from checkpoints
  - SpecAugment for data augmentation
- `SpectralTrainer`: Training for spectral model
**Depends on:** All layers above
**Used by:** `train.py` entry point

## Data Flow

### Training Flow

1. **Data Generation:**
   - Synthetic dataset or MUSDB18 generates (dry_audio, wet_audio, params)
   - Wet audio → STFT → log-mel spectrogram (B, n_mels, T)

2. **Forward Pass:**
   - Mel spectrogram → FrequencyAwareEncoder
     - 2D conv blocks (spatial feature extraction)
     - Reshape → grouped TCN stacks (temporal modeling)
     - Attention pooling → embedding (B, D)
     - Spectral bypass: mean mel profile (B, n_mels)
   - Embedding + mel_profile → MultiTypeEQParameterHead
     - Gain: MLP regression
     - Frequency: Attention over mel profile
     - Q: Trunk features
     - Filter type: Gumbel-Softmax classification
   - Predicted parameters → DifferentiableBiquadCascade → predicted H_mag

3. **Loss Computation:**
   - Hungarian matching: match predicted → target bands
   - Parameter loss: Huber on matched gains/freq/Q
   - Type loss: Cross-entropy on matched types
   - H_mag loss: L1 on log-frequency response
   - Anti-collapse losses: variance + contrastive
   - Combined loss → backprop

4. **Optimization:**
   - Gradient accumulation (4 steps for effective BS=1024)
   - Mixed precision (bf16) for memory efficiency
   - AdamW optimizer with per-group LR
   - CosineAnnealing + warmup scheduler
   - Curriculum learning: stage transitions update lambda_type, gumbel_tau, LR scale

5. **Validation:**
   - Hungarian matching for fair MAE computation
   - Metrics: gain_mae, freq_mae (octaves), q_mae (decades), type_accuracy
   - Checkpoint saving with early stopping

### Inference Flow

**Batch Mode:**
- Mel spectrogram (B, n_mels, T) → model.forward() → parameters + H_mag

**Streaming Mode:**
- mel_frame (B, n_mels) → model.process_frame() → parameters
- Buffer accumulation over receptive field frames
- Cumulative skip connection mean for stable embeddings

**Real-time DAW Plugin:**
- ONNX export: model → eq_estimator.onnx
- Plugin host: user inputs audio → mel_spectrogram → ONNX inference → biquad coefficients
- DSP layer stays in host (trivial computation, no ONNX benefit)

### Spectral Model Flow (Alternative)

1. Mel spectrogram → MLP → predicted H_db
2. Post-hoc scipy optimization: find params minimizing ||H_pred - H_target||
3. Faster inference, but not trained end-to-end

**Result:** Spectral model achieved 0.20 dB MAE vs 5.77 dB for parametric approach (avoids multi-band decomposition problem)

## Key Abstractions

### HungarianBandMatcher
**Purpose:** Solves band permutation problem for permutation-invariant loss
**Location:** `loss_multitype.py`
**Pattern:** DETR-style bipartite matching using scipy.linear_sum_assignment
**Trade-off:** Introduces O(B·N²) cost but enables fair comparison of unmatched predictions

### MultiTypeEQParameterHead
**Purpose:** Constrained parameter prediction from encoder embeddings
**Location:** `differentiable_eq.py`
**Pattern:** Multi-head architecture (gain, freq, Q, type) with attention and MLPs
**Key Innovation:** Spectral residual bypass ensures parameter head always receives mel information

### FrequencyAwareEncoder
**Purpose:** Hybrid TCN encoder avoiding catastrophic encoder collapse
**Location:** `model_tcn.py`
**Pattern:** 2D front-end (preserves frequency locality) + grouped 1D TCN (temporal modeling) + attention pooling
**Key Innovation:** Anti-collapse regularization (variance + contrastive) + spectral residual bypass

## Entry Points

### Training Script
**Location:** `train.py`
**Triggers:** `python train.py` or `python train.py --resume checkpoints/best.pt`
**Responsibilities:**
- Initialize Trainer with config from `conf/config.yaml`
- Load dataset (synthetic or MUSDB18) with optional precomputed cache
- Create StreamingTCNModel with FrequencyAwareEncoder
- Run curriculum-based training loop
- Save checkpoints, log metrics, implement early stopping

### Evaluation Script
**Location:** `training/evaluate.py` (alternative) or standalone eval script
**Triggers:** `python evaluate_model.py`
**Responsibilities:**
- Load best checkpoint
- Run validation with Hungarian-matched metrics
- Export predictions or audio reconstruction results

### Export Script
**Location:** `export.py`
**Triggers:** `python export.py --checkpoint checkpoints/best.pt --output eq_estimator.onnx`
**Responsibilities:**
- Export TCN encoder + parameter head to ONNX
- Supports dynamic batch and time frames for streaming
- Biquad coefficient computation stays in host code (plugin integration)

### Data Generation
**Location:** `generate_dataset.py`, `generate_dataset_200k.py`
**Triggers:** `python generate_dataset.py` or `python generate_dataset_200k.py`
**Responsibilities:**
- Generate synthetic multi-type EQ dataset with configurable parameters
- Support precomputation and caching for faster training
- Generate 200k samples for large-scale training

## Error Handling

**NaN Recovery:**
- Runtime NaN detection in weights and buffers (BatchNorm stats)
- Checkpoint recovery: find clean checkpoint, reset optimizer state
- Nan batch skipping during training

**Gradient Issues:**
- Per-parameter gradient norm monitoring
- NaN gradient clipping before optimizer step
- Gumbel-Softmax temperature annealing to prevent vanishing gradients

**Stability Issues:**
- BF16 mixed precision with gradient-safe operations
- Straight-through estimator for gradient flow
- Numerical clamping in biquad coefficient computation

## Cross-Cutting Concerns

**Logging:**
- Per-component loss logging every N steps
- Gradient norm monitoring per parameter group
- Embedding variance diagnostic (anti-collapse health)
- Per-stage curriculum announcements

**Validation:**
- Hungarian matching for fair parameter metrics
- Separate validation dataset for model selection
- Early stopping based on validation loss

**Configuration:**
- YAML-based hyperparameter configuration
- Curriculum stages with configurable lambda weights, gumbel temperature, LR scale
- Per-stage warmup for smooth LR transitions

**Testing:**
- Standalone test scripts (no pytest): `test_eq.py`, `test_model.py`, `test_streaming.py`
- Gradient flow verification, model forward/inverse/cycle tests
- Streaming consistency validation

---

*Architecture analysis: 2026-04-05*
