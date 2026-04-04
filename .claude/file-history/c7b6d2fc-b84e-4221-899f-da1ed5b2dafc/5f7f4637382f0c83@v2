# GPU Optimization Plan: IDSP EQ Estimator Training

## Context

**Hardware**: NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM), CUDA 13.0, Blackwell arch
**Framework**: PyTorch 2.8.0+cu128
**Current state**: Training crashes before completing a single epoch. GPU utilization is 0%, VRAM usage 559 MiB / 102 GB.

**Root causes identified**:
1. The custom `train.py` crashes immediately: `trainer_cfg` referenced on line 45 before assignment on line 56
2. The Lightning variant (`training/train.py`) crashes at validation: `pin_memory=True` applied to GPU-resident tensors causes `RuntimeError: cannot pin 'torch.cuda.FloatTensor'`
3. Even if both bugs were fixed, GPU utilization would be terrible: batch_size=1024 with a 1.6M-param model on a 102 GB GPU

---

## Changes

### 1. Fix crash: `trainer_cfg` used before definition (`train.py:45-48`)

**Why**: `trainer_cfg` is referenced before it's assigned. This is a NameError at init time.

**Before** (`train.py:44-48`):
```python
torch.backends.cudnn.deterministic = trainer_cfg.get("cudnn_deterministic", False)
torch.backends.cudnn.benchmark = trainer_cfg.get("cudnn_benchmark", True)
# ...
trainer_cfg = self.cfg["trainer"]  # line 56 — too late
```

**After**: Move lines 45-48 to after line 56 where `trainer_cfg` is assigned.

**File**: `insight/train.py`

### 2. Fix crash: `self.scaler` never initialized (`train.py`)

**Why**: The AMP branch calls `self.scaler.scale()`, `self.scaler.step()`, `self.scaler.update()` but `GradScaler` is never created. BF16 on Blackwell does NOT need a scaler (same exponent range as FP32). The comment on line 156 even says "no GradScaler needed" but the code still uses one.

**Before**: `self.scaler.scale(total_loss).backward()` etc.
**After**: Plain `total_loss.backward()` → `clip_grad_norm_` → `optimizer.step()` — no scaler needed with BF16.

**File**: `insight/train.py` — `train_one_epoch()` method (lines ~380-451)

### 3. Fix crash: `pin_memory=True` on GPU-resident data (`data_module.py`)

**Why**: The dataset precomputes to GPU (`_preload_to_gpu()`), so `__getitem__` returns CUDA tensors. `pin_memory=True` in DataLoader calls `.pin_memory()` on the batch, which fails: "cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned". This is visible in the training log at the val sanity check.

**Before** (`data_module.py:157-175`):
```python
def _dataloader(self, dataset, shuffle=False):
    ...
    return DataLoader(
        dataset, ..., pin_memory=not has_gpu_cache, ...
    )
```

The `_dataloader` already checks for `_gpu_cache`, but the attribute is named `_gpu_cache` while the dataset stores it on `self._gpu_cache` on the underlying dataset (not the `random_split` Subset wrapper). The check `hasattr(dataset, 'dataset') and hasattr(dataset.dataset, '_gpu_cache')` should catch this, but the `_dataloader` is called with the Subset from `random_split`, and `_gpu_cache` may not exist if precompute hasn't run yet.

**After**: Always disable `pin_memory` when data is on GPU. Also fix the same issue in `train.py`'s `_setup_stage()` and `_setup_flat_training()`.

**Files**: `insight/training/data_module.py`, `insight/train.py`

### 4. Increase batch size from 1024 to 4096

**Why**: batch_size=1024 with BF16 on a 102 GB GPU uses ~2 GB VRAM (model ~6 MB, optimizer ~12 MB, activations ~1.5 GB, dataset preloaded ~21 GB). Over 75 GB is completely wasted. Larger batches increase SM occupancy and kernel launch efficiency.

Current VRAM budget estimate (batch=4096, BF16):
- Model params: 6 MB
- Optimizer states (AdamW, FP32): 24 MB
- Preloaded dataset: 21 GB
- Input mel (4096 × 256 × 517 × 2 bytes): ~270 MB
- Skip connections (4096 × 128 × 517 × 2 bytes): ~135 MB
- Activations + gradients: ~2 GB
- Total: ~24 GB — well within 102 GB

**File**: `insight/conf/config.yaml` — `batch_size: 1024` → `batch_size: 4096`

### 5. Add gradient accumulation (effective batch 16K)

**Why**: Batch 4096 is still small for this model. Accumulating 4 micro-batches (4096 × 4 = 16384) gives more stable gradients at zero extra VRAM cost. This is especially important because the Hungarian matching loss benefits from diverse batch statistics.

**File**: `insight/conf/config.yaml` — `accumulate_grad_batches: 1` → `accumulate_grad_batches: 4`
**File**: `insight/train.py` — implement accumulation in `train_one_epoch()`:
```python
# Accumulate gradients over grad_accum_steps micro-batches
total_loss = total_loss / self.grad_accum_steps
total_loss.backward()
if (batch_idx + 1) % self.grad_accum_steps == 0:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
```

### 6. Enable `torch.compile` for the encoder

**Why**: The TCN encoder has many small ops (tanh, sigmoid, multiply for gating, batch norm) that benefit from kernel fusion. `torch.compile(mode="reduce-overhead")` fuses these into fewer CUDA kernels, reducing launch overhead and improving memory locality. The code already has the compile logic gated behind a config flag.

**File**: `insight/conf/config.yaml` — add `torch_compile: true` under `trainer:`
No code change needed — `train.py:171-177` already handles this.

### 7. Fuse wet/dry STFT into single batched operation

**Why**: `_prepare_input()` and `_prepare_dry_mel()` each compute a separate STFT on the same-length audio. Two STFTs per batch doubles the cuFFT overhead. By stacking wet+dry audio, running one STFT, and splitting the mel spectrograms, we halve the STFT time.

**Before** (`train.py:353-362`):
```python
mel_frames = self._prepare_input(batch)       # STFT on wet_audio
dry_mel = self._prepare_dry_mel(batch)         # STFT on dry_audio (redundant)
```

**After**:
```python
def _prepare_both_mels(self, batch):
    """Compute wet and dry mel-spectrograms in a single batched STFT."""
    wet = batch["wet_audio"].to(self.device, non_blocking=True)
    dry = batch["dry_audio"].to(self.device, non_blocking=True)
    stacked = torch.cat([wet, dry], dim=0)    # (2B, T)
    mel = self.frontend.mel_spectrogram(stacked).squeeze(1)  # (2B, n_mels, T)
    B = wet.shape[0]
    return mel[:B], mel[B:]                    # wet_mel, dry_mel
```

**File**: `insight/train.py` — new method + update `train_one_epoch()` and `validate()`

### 8. Disable `pin_memory` for GPU-resident precomputed data

**Why**: The `_preload_to_gpu()` method moves the entire dataset to VRAM. `pin_memory=True` allocates pinned host memory for faster CPU→GPU transfer, but when data is already on GPU, it's wasteful and causes crashes (as seen in the training log). Set `pin_memory=False` when using precomputed GPU data.

**File**: `insight/train.py` — `_setup_stage()` and `_setup_flat_training()`:
```python
# Detect GPU-resident data (precompute moves everything to VRAM)
pin_memory = False  # data is already on GPU via _preload_to_gpu()
```

### 9. Add batch size auto-finder utility

**Why**: Hardcoded batch sizes are fragile across different GPU configurations. A one-time calibration step at training start finds the maximum batch size that fits VRAM, then applies gradient accumulation to reach the target effective batch size.

**Implementation**: Add `_find_max_batch_size()` method to `Trainer` class:
1. Start at batch_size=512
2. Double until OOM
3. Binary search between last success and OOM
4. Apply 80% safety margin
5. Use result for the rest of training

**File**: `insight/train.py` — new method, called from `fit()`

---

## Files to modify

| File | Changes |
|------|---------|
| `insight/train.py` | Fix `trainer_cfg` bug (#1), remove scaler (#2), fix pin_memory (#8), fuse STFT (#7), add grad accum (#5), add batch size finder (#9) |
| `insight/conf/config.yaml` | batch_size → 4096 (#4), accumulate_grad_batches → 4 (#5), torch_compile → true (#6) |
| `insight/training/data_module.py` | Fix pin_memory crash for Lightning variant (#3) |

---

## Verification

1. **Smoke test**: `cd insight && python train.py --config conf/config.yaml` — should start training without any NameError or RuntimeError
2. **GPU utilization**: While training, run `nvidia-smi -l 1` in another terminal. Target: >70% GPU-Util, >30 GB Memory-Usage
3. **Throughput**: Compare time-per-epoch before and after. Expected ~3-4x from batch size + compile + STFT fusion
4. **Numerical correctness**: BF16 encoder + FP32 DSP should produce no NaN. Loss should converge at similar rate per effective batch (same hyperparameters)
5. **`torch.compile` warmup**: First 2-3 steps will be slow (JIT compilation). After warmup, per-step time should decrease ~10-30%
