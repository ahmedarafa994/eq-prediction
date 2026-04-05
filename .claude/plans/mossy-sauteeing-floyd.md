# Training Pipeline Audit Fix Plan

## Context

A comprehensive audit identified 20 issues (3 Critical, 6 High, 6 Medium, 5 Low) across security, reproducibility, correctness, performance, and operational reliability in the IDSP EQ estimator training pipelines. All 20 issues will be fixed using a 3-agent team executing sequentially (Agent 2 first for the largest `train.py` surface area, then Agents 1 and 3 in parallel on non-overlapping insertions).

---

## Execution Strategy

| Phase | Agent | Scope | Blocked by |
|-------|-------|-------|----------|
| A | Agent 2 | Training Pipeline Core | — |
| B | Agent 1 + Agent 3 | Security, Reproducibility, Stability | Agent 2 |

All agents use `isolation: "worktree"`. Changes are merged back after Phase B completes.

---

## File Ownership Matrix

| File | Agent 2 | Agent 1 | Agent 3 |
|------|---------|---------|---------|
| `train.py` | 2.1-2.7 (bulk) | 1.3, 1.4 (top of init) | 3.1, 3.5 (loop + end of fit) |
| `conf/config.yaml` | 2.7 | — | — |
| `dataset.py` | — | 1.1, 1.2 | — |
| `export.py` | — | 1.2 | 3.6 |
| `model_tcn.py` | — | — | 3.2, 3.3 |
| `loss_multitype.py` | — | — | 3.4 |

**train.py insertion points** (non-overlapping):
- Agent 2: modifies `_setup_stage`, `save_checkpoint`, `fit`, `_setup_flat_training`, DataLoader creation
 and adds resume logic + adds config keys
- Agent 1: inserts seeding block at top of `__init__` (after line 31)
 + fixes `random_split` call (inside `_setup_stage` and `_setup_flat_training`)
 — Agent 3: inserts NaN check inside `train_one_epoch` loop (after loss computation) + adds test evaluation at end of `fit`



---

## Phase A: Agent 2 — Training Pipeline Core

**Files**: `insight/train.py`, `insight/conf/config.yaml`

### 2.1 Fix curriculum dataset reuse bug (CRITICAL)
**Where**: `train.py:148-150`
- Remove `if self.train_loader is None` guard
 Always call `_build_stage_dataset()`)
- Free old loaders with `del` + `gc.collect()` before creating new ones
- Each stage gets its fresh `train_loader`/`val_loader` with correct param ranges
- Print confirmation of actual param ranges per stage transition

 **After**:
 ```python
 # Free old loaders
 if self.train_loader is not None:
     del self.train_loader
     del self.val_loader
     gc.collect()
 dataset = self._build_stage_dataset(stage_cfg)
 # ... split and build loaders
 ```

### 2.2 Add mixed precision training (HIGH)
**Where**: `train.py`
- `__init__`: add `self.use_amp = self.device.type == "cuda"` and `self.scaler`
 accordingly
- `train_one_epoch`: wrap forward+loss in `torch.amp.autocast("cuda", enabled=self.use_amp)`:
  ```python
  with torch.amp.autocast("cuda", enabled=self.use_amp):
      output = self.model(mel_frames)
      # ... loss computation ...
  if self.use_amp:
      self.scaler.scale(total_loss).backward()
      self.scaler.unscale_(self.optimizer)
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
      self.scaler.step(self.optimizer)
      self.scaler.update()
  else:
      total_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
      self.optimizer.step()
  ```
- `validate`: same autocast pattern with `torch.no_grad()`

### 2.3 Fix data pipeline efficiency (MEDIUM)
**Where**: `train.py:163,169`
- Read `num_workers` from `self.data_cfg` (default 0)
- Set `pin_memory=True` when CUDA available
- Add `persistent_workers=True` and `prefetch_factor=2` when `num_workers > 0`

### 2.4 Add checkpoint cleanup (MEDIUM)
**Where**: `train.py:save_checkpoint()`
- Keep `best.pt` + last 3 epoch checkpoints
- In `save_checkpoint()`, after saving new checkpoint, scan for old ones and delete:
  ```python
  ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  for old in ckpts[:-3]:
      if old.name != "best.pt":
          old.unlink()
  ```

### 2.5 Save scheduler state in checkpoints (MEDIUM)
**Where**: `train.py:316-323`
- Add `"scheduler_state_dict": self.scheduler.state_dict()` to checkpoint dict
- Add `"best_val_loss": self.best_val_loss`

### 2.6 Add resume-from-checkpoint (MEDIUM)
**Where**: `train.py`
- Add `resume_from` param to `Trainer.__init__`
- If path provided: load checkpoint, restore model, optimizer, scheduler, global_step, best_val_loss, epoch offset
 Resume training from correct epoch

### 2.7 Fix config inconsistencies (LOW)
**Where**: `train.py`, `conf/config.yaml`
- `train.py`: Read `duration` from `self.data_cfg.get("audio_duration", 3.0)` instead of hardcoding `1.5`
- `train.py`: Read `size` from `self.data_cfg.get("dataset_size", 50000)` instead of hardcoding `50000`
- `train.py`: Pass `type_weights` from `self.data_cfg` through to dataset instead of relying on dataset default
- `conf/config.yaml`: Add `dataset_size: 50000` and `audio_duration: 3.0` fields

---

## Phase B: Agent 1 — Security & Reproducibility (parallel with Agent 3)

**Files**: `insight/dataset.py`, `insight/export.py`, `insight/train.py`

### 1.1 Remove `allow_pickle=True` (CRITICAL)
**Where**: `dataset.py:259`
- Change `np.load(cache_path, allow_pickle=True)` to use NPZ-based caching that Store individual arrays (wet_mel, gain, freq, q, filter_type) as separate keys in `.npz`:
  ```python
  # Save
  np.savez_compressed(cache_path,
      wet_mels=np.stack([s["wet_mel"].numpy() for s in self._cache]),
  #       gain=gain_batch, freq=freq_batch, q=q_batch, filter_type=ft_batch.astype(int))

  )
  # Load
  data = np.load(cache_path)  # no allow_pickle needed for .npz
  wet_mels = data["wet_mels"]  # (N, n_mels, T) float array
  # etc.
  ```

### 1.2 Add `weights_only=True` to all `torch.load` (CRITICAL)
**Where**: `export.py:225`
- Change `torch.load(args.checkpoint, map_location="cpu")` to `torch.load(args.checkpoint, map_location="cpu", weights_only=True)`
- Scan entire `insight/` for other `torch.load` calls and fix them all

 Check `generate_dataset.py`, `test_checkpoint.py`, etc.)

### 1.3 Add seeding to custom trainer (HIGH)
**Where**: `train.py` — insert at top of `__init__` after line 31
- Add `import random` to imports
- Add seed block reading from `self.cfg.get("seed", 42)`:
  ```python
  seed = self.cfg.get("seed", 42)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

### 1.4 Fix `random_split` reproducibility (HIGH)
**Where**: `train.py:155-156` and `train.py:436-438`
- Pass seeded `Generator` to both `random_split` calls:
  ```python
  gen = torch.Generator().manual_seed(self.cfg.get("seed", 42))
  train_set, val_set, test_set = torch.utils.data.random_split(
      dataset, [n_train, n_val, n_test], generator=gen
 )
  ```

---

## Phase B: Agent 3 — Model, Loss & Stability (parallel with Agent 1)
**Files**: `insight/model_tcn.py`, `insight/loss_multitype.py`, `insight/export.py`, `insight/train.py`

### 3.1 Add NaN/Inf monitoring (HIGH)
**Where**: `train.py:train_one_epoch` — after loss computation (~line 237)
- After computing `total_loss`:
  ```python
  if torch.isnan(total_loss) or torch.isinf(total_loss):
      print(f"  WARNING: NaN/Inf loss at step {self.global_step}, skipping batch")
 + total_loss.item()
      continue
  ```

### 3.2 Fix streaming BatchNorm behavior (HIGH)
**Where**: `model_tcn.py:process_frame` (~line 260)
- Add training-mode check at start of `process_frame`:
  ```python
  if self.training:
      import warnings
      warnings.warn(
          "process_frame() called in training mode. "
          "Call model.eval() first for correct BatchNorm behavior in streaming mode."
 + total_loss.item()
  )
  ```

### 3.3 Fix streaming memory accumulation (LOW)
**Where**: `model_tcn.py:291-297`
- Replace cumulative sum with exponential moving average:
  ```python
  if self._cumulative_skip_sum is None:
      self._cumulative_skip_sum = skip_total
 else:
      alpha = min(1.0, 1.0 / self._frame_count)
      self._cumulative_skip_sum = (1 - alpha) * self._cumulative_skip_sum + alpha * skip_total
 ```
- Add `max_frames` parameter to `init_streaming` (default 1000) and reset state when exceeded

### 3.4 Harden Hungarian matching against NaN (HIGH)
**Where**: `loss_multitype.py:94-98`
- Already has `np.nan_to_num` (good). Add warning log before the conversion:
  ```python
  if np.any(np.isnan(cost_np)) or np.any(np.isinf(cost_np)):
      print(f"  WARNING: NaN/Inf in Hungarian cost matrix at batch {b}")
 + total_loss.item()
  ```

### 3.5 Add test set evaluation at training end (LOW)
**Where**: `train.py:fit` — after training loop completes
- If `test_loader` exists, run evaluation and print final test metrics
- Build `test_loader` if not already built (same as val_loader but with test split)

### 3.6 Validate ONNX export checkpoint (MEDIUM)
**Where**: `export.py:226-228`
- Add `strict=True` to `load_state_dict` call:
  ```python
  model.load_state_dict(checkpoint["model_state_dict"], strict=True)
  ```

### 3.7 Add octave error clipping in validation (LOW)
**Where**: `train.py:292`
- Clip octave error to [-4, 4] range for meaningful metric:
  ```python
  param_maes["freq"].append(
      torch.clamp(
          (torch.log2(pred_freq / (target_freq + 1e-8))).abs(), max=4.0
      ).mean().item()
  )
  ```

---

## Verification Plan

1. **Security fixes**: `grep -r "allow_pickle=True" insight/` and `grep -r "torch.load" insight/ | grep -v "weights_only"` — confirm no insecure calls remain
2. **All existing tests**: `cd insight && python test_eq.py && python test_model.py && python test_streaming.py && python test_multitype_eq.py`
 — all should pass
3. **Reproducibility**: Run `python train.py` twice with same seed — compare `training_history.json` outputs
4. **Curriculum fix**: Check printed param ranges per stage transition — verify they differ from stage 1's ranges
5. **Mixed precision**: `nvidia-smi` shows ~30% less GPU memory vs. baseline
6. **Resume**: Start training, interrupt at epoch 5, resume with `--resume checkpoints/epoch_005.pt` — continues from epoch 6
7. **NaN monitoring**: Verify NaN warning fires when injected (or observe natural training stability)
 improved)
8. **Streaming**: `python test_streaming.py` — passes with EMA fix
 streaming memory fix

## team_name: audit-fixes
