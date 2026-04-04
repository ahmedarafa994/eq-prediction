# Apply Lightning Training Best Practices to IDSP EQ Estimator

## Context

The project has two parallel training systems in `insight/`:
1. **Custom `Trainer`** (`train.py`) — production-ready with NaN handling, AMP with FP32 bypass, curriculum dataset rebuilding, Hungarian matching, detailed metrics
2. **Lightning module** (`training/`) — minimal skeleton missing many critical features

The goal is to upgrade the Lightning training infrastructure to match the custom trainer's capabilities using idiomatic Lightning patterns, so the Lightning path becomes a viable alternative for training.

## Files to Modify

| File | Action |
|------|--------|
| `insight/training/data_module.py` | **New** — `SyntheticEQDataModule(pl.LightningDataModule)` |
| `insight/training/lightning_module.py` | **Major upgrade** — NaN handling, AMP bypass, Hungarian matching, model kwargs |
| `insight/training/curriculum.py` | **Major upgrade** — dataset rebuilding, full loss weight updates, scheduler reset |
| `insight/training/train.py` | **Moderate upgrade** — use DataModule, gradient clipping, curriculum callback, logger fallback |

**Not modified**: `train.py` (root custom trainer), `dataset.py`, `loss_multitype.py`, `model_tcn.py` — these are reference implementations the Lightning system wraps.

---

## Step 1: Create `insight/training/data_module.py`

New `SyntheticEQDataModule(pl.LightningDataModule)` wrapping the root `dataset.SyntheticEQDataset`:

- **`__init__(data_cfg, model_cfg, seed, initial_stage_cfg)`** — store config, determine precompute mode (mel vs full spectrum from `use_full_spectrum`)
- **`setup(stage=None)`** — build `SyntheticEQDataset` with current stage's param ranges, call `precompute()`, create seeded `random_split` for train/val/test
- **`setup_for_stage(stage_cfg)`** — public method for curriculum transitions: clears old dataset refs, runs `gc.collect()` + `torch.cuda.empty_cache()`, rebuilds with new param ranges (gain, Q, min_gain_db, type_weights)
- **`train_dataloader()` / `val_dataloader()` / `test_dataloader()`** — create DataLoaders with `collate_fn` from root `dataset.py`, matching custom trainer's kwargs: `pin_memory=False`, `drop_last=True` (train only), `prefetch_factor`, `persistent_workers`
- Type weight construction: uniform over active filter types, 0 for inactive (matching `train.py:170-175`)

## Step 2: Upgrade `insight/training/lightning_module.py`

### 2A: Fix missing model kwargs
Add `kernel_size=enc_cfg.get("kernel_size", 3)` and `use_full_spectrum=model_cfg.get("use_full_spectrum", False)` to `StreamingTCNModel(...)`. Store `self.n_fft` and `self.use_full_spectrum`.

### 2B: NaN-safe training
In `training_step()`: check `torch.isfinite(total_loss)`. If not finite, log `train/nan_batches` and return `torch.tensor(0.0, requires_grad=True)` to skip the batch.

Add `on_before_optimizer_step(optimizer)` hook: check all gradients for NaN, call `optimizer.zero_grad(set_to_none=True)` if detected.

### 2C: AMP with FP32 bypass for DSP
In `_common_step()`:
- Encoder runs under Lightning's autocast (automatic with `precision="16-mixed"`)
- Wrap param head + DSP + loss in `torch.amp.autocast("cuda", enabled=False)` with explicit `.float()` casts
- This prevents FP16 overflow in biquad coefficient computation (epsilon/products)

### 2D: Hungarian matching in validation
Add `self.val_matcher = HungarianBandMatcher(...)` in `__init__`. In `validation_step()`:
- Use `val_matcher()` for permutation-invariant gain/freq/Q/type matching
- Log: `val/gain_mae_db`, `val/freq_mae_oct`, `val/q_mae_dec`, `val/type_acc`
- Add per-region frequency error: low (<500Hz), mid (500-4000Hz), high (>4000Hz)
- Add frequency span metric

### 2E: Model health monitoring
Add `on_train_batch_end()` hook checking for NaN in model parameters every 100 steps.

### 2F: Input handling
Add `_prepare_input(batch)` supporting three modes: precomputed mel, precomputed full spectrum, raw audio via frontend.

### 2G: Upgrade `configure_optimizers()`
Add `eta_min=base_lr * 0.01` to `CosineAnnealingLR`. Return proper dict with `"interval": "epoch"`. Helper `_get_max_epochs()` computes total from curriculum stages.

## Step 3: Upgrade `insight/training/curriculum.py`

`CurriculumCallback(pl.Callback)` — accepts `config` and `data_module`:

On stage transition in `on_train_epoch_start()`:
1. **Rebuild dataset**: `self.data_module.setup_for_stage(stage_cfg)`
2. **Reset trainer dataloaders**: `trainer.reset_train_val_dataloaders(self.data_module)`
3. **Update ALL loss weights**: `lambda_type`, `lambda_spread`, `lambda_coverage`, `lambda_freq_anchor`, `lambda_freq_match`, `lambda_q_match` (on both criterion and val_matcher)
4. **Update Gumbel temperature**: `pl_module.model.param_head.gumbel_temperature.fill_(temp)`
5. **Reset optimizer LR**: `base_lr * lr_scale` for all param groups
6. **Replace scheduler**: new `CosineAnnealingLR(optimizer, T_max=stage_epochs, eta_min=...)`

## Step 4: Upgrade `insight/training/train.py`

1. **Replace `create_dataloaders`** with `SyntheticEQDataModule`
2. **Add `gradient_clip_val=1.0, gradient_clip_algorithm="norm"`** to Trainer
3. **Add `CurriculumCallback(config, data_module)`** to callbacks
4. **Fix checkpoint filename** to reference `val/total_loss`
5. **Add WandB → CSV logger fallback** (try/except on import)
6. **Pass `datamodule=data_module`** to `trainer.fit()` and `trainer.test()`

---

## Verification

1. **DataModule unit test**: Create small dataset (size=100), verify dataloaders return correct shapes, test `setup_for_stage()` transitions
2. **Lightning module smoke test**: Run existing `test_lightning_dummy.py` with the new DataModule — verify one train/val/test step completes without NaN
3. **Curriculum integration**: Run 2-3 epochs of training, verify:
   - Stage transitions print correctly
   - Loss weights update
   - Metrics appear in logs
   - Checkpoints save with correct monitor key
4. **Gradient clipping**: Verify `Trainer` logs clipped gradient norms
