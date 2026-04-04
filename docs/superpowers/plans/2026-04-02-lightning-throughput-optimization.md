# Lightning Throughput Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate default training to the Lightning-based trainer and apply Blackwell-optimized throughput settings (BF16, torch.compile, TF32, pin_memory, zero_grad set_to_none).

**Architecture:** Four targeted file edits — config, training script, data module, lightning module — then a fast_dev_run smoke test before switching the live training run.

**Tech Stack:** PyTorch Lightning 2.x, torch.compile (reduce-overhead), BF16 mixed precision, AdamW

---

## Files

| File | Change |
|------|--------|
| `insight/conf/config.yaml` | Update `trainer:` section — bf16-mixed, max_epochs 90, check_val_every_n_epoch 5 |
| `insight/training/train.py` | Add TF32 flag, torch.compile, deterministic=False, check_val_every_n_epoch |
| `insight/training/data_module.py` | pin_memory=True, prefetch_factor=4 |
| `insight/training/lightning_module.py` | Add optimizer_zero_grad override |
| `insight/test_lightning_dummy.py` | Smoke test — no changes, just run it |

---

## Task 1: Update `conf/config.yaml` trainer section

**Files:**
- Modify: `insight/conf/config.yaml`

- [ ] **Step 1: Open the file and locate the `trainer:` block (currently around line 57)**

- [ ] **Step 2: Replace the entire `trainer:` block with**

```yaml
trainer:
  max_epochs: 90
  accelerator: gpu
  devices: 1
  precision: bf16-mixed
  log_every_n_steps: 10
  check_val_every_n_epoch: 5
  accumulate_grad_batches: 1
  fast_dev_run: false
```

- [ ] **Step 3: Verify the change**

```bash
grep -A 10 "^trainer:" insight/conf/config.yaml
```

Expected output:
```
trainer:
  max_epochs: 90
  accelerator: gpu
  devices: 1
  precision: bf16-mixed
  log_every_n_steps: 10
  check_val_every_n_epoch: 5
  accumulate_grad_batches: 1
  fast_dev_run: false
```

---

## Task 2: Update `training/train.py`

**Files:**
- Modify: `insight/training/train.py`

- [ ] **Step 1: Add `import torch` at the top** (it's not currently imported directly in this file)

Add after the existing imports (after `import lightning as L`):
```python
import torch
```

- [ ] **Step 2: Add TF32 and torch.compile before `trainer.fit()`**

Locate the line `data_module.setup()` (currently line 106). Insert these lines immediately before it:

```python
    torch.set_float32_matmul_precision("high")

    # Compile inner model only — avoids compiling Lightning internals
    model.model = torch.compile(model.model, mode="reduce-overhead")
```

- [ ] **Step 3: Update the `Trainer(...)` call**

Find the existing `Trainer(...)` instantiation. Replace these two lines:
```python
        deterministic="warn",
```
with:
```python
        deterministic=False,
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
```

The full updated `Trainer(...)` call should look like:
```python
    trainer = L.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 30),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "16-mixed"),
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        deterministic=False,
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        fast_dev_run=trainer_cfg.get("fast_dev_run", False),
    )
```

---

## Task 3: Update `training/data_module.py`

**Files:**
- Modify: `insight/training/data_module.py`

- [ ] **Step 1: Update `_dataloader()` — pin_memory and prefetch_factor**

Find the `DataLoader(...)` call inside `_dataloader()` (currently around line 147). Replace:
```python
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=shuffle and bs < len(dataset),
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )
```
with:
```python
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=shuffle and bs < len(dataset),
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )
```

---

## Task 4: Update `training/lightning_module.py`

**Files:**
- Modify: `insight/training/lightning_module.py`

- [ ] **Step 1: Add `optimizer_zero_grad` override to `EQEstimatorLightning`**

Find the `on_before_optimizer_step` method (currently around line 312). Insert the following method directly before it:

```python
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """Use set_to_none=True for faster gradient reset."""
        optimizer.zero_grad(set_to_none=True)

```

---

## Task 5: Smoke test

**Files:**
- Run: `insight/test_lightning_dummy.py` (no changes — runs fast_dev_run on CPU)

- [ ] **Step 1: Kill the current training process**

```bash
pkill -f "python train.py" 2>/dev/null; sleep 1; echo "done"
```

- [ ] **Step 2: Run the smoke test from `insight/`**

```bash
cd /teamspace/studios/this_studio/insight && python test_lightning_dummy.py
```

Expected last line:
```
Lightning Verification Passed
```

If it fails with a compile error, check that `model.model` exists on the `EQEstimatorLightning` instance — it's set in `__init__` at `self.model = StreamingTCNModel(...)`.

---

## Task 6: Start Lightning training

- [ ] **Step 1: Clear any stale dataset cache from the interrupted run**

```bash
rm -f /teamspace/studios/this_studio/insight/data_cache/*.npz
```

- [ ] **Step 2: Start training**

```bash
cd /teamspace/studios/this_studio/insight/training && nohup python train.py --config ../conf/config.yaml > /tmp/train_lightning.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 3: Confirm GPU is active after ~60s (dataset precomputation then first batch)**

```bash
sleep 60 && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Expected: utilization > 0%, memory.used > 2000 MiB

- [ ] **Step 4: Confirm first epoch logged**

```bash
tail -30 /tmp/train_lightning.log
```

Expected: lines containing `Epoch` and loss values (e.g. `train/loss`).
