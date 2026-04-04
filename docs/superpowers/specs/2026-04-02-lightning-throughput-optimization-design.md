# Lightning Throughput Optimization Design

**Date:** 2026-04-02
**Goal:** Migrate default training from custom `train.py` to the Lightning-based trainer (`training/train.py`) and apply throughput-maximizing optimizations for the NVIDIA RTX PRO 6000 Blackwell GPU (98 GB VRAM).

---

## Context

The project has two training paths:
- **Custom trainer** (`insight/train.py`) — 1021 lines, currently default, handles curriculum learning manually
- **Lightning trainer** (`insight/training/train.py`) — full Lightning 2.x stack with `CurriculumCallback`, `EQDataModule`, and `EQEstimatorLightning`; not currently the default entry point

The Lightning stack is already feature-complete (curriculum, loss weights, Gumbel temperature, scheduler reset). The migration is primarily about switching the entry point and applying performance config.

Hardware target: NVIDIA RTX PRO 6000 Blackwell, 98 GB VRAM, 48 CPU cores.

---

## Changes

### 1. `conf/config.yaml` — Trainer section

| Key | Before | After | Reason |
|-----|--------|-------|--------|
| `max_epochs` | 30 | 90 | Curriculum totals 15+25+10+30+10=90 |
| `precision` | `16-mixed` | `bf16-mixed` | Blackwell-native BF16, no GradScaler needed |
| `accelerator` | `auto` | `gpu` | Explicit, avoids auto-detection overhead |
| `check_val_every_n_epoch` | _(absent)_ | 5 | Validation every epoch is wasteful at 1024 batch size |

### 2. `training/train.py` — Script-level additions

Before `trainer.fit()`:

```python
torch.set_float32_matmul_precision("high")  # Enable TF32 on Blackwell
model = torch.compile(model, mode="reduce-overhead")  # JIT compile
```

In the `Trainer(...)` call:
- Change `deterministic="warn"` → `deterministic=False` — removes cuDNN benchmark disable
- Add `check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1)` — reads the new config key

### 3. `training/data_module.py` — DataLoader tuning

In `_dataloader()`:
- `pin_memory=False` → `pin_memory=True` — faster CPU→GPU transfer for precomputed tensors
- `prefetch_factor=2` → `prefetch_factor=4` — keeps GPU fed between batches

### 4. `training/lightning_module.py` — zero_grad override

Add to `EQEstimatorLightning`:

```python
def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
    optimizer.zero_grad(set_to_none=True)
```

Deallocates gradient tensors entirely instead of zeroing — modest but free speedup.

---

## What is NOT changing

- Curriculum learning logic — `CurriculumCallback` and `EQDataModule.setup_for_stage()` are already correct
- Loss functions, model architecture, DSP layer — untouched
- `num_workers: 4` — already set; in-memory precomputed dataset doesn't benefit from more
- FP8 / TransformerEngine — deferred; DSP layer ops (`torch.where`, biquad coefficient math) risk graph breaks

---

## Entry point after migration

```bash
cd insight/training
python train.py --config ../conf/config.yaml
```

The custom `train.py` remains in place as a fallback but is no longer the default.

---

## Expected impact

| Optimization | Expected gain |
|---|---|
| BF16-mixed | ~1.5× (removes FP16 GradScaler overhead, better tensor core utilization) |
| TF32 matmuls | ~1.3× on float32 ops |
| torch.compile reduce-overhead | ~1.2× after warmup |
| pin_memory + prefetch_factor=4 | ~5–10% (reduces DataLoader stall) |
| zero_grad(set_to_none=True) | ~1–2% |
| check_val_every_n_epoch=5 | Reduces total wall time by ~15% (validation is non-trivial with Hungarian matching) |
