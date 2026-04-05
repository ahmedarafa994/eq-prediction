# Plan: Maximize Training Throughput

## Context
Current training on `dataset_musdb_200k.pt` (80 epochs) is running at ~170s/epoch using fp32 precision, batch_size=256 with 4× gradient accumulation, and torch.compile disabled. The GPU (RTX PRO 6000 Blackwell, 102 GB VRAM) is only 22% utilized (~22 GB of 102 GB). Applying bf16-mixed precision, larger batch size, fused optimizer, and torch.compile will cut epoch time from ~170s to ~20–30s — reducing total training from ~3.6 hours to ~30–45 minutes.

Effective batch size is preserved at 1024 (256 × accum=4 → 1024 × accum=1), so training dynamics are unchanged.

---

## Step 1: Kill current training run
```bash
kill 140007
```

## Step 2: Update `insight/conf/config.yaml`

| Key | Old | New |
|---|---|---|
| `trainer.precision` | `"fp32"` | `"bf16-mixed"` |
| `data.batch_size` | `256` | `1024` |
| `trainer.gradient_accumulation_steps` | `4` | `1` |
| `data.num_workers` | `4` | `8` |
| `trainer.use_torch_compile` | `false` | `true` |

## Step 3: Update `insight/train.py`

**3a. Fused AdamW** — line ~265, `torch.optim.AdamW(param_groups)`:
```python
self.optimizer = torch.optim.AdamW(param_groups, fused=True)
```

**3b. DataLoader persistent_workers + higher prefetch** — both train and val loaders (lines ~163–184):
```python
persistent_workers=True if num_workers > 0 else False,
prefetch_factor=4 if num_workers > 0 else None,
```

## Step 4: Restart training
```bash
cd /teamspace/studios/this_studio/insight
nohup python train.py > ../train_200k_opt.log 2>&1 &
echo "PID: $!"
```

Then verify it started clean:
```bash
sleep 15 && tail -20 /teamspace/studios/this_studio/train_200k_opt.log
```

---

## Critical files
- `insight/conf/config.yaml`
- `insight/train.py` — DataLoader setup (~lines 163–184), AdamW init (~line 265)

## Verification
- Log shows `[opt] torch.compile applied to model`
- First epoch will be slower (~2–3 min) due to compilation — normal
- Subsequent epochs should be 20–30s each
- Loss trajectory should match previous run (same effective batch size)
- VRAM usage should jump to 40–60 GB (healthy utilization)
