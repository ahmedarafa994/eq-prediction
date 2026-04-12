# Runbook: Debug NaN Loss

## Symptoms

- `[nan] Non-finite loss at train step N: components=['type_loss', ...]`
- `[nan] NaN in model state after epoch N. Attempting checkpoint recovery...`
- Loss values suddenly spike to `inf` or `nan`

## Step 1: Identify the Component

Check the training log for the NaN diagnostic:
```
[nan] Non-finite loss at train step 1234: components=['typed_spectral_loss', 'hdb_loss']
```

The `components` list identifies which sub-losses produced NaN.

### Common Culprits

| Component | Likely Cause | Fix |
|-----------|-------------|-----|
| `type_loss` | Gumbel tau too low → one-hot collapse | Increase `gumbel.min_tau` to 0.3+ |
| `spectral_loss` | Extreme Q values → numerical instability | Reduce `q_bounds` upper limit to 10 |
| `typed_spectral_loss` | Wrong type assigned → type-conditional response mismatch | Increase `lambda_type` in curriculum |
| `hdb_loss` | Per-band H_db prediction divergence | Check `lambda_hdb` — reduce if too high |
| `embed_var_loss` | Embedding collapse (all outputs identical) | Check encoder — may need pretraining |
| `contrastive_loss` | All embeddings identical | Early training — normal, resolves with warmup |

## Step 2: Check Gradient Norms

Look for gradient explosion in logs:
```
[grads] step=1234 grad_gain=5.2341 | grad_type=0.0012 | grad_encoder=45.1230
```

- `grad_encoder > 10`: Likely cause of NaN. Check learning rate — reduce by 10×.
- `grad_gain >> 5`: Gain head may be overfitting. Reduce `lambda_gain`.

## Step 3: Recovery

The training loop automatically recovers from the last clean checkpoint:
```
[recovery] Reloaded clean checkpoint from checkpoints/epoch_038.pt, reset optimizer
```

If no clean checkpoint is found:
```
[recovery] No clean checkpoint found. Cannot recover.
ERROR: Cannot recover. Stopping.
```

### Manual Recovery

1. Find the last epoch with finite val_loss:
   ```bash
   grep "val_loss" checkpoints/training_events.jsonl | tail -5
   ```

2. Resume from that epoch's checkpoint:
   ```bash
   python train.py --resume checkpoints/epoch_NNN.pt
   ```

## Step 4: Prevention

| Setting | Recommended | Why |
|---------|------------|-----|
| `trainer.gradient_clip_val` | 1.0 | Prevents gradient explosion |
| `gumbel.min_tau` | 0.3+ | Prevents type classifier collapse |
| `data.q_bounds` | `[0.05, 10.0]` | Extreme Q causes numerical instability |
| `loss.lambda_type` | 3.0+ (early) | Ensures type classifier learns before spectral |
| `curriculum.stages[0].lambda_type` | 5.0 | Type pretrain in first stage |

## Diagnostic Tools

### Check Model Weight Health
```python
import torch
state = torch.load("checkpoints/best.pt", weights_only=False)
for k, v in state["model_state_dict"].items():
    if not torch.isfinite(v).all():
        print(f"NaN in {k}")
```

### Check Structured Logs
```bash
# View recent events
tail -20 checkpoints/training_events.jsonl | python -m json.tool

# View metric timeline
cat checkpoints/structured_log.jsonl | grep '"type":"metric"' | python -m json.tool
```
