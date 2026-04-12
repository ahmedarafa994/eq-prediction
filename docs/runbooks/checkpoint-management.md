# Runbook: Checkpoint Management

## Checkpoint Locations

All checkpoints are saved to `insight/checkpoints/`:

| File | Description |
|------|-------------|
| `best.pt` | Best overall checkpoint by `primary_val_score` |
| `best_primary.pt` | Best by composite primary validation score |
| `best_gain.pt` | Best gain MAE (Hungarian-matched) |
| `best_type.pt` | Best type classification accuracy |
| `best_audio.pt` | Best spectral loss |
| `epoch_NNN.pt` | Per-epoch checkpoints (pruned after 3 most recent) |
| `interrupted.pt` | Emergency checkpoint from signal handler |

## Checkpoint Pruning

The trainer automatically prunes old checkpoints, keeping:
- Last 3 `epoch_NNN.pt` files
- All named checkpoints (`best*.pt`)

## Manual Cleanup

To free disk space:
```bash
cd insight/checkpoints
# Keep only best checkpoints
rm epoch_*.pt
ls -lh best*.pt
```

## Inspecting a Checkpoint

```python
import torch
state = torch.load("checkpoints/best.pt", weights_only=False)
print(f"Epoch: {state['epoch']}")
print(f"Val loss: {state['val_loss']:.4f}")
print(f"Gain MAE: {state.get('gain_mae_db_matched', 'N/A')}")
print(f"Type acc: {state.get('type_accuracy_matched', 'N/A')}")
print(f"Global step: {state['global_step']}")

# Check for NaN weights
for k, v in state["model_state_dict"].items():
    if not torch.isfinite(v).all():
        print(f"  WARNING: NaN in {k}")
```

## Exporting for Inference

To export a clean model for deployment:
```bash
cd insight
python export.py --checkpoint checkpoints/best.pt --output model.onnx
```
