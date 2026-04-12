# Runbook: Resume Training from Checkpoint

## Quick Start

```bash
cd insight
bash ../resume_training.sh
```

This loads the latest checkpoint from `checkpoints/best.pt` or the most recent `checkpoints/epoch_*.pt`.

## Resume from Specific Checkpoint

```bash
cd insight
python train.py --config conf/config.yaml --resume checkpoints/epoch_042.pt
```

## What Gets Restored

- **Model weights** — fully restored (incompatible keys dropped with warning)
- **Optimizer state** — restored if compatible; fresh optimizer on mismatch
- **LR schedule** — always restarted fresh (warm restart behavior)
- **Epoch counter** — resumed from `checkpoint["epoch"] + 1`
- **Best metrics** — restored from checkpoint for early stopping continuity

## Troubleshooting

### "Could not restore optimizer state"
This is normal when resuming across architecture changes. Training continues with a fresh optimizer. The first epoch may show higher loss as momentum rebuilds.

### "Dropped N incompatible keys"
Expected when resuming after model architecture changes. The incompatible keys are from layers that changed shape or were removed. Check that the dropped keys are from the changed component.

### NaN on Resume
If the checkpoint has NaN weights, the training loop will attempt recovery by loading an earlier clean checkpoint. If no clean checkpoint exists, training stops.

**Recovery**: Delete corrupted checkpoints and restart from `best.pt`:
```bash
rm checkpoints/epoch_*.pt  # Remove corrupted epoch checkpoints
python train.py --resume checkpoints/best.pt
```

## Signal Handling

Sending SIGINT (Ctrl+C) or SIGTERM triggers:
1. Emergency checkpoint saved to `checkpoints/interrupted.pt`
2. Training event logged to `checkpoints/training_events.jsonl`
3. Clean exit with training history saved

Resume normally from the interrupted checkpoint.
