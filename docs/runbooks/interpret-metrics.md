# Runbook: Interpret Training Metrics

## Key Metrics to Watch

### Primary Metrics

| Metric | Target | Good | Concerning |
|--------|--------|------|------------|
| `gain_mae_db_matched` | < 1.0 dB | < 2.0 dB | > 4.0 dB |
| `type_accuracy_matched` | > 95% | > 85% | < 70% |
| `freq_mae_oct_matched` | < 0.25 oct | < 0.5 oct | > 1.0 oct |
| `q_mae_dec_matched` | < 0.2 dec | < 0.4 dec | > 0.8 dec |
| `primary_val_score` | < 2.0 | < 4.0 | > 8.0 |

### Secondary Metrics

| Metric | What It Tells You |
|--------|-------------------|
| `val_spectral_loss` | How well the model matches the target frequency response |
| `val_loss_soft` | Loss with soft (Gumbel-Softmax) type selection |
| `val_loss_hard` | Loss with hard (argmax) type selection |
| `gumbel_tau` | Current Gumbel temperature — should anneal from start to min |

### Per-Type Accuracy

Look at `type_accuracy_{type}_matched` for each filter type:
- `peaking`: Usually learns first (most represented in training)
- `lowshelf` / `highshelf`: Should reach >80% accuracy
- `highpass` / `lowpass`: May lag if type weights are imbalanced

### Gradient Norms

Logged as `grad_norm/{component}` in structured logs:
- `grad_encoder`: Should be 0.1–5.0
- `grad_gain`, `grad_freq`, `grad_q`: Parameter head gradients, 0.01–2.0
- `grad_type`: Type classifier gradient, 0.1–3.0
- `grad_backbone`: Only when unfrozen, should be < 1.0

## Where to Find Metrics

### Console Output
```
Epoch 42/60 (12.3s) train=2.3451 primary_val_score=3.2100 val_spectral_loss=1.2345
```

### Structured Logs (JSONL)
```bash
cat checkpoints/structured_log.jsonl | grep '"name":"epoch/gain_mae_db_matched"'
```

### Training History
```bash
cat checkpoints/training_history.json | python -m json.tool
```

### WandB Dashboard
If `logging.enable_wandb: true`, all metrics stream to `wandb.ai/your-project`.

## Interpreting Curves

### Good Training Run
- `train_loss`: Decreasing smoothly
- `val_loss`: Decreasing, tracking train_loss within 2×
- `gain_mae_db_matched`: Decreasing toward target
- `type_accuracy_matched`: Increasing, all types above 80%
- `gumbel_tau`: Annealing from start (2.0) toward min (0.1–0.3)

### Encoder Collapse (catastrophic failure)
- `train_loss`: Flat or NaN
- `val_loss`: Flat or NaN
- `gain_mae_db_matched`: > 20 dB
- Type accuracy: ~20% (random for 5 types)
- All embeddings identical (cosine similarity = 1.0)

### Loss Competition
- `type_accuracy_matched` stuck at ~40%
- `gain_mae_db_matched` decreasing slowly
- `spectral_loss` dominating other components
- **Fix**: Increase `lambda_type`, reduce `lambda_spectral` in early stages

### Plateau
- All metrics flat for 10+ epochs
- `patience_counter` increasing toward `early_stopping_patience`
- **Fix**: Try LR restart (resume with fresh optimizer), or unfreeze backbone
