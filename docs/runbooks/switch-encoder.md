# Runbook: Switch Encoder Backend

## Available Backends

| Backend | License | Description | Memory |
|---------|---------|-------------|--------|
| `hybrid_tcn` | Custom | Default 2D spectral + grouped TCN | ~4 GB |
| `wav2vec2_frozen` | Apache 2.0 | Frozen Wav2Vec2 + trainable projection | ~6 GB |
| `wav2vec2` (unfrozen) | Apache 2.0 | Fine-tuned Wav2Vec2 | ~8 GB + gradient checkpointing |
| `ast` | Apache 2.0 | Audio Spectrogram Transformer (ViT) | ~8 GB |
| `mert` | CC-BY-NC | MERT music encoder | ~6 GB |
| `clap` | MIT | CLAP audio encoder | ~5 GB |

## How to Switch

### 1. Update Config

Edit `conf/config.yaml` (or create a new config):

```yaml
model:
  encoder:
    backend: wav2vec2_frozen  # Change this
    wav2vec2_checkpoint: facebook/wav2vec2-base
    n_mels: 128
    embedding_dim: 256
    channels: 256
    num_blocks: 8
    num_stacks: 3
    dropout: 0.2
    mel_noise_std: 0.0
```

### 2. Update Data Config (if needed)

Some backends require raw audio (not precomputed mel):

```yaml
data:
  precompute_mels: false  # Required for wav2vec2_frozen, ast, mert, clap
  audio_duration: 2.0     # Some backends expect minimum duration
```

### 3. Start Training

```bash
cd insight
python train.py --config conf/config.yaml
```

## Wav2Vec2 Unfreezing (Phase 6)

For the backbone unfreezing experiment, use the dedicated config:

```bash
bash launch_wav2vec2.sh  # Background training with config_wav2vec2_unfreeze.yaml
```

Key settings in `conf/config_wav2vec2_unfreeze.yaml`:
- `gradient_checkpointing: true` — Required to fit unfrozen backbone in VRAM
- `backbone_lr: 4.0e-06` — 5× lower than encoder LR to prevent catastrophic forgetting
- `freeze_epochs: 42` — Backbone frozen for first 42 epochs, then unfrozen
- `batch_size: 512, gradient_accumulation_steps: 2` — Effective batch of 1024

## Troubleshooting

### OOM with Unfrozen Backbone
- Enable `use_gradient_checkpointing: true`
- Reduce `batch_size` to 256
- Increase `gradient_accumulation_steps` to maintain effective batch size

### "X requires transformers but it is not installed"
```bash
pip install transformers timm
```

### "X requires raw wet_audio; precompute_mels must be false"
All non-TCN backends need raw audio. Set `data.precompute_mels: false`.
