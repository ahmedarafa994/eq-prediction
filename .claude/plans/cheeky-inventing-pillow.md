# Plan: Generate MUSDB18 Precomputed Datasets with Multiple EQ Configs

## Context

The training pipeline expects precomputed `.pt` files (lists of dicts with `wet_mel`, `gain`, `freq`, `q`, `filter_type` tensors). The existing `dataset_musdb_10k.pt` was generated from MUSDB18 but applies only one EQ configuration per audio segment. We need a new script that:

1. Loads each MUSDB18 mixture track (65 tracks: 32 train + 33 test)
2. Chops each track into random segments
3. Applies **multiple different random EQ configurations** to each segment
4. Computes mel-spectrograms (matching the precompute format: 128 mels, shape `[128, 259]`)
5. Saves the result as a `.pt` file directly loadable by `SyntheticEQDataset.load_precomputed()`

## Approach

Create a new script `insight/generate_musdb_dataset.py` that:

1. **Discovers** all 65 `mixture.wav` files under `data/musdb18_hq/`
2. **Iterates** over target sample count (configurable, default ~10k-20k)
3. For each sample:
   - Picks a random track, random start position
   - Crops to `duration * sample_rate` samples (1.5s matching training)
   - Generates **N random EQ configs** (configurable, e.g. 3-5 per segment) using the same parameter distribution as `dataset.py:_sample_multitype_params()`
   - Applies each EQ in frequency domain via `DifferentiableBiquadCascade.apply_to_spectrum()`
   - Computes mel-spectrogram using the same manual mel filterbank as `dataset.py:_audio_to_mel()`
   - Stores each (wet_mel, params) as a separate sample in the cache list
4. **Saves** to `data/dataset_musdb_multieq.pt` via `torch.save()`

Key design: Reuse the EQ parameter sampling and frequency-domain application from `dataset.py` (via `DifferentiableBiquadCascade` and `_sample_multitype_params`), but load real audio from MUSDB18 instead of generating synthetic signals.

## Files to Create

- **`insight/generate_musdb_dataset.py`** — New standalone script

## Files Referenced (read-only)

- `insight/dataset.py` — Reuse: `_sample_multitype_params()`, `_apply_eq_freq_domain()`, `_audio_to_mel()`, mel filterbank construction
- `insight/differentiable_eq.py` — `DifferentiableBiquadCascade` for freq-domain EQ application
- `insight/train.py` — Dataset instantiation pattern to match precompute format
- `insight/data/musdb18_hq/` — 65 mixture.wav tracks

## Script Design

```
generate_musdb_dataset.py
  --input_dir    data/musdb18_hq       (default)
  --output       data/dataset_musdb_multieq.pt  (default)
  --num_samples  50000                 (total samples to generate)
  --configs_per_segment  5             (EQ configs per audio segment)
  --duration     1.5                   (seconds, matching training)
  --sample_rate  44100
  --n_fft        2048
  --n_mels       128
  --num_bands    5
  --num_workers  4                     (multiprocessing)
```

Each audio segment yields `configs_per_segment` samples (same dry, different wet), so we need `num_samples / configs_per_segment` unique audio segments. For 50k samples with 5 configs each = 10,000 segments from 65 tracks (~154 segments per track).

The script will:
1. Build mel filterbank (same as `dataset.py:_build_mel_filterbank()`)
2. Instantiate `DifferentiableBiquadCascade` for freq-domain EQ
3. Load and chunk all MUSDB18 tracks into memory
4. For each segment: sample N EQ configs, apply, compute mel, store
5. `torch.save()` the cache list

## Output Format

Identical to existing precomputed datasets — a list of dicts:
```python
{
    "wet_mel": Tensor[128, 259],      # log-mel spectrogram of wet audio
    "gain":    Tensor[5],             # gain per band
    "freq":    Tensor[5],             # frequency per band
    "q":       Tensor[5],             # Q per band
    "filter_type": Tensor[5],         # type index per band
}
```

This is directly loadable by `SyntheticEQDataset.load_precomputed()`.

## Verification

1. Run the script: `cd insight && python generate_musdb_dataset.py`
2. Verify output: `python -c "import torch; ds=torch.load('data/dataset_musdb_multieq.pt', weights_only=False); print(len(ds), ds[0].keys(), {k: v.shape for k,v in ds[0].items() if hasattr(v,'shape')})"`
3. Test training loads it: update `conf/config.yaml` `precompute_cache_path` to `data/dataset_musdb_multieq.pt` and run one training step
