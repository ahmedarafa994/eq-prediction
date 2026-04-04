# Lightning Best-Practices Audit: Remaining Gaps

## Context

The Lightning trainer (`insight/training/`) was already migrated and optimized (bf16-mixed, torch.compile, pin_memory=True, prefetch_factor=4, optimizer_zero_grad set_to_none, check_val_every_n_epoch=5). An audit of `https://lightning.ai/docs/` against the current codebase found three remaining gaps that materially affect training speed and stability:

1. **CRITICAL — `training_step` disables bf16 for the encoder** (`lightning_module.py:133`): The entire forward pass (including the TCN encoder) is wrapped in `autocast("cuda", enabled=False)`, forcing fp32. The encoder should run in bf16 under AMP; only the DSP biquad + loss computation needs fp32. This negates the bf16-mixed precision setting.
2. **Missing SWA callback** (`train.py`): StochasticWeightAveraging is a free generalization win supported natively by Lightning.
3. **Missing `num_sanity_val_steps`** (`config.yaml` + `train.py`): Default is 2 sanity batches; setting to 1 saves one forward pass at startup.

---

## Task 1: Fix `training_step` autocast scope — `insight/training/lightning_module.py`

**File:** `insight/training/lightning_module.py`

**Problem:** Lines 133–162: `autocast("cuda", enabled=False)` wraps the `model.encoder()` call, forcing the TCN encoder to compute in fp32. The model's own `forward()` (in `model_tcn.py`) correctly scopes the `autocast(enabled=False)` to only `_predict_from_embedding` — the Lightning `training_step` should match this pattern.

**Change:** Replace lines 131–163 in `training_step`:

Current (lines 131–162):
```python
        mel_profile = mel_frames.mean(dim=-1)

        with torch.amp.autocast("cuda", enabled=False):
            mel_frames_f = mel_frames.float()
            mel_profile_f = mel_profile.float()

            embedding, skip_sum = self.model.encoder(mel_frames_f)
            output = self.model._predict_from_embedding(
                embedding, mel_profile=mel_profile_f
            )

            target_H_mag = self.model.dsp_cascade(
                target_gain.float(),
                target_freq.float(),
                target_q.float(),
                n_fft=self.n_fft,
                filter_type=target_ft,
            ).clamp(min=1e-4, max=1e4)

            total_loss, components = self.criterion(
                output["params"][0],
                output["params"][1],
                output["params"][2],
                output["type_logits"].float(),
                output["H_mag"].float(),
                target_gain,
                target_freq,
                target_q,
                target_ft,
                target_H_mag,
                freq_attn=output.get("freq_attn"),
            )
```

Replace with:
```python
        mel_profile = mel_frames.mean(dim=-1)

        # Encoder runs in bf16 under AMP — matches model_tcn.forward() pattern
        embedding, skip_sum = self.model.encoder(mel_frames)

        # DSP biquad coefficients and loss require fp32 precision
        with torch.amp.autocast("cuda", enabled=False):
            output = self.model._predict_from_embedding(
                embedding.float(), mel_profile=mel_profile.float()
            )

            target_H_mag = self.model.dsp_cascade(
                target_gain.float(),
                target_freq.float(),
                target_q.float(),
                n_fft=self.n_fft,
                filter_type=target_ft,
            ).clamp(min=1e-4, max=1e4)

            total_loss, components = self.criterion(
                output["params"][0],
                output["params"][1],
                output["params"][2],
                output["type_logits"].float(),
                output["H_mag"].float(),
                target_gain,
                target_freq,
                target_q,
                target_ft,
                target_H_mag,
                freq_attn=output.get("freq_attn"),
            )
```

---

## Task 2: Add `StochasticWeightAveraging` — `insight/training/train.py`

**File:** `insight/training/train.py`

**Change 1:** Add `StochasticWeightAveraging` to the import block (lines 17–22):

Current:
```python
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
```

Replace with:
```python
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    StochasticWeightAveraging,
)
```

**Change 2:** Add SWA to the callbacks list (after `RichProgressBar()`):

Current (lines 81–86):
```python
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        lr_monitor,
        RichProgressBar(),
    ]
```

Replace with:
```python
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        lr_monitor,
        RichProgressBar(),
        StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=82, annealing_epochs=8),
    ]
```

Rationale: SWA averages weights over the final 8 epochs (82–90) after curriculum completes. `swa_lrs=1e-5` is one order below the base lr (1e-4).

---

## Task 3: Add `num_sanity_val_steps` — `insight/conf/config.yaml` + `insight/training/train.py`

**File 1:** `insight/conf/config.yaml` — add to `trainer:` block:

Current trainer block ends at line 66 (`fast_dev_run: false`). Add one line after `fast_dev_run`:
```yaml
  num_sanity_val_steps: 1
```

**File 2:** `insight/training/train.py` — add to `L.Trainer(...)` call (after the `fast_dev_run` line, currently line 105):

Current:
```python
        fast_dev_run=trainer_cfg.get("fast_dev_run", False),
    )
```

Replace with:
```python
        fast_dev_run=trainer_cfg.get("fast_dev_run", False),
        num_sanity_val_steps=trainer_cfg.get("num_sanity_val_steps", 2),
    )
```

---

## Verification

After applying all three changes:

1. **Smoke test** (CPU, no GPU needed):
   ```bash
   cd /teamspace/studios/this_studio/insight && python test_lightning_dummy.py
   ```
   Expected: `Lightning Verification Passed`

2. **BF16 check** — confirm encoder runs in bf16 after the autocast fix:
   ```bash
   cd /teamspace/studios/this_studio/insight/training
   python -c "
   import yaml, torch
   from lightning_module import EQEstimatorLightning
   cfg = yaml.safe_load(open('../conf/config.yaml'))
   m = EQEstimatorLightning(cfg).cuda().bfloat16()
   x = torch.randn(2, 256, 128, device='cuda', dtype=torch.bfloat16)
   emb, _ = m.model.encoder(x)
   print('encoder output dtype:', emb.dtype)  # expected: torch.bfloat16
   "
   ```

3. **Kill old training, clear cache, restart**:
   ```bash
   pkill -f "python train.py"; rm -f /teamspace/studios/this_studio/insight/data_cache/*.npz
   cd /teamspace/studios/this_studio/insight/training
   nohup python train.py --config ../conf/config.yaml > /tmp/train_lightning.log 2>&1 &
   echo "PID: $!"
   ```

4. **Confirm first epoch** after ~60s:
   ```bash
   tail -30 /tmp/train_lightning.log
   ```
   Expected: lines with `Epoch`, `train/loss` values.
