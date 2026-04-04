# Plan: Robust Model Evaluation and Diagnostic Report

## Context

The user reported poor inference results from `best.pt` (Gain MAE: 5.07 dB, Freq MAE: 2.3 octaves, Type Accuracy: 0%) and suspects the model may have mode collapse or the test signal (white noise) may be mismatched from training data. There is **no existing inference.py** and **no groundtruth audio file** — only synthetic test wavs in `test_output/`.

**Critical finding from training history**: The model itself has not learned effectively. Across 20 recorded epochs, validation metrics barely improved:
- Type accuracy: 49-51% (essentially random for the dominant peaking class)
- Freq MAE: 3.0 octaves (stagnant)
- Gain MAE: 7.0 dB (stagnant)
- Loss plateaued by epoch 10

Training stopped at epoch 37 (medium_multitype stage) — the curriculum has 4 stages totaling 80 epochs, so the model never completed training. The `training_history.json` only has 20 entries (epochs 1-20), meaning the rest may have failed or the file was overwritten.

**Answer to user's question**: There is no groundtruth audio file. We should proceed with synthetic evaluation.

---

## Step 1: Create a comprehensive evaluation script (`insight/evaluate_model.py`)

This script will:

1. **Load the checkpoint** and print metadata (epoch, stage, val_loss, global_step)
2. **Generate structured test signals** using the same `SyntheticEQDataset` infrastructure:
   - 50 samples each of: `noise`, `sweep`, `harmonic`, `speech_like`
   - All with known ground-truth EQ parameters (gain, freq, Q, filter_type)
3. **Run batch inference** using the model's `forward()` method with precomputed mel-spectrograms (same path as training)
4. **Run streaming inference** using `init_streaming()` + `process_frame()` for one sample
5. **Report per-signal-type metrics**:
   - Gain MAE (dB)
   - Freq MAE (octaves)
   - Q MAE (decades)
   - Filter type accuracy (%)
   - Hungarian-matched parameter errors (since band ordering is arbitrary)
6. **Print sample predictions vs ground truth** for visual inspection
7. **Compare batch vs streaming consistency**

### Key files to reuse:
- `dataset.py:SyntheticEQDataset` — generate test samples with known params
- `dataset.py:_generate_dry_signal()` — signal generation for all 4 types
- `dataset.py:_sample_multitype_params()` — ground truth EQ params
- `dataset.py:_apply_eq_freq_domain()` — apply EQ to create wet signals
- `dataset.py:_audio_to_mel()` — convert to mel-spectrogram
- `model_tcn.py:StreamingTCNModel.forward()` — batch inference
- `model_tcn.py:StreamingTCNModel.process_frame()` — streaming inference
- `loss_multitype.py:HungarianBandMatcher` — for permutation-invariant matching
- `dsp_frontend.py:STFTFrontend` — mel-spectrogram computation
- `conf/config.yaml` — model architecture params

### Why structured signals matter:
The dataset uses 4 signal types with different spectral characteristics:
- `noise`: flat spectrum — hardest for EQ estimation (no structure to exploit)
- `harmonic`: strong tonal content — EQ creates clearly audible changes at harmonic frequencies
- `sweep`: chirp signal — covers full frequency range sequentially
- `speech_like`: bursty, naturalistic — closest to real-world use

## Step 2: Run the evaluation and produce a diagnostic report

Execute the script and produce a structured report covering:
1. Per-signal-type performance breakdown
2. Whether performance varies by filter type (peaking vs shelf vs HP/LP)
3. Whether batch and streaming modes agree
4. Specific failure modes (e.g., frequency collapse to ~500Hz)
5. Comparison of best.pt vs epoch_070.pt (later checkpoint)

## Step 3: Determine root cause and recommend next steps

Based on findings, determine:
- Is the model genuinely collapsed (training issue) or just poorly tested?
- Did training complete properly? (history only has 20/80 epochs)
- Are there obvious fixes (longer training, learning rate, loss weights)?

---

## Verification

1. Run `python evaluate_model.py` — produces metrics table and sample predictions
2. Verify mel-spectrogram shapes match training: `(batch, 128, T)`
3. Compare batch vs streaming: predictions should be within 5% of each other
4. Check that ground-truth params are in expected ranges from config
