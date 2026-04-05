# Fix Poor Model Performance (Gain 10dB MAE, Type 42%, Freq 2.3 oct)

## Context

The trained IDSP EQ estimator (epoch 36, 50K MUSDB18 samples) performs poorly on ground-truth tests:
- **Gain MAE: 10.13 dB** (val: 4.5 dB — the gap is huge)
- **Freq MAE: 2.28 octaves** (val: 2.0 oct)
- **Type accuracy: 42.4%** (val: 58%, random=20%)
- **Spectral MAE: 9.32 dB**

After deep code analysis, I identified **6 root causes** ordered by impact.

---

## Root Cause 1: Test Script Uses No Hungarian Matching (BIGGEST GAP DRIVER)

**File:** `test_checkpoint.py` lines 127-142

The test script compares bands 1:1 (pred band 0 vs GT band 0) while training uses Hungarian matching to find optimal band assignments. Since the model has no notion of "band 0 = low shelf, band 2 = peak", it freely assigns bands in any order. With 5 bands, there are 120 permutations — the test picks the worst one.

**Evidence:** Val gain MAE (with Hungarian) = 4.5 dB. Test gain MAE (no Hungarian) = 10 dB. The 2x gap is almost entirely permutation mismatch.

**Fix:** Add Hungarian matching to `test_checkpoint.py` and `test_checkpoint_multi.py`, reusing `HungarianBandMatcher` from `loss_multitype.py`.

---

## Root Cause 2: hop_length Mismatch Between Train and Test

**Files:**
- `dataset.py` line 61: `self.hop_length = n_fft // 4` = **512**
- `test_checkpoint.py` line 25: `HOP_LENGTH = 512`
- `train.py` uses precomputed mel from dataset (hop=512)
- **But** `dsp_frontend.py` `STFTFrontend` uses config `hop_length: 256`

The precomputed training data uses hop=512, the test script also uses 512, so train/test are consistent with each other. However, the `STFTFrontend` (used as fallback) uses 256. This is not the primary issue since precomputed mels are used.

**Fix:** No code change needed — already consistent. Add an assertion to verify.

---

## Root Cause 3: Gain Head Has 0.7/0.3 Blend Diluting MLP Output

**File:** `differentiable_eq.py` line 758-776

```python
gain_db = self.gain_mlp(trunk_out).squeeze(-1) * self.gain_output_scale  # bounded [-24,24]
# ... then ...
gain_db = 0.7 * gain_db + 0.3 * gain_aux  # gain_aux is unbounded!
```

The `gain_mel_aux` path is a 2-layer MLP ending in **linear** (no Tanh), so it's unbounded. The 30% blend from an unbounded auxiliary path injects noise into the gain prediction. The primary MLP (with Tanh) has learned bounded gains, but the aux can push predictions far outside the valid range.

**Fix:** Replace the fixed 0.7/0.3 blend with a learned scalar gate (sigmoid-bounded), and add Tanh to `gain_mel_aux` output.

---

## Root Cause 4: HP/LP Gain Near-Zero Confuses the Model

**File:** `dataset.py` lines 186-189, `dataset_musdb.py` same pattern

```python
elif ftype == FILTER_HIGHPASS:
    g = random.uniform(-1.0, 1.0)  # near-zero gain
```

HP/LP filters have gain ~0 (they're gainless by nature), but the model still predicts a full gain value for every band. During training, the model sees ~20% of bands with near-zero gain and must learn "if type is HP/LP, output ~0 dB gain." This creates an implicit dependency: gain learning depends on correct type prediction, but type accuracy is only 42%.

**Fix:** Two options:
- **(A) Quick:** Set HP/LP gain to exactly 0.0 in training data (already nearly there)
- **(B) Better:** Add a `band_activity` gate that predicts whether each band is gainless (HP/LP) and masks the gain loss accordingly

---

## Root Cause 5: Type Classification Head Too Shallow + Imbalanced Data

**File:** `differentiable_eq.py` lines 603-609

The type classification head is just 2 linear layers (input → 64 → 5). With 5 types and complex spectral shapes to distinguish (peaking peak vs lowshelf tilt vs highpass rolloff), this is too shallow. Additionally, the data has 50% peaking, so the model has a strong prior toward "peaking" — which explains why most test predictions are "peaking."

**Evidence:** In the single-song test, **all 5 predicted bands were "peaking"** while GT had highpass, lowshelf, peaking×2, highshelf.

**Fix:** Deepen the type head to 3 layers (input → 128 → 64 → 5) and balance the type weights more aggressively toward underrepresented types during training.

---

## Root Cause 6: Spread Loss Pushes Frequencies Apart Artificially

**File:** `loss_multitype.py` lines 450-463

```python
loss_spread = -spread  # Maximize spread
```

This repulsive force pushes predicted frequencies apart. With 5 bands, it encourages the model to spread predictions across the spectrum even when GT bands cluster (e.g., two bands near 1kHz). This directly hurts frequency MAE.

**Fix:** Reduce `lambda_spread` from 0.02 to 0.0, or cap the repulsive force when bands are already >1 octave apart.

---

## Implementation Plan

### Step 1: Fix test evaluation (Hungarian matching)
**Files:** `test_checkpoint.py`, `test_checkpoint_multi.py`
- Import `HungarianBandMatcher` from `loss_multitype.py`
- Apply matching before computing per-band errors
- This alone should halve the reported gain MAE

### Step 2: Fix gain head blend
**File:** `differentiable_eq.py` (lines 556-581, 755-778)
- Add `self.gain_blend_gate = nn.Parameter(torch.tensor(0.7))` (learnable)
- Add Tanh to `gain_mel_aux` final layer
- Replace fixed 0.7/0.3 with `sigmoid(gain_blend_gate)` blend

### Step 3: Deepen type classification head
**File:** `differentiable_eq.py` (lines 603-609)
- Change `classification_head` from 2 layers to 3: `input → 128 → 64 → 5`
- Add proper init

### Step 4: Reduce spread loss
**File:** `conf/config_musdb_50k.yaml`
- Set `lambda_spread: 0.0`
- Increase `lambda_type: 5.0` (was 3.0)

### Step 5: Retrain and evaluate
- Run `python train.py --config conf/config_musdb_50k.yaml`
- Run `python test_checkpoint.py --checkpoint checkpoints/best.pt`
- Run `python test_checkpoint_multi.py`
- Compare metrics

---

## Verification

1. **After Step 1 (test fix only):** Gain MAE should drop from ~10 dB to ~5 dB (matching val) just from Hungarian matching
2. **After full fix + retrain:** Target metrics:
   - Gain MAE: < 4 dB
   - Freq MAE: < 1.5 oct
   - Type accuracy: > 65%
   - Spectral MAE: < 4 dB
3. Run existing tests: `python test_eq.py`, `python test_model.py`, `python test_multitype_eq.py`
