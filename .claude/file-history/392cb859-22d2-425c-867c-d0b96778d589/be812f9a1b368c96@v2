# Fix Training Plateau: Frequency & Q Stagnation

## Context

Training on the `easy_peaking` curriculum stage plateaued at epoch 8. The model learned band presence (F1=1.0) and type (acc=1.0) quickly, but continuous parameter estimation stalled: frequency error ~1.81 octaves (flat across all 12 epochs), gain ~5.2 dB, Q ~0.29. Val loss flat at ~22.2.

**Root cause analysis** reveals three structural issues, not just hyperparameter tuning:

1. **Attention-direct blend starts at 0.88** (`freq_blend_weight=2.0` → `sigmoid(2)=0.88`). When attention hasn't learned to localize (entropy high early in training), the direct regression path is attenuated to 12%, starving frequency learning of gradient signal.

2. **No frequency curriculum narrowing.** The `easy_peaking` stage limits gain and Q ranges but uses the full 20Hz–20kHz range. The model must learn to localize across 10 octaves from the start.

3. **`lambda_freq_internal=3.0` is hardcoded** in `PermutationInvariantParamLoss.__init__` and cannot be overridden by curriculum stages. Combined with `lambda_freq_anchor=3.0` (from stage override), frequency has ~6× total weight — but this hasn't helped because the bottleneck is gradient flow through the attention mechanism, not loss magnitude.

## Changes

### 1. Equalize attention-direct blend at initialization
**File:** `insight/differentiable_eq.py` line 760
**Change:** `self.freq_blend_weight = nn.Parameter(torch.tensor(2.0))` → `torch.tensor(0.0)`
**Effect:** `sigmoid(0) = 0.5` — both paths contribute equally. Direct regression provides strong gradient even when attention is diffuse. As training progresses, the model can learn to favor whichever path works better.

### 2. Add frequency range to curriculum stages
**File:** `insight/conf/config.yaml`
**Change:** Add `freq_range` to each stage:
- `easy_peaking`: `[100, 8000]` — narrower, only 6.6 octaves
- `medium_peaking`: `[60, 12000]` — expanding
- `frequency_spread`: `[40, 16000]` — expanding
- `shelf_types` and later: full `[20, 20000]`

**File:** `insight/train.py` (~line 360-390 where dataset is rebuilt per stage)
**Change:** Read `freq_range` from stage config and pass to `SyntheticEQDataset` constructor. The dataset already accepts a `freq_range` parameter — just needs to be wired up.

### 3. Make `lambda_freq_internal` configurable via curriculum
**File:** `insight/loss_multitype.py` line 298
**Change:** `PermutationInvariantParamLoss.__init__` already accepts `lambda_freq_internal` as a parameter. Wire it through `MultiTypeEQLoss.__init__` and add to curriculum stage overrides.

**File:** `insight/conf/config.yaml`
**Change:** Add to loss section: `lambda_freq_internal: 3.0` (base default). Override in stages:
- `easy_peaking`: `lambda_freq_internal: 1.5` (reduced — frequency already has anchor loss at 3.0)

**File:** `insight/train.py` (~line 420)
**Change:** Add curriculum override for `lambda_freq_internal` (same pattern as existing `lambda_freq_match` override).

### 4. Gentler LR schedule
**File:** `insight/train.py` line 471-474
**Change:**
- `max_lr=base_lr * 3` → `base_lr * 2` (less aggressive peak)
- `pct_start=0.3` → `0.4` (longer exploration before annealing)

## Files to Modify

1. `insight/differentiable_eq.py` — blend weight init (line 760)
2. `insight/conf/config.yaml` — add freq_range to stages, add lambda_freq_internal
3. `insight/train.py` — wire freq_range to dataset, add lambda_freq_internal override, adjust LR schedule
4. `insight/loss_multitype.py` — pass lambda_freq_internal through MultiTypeEQLoss

## Verification

1. Run a short training test (2-3 epochs of easy_peaking):
   ```bash
   cd insight
   # Edit config to set curriculum.easy_peaking.epochs=3 and trainer.max_epochs=3
   python train.py
   ```
2. Check that:
   - Frequency error starts decreasing within the first 3 epochs (previously flat)
   - Val loss is decreasing (not flat)
   - Loss component logs show freq_anchor and param_loss contributions
3. Run existing tests to verify nothing is broken:
   ```bash
   python test_model.py
   python test_multitype_eq.py
   ```
