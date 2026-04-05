# Plan: Fix Type Accuracy Plateau and Curriculum Stage Regression

## Context

The model plateaus at ~38% type accuracy (below the 50% random-baseline of always predicting
"peaking"). Three root causes were confirmed by code inspection:

1. **Critical bug** — `loss_type` uses unmatched filter types. `MultiTypeEQLoss.forward` runs
   Hungarian matching on gain/freq/Q but then computes `CrossEntropyLoss` against the original
   (unmatched) `target_filter_type`. If Hungarian assigns predicted band 0 → target band 3, the
   param losses correctly use band 3's gain/freq/Q but the type loss incorrectly penalises band
   0's type for predicted band 0. The supervision signal for the type head is systematically
   wrong.

2. **Stage transition shock** — Stage 1 (`peaking_warmup`) forces `type_weights = [1,0,0,0,0]`
   (100% peaking). Stage 2 jumps to `[0.5, 0.15, 0.15, 0.1, 0.1]`. The type head has never seen
   non-peaking examples; 50% of stage-2 data is suddenly foreign, causing a sharp accuracy drop.

3. **Shifting validation set** — `_apply_curriculum_stage` re-precomputes all 30 K samples and
   re-splits train/val at each stage boundary. The val set changes composition, making
   epoch-to-epoch val metrics incomparable and making regressions look larger than they are.

## Files to Modify

- `insight/loss_multitype.py` — Fix 1 (Hungarian type matching + loss_type fix)
- `insight/train.py` — Fix 2 (per-stage type_weights) + Fix 3 (fixed val set)
- `insight/conf/config.yaml` — Fix 2 (stage 1 type_weights key)

---

## Fix 1 — Match filter types through Hungarian before computing `loss_type`

### 1a. `HungarianBandMatcher.__call__` (`loss_multitype.py:87-120`)

Add optional `target_filter_type` parameter and return matched types as a 4th value.

```python
def __call__(self, pred_gain, pred_freq, pred_q,
             target_gain, target_freq, target_q,
             target_filter_type=None):          # NEW
    cost = self.compute_cost_matrix(...)
    assignments = self.match(cost)

    B, N = pred_gain.shape
    device = pred_gain.device
    matched_gain = torch.zeros_like(target_gain)
    matched_freq = torch.zeros_like(target_freq)
    matched_q    = torch.zeros_like(target_q)
    if target_filter_type is not None:          # NEW
        matched_filter_type = torch.zeros_like(target_filter_type)  # NEW

    for b in range(B):
        row_ind, col_ind = assignments[b]
        perm = torch.zeros(N, dtype=torch.long, device=device)
        for r, c in zip(row_ind, col_ind):
            perm[r] = c
        matched_gain[b] = target_gain[b, perm]
        matched_freq[b] = target_freq[b, perm]
        matched_q[b]    = target_q[b, perm]
        if target_filter_type is not None:      # NEW
            matched_filter_type[b] = target_filter_type[b, perm]  # NEW

    if target_filter_type is not None:          # NEW
        return matched_gain, matched_freq, matched_q, matched_filter_type  # NEW
    return matched_gain, matched_freq, matched_q   # unchanged 3-tuple when types not passed
```

Backward-compatible: every existing 3-value call site (no `target_filter_type` argument) still
works unchanged.

### 1b. `MultiTypeEQLoss.forward` (`loss_multitype.py:276-298`)

Replace the `self.param_loss(...)` call + unmatched `loss_type` block with a single matcher
call that returns matched params AND types. Then compute Huber losses inline (same formulas as
`PermutationInvariantParamLoss.forward`). `self.param_loss` is kept on the class — this
just bypasses it in the one place that needs types.

Replace this block:
```python
# lines 276-298 (current)
loss_gain, loss_freq, loss_q = self.param_loss(
    pred_gain, pred_freq, pred_q,
    target_gain, target_freq, target_q
)
components["loss_gain"] = loss_gain
components["loss_freq"] = loss_freq
components["loss_q"]    = loss_q
...
B, N, C = pred_type_logits.shape
loss_type = self.type_loss(
    pred_type_logits.reshape(B * N, C),
    target_filter_type.reshape(B * N)   # ← BUG: unmatched
)
```

With:
```python
# Single matcher call — solves Hungarian once, aligns params AND types
matched_gain, matched_freq, matched_q, matched_filter_type = self.param_loss.matcher(
    pred_gain, pred_freq, pred_q,
    target_gain, target_freq, target_q,
    target_filter_type=target_filter_type,
)
# Huber on matched params  (same delta=5.0 as PermutationInvariantParamLoss)
loss_gain = self.param_loss.huber(pred_gain, matched_gain)
loss_freq = self.param_loss.huber(
    torch.log(pred_freq + 1e-8), torch.log(matched_freq + 1e-8))
loss_q    = self.param_loss.huber(
    torch.log(pred_q   + 1e-8), torch.log(matched_q   + 1e-8))
components["loss_gain"] = loss_gain
components["loss_freq"] = loss_freq
components["loss_q"]    = loss_q

# Type CE using correctly matched targets
B, N, C = pred_type_logits.shape
loss_type = self.type_loss(
    pred_type_logits.reshape(B * N, C),
    matched_filter_type.reshape(B * N),   # ← FIXED: matched types
)
```

Note: `self.param_loss.huber` is the `nn.HuberLoss(delta=5.0)` instance; `self.huber` on
`MultiTypeEQLoss` itself is also `delta=5.0` — either works. Using `self.param_loss.huber`
makes the formula self-documenting.

Also remove the `param_loss` component entry added later (`components["param_loss"] = loss_param`)
— that block will now be stale. Keep the scalar `loss_param` only for the total_loss line if
`lambda_param > 0` (keep existing guard).

### 1c. `Trainer.validate` — fix `type_acc` metric (`train.py:498-512`)

The validation loop already calls `self.matcher(...)` to compute matched gain/freq/Q (line 498).
Extend this call to also get matched types, then compare against those:

```python
# current line 498-499
matched_gain, matched_freq, matched_q = self.matcher(
    pred_gain, pred_freq, pred_q, target_gain, target_freq, target_q
)
# current line 511-512
param_maes["type_acc"].append(
    (output["filter_type"] == target_ft).float().mean().item()  # ← BUG
)
```

Replace with:
```python
matched_gain, matched_freq, matched_q, matched_ft = self.matcher(
    pred_gain, pred_freq, pred_q, target_gain, target_freq, target_q,
    target_filter_type=target_ft,
)
# ...
param_maes["type_acc"].append(
    (output["filter_type"] == matched_ft).float().mean().item()  # ← FIXED
)
```

---

## Fix 2 — Per-stage `type_weights` to smooth the stage-1 → stage-2 transition

### 2a. `_apply_curriculum_stage` (`train.py:610-618`)

Add a check for `type_weights` in the stage dict *before* the `filter_types` derivation:

```python
# NEW: explicit per-stage type_weights override
stage_type_weights = stage.get("type_weights", None)
active_types = stage.get("filter_types", None)
if stage_type_weights is not None:
    self.train_dataset.type_weights = stage_type_weights
elif active_types is not None:
    # existing fallback: uniform weights across listed filter_types
    all_types = ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]
    weights = [0.0] * len(all_types)
    for t in active_types:
        if t in all_types:
            weights[all_types.index(t)] = 1.0 / len(active_types)
    self.train_dataset.type_weights = weights
```

Update the print statement at line 664 to log whichever was used:
```python
type_info = stage_type_weights if stage_type_weights else active_types
print(f"  [curriculum] Epoch {epoch}: stage '{stage['name']}' "
      f"(types={type_info}, gumbel_tau={target_tau}, data_recomputed)")
```

### 2b. `conf/config.yaml` — stage 1 type_weights

Replace stage 1's `filter_types: ["peaking"]` with an explicit `type_weights` key that
introduces non-peaking types from the start (but heavily weights peaking):

```yaml
- name: "peaking_warmup"
  epochs: 10
  type_weights: [0.75, 0.075, 0.075, 0.05, 0.05]  # ~75% peaking, 25% other types
  param_ranges:
    gain: [-12.0, 12.0]
    q: [0.3, 5.0]
  lambda_type: 1.0
  gumbel_temperature: 1.0
```

Remove `filter_types: ["peaking"]` from stage 1 (no longer needed; `type_weights` takes
precedence). Stages 2-4 keep their existing `filter_types` keys (they have no `type_weights`
key, so the existing logic runs unchanged).

---

## Fix 3 — Fixed validation set (never reshuffled at stage transitions)

### 3a. `Trainer.__init__` — create a dedicated `val_dataset` (`train.py:94-152`)

After the train_dataset cache is populated (after the `precompute()` / `load_precomputed()` block),
add:

```python
# Fixed validation dataset — generated once with full type distribution, never changes
val_dataset_size = data_cfg.get("val_dataset_size", 2000)
self.val_dataset = SyntheticEQDataset(
    num_bands=self.num_bands,
    sample_rate=self.sample_rate,
    duration=1.5,
    n_fft=self.n_fft,
    size=val_dataset_size,
    gain_range=tuple(data_cfg["gain_bounds"]),
    freq_range=tuple(data_cfg["freq_bounds"]),
    q_range=tuple(data_cfg["q_bounds"]),
    type_weights=data_cfg.get("type_weights", None),  # global balanced weights
    precompute_mels=True,
    n_mels=data_cfg.get("n_mels", 128),
)
self.val_dataset.precompute()
```

Change the three-way `random_split` to a two-way split (train + test only), and build
`val_loader` from `val_dataset`:

```python
# Two-way split — val portion folds into training
n_train = int(len(self.train_dataset) * (data_cfg["train_split"] + data_cfg["val_split"]))
n_test  = len(self.train_dataset) - n_train
self.train_set, self.test_set = random_split(
    self.train_dataset, [n_train, n_test],
    generator=torch.Generator().manual_seed(42),
)

# val_loader from fixed dataset
self.val_loader = DataLoader(
    self.val_dataset,
    batch_size=data_cfg["batch_size"],
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
    pin_memory=pin_memory,
    prefetch_factor=2 if num_workers > 0 else None,
)
```

Add `val_dataset_size: 2000` under `data:` in `conf/config.yaml`.

### 3b. `_apply_curriculum_stage` — stop recreating `val_loader` (`train.py:631-661`)

After `self.train_dataset.precompute()`, replace the three-way split + two DataLoader
recreations with a two-way split + only a train_loader recreation:

```python
# Re-split training data (val_loader untouched — points to fixed val_dataset)
n_train = int(len(self.train_dataset) * (
    self.cfg["data"]["train_split"] + self.cfg["data"]["val_split"]
))
n_test = len(self.train_dataset) - n_train
self.train_set, self.test_set = random_split(
    self.train_dataset, [n_train, n_test],
    generator=torch.Generator().manual_seed(42),
)

self.train_loader = DataLoader(
    self.train_set,
    batch_size=self.cfg["data"]["batch_size"],
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
    pin_memory=pin_memory,
    drop_last=True,
    prefetch_factor=2 if num_workers > 0 else None,
)
# self.val_loader NOT recreated — fixed val_dataset from __init__
```

Remove the `self.val_loader = DataLoader(self.val_set, ...)` block entirely.

---

## Verification

```bash
cd insight

# 1. Confirm matched type targets are used in loss (gradient direction test)
python -c "
import torch
from loss_multitype import MultiTypeEQLoss, HungarianBandMatcher

matcher = HungarianBandMatcher()
pred_gain = torch.randn(2, 5)
target_gain = torch.randn(2, 5)
pred_freq = torch.rand(2, 5) * 19980 + 20
target_freq = torch.rand(2, 5) * 19980 + 20
pred_q = torch.rand(2, 5) * 9.9 + 0.1
target_q = torch.rand(2, 5) * 9.9 + 0.1
target_ft = torch.randint(0, 5, (2, 5))
mg, mf, mq, mft = matcher(pred_gain, pred_freq, pred_q,
                           target_gain, target_freq, target_q,
                           target_filter_type=target_ft)
assert mft.shape == target_ft.shape
print('PASS: matcher returns matched filter types')

loss = MultiTypeEQLoss()
pred_logits = torch.randn(2, 5, 5)
pred_H = torch.ones(2, 1025)
tH = torch.ones(2, 1025)
total, comps = loss(pred_gain, pred_freq, pred_q, pred_logits, pred_H,
                    target_gain, target_freq, target_q, target_ft, tH)
print(f'PASS: loss computed = {total.item():.4f}')
"

# 2. Quick 3-epoch smoke test (fast config)
python -c "
import yaml, tempfile, os
import train as tr_mod
with open('conf/config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['dataset_size'] = 400
cfg['data']['val_dataset_size'] = 100
cfg['trainer']['max_epochs'] = 3
cfg['data']['num_workers'] = 0
cfg['curriculum']['stages'][0]['epochs'] = 2
cfg['curriculum']['stages'][1]['epochs'] = 1
cfg['curriculum']['stages'][2]['epochs'] = 0
cfg['curriculum']['stages'][3]['epochs'] = 0
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(cfg, f); tmp = f.name
try:
    t = tr_mod.Trainer(config_path=tmp); t.fit()
    for e in t.history:
        print(f'epoch {e[\"epoch\"]}: type_acc={e[\"metrics\"].get(\"type_acc\",0):.1%}')
    print('PASS')
finally:
    os.unlink(tmp)
"
```

Expected: type_acc should improve steadily (even over 3 epochs) rather than staying flat or
regressing at stage 2 transition.
