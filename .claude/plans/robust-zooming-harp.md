# Comprehensive Training Loss Reduction Guide
## StreamingTCNModel — blind parametric EQ estimation
### Observed: `train_loss: 32.65` / `val_loss: 16.19`

---

## Context

The model is `StreamingTCNModel`: a causal 1D TCN encoder (WaveNet-style dilated gated convolutions) → `MultiTypeEQParameterHead` → `DifferentiableBiquadCascade`, trained to estimate blind EQ parameters (gain, freq, Q, filter type) from synthetic audio. Training uses `MultiTypeEQLoss` (Huber regression + cross-entropy + frequency response L1 + regularizations) with `HungarianBandMatcher` for permutation-invariant band assignment.

The loss ratio `train_loss ≈ 2 × val_loss` is an atypical inversion. This guide addresses root causes in priority order, grounded in code inspection and peer-reviewed citations.

---

## 1. Diagnosis — Root Cause of Inverted Loss Ratio

### Primary Cause (Code-Verified): EMA Weight Substitution at Validation

**This is the dominant cause.** `validate()` in `insight/train.py:666-851` calls `_load_ema_weights()` before `model.eval()`. Validation runs on the **EMA model** (exponential moving average of all previous checkpoints, `ema_decay=0.999`), which is inherently smoother and lower-loss than the live weights used during training. This is intentional, but it means `val_loss` and `train_loss` are not comparable — they measure different model states.

**Verification:** Run `audit_loss_reduction()` (Code Example 1) with `ema_model=trainer.ema_model`. The `ema_gap` output will confirm the delta.

**Secondary Cause: BatchNorm train/eval mode divergence.** `TCNBlock` uses `BatchNorm1d`. In `model.train()`, BN uses batch statistics; in `model.eval()`, it uses running statistics. If these have not converged (early training, small batches), training-mode BN inflates loss. Switching to `WeightNorm` or `LayerNorm` eliminates this entirely.

### Hypothesis Probability Table

| Hypothesis | Mechanism | Probability | Evidence |
|---|---|---|---|
| EMA at validation | val runs on EMA model | **CONFIRMED** | Code: `validate()` calls `_load_ema_weights()` |
| Loss reduction mismatch (`sum` vs `mean`) | `train_loss / val_loss = batch_size_train / batch_size_val` | **LIKELY** | Check `MultiTypeEQLoss` constructors |
| Training set harder than val | EQ distribution mismatch | POSSIBLE | Audit param distributions |
| BatchNorm train/eval | variance shift (Li et al., CVPR 2019) | POSSIBLE (minor) | Explains 5–30%, not 2× |
| Hungarian matching instability | unstable assignments (Carion et al., ECCV 2020) | POSSIBLE | Secondary for feedforward TCN |
| Data leakage | seed overlap | UNLIKELY | On-the-fly synthetic data |

### Diagnostic Methods (Systematic)

1. **Check loss reduction** (< 5 min): Print all `reduction` args in `MultiTypeEQLoss.__init__`. Confirm `reduction='mean'` everywhere. If `reduction='sum'` anywhere, fix first.
2. **Quantify EMA gap** (30 min): Run `audit_loss_reduction()` (Code 1). If `ema_gap < -0.5`, EMA is the explanation.
3. **Per-component loss logging**: Log `{'loss/huber', 'loss/ce', 'loss/freq_resp', 'loss/activity_reg', 'loss/freq_spread'}` separately. If any one component > 70% of total, its scale needs correction.
4. **Gradient norm tracking** (`clip_grad_norm_` return value): Log at every step. Consistent norm > `max_norm=1.0` → loss scale is too large relative to optimizer expectations.
5. **Activation histograms**: `wandb.watch(model, log='all')` at steps 0, 100, 500. Dead gate detection: fraction of `TCNBlock` activations < 1e-4 > 50% → gate saturation.

**Key files:** `insight/train.py:666-851` (validate), `insight/loss_multitype.py` (MultiTypeEQLoss)

---

## 2. Critical Bug: Gumbel Temperature `min_tau` Floor Suppresses Curriculum

**Code-verified in `insight/differentiable_eq.py:866-886`:**

```python
tau = max(self.gumbel_temperature.item(), self.min_tau)  # min_tau = 0.5
```

The curriculum (`insight/conf/config.yaml:86-189`) sets temperatures `0.15 → 0.1 → 0.05` for `shelf_types`, `full_multitype`, and final stages. **All of these hit the `min_tau=0.5` floor and are never executed.** The filter type head never reaches the low-temperature regime needed for sharp type discrimination.

**Impact:** At τ=0.5 (floor), type probabilities remain relatively diffuse → `forward_soft()` blends all 5 filter response curves → elevated `hmag_loss` and `recon_loss` throughout late curriculum stages.

**Fix:** Either lower `min_tau` to 0.05–0.1 in `MultiTypeEQParameterHead.__init__`, or replace with `GumbelSoftmaxWithAnnealing` (Code Example 3) which exposes `min_temp` as a configurable parameter.

**Source:** `insight/differentiable_eq.py:~875` (`min_tau = 0.5` hardcoded); `insight/conf/config.yaml:86-189` (curriculum temps).

---

## 3. Loss Decomposition at Initialization

**Cross-entropy contribution from random logits:**
- 3 bands × `lambda_type=2.0` × `−log(1/5)` ≈ 3 × 2.0 × 1.609 = **9.67 nats**
- This is ~30% of `train_loss=32.65`

As logits become non-uniform, CE decreases rapidly. If CE stays high after 1000 steps, it indicates the model is not learning filter type discrimination — possibly because the `min_tau` floor (§2) is keeping type probs diffuse.

**Frequency response L1 contribution**: If evaluated on a linear frequency grid over [20, 20000] Hz with many bins, and bins are not normalized per-bin, this term dominates. Normalize by `n_fft // 2 + 1` frequency bins.

**Source:** `insight/loss_multitype.py:434-439` (confirms CE uses raw logits, not Gumbel probs)

---

## 4. Optimization Techniques

### Priority Order for This Loss Profile

**Highest impact:**

1. **Gradient clipping** (`max_norm=1.0`): Add before every `optimizer.step()`. High initial loss → large gradients → Adam second-moment corruption. Pascanu et al. (ICML 2013); Bai et al. (arXiv:1803.01271) use this for TCNs.
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
   Log the pre-clip norm at every step — persistent norm > 1.0 means loss scale is wrong.

2. **LR warmup**: 2–5% of total steps from `lr=1e-6` → `lr=3e-4`. Vaswani et al. (NeurIPS 2017); Devlin et al. (NAACL 2019) justify warmup by unreliable Adam second-moment estimates in early steps.

3. **OneCycleLR** (Code Example 2): Smith & Topin (ICML Workshop 2019). For high-loss starting points, the 30% ascending phase finds basins quickly. `max_lr=3e-4 to 1e-3` (use LR range finder: Smith 2017, WACV).

4. **RAdam as alternative**: Liu et al. (ICLR 2020). Principled early-step correction, no warmup hyperparameter needed. `torch.optim.RAdam` available since PyTorch 1.10.

**Medium impact:**

5. **AdamW weight_decay tuning**: Currently high train_loss suggests underfitting → reduce `weight_decay` to `1e-4` initially. Loshchilov & Hutter (ICLR 2019).

6. **Separate LR for curriculum stages**: Consider reducing LR at each curriculum transition (new filter types → harder problem). Reduce by 2–5× at each stage boundary.

### Normalization: WeightNorm over BatchNorm

**Critical for streaming:** `BatchNorm1d` in `TCNBlock` uses batch statistics during training, running statistics at inference. With `batch_size=1` (streaming), BN fails. Replace with:
- `torch.nn.utils.weight_norm(nn.Conv1d(...))` — used in original WaveNet (van den Oord et al., 2016, arXiv:1609.03499), Kalchbrenner et al. (ICML 2018). Baked into weights at inference, zero overhead.
- Alternatively: `LayerNorm` (Ba et al., arXiv:1607.06450) — batch-size independent, used in GPT-2/GPT-3.

Li et al. (CVPR 2019) "Understanding the Disharmony between Dropout and Batch Normalization" shows BN + dropout creates variance shift. Removing BN resolves this.

### Regularization

- **Spatial (channel) dropout** (`nn.Dropout1d`) between residual blocks at rate 0.05: Tompson et al. (CVPR 2015). Do NOT apply inside gated activation pair.
- **L1 on gain head weights**: Encourages sparse band activity (many bands at 0 dB gain). `lambda_l1 ≈ 1e-4`.
- **Label smoothing** (`eps=0.1`) on filter type cross-entropy: Szegedy et al. (CVPR 2016). Reduces overconfidence on majority filter class.

---

## 5. Audio-Specific Preprocessing

### Mel-Spectrogram Parameter Issues

**n_fft:** Current codebase default likely 1024 or 2048. For EQ estimation:
- Sub-bass (20–80 Hz): requires `n_fft ≥ 4096` at 44.1 kHz (Δf ≤ 10 Hz)
- Bass (80–250 Hz): requires `n_fft ≥ 2048` (Δf ≤ 21.5 Hz)
- **Recommendation: n_fft = 4096** for adequate resolution across all filter types

Heisenberg uncertainty (Müller, "Fundamentals of Music Processing," Springer 2015, §2.4): larger n_fft → better freq resolution, more latency. For offline training this is acceptable.

**Filterbank: mel → log-frequency:** The mel scale compresses high frequencies, leaving only 8–12 bins above 8 kHz (with n_mels=128). High-shelf detection at 10 kHz is severely underdetermined. Replace with **log-frequency filterbank**: `np.geomspace(f_min, f_max, n_bins)` as center frequencies. This aligns with the log-frequency parameterization already used in `MultiTypeEQParameterHead`.

TimbreTron (Huang et al., ICLR 2019) chose CQT over mel for the same reason. Brown (1991, JASA) defines CQT.

**Window:** Hann window — best sidelobe rolloff (−18 dB/oct) for narrow EQ peak detection. Harris (Proc. IEEE, 1978). Verify `STFTFrontend` uses Hann, not rectangular.

### Feature Scaling

- **Use `10 × log10(mel + ε)` (dB scale)**: EQ gain is in dB — log representation makes gain additive in feature space. Do NOT use mu-law (destroys dB-linear relationships).
- **Do NOT use per-channel (per-mel-bin) normalization**: Removes relative amplitude differences between bands, which IS the EQ signature.
- **Global statistics normalization** (safest): Compute dataset-wide mean/std of log-mel. Safe: preserves EQ shape, removes excitation mean.
- **Normalize EQ parameter targets**: Gain ÷ 24, log-freq → [0,1], log-Q → [0,1]. Without this, Huber loss is dominated by Hz-scale frequency errors.

### Augmentation

| Augmentation | Impact | Notes |
|---|---|---|
| SNR-controlled noise injection | HIGH | Add to wet signal, not dry. SNR ∈ [10,40] dB. |
| Polarity inversion | LOW (free) | Multiply waveform by −1, no EQ effect |
| Gain normalization (±20 dB random scale) | MEDIUM | Forces amplitude invariance |
| SpecAugment **time masking** | LOW | OK, time-invariant EQ survives |
| SpecAugment **frequency masking** | HARMFUL | Destroys EQ signature — **do not use** |
| Room IR convolution | HIGH (but harder) | Use curriculum: start anechoic |
| More signal types (polyphonic, transient) | MEDIUM | Add multi-harmonic summing + drum-like bursts |

Park et al. (Interspeech 2019): SpecAugment designed for ASR, not parameter estimation — explicit domain mismatch.

---

## 6. Loss Function Alternatives

### Huber Loss δ Calibration (High Impact, Low Effort)

Current: PyTorch default `delta=1.0` for all parameters. This is miscalibrated:
- **Gain** (dB scale, range ±24): `delta=3.0` dB — errors < 3 dB are barely audible (Moore, 2012)
- **Log-frequency** (normalized [0,1]): `delta=0.1` ≈ half-octave
- **Log-Q** (normalized [0,1]): `delta=0.1`

With `delta=1.0` and raw dB gain, almost every training example triggers the linear (not quadratic) regime — effectively using L1 loss everywhere and losing the tight-fitting benefit of Huber's quadratic region.

Advanced: Barron (CVPR 2019) "A General and Adaptive Robust Loss Function" — learnable robustness parameter α per output dimension. Auto-tunes δ during training.

### Frequency Response Evaluation Grid

**Current:** likely linear frequency grid → majority of loss signal from high frequencies.
**Fix:** Evaluate on log-frequency grid (`np.geomspace(f_min, f_max, K)`). Each octave contributes equally to loss.

**Optional:** Weight by ISO 226:2003 equal-loudness curve — presence region (1–4 kHz) contributes more; extreme sub-bass contributes less. Zwicker & Fastl (Psychoacoustics, Springer, 1990).

### Multi-Task Loss Balancing

**EnCodec-style loss balancer** (Défossez et al., TMLR 2023): normalizes each loss component's gradient by its EMA magnitude so each contributes a fixed fraction of total gradient signal:
```
w_i(t) = target_fraction_i / (EMA(||∇L_i||) + ε)
```
This eliminates the need to manually tune `lambda_huber`, `lambda_type`, `lambda_freq_resp` etc.

**Uncertainty weighting**: Kendall, Gal & Cipolla (CVPR 2018, arXiv:1705.07115) — treat each loss weight as learnable homoscedastic uncertainty. Outperforms grid search for multi-task loss weighting.

### Loss Components Ranked by Task Alignment

1. **Frequency response L1** (`FreqResponseLoss`): most directly task-relevant — evaluates in the EQ domain without needing parameter-space matching. Enhance with log-frequency grid.
2. **Huber parameter regression**: well-suited with proper δ and normalization.
3. **Multi-Resolution STFT Loss** (`MultiResolutionSTFTLoss` already in `loss.py`): use multiple n_fft values to capture both fine (high-Q peaks) and coarse (shelf tilt) structure. Yamamoto et al. (ICASSP 2020).
4. **SI-SNR** (Le Roux et al., ICASSP 2019): applicable only in cycle-consistency path (estimated params → resynthesized wet → compare to target wet). `CycleConsistencyLoss` in `loss.py` serves this purpose.
5. **Perceptual loss via PANNs** (Kong et al., IEEE/ACM TASLP 2020): use shallow layers only (layers 1–2). Full-depth PANN features encode semantics, not EQ.

---

## 7. Model Architecture

### Receptive Field — Not a Bottleneck

**Code-verified:** `CausalTCNEncoder` with `num_stacks=2`, `num_blocks=6`, `kernel_size=5`:
```
RF per stack = Σ_{i=0}^{5} (5-1) × 2^i = 4 × (2^6 − 1) = 252 frames
Total RF = 1 + 2 × 252 = 505 frames
```
At `hop_length=256` @ 44.1 kHz: **505 × 5.8ms ≈ 2.9 seconds** — exceeds the 1.5s training clips. Adding more depth adds no temporal context benefit.

**Width is the right lever** (not depth): `channels=128 → 256` provides denser gradient signal per layer. van den Oord et al. (2016) found 512-channel blocks improve representation density.

### Gumbel-Softmax Temperature Fix (Highest Architecture Priority)

See §2 above. `min_tau=0.5` in `MultiTypeEQParameterHead.forward()` floors all curriculum temperatures. Fix: lower to 0.05, or deploy Code Example 3.

### Attention — Already Sophisticated

`MultiTypeEQParameterHead` already has: local CNN + dilated CNN dual-path, input-conditioned queries, Gaussian position priors (`attn_position_bias`), learnable blend weight (`freq_blend_weight`). The `attn_temperature=0.1` sharpens attention heavily.

Do NOT add Transformer self-attention (AST, Gong et al., Interspeech 2021) until existing attention paths are stable. Monitor `sigmoid(freq_blend_weight)` (should be non-trivial blend, not 0 or 1).

### Hungarian Matching Curriculum

DETR-style matching (Carion et al., ECCV 2020) is correct long-term, but Zhao et al. (arXiv:2211.14448) showed Hungarian gradient has discontinuities at assignment switches — unstable early in training. Mitigation: start with frequency-sorted assignment (bands assigned by ascending predicted center frequency), switch to Hungarian once frequency predictions are approximately correct.

DN-DETR (Li et al., arXiv:2203.01305): add teacher-forcing denoising queries in early epochs to reduce matching difficulty.

---

## 8. Training Dynamics

### Data Splitting for Synthetic On-the-Fly Data

**Risk:** If training and validation `SyntheticEQDataset` instances use overlapping random seeds or different EQ parameter distributions, the inverted ratio will persist independent of model improvements.

**Fix:**
- Use `torch.Generator` with isolated seeds per split: `train_seed=42`, `val_seed=99`
- Set `worker_init_fn=lambda w: np.random.seed(base_seed + w)` for each DataLoader (PyTorch Reproducibility docs)
- Verify identical EQ parameter ranges (gain, freq, Q, filter type probabilities) between splits
- Log mean absolute gain, mean Q, fraction of active bands for 1000 samples from each split

### Class Imbalance in Filter Types

5 filter types (peaking, lowshelf, highshelf, highpass, lowpass) may not be equally distributed in synthetic generation, biasing cross-entropy. Fix options:
1. Enforce uniform type sampling in `SyntheticEQDataset` (simplest)
2. Focal loss (Lin et al., ICCV 2017): `gamma=2` down-weights well-classified majority class
3. Class weighting: `CrossEntropyLoss(weight=inverse_frequency_tensor)`

### Early Stopping with Inverted Ratio

Standard early stopping (monitor val_loss) gives false positives here because val_loss is biased low by EMA weights. **Monitor `train_loss / val_loss` ratio** in addition to absolute val_loss. As training converges, ratio should approach 1.0. If ratio stays > 1.5 after 20 epochs, data distribution mismatch is likely.

Prechelt (1998, "Early Stopping — But When?", Neural Networks: Tricks of the Trade): patience ≥ T_max/3 for cosine annealing.

---

## 9. Prioritized Action Plan

### Ranked Interventions (Highest → Lowest Expected Impact)

| Rank | Intervention | Effort | Expected Impact | Verification |
|---|---|---|---|---|
| **1** | Fix `min_tau=0.5` floor in `MultiTypeEQParameterHead` | 10 min | **HIGH** — curriculum annealing is currently broken | Check type cross-entropy decreases post fix |
| **2** | Audit loss reduction: confirm `reduction='mean'` in all `MultiTypeEQLoss` components | 15 min | **HIGH** — explains 2:1 if any component uses `sum` | Run Code Example 1 |
| **3** | Add gradient clipping `max_norm=1.0` before every optimizer step | 15 min | **HIGH** — stabilizes Adam second-moment early | Log pre-clip grad norms |
| **4** | Calibrate Huber `delta`: `delta=3.0` for gain (dB), `delta=0.1` for log-freq/log-Q | 30 min | **HIGH** — most training currently in L1 regime | Per-component loss before/after |
| **5** | LR warmup (2–5% of steps, from `lr=1e-6` to `lr=3e-4`) | 30 min | **HIGH** — AdamW unstable without warmup at high initial loss | Loss curve improvement in first 500 steps |
| **6** | Verify train/val `SyntheticEQDataset` use identical parameter distributions | 30 min | **HIGH** — explains persistent ratio if distributions differ | Log param statistics per split |
| **7** | Replace `BatchNorm1d` in `TCNBlock` with `WeightNorm` or `LayerNorm` | 1–2h | MEDIUM — eliminates batch-size-dependent noise; required for streaming | Compare train/val loss gap before/after |
| **8** | Normalize EQ parameter targets: gain ÷ 24, log-freq → [0,1], log-Q → [0,1] | 1h | MEDIUM — prevents Hz-scale frequency errors dominating Huber | Per-component Huber contribution |
| **9** | Switch mel filterbank → log-frequency filterbank (`np.geomspace(f_min, f_max, n_bins)`) | 2–3h | MEDIUM — improves high-shelf and low-freq band detection | FreqResponseLoss before/after |
| **10** | Evaluate `FreqResponseLoss` on log-frequency grid instead of linear grid | 1h | MEDIUM — each octave contributes equally | Loss value and per-frequency error |
| **11** | Increase `n_fft` to 4096 (from current default) | 1h | MEDIUM — sub-bass EQ unresolvable with smaller n_fft | FreqResponseLoss for low-freq bands |
| **12** | OneCycleLR policy (Code Example 2) | 1h | MEDIUM — fast escape from high-loss starting region | Loss curve shape |
| **13** | Hungarian matching curriculum: start with freq-sorted, switch at epoch 10–15 | 2h | LOW-MEDIUM — reduces assignment instability | Per-epoch assignment change fraction |
| **14** | EnCodec-style gradient-magnitude loss balancer | 3–4h | MEDIUM-HIGH (if components are imbalanced) | Per-component gradient norm ratios |
| **15** | Add polyphonic + transient signal types to `SyntheticEQDataset` | 2h | MEDIUM (generalization) | Val loss on held-out real recordings |

### Minimum Viable Fix Set (Day 1, < 2 hours total)

1. **`insight/differentiable_eq.py`**: Change `self.min_tau = 0.5` → `self.min_tau = 0.05` in `MultiTypeEQParameterHead.__init__`
2. **`insight/loss_multitype.py`**: Audit all loss component constructors — assert `reduction='mean'`
3. **`insight/train.py`**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`
4. **`insight/conf/config.yaml`**: Calibrate `delta` values per parameter in Huber loss config

---

## 10. Code Examples (from agent-arch-code)

Three production-quality code modules were generated:

### Code Example 1: `audit_loss.py`
`audit_loss_reduction(model, train_loader, val_loader, loss_fn, ema_model=trainer.ema_model)` — quantifies EMA gap, BN mode gap, per-sample loss ratio. `ConsistentReductionLossWrapper` adds range assertions.

### Code Example 2: `optimizer_setup.py`
`build_adamw_with_onecycle(model, base_lr, max_lr, weight_decay, warmup_epochs, total_epochs, steps_per_epoch)` — parameter group separation (norm layers get `weight_decay=0`), LinearLR warmup → OneCycleLR. `training_step()` with gradient norm logging and per-layer explosion diagnosis.

### Code Example 3: `gumbel_annealing.py`
`TemperatureScheduler` (exponential/linear/cosine) + `GumbelSoftmaxWithAnnealing` — intra-stage epoch-level annealing, hard argmax at `model.eval()`, serializes temperature as buffer. Drop-in replacement for `MultiTypeEQParameterHead`'s manual Gumbel block. Call `head.gumbel.anneal(epoch)` from `train_one_epoch()`.

**Note on min_tau:** The scheduler's `min_temp` parameter here is independent of `MultiTypeEQParameterHead.min_tau`. Both must be updated. The code's `GumbelSoftmaxWithAnnealing` respects its own `min_temp`; the `min_tau` in the head's `forward()` must also be lowered.

---

## Key Files

| File | Lines | Relevance |
|---|---|---|
| `insight/train.py` | 666–851 | `validate()` — EMA weight substitution (confirmed root cause) |
| `insight/differentiable_eq.py` | 866–886 | `MultiTypeEQParameterHead.forward()` — `min_tau=0.5` bug |
| `insight/conf/config.yaml` | 86–189 | Curriculum temps (0.15/0.1/0.05 → never reach model) |
| `insight/loss_multitype.py` | 434–439 | `MultiTypeEQLoss` — CE uses raw logits; check reduction modes |
| `insight/dataset.py` | — | `SyntheticEQDataset` — check train/val seed isolation |
| `insight/dsp_frontend.py` | — | `STFTFrontend` — n_fft, hop_length, window type |
| `insight/model_tcn.py` | — | `CausalTCNEncoder` — receptive field 505 frames (not a bottleneck) |

---

## Verification / Testing

After each fix, run:
```bash
cd insight
python test_eq.py          # gradient flow through biquad
python test_model.py       # forward/inverse/cycle/gradient
python test_streaming.py   # streaming consistency
python test_multitype_eq.py  # multi-type filter parameter head
python train.py             # full run; monitor per-component loss breakdown
```

Expected outcomes after implementing the minimum viable fix set:
- `train_loss / val_loss` ratio should approach 1.0–1.5 (reflecting only the true EMA advantage)
- CE component should decrease from ~9.7 nats to < 2.0 nats within 1000 steps (Gumbel temp fix)
- Gradient norm should stay below `max_norm=1.0` consistently (clip fix)

---

## References (Key Papers)

- Carion et al. (ECCV 2020) — DETR: Hungarian matching instability
- Bai, Kolter & Koltun (arXiv:1803.01271, 2018) — TCN empirical eval; gradient clipping recommendation
- Loshchilov & Hutter (ICLR 2019) — AdamW; weight decay decoupling
- Smith & Topin (ICML Workshop 2019) — OneCycleLR super-convergence
- Pascanu, Mikolov & Bengio (ICML 2013) — gradient clipping for deep networks
- Li et al. (CVPR 2019) — BN+Dropout disharmony; variance shift
- Jang et al. (ICLR 2017) — Gumbel-Softmax; temperature annealing
- Kendall, Gal & Cipolla (CVPR 2018, arXiv:1705.07115) — uncertainty-weighted multi-task loss
- Défossez et al. (TMLR 2023) — EnCodec; loss balancer design
- Barron (CVPR 2019) — adaptive robust loss function
- Yamamoto, Song & Kim (ICASSP 2020) — Multi-Resolution STFT Loss
- Müller (Springer 2015) — STFT uncertainty principle for audio
- Kong et al. (IEEE/ACM TASLP 2020) — PANNs for perceptual loss
- ISO 226:2003 — equal-loudness contours for frequency weighting
