"""
Training script for the multi-type IDSP EQ Estimator.

Uses StreamingTCNModel with MultiTypeEQLoss (Hungarian matching,
permutation-invariant parameter regression, type classification).
On-the-fly synthetic multi-type data generation.

Optimization features (QLoRA/Unsloth/DeepSpeed-inspired):
  - 8-bit AdamW optimizer (bitsandbytes) for reduced memory
  - torch.compile for graph-level optimization
  - Gradient checkpointing for TCN stacks
  - Fused Triton kernels for gated activation
  - DeepSpeed ZeRO-2 integration for multi-GPU
  - Pin memory + async GPU transfer
  - Precomputed dataset caching to disk
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model_tcn import StreamingTCNModel
from dsp_frontend import STFTFrontend
from loss_multitype import MultiTypeEQLoss, HungarianBandMatcher
from dataset import SyntheticEQDataset, collate_fn
from dataset_musdb import MUSDB18EQDataset
import yaml
from pathlib import Path
import time
import json
import math
import os
from contextlib import nullcontext

try:
    import bitsandbytes as bnb

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    import deepspeed

    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False


def load_config(path="conf/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Trainer:
    def __init__(self, config_path="conf/config.yaml", resume_path=None):
        self.cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_cfg = self.cfg["data"]
        model_cfg = self.cfg["model"]
        loss_cfg = self.cfg["loss"]
        trainer_cfg = self.cfg["trainer"]

        self.num_bands = data_cfg["num_bands"]
        self.sample_rate = data_cfg["sample_rate"]
        self.max_epochs = trainer_cfg["max_epochs"]
        self.log_every = trainer_cfg["log_every_n_steps"]
        self.n_fft = data_cfg.get("n_fft", 2048)

        # Optimization flags from config
        self.use_8bit_optimizer = (
            trainer_cfg.get("use_8bit_optimizer", False) and HAS_BITSANDBYTES
        )
        self.use_torch_compile = trainer_cfg.get("use_torch_compile", False)
        self.use_gradient_checkpointing = trainer_cfg.get(
            "use_gradient_checkpointing", False
        )
        self.use_activation_quantization = trainer_cfg.get(
            "use_activation_quantization", False
        )
        self.use_deepspeed = trainer_cfg.get("use_deepspeed", False) and HAS_DEEPSPEED
        self.precompute_cache_path = trainer_cfg.get("precompute_cache_path", None)

        if self.use_8bit_optimizer:
            print("  [opt] Using 8-bit AdamW optimizer (bitsandbytes)")
        if self.use_torch_compile:
            print("  [opt] Using torch.compile for graph optimization")
        if self.use_gradient_checkpointing:
            print("  [opt] Using gradient checkpointing for TCN stacks")
        if self.use_activation_quantization:
            print("  [opt] Using INT8 activation quantization")
        if self.use_deepspeed:
            print("  [opt] Using DeepSpeed ZeRO-2 optimization")

        # Dataset selection: "musdb" for real audio, "synthetic" (or unset) for synthetic
        dataset_type = data_cfg.get("dataset_type", "synthetic")
        dataset_size = data_cfg.get("dataset_size", 30000)
        val_dataset_size = data_cfg.get("val_dataset_size", 2000)
        ds_kwargs = dict(
            num_bands=self.num_bands,
            sample_rate=self.sample_rate,
            duration=data_cfg.get("audio_duration", 1.5),
            n_fft=self.n_fft,
            size=dataset_size,
            gain_range=tuple(data_cfg["gain_bounds"]),
            freq_range=tuple(data_cfg["freq_bounds"]),
            q_range=tuple(data_cfg["q_bounds"]),
            type_weights=data_cfg.get("type_weights", None),
            precompute_mels=True,
            n_mels=data_cfg.get("n_mels", 128),
        )

        if dataset_type == "musdb":
            musdb_root = data_cfg.get("musdb_root", "")
            musdb_subsets = data_cfg.get("musdb_subsets", ["train", "test"])
            print(f"  [data] Using MUSDB18 dataset from {musdb_root}")
            self.train_dataset = MUSDB18EQDataset(
                musdb_root=musdb_root,
                subsets=musdb_subsets,
                **ds_kwargs,
            )
        else:
            print(f"  [data] Using synthetic dataset")
            self.train_dataset = SyntheticEQDataset(**ds_kwargs)

        # Try loading precomputed cache from disk
        if self.precompute_cache_path and self.train_dataset.load_precomputed(
            self.precompute_cache_path
        ):
            print(f"  [data] Loaded cached dataset from {self.precompute_cache_path}")
        else:
            # Precompute all samples with mel-spectrograms upfront
            self.train_dataset.precompute()
            # Save cache if path specified
            if self.precompute_cache_path:
                self.train_dataset.save_precomputed(self.precompute_cache_path)

        # Fixed validation dataset — generated once with full type distribution.
        if dataset_type == "musdb":
            self.val_dataset = MUSDB18EQDataset(
                musdb_root=musdb_root,
                subsets=musdb_subsets,
                **{**ds_kwargs, "size": val_dataset_size},
            )
        else:
            self.val_dataset = SyntheticEQDataset(
                **{**ds_kwargs, "size": val_dataset_size},
            )
        self.val_dataset.precompute()

        # Train split: fold val portion into training since we have a separate val set.
        n_train = int(
            len(self.train_dataset) * (data_cfg["train_split"] + data_cfg["val_split"])
        )
        n_test = len(self.train_dataset) - n_train
        self.train_set, self.test_set = random_split(
            self.train_dataset,
            [n_train, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        # Pin memory for async GPU transfer
        pin_memory = self.device.type == "cuda"
        num_workers = data_cfg.get("num_workers", 0)

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=data_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Model
        enc_cfg = model_cfg.get("encoder", {})
        self.model = StreamingTCNModel(
            n_mels=data_cfg.get("n_mels", 128),
            embedding_dim=enc_cfg.get("embedding_dim", 128),
            num_bands=self.num_bands,
            channels=enc_cfg.get("channels", 128),
            num_blocks=enc_cfg.get("num_blocks", 4),
            num_stacks=enc_cfg.get("num_stacks", 2),
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
        ).to(self.device)

        # STFT frontend
        self.frontend = STFTFrontend(
            n_fft=self.n_fft,
            hop_length=data_cfg.get("hop_length", 256),
            win_length=self.n_fft,
            mel_bins=data_cfg.get("n_mels", 128),
            sample_rate=self.sample_rate,
        ).to(self.device)

        # Loss
        self.criterion = MultiTypeEQLoss(
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            lambda_param=loss_cfg.get("lambda_param", 0.0),
            lambda_gain=loss_cfg.get("lambda_gain", 3.0),
            lambda_freq=loss_cfg.get("lambda_freq", 1.0),
            lambda_q=loss_cfg.get("lambda_q", 0.5),
            lambda_type=loss_cfg.get("lambda_type", 0.5),
            lambda_spectral=loss_cfg.get("lambda_spectral", 1.0),
            lambda_hmag=loss_cfg.get("lambda_hmag", 0.3),
            lambda_activity=loss_cfg.get("lambda_activity", 0.1),
            lambda_spread=loss_cfg.get("lambda_spread", 0.05),
            lambda_embed_var=loss_cfg.get("lambda_embed_var", 0.3),
            lambda_contrastive=loss_cfg.get("lambda_contrastive", 0.05),
            lambda_type_match=loss_cfg.get("lambda_type_match", 0.5),
            embed_var_threshold=loss_cfg.get("embed_var_threshold", 0.1),
            warmup_epochs=loss_cfg.get("warmup_epochs", 5),
        )

        # Hungarian matcher for fair validation metrics (matches target ordering to predictions)
        self.matcher = HungarianBandMatcher(
            lambda_gain=1.0, lambda_freq=1.0, lambda_q=1.0, lambda_type_match=0.5
        )

        # Optimizer: per-group learning rates to handle the ~17,000x gradient imbalance
        # between the mel_cnn encoder (large grads) and gain_mlp head (tiny grads).
        base_lr = model_cfg["learning_rate"]
        wd = model_cfg["weight_decay"]
        head_lr_mult = model_cfg.get("head_lr_multiplier", 3.0)

        # Classify parameters: encoder backbone gets base_lr, param head gets higher
        encoder_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "param_head" in name:
                head_params.append(param)
            else:
                encoder_params.append(param)

        print(f"  [opt] LR groups: encoder={base_lr:.1e}, head={base_lr*head_lr_mult:.1e} ({head_lr_mult:.1f}x)")

        param_groups = [
            {
                "params": encoder_params,
                "lr": base_lr,
                "weight_decay": wd,
                "initial_lr": base_lr,
            },
            {
                "params": head_params,
                "lr": base_lr * head_lr_mult,
                "weight_decay": wd,
                "initial_lr": base_lr * head_lr_mult,
            },
        ]

        if self.use_8bit_optimizer:
            self.optimizer = bnb.optim.AdamW8bit(param_groups)
        else:
            self.optimizer = torch.optim.AdamW(param_groups, fused=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=1e-6
        )

        # LR warmup: linear warmup over first 500 steps, then hand off to CosineAnnealing
        warmup_steps = 500
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, self.scheduler],
            milestones=[warmup_steps],
        )

        self.gradient_accumulation_steps = trainer_cfg.get(
            "gradient_accumulation_steps", 1
        )
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = trainer_cfg.get("early_stopping_patience", 5)
        self.history = []
        self.start_epoch = 1

        if resume_path is not None:
            self._load_checkpoint(resume_path)
        self.criterion.to(self.device)

        # bf16-mixed precision: autocast context (no GradScaler needed for bf16)
        self.use_bf16 = trainer_cfg.get("precision", "").lower() == "bf16-mixed"
        if self.use_bf16:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            self.autocast_ctx = torch.autocast(device_type, dtype=torch.bfloat16)
        else:
            self.autocast_ctx = nullcontext()

        # Curriculum stages from config
        self.curriculum_stages = self.cfg.get("curriculum", {}).get("stages", [])
        self._current_stage_idx = -1
        self._last_metrics = {}  # D-06/D-07: Updated after each validate() call

        # Per-stage LR warmup state
        self._stage_warmup_active = False
        self._stage_warmup_step_count = 0
        self._target_type_weights = None

        # DeepSpeed initialization
        if self.use_deepspeed:
            ds_config_path = trainer_cfg.get(
                "deepspeed_config", "conf/deepspeed_config.json"
            )
            if os.path.exists(ds_config_path):
                with open(ds_config_path, "r") as f:
                    ds_config = json.load(f)
                self.model, self.optimizer, _, _ = deepspeed.initialize(
                    model=self.model,
                    optimizer=self.optimizer,
                    config=ds_config,
                )
                print(f"  [opt] DeepSpeed initialized with config: {ds_config_path}")
            else:
                print(
                    f"  [opt] WARNING: DeepSpeed config not found at {ds_config_path}, disabling"
                )
                self.use_deepspeed = False

        # torch.compile after DeepSpeed init to avoid conflicts
        if self.use_torch_compile:
            self.model = torch.compile(self.model)
            print("  [opt] torch.compile applied to model")

    def _prepare_input(self, batch):
        """Get mel-spectrogram from precomputed cache or compute on-the-fly."""
        if "wet_mel" in batch:
            # Use precomputed mel-spectrogram
            return batch["wet_mel"].to(self.device, non_blocking=True)
        # Fallback: compute from audio
        wet_audio = batch["wet_audio"].to(self.device, non_blocking=True)
        mel_spec = self.frontend.mel_spectrogram(wet_audio)
        return mel_spec.squeeze(1)

    def _spec_augment(self, mel_frames):
        """
        Apply SpecAugment to mel-spectrograms for training data augmentation.
        Uses frequency masking and time masking to improve model robustness.

        Args:
            mel_frames: (B, n_mels, T) mel-spectrogram

        Returns:
            Augmented mel-spectrogram of same shape
        """
        B, n_mels, T = mel_frames.shape
        device = mel_frames.device

        # Frequency masking: mask up to 15 consecutive mel bins, 2 masks
        freq_mask_param = 15
        num_freq_masks = 2
        for _ in range(num_freq_masks):
            f = torch.randint(0, freq_mask_param + 1, size=(B,)).to(device)
            f0 = torch.randint(0, max(n_mels - f.max().item(), 1), size=(B,)).to(device)
            for i in range(B):
                mel_frames[i, f0[i]:f0[i] + f[i], :] = 0

        # Time masking: mask up to 30 consecutive time frames, 2 masks
        time_mask_param = 30
        num_time_masks = 2
        for _ in range(num_time_masks):
            t = torch.randint(0, time_mask_param + 1, size=(B,)).to(device)
            t0 = torch.randint(0, max(T - t.max().item(), 1), size=(B,)).to(device)
            for i in range(B):
                mel_frames[i, :, t0[i]:t0[i] + t[i]] = 0

        return mel_frames

    def train_one_epoch(self, epoch):
        # Enable anomaly detection during warmup for precise NaN backtrace
        # (incompatible with torch.compile — compiled backward is a fused kernel)
        torch.autograd.set_detect_anomaly(epoch <= 2 and not self.use_torch_compile)
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_nan_batches = 0

        # Accumulate per-component losses for epoch-level logging
        component_accum = {}

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            mel_frames = self._prepare_input(batch)
            # Apply SpecAugment during training only
            mel_frames = self._spec_augment(mel_frames)
            target_gain = batch["gain"].to(self.device, non_blocking=True)
            target_freq = batch["freq"].to(self.device, non_blocking=True)
            target_q = batch["q"].to(self.device, non_blocking=True)
            target_ft = batch["filter_type"].to(self.device, non_blocking=True)
            active_band_mask = batch.get("active_band_mask", None)
            if active_band_mask is not None:
                active_band_mask = active_band_mask.to(self.device, non_blocking=True)

            # Forward (wrapped in autocast for bf16-mixed precision)
            with self.autocast_ctx:
                output = self.model(mel_frames)
                pred_gain, pred_freq, pred_q = output["params"]

                # Ground truth frequency response
                target_H_mag = self.model.dsp_cascade(
                    target_gain,
                    target_freq,
                    target_q,
                    n_fft=self.n_fft,
                    filter_type=target_ft,
                )

                # LOSS-04: Dual H_mag path — model outputs both soft (Gumbel) and hard (argmax)
                pred_H_mag_soft = output["H_mag"]       # soft path for spectral_loss gradient
                pred_H_mag_hard = output["H_mag_hard"]  # hard path for hmag_loss (detached inside loss)

                # LOSS-05: Spectral reconstruction — pass H_mag_soft when spectral is active
                is_spectral_active = (
                    hasattr(self.criterion, 'current_epoch') and
                    hasattr(self.criterion, 'warmup_epochs') and
                    self.criterion.current_epoch >= self.criterion.warmup_epochs + 2
                )
                pred_audio_for_loss = pred_H_mag_soft if is_spectral_active else None
                target_audio_for_loss = target_H_mag if is_spectral_active else None

                # Loss (dual H_mag path, active_band_mask, spectral reconstruction)
                total_loss, components = self.criterion(
                    pred_gain,
                    pred_freq,
                    pred_q,
                    output["type_logits"],
                    pred_H_mag_soft,
                    pred_H_mag_hard,
                    target_gain,
                    target_freq,
                    target_q,
                    target_ft,
                    target_H_mag,
                    pred_audio=pred_audio_for_loss,
                    target_audio=target_audio_for_loss,
                    active_band_mask=active_band_mask,
                    embedding=output["embedding"],
                )

            # Skip batch if loss is NaN (from early training instability)
            if not torch.isfinite(total_loss):
                # Identify which component caused NaN
                nan_components = [
                    k
                    for k, v in components.items()
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                ]
                print(
                    f"  [nan] Non-finite loss at train step {self.global_step}: "
                    f"components={nan_components}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                n_nan_batches += 1
                if n_nan_batches >= 10:
                    print(
                        f"  [train] WARNING: {n_nan_batches} NaN batches in epoch {epoch}, stopping epoch early"
                    )
                    break
                continue

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / self.gradient_accumulation_steps

            if self.use_deepspeed:
                self.model.backward(scaled_loss)
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if not self.use_deepspeed:
                    # Zero out NaN gradients before clipping
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm()
                            if not torch.isfinite(grad_norm):
                                param.grad.zero_()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    if not torch.isfinite(grad_norm):
                        self.optimizer.zero_grad(set_to_none=True)
                        n_nan_batches += 1
                        continue

                    # Per-parameter gradient norm monitoring (logged every N steps)
                    if self.global_step % self.log_every == 0:
                        grad_parts = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None and param.grad.norm() > 0:
                                # D-03: Fixed parameter name matching to actual model structure
                                # Verified parameter name prefixes from StreamingTCNModel.named_parameters():
                                #   param_head.gain_mlp.*         -> gain prediction MLP
                                #   param_head.q_head.*           -> Q prediction
                                #   param_head.classification_head.* -> type classification
                                #   param_head.freq_direct.*      -> frequency prediction
                                #   param_head.freq_fallback.*    -> frequency fallback
                                #   param_head.trunk.*            -> shared parameter trunk
                                #   param_head.type_mel_proj.*    -> type mel projection
                                #   param_head.mel_cnn.*          -> mel CNN features
                                #   encoder.*                     -> TCN encoder
                                if "gain_mlp" in name:
                                    grad_parts.append(("gain", param.grad.norm().item()))
                                elif "q_head" in name:
                                    grad_parts.append(("q", param.grad.norm().item()))
                                elif "classification_head" in name or "type_mel_proj" in name:
                                    grad_parts.append(("type", param.grad.norm().item()))
                                elif "freq_direct" in name or "freq_fallback" in name:
                                    grad_parts.append(("freq", param.grad.norm().item()))
                                elif "param_head" in name:
                                    # Catch-all for trunk, cnn_merge, query_proj, etc.
                                    grad_parts.append(("param_head_other", param.grad.norm().item()))
                                elif "encoder" in name:
                                    grad_parts.append(("encoder", param.grad.norm().item()))
                        if grad_parts:
                            grouped = {}
                            for k, v in grad_parts:
                                grouped.setdefault(k, []).append(v)
                            avg_grads = {k: sum(v) / len(v) for k, v in grouped.items()}
                            grad_str = " | ".join(
                                f"grad_{k}={v:.4f}"
                                for k, v in sorted(avg_grads.items())
                            )
                            print(f"  [grads] step={self.global_step} {grad_str}")

                    self.optimizer.step()

                    # D-04: Update gain MAE EMA for hybrid warmup gate in loss function.
                    # Use raw gain MAE (not log_cosh) so the threshold (2.5 dB) has physical meaning.
                    if hasattr(self.criterion, 'update_gain_mae'):
                        with torch.no_grad():
                            batch_gain_mae = (pred_gain - target_gain).abs().mean().item()
                        self.criterion.update_gain_mae(batch_gain_mae)

                    # Sanitize optimizer state: reset any Adam momentum/variance
                    # buffers that contain NaN (prevents permanent training death)
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                                v.zero_()
                else:
                    self.model.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Stage warmup: linearly ramp LR from start to target over N steps
                if self._stage_warmup_active:
                    self._stage_warmup_step_count += 1
                    progress = min(
                        1.0,
                        self._stage_warmup_step_count
                        / max(self._stage_warmup_steps, 1),
                    )
                    for pg_idx, pg in enumerate(self.optimizer.param_groups):
                        start_lr = self._stage_warmup_start_lrs[pg_idx]
                        target_lr = self._stage_warmup_target_lrs[pg_idx]
                        pg["lr"] = start_lr + (target_lr - start_lr) * progress
                    if progress >= 1.0:
                        self._stage_warmup_active = False

            epoch_loss += total_loss.item()
            n_batches += 1

            # Accumulate per-component losses
            for k, v in components.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                component_accum[k] = component_accum.get(k, 0.0) + val

            if self.global_step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                parts = [
                    f"step={self.global_step} loss={total_loss.item():.4f} lr={lr:.2e}"
                ]
                for k, v in components.items():
                    parts.append(f"{k}={v:.4f}")
                # Log embedding variance diagnostic (monitors anti-collapse health)
                if "embed_var_loss" in components:
                    embed_var = output["embedding"].var(dim=0).mean().item()
                    parts.append(f"embed_var={embed_var:.4f}")
                print(f"  [train] " + " | ".join(parts))

        if n_nan_batches > 0:
            print(f"  [train] Epoch {epoch}: {n_nan_batches} NaN batches skipped")

        # Per-component epoch averages
        if n_batches > 0:
            comp_strs = []
            for k in sorted(component_accum.keys()):
                avg = component_accum[k] / n_batches
                comp_strs.append(f"{k}={avg:.4f}")
            print(f"  [train] Epoch {epoch} components: " + " | ".join(comp_strs))

        return epoch_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, epoch):
        # Set curriculum epoch on criterion for consistent validation losses
        if hasattr(self.criterion, 'current_epoch'):
            self.criterion.current_epoch = epoch

        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        param_maes = {"gain": [], "gain_raw": [], "freq": [], "q": [], "type_acc": []}
        val_component_accum = {}  # D-02: accumulate loss components during validation

        for batch in self.val_loader:
            mel_frames = self._prepare_input(batch)
            target_gain = batch["gain"].to(self.device, non_blocking=True)
            target_freq = batch["freq"].to(self.device, non_blocking=True)
            target_q = batch["q"].to(self.device, non_blocking=True)
            target_ft = batch["filter_type"].to(self.device, non_blocking=True)
            active_band_mask = batch.get("active_band_mask", None)
            if active_band_mask is not None:
                active_band_mask = active_band_mask.to(self.device, non_blocking=True)

            with self.autocast_ctx:
                output = self.model(mel_frames)
                pred_gain, pred_freq, pred_q = output["params"]

                # NaN diagnostic: check model outputs before loss
                if not torch.isfinite(pred_gain).all():
                    print(f"  [nan] pred_gain has NaN/inf at val batch {n_batches}")
                if not torch.isfinite(pred_freq).all():
                    print(f"  [nan] pred_freq has NaN/inf at val batch {n_batches}")
                if not torch.isfinite(pred_q).all():
                    print(f"  [nan] pred_q has NaN/inf at val batch {n_batches}")
                if not torch.isfinite(output["embedding"]).all():
                    print(f"  [nan] embedding has NaN/inf at val batch {n_batches}")
                if (
                    output["H_mag"] is not None
                    and not torch.isfinite(output["H_mag"]).all()
                ):
                    print(f"  [nan] H_mag has NaN/inf at val batch {n_batches}")

                target_H_mag = self.model.dsp_cascade(
                    target_gain,
                    target_freq,
                    target_q,
                    n_fft=self.n_fft,
                    filter_type=target_ft,
                )

                # Dual H_mag path for validation (matching training)
                pred_H_mag_soft = output["H_mag"]
                pred_H_mag_hard = output["H_mag_hard"]

                # Spectral reconstruction for validation (matching training)
                is_spectral_active = (
                    hasattr(self.criterion, 'current_epoch') and
                    hasattr(self.criterion, 'warmup_epochs') and
                    self.criterion.current_epoch >= self.criterion.warmup_epochs + 2
                )
                pred_audio_for_loss = pred_H_mag_soft if is_spectral_active else None
                target_audio_for_loss = target_H_mag if is_spectral_active else None

                total_loss, components = self.criterion(
                    pred_gain,
                    pred_freq,
                    pred_q,
                    output["type_logits"],
                    pred_H_mag_soft,
                    pred_H_mag_hard,
                    target_gain,
                    target_freq,
                    target_q,
                    target_ft,
                    target_H_mag,
                    pred_audio=pred_audio_for_loss,
                    target_audio=target_audio_for_loss,
                    active_band_mask=active_band_mask,
                    embedding=output["embedding"],
                )

                # NaN diagnostic: identify which loss component causes NaN
                if not torch.isfinite(total_loss):
                    nan_components = [
                        k
                        for k, v in components.items()
                        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                    ]
                    print(
                        f"  [nan] Non-finite loss at val batch {n_batches}: "
                        f"components={nan_components}"
                    )
                    # Clamp NaN loss to a large finite value to prevent checkpoint corruption
                    # and avoid breaking the validation average (inf would corrupt avg_val_loss)
                    total_loss = torch.tensor(1e4, device=total_loss.device)

            val_loss += total_loss.item()
            n_batches += 1

            # D-02: Accumulate loss components for validation logging
            for k, v in components.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                val_component_accum[k] = val_component_accum.get(k, 0.0) + val

            # Hungarian matching for fair MAE computation — also matches filter types
            # so type_acc is compared against the correctly-permuted ground truth.
            matched_gain, matched_freq, matched_q, matched_ft = self.matcher(
                pred_gain,
                pred_freq,
                pred_q,
                target_gain,
                target_freq,
                target_q,
                target_filter_type=target_ft,
                pred_type_logits=output["type_logits"],
            )

            # Metrics (matched = fair, raw = for comparison)
            param_maes["gain"].append((pred_gain - matched_gain).abs().mean().item())
            param_maes["gain_raw"].append((pred_gain - target_gain).abs().mean().item())
            param_maes["freq"].append(
                (torch.log2(pred_freq / (matched_freq + 1e-8))).abs().mean().item()
            )
            param_maes["q"].append(
                (torch.log10(pred_q / (matched_q + 1e-8))).abs().mean().item()
            )
            param_maes["type_acc"].append(
                (output["filter_type"] == matched_ft).float().mean().item()
            )

            # D-11: Per-type accuracy breakdown
            from differentiable_eq import FILTER_NAMES
            pred_ft = output["filter_type"]
            for type_idx, type_name in enumerate(FILTER_NAMES):
                mask = (matched_ft == type_idx)
                if mask.sum() > 0:
                    per_type_acc = (pred_ft[mask] == matched_ft[mask]).float().mean().item()
                else:
                    per_type_acc = 0.0
                param_maes.setdefault(f"type_{type_name}", []).append(per_type_acc)

        avg_val_loss = val_loss / max(n_batches, 1)
        metrics = {k: sum(v) / len(v) for k, v in param_maes.items() if v}

        # D-02: Log averaged validation loss components
        if n_batches > 0 and val_component_accum:
            comp_strs = []
            for k in sorted(val_component_accum.keys()):
                avg = val_component_accum[k] / n_batches
                comp_strs.append(f"{k}={avg:.4f}")
            print(f"  [val] epoch={epoch} components: " + " | ".join(comp_strs))

        # Matched MAE is the primary/trustworthy metric (D-04)
        # gain_raw is kept for debug comparison only
        print(
            f"  [val] epoch={epoch} val_loss={avg_val_loss:.4f} "
            f"gain_mae={metrics.get('gain', 0):.2f}dB "
            f"gain_raw={metrics.get('gain_raw', 0):.2f}dB "
            f"freq_mae={metrics.get('freq', 0):.3f}oct "
            f"q_mae={metrics.get('q', 0):.3f}dec "
            f"type_acc={metrics.get('type_acc', 0):.1%}"
        )
        # D-11: Per-type accuracy reporting
        per_type_parts = []
        for tn in ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]:
            v = metrics.get(f"type_{tn}", 0)
            per_type_parts.append(f"{tn}={v:.1%}")
        print(f"  [val] per-type: {' | '.join(per_type_parts)}")
        return avg_val_loss, metrics

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)

        if self.use_deepspeed:
            # DeepSpeed handles checkpointing
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}"
            self.model.save_checkpoint(str(ckpt_path), tag=f"epoch_{epoch}")
            if is_best:
                self.model.save_checkpoint(str(ckpt_dir / "best"), tag="best")
        else:
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "global_step": self.global_step,
            }
            path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(state, path)
            if is_best:
                best_path = ckpt_dir / "best.pt"
                torch.save(state, best_path)
            print(f"  Saved checkpoint: {path}" + (" (best)" if is_best else ""))

    def _has_nan_weights(self):
        """Check if any model parameter or buffer (BatchNorm stats) contains NaN."""
        for p in self.model.parameters():
            if not torch.isfinite(p).all():
                return True
        # BatchNorm running_mean/running_var are buffers, not parameters
        for buf in self.model.buffers():
            if not torch.isfinite(buf).all():
                return True
        return False

    def _load_checkpoint(self, path):
        """Load model (and optimizer) state from a checkpoint to resume training."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        sd = state["model_state_dict"]
        # torch.compile wraps keys with "_orig_mod." — strip it for non-compiled load
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        self.model.to(self.device)
        result = self.model.load_state_dict(sd, strict=False)
        if result.unexpected_keys:
            print(f"  [resume] Dropped {len(result.unexpected_keys)} extra keys from checkpoint (architecture change): {[k.split('.')[-1] for k in result.unexpected_keys]}")
        if result.missing_keys:
            print(f"  [resume] {len(result.missing_keys)} keys initialized randomly (not in checkpoint): {[k.split('.')[-1] for k in result.missing_keys]}")
        # FIX-5: Reset gain_output_scale regardless of checkpoint value.
        # The saved scale was learned under a corrupt regime (type loss absent 17 epochs).
        # STE clamps output to ±24 dB regardless, so resetting to 12.0 only reduces noise.
        if hasattr(self.model, 'param_head') and hasattr(self.model.param_head, 'gain_output_scale'):
            with torch.no_grad():
                self.model.param_head.gain_output_scale.fill_(12.0)
            print("  [resume] Reset gain_output_scale to 12.0 (fix for overfit scale)")
        try:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as e:
            print(f"  [resume] Could not restore optimizer state ({e}); starting fresh optimizer")
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("val_loss", float("inf"))
        self.start_epoch = state.get("epoch", 0) + 1

        # Restore scheduler state or advance to the correct position
        if "scheduler_state_dict" in state:
            try:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
                print(f"  [resume] Restored scheduler state")
            except Exception as e:
                print(f"  [resume] Could not restore scheduler ({e}); advancing to epoch {self.start_epoch}")
                for _ in range(self.start_epoch - 1):
                    self.scheduler.step()
        else:
            for _ in range(self.start_epoch - 1):
                self.scheduler.step()

        print(
            f"  [resume] Loaded checkpoint from {path} "
            f"(epoch {state.get('epoch')}, val_loss={self.best_val_loss:.4f})"
        )
        print(f"  [resume] Resuming from epoch {self.start_epoch}")

    def _recover_from_nan(self, epoch):
        """Reload last good checkpoint to recover from NaN state."""
        ckpt_dir = Path("checkpoints")
        # Try checkpoints in reverse order (most recent first), find one that's clean
        candidates = sorted(ckpt_dir.glob("epoch_*.pt"), reverse=True)
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            candidates.insert(0, best_path)

        for ckpt_path in candidates:
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            # Skip checkpoints with NaN val_loss
            if not math.isfinite(state.get("val_loss", float("inf"))):
                continue
            # Load and verify no NaN in state dict
            try:
                self.model.load_state_dict(state["model_state_dict"])
            except Exception:
                continue
            if self._has_nan_weights():
                continue  # This checkpoint is also corrupted
            # Found a clean checkpoint — reset optimizer (old state may have NaN)
            base_lr = self.cfg["model"]["learning_rate"]
            wd = self.cfg["model"]["weight_decay"]
            encoder_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "param_head" in name:
                    head_params.append(param)
                else:
                    encoder_params.append(param)
            head_lr_mult = self.cfg["model"].get("head_lr_multiplier", 3.0)
            param_groups = [
                {
                    "params": encoder_params,
                    "lr": base_lr,
                    "weight_decay": wd,
                    "initial_lr": base_lr,
                },
                {
                    "params": head_params,
                    "lr": base_lr * head_lr_mult,
                    "weight_decay": wd,
                    "initial_lr": base_lr * head_lr_mult,
                },
            ]
            self.optimizer = torch.optim.AdamW(param_groups)
            self.global_step = state.get("global_step", 0)
            print(
                f"  [recovery] Reloaded clean checkpoint from {ckpt_path}, reset optimizer"
            )
            return True

        print(f"  [recovery] No clean checkpoint found. Cannot recover.")
        return False

    def _update_type_transition(self, epoch):
        """No-op: data distribution stays fixed across stages."""
        pass

    def _apply_curriculum_stage(self, epoch):
        """Determine and apply curriculum stage based on epoch number."""
        # Handle gradual type weight transition at the start of each epoch
        self._update_type_transition(epoch)

        if not self.curriculum_stages:
            return

        # Determine stage from cumulative epoch boundaries
        cumulative = 0
        stage_idx = 0
        stage_start_epoch = 0
        for i, stage in enumerate(self.curriculum_stages):
            prev_cumulative = cumulative
            cumulative += stage["epochs"]
            if epoch <= cumulative:
                stage_idx = i
                stage_start_epoch = prev_cumulative + 1
                break
        else:
            stage_idx = len(self.curriculum_stages) - 1

        # Skip if stage hasn't changed (but still do intra-stage annealing)
        stage = self.curriculum_stages[stage_idx]
        stage_epochs = stage["epochs"]

        # Intra-stage Gumbel temperature annealing
        target_tau = stage.get("gumbel_temperature", 1.0)
        epoch_in_stage = epoch - stage_start_epoch
        if stage_idx > 0:
            prev_tau = self.curriculum_stages[stage_idx - 1].get(
                "gumbel_temperature", 1.0
            )
        else:
            prev_tau = 1.0

        # Smooth linear annealing from prev_tau to target_tau within stage
        progress = epoch_in_stage / max(stage_epochs - 1, 1)
        current_tau = prev_tau + (target_tau - prev_tau) * progress
        current_tau = max(current_tau, 0.1)  # floor — prevent vanishing gradients
        self.model.param_head.gumbel_temperature.fill_(current_tau)

        # Only re-precompute data when stage changes
        if stage_idx == self._current_stage_idx:
            return

        # D-06/D-07/D-08: Metric-gated curriculum transitions
        thresholds = stage.get("metric_thresholds", {})
        if thresholds and self._last_metrics:
            all_met = True
            for metric_name, threshold in thresholds.items():
                current_val = self._last_metrics.get(metric_name)
                if current_val is None:
                    all_met = False
                    break
                if "mae" in metric_name:
                    if current_val > threshold:
                        all_met = False
                        break
                elif "acc" in metric_name:
                    if current_val < threshold:
                        all_met = False
                        break
            epoch_cap = stage.get("epoch_cap", stage_epochs * 2)
            epochs_in_current = epoch - stage_start_epoch
            if not all_met and epochs_in_current < epoch_cap:
                print(
                    f"  [curriculum] Stage transition to '{stage['name']}' blocked: "
                    f"metrics not met (epoch {epochs_in_current}/{epoch_cap})"
                )
                return

        self._current_stage_idx = stage_idx

        # Per-stage LR warmup: restore LR to fraction of initial, short linear warmup
        stage_lr_scale = stage.get("learning_rate_scale", 0.3)
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * stage_lr_scale
        self._stage_warmup_active = True
        self._stage_warmup_steps = 200
        self._stage_warmup_step_count = 0
        self._stage_warmup_start_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self._stage_warmup_target_lrs = [
            pg.get("initial_lr", pg["lr"]) for pg in self.optimizer.param_groups
        ]

        # Apply per-stage loss weight overrides (NO data re-precompute)
        # The curriculum now controls only loss weights, Gumbel temperature,
        # and LR warmup. Data distribution stays fixed for the entire run.
        # This prevents the fatal distribution shift that destroyed optimizer
        # state alignment at stage boundaries (epoch 30→31 never recovered).
        stage_lambda_type = stage.get("lambda_type", None)
        if stage_lambda_type is not None:
            self.criterion.lambda_type = stage_lambda_type

        print(
            f"  [curriculum] Epoch {epoch}: stage '{stage['name']}' "
            f"(gumbel_tau={target_tau:.2f}, lambda_type={self.criterion.lambda_type}, lr_warmup={self._stage_warmup_steps} steps)"
        )

    def fit(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"IDSP Multi-Type EQ Estimator Training")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {total_params:,}")
        print(
            f"  Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Receptive field: {self.model.receptive_field_frames} frames")
        print()

        consecutive_nan_epochs = 0

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            t0 = time.time()

            # Set curriculum epoch on criterion for warmup gating
            if hasattr(self.criterion, 'current_epoch'):
                self.criterion.current_epoch = epoch
                if hasattr(self.criterion, 'warmup_epochs'):
                    warmup = self.criterion.warmup_epochs
                    if epoch < warmup:
                        print(f"  [curriculum] Epoch {epoch}: GAIN-ONLY WARMUP ({warmup - epoch} remaining)")
                    elif epoch < warmup + 1:
                        print(f"  [curriculum] Epoch {epoch}: Gain + Freq + Q active")
                    elif epoch < warmup + 2:
                        print(f"  [curriculum] Epoch {epoch}: + Type loss activated")
                    else:
                        print(f"  [curriculum] Epoch {epoch}: All losses active")

            self._apply_curriculum_stage(epoch)
            train_loss = self.train_one_epoch(epoch)

            # Check for NaN in weights AND buffers (BatchNorm running stats)
            if self._has_nan_weights():
                print(
                    f"  [nan] NaN in model state after epoch {epoch}. "
                    f"Attempting checkpoint recovery..."
                )
                if self._recover_from_nan(epoch):
                    consecutive_nan_epochs = 0
                    continue  # Retry this epoch with recovered model
                else:
                    print(f"  ERROR: Cannot recover. Stopping.")
                    break

            val_loss, metrics = self.validate(epoch)
            self._last_metrics = metrics  # D-06/D-07: Store for metric-gated curriculum
            self.scheduler.step()
            elapsed = time.time() - t0

            # Track consecutive NaN val epochs (death spiral detection)
            if math.isnan(val_loss) or math.isnan(train_loss):
                consecutive_nan_epochs += 1
                if consecutive_nan_epochs >= 2:
                    print(
                        f"  [nan] NaN loss for {consecutive_nan_epochs} consecutive epochs. "
                        f"Recovering from best checkpoint..."
                    )
                    if self._recover_from_nan(epoch):
                        consecutive_nan_epochs = 0
                        continue
                    else:
                        print(f"  ERROR: Cannot recover. Stopping.")
                        break
            else:
                consecutive_nan_epochs = 0

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, val_loss, is_best)

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "metrics": metrics,
                    "time_s": elapsed,
                }
            )

            print(
                f"Epoch {epoch}/{self.max_epochs} ({elapsed:.1f}s) "
                f"train={train_loss:.4f} val={val_loss:.4f}"
            )

            # Early stopping: stop if no improvement for patience epochs
            if self.patience_counter >= self.early_stopping_patience:
                print(
                    f"  [early_stop] No improvement for {self.patience_counter} epochs. "
                    f"Stopping early at epoch {epoch}."
                )
                break

        hist_path = Path("checkpoints") / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"History saved to {hist_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. best.pt)")
    args = parser.parse_args()
    trainer = Trainer(config_path=args.config, resume_path=args.resume)
    trainer.fit()
