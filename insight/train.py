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
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from model_tcn import StreamingTCNModel
from dsp_frontend import STFTFrontend
from differentiable_eq import FILTER_HIGHSHELF, FILTER_LOWSHELF
from loss_v2 import SimplifiedEQLoss
from loss_multitype import HungarianBandMatcher
from metrics import compute_eq_metrics
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
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    lambda_param = loss_cfg.get("lambda_param", 1.0)
    lambda_spectral = loss_cfg.get("lambda_spectral", 5.0)
    hp_lp_gain_target = data_cfg.get("hp_lp_gain_target", "zero")
    if hp_lp_gain_target != "zero":
        raise ValueError(
            "Only `data.hp_lp_gain_target: zero` is supported for the "
            "multi-type EQ label contract."
        )
    if lambda_param <= 0.0 and lambda_spectral <= 0.0:
        raise ValueError(
            "SimplifiedEQLoss requires at least one of "
            "`loss.lambda_param` or `loss.lambda_spectral` to be > 0."
        )


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
        self.monitor_val_metric = trainer_cfg.get("monitor_val_metric", "val_loss_soft")

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
            duration_range=tuple(data_cfg["duration_range"])
            if data_cfg.get("duration_range") is not None
            else None,
            n_fft=self.n_fft,
            size=dataset_size,
            gain_range=tuple(data_cfg["gain_bounds"]),
            freq_range=tuple(data_cfg["freq_bounds"]),
            q_range=tuple(data_cfg["q_bounds"]),
            type_weights=data_cfg.get("type_weights", None),
            hp_lp_gain_target=data_cfg.get("hp_lp_gain_target", "zero"),
            precompute_mels=data_cfg.get("precompute_mels", True),
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
        elif data_cfg.get("precompute_mels", True):
            # Precompute all samples with mel-spectrograms upfront
            self.train_dataset.precompute()
            # Save cache if path specified
            if self.precompute_cache_path:
                self.train_dataset.save_precomputed(self.precompute_cache_path)
        else:
            print(f"  [data] On-the-fly generation (no precompute)")

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
        if data_cfg.get("precompute_mels", True) or data_cfg.get(
            "freeze_val_set", True
        ):
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
            type_conditioned_frequency=model_cfg.get(
                "type_conditioned_frequency", True
            ),
            dropout=enc_cfg.get("dropout", 0.1),
            mel_noise_std=enc_cfg.get("mel_noise_std", 0.0),
            n_shelf_bands=model_cfg.get("n_shelf_bands", 16),
        ).to(self.device)

        # STFT frontend
        self.frontend = STFTFrontend(
            n_fft=self.n_fft,
            hop_length=data_cfg.get("hop_length", 256),
            win_length=self.n_fft,
            mel_bins=data_cfg.get("n_mels", 128),
            sample_rate=self.sample_rate,
        ).to(self.device)

        # Loss — simplified 3-term (spectral-dominant, ICASSP 2024 BE-AFX)
        self.criterion = SimplifiedEQLoss(
            lambda_spectral=loss_cfg.get("lambda_spectral", 5.0),
            lambda_param=loss_cfg.get("lambda_param", 1.0),
            lambda_type=loss_cfg.get("lambda_type", 2.0),
            lambda_gain=loss_cfg.get("lambda_gain", 1.0),
            lambda_freq=loss_cfg.get("lambda_freq", 1.0),
            lambda_q=loss_cfg.get("lambda_q", 1.0),
            lambda_gain_zero=loss_cfg.get("lambda_gain_zero", 1.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.02),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            type_class_priors=data_cfg.get("type_weights", None),
            lambda_type_match=loss_cfg.get("lambda_type_match", 0.0),
            lambda_type_prior=loss_cfg.get("lambda_type_prior", 0.0),
            lambda_embed_var=loss_cfg.get("lambda_embed_var", 0.0),
            lambda_contrastive=loss_cfg.get("lambda_contrastive", 0.0),
            embed_var_threshold=loss_cfg.get("embed_var_threshold", 0.1),
            class_weight_multipliers=self.cfg.get("loss", {}).get("class_weight_multipliers", None),
            sign_penalty_weight=self.cfg.get("loss", {}).get("sign_penalty_weight", 0.0),
            lambda_hdb=self.cfg.get("loss", {}).get("lambda_hdb", 2.0),
        )
        self.metric_lambda_type_match = loss_cfg.get("lambda_type_match", 0.0)

        # Hungarian matcher for fair validation metrics (matches target ordering to predictions)
        self.matcher = HungarianBandMatcher(
            lambda_gain=1.0,
            lambda_freq=1.0,
            lambda_q=1.0,
            lambda_type_match=loss_cfg.get("lambda_type_match", 0.0),
        )

        # Optimizer: the parameter head runs at a higher LR than the encoder
        # because the spectral trunk is deeper and adapts more slowly.
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
        # CosineAnnealing LR schedule — steps once per epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=1e-6
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

    def _gain_aux_alignment(self, batch, wet_mel, output):
        gain_aux_summary = output.get("gain_aux_summary")
        if "dry_audio" not in batch or gain_aux_summary is None:
            return None
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        dry_audio = batch["dry_audio"].to(self.device, non_blocking=True)
        with torch.no_grad():
            dry_mel = self.frontend.mel_spectrogram(dry_audio).squeeze(1).float()
            mel_delta = (wet_mel.detach().float() - dry_mel).mean(dim=-1)
            gain_delta_summary = model_ref.param_head.summarize_gain_aux_features(
                mel_delta
            )
            if gain_delta_summary is None:
                return None
            lhs = F.normalize(gain_aux_summary.detach().float(), dim=-1)
            rhs = F.normalize(gain_delta_summary.detach().float(), dim=-1)
            alignment = F.cosine_similarity(lhs, rhs, dim=-1).mean()
            if torch.isfinite(alignment):
                return alignment.item()
        return None

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_nan_batches = 0

        # Accumulate per-component losses for epoch-level logging
        component_accum = {}

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            mel_frames = self._prepare_input(batch)
            target_gain = batch["gain"].to(self.device, non_blocking=True)
            target_freq = batch["freq"].to(self.device, non_blocking=True)
            target_q = batch["q"].to(self.device, non_blocking=True)
            target_ft = batch["filter_type"].to(self.device, non_blocking=True)

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

                # H_db target for direct prediction loss (hybrid spectral-parametric)
                h_db_target = 20.0 * torch.log10(target_H_mag.clamp(min=1e-6))
                h_db_pred = output.get("h_db_pred")

                # Use soft H_mag (differentiable w.r.t. type_logits) for spectral loss
                total_loss, components = self.criterion(
                    pred_gain,
                    pred_freq,
                    pred_q,
                    output["type_logits"],
                    output["H_mag_soft"],
                    target_gain,
                    target_freq,
                    target_q,
                    target_ft,
                    target_H_mag,
                    embedding=output["embedding"],
                    h_db_pred=h_db_pred,
                    h_db_target=h_db_target,
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
                                if "gain_head" in name:
                                    grad_parts.append(("gain", param.grad.norm().item()))
                                elif "q_head" in name:
                                    grad_parts.append(("q", param.grad.norm().item()))
                                elif "type_head" in name or "type_mel_proj" in name:
                                    grad_parts.append(("type", param.grad.norm().item()))
                                elif "freq_head" in name or "freq_context_proj" in name:
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
                gain_aux_alignment = self._gain_aux_alignment(batch, mel_frames, output)
                if gain_aux_alignment is not None:
                    parts.append(f"gain_aux_delta_cos={gain_aux_alignment:.4f}")
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
        self.model.eval()
        val_loss_soft = 0.0
        val_loss_hard = 0.0
        n_batches = 0
        metric_tensors = {
            "gain_pred": [],
            "gain_gt": [],
            "freq_pred": [],
            "freq_gt": [],
            "q_pred": [],
            "q_gt": [],
            "type_pred": [],
            "type_gt": [],
            "type_logits": [],
        }
        val_component_accum = {}
        shelf_diag = {
            "pred_lowshelf_count": 0,
            "pred_highshelf_count": 0,
            "low_shelf_bias_sum": 0.0,
            "low_shelf_bias_count": 0,
            "high_shelf_bias_sum": 0.0,
            "high_shelf_bias_count": 0,
        }

        for batch in self.val_loader:
            mel_frames = self._prepare_input(batch)
            target_gain = batch["gain"].to(self.device, non_blocking=True)
            target_freq = batch["freq"].to(self.device, non_blocking=True)
            target_q = batch["q"].to(self.device, non_blocking=True)
            target_ft = batch["filter_type"].to(self.device, non_blocking=True)

            with self.autocast_ctx:
                output_hard = self.model(mel_frames)
                output_soft = self.model(
                    mel_frames,
                    hard_types=False,
                    force_soft_response=True,
                )
                pred_gain, pred_freq, pred_q = output_hard["params"]

                # NaN diagnostic
                for name, tensor in [
                    ("pred_gain", pred_gain),
                    ("pred_freq", pred_freq),
                    ("pred_q", pred_q),
                    ("embedding", output_hard["embedding"]),
                    ("H_mag", output_hard["H_mag"]),
                    ("H_mag_soft", output_soft["H_mag_soft"]),
                ]:
                    if not torch.isfinite(tensor).all():
                        print(f"  [nan] {name} has NaN/inf at val batch {n_batches}")

                model_ref = self.model.module if hasattr(self.model, "module") else self.model
                target_H_mag = model_ref.dsp_cascade(
                    target_gain,
                    target_freq,
                    target_q,
                    n_fft=self.n_fft,
                    filter_type=target_ft,
                )

                total_loss_soft, components = self.criterion(
                    output_soft["params"][0],
                    output_soft["params"][1],
                    output_soft["params"][2],
                    output_soft["type_logits"],
                    output_soft["H_mag_soft"],
                    target_gain,
                    target_freq,
                    target_q,
                    target_ft,
                    target_H_mag,
                    embedding=output_soft["embedding"],
                )
                total_loss_hard, _ = self.criterion(
                    output_hard["params"][0],
                    output_hard["params"][1],
                    output_hard["params"][2],
                    output_hard["type_logits"],
                    output_hard["H_mag"],
                    target_gain,
                    target_freq,
                    target_q,
                    target_ft,
                    target_H_mag,
                    embedding=output_hard["embedding"],
                )

                # NaN diagnostic: identify which loss component causes NaN
                if not torch.isfinite(total_loss_soft):
                    nan_components = [
                        k
                        for k, v in components.items()
                        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                    ]
                    print(
                        f"  [nan] Non-finite loss at val batch {n_batches}: "
                        f"components={nan_components}"
                    )
                    total_loss_soft = torch.tensor(1e4, device=pred_gain.device)
                if not torch.isfinite(total_loss_hard):
                    total_loss_hard = torch.tensor(1e4, device=pred_gain.device)

            val_loss_soft += total_loss_soft.item()
            val_loss_hard += total_loss_hard.item()
            n_batches += 1

            # D-02: Accumulate loss components for validation logging
            for k, v in components.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                val_component_accum[k] = val_component_accum.get(k, 0.0) + val

            matched_filter_type = self.matcher(
                pred_gain,
                pred_freq,
                pred_q,
                target_gain,
                target_freq,
                target_q,
                target_filter_type=target_ft,
                pred_type_logits=output_hard["type_logits"],
            )[3]
            pred_filter_type = output_hard["filter_type"]
            shelf_diag["pred_lowshelf_count"] += int(
                (pred_filter_type == FILTER_LOWSHELF).sum().item()
            )
            shelf_diag["pred_highshelf_count"] += int(
                (pred_filter_type == FILTER_HIGHSHELF).sum().item()
            )
            shelf_bias = output_hard.get("shelf_bias")
            if shelf_bias is not None:
                low_mask = matched_filter_type == FILTER_LOWSHELF
                high_mask = matched_filter_type == FILTER_HIGHSHELF
                if low_mask.any():
                    shelf_diag["low_shelf_bias_sum"] += (
                        shelf_bias[..., 0][low_mask].float().sum().item()
                    )
                    shelf_diag["low_shelf_bias_count"] += int(low_mask.sum().item())
                if high_mask.any():
                    shelf_diag["high_shelf_bias_sum"] += (
                        shelf_bias[..., 1][high_mask].float().sum().item()
                    )
                    shelf_diag["high_shelf_bias_count"] += int(high_mask.sum().item())

            metric_tensors["gain_pred"].append(pred_gain.detach().float().cpu())
            metric_tensors["gain_gt"].append(target_gain.detach().float().cpu())
            metric_tensors["freq_pred"].append(pred_freq.detach().float().cpu())
            metric_tensors["freq_gt"].append(target_freq.detach().float().cpu())
            metric_tensors["q_pred"].append(pred_q.detach().float().cpu())
            metric_tensors["q_gt"].append(target_q.detach().float().cpu())
            metric_tensors["type_pred"].append(output_hard["filter_type"].detach().cpu())
            metric_tensors["type_gt"].append(target_ft.detach().cpu())
            metric_tensors["type_logits"].append(
                output_hard["type_logits"].detach().float().cpu()
            )

        avg_val_loss_soft = val_loss_soft / max(n_batches, 1)
        avg_val_loss_hard = val_loss_hard / max(n_batches, 1)
        if metric_tensors["gain_pred"]:
            metrics = compute_eq_metrics(
                torch.cat(metric_tensors["gain_pred"]),
                torch.cat(metric_tensors["freq_pred"]),
                torch.cat(metric_tensors["q_pred"]),
                torch.cat(metric_tensors["type_pred"]),
                torch.cat(metric_tensors["type_logits"]),
                torch.cat(metric_tensors["gain_gt"]),
                torch.cat(metric_tensors["freq_gt"]),
                torch.cat(metric_tensors["q_gt"]),
                torch.cat(metric_tensors["type_gt"]),
                lambda_type_match=self.metric_lambda_type_match,
            )
        else:
            metrics = {}
        metrics["val_loss_soft"] = avg_val_loss_soft
        metrics["val_loss_hard"] = avg_val_loss_hard
        metrics["val_loss"] = avg_val_loss_soft
        metrics["pred_lowshelf_count"] = shelf_diag["pred_lowshelf_count"]
        metrics["pred_highshelf_count"] = shelf_diag["pred_highshelf_count"]
        metrics["mean_low_shelf_bias_on_lowshelf"] = (
            shelf_diag["low_shelf_bias_sum"] / shelf_diag["low_shelf_bias_count"]
            if shelf_diag["low_shelf_bias_count"] > 0
            else 0.0
        )
        metrics["mean_high_shelf_bias_on_highshelf"] = (
            shelf_diag["high_shelf_bias_sum"] / shelf_diag["high_shelf_bias_count"]
            if shelf_diag["high_shelf_bias_count"] > 0
            else 0.0
        )

        # D-02: Log averaged validation loss components
        if n_batches > 0 and val_component_accum:
            comp_strs = []
            for k in sorted(val_component_accum.keys()):
                avg = val_component_accum[k] / n_batches
                comp_strs.append(f"{k}={avg:.4f}")
            print(f"  [val] epoch={epoch} components: " + " | ".join(comp_strs))

        print(
            f"  [val] epoch={epoch} val_loss_soft={avg_val_loss_soft:.4f} "
            f"val_loss_hard={avg_val_loss_hard:.4f} "
            f"gain_mae_db_matched={metrics.get('gain_mae_db_matched', 0):.2f} "
            f"gain_mae_db_raw={metrics.get('gain_mae_db_raw', 0):.2f} "
            f"freq_mae_oct_matched={metrics.get('freq_mae_oct_matched', 0):.3f} "
            f"freq_mae_oct_raw={metrics.get('freq_mae_oct_raw', 0):.3f} "
            f"q_mae_dec_matched={metrics.get('q_mae_dec_matched', 0):.3f} "
            f"type_accuracy_matched={metrics.get('type_accuracy_matched', 0):.1%} "
            f"type_accuracy_raw={metrics.get('type_accuracy_raw', 0):.1%}"
        )
        # D-11: Per-type accuracy reporting
        per_type_parts = []
        for tn in ["peaking", "lowshelf", "highshelf", "highpass", "lowpass"]:
            v = metrics.get(f"type_accuracy_{tn}_matched", 0)
            per_type_parts.append(f"{tn}={v:.1%}")
        print(f"  [val] per-type: {' | '.join(per_type_parts)}")
        print(
            f"  [val] shelf: pred_lowshelf={metrics['pred_lowshelf_count']} "
            f"pred_highshelf={metrics['pred_highshelf_count']} "
            f"low_bias_on_lowshelf={metrics['mean_low_shelf_bias_on_lowshelf']:.3f} "
            f"high_bias_on_highshelf={metrics['mean_high_shelf_bias_on_highshelf']:.3f}"
        )
        return avg_val_loss_soft, metrics

    def save_checkpoint(self, epoch, val_loss, is_best=False, metrics=None):
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
            if metrics is not None:
                state["val_loss_soft"] = metrics.get("val_loss_soft", val_loss)
                state["val_loss_hard"] = metrics.get("val_loss_hard", val_loss)
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
        result, skipped_keys = self.model.load_compatible_state_dict(sd)
        if skipped_keys:
            print(
                f"  [resume] Dropped {len(skipped_keys)} incompatible keys "
                f"(shape/name mismatch after architecture change)"
            )
        if result.unexpected_keys:
            print(f"  [resume] Dropped {len(result.unexpected_keys)} extra keys from checkpoint (architecture change): {[k.split('.')[-1] for k in result.unexpected_keys]}")
        if result.missing_keys:
            print(f"  [resume] {len(result.missing_keys)} keys initialized randomly (not in checkpoint): {[k.split('.')[-1] for k in result.missing_keys]}")
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
            self.scheduler.step()

            # Curriculum schedule for H_db gain mix: ramp 0.1 → 0.9 over training
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            mix_schedule = min(0.9, 0.1 + 0.8 * (epoch - 1) / max(self.max_epochs - 1, 1))
            model_ref.param_head.gain_mix_value.fill_(mix_schedule)
            if epoch <= 3 or epoch % 5 == 0:
                print(f"  [schedule] gain_mix={mix_schedule:.3f} (H_db weight)")
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

            monitor_value = metrics.get(self.monitor_val_metric, val_loss)
            is_best = monitor_value < self.best_val_loss
            if is_best:
                self.best_val_loss = monitor_value
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, monitor_value, is_best, metrics=metrics)

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": monitor_value,
                    "metrics": metrics,
                    "time_s": elapsed,
                }
            )

            print(
                f"Epoch {epoch}/{self.max_epochs} ({elapsed:.1f}s) "
                f"train={train_loss:.4f} "
                f"{self.monitor_val_metric}={monitor_value:.4f} "
                f"val_loss_hard={metrics.get('val_loss_hard', float('nan')):.4f}"
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
