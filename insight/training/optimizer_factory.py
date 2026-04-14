"""
Optimizer factory for the IDSP EQ Estimator training pipeline.

AUDIT: P2-13 — Extracted from the monolithic Trainer class (train.py, 2700+ lines).
Constructs optimizers with support for AdamW, 8-bit AdamW (bitsandbytes),
and DeepSpeed ZeRO-2, with learning rate scheduling.
"""
import math
from typing import Optional, Tuple

import torch


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    head_lr_multiplier: float = 1.0,
    use_8bit: bool = False,
    use_deepspeed: bool = False,
) -> torch.optim.Optimizer:
    """
    Build optimizer with parameter groups for different learning rates.

    Args:
        model: The model to optimize
        lr: Base learning rate for the encoder
        weight_decay: Weight decay for regularization
        head_lr_multiplier: Multiplier for the parameter head learning rate
        use_8bit: Whether to use 8-bit AdamW (bitsandbytes)
        use_deepspeed: Whether to use DeepSpeed ZeRO-2

    Returns:
        Configured optimizer
    """
    # Separate parameter groups for encoder vs parameter head
    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "param_head" in name or "type_head" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr * head_lr_multiplier})

    print(f"  [opt] LR groups: encoder={lr:.1e}, head={lr * head_lr_multiplier:.1e} ({head_lr_multiplier}x)")
    print(f"  [opt] Criterion trainables: {len(param_groups)}")

    # Select optimizer implementation
    if use_deepspeed:
        # DeepSpeed handles optimizer internally
        return _build_deepspeed_optimizer(model, lr, weight_decay)
    elif use_8bit:
        return _build_8bit_adamw(param_groups, weight_decay)
    else:
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def _build_8bit_adamw(param_groups: list, weight_decay: float) -> torch.optim.Optimizer:
    """Build 8-bit AdamW optimizer using bitsandbytes."""
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(param_groups, weight_decay=weight_decay)
    except ImportError:
        print("  [opt] WARNING: bitsandbytes not available, falling back to standard AdamW")
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _build_deepspeed_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Configure DeepSpeed ZeRO-2 optimizer."""
    try:
        import deepspeed
        ds_config = {
            "train_batch_size": 1,
            "zero_optimization": {"stage": 2},
            "optimizer": {"type": "Adam", "params": {"lr": lr, "weight_decay": weight_decay}},
        }
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=ds_config
        )
        return optimizer
    except ImportError:
        print("  [opt] WARNING: DeepSpeed not available, falling back to standard AdamW")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int = 0,
    cycles: int = 1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build learning rate scheduler with optional warmup and cosine annealing.

    Args:
        optimizer: The optimizer to schedule
        max_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs (linear ramp from 0 to base LR)
        cycles: Number of cosine annealing cycles

    Returns:
        Configured scheduler (or None if no scheduling needed)
    """
    if warmup_epochs > 0:
        # Linear warmup followed by cosine annealing with restarts
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=max_epochs,
            cycle_mult=1.0,
            max_lr=optimizer.param_groups[0]["lr"],
            min_lr=optimizer.param_groups[0]["lr"] * 0.01,
            warmup_steps=warmup_epochs,
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=optimizer.param_groups[0]["lr"] * 0.01
        )


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with linear warmup and restarts.

    Args:
        optimizer: Wrapped optimizer
        first_cycle_steps: Number of steps in the first cycle
        cycle_mult: Multiplier for subsequent cycle lengths
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.001,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.cur_cycle_steps = first_cycle_steps
        self.step_in_cycle = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            # Linear warmup
            return [
                self.min_lr + (self.max_lr - self.min_lr) * (self.step_in_cycle / self.warmup_steps)
                for _ in self.base_lrs
            ]
        else:
            # Cosine annealing
            decay_steps = self.cur_cycle_steps - self.warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / decay_steps))
            return [
                self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
                for _ in self.base_lrs
            ]

    def step(self, epoch=None):
        self.step_in_cycle += 1
        super().step(epoch)
