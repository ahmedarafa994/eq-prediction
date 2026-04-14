"""
Checkpoint manager for the IDSP EQ Estimator training pipeline.

AUDIT: P2-13 — Extracted from the monolithic Trainer class (train.py, 2700+ lines).
Handles checkpoint saving, loading, pruning, and emergency recovery.
"""
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from pipeline_utils import compute_version_hash, utc_now_iso


class CheckpointManager:
    """Manages model checkpoint lifecycle: save, load, prune, and emergency recovery."""

    def __init__(self, checkpoint_dir: str = "checkpoints", keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_monitor_value: Optional[float] = None
        self.best_named_metrics: Dict[str, float] = {
            "primary_val_score": float("inf"),
            "gain_mae_db_matched": float("inf"),
            "type_accuracy_matched": 0.0,
            "val_spectral_loss": float("inf"),
        }
        self._events: list = []  # training events log

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        monitor_value: float,
        metrics: Optional[Dict[str, Any]] = None,
        save_tags: Optional[list] = None,
        global_step: int = 0,
        curriculum_stage_idx: int = 0,
        gain_mae_ema: float = 0.0,
        config_hash: str = "",
        dataset_fingerprint: str = "",
        batch_size: int = 0,
        is_best: bool = False,
    ) -> None:
        """
        Save a checkpoint with metadata.

        Args:
            epoch: Current training epoch
            model: The model to save
            optimizer: The optimizer to save
            scheduler: The LR scheduler to save
            monitor_value: Current validation metric value
            metrics: Dict of validation metrics
            save_tags: List of tags (e.g., ["best", "last"])
            global_step: Global training step counter
            curriculum_stage_idx: Current curriculum stage index
            gain_mae_ema: Exponential moving average of gain MAE
            config_hash: Hash of the training configuration
            dataset_fingerprint: Fingerprint of the training dataset
            batch_size: Current batch size
            is_best: Whether this is the best model so far
        """
        save_tags = list(save_tags or [])

        state = self._build_state(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            monitor_value=monitor_value,
            metrics=metrics,
            global_step=global_step,
            curriculum_stage_idx=curriculum_stage_idx,
            gain_mae_ema=gain_mae_ema,
            config_hash=config_hash,
            dataset_fingerprint=dataset_fingerprint,
            batch_size=batch_size,
        )

        # Save epoch checkpoint
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(state, tmp_path)
        tmp_path.rename(path)
        print(f"  Saved checkpoint: {path}")

        # Save tagged checkpoints
        for tag in save_tags:
            tag_path = self.checkpoint_dir / f"{tag}.pt"
            tmp_tag = tag_path.with_suffix(".pt.tmp")
            torch.save(state, tmp_tag)
            tmp_tag.rename(tag_path)
            print(f"  Updated checkpoint: {tag_path}")
            # AUDIT: HIGH-09 — When 'best' tag is set, also create best.pt alias
            if tag == "best":
                best_path = self.checkpoint_dir / "best.pt"
                tmp_best = best_path.with_suffix(".pt.tmp")
                torch.save(state, tmp_best)
                tmp_best.rename(best_path)
                print(f"  Updated checkpoint: {best_path}")

        self._log_event("checkpoint_saved", epoch=epoch, monitor_value=monitor_value, tags=save_tags)

        # Prune old checkpoints
        self._prune_old_checkpoints()

    def _build_state(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        monitor_value: float,
        metrics: Optional[Dict[str, Any]] = None,
        global_step: int = 0,
        curriculum_stage_idx: int = 0,
        gain_mae_ema: float = 0.0,
        config_hash: str = "",
        dataset_fingerprint: str = "",
        batch_size: int = 0,
    ) -> Dict[str, Any]:
        """Build checkpoint state dictionary with metadata."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": monitor_value,
            "monitor_metric": getattr(self, "_monitor_metric", "primary_val_score"),
            "monitor_value": monitor_value,
            "global_step": global_step,
            "curriculum_stage_idx": curriculum_stage_idx,
            "gain_mae_ema": gain_mae_ema,
            "config_hash": config_hash,
            "dataset_fingerprint": dataset_fingerprint,
            "batch_size": batch_size,
        }

        if metrics is not None:
            state["primary_val_score"] = metrics.get("primary_val_score", monitor_value)
            state["val_spectral_loss"] = metrics.get("val_spectral_loss", monitor_value)
            state["val_spectral_loss_soft"] = metrics.get("val_spectral_loss_soft", monitor_value)
            state["val_spectral_loss_hard"] = metrics.get("val_spectral_loss_hard", monitor_value)
            state["val_audio_loss"] = state["val_spectral_loss"]
            state["val_loss_soft"] = metrics.get("val_loss_soft", monitor_value)
            state["val_loss_hard"] = metrics.get("val_loss_hard", monitor_value)
            state["gain_mae_db_matched"] = metrics.get("gain_mae_db_matched")
            state["type_accuracy_matched"] = metrics.get("type_accuracy_matched")
            state["best_named_metrics"] = dict(self.best_named_metrics)

        # AUDIT: P2-17 — Version tracking for reproducibility
        try:
            state["code_version_hash"] = compute_version_hash()
        except Exception:
            state["code_version_hash"] = "unknown"
        state["torch_version"] = torch.__version__
        state["numpy_version"] = np.__version__
        import sys
        state["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        return state

    def _prune_old_checkpoints(self) -> None:
        """Delete old epoch_NNN.pt checkpoints, keeping the last N plus named ones."""
        named_checkpoints = {"best.pt", "last.pt"}
        epoch_ckpts = sorted(
            p for p in self.checkpoint_dir.glob("epoch_*.pt")
            if p.name not in named_checkpoints
        )
        if len(epoch_ckpts) <= self.keep_last_n:
            return
        to_delete = epoch_ckpts[:-self.keep_last_n]
        for p in to_delete:
            try:
                p.unlink()
                print(f"  Pruned old checkpoint: {p}")
            except OSError:
                pass

    def load(self, path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler=None) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model/optimizer/scheduler state.

        Returns:
            Dict with checkpoint metadata (epoch, metrics, etc.)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"  Loading checkpoint from {path}...")
        state = torch.load(path, map_location="cpu", weights_only=False)

        model.load_state_dict(state["model_state_dict"])
        print(f"  Restored model state from epoch {state.get('epoch', '?')}")

        if optimizer is not None and "optimizer_state_dict" in state:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
                print("  Restored optimizer state")
            except Exception as e:
                print(f"  [resume] Could not restore optimizer state ({e}); starting fresh optimizer")

        if scheduler is not None and "scheduler_state_dict" in state:
            try:
                scheduler.load_state_dict(state["scheduler_state_dict"])
                print("  Restored scheduler state")
            except Exception as e:
                print(f"  [resume] Could not restore scheduler state ({e}); starting fresh scheduler")

        return state

    def save_emergency(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, batch_idx: int, **kwargs) -> None:
        """
        Save an emergency checkpoint at the current batch boundary.

        AUDIT: mid_epoch_ckpt — Signal handlers check only at epoch boundaries,
        potentially losing up to 72 seconds of compute. This method saves a
        checkpoint that can be resumed from the exact batch where the signal
        was received.
        """
        emergency_path = self.checkpoint_dir / f"emergency_epoch_{epoch:03d}_batch_{batch_idx:05d}.pt"
        tmp_path = emergency_path.with_suffix(".pt.tmp")

        state = self._build_state(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, **kwargs)
        state["emergency_batch_idx"] = batch_idx
        state["saved_at"] = utc_now_iso()

        torch.save(state, tmp_path)
        tmp_path.rename(emergency_path)
        print(f"  [signal] Emergency checkpoint saved: {emergency_path}")
        self._log_event("emergency_checkpoint_saved", epoch=epoch, batch_idx=batch_idx)

    def _log_event(self, event_type: str, **kwargs) -> None:
        """Log a training event for audit trail."""
        event = {"type": event_type, "timestamp": utc_now_iso(), **kwargs}
        self._events.append(event)

    def get_events(self) -> list:
        """Return all logged events."""
        return list(self._events)
