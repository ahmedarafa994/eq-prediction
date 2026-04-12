"""
Structured logging for training pipeline observability.

AUDIT: CRITICAL-30 — Provides machine-readable JSON logs and optional
WandB/TensorBoard integration for real-time metric visualization.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class StructuredLogger:
    """
    Writes structured JSON logs to a file and optionally to WandB.

    Usage:
        logger = StructuredLogger(log_dir="checkpoints", enable_wandb=True)
        logger.log_metric("train_loss", 0.5, epoch=1, step=100)
        logger.log_event("checkpoint_saved", {"path": "checkpoints/best.pt"})
    """

    def __init__(
        self,
        log_dir: str = "checkpoints",
        enable_wandb: bool = False,
        wandb_project: str = "idsp-eq",
        wandb_run_name: str = None,
        enable_tensorboard: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "structured_log.jsonl"
        self.enable_wandb = enable_wandb
        self.enable_tensorboard = enable_tensorboard
        self._wandb_initialized = False
        self._tensorboard_initialized = False

        if enable_wandb:
            self._init_wandb(wandb_project, wandb_run_name)
        if enable_tensorboard:
            self._init_tensorboard()

    # ------------------------------------------------------------------
    # WandB / TensorBoard initialization
    # ------------------------------------------------------------------

    def _init_wandb(self, project: str, run_name: str = None):
        try:
            import wandb
            wandb.init(project=project, name=run_name)
            self._wandb_initialized = True
            print(f"  [logger] WandB initialized: project={project}, name={run_name}")
        except ImportError:
            print("  [logger] WandB not installed — structured logs only")
            self.enable_wandb = False
        except Exception as e:
            print(f"  [logger] WandB init failed: {e}")
            self.enable_wandb = False

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            self._tensorboard_initialized = True
            print(f"  [logger] TensorBoard initialized: {self.log_dir / 'tensorboard'}")
        except ImportError:
            print("  [logger] TensorBoard not installed — structured logs only")
            self.enable_tensorboard = False
        except Exception as e:
            print(f"  [logger] TensorBoard init failed: {e}")
            self.enable_tensorboard = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metric(
        self,
        name: str,
        value: float,
        epoch: int = None,
        step: int = None,
        extra: dict = None,
    ):
        """
        Log a single numeric metric.

        Writes to JSONL file and optionally to WandB / TensorBoard.
        """
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return  # Skip non-finite values

        record = {
            "timestamp": utc_now_iso(),
            "type": "metric",
            "name": name,
            "value": value,
        }
        if epoch is not None:
            record["epoch"] = epoch
        if step is not None:
            record["step"] = step
        if extra:
            record.update(extra)

        self._write_jsonl(record)

        if self._wandb_initialized:
            import wandb
            log_dict = {name: value}
            if epoch is not None:
                log_dict["epoch"] = epoch
            if step is not None:
                log_dict["step"] = step
            wandb.log(log_dict)

        if self._tensorboard_initialized:
            tag = name
            if epoch is not None:
                self._tb_writer.add_scalar(tag, value, epoch)
            elif step is not None:
                self._tb_writer.add_scalar(tag, value, step)

    def log_event(self, event_name: str, payload: dict = None):
        """
        Log a lifecycle event (checkpoint saved, NaN detected, etc.).
        """
        record = {
            "timestamp": utc_now_iso(),
            "type": "event",
            "event": event_name,
            **(payload or {}),
        }
        self._write_jsonl(record)

        if self._wandb_initialized:
            import wandb
            wandb.log({f"event/{event_name}": 1})

    def log_metrics_batch(self, metrics: dict, epoch: int = None, step: int = None):
        """
        Log a batch of metrics at once (e.g., all validation metrics per epoch).
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(name, value, epoch=epoch, step=step)

    def log_grad_norms(self, grad_norms: dict, step: int):
        """
        Log per-component gradient norms.
        """
        for component, norm in grad_norms.items():
            if isinstance(norm, (int, float)) and math.isfinite(norm):
                self.log_metric(f"grad_norm/{component}", norm, step=step)

    def close(self):
        """Close all logging backends."""
        if self._wandb_initialized:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        if self._tensorboard_initialized:
            try:
                self._tb_writer.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
