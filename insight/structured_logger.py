"""
Structured logging for training pipeline observability.

AUDIT: CRITICAL-30 — Provides machine-readable JSON logs and optional
WandB/TensorBoard integration for real-time metric visualization.
AUDIT: MEDIUM-30 — Optional Prometheus metrics export for production monitoring.
AUDIT: MEDIUM-29 — SLA tracking for epoch duration, GPU utilization, data loading overhead.
"""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Prometheus metrics (AUDIT: MEDIUM-30)
# ---------------------------------------------------------------------------

_PROMETHEUS_REGISTRY = None

def _get_prometheus_registry():
    """Get or create Prometheus registry (lazy init)."""
    global _PROMETHEUS_REGISTRY
    if _PROMETHEUS_REGISTRY is None:
        try:
            from prometheus_client import CollectorRegistry
            _PROMETHEUS_REGISTRY = CollectorRegistry()
        except ImportError:
            pass
    return _PROMETHEUS_REGISTRY


class PrometheusMetrics:
    """
    Optional Prometheus metrics for ML training monitoring.

    Usage:
        metrics = PrometheusMetrics()
        metrics.record_metric("training_loss", 0.5, labels={"component": "gain"})
        metrics.start_http_server(port=9090)  # Optional: expose /metrics endpoint
    """

    def __init__(self):
        self._initialized = False
        self._gauges = {}
        self._counters = {}
        self._histograms = {}

        try:
            from prometheus_client import Gauge, Counter, Histogram
            self._Gauge = Gauge
            self._Counter = Counter
            self._Histogram = Histogram
            self._has_prometheus = True
        except ImportError:
            self._has_prometheus = False
            return

        reg = _get_prometheus_registry()
        if reg is None:
            self._has_prometheus = False
            return

        # Training metrics
        self.training_loss = self._Gauge(
            "idsp_training_loss", "Current training loss value",
            ["component"], registry=reg,
        )
        self.validation_metric = self._Gauge(
            "idsp_validation_metric", "Validation metric values",
            ["metric_name"], registry=reg,
        )
        self.epoch_duration = self._Histogram(
            "idsp_epoch_duration_seconds", "Time taken per epoch",
            buckets=[30, 60, 120, 180, 300, 600], registry=reg,
        )
        self.data_loading_overhead = self._Histogram(
            "idsp_data_loading_overhead_seconds", "Time spent on data loading per epoch",
            buckets=[1, 5, 10, 30, 60, 120], registry=reg,
        )
        self.gpu_utilization = self._Gauge(
            "idsp_gpu_utilization_percent", "GPU utilization percentage",
            registry=reg,
        )
        self.gradient_norm = self._Gauge(
            "idsp_gradient_norm", "Gradient norm per component",
            ["component"], registry=reg,
        )
        self.alert_triggered = self._Counter(
            "idsp_alert_total", "Total number of quality alerts triggered",
            ["severity", "metric"], registry=reg,
        )
        self._initialized = True

    def record_metric(self, name: str, value: float, labels: dict = None):
        if not self._initialized:
            return
        try:
            if "loss" in name.lower():
                self.training_loss.labels(component=name).set(value)
            elif any(m in name for m in ["val", "accuracy", "mae", "score"]):
                self.validation_metric.labels(metric_name=name).set(value)
        except Exception:
            pass

    def record_epoch_duration(self, seconds: float):
        if self._initialized:
            self.epoch_duration.observe(seconds)

    def record_data_loading_overhead(self, seconds: float):
        if self._initialized:
            self.data_loading_overhead.observe(seconds)

    def record_gpu_utilization(self, pct: float):
        if self._initialized:
            self.gpu_utilization.set(pct)

    def record_gradient_norm(self, component: str, norm: float):
        if self._initialized:
            self.gradient_norm.labels(component=component).set(norm)

    def record_alert(self, severity: str, metric: str):
        if self._initialized:
            self.alert_triggered.labels(severity=severity, metric=metric).inc()

    def start_http_server(self, port: int = 9090):
        """Expose /metrics endpoint for Prometheus scraping."""
        if not self._initialized:
            return
        try:
            from prometheus_client import start_http_server
            start_http_server(port, registry=_get_prometheus_registry())
            print(f"  [prometheus] Metrics server started on port {port}")
        except Exception as e:
            print(f"  [prometheus] Failed to start HTTP server: {e}")


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
