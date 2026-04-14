"""
Training pipeline modules for the IDSP EQ Estimator.

AUDIT: P2-13 — Decomposed from the monolithic Trainer class (train.py, 2700+ lines)
into focused, testable modules:

- checkpoint_manager.py: Checkpoint save/load/prune/emergency recovery
- dataset_manager.py: Dataset creation, splitting, caching, DataLoader construction
- optimizer_factory.py: Optimizer construction (AdamW, 8-bit, DeepSpeed) + LR scheduling
- validation_loop.py: Validation metric computation, Hungarian matching

The thin trainer.py orchestrator wires these modules together.
"""

from training.checkpoint_manager import CheckpointManager
from training.dataset_manager import DatasetManager
from training.optimizer_factory import build_optimizer, build_scheduler
from training.validation_loop import ValidationLoop

__all__ = [
    "CheckpointManager",
    "DatasetManager",
    "build_optimizer",
    "build_scheduler",
    "ValidationLoop",
]
