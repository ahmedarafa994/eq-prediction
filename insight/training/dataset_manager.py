"""
Dataset manager for the IDSP EQ Estimator training pipeline.

AUDIT: P2-13 — Extracted from the monolithic Trainer class (train.py, 2700+ lines).
Handles dataset creation, train/val/test splitting, precompute caching,
and DataLoader construction.
"""
from functools import partial
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from pipeline_utils import seed_worker


class DatasetManager:
    """Manages dataset lifecycle: creation, splitting, caching, and loading."""

    def __init__(
        self,
        dataset_cls,
        dataset_size: int = 200000,
        val_dataset_size: int = 10000,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 2048,
        num_workers: int = 0,
        precompute_mels: bool = False,
        precompute_cache_path: Optional[str] = None,
        freeze_val_set: bool = False,
        collate_fn=None,
        base_seed: int = 42,
        **dataset_kwargs,
    ):
        self.dataset_cls = dataset_cls
        self.dataset_size = dataset_size
        self.val_dataset_size = val_dataset_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.precompute_mels = precompute_mels
        self.precompute_cache_path = precompute_cache_path
        self.freeze_val_set = freeze_val_set
        self.collate_fn = collate_fn
        self.base_seed = base_seed
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def create_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Create full dataset
        ds = self.dataset_cls(
            size=self.dataset_size,
            precompute_mels=self.precompute_mels,
            base_seed=self.base_seed,
            **self.dataset_kwargs,
        )

        # Try loading precomputed cache from disk
        if self.precompute_cache_path and hasattr(ds, "load_precomputed"):
            if ds.load_precomputed(self.precompute_cache_path):
                print(f"  [data] Loaded cached dataset from {self.precompute_cache_path}")
            else:
                # Precompute if cache was not loaded
                if self.precompute_mels:
                    ds.precompute()
                    if self.precompute_cache_path:
                        ds.save_precomputed(self.precompute_cache_path)

        self.train_dataset = ds

        # Create separate validation dataset if val_dataset_size is specified
        if self.val_dataset_size > 0:
            val_ds = self.dataset_cls(
                size=self.val_dataset_size,
                precompute_mels=False,
                base_seed=self.base_seed + 1000000,
                **self.dataset_kwargs,
            )
            self.val_dataset = val_ds

        # Create test dataset (smaller subset)
        test_size = max(1000, self.dataset_size // 20)
        self.test_dataset = self.dataset_cls(
            size=test_size,
            precompute_mels=False,
            base_seed=self.base_seed + 2000000,
            **self.dataset_kwargs,
        )

        return self.train_dataset, self.val_dataset, self.test_dataset

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for train, validation, and test datasets.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Train loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            worker_init_fn=partial(seed_worker, base_seed=self.base_seed),
        )

        # Validation loader
        val_batch_size = self.batch_size
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        # Test loader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        print(f"  [data] Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        return self.train_loader, self.val_loader, self.test_loader

    def apply_curriculum(self, stage: dict) -> None:
        """Apply curriculum stage overrides to the training dataset."""
        if self.train_dataset and hasattr(self.train_dataset, "apply_curriculum_stage"):
            self.train_dataset.apply_curriculum_stage(stage)
        else:
            print("  [curriculum] train dataset is precomputed; dataset-level curriculum updates are skipped")
