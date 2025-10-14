"""Data set and DataLoader helpers shared by Week 4 training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

try:  # DistributedSampler is optional for non-DDP scenarios
    from torch.utils.data.distributed import DistributedSampler
except ImportError:  # pragma: no cover - DDP requires PyTorch distributed build
    DistributedSampler = None  # type: ignore

from .synthetic_data import make_classification_data


@dataclass
class ClassificationDataLoaders:
    """Package DataLoader handles and their (optional) samplers."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    samplers: Optional[Dict[str, "DistributedSampler"]] = None


def build_classification_dataset(
    n_train: int,
    n_val: int,
    n_test: int,
    *,
    n_features: int,
    n_classes: int,
    seed: int,
) -> Tuple[Subset, Subset, Subset]:
    """Generate deterministic train/validation/test subsets.

    Synthetic tensors are produced via :func:`make_classification_data` and then split
    according to the exact counts requested by the caller.
    """
    total = n_train + n_val + n_test
    features, labels = make_classification_data(
        n_samples=total,
        n_features=n_features,
        n_classes=n_classes,
        seed=seed,
    )
    dataset = TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(seed)
    splits = random_split(dataset, [n_train, n_val, n_test], generator=generator)
    train_ds, val_ds, test_ds = splits  # type: ignore[assignment]
    return train_ds, val_ds, test_ds


def build_classification_loaders(
    train_dataset: Subset,
    val_dataset: Subset,
    test_dataset: Subset,
    *,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> ClassificationDataLoaders:
    """Construct DataLoader objects (optionally rank-aware for DDP)."""
    use_pin_memory = torch.cuda.is_available()
    samplers: Optional[Dict[str, "DistributedSampler"]] = None

    if distributed:
        if DistributedSampler is None:  # pragma: no cover - guard for CPU-only builds
            raise RuntimeError("DistributedSampler is unavailable; rebuild with distributed support")

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        samplers = {"train": train_sampler, "val": val_sampler, "test": test_sampler}

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )

    return ClassificationDataLoaders(train=train_loader, val=val_loader, test=test_loader, samplers=samplers)
