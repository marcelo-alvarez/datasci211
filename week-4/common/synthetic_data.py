"""Synthetic dataset generation utilities.

Provides DataLoader-compatible datasets for training demonstrations
without requiring external data downloads.
"""

from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset


__all__ = [
    "make_classification_data",
    "split_dataset",
    "create_dataloaders",
]


def make_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification dataset with separable classes.

    Creates a simple synthetic dataset where each class has a distinct pattern
    in feature space, making it suitable for testing classification pipelines.

    Args:
        n_samples: Total number of samples to generate
        n_features: Number of input features
        n_classes: Number of output classes
        seed: Random seed for reproducibility

    Returns:
        Tuple of (features, labels) where:
            - features: FloatTensor of shape (n_samples, n_features)
            - labels: LongTensor of shape (n_samples,) with values in [0, n_classes)

    Example:
        >>> X, y = make_classification_data(n_samples=100, n_features=20, n_classes=4)
        >>> X.shape, y.shape
        (torch.Size([100, 20]), torch.Size([100]))
        >>> y.dtype
        torch.int64
    """
    generator = torch.Generator().manual_seed(seed)

    # Generate class labels uniformly
    labels = torch.randint(0, n_classes, (n_samples,), generator=generator)

    # Generate base features from normal distribution
    features = torch.randn(n_samples, n_features, generator=generator)

    # Add class-specific offsets to make classes separable
    # Each class gets a distinct offset in feature space
    # Offsets (1.4/-1.0) with moderate noise make the task more challenging
    for class_idx in range(n_classes):
        class_mask = labels == class_idx
        # Create a distinctive pattern for this class
        offset = torch.zeros(n_features)
        offset[class_idx % n_features] = 1.4  # Moderate signal in one feature
        offset[(class_idx + 1) % n_features] = -1.0  # Moderate negative signal in another
        features[class_mask] += offset

    # Add Gaussian noise to features to increase task difficulty
    features += 0.2 * torch.randn_like(features)

    return features, labels


def split_dataset(
    dataset: TensorDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset into train/validation/test subsets deterministically.

    Args:
        dataset: TensorDataset to split
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.15)
        seed: Random seed for reproducible shuffling

    Returns:
        Tuple of (train_subset, val_subset, test_subset). Each is a Subset
        of the input dataset (not a TensorDataset).

    Note:
        Test ratio is computed as 1 - train_ratio - val_ratio.
        Ratios should sum to 1.0 (within floating point tolerance).

    Example:
        >>> X, y = make_classification_data(n_samples=1000)
        >>> dataset = TensorDataset(X, y)
        >>> train_ds, val_ds, test_ds = split_dataset(dataset, 0.7, 0.15)
        >>> len(train_ds), len(val_ds), len(test_ds)
        (700, 150, 150)
    """
    assert 0 < train_ratio < 1, "train_ratio must be in (0, 1)"
    assert 0 < val_ratio < 1, "val_ratio must be in (0, 1)"
    assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be < 1"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    return train_ds, val_ds, test_ds


def create_dataloaders(
    train_dataset: Union[TensorDataset, Subset],
    val_dataset: Union[TensorDataset, Subset],
    test_dataset: Union[TensorDataset, Subset],
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/validation/test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders (default: 32)
        num_workers: Number of workers for data loading (default: 2)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Note:
        - Training loader shuffles data each epoch
        - Validation and test loaders do not shuffle
        - pin_memory is enabled for faster GPU transfer

    Example:
        >>> X, y = make_classification_data(n_samples=1000)
        >>> dataset = TensorDataset(X, y)
        >>> train_ds, val_ds, test_ds = split_dataset(dataset)
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_ds, val_ds, test_ds, batch_size=32
        ... )
        >>> for batch_X, batch_y in train_loader:
        ...     print(batch_X.shape, batch_y.shape)
        ...     break
        torch.Size([32, 20]) torch.Size([32])
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
