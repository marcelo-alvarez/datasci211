"""Shared training utility functions.

Common helpers for training loops, metric logging, checkpointing,
and other reusable components across different training scripts.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


__all__ = [
    "set_seed",
    "batch_to_device",
    "AverageMeter",
    "train_one_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
]


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Seeds random, numpy (if available), and torch (CPU + CUDA).

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
        >>> torch.randn(3)
        tensor([...])  # Deterministic output
    """
    random.seed(seed)
    try:
        np.random.seed(seed)
    except NameError:
        pass  # numpy not imported/available
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional settings for determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batch_to_device(
    batch: Union[torch.Tensor, tuple, list],
    device: torch.device,
) -> Union[torch.Tensor, tuple, list]:
    """Move batch to specified device.

    Handles tensor, tuple, or list inputs recursively.

    Args:
        batch: Input batch (tensor, tuple of tensors, or list of tensors)
        device: Target device

    Returns:
        Batch moved to device (preserving original structure)

    Example:
        >>> batch = (torch.randn(32, 20), torch.randint(0, 4, (32,)))
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> batch_gpu = batch_to_device(batch, device)
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (tuple, list)):
        return type(batch)(batch_to_device(item, device) for item in batch)
    else:
        return batch


class AverageMeter:
    """Computes and stores the average and current value.

    Useful for tracking running averages of metrics during training.

    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch_loss in [0.5, 0.4, 0.3]:
        ...     loss_meter.update(batch_loss)
        >>> loss_meter.avg
        0.4
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update statistics with new value.

        Args:
            val: New value to add
            n: Weight/count for this value (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = "auto",
) -> Dict[str, float]:
    """Compute task-appropriate metrics.

    Args:
        outputs: Model predictions
        targets: Ground truth values
        task_type: "classification", "regression", or "auto" (infer from target dtype)

    Returns:
        Dictionary of metrics (accuracy for classification, MAE/MSE for regression)
    """
    # Auto-detect task type from target dtype
    if task_type == "auto":
        if targets.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            task_type = "classification"
        else:
            task_type = "regression"

    metrics = {}

    if task_type == "classification":
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        metrics["accuracy"] = correct / targets.size(0)
    else:  # regression
        # Mean Absolute Error
        mae = (outputs.squeeze() - targets).abs().mean().item()
        metrics["mae"] = mae
        # Mean Squared Error
        mse = ((outputs.squeeze() - targets) ** 2).mean().item()
        metrics["mse"] = mse

    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    task_type: str = "auto",
) -> Dict[str, float]:
    """Train model for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to run training on
        epoch: Current epoch number (for logging)
        task_type: Task type for metric computation. "classification" computes accuracy,
                   "regression" computes MAE/MSE. "auto" infers from target dtype (default).

    Returns:
        Dictionary with "loss" and task-specific metrics:
            - Classification: {"loss": float, "accuracy": float}
            - Regression: {"loss": float, "mae": float, "mse": float}

    Example (classification):
        >>> model = SimpleModel(ModelConfig())
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> criterion = nn.CrossEntropyLoss()
        >>> device = torch.device("cpu")
        >>> metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        >>> print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")

    Example (regression):
        >>> config = ModelConfig(task_type="regression", output_dim=1)
        >>> model = build_model(config).to(device)
        >>> criterion = nn.MSELoss()
        >>> metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, task_type="regression")
        >>> print(f"Loss: {metrics['loss']:.4f}, MAE: {metrics['mae']:.4f}")
    """
    model.train()
    loss_meter = AverageMeter()
    metric_meters = {}  # Dynamic metrics based on task type

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        inputs, targets = batch_to_device(batch, device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute task-appropriate metrics
        batch_metrics = _compute_metrics(outputs, targets, task_type)

        # Update meters
        loss_meter.update(loss.item(), targets.size(0))
        for key, value in batch_metrics.items():
            if key not in metric_meters:
                metric_meters[key] = AverageMeter()
            metric_meters[key].update(value, targets.size(0))

        # Log progress periodically
        if batch_idx % 10 == 0:
            metric_str = " ".join([f"{k.capitalize()}: {v.avg:.4f}" for k, v in metric_meters.items()])
            logger.info(
                f"Epoch: {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f} {metric_str}"
            )

    result = {"loss": loss_meter.avg}
    result.update({k: v.avg for k, v in metric_meters.items()})
    return result


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_type: str = "auto",
) -> Dict[str, float]:
    """Evaluate model on validation/test set.

    Args:
        model: Neural network model
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to run evaluation on
        task_type: Task type for metric computation. "classification" computes accuracy,
                   "regression" computes MAE/MSE. "auto" infers from target dtype (default).

    Returns:
        Dictionary with "loss" and task-specific metrics:
            - Classification: {"loss": float, "accuracy": float}
            - Regression: {"loss": float, "mae": float, "mse": float}

    Example (classification):
        >>> model = SimpleModel(ModelConfig())
        >>> criterion = nn.CrossEntropyLoss()
        >>> device = torch.device("cpu")
        >>> metrics = evaluate(model, val_loader, criterion, device)
        >>> print(f"Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.4f}")

    Example (regression):
        >>> config = ModelConfig(task_type="regression", output_dim=1)
        >>> model = build_model(config).to(device)
        >>> criterion = nn.MSELoss()
        >>> metrics = evaluate(model, val_loader, criterion, device, task_type="regression")
        >>> print(f"Val Loss: {metrics['loss']:.4f}, Val MAE: {metrics['mae']:.4f}")
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meters = {}  # Dynamic metrics

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            inputs, targets = batch_to_device(batch, device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute task-appropriate metrics
            batch_metrics = _compute_metrics(outputs, targets, task_type)

            # Update meters
            loss_meter.update(loss.item(), targets.size(0))
            for key, value in batch_metrics.items():
                if key not in metric_meters:
                    metric_meters[key] = AverageMeter()
                metric_meters[key].update(value, targets.size(0))

    metric_str = " ".join([f"{k.capitalize()}: {v.avg:.4f}" for k, v in metric_meters.items()])
    logger.info(f"Eval - Loss: {loss_meter.avg:.4f} {metric_str}")

    result = {"loss": loss_meter.avg}
    result.update({k: v.avg for k, v in metric_meters.items()})
    return result


def save_checkpoint(
    path: Union[str, Path],
    state: Dict[str, Any],
) -> None:
    """Save checkpoint to disk.

    Creates parent directories if they don't exist.

    Args:
        path: Path to save checkpoint
        state: Dictionary containing model state, optimizer state, etc.

    Example:
        >>> state = {
        ...     "epoch": 10,
        ...     "model_state_dict": model.state_dict(),
        ...     "optimizer_state_dict": optimizer.state_dict(),
        ...     "loss": 0.123,
        ... }
        >>> save_checkpoint("checkpoints/epoch_10.pt", state)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load checkpoint from disk.

    Args:
        path: Path to checkpoint file
        map_location: Device to map tensors to (default: None, uses saved device)

    Returns:
        Dictionary containing checkpoint state

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist

    Example:
        >>> checkpoint = load_checkpoint("checkpoints/epoch_10.pt", map_location="cpu")
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>> optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        >>> start_epoch = checkpoint["epoch"] + 1
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=map_location)
    logger.info(f"Checkpoint loaded from {path}")

    return checkpoint
