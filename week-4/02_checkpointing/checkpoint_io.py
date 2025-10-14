"""Checkpoint I/O utilities for safe checkpoint saving and resumption.

Provides CheckpointManager for atomic checkpoint writes, metadata tracking,
retention policies, and resume capabilities.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


__all__ = ["CheckpointManager"]


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving, loading, and retention.

    Features:
    - Atomic writes via temporary files to prevent partial checkpoints
    - JSON metadata tracking (latest.json) with epoch, timestamp, best flag
    - Automatic pruning to keep only last N checkpoints
    - Resume helpers to load latest checkpoint and metadata

    Example:
        >>> manager = CheckpointManager(checkpoint_dir="./checkpoints", keep_last=3)
        >>> state = {"epoch": 5, "model_state_dict": model.state_dict(), ...}
        >>> path = manager.save(state, epoch=5, is_best=False)
        >>> checkpoint, metadata = manager.load_latest()
        >>> print(f"Resumed from epoch {metadata['epoch']}")
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        prefix: str = "checkpoint_epoch",
        keep_last: int = 3,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            prefix: Prefix for checkpoint filenames (default: "checkpoint_epoch")
            keep_last: Number of recent checkpoints to keep (minimum 1)

        Raises:
            ValueError: If keep_last < 1
        """
        if keep_last < 1:
            raise ValueError(f"keep_last must be >= 1, got {keep_last}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.prefix = prefix
        self.keep_last = keep_last
        self.metadata_file = self.checkpoint_dir / "latest.json"

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"CheckpointManager initialized: dir={self.checkpoint_dir}, "
            f"prefix={self.prefix}, keep_last={self.keep_last}"
        )

    def _extract_epoch(self, path: Path) -> int:
        """Extract epoch number from checkpoint filename."""
        try:
            return int(path.stem.split('_')[-1])
        except ValueError:
            return -1

    def save(
        self,
        state: Dict[str, Any],
        epoch: int,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint with atomic write and metadata tracking.

        Args:
            state: Dictionary containing model state, optimizer state, etc.
            epoch: Current epoch number
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint file

        Example:
            >>> state = {
            ...     "epoch": 5,
            ...     "model_state_dict": model.state_dict(),
            ...     "optimizer_state_dict": optimizer.state_dict(),
            ... }
            >>> path = manager.save(state, epoch=5, is_best=True)
        """
        # Generate checkpoint filename
        checkpoint_name = f"{self.prefix}_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Atomic write: save to temp file then rename
        tmp_path = checkpoint_path.with_suffix(".tmp")
        try:
            torch.save(state, tmp_path)
            tmp_path.rename(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            logger.error(f"Failed to save checkpoint: {e}")
            raise

        # Update metadata file
        metadata = {
            "epoch": epoch,
            "is_best": is_best,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "path": checkpoint_name,
        }

        try:
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Metadata updated: {metadata}")
        except Exception as e:
            logger.warning(f"Failed to update metadata file: {e}")

        # Prune old checkpoints
        self._prune_old_checkpoints()

        return checkpoint_path

    def load_latest(
        self,
        map_location=None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load the most recent checkpoint and its metadata.

        Args:
            map_location: Device to map tensors to (e.g., "cpu", "cuda:0")

        Returns:
            Tuple of (checkpoint_state, metadata_dict)

        Raises:
            FileNotFoundError: If no checkpoints exist

        Example:
            >>> checkpoint, metadata = manager.load_latest(map_location="cpu")
            >>> model.load_state_dict(checkpoint["model_state_dict"])
            >>> start_epoch = metadata["epoch"] + 1
        """
        latest = self.latest_path()
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        # Load checkpoint
        checkpoint = self.load(latest, map_location=map_location)

        # Load metadata
        metadata = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

        return checkpoint, metadata

    def load(
        self,
        path: Path,
        map_location=None,
    ) -> Dict[str, Any]:
        """Load checkpoint from specific path.

        Args:
            path: Path to checkpoint file
            map_location: Device to map tensors to

        Returns:
            Dictionary containing checkpoint state

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist

        Example:
            >>> checkpoint = manager.load("checkpoint_epoch_5.pt", map_location="cpu")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location=map_location)
        logger.info(f"Checkpoint loaded from {path}")

        return checkpoint

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files, sorted by modification time (newest first).

        Returns:
            List of checkpoint paths

        Example:
            >>> checkpoints = manager.list_checkpoints()
            >>> print(f"Found {len(checkpoints)} checkpoints")
        """
        if not self.checkpoint_dir.exists():
            return []

        # Find all .pt files matching prefix
        pattern = f"{self.prefix}_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        # Sort newest first using epoch number (with mtime tie-breaker)
        checkpoints.sort(
            key=lambda p: (self._extract_epoch(p), p.stat().st_mtime), reverse=True
        )

        return checkpoints

    def latest_path(self) -> Optional[Path]:
        """Get path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist

        Example:
            >>> latest = manager.latest_path()
            >>> if latest:
            ...     print(f"Latest checkpoint: {latest}")
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond keep_last limit."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.keep_last:
            to_remove = checkpoints[self.keep_last:]
            for ckpt in to_remove:
                try:
                    ckpt.unlink()
                    logger.info(f"Pruned old checkpoint: {ckpt.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {ckpt}: {e}")
