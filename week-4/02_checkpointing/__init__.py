"""Checkpointing module for robust training with signal handling.

Provides utilities for:
- Atomic checkpoint I/O with automatic pruning
- SLURM signal monitoring for graceful job termination
- Resume-capable training workflows
"""

from .checkpoint_io import CheckpointManager
from .signal_handler import SlurmSignalMonitor, install_default_signal_handlers

__all__ = ["CheckpointManager", "SlurmSignalMonitor", "install_default_signal_handlers"]
