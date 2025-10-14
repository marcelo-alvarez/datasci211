"""SLURM signal handling for graceful checkpointing on preemption.

Monitors SIGUSR1 (preemption warning) and SIGTERM (timeout) signals,
triggering checkpoint callbacks and graceful shutdown.
"""

import logging
import signal
from typing import Callable, Iterable, Optional


__all__ = ["SlurmSignalMonitor", "install_default_signal_handlers"]


logger = logging.getLogger(__name__)


class SlurmSignalMonitor:
    """Monitor SLURM signals and trigger checkpoint callbacks.

    Handles SIGUSR1 (preemption warning) and SIGTERM (timeout) signals,
    ensuring checkpoints are saved exactly once before shutdown.

    Example:
        >>> def save_checkpoint():
        ...     print("Saving checkpoint...")
        >>> monitor = SlurmSignalMonitor(checkpoint_cb=save_checkpoint)
        >>> monitor.install()
        >>> # Training loop can check monitor.should_stop
        >>> while not monitor.should_stop:
        ...     train_one_epoch()
    """

    def __init__(
        self,
        checkpoint_cb: Callable[[], None],
        exit_cb: Optional[Callable[[], None]] = None,
    ):
        """Initialize signal monitor.

        Args:
            checkpoint_cb: Callback to save checkpoint (called exactly once)
            exit_cb: Optional callback to run after checkpoint (e.g., cleanup)
        """
        self.checkpoint_cb = checkpoint_cb
        self.exit_cb = exit_cb
        self.stop_requested = False
        self.received_signal = None
        self._checkpoint_triggered = False

    @property
    def should_stop(self) -> bool:
        """Check if training should stop due to signal."""
        return self.stop_requested

    def install(self, signals: Optional[Iterable[int]] = None) -> None:
        """Install signal handlers.

        Args:
            signals: Signals to monitor (default: {SIGUSR1, SIGTERM})

        Example:
            >>> monitor.install()  # Use defaults
            >>> monitor.install([signal.SIGUSR1])  # Custom signals
        """
        if signals is None:
            signals = {signal.SIGUSR1, signal.SIGTERM}

        for sig in signals:
            signal.signal(sig, self.handle)
            logger.info(f"Installed signal handler for {sig}")

    def handle(self, signum: int, frame) -> None:
        """Handle received signal.

        Called by signal module when signal is received.
        Can also be invoked directly for testing.

        Args:
            signum: Signal number
            frame: Current stack frame (unused)
        """
        if self._checkpoint_triggered:
            logger.debug(f"Signal {signum} received but checkpoint already triggered")
            return

        logger.warning(f"Received signal {signum}, initiating graceful shutdown")

        # Record signal details
        self.received_signal = signum
        self._checkpoint_triggered = True

        # Trigger checkpoint callback exactly once
        try:
            logger.info("Calling checkpoint callback...")
            self.checkpoint_cb()
            logger.info("Checkpoint callback completed")
        except Exception as e:
            logger.error(f"Checkpoint callback failed: {e}", exc_info=True)

        # Set stop flag
        self.stop_requested = True

        # Call exit callback if provided
        if self.exit_cb is not None:
            try:
                logger.info("Calling exit callback...")
                self.exit_cb()
            except Exception as e:
                logger.error(f"Exit callback failed: {e}", exc_info=True)


def install_default_signal_handlers(
    checkpoint_cb: Callable[[], None],
    exit_cb: Optional[Callable[[], None]] = None,
) -> SlurmSignalMonitor:
    """Create and install signal monitor with default SLURM signals.

    Convenience helper for common use case.

    Args:
        checkpoint_cb: Callback to save checkpoint
        exit_cb: Optional exit/cleanup callback

    Returns:
        Configured and installed SlurmSignalMonitor

    Example:
        >>> def save_checkpoint():
        ...     manager.save(state, epoch=current_epoch)
        >>> monitor = install_default_signal_handlers(save_checkpoint)
        >>> # Training loop checks monitor.should_stop
    """
    monitor = SlurmSignalMonitor(checkpoint_cb=checkpoint_cb, exit_cb=exit_cb)
    monitor.install()
    return monitor
