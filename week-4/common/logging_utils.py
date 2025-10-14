"""Logging configuration utilities shared by Week 4 scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def format_timespan(duration_seconds: float) -> str:
    """Return a human-readable representation of a duration.

    Examples:
        74.2 -> "1m 14.2s"
        8.9 -> "8.90s"
    """

    if duration_seconds < 0:
        duration_seconds = 0.0

    total_seconds = int(duration_seconds)
    fractional = duration_seconds - total_seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds_with_fraction = seconds + fractional

    if hours:
        return f"{hours}h {minutes:02d}m {seconds_with_fraction:04.1f}s"
    if minutes:
        return f"{minutes}m {seconds_with_fraction:04.1f}s"
    return f"{seconds_with_fraction:05.2f}s"


_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_CONSOLE_FORMAT = "%(levelname)s - %(message)s"


def _reset_root_logger(level: int) -> logging.Logger:
    """Remove existing handlers and configure the root logger."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(level)
    root_logger.propagate = False
    return root_logger


def setup_basic_logging(output_dir: Path, log_name: str = "training.log") -> Path:
    """Initialise console + file logging for single-process scripts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_logger = _reset_root_logger(logging.INFO)
    log_path = output_dir / log_name

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_path


def setup_rank_logging(
    output_dir: Path,
    rank: int,
    verbose: bool,
    log_prefix: str = "training",
) -> Path:
    """Initialise per-rank logging for DistributedDataParallel runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO
    root_logger = _reset_root_logger(level)

    log_path = output_dir / f"{log_prefix}_rank{rank}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(file_handler)

    if verbose or rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        if verbose and rank != 0:
            console_format = logging.Formatter(f"[rank {rank}] {_CONSOLE_FORMAT}")
        else:
            console_format = logging.Formatter(_CONSOLE_FORMAT)
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

    return log_path
