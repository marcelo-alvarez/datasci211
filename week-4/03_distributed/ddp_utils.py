"""Distributed training utility functions.

Helper functions for DDP setup, rank/world size management, distributed metrics
aggregation, and SLURM environment variable parsing for multi-node training.
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.training_utils import set_seed

__all__ = [
    "DistributedConfig",
    "init_distributed",
    "cleanup_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_master",
    "synchronize",
    "reduce_dict",
    "seed_everything_for_ddp",
    "rank_log",
]


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device
    master_addr: str
    master_port: int
    verbose: bool = False


class _RankFilter(logging.Filter):
    """Logging filter that blocks non-master ranks unless verbose mode is enabled."""

    def __init__(self, rank: int, verbose: bool):
        super().__init__()
        self.rank = rank
        self.verbose = verbose

    def filter(self, record):
        return self.rank == 0 or self.verbose


class _DistState:
    """Module-level state for distributed training."""

    def __init__(self):
        self.config: Optional[DistributedConfig] = None
        self.original_handler_state: List[tuple] = []


_DIST_STATE = _DistState()


def _infer_distributed_config(
    backend: str,
    rank: Optional[int],
    world_size: Optional[int],
    local_rank: Optional[int],
    master_addr: Optional[str],
    master_port: Optional[int],
    verbose: bool,
    set_cuda_device: bool,
) -> DistributedConfig:
    """Infer distributed config from CLI overrides, torchrun, or SLURM environment."""

    # Priority 1: CLI overrides
    if rank is not None and world_size is not None and local_rank is not None:
        resolved_rank = rank
        resolved_world_size = world_size
        resolved_local_rank = local_rank
        resolved_master_addr = master_addr or "127.0.0.1"
        resolved_master_port = master_port or 29500
    # Priority 2: torchrun environment variables
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        resolved_rank = int(os.environ["RANK"])
        resolved_world_size = int(os.environ["WORLD_SIZE"])
        resolved_local_rank = int(os.environ["LOCAL_RANK"])
        resolved_master_addr = master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
        resolved_master_port = master_port or int(os.environ.get("MASTER_PORT", "29500"))
    # Priority 3: SLURM environment variables
    elif "SLURM_PROCID" in os.environ:
        if "SLURM_NTASKS" not in os.environ or "SLURM_LOCALID" not in os.environ:
            raise RuntimeError(
                "Running under SLURM but required environment variables are missing. "
                "Expected SLURM_PROCID, SLURM_NTASKS, and SLURM_LOCALID. "
                "Ensure your SLURM job is configured correctly or provide explicit "
                "overrides via --rank, --world-size, and --local-rank arguments."
            )

        resolved_rank = int(os.environ["SLURM_PROCID"])
        resolved_world_size = int(os.environ["SLURM_NTASKS"])
        resolved_local_rank = int(os.environ["SLURM_LOCALID"])

        # Parse SLURM node list for master address
        if master_addr is None and "SLURM_JOB_NODELIST" in os.environ:
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            # Simple extraction: handle single node or take first node from list
            # Format examples: "node01" or "node[01-04]" or "node01,node02"
            if "[" in nodelist:
                # Extract first node from range: "node[01-04]" -> "node01"
                base = nodelist.split("[")[0]
                first_num = nodelist.split("[")[1].split("-")[0].split(",")[0].rstrip("]")
                resolved_master_addr = f"{base}{first_num}"
            elif "," in nodelist:
                # Take first from comma-separated list
                resolved_master_addr = nodelist.split(",")[0]
            else:
                resolved_master_addr = nodelist
        else:
            resolved_master_addr = master_addr or "127.0.0.1"

        resolved_master_port = master_port or int(os.environ.get("SLURM_STEP_RESV_PORTS", "29500").split("-")[0])
    else:
        raise RuntimeError(
            "Cannot infer distributed configuration. "
            "Please provide explicit arguments (--rank, --world-size, --local-rank) "
            "or run via torchrun (sets RANK, WORLD_SIZE, LOCAL_RANK) "
            "or SLURM (sets SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID)."
        )

    # Determine device
    if set_cuda_device and torch.cuda.is_available():
        visible_device_count = torch.cuda.device_count()
        if visible_device_count == 0:
            raise RuntimeError(
                "Requested CUDA device assignment but no GPUs are visible. "
                "Ensure CUDA is available or disable automatic device selection."
            )

        if resolved_local_rank >= visible_device_count:
            # SLURM task counts can exceed the CUDA_VISIBLE_DEVICES fan-out when a node
            # exposes fewer GPUs than tasks; remap by modulo so ranks still land on a
            # valid device instead of raising "invalid device ordinal".
            assigned_index = resolved_local_rank % visible_device_count
            if visible_device_count == 1:
                logging.info(
                    "Local rank %s restricted to CUDA:0 because only one GPU is visible in this task.",
                    resolved_local_rank,
                )
            else:
                logging.warning(
                    "Local rank %s exceeds visible GPUs (%s); remapping to CUDA:%s.",
                    resolved_local_rank,
                    visible_device_count,
                    assigned_index,
                )
        else:
            assigned_index = resolved_local_rank

        device = torch.device(f"cuda:{assigned_index}")
    else:
        device = torch.device("cpu")

    return DistributedConfig(
        rank=resolved_rank,
        world_size=resolved_world_size,
        local_rank=resolved_local_rank,
        backend=backend,
        device=device,
        master_addr=resolved_master_addr,
        master_port=resolved_master_port,
        verbose=verbose,
    )


def init_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    timeout: Optional[timedelta] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    verbose: bool = False,
    set_cuda_device: bool = True,
) -> DistributedConfig:
    """Initialize distributed training process group.

    Args:
        backend: Communication backend ("nccl", "gloo", "mpi")
        init_method: Initialization method (default "env://")
        timeout: Timeout for operations
        rank: Process rank (optional, inferred if not provided)
        world_size: Total number of processes (optional, inferred if not provided)
        local_rank: Local rank on node (optional, inferred if not provided)
        master_addr: Master node address (optional, inferred if not provided)
        master_port: Master node port (optional, inferred if not provided)
        verbose: Enable verbose logging for all ranks
        set_cuda_device: Automatically set CUDA device based on local_rank

    Returns:
        DistributedConfig with resolved configuration
    """
    # Validate backend availability
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError(
            "Backend 'nccl' requires CUDA but torch.cuda.is_available() is False. "
            "For CPU-only testing, use backend='gloo'."
        )

    # Infer configuration
    config = _infer_distributed_config(
        backend=backend,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        verbose=verbose,
        set_cuda_device=set_cuda_device,
    )

    # Set environment variables for init_process_group
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)

    # Set CUDA device if appropriate
    if set_cuda_device and torch.cuda.is_available():
        torch.cuda.set_device(config.device)

    # Initialize process group
    init_kwargs = {
        "backend": backend,
        "init_method": init_method,
        "rank": config.rank,
        "world_size": config.world_size,
    }
    if timeout is not None:
        init_kwargs["timeout"] = timeout

    dist.init_process_group(**init_kwargs)

    # Configure logging: only rank 0 logs by default
    root_logger = logging.getLogger()
    rank_filter = _RankFilter(config.rank, config.verbose)

    # Save original handler state so cleanup can restore defaults later
    for handler in root_logger.handlers:
        _DIST_STATE.original_handler_state.append((handler, handler.level, handler.filters.copy()))
        handler.addFilter(rank_filter)

    # Store config in module state
    _DIST_STATE.config = config

    # Log configuration from rank 0
    if config.rank == 0:
        logging.info(
            f"Initialized distributed training: rank={config.rank}, "
            f"world_size={config.world_size}, backend={config.backend}, "
            f"master={config.master_addr}:{config.master_port}, device={config.device}"
        )

    return config


def cleanup_distributed() -> None:
    """Clean up distributed training process group and restore logging state."""
    # Restore logging handlers
    for handler, level, filters in _DIST_STATE.original_handler_state:
        handler.filters = filters
        handler.level = level
    _DIST_STATE.original_handler_state.clear()

    # Synchronize and destroy process group if initialized
    if dist.is_initialized():
        if _DIST_STATE.config and _DIST_STATE.config.world_size > 1:
            dist.barrier()
        dist.destroy_process_group()

    # Reset module state
    _DIST_STATE.config = None


def get_rank() -> int:
    """Get current process rank (0 if not initialized)."""
    if _DIST_STATE.config is not None:
        return _DIST_STATE.config.rank
    return 0


def get_world_size() -> int:
    """Get total number of processes (1 if not initialized)."""
    if _DIST_STATE.config is not None:
        return _DIST_STATE.config.world_size
    return 1


def get_local_rank() -> int:
    """Get local rank on current node (0 if not initialized)."""
    if _DIST_STATE.config is not None:
        return _DIST_STATE.config.local_rank
    return 0


def is_master() -> bool:
    """Check if current process is rank 0 (True if not initialized)."""
    return get_rank() == 0


def synchronize() -> None:
    """Synchronize all processes (barrier only when world_size > 1)."""
    if dist.is_initialized() and get_world_size() > 1:
        dist.barrier()


def reduce_dict(
    input_dict: Dict[str, Union[float, torch.Tensor]],
    average: bool = True,
) -> Dict[str, Union[float, torch.Tensor, List[torch.Tensor]]]:
    """Reduce dictionary of metrics across all processes.

    Scalar values (floats or 0-d tensors) are averaged across ranks.
    Higher-dimensional tensors are gathered into lists.

    Args:
        input_dict: Dictionary of metrics to reduce
        average: Whether to average scalar values (vs sum)

    Returns:
        Dictionary with reduced values
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return input_dict

    device = _DIST_STATE.config.device if _DIST_STATE.config else torch.device("cpu")
    output_dict = {}

    for key, value in input_dict.items():
        # Convert to tensor if needed
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, dtype=torch.float32, device=device)
        else:
            tensor = value.to(device) if not value.device == device else value

        # Handle scalars: all-reduce and average
        if tensor.dim() == 0 or (tensor.dim() == 1 and tensor.numel() == 1):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if average:
                tensor = tensor / get_world_size()
            output_dict[key] = tensor.item() if tensor.numel() == 1 else tensor
        else:
            # Handle higher-dimensional tensors: all-gather produces per-rank replicas
            gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
            dist.all_gather(gathered, tensor)
            output_dict[key] = gathered

    return output_dict


def seed_everything_for_ddp(base_seed: int) -> None:
    """Set random seeds for DDP training (base_seed + rank for per-process variation).

    Args:
        base_seed: Base random seed
    """
    seed = base_seed + get_rank()
    set_seed(seed)


def rank_log(
    message: str,
    level: str = "info",
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
) -> None:
    """Log a message with rank awareness.

    By default, only rank 0 logs. If verbose=True, all ranks log with rank prefix.

    Args:
        message: Message to log
        level: Logging level ("info", "debug", "warning", "error")
        logger: Logger instance (uses root logger if None)
        verbose: Force logging on all ranks with rank prefix
    """
    rank = get_rank()

    if rank == 0 or verbose:
        log_func = getattr(logger or logging.getLogger(), level.lower())
        prefix = f"[rank {rank}] " if verbose and rank != 0 else ""
        log_func(f"{prefix}{message}")
