#!/usr/bin/env python3
"""Multi-GPU training with DistributedDataParallel (DDP).

Data-parallel training across multiple GPUs using PyTorch DDP with proper
process group initialization, synchronized training, checkpointing, and signal handling.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.cli import add_classification_args, finalize_classification_args, resolve_output_paths
from common.data_utils import build_classification_dataset, build_classification_loaders
from common.logging_utils import format_timespan, setup_rank_logging
from common.metrics import append_metrics_jsonl
from common.random_state import restore_rng_state, save_rng_state
from common.simple_model import ModelConfig, build_model

from ddp_utils import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_master,
    rank_log,
    reduce_dict,
    seed_everything_for_ddp,
    synchronize,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "02_checkpointing"))
from checkpoint_io import CheckpointManager
from signal_handler import install_default_signal_handlers


__all__ = ["main"]


logger = logging.getLogger(__name__)


def write_metrics_entry(
    metrics_file: Path,
    epoch: int,
    split: str,
    loss: float,
    accuracy: float,
    samples: int,
    global_step: int,
) -> None:
    """Write metrics entry to JSONL file (rank 0 only)."""

    append_metrics_jsonl(
        metrics_file,
        {
            "epoch": epoch,
            "split": split,
            "loss": loss,
            "accuracy": accuracy,
            "samples": samples,
            "global_step": global_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        master_only=True,
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    accumulation_steps: int,
    global_step: int,
    epoch: int,
    split: str,
    log_every: int,
) -> tuple:
    """Run one epoch of training or evaluation.

    Args:
        model: DDP model
        dataloader: Data loader
        criterion: Loss function
        optimizer: Optimizer (None for evaluation)
        device: Device
        accumulation_steps: Gradient accumulation steps
        global_step: Current global step
        epoch: Current epoch
        split: "train", "val", or "test"
        log_every: Log frequency

    Returns:
        Tuple of (metrics_dict, updated_global_step)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    # Local accumulators
    local_loss_sum = 0.0
    local_correct = 0
    local_total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    if is_train:
        optimizer.zero_grad()

    # Forward/backward loop; gradients enabled only when training
    with context:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Training: gradient accumulation
            if is_train:
                loss_scaled = loss / accumulation_steps
                loss_scaled.backward()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Accumulate metrics
            local_loss_sum += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            local_correct += predicted.eq(targets).sum().item()
            local_total += targets.size(0)

            # Log progress
            if batch_idx % log_every == 0:
                batch_loss = loss.item()
                batch_acc = predicted.eq(targets).sum().item() / targets.size(0)
                rank_log(
                    f"Epoch {epoch} [{split}] [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {batch_loss:.4f} Acc: {batch_acc:.4f}",
                    verbose=True,
                )

    # Reduce metrics across all ranks
    metrics_to_reduce = {
        "loss_sum": local_loss_sum,
        "correct": float(local_correct),
        "total": float(local_total),
    }
    # Use distributed reduction to aggregate metrics across ranks
    reduced = reduce_dict(metrics_to_reduce, average=False)

    # Compute final metrics (rank 0 only, but all ranks compute for consistency)
    total_samples = int(reduced["total"])
    avg_loss = reduced["loss_sum"] / total_samples if total_samples > 0 else 0.0
    accuracy = reduced["correct"] / total_samples if total_samples > 0 else 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "samples": total_samples,
    }

    return metrics, global_step


def save_checkpoint(
    checkpoint_manager: CheckpointManager,
    epoch: int,
    global_step: int,
    model: DistributedDataParallel,
    optimizer: optim.Optimizer,
    sampler_epoch: int,
    rng_state: Dict[str, Any],
    config_dict: Dict[str, Any],
) -> None:
    """Save checkpoint (rank 0 only).

    Args:
        checkpoint_manager: Checkpoint manager
        epoch: Current epoch
        global_step: Global step count
        model: DDP model
        optimizer: Optimizer
        sampler_epoch: Sampler epoch for resumption
        rng_state: RNG state dictionary
        config_dict: Distributed config dictionary
    """
    if not is_master():
        return

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sampler_epoch": sampler_epoch,
        "rng_state": rng_state,
        "config": config_dict,
    }

    checkpoint_manager.save(state, epoch=epoch)


def resume_from_checkpoint(
    checkpoint_manager: CheckpointManager,
    model: DistributedDataParallel,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Resume from latest checkpoint if available.

    Args:
        checkpoint_manager: Checkpoint manager
        model: DDP model
        optimizer: Optimizer
        device: Device

    Returns:
        Tuple of (start_epoch, global_step, sampler_epoch, should_resume)
    """
    # Check for checkpoint on rank 0
    resume_path = None
    if is_master():
        resume_path = checkpoint_manager.latest_path()

    # Broadcast resume decision
    should_resume_tensor = torch.tensor(1 if resume_path is not None else 0, device=device)
    if dist.is_initialized():
        dist.broadcast(should_resume_tensor, src=0)
    should_resume = should_resume_tensor.item() == 1

    if not should_resume:
        return 0, 0, 0, False

    # Load checkpoint on all ranks
    # All ranks load the same checkpoint path resolved by rank 0
    checkpoint, metadata = checkpoint_manager.load_latest(map_location=device)

    # Restore model and optimizer
    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore RNG state
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint.get("global_step", 0)
    sampler_epoch = checkpoint.get("sampler_epoch", checkpoint["epoch"])

    rank_log(f"Resumed from epoch {checkpoint['epoch']}, global_step {global_step}")

    return start_epoch, global_step, sampler_epoch, True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DDP training with checkpointing")
    add_classification_args(parser, include_checkpoint=True, include_ddp=True)
    parser.set_defaults(
        save_every=2,
        global_batch_size=256,
    )
    args = parser.parse_args()
    args = finalize_classification_args(args)
    return args


def main():
    """Main training function."""
    args = parse_args()

    # Initialize distributed
    dist_config = init_distributed(
        backend=args.backend,
        init_method=args.init_method,
        rank=args.rank,
        world_size=args.world_size,
        local_rank=args.local_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        verbose=args.verbose_logs,
        set_cuda_device=not args.no_set_cuda_device,
    )

    rank = dist_config.rank
    world_size = dist_config.world_size
    device = dist_config.device

    output_dir, checkpoint_dir, metrics_path = resolve_output_paths(
        args,
        default_prefix="ddp",
        metrics_filename="metrics.jsonl",
    )
    checkpoint_dir = checkpoint_dir or (output_dir / "checkpoints")
    metrics_file = metrics_path or (output_dir / "metrics.jsonl")

    synchronize()

    # Setup logging
    log_path = setup_rank_logging(output_dir, rank, args.verbose_logs)
    rank_log(f"Logging initialized: log_file={log_path}", verbose=args.verbose_logs)

    # Seed randomness
    seed_everything_for_ddp(args.seed)

    wall_start = time.perf_counter()

    # Compute micro-batch size and accumulation steps
    micro_batch_size = args.micro_batch_size or max(1, args.global_batch_size // world_size)
    accumulation_steps = math.ceil(args.global_batch_size / (micro_batch_size * world_size))
    effective_batch_size = micro_batch_size * world_size * accumulation_steps

    rank_log(
        f"Batch configuration: global_batch={args.global_batch_size}, "
        f"micro_batch={micro_batch_size}, accumulation_steps={accumulation_steps}, "
        f"effective_batch={effective_batch_size}"
    )

    # Build dataset and dataloaders using shared helpers
    train_ds, val_ds, test_ds = build_classification_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_features=args.n_features,
        n_classes=args.n_classes,
        seed=args.seed,
    )
    loaders = build_classification_loaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=micro_batch_size,
        num_workers=args.num_workers,
        distributed=True,
        world_size=world_size,
        rank=rank,
    )
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    if not loaders.samplers or "train" not in loaders.samplers:
        raise RuntimeError("Distributed training requires train sampler")
    train_sampler = loaders.samplers["train"]

    rank_log(
        f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # Build model
    model_config = ModelConfig(
        input_dim=args.n_features,
        hidden_dim=args.hidden_dim,
        num_classes=args.n_classes,
        dropout=args.dropout,
        task_type="classification",
    )
    model = build_model(model_config).to(device)
    if dist_config.device.type == "cuda":
        # CUDA path: pin module parameters to the local GPU so NCCL avoids cross-device copies
        model = DistributedDataParallel(
            model,
            device_ids=[dist_config.local_rank],
            output_device=dist_config.local_rank,
        )
    else:
        # CPU path: rely on PyTorch defaults (no device_ids argument allowed)
        model = DistributedDataParallel(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Checkpoint manager (rank 0 only, but all ranks need instance for resume)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        prefix="ddp_epoch",
        keep_last=args.keep_last,
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    global_step = 0
    sampler_epoch = 0

    if args.resume:
        start_epoch, global_step, sampler_epoch, resumed = resume_from_checkpoint(
            checkpoint_manager, model, optimizer, device
        )
        if resumed:
            train_sampler.set_epoch(sampler_epoch)

    # Signal handler setup (rank 0 only)
    stop_requested = False
    current_epoch = start_epoch

    def checkpoint_callback():
        nonlocal stop_requested
        rank_log("Signal received, saving checkpoint...")
        # Save an immediate checkpoint so sbatch resume can pick up progress
        save_checkpoint(
            checkpoint_manager,
            epoch=current_epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            sampler_epoch=train_sampler.epoch if hasattr(train_sampler, 'epoch') else current_epoch,
            rng_state=save_rng_state(),
            config_dict={
                "world_size": world_size,
                "rank": rank,
                "backend": args.backend,
                "master_addr": dist_config.master_addr,
                "master_port": dist_config.master_port,
            },
        )
        stop_requested = True

    signal_monitor = None
    if is_master():
        signal_monitor = install_default_signal_handlers(checkpoint_callback)

    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch

            # Check stop flag and broadcast to all ranks
            local_stop = stop_requested or (signal_monitor.should_stop if signal_monitor else False)
            stop_tensor = torch.tensor(1 if local_stop else 0, device=device)
            if dist.is_initialized():
                dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                if local_stop:
                    rank_log("Stop requested, exiting training loop")
                break

            # Set sampler epoch for shuffling
            train_sampler.set_epoch(epoch)
            sampler_epoch = epoch

            # Training
            train_metrics, global_step = run_epoch(
                model, train_loader, criterion, optimizer, device,
                accumulation_steps, global_step, epoch, "train", args.log_every
            )

            rank_log(
                f"Epoch {epoch} [train] Loss: {train_metrics['loss']:.4f} "
                f"Acc: {train_metrics['accuracy']:.4f}"
            )

            write_metrics_entry(
                metrics_file, epoch, "train",
                train_metrics["loss"], train_metrics["accuracy"],
                train_metrics["samples"], global_step
            )

            # Validation
            val_metrics, _ = run_epoch(
                model, val_loader, criterion, None, device,
                1, global_step, epoch, "val", args.log_every
            )

            rank_log(
                f"Epoch {epoch} [val] Loss: {val_metrics['loss']:.4f} "
                f"Acc: {val_metrics['accuracy']:.4f}"
            )

            write_metrics_entry(
                metrics_file, epoch, "val",
                val_metrics["loss"], val_metrics["accuracy"],
                val_metrics["samples"], global_step
            )

            # Checkpoint saving
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    checkpoint_manager,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    sampler_epoch=sampler_epoch,
                    rng_state=save_rng_state(),
                    config_dict={
                        "world_size": world_size,
                        "rank": rank,
                        "backend": args.backend,
                        "master_addr": dist_config.master_addr,
                        "master_port": dist_config.master_port,
                    },
                )

            synchronize()

        # Final test evaluation
        test_metrics, _ = run_epoch(
            model, test_loader, criterion, None, device,
            1, global_step, args.epochs - 1, "test", args.log_every
        )

        rank_log(
            f"Final [test] Loss: {test_metrics['loss']:.4f} "
            f"Acc: {test_metrics['accuracy']:.4f}"
        )

        write_metrics_entry(
            metrics_file, args.epochs - 1, "test",
            test_metrics["loss"], test_metrics["accuracy"],
            test_metrics["samples"], global_step
        )

        synchronize()

    finally:
        elapsed = time.perf_counter() - wall_start
        rank_log("Training complete")
        rank_log(f"Total runtime: {format_timespan(elapsed)}")
        cleanup_distributed()


if __name__ == "__main__":
    main()
