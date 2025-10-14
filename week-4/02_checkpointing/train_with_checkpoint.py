#!/usr/bin/env python3
"""Training script with checkpoint support and signal handling.

Demonstrates:
- Periodic checkpoint saving with atomic writes
- Automatic resume from latest checkpoint
- SLURM signal handling for graceful preemption
- Metrics tracking in JSONL format
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.cli import add_classification_args, finalize_classification_args, resolve_output_paths
from common.data_utils import build_classification_dataset, build_classification_loaders
from common.logging_utils import format_timespan, setup_basic_logging
from common.metrics import append_metrics_jsonl
from common.random_state import restore_rng_state, save_rng_state
from common.simple_model import ModelConfig, build_model
from common.training_utils import evaluate, set_seed, train_one_epoch

from checkpoint_io import CheckpointManager
from signal_handler import install_default_signal_handlers


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Training with checkpointing and signal handling"
    )
    add_classification_args(parser, include_checkpoint=True)
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    args = finalize_classification_args(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup output directory and logging
    output_dir, checkpoint_dir, metrics_path = resolve_output_paths(
        args, default_prefix="run", metrics_filename="metrics.jsonl"
    )
    log_path = setup_basic_logging(output_dir)
    wall_start = time.perf_counter()
    logging.info(f"Logging to {log_path}")
    if checkpoint_dir is None:
        checkpoint_dir = output_dir / "checkpoints"

    # Log configuration
    logging.info("=" * 60)
    logging.info("Training with Checkpointing")
    logging.info("=" * 60)
    logging.info(f"Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    logging.info("=" * 60)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(checkpoint_dir),
        prefix="checkpoint_epoch",
        keep_last=args.keep_last,
    )

    # Metrics file (JSONL format)
    metrics_file = metrics_path or (output_dir / "metrics.jsonl")

    # Attempt to resume from checkpoint
    start_epoch = 0
    model = None
    optimizer = None
    last_metrics = None

    if args.resume:
        try:
            checkpoint, metadata = checkpoint_manager.load_latest(map_location=device)
            logging.info(f"Resuming from checkpoint at epoch {checkpoint['epoch']}")

            # Restore epoch counter
            start_epoch = checkpoint["epoch"] + 1

            # Will restore model/optimizer state after building them
            saved_model_state = checkpoint.get("model_state_dict")
            saved_optimizer_state = checkpoint.get("optimizer_state_dict")
            saved_rng_state = checkpoint.get("rng_state")
            last_metrics = checkpoint.get("last_metrics")

            # Restore RNG state
            if saved_rng_state:
                restore_rng_state(saved_rng_state)
                logging.info("RNG state restored from checkpoint")

        except FileNotFoundError:
            logging.info("No checkpoint found, starting from scratch")
            saved_model_state = None
            saved_optimizer_state = None

    else:
        logging.info("Resume disabled, starting from scratch")
        saved_model_state = None
        saved_optimizer_state = None

    # Generate synthetic data
    logging.info("Generating synthetic classification data...")
    train_ds, val_ds, test_ds = build_classification_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_features=args.n_features,
        n_classes=args.n_classes,
        seed=args.seed,
    )
    logging.info(
        "Splitting dataset: train=%s, val=%s, test=%s",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    loaders = build_classification_loaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test

    # Build model
    logging.info("Building model...")
    config = ModelConfig(
        input_dim=args.n_features,
        hidden_dim=args.hidden_dim,
        num_classes=args.n_classes,
        dropout=args.dropout,
        task_type="classification",
    )
    model = build_model(config).to(device)

    # Restore model state if resuming
    if saved_model_state:
        model.load_state_dict(saved_model_state)
        logging.info("Model state restored from checkpoint")

    logging.info(f"Model architecture:\n{model}")

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Restore optimizer state if resuming
    if saved_optimizer_state:
        optimizer.load_state_dict(saved_optimizer_state)
        logging.info("Optimizer state restored from checkpoint")

    criterion = nn.CrossEntropyLoss()

    # Setup signal handler
    signal_triggered = {"value": False}

    def checkpoint_callback():
        """Save checkpoint when signal received."""
        signal_triggered["value"] = True
        logging.warning("Signal received, saving emergency checkpoint...")

    monitor = install_default_signal_handlers(checkpoint_callback)

    # Training loop
    logging.info("=" * 60)
    logging.info(f"Starting training from epoch {start_epoch}...")
    logging.info("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\nEpoch {epoch}/{args.epochs - 1}")

        # Train for one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch=epoch
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log epoch summary
        logging.info(
            f"Epoch {epoch} Summary - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Check for NaN loss
        if torch.isnan(torch.tensor(train_metrics["loss"])):
            logging.error("Training diverged: NaN loss detected!")
            sys.exit(1)

        # Write metrics to JSONL
        append_metrics_jsonl(
            metrics_file,
            {
                "epoch": epoch,
                "split": "train",
                "loss": train_metrics["loss"],
                "accuracy": train_metrics["accuracy"],
                "timestamp": datetime.now().isoformat(),
            },
        )
        append_metrics_jsonl(
            metrics_file,
            {
                "epoch": epoch,
                "split": "val",
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Update best validation loss
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        # Save checkpoint periodically or if signal received
        should_save = (epoch % args.save_every == 0) or signal_triggered["value"]

        if should_save:
            triggered_by_signal = signal_triggered["value"]
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": save_rng_state(),
                "last_metrics": {"train": train_metrics, "val": val_metrics},
                "args": vars(args),
                "signal": triggered_by_signal,
            }
            checkpoint_manager.save(state, epoch=epoch, is_best=is_best)
            if triggered_by_signal:
                signal_triggered["value"] = False

        # Check if we should stop due to signal
        if monitor.should_stop:
            logging.warning("Stopping training due to signal, exiting gracefully")
            sys.exit(0)

    # Final test evaluation
    logging.info("=" * 60)
    logging.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    logging.info(
        f"Test Results - Loss: {test_metrics['loss']:.4f}, "
        f"Accuracy: {test_metrics['accuracy']:.4f}"
    )

    # Write test metrics
    append_metrics_jsonl(
        metrics_file,
        {
            "epoch": args.epochs - 1,
            "split": "test",
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "timestamp": datetime.now().isoformat(),
        },
    )

    logging.info("=" * 60)
    logging.info("Training complete!")
    logging.info("=" * 60)
    logging.info("Total runtime: %s", format_timespan(time.perf_counter() - wall_start))


if __name__ == "__main__":
    main()
