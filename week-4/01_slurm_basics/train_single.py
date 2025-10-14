#!/usr/bin/env python3
"""Single-GPU training script for the SLURM basics example."""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.cli import add_classification_args, finalize_classification_args, resolve_output_paths
from common.data_utils import build_classification_dataset, build_classification_loaders
from common.logging_utils import format_timespan, setup_basic_logging
from common.metrics import append_metrics_json
from common.simple_model import ModelConfig, build_model
from common.training_utils import evaluate, save_checkpoint, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Single-GPU training for SLURM basics demonstration"
    )
    add_classification_args(parser)
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    args = finalize_classification_args(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup output directory and logging
    output_dir, _, _ = resolve_output_paths(args, default_prefix="run")
    log_path = setup_basic_logging(output_dir)
    wall_start = time.perf_counter()
    logging.info(f"Logging to {log_path}")

    # Log configuration
    logging.info("=" * 60)
    logging.info("Single-GPU Training - SLURM Basics")
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
    logging.info(f"Model architecture:\n{model}")

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logging.info("=" * 60)
    logging.info("Starting training...")
    logging.info("=" * 60)

    metrics_path = output_dir / "metrics.json"
    append_metrics_json(metrics_path, {"config": vars(args)})
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train for one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch=epoch
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log epoch summary
        logging.info(
            f"Epoch {epoch + 1} Summary - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Check for NaN loss
        if torch.isnan(torch.tensor(train_metrics["loss"])):
            logging.error("Training diverged: NaN loss detected!")
            sys.exit(1)

        # Save metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        append_metrics_json(metrics_path, epoch_metrics)

        # Save checkpoint
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(
            checkpoint_path,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": vars(args),
            },
        )

        if is_best:
            best_checkpoint_path = output_dir / "checkpoint_best.pt"
            save_checkpoint(
                best_checkpoint_path,
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": vars(args),
                },
            )

    # Final test evaluation
    logging.info("=" * 60)
    logging.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    logging.info(
        f"Test Results - Loss: {test_metrics['loss']:.4f}, "
        f"Accuracy: {test_metrics['accuracy']:.4f}"
    )

    # Save metrics to JSON
    append_metrics_json(metrics_path, {"test": test_metrics})
    logging.info(f"Metrics saved to {metrics_path}")

    logging.info("=" * 60)
    logging.info("Training complete!")
    logging.info("=" * 60)
    logging.info("Total runtime: %s", format_timespan(time.perf_counter() - wall_start))


if __name__ == "__main__":
    main()
