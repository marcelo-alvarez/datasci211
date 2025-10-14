"""Command-line interface helpers shared across Week 4 training scripts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


_DEFAULT_RUN_ROOT = Path("./runs")

# Heavy (production) defaults for Week 4 workloads
_HEAVY_DEFAULTS = {
    "epochs": 16,
    "batch_size": 256,
    "hidden_dim": 1536,
    "n_train": 600000,
    "n_val": 144000,
    "n_test": 144000,
}

# Lightweight (test) defaults for fast validation runs
_TEST_DEFAULTS = {
    "epochs": 10,
    "batch_size": 32,
    "hidden_dim": 64,
    "n_train": 7000,
    "n_val": 1500,
    "n_test": 1500,
}


def add_classification_args(
    parser: argparse.ArgumentParser,
    *,
    include_checkpoint: bool = False,
    include_ddp: bool = False,
) -> argparse.ArgumentParser:
    """Attach common classification-training arguments to ``parser``."""

    # Core optimisation/training knobs
    parser.add_argument(
        "--epochs",
        type=int,
        default=_HEAVY_DEFAULTS["epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_HEAVY_DEFAULTS["batch_size"],
        help="Mini-batch size for DataLoader instances",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="lr",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=_HEAVY_DEFAULTS["hidden_dim"],
        help="Hidden layer width for SimpleModel",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability applied in the model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for DataLoader instances",
    )

    # Dataset sizing knobs
    parser.add_argument(
        "--n-train",
        type=int,
        default=_HEAVY_DEFAULTS["n_train"],
        help="Number of synthetic samples allocated to the training split",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=_HEAVY_DEFAULTS["n_val"],
        help="Number of synthetic samples allocated to the validation split",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=_HEAVY_DEFAULTS["n_test"],
        help="Number of synthetic samples allocated to the test split",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Feature dimensionality for the generated dataset",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=4,
        help="Number of target classes for the dataset",
    )

    # Output handling
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where logs, metrics, and checkpoints are written",
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Activate lightweight defaults suitable for quick testing",
    )

    if include_checkpoint:
        parser.add_argument(
            "--checkpoint-dir",
            type=str,
            default=None,
            help="Directory used to persist checkpoints (defaults to <output-dir>/checkpoints)",
        )
        parser.add_argument(
            "--save-every",
            type=int,
            default=1,
            help="Save a checkpoint every N epochs",
        )
        parser.add_argument(
            "--keep-last",
            type=int,
            default=3,
            help="Retain only the most recent N checkpoints",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            default=True,
            help="Resume training from the latest checkpoint when available",
        )
        parser.add_argument(
            "--no-resume",
            dest="resume",
            action="store_false",
            help="Disable automatic resume from checkpoints",
        )

    if include_ddp:
        parser.add_argument(
            "--global-batch-size",
            type=int,
            default=256,
            help="Effective batch size aggregated across all ranks",
        )
        parser.add_argument(
            "--micro-batch-size",
            type=int,
            default=None,
            help="Per-rank mini-batch size; computed from global batch if omitted",
        )
        parser.add_argument(
            "--weight-decay",
            type=float,
            default=1e-4,
            help="L2 weight decay applied by the optimizer",
        )
        parser.add_argument(
            "--log-every",
            type=int,
            default=10,
            help="Emit progress logs every N batches",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="nccl",
            help="Distributed backend handed to torch.distributed.init_process_group",
        )
        parser.add_argument(
            "--init-method",
            type=str,
            default="env://",
            help="Process-group init method (e.g., env://, tcp://<host>:<port>)",
        )
        parser.add_argument("--rank", type=int, default=None, help="Explicit global rank override")
        parser.add_argument(
            "--world-size", type=int, default=None, help="Explicit world-size override"
        )
        parser.add_argument(
            "--local-rank",
            type=int,
            default=None,
            help="Explicit local-rank override (torchrun/SLURM typically set this)",
        )
        parser.add_argument(
            "--master-addr",
            type=str,
            default=None,
            help="Master address used for rendezvous (defaults to SLURM or torchrun env)",
        )
        parser.add_argument(
            "--master-port",
            type=int,
            default=None,
            help="Master port used for rendezvous (defaults to SLURM or torchrun env)",
        )
        parser.add_argument(
            "--metrics-file",
            type=str,
            default=None,
            help="Optional path for aggregated JSONL metrics",
        )
        parser.add_argument(
            "--verbose-logs",
            action="store_true",
            help="Emit console logs from every rank instead of rank 0 only",
        )
        parser.add_argument(
            "--no-set-cuda-device",
            action="store_true",
            help="Skip automatic torch.cuda.set_device(local_rank)",
        )

    return parser


def finalize_classification_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply test-mode overrides when requested and return args."""

    if getattr(args, "test_mode", False):
        if args.epochs == _HEAVY_DEFAULTS["epochs"]:
            args.epochs = _TEST_DEFAULTS["epochs"]
        if args.batch_size == _HEAVY_DEFAULTS["batch_size"]:
            args.batch_size = _TEST_DEFAULTS["batch_size"]
        if args.hidden_dim == _HEAVY_DEFAULTS["hidden_dim"]:
            args.hidden_dim = _TEST_DEFAULTS["hidden_dim"]
        if args.n_train == _HEAVY_DEFAULTS["n_train"]:
            args.n_train = _TEST_DEFAULTS["n_train"]
        if args.n_val == _HEAVY_DEFAULTS["n_val"]:
            args.n_val = _TEST_DEFAULTS["n_val"]
        if args.n_test == _HEAVY_DEFAULTS["n_test"]:
            args.n_test = _TEST_DEFAULTS["n_test"]
    return args


def resolve_output_paths(
    args: argparse.Namespace,
    default_prefix: str,
    *,
    checkpoint_subdir: str = "checkpoints",
    metrics_filename: Optional[str] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Resolve output paths based on CLI arguments."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_RUN_ROOT / f"{default_prefix}_{timestamp}"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    checkpoint_dir: Optional[Path] = None
    if hasattr(args, "checkpoint_dir"):
        checkpoint_dir = (
            Path(args.checkpoint_dir).expanduser().resolve()
            if args.checkpoint_dir
            else output_dir / checkpoint_subdir
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        args.checkpoint_dir = str(checkpoint_dir)

    metrics_path: Optional[Path] = None
    provided_metrics = getattr(args, "metrics_file", None)
    if provided_metrics:
        metrics_path = Path(provided_metrics).expanduser().resolve()
    elif metrics_filename is not None:
        metrics_path = output_dir / metrics_filename

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(args, "metrics_file"):
            args.metrics_file = str(metrics_path)

    return output_dir, checkpoint_dir, metrics_path
