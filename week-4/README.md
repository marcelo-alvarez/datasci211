# Week 4 – SLURM Workflows, Checkpointing, Distributed Training

## Scope

Week 4 extends the course materials from simple GPU benchmarks to training-oriented SLURM jobs:

1. `01_slurm_basics/` — single-GPU submission practice
2. `02_checkpointing/` — signal-aware checkpoint/resume flow
3. `03_distributed/` — multi-GPU PyTorch DDP launchers

## Environment

Week 4 runs in a dedicated Micromamba environment (`ds211-week4`). Keep `ds211-python` (Weeks 1–2) unchanged.

### Option 1 — Use the class environment (recommended)

The shared week 4 environment has already been created on Marlowe. Activate it in any shell or SBATCH script as follows:

```bash
module purge
export MAMBA_ROOT_PREFIX="/scratch/c001/sw/micromamba"
eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook --shell=bash)"
micromamba activate ds211-week4
```

The Week 4 Slurm scripts already follow this pattern.

### Option 2 — Create your own environment

If you prefer to create your own personal copy, build it from the spec file:

```bash
export MAMBA_ROOT_PREFIX="/scratch/c001/users/$USER/micromamba"
mkdir -p "$MAMBA_ROOT_PREFIX"
eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook --shell=bash)"
micromamba env create -f environment-week4.yml   # rerun with `micromamba env update -f ...` to refresh
micromamba activate ds211-week4
```

## Directory overview

- `common/` — shared CLI, logging, data, metrics, and model utilities reused by all modules.
- `01_slurm_basics/` — single-GPU job script, training entry point, and instructor notes.
- `02_checkpointing/` — checkpoint manager, signal handler, and checkpoint launcher.
- `03_distributed/` — DDP helpers, two-GPU launchers, and notes.

See each subdirectory README for command examples.

## Quick commands

Activate `ds211-week4` (Option 1 or 2) and run the single-GPU job:

```bash
cd week-4
sbatch 01_slurm_basics/single_gpu_train.sh
```

The launcher trains for 16 epochs on ~200k synthetic samples (with ~48k validation/test) using a 256-wide hidden layer; expect roughly a 2 minute runtime on a single H100 unless you trim the CLI flags.

Checkpoint-aware submission:

```bash
sbatch 02_checkpointing/checkpoint_train.sh
```

DDP submission and resume (adjust `CHECKPOINT_DIR` to an existing run):

```bash
sbatch 03_distributed/ddp_train.sh
# Reuse an existing checkpoint directory to continue a run
CHECKPOINT_DIR=/path/to/checkpoints sbatch 03_distributed/ddp_train.sh
```
