#!/bin/bash
#
# Amdahl's Law Visualization Runner
# Generates speedup curves and plots for multi-GPU scaling analysis
#
# Usage: bash run_amdahl.sh [--measure]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Amdahl's Law Analysis"
echo "========================================"

# Prepare environment
export MAMBA_ROOT_PREFIX="/scratch/c001/sw/micromamba"
MAMBA_BIN="${MAMBA_ROOT_PREFIX}/bin/micromamba"
eval "$($MAMBA_BIN shell hook --shell=bash)"
micromamba activate ds211-python

# Check if we should run measurements
if [[ "$*" == *"--measure"* ]]; then
    echo "Running with GPU measurements..."
    echo ""

    # Need to run on GPU node with multiple GPUs
    srun \
      -A marlowe-c001 \
      -p class \
      --qos=class \
      --gpus=8 \
      --ntasks=1 \
      --cpus-per-task=1 \
      -t 00:10:00 \
      python amdahl.py --measure
else
    # Just generate theoretical curves (can run on login node)
    echo "Generating theoretical curves only..."
    echo "(Use --measure flag to run actual multi-GPU measurements)"
    echo ""
    python amdahl.py
fi

# Generate plot (runs on login node)
echo ""
python plot_amdahl.py