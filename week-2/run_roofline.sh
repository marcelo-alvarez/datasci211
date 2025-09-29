#!/bin/bash
#
# Roofline Runner - Executes CUDA and CuPy arithmetic intensity sweeps
# Uses srun for GPU allocation, then plots results locally (no GPU needed)
#
# Usage: bash run_roofline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Roofline Sweep on H100"
echo "========================================"

# Prepare environment
module load nvhpc
export CUDA_PATH="${NVHPC_ROOT}/cuda"
export MAMBA_ROOT_PREFIX="/scratch/c001/sw/micromamba"
eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell=bash)"
micromamba activate ds211-python

# Compile CUDA version
echo "Compiling CUDA roofline kernel..."
nvcc -O3 -arch=sm_90 -std=c++11 -o roofline_cuda roofline.cu

# Run GPU work via srun
echo "Running GPU benchmarks..."
srun \
  -A marlowe-c001 \
  -p class \
  --qos=class \
  --gpus=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  -t 00:10:00 \
  bash -c './roofline_cuda && python roofline.py'

# Generate summary and plot
echo ""
python plot_roofline.py
