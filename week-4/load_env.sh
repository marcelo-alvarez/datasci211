#!/bin/bash

# Detect if script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced, not executed directly." >&2
    echo "Usage: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# Purge existing modules
module purge
module load slurm

# Set MAMBA_ROOT_PREFIX if not already set
if [[ -z "${MAMBA_ROOT_PREFIX}" ]]; then
    export MAMBA_ROOT_PREFIX=/scratch/c001/sw/micromamba
fi

# Initialize micromamba for bash
eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell bash)"

# Activate ds211-week4 environment
micromamba activate ds211-week4

# Compute CUDA MPS directories
if [[ -n "${SLURM_JOB_ID}" ]]; then
    MPS_ID="${SLURM_JOB_ID}"
else
    MPS_ID="${USER}_$$"
fi

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${MPS_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${MPS_ID}"

# Create directories if they don't exist
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
mkdir -p "${CUDA_MPS_LOG_DIRECTORY}"

# Print confirmation message
echo "Environment loaded successfully:"
echo "  Python: $(which python)"
echo "  CUDA MPS Pipe: ${CUDA_MPS_PIPE_DIRECTORY}"
echo "  CUDA MPS Log: ${CUDA_MPS_LOG_DIRECTORY}"
