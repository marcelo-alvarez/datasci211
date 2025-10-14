#!/bin/bash
#
# Setup Week 4 environment for DataSci 211
# Creates ds211-week4 environment with PyTorch for ML training examples
#
# This script reuses the existing micromamba installation from weeks 1-2
# and creates a separate environment to avoid conflicts with ds211-python.
#
# Usage: bash setup_week4_env.sh

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================
MAMBA_ROOT="/scratch/c001/sw/micromamba"
MAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"
ENV_NAME="ds211-week4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment-week4.yml"

echo "=========================================="
echo "Week 4 Environment Setup for DataSci 211"
echo "=========================================="
echo "Micromamba root:  ${MAMBA_ROOT}"
echo "Environment name: ${ENV_NAME}"
echo "Spec file:        ${ENV_FILE}"
echo ""

# ============================================================================
# Step 1: Verify micromamba installation
# ============================================================================
echo "[1/4] Checking micromamba installation..."
if [ ! -f "${MAMBA_BIN}" ]; then
    echo "ERROR: Micromamba not found at ${MAMBA_BIN}"
    echo ""
    echo "Please run the Week 2 setup script first to install micromamba:"
    echo "  bash ../week-2/setup_shared_micromamba.sh"
    echo ""
    exit 1
fi

echo "  Micromamba found: $(${MAMBA_BIN} --version)"

# ============================================================================
# Step 2: Verify environment specification file
# ============================================================================
echo "[2/4] Checking environment specification..."
if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: Environment file not found at ${ENV_FILE}"
    exit 1
fi

echo "  Environment spec: ${ENV_FILE}"

# ============================================================================
# Step 3: Initialize shell environment
# ============================================================================
echo "[3/4] Initializing micromamba environment..."
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
eval "$("${MAMBA_BIN}" shell hook --shell=bash)"

# ============================================================================
# Step 4: Create or update ds211-week4 environment
# ============================================================================
echo "[4/4] Setting up ${ENV_NAME} environment..."

# Check if environment already exists
if micromamba env list | grep -q "^  ${ENV_NAME}"; then
    echo "  Environment '${ENV_NAME}' already exists."
    read -p "  Recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing existing environment..."
        micromamba env remove -n "${ENV_NAME}" -y
    else
        echo "  Keeping existing environment."
        echo ""
        echo "=========================================="
        echo "Setup complete (using existing environment)"
        echo "=========================================="
        exit 0
    fi
fi

echo "  Creating environment from ${ENV_FILE}..."
micromamba env create -f "${ENV_FILE}" -y

echo ""
echo "  Verifying PyTorch GPU installation..."
micromamba run -n "${ENV_NAME}" python -c "
import torch
print(f'  ✓ PyTorch {torch.__version__}')
print(f'  ✓ CUDA runtime version (build): {torch.version.cuda}')

# Check GPU availability (advisory only - login nodes have no GPUs)
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f'  ✓ CUDA available: True')
    print(f'  ✓ GPU count: {torch.cuda.device_count()}')
    print(f'  ✓ GPU 0: {torch.cuda.get_device_name(0)}')
else:
    print(f'  ⚠ CUDA available: False (expected on login nodes)')
    print(f'    GPU access will be available on compute nodes with --gpus allocation')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment '${ENV_NAME}' is ready with PyTorch 2.3.x (GPU/CUDA 12.1)."
echo ""
echo "Note: CUDA runtime is installed, but GPU access requires compute node allocation."
echo "      Use 'srun --gpus=1 python your_script.py' or submit SBATCH jobs to access GPUs."
echo ""
echo "To use in your scripts or interactive sessions, add:"
echo ""
echo "  export MAMBA_ROOT_PREFIX=\"${MAMBA_ROOT}\""
echo "  eval \"\$(\"\\$MAMBA_ROOT_PREFIX/bin/micromamba\" shell hook --shell=bash)\""
echo "  micromamba activate ${ENV_NAME}"
echo ""
echo "To verify the installation:"
echo ""
echo "  micromamba activate ${ENV_NAME}"
echo "  python -c 'import torch; print(\"PyTorch version:\", torch.__version__)'"
echo ""
echo "NOTE: This environment is separate from 'ds211-python' (weeks 1-2)."
echo "      Do not modify ds211-python to avoid breaking earlier weeks."
echo ""
