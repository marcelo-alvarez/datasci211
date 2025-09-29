#!/bin/bash
#
# Setup shared micromamba installation for DataSci 211
# This script installs micromamba and creates the ds211-cupy environment
# in a shared location for use across multiple users/sessions.
#
# Usage: bash setup_shared_micromamba.sh

set -euo pipefail

# ============================================================================
# CONFIGURATION - Change this to install in a different location
# ============================================================================
INSTALL_BASE="/scratch/c001/sw"

# ============================================================================
# Derived paths (no need to change these)
# ============================================================================
MAMBA_ROOT="${INSTALL_BASE}/micromamba"
MAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"
ENV_NAME="ds211-python"

echo "=========================================="
echo "Micromamba Setup for DataSci 211"
echo "=========================================="
echo "Install location: ${INSTALL_BASE}"
echo "Micromamba root:  ${MAMBA_ROOT}"
echo "Environment name: ${ENV_NAME}"
echo ""

# ============================================================================
# Step 1: Create directory structure
# ============================================================================
echo "[1/4] Creating directory structure..."
mkdir -p "${MAMBA_ROOT}"
cd "${MAMBA_ROOT}"

# ============================================================================
# Step 2: Download and extract micromamba binary
# ============================================================================
echo "[2/4] Downloading micromamba binary..."
if [ -f "${MAMBA_BIN}" ]; then
    echo "  Micromamba binary already exists, skipping download."
else
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    echo "  Downloaded micromamba version:"
    "${MAMBA_BIN}" --version
fi

# ============================================================================
# Step 3: Initialize shell environment
# ============================================================================
echo "[3/4] Initializing micromamba environment..."
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
eval "$("${MAMBA_BIN}" shell hook --shell=bash)"

# ============================================================================
# Step 4: Create ds211-cupy environment
# ============================================================================
echo "[4/4] Creating ${ENV_NAME} environment..."

# Check if environment already exists
if micromamba env list | grep -q "^${ENV_NAME}"; then
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

echo "  Installing: python=3.11, pip, matplotlib, numpy from conda-forge..."
micromamba create -n "${ENV_NAME}" -y \
    python=3.11 \
    pip \
    matplotlib \
    numpy \
    -c conda-forge

echo "  Installing cupy-cuda12x from PyPI..."
micromamba run -n "${ENV_NAME}" pip install cupy-cuda12x

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Micromamba is installed at: ${MAMBA_ROOT}"
echo "Environment '${ENV_NAME}' is ready."
echo ""
echo "To use in your scripts, add:"
echo ""
echo "  export MAMBA_ROOT_PREFIX=\"${MAMBA_ROOT}\""
echo "  eval \"\$(\"\\$MAMBA_ROOT_PREFIX/bin/micromamba\" shell hook --shell=bash)\""
echo "  micromamba activate ${ENV_NAME}"
echo ""
echo "To test the installation:"
echo ""
echo "  export MAMBA_ROOT_PREFIX=\"${MAMBA_ROOT}\""
echo "  eval \"\$(\"\\$MAMBA_ROOT_PREFIX/bin/micromamba\" shell hook --shell=bash)\""
echo "  micromamba activate ${ENV_NAME}"
echo "  python -c 'import cupy; print(\"CuPy version:\", cupy.__version__)'"
echo ""