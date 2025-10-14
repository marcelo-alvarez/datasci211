#!/bin/bash
#
# Single-GPU Training Job for SLURM (~2 minute baseline run)
#
# Marlowe cluster configuration (Week 2 defaults):
# - Account: marlowe-c001
# - Partition: class
# - QoS: class
#
#SBATCH -A marlowe-c001
#SBATCH -p class
#SBATCH --qos=class
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -J single_gpu_train
#SBATCH -o logs/single_gpu_%j.out
#SBATCH -e logs/single_gpu_%j.err

# Ensure log directory exists before SLURM redirects
mkdir -p logs

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  LOAD_ENV_ROOT="${SLURM_SUBMIT_DIR}"
else
  LOAD_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${LOAD_ENV_ROOT}/load_env.sh"

echo "Environment prepared via load_env.sh"

echo "Starting single-GPU training job on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Environment: $(which python)"

# Create run directory using SLURM job ID
mkdir -p runs
RUN_DIR="runs/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "Run directory: ${RUN_DIR}"
echo "Checkpoint directory: n/a"
echo "Resume enabled: no"
echo "Launching with ${SLURM_NTASKS:-1} tasks"

# Launch training script with outputs scoped to the job-specific directory
python 01_slurm_basics/train_single.py --output-dir "${RUN_DIR}"

echo "Training complete. Results saved to ${RUN_DIR}"
