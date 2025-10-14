#!/bin/bash
#
# Checkpointing Training Job for SLURM
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
#SBATCH -J checkpoint_train
#SBATCH -o logs/checkpoint_%j.out
#SBATCH -e logs/checkpoint_%j.err
#SBATCH --signal=SIGUSR1@90

mkdir -p logs

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  LOAD_ENV_ROOT="${SLURM_SUBMIT_DIR}"
else
  LOAD_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${LOAD_ENV_ROOT}/load_env.sh"

echo "Environment prepared via load_env.sh"

echo "Starting checkpointing training job on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Environment: $(which python)"

# Create run directory based on SLURM job ID
mkdir -p runs
RUN_DIR="runs/${SLURM_JOB_ID}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"

echo "Run directory: ${RUN_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Resume enabled: yes"
echo "Launching with ${SLURM_NTASKS:-1} tasks"

# Launch training script with checkpoint resume capability
python 02_checkpointing/train_with_checkpoint.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${RUN_DIR}" \
    "$@"

echo "Training complete. Results saved to ${RUN_DIR}"
