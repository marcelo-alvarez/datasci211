#!/bin/bash
#
# Multi-GPU DDP Training Job for SLURM (Marlowe Week 4)
# - Account: marlowe-c001
# - Partition: class
# - QoS: class
# - Default run: single node, 2 GPUs (adjust SLURM directives as needed)
#
#SBATCH -A marlowe-c001
#SBATCH -p class
#SBATCH --qos=class
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -J ddp_train
#SBATCH -o logs/ddp_train_%j.out
#SBATCH -e logs/ddp_train_%j.err

mkdir -p logs

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  LOAD_ENV_ROOT="${SLURM_SUBMIT_DIR}"
else
  LOAD_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${LOAD_ENV_ROOT}/load_env.sh"

echo "Environment prepared via load_env.sh"

echo "Starting distributed training job on $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Environment: $(which python)"

# Create run directory using SLURM job ID (allowing overrides via OUTPUT_DIR)
mkdir -p runs
DEFAULT_RUN_DIR="runs/${SLURM_JOB_ID}"
RUN_DIR="${OUTPUT_DIR:-${DEFAULT_RUN_DIR}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"

echo "Run directory: ${RUN_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Resume enabled: yes"
echo "Launching with ${SLURM_NTASKS:-1} tasks"

# Launch training script with DDP
srun --gpus-per-task=1 --gpu-bind=closest \
  python 03_distributed/train_ddp.py \
    --output-dir "${RUN_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    "$@"

echo "Training complete. Results saved to ${RUN_DIR}"
