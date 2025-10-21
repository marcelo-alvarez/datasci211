#!/bin/bash
#SBATCH -A marlowe-c001
#SBATCH -p class
#SBATCH -J ddp_single_node
#SBATCH -t 00:06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH -o ddp_%j.out
#SBATCH -e ddp_%j.err

module load cuda || true
#module load pytorch || true
source ./load_env.sh


GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-1}
echo "Visible GPUs: $GPUS_ON_NODE"
echo "Launching torchrun with nproc_per_node=$GPUS_ON_NODE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=warn
export TORCH_NCCL_BLOCKING_WAIT=1

torchrun --nproc_per_node=${GPUS_ON_NODE} python python/ddp_synthetic.py   --epochs 3 --batch-size 256 --samples 20000 --model conv --flops-scale 1.0
