#!/bin/bash

# ---- Preprocessing Directives ---
# Anything prepended with the # character is a comment
# with the exception of lines starting with #SBATCH 
# which is a pre-processing directive the batch scheduler
# (Slurm) that Marlowe uses

#SBATCH -J hello-gpu        # Job name
#SBATCH -o hello-gpu.%j.out # Output file
#SBATCH -t 0-00:02:00       # Time limit format: DD-HH:MM:SS
#SBATCH -N 1                # Number of Nodes
#SBATCH --gpus=1            # In Slurm you must explicitly request GPU(s)
#SBATCH --ntasks=2          # Number of tasks
#SBATCH --cpus-per-task=1   # Number of CPUs per Task
#SBATCH -A marlowe-c001     # Marlowe Slurm Account for DataSci 211
#SBATCH --partition=class   # Partition to be used for DataSci 211

# --- Load environment modules ---
module load nvhpc

# --- Run commands --- 
echo "Job $SLURM_JOB_ID on $(date)"
echo "Node list: $SLURM_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS  SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "Requested GPU(s): $SLURM_GPUS"
echo -n "CPU model: "; grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //'

# echo "=== nvidia-smi ==="
if command -v nvidia-smi >/dev/null; then nvidia-smi; else echo "nvidia-smi not found"; fi


# Compile (fast, a couple of seconds)
echo "Compiling hello_gpu.cu ..."
nvcc -O2 -arch=native -lineinfo -o hello_gpu hello_gpu.cu

srun ./hello_gpu

